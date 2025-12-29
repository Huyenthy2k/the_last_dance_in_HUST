# ！/usr/bin/python
# -*- coding: utf-8 -*-#
"""
---------------------------------
 Name: tradeEnv.py
 Description: Define the trading environment for the trading agent.
--------------------------------
"""

from collections import deque
import numpy as np
import sys
import os
import pandas as pd
import time
import copy

import gymnasium as gym
from gymnasium import spaces

GYMNASIUM_AVAILABLE = True

# Seeding compatibility
try:
    from gymnasium.utils import seeding
    GYM_SEEDING_AVAILABLE = True
except ImportError:
    GYM_SEEDING_AVAILABLE = False
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.stats import entropy
import scipy.stats as spstats

try:
    from utils.weight_symbol_mapper import get_top_stocks
except ImportError:
    # Fallback if weight_symbol_mapper is not available
    get_top_stocks = None

# Import LiveDisplay integration
try:
    from utils.display_integration import (
        update_step as display_update_step,
        update_selection as display_update_selection,
        update_controller as display_update_controller,
        update_regime_shift as display_update_regime_shift,
        update_reward_components as display_update_reward_components,
        get_display,
        smart_print,
    )

    LIVE_DISPLAY_AVAILABLE = True
except ImportError:
    LIVE_DISPLAY_AVAILABLE = False
    smart_print = print  # Fallback to normal print

    def get_display():
        return None


def _is_live_display_active() -> bool:
    """Check if LiveDisplay is currently active and rendering."""
    if not LIVE_DISPLAY_AVAILABLE:
        return False
    display = get_display()
    return display is not None and display.enabled


# Import Portfolio Allocator reward function (new simplified 2-component reward)
try:
    from RL_controller.portfolio_allocator_reward import (
        compute_reward as compute_pa_reward,
        RewardNormalizer,
    )

    PA_REWARD_AVAILABLE = True
except ImportError:
    PA_REWARD_AVAILABLE = False
    smart_print(
        "[Warning] portfolio_allocator_reward module not available; using legacy reward",
        flush=True,
    )


def _safe_array_from_values(values):
    """Safely convert values to numpy array, handling inhomogeneous shapes."""
    try:
        arr = np.array(values)
        # If array has object dtype (inhomogeneous), extract scalars
        if arr.dtype == object:
            # Extract scalar values safely
            result = []
            for v in values:
                if isinstance(v, np.ndarray):
                    # If it's an array, take first element if size > 1, or convert if size == 1
                    if v.size == 1:
                        result.append(float(v.item()))
                    elif v.size > 1:
                        # Take first element
                        result.append(float(v.flat[0]))
                    else:
                        result.append(0.0)
                elif hasattr(v, "item"):
                    try:
                        result.append(float(v.item()))
                    except (ValueError, TypeError):
                        result.append(
                            float(v) if isinstance(v, (int, float, np.number)) else 0.0
                        )
                else:
                    result.append(
                        float(v) if isinstance(v, (int, float, np.number)) else 0.0
                    )
            arr = np.array(result)
        # Ensure 1D array
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr
    except (ValueError, TypeError):
        # Fallback: extract scalar values manually
        result = []
        for v in values:
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    result.append(float(v.item()))
                elif v.size > 1:
                    result.append(float(v.flat[0]))
                else:
                    result.append(0.0)
            elif hasattr(v, "item"):
                try:
                    result.append(float(v.item()))
                except (ValueError, TypeError):
                    result.append(
                        float(v) if isinstance(v, (int, float, np.number)) else 0.0
                    )
            else:
                result.append(
                    float(v) if isinstance(v, (int, float, np.number)) else 0.0
                )
        return np.array(result)


def _fillna_infer(series, value=0.0):
    """
    Fill NaN values and infer numeric dtypes without triggering pandas FutureWarning.
    """
    if hasattr(series, "infer_objects"):
        series = series.infer_objects(copy=False)
    filled = series.fillna(value)
    if hasattr(filled, "infer_objects"):
        filled = filled.infer_objects(copy=False)
    return filled


def _normalize_prob(vec):
    vec = np.array(vec, dtype=float).flatten()
    if vec.size == 0:
        return vec
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = np.clip(vec, 1e-12, None)
    total = np.sum(vec)
    if total <= 1e-12:
        return np.ones_like(vec) / len(vec)
    return vec / total


def _build_cov_from_indicator(values, target_size, fallback):
    """
    Build a covariance matrix compatible with target_size based on indicator values.
    Falls back to scaled identity when insufficient data.
    """
    arr = _safe_array_from_values(values)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.flatten()

    if arr.size < 2:
        return np.eye(target_size) * fallback

    try:
        cov = np.cov(arr)
    except Exception:
        cov = np.array([[np.var(arr)]])

    cov = np.nan_to_num(cov, nan=fallback, posinf=fallback, neginf=fallback)
    if cov.ndim == 0:
        cov = np.eye(target_size) * float(cov)
    elif cov.ndim == 1:
        cov = np.diag(cov)

    if cov.shape[0] != target_size or cov.shape[1] != target_size:
        if cov.size == 1:
            cov = np.eye(target_size) * float(cov.flat[0])
        else:
            if cov.shape[0] < target_size:
                pad = target_size - cov.shape[0]
                cov = np.pad(
                    cov, ((0, pad), (0, pad)), mode="constant", constant_values=fallback
                )
                np.fill_diagonal(cov[-pad:, -pad:], fallback)
            else:
                cov = cov[:target_size, :target_size]

    cov = 0.5 * (cov + cov.T)
    cov += np.eye(target_size) * 1e-9
    return cov


class StockPortfolioEnv(gym.Env):
    def __init__(
        self,
        config,
        rawdata,
        mode,
        stock_num,
        action_dim,
        tech_indicator_lst=None,
        max_shares=None,
        initial_asset=1000000,
        reward_scaling=1,
        norm_method="sum",
        transaction_cost=0.001,
        slippage=0.001,
        seed_num=2022,
        extra_data=None,
        mkt_observer=None,
    ):
        self.config = config
        self.rawdata = rawdata
        self.mode = mode  # train, valid, test
        self.stock_num = stock_num  # Number of stocks in universe
        # RL acts only on Top-K; keep action_dim_rl = mafia_top_k for RL outputs/actions
        self.action_dim = getattr(config, "mafia_top_k", action_dim)
        self.action_dim_rl = self.action_dim
        self.validation_mode = (
            False  # Flag to indicate if environment is in validation mode
        )
        # State mode: enforce compact Top-K flow per refactor spec
        self.mafia_state_mode = getattr(self.config, "mafia_state_mode", "compact")
        if self.mafia_state_mode != "compact":
            smart_print(
                f"[MAFIA][ENV] Forcing mafia_state_mode=compact (was {self.mafia_state_mode}) to align with Top-K-only flow",
                flush=True,
            )
            self.mafia_state_mode = "compact"
            try:
                self.config.mafia_state_mode = "compact"
            except Exception:
                pass
        # Enforce Top-K-only action mode (no full-universe RL actions)
        self.mafia_action_full_universe = False
        try:
            self.config.mafia_action_full_universe = False
        except Exception:
            pass
        # Handle tech_indicator_lst (optional for MAFIA, required for legacy)
        if tech_indicator_lst is None:
            tech_indicator_lst = []
        self.tech_indicator_lst = tech_indicator_lst
        self.tech_indicator_lst_wocov = copy.deepcopy(
            self.tech_indicator_lst
        )  # without cov feature
        if "cov" in self.tech_indicator_lst_wocov:
            self.tech_indicator_lst_wocov.remove("cov")
        self.max_shares = max_shares  # Maximum number of shares
        self.seed_num = seed_num
        # Handle gym version compatibility for seeding
        try:
            self.seed(seed=self.seed_num)
        except TypeError:
            # Newer gym versions might use different seed signature
            try:
                self.seed(self.seed_num)
            except Exception:
                # Fallback: manually set numpy random seed
                np.random.seed(self.seed_num)
        self.epoch = 0
        self.curTradeDay = 0
        self.eps = 1e-6

        self.initial_asset = initial_asset  # Initial portfolio value
        self.reward_scaling = reward_scaling
        self.norm_method = norm_method
        self.transaction_cost = transaction_cost  # 0.001
        self.lambda_tc = getattr(self.config, "lambda_tc", self.transaction_cost)
        self.rebalance_interval = getattr(self.config, "rebalance_interval", 15)
        self.topk_rebalance_interval = max(
            1,
            int(
                getattr(
                    self.config,
                    "mafia_topk_rebalance_interval",
                    self.rebalance_interval,
                )
            ),
        )
        self._last_selection_triggered = False
        self._last_selection_reason = None
        self._last_eta_value = None
        self.slippage = slippage  # 0.001 for one-side, 0.002 for two-side
        self.cur_slippage_drift = (
            np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
        )
        # Track rebalance cadence and emergency triggers
        self.days_since_rebalance = 0
        self.force_rebalance = False
        self.emergency_rebalance = False
        # Contextual Awareness Features (per spec Section 3.4)
        self.last_topk_avg_return = 0.0  # R_topk_avg for relative_alpha calculation
        self.last_relative_alpha = 0.0  # α_rel = R_portfolio - R_topk_avg
        self.last_risk_violation = 0.0  # δ_risk = max(0, σ(a^RL) - σ_target)
        # Rebalance Signals for TD3 (reset each step, only 1 on rebalance day)
        self.last_rebalance_scheduled = False  # True on scheduled rebalance day
        self.last_rebalance_regime = False  # True when regime shift triggers rebalance
        # Buffer for market index close prices used in continuous eta mapping
        self._eta_price_buffer = deque(
            maxlen=getattr(self.config, "risk_eta_window", 50)
        )
        if extra_data is not None:
            self.extra_data = extra_data
        else:
            self.extra_data = None

        if self.norm_method == "softmax":
            self.weights_normalization = self.softmax_normalization
        elif self.norm_method == "sum":
            self.weights_normalization = self.sum_normalization
        else:
            raise ValueError(
                "Unexpected normalization method of stock weights: {}".format(
                    self.norm_method
                )
            )
        if self.config.enable_cov_features:
            self.state_dim = (
                (len(self.tech_indicator_lst_wocov) + self.stock_num) * self.stock_num
            ) + 1  # +1: current portfolio value
        else:
            self.state_dim = (
                len(self.tech_indicator_lst_wocov) * self.stock_num
            ) + 1  # +1: current portfolio value
        if self.config.enable_market_observer:
            self.mkt_observer = mkt_observer
            # If using MAFIA observer, override state dimension based on mode
            try:
                from RL_controller.mafia_observer import MAFIAObserver

                if isinstance(self.mkt_observer, MAFIAObserver):
                    risk_dim = (
                        1
                        if getattr(
                            self.config, "mafia_include_risk_boundary_in_state", False
                        )
                        else 0
                    )
                    # Compact mode: state = [market_vector(K), portfolio_value, (optional) risk_boundary]
                    # Removed market_index_state - rely on Observer's learned representation
                    k = self.config.mafia_top_k
                    self.state_dim = k + 1 + risk_dim
            except Exception:
                pass
        else:
            self.mkt_observer = None

        # Initialize market_scores_full and topk_indices for MAFIA observer (will be populated in run_mkt_observer)
        self.market_scores_full = None
        self.observer_topk_indices = (
            None  # Top-K indices selected by Observer (for mode 'compact')
        )

        # Validate MAFIA state mode if using MAFIA observer
        if self.config.enable_market_observer and self.mkt_observer is not None:
            try:
                from RL_controller.mafia_observer import MAFIAObserver

                if isinstance(self.mkt_observer, MAFIAObserver):
                    smart_print(
                        f"[MAFIA] State mode: COMPACT (K={self.config.mafia_top_k} stocks in state)"
                    )
            except Exception:
                pass

        if self.config.benchmark_algo in self.config.only_long_algo_lst:
            # Long only
            self.action_space = spaces.Box(
                low=0, high=1, shape=(self.action_dim_rl,), dtype=np.float32
            )
            self.bound_flag = 1  # 1 for long and long+short, -1 for short
        else:
            if self.config.trade_pattern == 1:
                # Long only
                self.action_space = spaces.Box(
                    low=0, high=1, shape=(self.action_dim_rl,), dtype=np.float32
                )
                self.bound_flag = 1  # 1 for long and long+short, -1 for short
            else:
                raise ValueError(
                    "Unexpected trade pattern: {}".format(self.config.trade_pattern)
                )

        self.rawdata.sort_values(["date", "stock"], ascending=True, inplace=True)
        self.rawdata.index = self.rawdata.date.factorize()[0]
        self.totalTradeDay = len(self.rawdata["date"].unique())
        self.stock_lst = np.sort(self.rawdata["stock"].unique())
        # Mini-epoch tracking for observer mid-epoch training
        self.observer_mini_epoch_steps = getattr(
            self.config, "observer_mini_epoch_steps", 0
        )
        self._observer_mini_step_counter = 0

        self.stock_index_map = {stock: idx for idx, stock in enumerate(self.stock_lst)}
        # RL only allocates on Observer-selected Top-K
        self.rl_stock_num = getattr(self.config, "mafia_top_k", self.stock_num)
        self.use_multibranch_state = (
            getattr(self.config, "rl_obs_use_multibranch_state", True)
            and self.config.enable_market_observer
        )
        if self.use_multibranch_state:
            self._init_rl_multibranch_spec()
            self.observation_space = spaces.Dict(
                {
                    "global_context": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.rl_global_dim,),
                        dtype=np.float32,
                    ),
                    "per_stock": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.rl_stock_num, self.rl_per_stock_feature_dim),
                        dtype=np.float32,
                    ),
                    "history": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.rl_obs_history_len, self.rl_history_feature_dim),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
            )

        self.curData = self._ensure_dataframe(
            self.rawdata.loc[self.curTradeDay, :].copy()
        )
        self.curData.sort_values(["stock"], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        self._ensure_minimal_features()
        self.lastDayData = None  # Will be set after first step
        self.last_slippage_drift = np.zeros(
            self.stock_num, dtype=np.float32
        )  # Will be updated after first step
        if self.use_multibranch_state:
            self._reset_rl_multibranch_buffers()

        # Initialize state (will be built in run_mkt_observer)
        if self.use_multibranch_state:
            self.state = {
                "global_context": np.zeros(self.rl_global_dim, dtype=np.float32),
                "per_stock": np.zeros(
                    (self.rl_stock_num, self.rl_per_stock_feature_dim), dtype=np.float32
                ),
                "history": np.zeros(
                    (self.rl_obs_history_len, self.rl_history_feature_dim),
                    dtype=np.float32,
                ),
            }
        else:
            self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.ctl_state = self._build_ctl_state()
        self.terminal = False

        # Initialize capital before running market observer (needed for state building)
        self.cur_capital = self.initial_asset
        self.last_valid_capital = self.cur_capital
        self.last_valid_capital = self.cur_capital

        self.profit_lst = [0]  # percentage of portfolio daily returns

        # Initialize Portfolio Allocator reward components
        self.use_pa_reward = PA_REWARD_AVAILABLE and getattr(
            self.config, "use_portfolio_allocator_reward", False
        )
        if self.use_pa_reward:
            # Parameters for new 2-component reward function
            self.pa_w_return = getattr(self.config, "allocator_return_weight", 1.0)
            self.pa_lambda_js = getattr(self.config, "allocator_lambda_js", 0.1)
            self.pa_reward_scale = getattr(self.config, "allocator_reward_scale", 1.0)
            self.pa_reward_normalizer = RewardNormalizer(
                alpha=getattr(self.config, "allocator_reward_norm_alpha", 0.01)
            )
            # Track reward components for logging
            self.rl_reward_return_lst = []
            self.rl_reward_divergence_lst = []
            smart_print(
                f"[ENV] Portfolio Allocator reward enabled: w_return={self.pa_w_return}, "
                f"lambda_js={self.pa_lambda_js}, reward_scale={self.pa_reward_scale}",
                flush=True,
            )
        else:
            # Legacy reward tracking
            self.pa_reward_normalizer = None

        # Initialize action memories before observer builds history features
        init_topk = max(1, getattr(self, "rl_stock_num", self.stock_num))
        self.action_topk_indices_memory = [
            np.arange(min(init_topk, self.stock_num), dtype=int)
        ]
        self.action_cbf_memeory = [np.zeros(init_topk, dtype=float)]
        self.actions_memory = [
            np.array([1 / self.stock_num] * self.stock_num) * self.bound_flag
        ]
        self.action_rl_memory = [
            np.array([1 / init_topk] * init_topk) * self.bound_flag
        ]
        # Initialize risk history before running market observer (observer builds state that reads these lists)
        self.risk_raw_lst = [0]  # Raw risk without controller
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset]
        # Track selection mask for cadence alignment (1 = rebalance day, 0 = holding day)
        self._selection_mask_lst = [1]  # First day is always rebalance
        # Regime shift detection buffers
        self._direction_history = deque(
            maxlen=getattr(self.config, "regime_confirmation_window", 3) + 1
        )
        self._index_return_buffer = deque(
            maxlen=getattr(self.config, "regime_vol_window", 20) + 1
        )
        self._vol_history = deque(
            maxlen=max(10, getattr(self.config, "regime_vol_window", 20))
        )
        self._last_index_pivot = None
        self._last_dc_direction = 0
        self.regime_shift_detected = False
        self.regime_shift_reason = None

        cur_risk_boundary, stock_ma_price = self.run_mkt_observer(
            stage="init"
        )  # after curData and state, before cur_risk_boundary
        if stock_ma_price is not None:
            self.ctl_state["MA-{}".format(self.config.otherRef_indicator_ma_window)] = (
                stock_ma_price
            )

        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]

        self.asset_lst = [self.initial_asset]
        self.date_memory = [self.curData["date"].unique()[0]]
        self.reward_lst = [0]

        self.risk_adj_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.solver_stat = {
            "solvable": 0,
            "insolvable": 0,
            "stochastic_solvable": 0,
            "stochastic_time": [],
            "socp_solvable": 0,
            "socp_time": [],
        }

        self.ctrl_weight_lst = [1.0]
        self.solvable_flag = []
        self.risk_pred_lst = []

        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.latest_invest_profile = None
        # Persist last completed epoch profile even after reset so callbacks can log correctly
        self.last_epoch_profile = None
        self.cnt1 = 0
        self.cnt2 = 0
        self.stepcount = 0

        risk_free = self.config.mkt_rf[self.config.market_name] / 100
        self.start_cputime = time.process_time()
        self.start_systime = time.perf_counter()
        if self.mode == "train":
            self.exclusive_cputime = 0
            self.exclusive_systime = 0
        # For saving profile
        self.profile_hist_field_lst = [
            "ep",
            "trading_days",
            "annualReturn_pct",
            "mdd",
            "sharpeRatio",
            "final_capital",
            "volatility",
            "calmarRatio",
            "sterlingRatio",
            "netProfit",
            "netProfit_pct",
            "winRate",
            "vol_max",
            "vol_min",
            "vol_avg",
            "risk_max",
            "risk_min",
            "risk_avg",
            "riskRaw_max",
            "riskRaw_min",
            "riskRaw_avg",
            "dailySR_max",
            "dailySR_min",
            "dailySR_avg",
            "dailySR_wocbf_max",
            "dailySR_wocbf_min",
            "dailySR_wocbf_avg",
            "dailyReturn_pct_max",
            "dailyReturn_pct_min",
            "dailyReturn_pct_avg",
            "sigReturn_max",
            "sigReturn_min",
            "mdd_high",
            "mdd_low",
            "mdd_high_date",
            "mdd_low_date",
            "sharpeRatio_wocbf",
            "reward_sum",
            "final_capital_wocbf",
            "cbf_contribution",
            "risk_downsideAtVol",
            "risk_downsideAtVol_daily_max",
            "risk_downsideAtVol_daily_min",
            "risk_downsideAtVol_daily_avg",
            "risk_downsideAtValue_daily_max",
            "risk_downsideAtValue_daily_min",
            "risk_downsideAtValue_daily_avg",
            "cvar_max",
            "cvar_min",
            "cvar_avg",
            "cvar_raw_max",
            "cvar_raw_min",
            "cvar_raw_avg",
            "solver_solvable",
            "solver_insolvable",
            "cputime",
            "systime",
        ]
        self.profile_hist_ep = {k: [] for k in self.profile_hist_field_lst}
        # Track already-saved epochs and hydrate history when resuming a run
        self._profile_saved_epochs = set()
        self._load_existing_profile_history()

    def _ensure_dataframe(self, data):
        """Return a DataFrame even when pandas gives back a Series for single-row slices."""
        if isinstance(data, pd.Series):
            return data.to_frame().T
        return data

    def _build_ctl_state(self):
        """Populate control state arrays, defaulting missing indicators to zeros."""
        ctl_state = {}
        cur_len = (
            len(self.curData)
            if hasattr(self, "curData") and self.curData is not None
            else 0
        )
        for key in getattr(self.config, "otherRef_indicator_lst", []):
            if cur_len > 0 and key in getattr(self.curData, "columns", []):
                values = self.curData[key].values
            else:
                values = np.zeros(cur_len, dtype=float)
            ctl_state[key] = _safe_array_from_values(values)
        return ctl_state

    def _ensure_minimal_features(self):
        """Guarantee base feature columns exist for simplified test data."""
        if not hasattr(self, "curData") or self.curData is None:
            return
        daily_key = f"DAILYRETURNS-{getattr(self.config, 'dailyRetun_lookback', 30)}"
        if daily_key not in self.curData.columns:
            self.curData[daily_key] = 0.0
        if hasattr(self, "rawdata") and daily_key not in self.rawdata.columns:
            self.rawdata[daily_key] = 0.0

    def _get_current_topk_indices(self):
        """Return current observer Top-K indices or a deterministic fallback."""
        idx = getattr(self, "observer_topk_indices", None)
        if idx is None:
            return np.arange(min(self.rl_stock_num, self.stock_num), dtype=int)
        idx = np.array(idx, dtype=int).flatten()
        idx = idx[(idx >= 0) & (idx < self.stock_num)]
        if idx.size == 0:
            idx = np.arange(min(self.rl_stock_num, self.stock_num), dtype=int)
        return idx

    def _expand_action_to_full(self, action, indices=None):
        """Expand a Top-K action to full-universe length using provided indices."""
        action = np.array(action, dtype=float).flatten()
        if action.shape[0] == self.stock_num:
            # Already full length
            full = action
        else:
            full = np.zeros(self.stock_num, dtype=float)
            idx = (
                self._get_current_topk_indices()
                if indices is None
                else np.array(indices, dtype=int).flatten()
            )
            idx = idx[(idx >= 0) & (idx < self.stock_num)]
            fill_len = min(len(action), len(idx))
            if fill_len > 0:
                full[idx[:fill_len]] = action[:fill_len]
        full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
        return full

    def _record_action_memory(
        self, rl_action_topk, cbf_action_topk=None, topk_indices=None
    ):
        """
        Store RL/CBF actions in Top-K form plus associated indices for later expansion.
        """
        idx = (
            self._get_current_topk_indices()
            if topk_indices is None
            else np.array(topk_indices, dtype=int).flatten()
        )
        idx = idx[(idx >= 0) & (idx < self.stock_num)]
        if idx.size == 0:
            idx = self._get_current_topk_indices()
        self.action_topk_indices_memory.append(idx)
        rl_arr = np.array(rl_action_topk, dtype=float).flatten()
        self.action_rl_memory.append(rl_arr)
        cbf_arr = (
            np.zeros_like(rl_arr)
            if cbf_action_topk is None
            else np.array(cbf_action_topk, dtype=float).flatten()
        )
        self.action_cbf_memeory.append(cbf_arr)

    def _get_full_action_from_memory(self, memory_list, step_idx=-1):
        """Expand stored Top-K action at step to full length using recorded indices."""
        if len(memory_list) == 0:
            return np.zeros(self.stock_num, dtype=float)
        action = memory_list[step_idx]
        idx = None
        if (
            hasattr(self, "action_topk_indices_memory")
            and len(self.action_topk_indices_memory) > 0
        ):
            idx = self.action_topk_indices_memory[step_idx]
        return self._expand_action_to_full(action, indices=idx)

    def _expand_action_series(self, memory_list):
        """Expand an entire action history (Top-K) to full length per step."""
        if len(memory_list) == 0:
            return np.zeros((0, self.stock_num), dtype=float)
        expanded = []
        for i, act in enumerate(memory_list):
            idx = None
            if hasattr(self, "action_topk_indices_memory") and i < len(
                self.action_topk_indices_memory
            ):
                idx = self.action_topk_indices_memory[i]
            expanded.append(self._expand_action_to_full(act, indices=idx))
        return np.vstack(expanded)

    def _build_step_info(self) -> dict:
        """Build info dict for step() return. Contains metadata for replay buffer filtering."""
        # Get current date in YYYYMMDD format (as int)
        cur_date_str = self.date_memory[-1] if len(self.date_memory) > 0 else "19700101"
        try:
            # Handle various date formats
            if isinstance(cur_date_str, str):
                cur_trade_day_int = int(
                    cur_date_str.replace("-", "").replace("/", "")[:8]
                )
            else:
                cur_trade_day_int = int(str(cur_date_str)[:8].replace("-", ""))
        except (ValueError, TypeError):
            cur_trade_day_int = 0

        return {
            "curTradeDay": cur_trade_day_int,
            "regime_shift_event": getattr(self, "regime_shift_detected", False),
        }

    def _update_live_display(
        self,
        step_return: float = 0.0,
        reward: float = 0.0,
        selection_trigger: bool = False,
        selection_reason: str = None,
    ):
        """Update LiveDisplay with current step information."""
        if not LIVE_DISPLAY_AVAILABLE:
            return

        try:
            display = get_display()
            if display is None:
                return
            # Note: Don't check display.enabled - state updates should happen
            # for dashboard API even when terminal display is disabled

            import numpy as np

            # Get current date
            cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
            if hasattr(cur_date, "strftime"):
                cur_date = cur_date.strftime("%Y-%m-%d")
            else:
                cur_date = str(cur_date)

            # Get market direction from observer (0=Bear, 1=Flat, 2=Bull)
            direction = 1  # Default: Flat
            trend_z = getattr(self, "last_trend_zscore", 0.0)
            eta_risk = 1.0
            if hasattr(self, "mkt_observer") and self.mkt_observer is not None:
                dir_pred = getattr(self.mkt_observer, "last_direction_pred", "FLAT")
                if dir_pred == "Bull":
                    direction = 2
                elif dir_pred == "Bear":
                    direction = 0
                else:
                    direction = 1
                eta_risk = getattr(self.mkt_observer, "last_eta", 1.0)

            # Get regime info
            regime_shift_count = getattr(self, "regime_shift_count", 0)

            # Portfolio info
            top_k = getattr(self.config, "topK", 10)
            capital = getattr(self, "cur_capital", 1_000_000)
            days_since_rebal = getattr(self, "days_since_rebalance", 0)
            rebal_interval = getattr(self.config, "topk_rebalance_interval", 14)
            rebalance_count = getattr(self, "rebalance_count", 0)

            # Calculate cumulative return, volatility, sharpe, mdd, win_rate
            cumul_return = 0.0
            volatility = 0.0
            sharpe = 0.0
            mdd = 0.0
            win_rate = 0.5
            daily_return = step_return * 100  # Convert to percentage

            if hasattr(self, "asset_lst") and len(self.asset_lst) >= 2:
                initial_asset = (
                    self.asset_lst[0] if self.asset_lst[0] > 0 else 1_000_000
                )
                cumul_return = (capital - initial_asset) / initial_asset * 100

            if hasattr(self, "profit_lst") and len(self.profit_lst) > 1:
                profits = np.array(self.profit_lst[1:])  # Skip first 0
                if len(profits) > 0:
                    # Daily volatility (annualized)
                    volatility = (
                        np.std(profits) * np.sqrt(252) if len(profits) > 1 else 0.0
                    )
                    # Sharpe ratio
                    avg_return = np.mean(profits) * 252  # Annualized
                    rf_rate = getattr(self.config, "mkt_rf", {}).get(
                        getattr(self.config, "market_name", "VNINDEX"), 0.05
                    )
                    if volatility > 1e-8:
                        sharpe = (avg_return - rf_rate) / volatility
                    # Win rate
                    win_rate = (
                        np.sum(profits > 0) / len(profits) if len(profits) > 0 else 0.5
                    )

            # Calculate MDD
            if hasattr(self, "asset_lst") and len(self.asset_lst) > 1:
                assets = np.array(self.asset_lst)
                peak = np.maximum.accumulate(assets)
                drawdown = (peak - assets) / peak
                mdd = np.max(drawdown) * 100  # As percentage

            # Selection stats
            n_kept = len(getattr(self, "_cur_topk_stocks", []))
            n_added = 0
            n_removed = 0
            if hasattr(self, "_prev_topk_stocks") and hasattr(self, "_cur_topk_stocks"):
                prev_set = set(self._prev_topk_stocks or [])
                cur_set = set(self._cur_topk_stocks or [])
                n_kept = len(prev_set & cur_set)
                n_added = len(cur_set - prev_set)
                n_removed = len(prev_set - cur_set)

            # Update step display with simplified fields
            display_update_step(
                step=self.curTradeDay,
                date=cur_date,
                direction=direction,
                trend_z=trend_z,
                eta_risk=eta_risk,
                volatility=volatility,
                regime_shift_count=regime_shift_count,
                is_rebalance=selection_trigger,
                days_since_rebal=days_since_rebal,
                rebal_interval=rebal_interval,
                top_k=top_k,
                capital=capital,
                daily_return=daily_return,
                cumul_return=cumul_return,
                n_kept=n_kept,
                n_added=n_added,
                n_removed=n_removed,
                rebalance_count=rebalance_count,
                sharpe=sharpe,
                mdd=mdd,
                win_rate=win_rate,
                td3_reward=reward,
                # Note: total_steps managed by callback for accurate epoch tracking
                start_time=getattr(self, "_epoch_start_time", 0),
            )

            # Log selection event if triggered
            if selection_trigger and selection_reason:
                # Get kept/added/removed stocks
                kept = []
                added = []
                removed = []
                if hasattr(self, "_prev_topk_stocks") and hasattr(
                    self, "_cur_topk_stocks"
                ):
                    prev_set = set(self._prev_topk_stocks or [])
                    cur_set = set(self._cur_topk_stocks or [])
                    kept = list(prev_set & cur_set)
                    added = list(cur_set - prev_set)
                    removed = list(prev_set - cur_set)

                display_update_selection(
                    step=self.curTradeDay,
                    date=cur_date,
                    trigger=selection_reason,
                    kept=kept,
                    added=added,
                    removed=removed,
                )

        except Exception as e:
            # Silently ignore display errors to not disrupt training
            pass

    def step(self, actions):
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        allow_observer_training = bool(
            getattr(self.config, "mafia_allow_observer_training", False)
        )
        # Debug: Log when episode ends
        if self.terminal:
            smart_print(
                f"[ENV] ⚠️ Episode end detected! curTradeDay={self.curTradeDay}, totalTradeDay={self.totalTradeDay}, terminal={self.terminal}, epoch={self.epoch}",
                flush=True,
            )
            self.cur_capital = self.cur_capital * (1 - self.transaction_cost)
            self.asset_lst[-1] = self.cur_capital
            if len(self.asset_lst) >= 2:
                self.profit_lst[-1] = (
                    self.cur_capital - self.asset_lst[-2]
                ) / self.asset_lst[-2]
            else:
                # Single-step episode fallback
                self.profit_lst[-1] = 0.0
            # Ensure reward attribute exists for terminal return path
            if not hasattr(self, "reward"):
                self.reward = (
                    float(self.profit_lst[-1]) if len(self.profit_lst) > 0 else 0.0
                )
            if len(self.action_rl_memory) > 1:
                last_rl_full = self._get_full_action_from_memory(
                    self.action_rl_memory, -1
                )
                self.return_raw_lst[-1] = self.return_raw_lst[-1] * (
                    1 - (np.sum(np.abs(last_rl_full)) * self.transaction_cost)
                )

            if (
                (self.config.enable_market_observer)
                and (self.mkt_observer is not None)
                and (self.mode == "train")
                and allow_observer_training
            ):
                # Training at the end of epoch
                ori_profit_rate = np.append(
                    [1],
                    np.array(self.return_raw_lst)[1:]
                    / np.array(self.return_raw_lst)[:-1],
                    axis=0,
                )
                adj_profit_rate = np.array(self.profit_lst) + 1
                # Build selection mask for cadence alignment
                # Ensures Stock Selection PG loss is only computed on rebalance days
                selection_mask = getattr(self, "_selection_mask_lst", None)
                if selection_mask is not None:
                    selection_mask = np.array(selection_mask, dtype=np.float32)
                label_kwargs = {
                    "mode": self.mode,
                    "ori_profit": ori_profit_rate,
                    "adj_profit": adj_profit_rate,
                    "ori_risk": np.array(self.risk_raw_lst),
                    "adj_risk": np.array(self.risk_cbf_lst),
                    "topk_selection_mask": selection_mask,
                }
                # train() returns True if training happened, False if skipped (warmup)
                trained = self.mkt_observer.train(**label_kwargs)
                if trained:
                    smart_print(
                        f"[MAFIA][OBSERVER] ⭐ Epoch-end train @ epoch {self.epoch} | steps: {self.curTradeDay + 1}/{self.totalTradeDay}",
                        flush=True,
                    )
                    self._observer_mini_step_counter = 0
            elif (
                (self.config.enable_market_observer)
                and (self.mkt_observer is not None)
                and (self.mode == "train")
                and (not allow_observer_training)
            ):
                # Clear buffered tensors when observer training is frozen (Phase 2: RL-Only)
                # Use reset(preserve_warmup=False) to fully clear buffers
                self.mkt_observer.reset(preserve_warmup=False)
                self._observer_mini_step_counter = 0

            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True

            # Skip save_profile during validation to prevent array length conflicts
            if not self.validation_mode:
                smart_print(
                    f"[Profile Save] Epoch {self.epoch} complete (mode={self.mode}), calling get_results() and save_profile()...",
                    flush=True,
                )
                invest_profile = self.get_results()
                smart_print(
                    f"[Profile Save] get_results() completed, calling save_profile()...",
                    flush=True,
                )
                self.save_profile(invest_profile=invest_profile)
                smart_print(
                    f"[Profile Save] save_profile() completed for epoch {self.epoch}",
                    flush=True,
                )
                self.latest_invest_profile = invest_profile
                if self.latest_invest_profile is not None:
                    self.last_epoch_profile = self.latest_invest_profile
            else:
                try:
                    self.latest_invest_profile = self.get_results()
                    if self.latest_invest_profile is not None:
                        self.last_epoch_profile = self.latest_invest_profile
                except Exception:
                    self.latest_invest_profile = None

            if self.use_multibranch_state:
                self.peak_capital = max(self.peak_capital, self.cur_capital)
            # Return format compatible with both gym and gymnasium
            # Always use gymnasium format (terminated, truncated, info) for consistency with VecEnv
            # Debug: Log return value
            smart_print(
                f"[ENV] Returning terminal={self.terminal} (will be converted to dones array by VecEnv)",
                flush=True,
            )
            return (
                self.state,
                self.reward,
                self.terminal,
                False,
                self._build_step_info(),
            )

        # Non-terminal path
        actions = np.reshape(actions, (-1))  # [1, num_of_stocks] or [num_of_stocks, ]
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        raw_topk_action = np.array(actions, copy=True)
        # Map Top-K actions to full universe when Observer indices are available
        if actions.shape[0] != self.stock_num:
            full_actions = np.zeros(self.stock_num, dtype=float)
            idx = getattr(self, "observer_topk_indices", None)
            if idx is None or len(np.atleast_1d(idx)) == 0:
                idx = np.arange(min(len(actions), self.stock_num))
            idx = np.clip(np.array(idx, dtype=int), 0, self.stock_num - 1)
            fill_k = min(len(actions), len(idx))
            full_actions[idx[:fill_k]] = actions[:fill_k]
            actions = full_actions
        turnover_amt = 0.0
        # Selection scheduler: rebalance Top-K on cadence or force flag
        selection_trigger = False
        selection_reason = None
        interval = max(
            1, getattr(self, "topk_rebalance_interval", self.rebalance_interval)
        )
        if self.curTradeDay == 0:
            selection_trigger = True
            selection_reason = "init"
        elif getattr(self, "force_rebalance", False):
            selection_trigger = True
            selection_reason = "force_flag"
        elif (self.days_since_rebalance + 1) >= interval:
            selection_trigger = True
            selection_reason = "schedule"
        if selection_trigger and getattr(self.config, "mafia_log_scheduler", True):
            smart_print(
                f"[SCHED] trigger_topk_update=1 reason={selection_reason} day={self.curTradeDay} dsr={self.days_since_rebalance}",
                flush=True,
            )
        # Ensure curData and lastDayData prices are aligned with stock_lst
        cur_close_prices = np.zeros(self.stock_num)
        last_close_prices = np.zeros(self.stock_num)
        for i, stock in enumerate(self.stock_lst):
            if stock in self.curData["stock"].values:
                close_val = self.curData[self.curData["stock"] == stock][
                    "close"
                ].values[0]
                # Validate and sanitize NaN/inf/<=0 to prevent NaN propagation
                if pd.isna(close_val) or np.isinf(close_val) or close_val <= 0:
                    self.nan_stats["cur_close_nan"] += 1
                    cur_date = (
                        self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
                    )
                    fallback_used = None
                    # Fallback: use last known price or 1.0
                    if (
                        self.lastDayData is not None
                        and stock in self.lastDayData["stock"].values
                    ):
                        last_close_val = self.lastDayData[
                            self.lastDayData["stock"] == stock
                        ]["close"].values[0]
                        if not (
                            pd.isna(last_close_val)
                            or np.isinf(last_close_val)
                            or last_close_val <= 0
                        ):
                            cur_close_prices[i] = last_close_val
                            fallback_used = f"last_known_price({last_close_val:.4f})"
                        else:
                            cur_close_prices[i] = 1.0
                            fallback_used = "1.0_default"
                    else:
                        cur_close_prices[i] = 1.0
                        fallback_used = "1.0_default"
                    # Log details (limit to first 10 per epoch to avoid spam)
                    if len(self.nan_stats["cur_close_details"]) < 10:
                        self.nan_stats["cur_close_details"].append(
                            {
                                "day": self.curTradeDay,
                                "date": cur_date,
                                "stock": stock,
                                "value": close_val,
                                "fallback": fallback_used,
                            }
                        )
                else:
                    cur_close_prices[i] = close_val
            else:
                cur_close_prices[i] = 1.0  # Default to no change if missing
            # last_close_prices
            if (
                self.lastDayData is not None
                and stock in self.lastDayData["stock"].values
            ):
                last_close_val = self.lastDayData[self.lastDayData["stock"] == stock][
                    "close"
                ].values[0]
                # Validate and sanitize NaN/inf/<=0
                if (
                    pd.isna(last_close_val)
                    or np.isinf(last_close_val)
                    or last_close_val <= 0
                ):
                    self.nan_stats["last_close_nan"] += 1
                    cur_date = (
                        self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
                    )
                    fallback_used = None
                    # Fallback: use current price or 1.0
                    if not (
                        pd.isna(cur_close_prices[i])
                        or np.isinf(cur_close_prices[i])
                        or cur_close_prices[i] <= 0
                    ):
                        last_close_prices[i] = cur_close_prices[i]
                        fallback_used = f"current_price({cur_close_prices[i]:.4f})"
                    else:
                        last_close_prices[i] = 1.0
                        fallback_used = "1.0_default"
                    # Log details (limit to first 10 per epoch to avoid spam)
                    if len(self.nan_stats["last_close_details"]) < 10:
                        self.nan_stats["last_close_details"].append(
                            {
                                "day": self.curTradeDay,
                                "date": cur_date,
                                "stock": stock,
                                "value": last_close_val,
                                "fallback": fallback_used,
                            }
                        )
                else:
                    last_close_prices[i] = last_close_val
            else:
                # Validate cur_close_prices before using
                if (
                    pd.isna(cur_close_prices[i])
                    or np.isinf(cur_close_prices[i])
                    or cur_close_prices[i] <= 0
                ):
                    last_close_prices[i] = 1.0
                else:
                    last_close_prices[i] = cur_close_prices[
                        i
                    ]  # Use current if last not available

        cur_p = cur_close_prices * (1 + self.cur_slippage_drift)
        last_p = last_close_prices * (1 + self.last_slippage_drift)
        x_p = cur_p / last_p
        last_action = np.array(self.actions_memory[-1])
        x_p_adj = np.where((x_p >= 2) & (last_action < 0), 2, x_p)
        sgn = np.sign(last_action)
        # Check if loss the whole capital
        adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
        adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
        if (adj_cap <= 0) or np.all(adj_w_ay == 0):
            raise ValueError(
                "Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                    self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay
                )
            )
        last_w_adj = adj_w_ay / adj_cap
        # Apply RL allocation daily (Top-K stable via observer_topk_indices mapping)
        weights = self.weights_normalization(
            actions=actions
        )  # Unnormalized weights -> normalized weights
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        last_w_adj = np.nan_to_num(last_w_adj, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions_memory.append(weights)
        turnover_amt = float(
            np.nan_to_num(
                np.sum(np.abs(weights - last_w_adj)), nan=0.0, posinf=0.0, neginf=0.0
            )
        )
        self.cur_capital = self.cur_capital * (
            1 - (turnover_amt * self.transaction_cost)
        )
        if not np.isfinite(self.cur_capital):
            self.cur_capital = (
                self.last_valid_capital
                if np.isfinite(self.last_valid_capital)
                else self.initial_asset
            )
        self.asset_lst[-1] = self.cur_capital
        # Calculate profit: on first step, profit is 0; afterwards use asset difference
        if len(self.asset_lst) >= 2:
            self.profit_lst[-1] = (
                self.cur_capital - self.asset_lst[-2]
            ) / self.asset_lst[-2]
        else:
            self.profit_lst[-1] = 0.0
        # Track RL action memory in Top-K space
        topk_len = min(self.rl_stock_num, raw_topk_action.size)
        rl_topk_norm = raw_topk_action[:topk_len]
        rl_topk_norm = (
            rl_topk_norm / (np.sum(np.abs(rl_topk_norm)) + 1e-8)
            if rl_topk_norm.size > 0
            else rl_topk_norm
        )
        self._record_action_memory(
            rl_action_topk=rl_topk_norm, topk_indices=self._get_current_topk_indices()
        )
        # RL tracking only for post-pretrain steps (skip during uniform action pretrain)
        if len(self.action_rl_memory) > 1:
            last_rl_action = np.array(self.action_rl_memory[-2])
            last_rl_action = np.nan_to_num(
                last_rl_action, nan=0.0, posinf=0.0, neginf=0.0
            )
            if self.rl_stock_num == len(last_rl_action):
                sgn_rl = np.sign(last_rl_action)
                prev_rl_cap = self.return_raw_lst[-1]
                # Get Top-K prices for RL action tracking
                topk_indices = self._get_current_topk_indices()
                if topk_indices is not None and len(topk_indices) == len(
                    last_rl_action
                ):
                    x_p_topk = x_p[topk_indices]
                    x_p_adjrl = np.where(
                        (x_p_topk >= 2) & (last_rl_action < 0), 2, x_p_topk
                    )
                    adj_w_ay = sgn_rl * (
                        last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action)
                    )
                    adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1
                    if (adj_cap <= 0) or np.all(adj_w_ay == 0):
                        smart_print(
                            "Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                                self.curTradeDay,
                                self.date_memory[-1],
                                adj_cap,
                                adj_w_ay,
                            )
                        )
                        adj_w_ay = (
                            np.array([1 / self.rl_stock_num] * self.rl_stock_num)
                            * self.bound_flag
                        )
                        adj_cap = 1
                    last_rlw_adj = adj_w_ay / adj_cap
                    return_raw = prev_rl_cap * (
                        1
                        - (
                            np.sum(np.abs(self.action_rl_memory[-1] - last_rlw_adj))
                            * self.transaction_cost
                        )
                    )
                    self.return_raw_lst[-1] = return_raw

        curDay_ClosePrice_withSlippage = cur_close_prices * (
            1 + self.cur_slippage_drift
        )
        lastDay_ClosePrice_withSlippage = last_close_prices * (
            1 + self.last_slippage_drift
        )
        rate_of_price_change = curDay_ClosePrice_withSlippage / (
            lastDay_ClosePrice_withSlippage + 1e-8
        )  # Avoid division by zero
        rate_of_price_change_adj = np.where(
            (rate_of_price_change >= 2) & (weights < 0), 2, rate_of_price_change
        )
        sigDayReturn = (
            rate_of_price_change_adj - 1
        ) * weights  # [s1_pct, s2_pct, .., px_pct_returns]
        poDayReturn = np.sum(sigDayReturn)

        # Validate poDayReturn to prevent NaN propagation
        if np.isnan(poDayReturn) or np.isinf(poDayReturn):
            self.nan_stats["poDayReturn_nan"] += 1
            cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
            smart_print(
                f"Warning: poDayReturn is {poDayReturn}, using 0.0 as fallback (Day: {self.curTradeDay}, date: {cur_date})",
                flush=True,
            )
            # Log details (limit to first 10 per epoch to avoid spam)
            if len(self.nan_stats["poDayReturn_details"]) < 10:
                self.nan_stats["poDayReturn_details"].append(
                    {"day": self.curTradeDay, "date": cur_date, "value": poDayReturn}
                )
            poDayReturn = 0.0

        if poDayReturn <= (-1):
            raise ValueError(
                "Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(
                    self.curTradeDay, self.date_memory[-1], poDayReturn
                )
            )

        prev_cur_capital = (
            self.cur_capital
        )  # Save previous value for poDayReturn_withcost calculation
        updatePoValue = self.cur_capital * (poDayReturn + 1)
        # Validate updatePoValue to prevent NaN propagation
        if np.isnan(updatePoValue) or np.isinf(updatePoValue) or updatePoValue <= 0:
            self.nan_stats["updatePoValue_nan"] += 1
            cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
            fallback_cap = (
                self.last_valid_capital
                if np.isfinite(self.last_valid_capital)
                else self.initial_asset
            )
            smart_print(
                f"Warning: updatePoValue is {updatePoValue}, using fallback capital {fallback_cap} (Day: {self.curTradeDay}, date: {cur_date})",
                flush=True,
            )
            # Log details (limit to first 10 per epoch to avoid spam)
            if len(self.nan_stats["updatePoValue_details"]) < 10:
                self.nan_stats["updatePoValue_details"].append(
                    {
                        "day": self.curTradeDay,
                        "date": cur_date,
                        "value": updatePoValue,
                        "prev_capital": prev_cur_capital,
                        "poDayReturn": poDayReturn,
                    }
                )
            updatePoValue = fallback_cap  # Keep previous valid value

        self.cur_capital = updatePoValue
        if np.isfinite(self.cur_capital):
            self.last_valid_capital = self.cur_capital
        poDayReturn_withcost = (updatePoValue - prev_cur_capital) / (
            prev_cur_capital + 1e-8
        )  # Avoid division by zero

        # Compute relative_alpha for Contextual Awareness (spec Section 3.4)
        # α_rel = R_portfolio - R_topk_avg (Credit Assignment)
        topk_indices = getattr(self, "observer_topk_indices", None)
        if topk_indices is not None and len(topk_indices) > 0:
            idx = np.array(topk_indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(rate_of_price_change_adj))]
            if len(idx) > 0:
                # Unweighted average return of Top-K stocks (equal-weight benchmark)
                topk_returns = rate_of_price_change_adj[idx] - 1
                self.last_topk_avg_return = float(np.mean(topk_returns))
            else:
                self.last_topk_avg_return = 0.0
        else:
            self.last_topk_avg_return = 0.0
        self.last_relative_alpha = float(
            poDayReturn_withcost - self.last_topk_avg_return
        )

        self.profit_lst.append(poDayReturn_withcost)
        self.asset_lst.append(self.cur_capital)

        # Jump to the next day
        self.curTradeDay = self.curTradeDay + 1
        # Save current data as lastDayData (ensure it has all stocks)
        self.lastDayData = self.curData.copy()
        self.last_slippage_drift = self.cur_slippage_drift
        self.curData = self._ensure_dataframe(
            self.rawdata.loc[self.curTradeDay, :].copy()
        )
        self.curData.sort_values(["stock"], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)

        # Ensure curData has all expected stocks (from stock_lst)
        # If some stocks are missing, add them with NaN values (will be handled later)
        if len(self.curData) < len(self.stock_lst):
            missing_stocks = set(self.stock_lst) - set(self.curData["stock"].unique())
            if len(missing_stocks) > 0:
                # Add missing stocks with NaN values (will use forward fill from lastDayData if available)
                cur_date = (
                    self.curData["date"].unique()[0]
                    if len(self.curData) > 0
                    else self.date_memory[-1]
                    if len(self.date_memory) > 0
                    else None
                )
                if cur_date is not None:
                    for stock in missing_stocks:
                        # Try to get from lastDayData first
                        if (
                            self.lastDayData is not None
                            and stock in self.lastDayData["stock"].values
                        ):
                            last_row = (
                                self.lastDayData[self.lastDayData["stock"] == stock]
                                .iloc[0]
                                .copy()
                            )
                            last_row["date"] = cur_date
                            self.curData = pd.concat(
                                [self.curData, last_row.to_frame().T], ignore_index=True
                            )
                        else:
                            # Create row with NaN values (will be filled later)
                            new_row = pd.DataFrame(
                                {
                                    "stock": [stock],
                                    "date": [cur_date],
                                    "open": [np.nan],
                                    "close": [np.nan],
                                    "high": [np.nan],
                                    "low": [np.nan],
                                    "volume": [np.nan],
                                }
                            )
                            # Add other required columns if they exist
                            for col in self.curData.columns:
                                if col not in new_row.columns:
                                    new_row[col] = np.nan
                            self.curData = pd.concat(
                                [self.curData, new_row], ignore_index=True
                            )
                    # Re-sort after adding missing stocks
                    self.curData.sort_values(["stock"], ascending=True, inplace=True)
                    self.curData.reset_index(drop=True, inplace=True)

                    # Fill NaN values with forward fill from lastDayData or use last valid value
                    if self.lastDayData is not None:
                        for col in ["open", "close", "high", "low", "volume"]:
                            if col in self.curData.columns:
                                # Fill NaN with corresponding value from lastDayData
                                for idx, row in self.curData.iterrows():
                                    if (
                                        pd.isna(row[col])
                                        and row["stock"]
                                        in self.lastDayData["stock"].values
                                    ):
                                        last_val = self.lastDayData[
                                            self.lastDayData["stock"] == row["stock"]
                                        ][col].values
                                        if len(last_val) > 0:
                                            self.curData.at[idx, col] = last_val[0]
                                # If still NaN, fill with 0 (shouldn't happen with proper data)
                                self.curData[col] = _fillna_infer(
                                    self.curData[col], value=0.0
                                )

        self.ctl_state = self._build_ctl_state()
        cur_date = self.curData["date"].unique()[0]
        self.date_memory.append(cur_date)

        self.cur_slippage_drift = (
            np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
        )

        # Ensure curData and lastDayData have same stocks in same order
        cur_close_prices = np.zeros(self.stock_num)
        last_close_prices = np.zeros(self.stock_num)
        for i, stock in enumerate(self.stock_lst):
            if stock in self.curData["stock"].values:
                close_val = self.curData[self.curData["stock"] == stock][
                    "close"
                ].values[0]
                # Validate and sanitize NaN/inf/<=0 to prevent NaN propagation
                if pd.isna(close_val) or np.isinf(close_val) or close_val <= 0:
                    self.nan_stats["cur_close_nan"] += 1
                    cur_date = (
                        self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
                    )
                    fallback_used = None
                    # Fallback: use last known price or 1.0
                    if (
                        self.lastDayData is not None
                        and stock in self.lastDayData["stock"].values
                    ):
                        last_close_val = self.lastDayData[
                            self.lastDayData["stock"] == stock
                        ]["close"].values[0]
                        if not (
                            pd.isna(last_close_val)
                            or np.isinf(last_close_val)
                            or last_close_val <= 0
                        ):
                            cur_close_prices[i] = last_close_val
                            fallback_used = f"last_known_price({last_close_val:.4f})"
                        else:
                            cur_close_prices[i] = 1.0
                            fallback_used = "1.0_default"
                    else:
                        cur_close_prices[i] = 1.0
                        fallback_used = "1.0_default"
                    # Log details (limit to first 10 per epoch to avoid spam)
                    if len(self.nan_stats["cur_close_details"]) < 10:
                        self.nan_stats["cur_close_details"].append(
                            {
                                "day": self.curTradeDay,
                                "date": cur_date,
                                "stock": stock,
                                "value": close_val,
                                "fallback": fallback_used,
                            }
                        )
                else:
                    cur_close_prices[i] = close_val
            else:
                cur_close_prices[i] = 1.0  # Default to no change if missing

            if (
                self.lastDayData is not None
                and stock in self.lastDayData["stock"].values
            ):
                last_close_val = self.lastDayData[self.lastDayData["stock"] == stock][
                    "close"
                ].values[0]
                # Validate and sanitize NaN/inf/<=0
                if (
                    pd.isna(last_close_val)
                    or np.isinf(last_close_val)
                    or last_close_val <= 0
                ):
                    self.nan_stats["last_close_nan"] += 1
                    cur_date = (
                        self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
                    )
                    fallback_used = None
                    # Fallback: use current price or 1.0
                    if not (
                        pd.isna(cur_close_prices[i])
                        or np.isinf(cur_close_prices[i])
                        or cur_close_prices[i] <= 0
                    ):
                        last_close_prices[i] = cur_close_prices[i]
                        fallback_used = f"current_price({cur_close_prices[i]:.4f})"
                    else:
                        last_close_prices[i] = 1.0
                        fallback_used = "1.0_default"
                    # Log details (limit to first 10 per epoch to avoid spam)
                    if len(self.nan_stats["last_close_details"]) < 10:
                        self.nan_stats["last_close_details"].append(
                            {
                                "day": self.curTradeDay,
                                "date": cur_date,
                                "stock": stock,
                                "value": last_close_val,
                                "fallback": fallback_used,
                            }
                        )
                else:
                    last_close_prices[i] = last_close_val
            else:
                # Validate cur_close_prices before using
                if (
                    pd.isna(cur_close_prices[i])
                    or np.isinf(cur_close_prices[i])
                    or cur_close_prices[i] <= 0
                ):
                    last_close_prices[i] = 1.0
                else:
                    last_close_prices[i] = cur_close_prices[
                        i
                    ]  # Use current if last not available

        curDay_ClosePrice_withSlippage = cur_close_prices * (
            1 + self.cur_slippage_drift
        )
        lastDay_ClosePrice_withSlippage = last_close_prices * (
            1 + self.last_slippage_drift
        )
        rate_of_price_change = curDay_ClosePrice_withSlippage / (
            lastDay_ClosePrice_withSlippage + 1e-8
        )  # Avoid division by zero
        rate_of_price_change_adj = np.where(
            (rate_of_price_change >= 2) & (weights < 0), 2, rate_of_price_change
        )
        sigDayReturn = (
            rate_of_price_change_adj - 1
        ) * weights  # [s1_pct, s2_pct, .., px_pct_returns]
        poDayReturn = np.sum(sigDayReturn)

        # Validate poDayReturn to prevent NaN propagation
        if np.isnan(poDayReturn) or np.isinf(poDayReturn):
            self.nan_stats["poDayReturn_nan"] += 1
            cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
            smart_print(
                f"Warning: poDayReturn is {poDayReturn}, using 0.0 as fallback (Day: {self.curTradeDay}, date: {cur_date})",
                flush=True,
            )
            # Log details (limit to first 10 per epoch to avoid spam)
            if len(self.nan_stats["poDayReturn_details"]) < 10:
                self.nan_stats["poDayReturn_details"].append(
                    {"day": self.curTradeDay, "date": cur_date, "value": poDayReturn}
                )
            poDayReturn = 0.0

        if poDayReturn <= (-1):
            raise ValueError(
                "Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(
                    self.curTradeDay, self.date_memory[-1], poDayReturn
                )
            )

        prev_cur_capital = (
            self.cur_capital
        )  # Save previous value for poDayReturn_withcost calculation
        updatePoValue = self.cur_capital * (poDayReturn + 1)
        # Validate updatePoValue to prevent NaN propagation
        if np.isnan(updatePoValue) or np.isinf(updatePoValue) or updatePoValue <= 0:
            self.nan_stats["updatePoValue_nan"] += 1
            cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else "N/A"
            fallback_cap = (
                self.last_valid_capital
                if np.isfinite(self.last_valid_capital)
                else self.initial_asset
            )
            smart_print(
                f"Warning: updatePoValue is {updatePoValue}, using fallback capital {fallback_cap} (Day: {self.curTradeDay}, date: {cur_date})",
                flush=True,
            )
            # Log details (limit to first 10 per epoch to avoid spam)
            if len(self.nan_stats["updatePoValue_details"]) < 10:
                self.nan_stats["updatePoValue_details"].append(
                    {
                        "day": self.curTradeDay,
                        "date": cur_date,
                        "value": updatePoValue,
                        "prev_capital": prev_cur_capital,
                        "poDayReturn": poDayReturn,
                    }
                )
            updatePoValue = fallback_cap  # Keep previous valid value

        self.cur_capital = updatePoValue
        if np.isfinite(self.cur_capital):
            self.last_valid_capital = self.cur_capital
        poDayReturn_withcost = (updatePoValue - prev_cur_capital) / (
            prev_cur_capital + 1e-8
        )  # Avoid division by zero

        self.profit_lst.append(poDayReturn_withcost)
        self.asset_lst.append(self.cur_capital)

        # Build MAFIA state via market observer
        cur_risk_boundary, stock_ma_price = self.run_mkt_observer(
            stage="run",
            rate_of_price_change=np.array([rate_of_price_change]),
            selection_trigger=selection_trigger,
            selection_reason=selection_reason,
        )
        if stock_ma_price is not None:
            self.ctl_state["MA-{}".format(self.config.otherRef_indicator_ma_window)] = (
                stock_ma_price
            )
        self.risk_adj_lst.append(cur_risk_boundary)
        self.ctrl_weight_lst.append(1.0)
        selection_fired = bool(
            getattr(self, "_last_selection_triggered", selection_trigger)
        )
        if selection_fired:
            self.days_since_rebalance = 0
        else:
            self.days_since_rebalance += 1

        # Mid-epoch observer training to keep buffers small
        if (
            self.mode == "train"
            and self.config.enable_market_observer
            and self.mkt_observer is not None
            and allow_observer_training
            and getattr(self.config, "observer_mini_epoch_steps", 0) > 0
        ):
            self._observer_mini_step_counter += 1
            if (
                self._observer_mini_step_counter
                >= self.config.observer_mini_epoch_steps
            ):
                try:
                    # train() returns True if training happened, False if skipped (warmup)
                    trained = self.mkt_observer.train(mode=self.mode)
                    if trained:
                        smart_print(
                            f"[MAFIA][OBSERVER] ✅ Mini-epoch train @ step {self.curTradeDay}/{self.totalTradeDay} "
                            f"(window={self.config.observer_mini_epoch_steps})",
                            flush=True,
                        )
                        self._observer_mini_step_counter = 0  # Only reset if trained
                    # If not trained (warmup), don't reset counter - will try again next mini-epoch
                except Exception as e:
                    smart_print(
                        f"[MAFIA] Warning: observer mini-epoch train failed: {e}",
                        flush=True,
                    )
                    self._observer_mini_step_counter = 0

        daily_return_ay = self.curData[
            "DAILYRETURNS-{}".format(self.config.dailyRetun_lookback)
        ].values
        cur_cov = _build_cov_from_indicator(
            daily_return_ay, target_size=len(weights), fallback=self.config.risk_market
        )

        # Calculate risk with safe matmul
        try:
            # Ensure weights is 1D
            weights_1d = weights.flatten() if weights.ndim > 1 else weights
            # weights @ cur_cov @ weights.T
            temp = np.matmul(weights_1d, cur_cov)  # (N,) @ (N, N) -> (N,)
            risk_cbf = np.sqrt(np.matmul(temp, weights_1d))  # (N,) @ (N,) -> scalar
            if np.isnan(risk_cbf) or np.isinf(risk_cbf):
                risk_cbf = self.config.risk_market
            # Ensure scalar
            if isinstance(risk_cbf, np.ndarray):
                risk_cbf = float(
                    risk_cbf.item() if risk_cbf.size == 1 else risk_cbf.flat[0]
                )
            self.risk_cbf_lst.append(float(risk_cbf))
        except (ValueError, TypeError, IndexError) as e:
            self.risk_cbf_lst.append(self.config.risk_market)

        w_rl = self._get_full_action_from_memory(self.action_rl_memory, -1)
        w_rl = w_rl / (np.sum(np.abs(w_rl)) + 1e-8)  # Avoid division by zero
        try:
            # w_rl @ cur_cov @ w_rl.T
            temp = np.matmul(w_rl, cur_cov)  # (N,) @ (N, N) -> (N,)
            risk_raw = np.sqrt(np.matmul(temp, w_rl))  # (N,) @ (N,) -> scalar
            if np.isnan(risk_raw) or np.isinf(risk_raw):
                risk_raw = self.config.risk_market
            # Ensure scalar
            if isinstance(risk_raw, np.ndarray):
                risk_raw = float(
                    risk_raw.item() if risk_raw.size == 1 else risk_raw.flat[0]
                )
            self.risk_raw_lst.append(float(risk_raw))
        except (ValueError, TypeError, IndexError) as e:
            self.risk_raw_lst.append(self.config.risk_market)

        # Compute last_risk_violation for next step observation (Spec Section 3.4)
        # δ_risk = max(0, σ(a^RL) - σ_target)
        try:
            sigma_rl = self.risk_raw_lst[-1]
            sigma_target = float(cur_risk_boundary)
            self.last_risk_violation = float(max(0.0, sigma_rl - sigma_target))
        except (IndexError, TypeError, ValueError):
            self.last_risk_violation = 0.0

        if self.curTradeDay == 1:
            prev_rl_cap = self.return_raw_lst[-1] * (1 - self.transaction_cost)
        else:
            prev_rl_cap = self.return_raw_lst[-1]

        rate_of_price_change_adj_rawrl = np.where(
            (rate_of_price_change >= 2) & (w_rl < 0), 2, rate_of_price_change
        )
        po_r_rl = np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl)
        if po_r_rl <= (-1):
            raise ValueError(
                "Loss the whole capital if using RL actions only! [Day: {}, date: {}, po_r_rl: {}]".format(
                    self.curTradeDay, self.date_memory[-1], po_r_rl
                )
            )
        return_raw = prev_rl_cap * (po_r_rl + 1)
        self.return_raw_lst.append(return_raw)

        # CVaR
        # daily_return_ay is 1D, take last 21 elements
        if daily_return_ay.ndim == 1:
            expected_r_series = (
                daily_return_ay[-21:] if len(daily_return_ay) >= 21 else daily_return_ay
            )
            expected_r_prev = (
                np.mean(expected_r_series[-1:]) if len(expected_r_series) > 0 else 0.0
            )
            # For 1D, expected_r_prev is scalar, convert to array matching weights
            expected_r_prev = np.full(len(weights), expected_r_prev)
            expected_r_prev = np.where(
                (expected_r_prev >= 1) & (weights < 0), 1, expected_r_prev
            )
            expected_r = np.sum(expected_r_prev * weights)
            # For 1D series, covariance is scalar, create diagonal matrix
            if len(expected_r_series) < 2:
                expected_cov = np.eye(len(weights)) * self.config.risk_market
            else:
                expected_cov_val = np.var(expected_r_series)
                expected_cov = np.eye(len(weights)) * expected_cov_val
        else:
            expected_r_series = (
                daily_return_ay[:, -21:]
                if daily_return_ay.shape[1] >= 21
                else daily_return_ay
            )
            expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prev = np.where(
                (expected_r_prev >= 1) & (weights < 0), 1, expected_r_prev
            )
            expected_r = np.sum(
                np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights, (-1, 1))
            )
            expected_cov = np.cov(expected_r_series)
            # Ensure expected_cov is 2D and matches weights shape
            if expected_cov.ndim == 0:
                expected_cov = np.eye(len(weights)) * float(expected_cov)
            elif expected_cov.ndim == 1:
                expected_cov = np.diag(expected_cov)
            if expected_cov.shape[0] != len(weights) or expected_cov.shape[1] != len(
                weights
            ):
                if expected_cov.size == 1:
                    expected_cov = np.eye(len(weights)) * float(expected_cov.flat[0])
                else:
                    target_size = len(weights)
                    if expected_cov.shape[0] < target_size:
                        pad_size = target_size - expected_cov.shape[0]
                        expected_cov = np.pad(
                            expected_cov,
                            ((0, pad_size), (0, pad_size)),
                            mode="constant",
                            constant_values=self.config.risk_market,
                        )
                    elif expected_cov.shape[0] > target_size:
                        expected_cov = expected_cov[:target_size, :target_size]

        # Calculate expected_std safely
        weights_1d = weights.flatten() if weights.ndim > 1 else weights
        try:
            temp = np.matmul(weights_1d, expected_cov)  # (N,) @ (N, N) -> (N,)
            expected_std = np.sqrt(np.matmul(temp, weights_1d))  # (N,) @ (N,) -> scalar
            if isinstance(expected_std, np.ndarray):
                expected_std = float(
                    expected_std.item()
                    if expected_std.size == 1
                    else expected_std.flat[0]
                )
        except (ValueError, TypeError, IndexError):
            expected_std = self.config.risk_market
        cvar_lz = spstats.norm.ppf(
            1 - 0.05
        )  # positive 1.65 for 95%(=1-alpha) confidence level.
        cvar_Z = np.exp(-0.5 * np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2 * np.pi)
        cvar_expected = -expected_r + expected_std * cvar_Z
        self.cvar_lst.append(cvar_expected)

        # CVaR without risk controller
        if expected_r_series.ndim == 1:
            expected_r_prevrl = (
                np.mean(expected_r_series[-1:]) if len(expected_r_series) > 0 else 0.0
            )
            expected_r_prevrl = np.full(len(w_rl), expected_r_prevrl)
            expected_r_prevrl = np.where(
                (expected_r_prevrl >= 1) & (w_rl < 0), 1, expected_r_prevrl
            )
            expected_r_raw = np.sum(expected_r_prevrl * w_rl)
        else:
            expected_r_prevrl = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prevrl = np.where(
                (expected_r_prevrl >= 1) & (w_rl < 0), 1, expected_r_prevrl
            )
            expected_r_raw = np.sum(
                np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1))
            )

        # Calculate expected_std_raw safely
        w_rl_1d = w_rl.flatten() if w_rl.ndim > 1 else w_rl
        try:
            temp = np.matmul(w_rl_1d, expected_cov)  # (N,) @ (N, N) -> (N,)
            expected_std_raw = np.sqrt(
                np.matmul(temp, w_rl_1d)
            )  # (N,) @ (N,) -> scalar
            if isinstance(expected_std_raw, np.ndarray):
                expected_std_raw = float(
                    expected_std_raw.item()
                    if expected_std_raw.size == 1
                    else expected_std_raw.flat[0]
                )
        except (ValueError, TypeError, IndexError):
            expected_std_raw = self.config.risk_market
        cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
        self.cvar_raw_lst.append(cvar_expected_raw)

        # ============ PORTFOLIO ALLOCATOR REWARD (NEW 2-COMPONENT) ============
        if self.use_pa_reward:
            # Extract a_alloc (raw RL output) and a_final (Controller adjusted)
            a_alloc = None
            a_final = None
            if len(self.action_rl_memory) > 0:
                a_alloc = np.array(self.action_rl_memory[-1], dtype=np.float64)
            if len(self.action_cbf_memeory) > 0:
                cbf_adj = np.array(self.action_cbf_memeory[-1], dtype=np.float64)
                if (
                    a_alloc is not None
                    and cbf_adj is not None
                    and len(cbf_adj) == len(a_alloc)
                ):
                    a_final = a_alloc + cbf_adj
                else:
                    a_final = a_alloc

            # Fallback: use current weights as a_final if not available
            if a_alloc is None:
                a_alloc = weights
            if a_final is None:
                a_final = weights

            # Compute reward using new 2-component formula
            reward_components = compute_pa_reward(
                portfolio_return=poDayReturn_withcost,
                a_alloc=a_alloc,
                a_final=a_final,
                w_return=self.pa_w_return,
                lambda_js=self.pa_lambda_js,
                reward_scale=self.pa_reward_scale,
            )

            cur_reward = reward_components["reward_total"]

            # Track components for logging
            self.rl_reward_return_lst.append(reward_components["r_return"])
            self.rl_reward_divergence_lst.append(reward_components["r_divergence"])

            # Store latest reward for realtime logging
            self._last_reward_info = {
                "log_return": reward_components["log_return"],
                "total_reward": cur_reward,
            }

            # Update LiveDisplay with reward components
            if LIVE_DISPLAY_AVAILABLE:
                try:
                    display_update_reward_components(
                        reward_return=reward_components["r_return"],
                        reward_js=reward_components["r_divergence"],
                        reward_total=cur_reward,
                        reward_unscaled=reward_components["reward_unscaled"],
                        js_divergence=reward_components["js_divergence"],
                        w_return=self.pa_w_return,
                        lambda_js=self.pa_lambda_js,
                        reward_scale=self.pa_reward_scale,
                    )
                except Exception:
                    pass

            # Optional verbose debug log - Portfolio Allocator reward components (separate lines)
            if getattr(self.config, "mafia_log_reward", False) and (
                len(self.reward_lst) < 10 or len(self.reward_lst) % 100 == 0
            ):
                smart_print(
                    f"\n[PA-REWARD] Portfolio Allocator reward breakdown: "
                    f"log_return={reward_components['log_return']:.6f} | "
                    f"r_return={reward_components['r_return']:.6f} | "
                    f"js_divergence={reward_components['js_divergence']:.6f} | "
                    f"r_divergence={reward_components['r_divergence']:.6f} | "
                    f"unscaled={reward_components['reward_unscaled']:.6f} | "
                    f"scaled(×{self.pa_reward_scale:.0f})={cur_reward:.6f}",
                    flush=True,
                )
        else:
            # ============ LEGACY REWARD (ORIGINAL) ============
            # Ensure poDayReturn_withcost is valid before taking log
            if np.isnan(poDayReturn_withcost) or np.isinf(poDayReturn_withcost):
                poDayReturn_withcost = 0.0
            # Avoid log(0) = -inf when poDayReturn_withcost = -1
            if poDayReturn_withcost <= -1:
                profit_part = -10.0  # Large negative value instead of -inf
            else:
                profit_part = np.log(poDayReturn_withcost + 1)
            # Unified reward: log-return + Jensen-Shannon diversity
            if len(self.action_rl_memory) > 0:
                w_rl_latest = self.action_rl_memory[-1]
                # Expand Top-K action to full universe space if needed
                if (
                    len(w_rl_latest) != len(weights)
                    and len(self.action_topk_indices_memory) > 0
                ):
                    w_rl_full = np.zeros(len(weights), dtype=np.float32)
                    topk_idx = self.action_topk_indices_memory[-1]
                    if len(topk_idx) == len(w_rl_latest):
                        w_rl_full[topk_idx] = w_rl_latest
                        w_rl_latest = w_rl_full
            else:
                w_rl_latest = weights
            weights_norm = _normalize_prob(weights)
            w_rl_norm = _normalize_prob(w_rl_latest)
            js_m = 0.5 * (w_rl_norm + weights_norm)
            js_divergence = 0.5 * entropy(
                pk=w_rl_norm, qk=js_m, base=2
            ) + 0.5 * entropy(pk=weights_norm, qk=js_m, base=2)
            if np.isnan(js_divergence) or np.isinf(js_divergence):
                js_divergence = 0.0
            j_return = profit_part  # log-return at current step
            scaled_profit_part = self.config.lambda_1 * j_return
            scaled_js_part = self.config.lambda_2 * js_divergence
            turnover_penalty = self.lambda_tc * turnover_amt
            # Membership change penalty: phạt số mã ra/vào Top-K giữa hai ngày
            change_penalty = 0.0
            try:
                if len(self.actions_memory) > 1:
                    k = getattr(self.config, "topK", 10)
                    prev_w = np.array(self.actions_memory[-2])
                    curr_w = np.array(weights)
                    # Bỏ tiền mặt nếu có: actions_memory lưu weights đã chuẩn hóa (không cash)
                    if prev_w.shape[0] == curr_w.shape[0] + 1:
                        prev_w = prev_w[1:]
                    if curr_w.shape[0] == prev_w.shape[0] + 1:
                        curr_w = curr_w[1:]
                    k = max(1, min(k, len(curr_w)))
                    prev_top = set(np.argsort(prev_w)[-k:])
                    curr_top = set(np.argsort(curr_w)[-k:])
                    sym_diff = prev_top.symmetric_difference(curr_top)
                    # Chuẩn hóa theo k để scale 0..2
                    change_penalty = (len(sym_diff) / float(k)) * getattr(
                        self.config, "lambda_change", 0.0
                    )
            except Exception as e:
                smart_print(f"[Reward] change_penalty skipped: {e}", flush=True)
                change_penalty = 0.0

            penalty_enabled = not getattr(
                self.config, "rl_reward_disable_turnover_change", True
            )
            total_penalty = (
                (turnover_penalty + change_penalty) if penalty_enabled else 0.0
            )
            cur_reward = scaled_profit_part - scaled_js_part - total_penalty

            # Optional debug log for reward components (early train steps)
            self._log_reward_debug(
                j_return,
                scaled_profit_part,
                js_divergence,
                scaled_js_part,
                turnover_amt,
                turnover_penalty if penalty_enabled else 0.0,
                change_penalty if penalty_enabled else 0.0,
                cur_reward,
                penalty_enabled,
            )

            self.rl_reward_risk_lst.append(scaled_js_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)

            # Update LiveDisplay with legacy reward components
            if LIVE_DISPLAY_AVAILABLE:
                # Update dashboard components
                update_reward_components(
                    reward_return=scaled_profit_part,
                    reward_js=-scaled_js_part,  # Negative because it's subtracted
                    reward_total=cur_reward,
                    reward_unscaled=j_return - js_divergence,
                    js_divergence=js_divergence,
                    w_return=self.config.lambda_1,
                    lambda_js=self.config.lambda_2,
                    reward_scale=1.0,  # Legacy reward has no scaling
                )

        # Ensure cur_reward is not NaN or inf (root cause fix)
        if np.isnan(cur_reward) or np.isinf(cur_reward):
            smart_print(
                f"Warning: cur_reward is {cur_reward}, replacing with 0.0",
                flush=True,
            )
            cur_reward = 0.0
        self.reward = cur_reward
        self.reward_lst.append(self.reward)
        self.model_save_flag = False
        if self.use_multibranch_state:
            self.peak_capital = max(self.peak_capital, self.cur_capital)

        # Update LiveDisplay with step info
        # Note: selection_trigger, selection_reason, poDayReturn_withcost are always defined
        # in the non-terminal path where this code runs
        self._update_live_display(
            step_return=poDayReturn_withcost,
            reward=self.reward,
            selection_trigger=selection_trigger,
            selection_reason=selection_reason,
        )

        # Return format compatible with both gym and gymnasium
        # Always use gymnasium format (terminated, truncated, info) for consistency with VecEnv
        return self.state, self.reward, self.terminal, False, self._build_step_info()

    def reset(self, seed=None, options=None):
        # Handle gymnasium compatibility (seed and options parameters)
        if seed is not None:
            self.seed_num = seed
            np.random.seed(seed)
        self.epoch = self.epoch + 1
        self.curTradeDay = 0

        # === OBSERVER WARMUP-AWARE RESET ===
        # Determine if we should preserve warmup buffers across episode boundary
        allow_observer_training = bool(
            getattr(self.config, "mafia_allow_observer_training", False)
        )
        preserve_warmup = False
        if (
            hasattr(self, "mkt_observer")
            and self.mkt_observer is not None
            and allow_observer_training
            and self.mode == "train"
        ):
            # Check if Observer is still in warmup phase
            warmup_complete = getattr(self.mkt_observer, "_warmup_complete", False)
            warmup_samples = getattr(self.config, "observer_warmup_samples", 300)
            current_buffer = len(getattr(self.mkt_observer, "topk_scores_lst", []))
            if not warmup_complete and current_buffer < warmup_samples:
                preserve_warmup = True

            # Reset observer with warmup preservation if needed
            self.mkt_observer.reset(preserve_warmup=preserve_warmup)

        # === UPDATE GUMBEL TEMPERATURE FOR MAFIA OBSERVER ===
        # Temperature annealing: temp = max(min, init * decay^episode)
        if (
            hasattr(self, "mkt_observer")
            and self.mkt_observer is not None
            and hasattr(self.mkt_observer, "update_temperature")
        ):
            new_temp = self.mkt_observer.update_temperature(self.epoch)
            if self.epoch % 10 == 0:
                smart_print(
                    f"[MAFIA] Epoch {self.epoch}: Gumbel Temperature = {new_temp:.4f}",
                    flush=True,
                )

        # Reset NaN statistics for new epoch
        self.nan_stats = {
            "cur_close_nan": 0,
            "last_close_nan": 0,
            "poDayReturn_nan": 0,
            "updatePoValue_nan": 0,
            "cur_close_details": [],
            "last_close_details": [],
            "poDayReturn_details": [],
            "updatePoValue_details": [],
        }

        self.curData = self._ensure_dataframe(
            self.rawdata.loc[self.curTradeDay, :].copy()
        )
        self.curData.sort_values(["stock"], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        self._ensure_minimal_features()

        # Initialize state (will be built in run_mkt_observer)
        self.state = np.zeros(self.state_dim, dtype=np.float32)

        self.ctl_state = self._build_ctl_state()
        self.terminal = False

        self.profit_lst = [0]
        cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage="reset")
        if stock_ma_price is not None:
            self.ctl_state["MA-{}".format(self.config.otherRef_indicator_ma_window)] = (
                stock_ma_price
            )

        self.cur_capital = self.initial_asset
        self.last_valid_capital = self.cur_capital

        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]

        self.asset_lst = [self.initial_asset]

        self.actions_memory = [
            np.array([1 / self.stock_num] * self.stock_num) * self.bound_flag
        ]
        self.date_memory = [self.curData["date"].unique()[0]]
        self.reward_lst = [0]
        init_topk = max(1, getattr(self, "rl_stock_num", self.stock_num))
        self.action_topk_indices_memory = [self._get_current_topk_indices()]
        self.action_cbf_memeory = [np.zeros(init_topk, dtype=float)]
        self.action_rl_memory = [
            np.array([1 / init_topk] * init_topk) * self.bound_flag
        ]

        self.risk_adj_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0]
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset]
        # Track selection mask for cadence alignment (1 = rebalance day, 0 = holding day)
        self._selection_mask_lst = [1]  # First day is always rebalance
        self.solver_stat = {
            "solvable": 0,
            "insolvable": 0,
            "stochastic_solvable": 0,
            "stochastic_time": [],
            "socp_solvable": 0,
            "socp_time": [],
        }

        self.ctrl_weight_lst = [1.0]
        self.solvable_flag = []
        self.risk_pred_lst = []

        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.cnt1 = 0
        self.cnt2 = 0
        self._observer_mini_step_counter = 0
        self.stepcount = 0
        self.latest_invest_profile = None
        self.days_since_rebalance = 0
        self.force_rebalance = False
        self.emergency_rebalance = False
        # Reset Contextual Awareness Features
        self.last_topk_avg_return = 0.0
        self.last_relative_alpha = 0.0
        self.last_risk_violation = 0.0
        # Reset Rebalance Signals
        self.last_rebalance_scheduled = False
        self.last_rebalance_regime = False
        self.start_cputime = time.process_time()
        self.start_systime = time.perf_counter()

        # Reset resuming flag after reset (normal training state)
        self._is_resuming = False

        # Return format compatible with both gym and gymnasium
        # gymnasium: (obs, info)
        # gym: obs
        if GYMNASIUM_AVAILABLE:
            return self.state, {}
        else:
            return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        if np.sum(np.abs(actions)) == 0:
            norm_weights = np.array([1 / len(actions)] * len(actions)) * self.bound_flag
        else:
            # Proper softmax: exp(x_i) / sum(exp(x_j))
            # For short positions (bound_flag = -1), negate actions first
            if self.bound_flag < 0:
                actions = -actions  # Flip actions for short positions
            exp_actions = np.exp(
                actions - np.max(actions)
            )  # Subtract max for numerical stability
            norm_weights = exp_actions / np.sum(exp_actions)
            norm_weights = norm_weights * self.bound_flag
        return norm_weights

    def sum_normalization(self, actions):
        if np.sum(np.abs(actions)) == 0:
            norm_weights = np.array([1 / len(actions)] * len(actions)) * self.bound_flag
        else:
            # Normalize by sum of absolute values, then apply bound_flag
            # For long positions: actions / sum(|actions|) -> sums to 1
            # For short positions: -actions / sum(|actions|) -> sums to -1
            if self.bound_flag < 0:
                actions = -actions  # Flip actions for short positions
            norm_weights = actions / np.sum(np.abs(actions))
            # Ensure the result has the correct sign and magnitude
            norm_weights = norm_weights * abs(self.bound_flag)
        return norm_weights

    def _estimate_min_variance_risk_for_mask(self, mask=None):
        """
        Estimate daily min-variance risk (sqrt(w^T Σ w)) for a given asset mask.
        mask: bool array length stock_num; if None uses all assets.
        """
        key = f"DAILYRETURNS-{self.config.dailyRetun_lookback}"
        daily_returns = self.ctl_state.get(key, None)
        if daily_returns is None:
            return None
        try:
            daily_returns = np.array(daily_returns, dtype=float)
            daily_returns = np.nan_to_num(
                daily_returns, nan=0.0, posinf=0.0, neginf=0.0
            )
            if daily_returns.ndim == 1:
                daily_returns = daily_returns.reshape(1, -1)
            if (
                daily_returns.shape[0] != self.stock_num
                and daily_returns.shape[1] == self.stock_num
            ):
                daily_returns = daily_returns.T
            # Only compute once enough history is available; skip to avoid cov warnings
            required_cols = max(2, getattr(self.config, "dailyRetun_lookback", 2))
            if daily_returns.shape[1] < required_cols:
                return None
            if mask is not None:
                mask = np.array(mask, dtype=bool)
                if mask.shape[0] != self.stock_num or not np.any(mask):
                    return None
                daily_returns = daily_returns[mask]
            cov_matrix = np.cov(daily_returns)
            cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[cov_matrix]])
            elif cov_matrix.ndim == 1:
                cov_matrix = np.diag(cov_matrix)
            cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            ones = np.ones(cov_matrix.shape[0])
            inv_cov = np.linalg.pinv(cov_matrix)
            w = inv_cov @ ones
            if np.sum(w) <= 1e-10:
                return None
            w = np.maximum(w, 0.0)
            if np.sum(w) <= 1e-10:
                return None
            w = w / np.sum(w)
            risk_val = float(np.matmul(w, np.matmul(cov_matrix, w.T)))
            return float(np.sqrt(max(risk_val, 0.0)))
        except Exception:
            return None

    def save_action_memory(self):
        action_arr = np.array(self.actions_memory)
        columns = list(self.stock_lst)
        if len(columns) != action_arr.shape[1]:
            if len(columns) > action_arr.shape[1]:
                columns = columns[: action_arr.shape[1]]
            else:
                columns = columns + [
                    f"UNNAMED_{i}" for i in range(action_arr.shape[1] - len(columns))
                ]
            smart_print(
                f"[SAVE_PROFILE] Warning: stock list length mismatch (actions: {action_arr.shape[1]}, columns: {len(self.stock_lst)}). Adjusting columns for consistency.",
                flush=True,
            )
        action_pd = pd.DataFrame(action_arr, columns=columns)
        action_pd["date"] = self.date_memory
        return action_pd

    def seed(self, seed=2022):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def save_state(self, filepath):
        """
        Save environment state to file for checkpoint resume.
        This allows resuming from mid-epoch checkpoints.
        """
        import pickle

        state = {
            "epoch": self.epoch,
            "curTradeDay": self.curTradeDay,
            "cur_capital": self.cur_capital,
            "curData": self.curData.copy()
            if hasattr(self, "curData") and self.curData is not None
            else None,
            "lastDayData": self.lastDayData.copy()
            if hasattr(self, "lastDayData") and self.lastDayData is not None
            else None,
            "cur_slippage_drift": self.cur_slippage_drift.copy()
            if hasattr(self, "cur_slippage_drift")
            else None,
            "last_slippage_drift": self.last_slippage_drift.copy()
            if hasattr(self, "last_slippage_drift")
            else None,
            "state": self._copy_observation(self.state)
            if hasattr(self, "state") and self.state is not None
            else None,
            "ctl_state": {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in self.ctl_state.items()
            }
            if hasattr(self, "ctl_state")
            else None,
            "terminal": self.terminal if hasattr(self, "terminal") else False,
            "profit_lst": self.profit_lst.copy()
            if hasattr(self, "profit_lst")
            else None,
            "asset_lst": self.asset_lst.copy() if hasattr(self, "asset_lst") else None,
            "actions_memory": [
                a.copy() if isinstance(a, np.ndarray) else a
                for a in self.actions_memory
            ]
            if hasattr(self, "actions_memory")
            else None,
            "date_memory": self.date_memory.copy()
            if hasattr(self, "date_memory")
            else None,
            "reward_lst": self.reward_lst.copy()
            if hasattr(self, "reward_lst")
            else None,
            "action_cbf_memeory": [
                a.copy() if isinstance(a, np.ndarray) else a
                for a in self.action_cbf_memeory
            ]
            if hasattr(self, "action_cbf_memeory")
            else None,
            "action_rl_memory": [
                a.copy() if isinstance(a, np.ndarray) else a
                for a in self.action_rl_memory
            ]
            if hasattr(self, "action_rl_memory")
            else None,
            "action_topk_indices_memory": [
                a.copy() if isinstance(a, np.ndarray) else np.array(a)
                for a in getattr(self, "action_topk_indices_memory", [])
            ],
            "risk_adj_lst": self.risk_adj_lst.copy()
            if hasattr(self, "risk_adj_lst")
            else None,
            "risk_raw_lst": self.risk_raw_lst.copy()
            if hasattr(self, "risk_raw_lst")
            else None,
            "risk_cbf_lst": self.risk_cbf_lst.copy()
            if hasattr(self, "risk_cbf_lst")
            else None,
            "return_raw_lst": self.return_raw_lst.copy()
            if hasattr(self, "return_raw_lst")
            else None,
            "cvar_lst": self.cvar_lst.copy() if hasattr(self, "cvar_lst") else None,
            "cvar_raw_lst": self.cvar_raw_lst.copy()
            if hasattr(self, "cvar_raw_lst")
            else None,
            "rl_reward_risk_lst": self.rl_reward_risk_lst.copy()
            if hasattr(self, "rl_reward_risk_lst")
            else None,
            "rl_reward_profit_lst": self.rl_reward_profit_lst.copy()
            if hasattr(self, "rl_reward_profit_lst")
            else None,
            "is_last_ctrl_solvable": self.is_last_ctrl_solvable
            if hasattr(self, "is_last_ctrl_solvable")
            else False,
            "nan_stats": self.nan_stats.copy() if hasattr(self, "nan_stats") else None,
            "latest_invest_profile": self.latest_invest_profile,
            "last_epoch_profile": self.last_epoch_profile,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        smart_print(f"[ENV STATE] Saved environment state to {filepath}", flush=True)

    def restore_state(self, filepath):
        """
        Restore environment state from file for checkpoint resume.
        This allows resuming from mid-epoch checkpoints.
        """
        import pickle

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.epoch = state.get("epoch", 0)
        self.curTradeDay = state.get("curTradeDay", 0)
        self.cur_capital = state.get("cur_capital", self.initial_asset)
        self.last_valid_capital = (
            self.cur_capital if np.isfinite(self.cur_capital) else self.initial_asset
        )

        if state.get("curData") is not None:
            self.curData = state["curData"].copy()
        if state.get("lastDayData") is not None:
            self.lastDayData = state["lastDayData"].copy()
        if state.get("cur_slippage_drift") is not None:
            self.cur_slippage_drift = state["cur_slippage_drift"].copy()
        if state.get("last_slippage_drift") is not None:
            self.last_slippage_drift = state["last_slippage_drift"].copy()
        if state.get("state") is not None:
            self.state = self._copy_observation(state["state"])
        if state.get("ctl_state") is not None:
            self.ctl_state = {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in state["ctl_state"].items()
            }
        if state.get("terminal") is not None:
            self.terminal = state["terminal"]
        if state.get("profit_lst") is not None:
            self.profit_lst = state["profit_lst"].copy()
        if state.get("asset_lst") is not None:
            self.asset_lst = state["asset_lst"].copy()
        if state.get("actions_memory") is not None:
            self.actions_memory = [
                a.copy() if isinstance(a, np.ndarray) else a
                for a in state["actions_memory"]
            ]
        if state.get("date_memory") is not None:
            self.date_memory = state["date_memory"].copy()
        if state.get("reward_lst") is not None:
            self.reward_lst = state["reward_lst"].copy()
        if state.get("action_cbf_memeory") is not None:
            self.action_cbf_memeory = [
                a.copy() if isinstance(a, np.ndarray) else a
                for a in state["action_cbf_memeory"]
            ]
        if state.get("action_rl_memory") is not None:
            self.action_rl_memory = [
                a.copy() if isinstance(a, np.ndarray) else a
                for a in state["action_rl_memory"]
            ]
        if state.get("action_topk_indices_memory") is not None:
            self.action_topk_indices_memory = [
                np.array(a) for a in state["action_topk_indices_memory"]
            ]
        if state.get("risk_adj_lst") is not None:
            self.risk_adj_lst = state["risk_adj_lst"].copy()
        if state.get("risk_raw_lst") is not None:
            self.risk_raw_lst = state["risk_raw_lst"].copy()
        if state.get("risk_cbf_lst") is not None:
            self.risk_cbf_lst = state["risk_cbf_lst"].copy()
        if state.get("return_raw_lst") is not None:
            self.return_raw_lst = state["return_raw_lst"].copy()
        if state.get("cvar_lst") is not None:
            self.cvar_lst = state["cvar_lst"].copy()
        if state.get("cvar_raw_lst") is not None:
            self.cvar_raw_lst = state["cvar_raw_lst"].copy()
        if state.get("rl_reward_risk_lst") is not None:
            self.rl_reward_risk_lst = state["rl_reward_risk_lst"].copy()
        if state.get("rl_reward_profit_lst") is not None:
            self.rl_reward_profit_lst = state["rl_reward_profit_lst"].copy()
        if state.get("is_last_ctrl_solvable") is not None:
            self.is_last_ctrl_solvable = state["is_last_ctrl_solvable"]
        if state.get("nan_stats") is not None:
            self.nan_stats = state["nan_stats"].copy()
        self.latest_invest_profile = state.get("latest_invest_profile", None)
        self.last_epoch_profile = state.get(
            "last_epoch_profile", self.latest_invest_profile
        )

        # Set flag to indicate we're resuming from checkpoint (affects save_profile logic)
        self._is_resuming = True

        smart_print(
            f"[ENV STATE] Restored environment state from {filepath} | epoch={self.epoch}, day={self.curTradeDay}, capital={self.cur_capital:.2f}",
            flush=True,
        )

    def get_results(self):
        # Create copies to avoid modifying original lists during ongoing episodes
        profit_lst = np.array(self.profit_lst)
        asset_lst = np.array(self.asset_lst)

        netProfit = self.cur_capital - self.initial_asset  # Profits
        netProfit_pct = netProfit / self.initial_asset  # Rate of overall returns

        diffPeriodAsset = np.diff(asset_lst)
        if len(diffPeriodAsset) > 0:
            sigReturn_max = np.max(
                diffPeriodAsset
            )  # Maximal returns in a single transaction.
            sigReturn_min = np.min(
                diffPeriodAsset
            )  # Minimal returns in a single transaction
        else:
            sigReturn_max = 0.0
            sigReturn_min = 0.0

        # Annual Returns
        annualReturn_pct = (
            np.power(
                (1 + netProfit_pct), (self.config.tradeDays_per_year / len(asset_lst))
            )
            - 1
        )

        dailyReturn_pct_max = np.max(profit_lst)
        dailyReturn_pct_min = np.min(profit_lst)
        avg_dailyReturn_pct = np.mean(profit_lst)
        # strategy volatility
        volatility = np.sqrt(
            np.sum(np.power((profit_lst - avg_dailyReturn_pct), 2))
            * self.config.tradeDays_per_year
            / (len(profit_lst) - 1)
        )
        # Avoid division by zero
        if volatility == 0 or np.isnan(volatility) or np.isinf(volatility):
            volatility = 1e-6  # Small epsilon to avoid division by zero

        # SR_Vol, Long-term risk
        sharpeRatio = (
            (annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name]
        ) / (volatility * 100)
        # Handle NaN/inf
        if np.isnan(sharpeRatio) or np.isinf(sharpeRatio):
            sharpeRatio = 0.0
        # sharpeRatio = np.max([sharpeRatio, 0])

        dailyAnnualReturn_lst = (
            np.power((1 + profit_lst), self.config.tradeDays_per_year) - 1
        )
        dailyRisk_lst = np.array(self.risk_cbf_lst) * np.sqrt(
            self.config.tradeDays_per_year
        )  # Daily Risk to Anuual Risk

        # Handle length mismatch between profit_lst and risk_cbf_lst
        # This can occur during epoch transitions or checkpoint resumption
        min_len = min(len(dailyAnnualReturn_lst), len(dailyRisk_lst))
        if min_len < 2:
            # Not enough data for daily SR calculation
            dailySR = np.array([0.0])
            dailySR_max = 0.0
            dailySR_min = 0.0
            dailySR_avg = 0.0
        else:
            # Truncate both arrays to same length
            dailyAnnualReturn_lst = dailyAnnualReturn_lst[:min_len]
            dailyRisk_lst = dailyRisk_lst[:min_len]

            # Avoid division by zero
            dailyRisk_lst_safe = np.where(dailyRisk_lst == 0, 1e-6, dailyRisk_lst)
            dailySR = (
                (dailyAnnualReturn_lst[1:] * 100)
                - self.config.mkt_rf[self.config.market_name]
            ) / (dailyRisk_lst_safe[1:] * 100)
            # Handle NaN/inf
            dailySR = np.where(np.isnan(dailySR) | np.isinf(dailySR), 0.0, dailySR)
            dailySR = np.append(0, dailySR)
            # dailySR = np.where(dailySR < 0, 0, dailySR)
            dailySR_max = np.max(dailySR)
            dailySR_nonzero = dailySR[dailySR != 0]
            dailySR_min = np.min(dailySR_nonzero) if len(dailySR_nonzero) > 0 else 0.0
            dailySR_avg = np.mean(dailySR)

        # For performance analysis
        # Avoid division by zero
        return_raw_array = np.array(self.return_raw_lst)
        return_raw_safe = np.where(return_raw_array == 0, 1e-6, return_raw_array)
        dailyReturnRate_wocbf = np.diff(return_raw_array) / return_raw_safe[:-1]
        dailyReturnRate_wocbf = np.append(0, dailyReturnRate_wocbf)
        # Handle NaN/inf
        dailyReturnRate_wocbf = np.where(
            np.isnan(dailyReturnRate_wocbf) | np.isinf(dailyReturnRate_wocbf),
            0.0,
            dailyReturnRate_wocbf,
        )
        dailyAnnualReturn_wocbf_lst = (
            np.power((1 + dailyReturnRate_wocbf), self.config.tradeDays_per_year) - 1
        )
        dailyRisk_wocbf_lst = np.array(self.risk_raw_lst) * np.sqrt(
            self.config.tradeDays_per_year
        )
        # Avoid division by zero
        dailyRisk_wocbf_lst_safe = np.where(
            dailyRisk_wocbf_lst == 0, 1e-6, dailyRisk_wocbf_lst
        )

        # Handle length mismatch between return and risk lists
        min_len_wocbf = min(
            len(dailyAnnualReturn_wocbf_lst), len(dailyRisk_wocbf_lst_safe)
        )
        if min_len_wocbf < 2:
            dailySR_wocbf = np.array([0.0])
            dailySR_wocbf_max = 0.0
            dailySR_wocbf_min = 0.0
            dailySR_wocbf_avg = 0.0
        else:
            dailyAnnualReturn_wocbf_lst = dailyAnnualReturn_wocbf_lst[:min_len_wocbf]
            dailyRisk_wocbf_lst_safe = dailyRisk_wocbf_lst_safe[:min_len_wocbf]

            dailySR_wocbf = (
                (dailyAnnualReturn_wocbf_lst[1:] * 100)
                - self.config.mkt_rf[self.config.market_name]
            ) / (dailyRisk_wocbf_lst_safe[1:] * 100)
            # Handle NaN/inf
            dailySR_wocbf = np.where(
                np.isnan(dailySR_wocbf) | np.isinf(dailySR_wocbf), 0.0, dailySR_wocbf
            )
            dailySR_wocbf = np.append(0, dailySR_wocbf)
            # dailySR_wocbf = np.where(dailySR_wocbf < 0, 0, dailySR_wocbf)
            dailySR_wocbf_max = np.max(dailySR_wocbf)
            dailySR_wocbf_nonzero = dailySR_wocbf[dailySR_wocbf != 0]
            dailySR_wocbf_min = (
                np.min(dailySR_wocbf_nonzero) if len(dailySR_wocbf_nonzero) > 0 else 0.0
            )
            dailySR_wocbf_avg = np.mean(dailySR_wocbf)

        annualReturn_wocbf_pct = (
            np.power(
                (
                    1
                    + (
                        (self.return_raw_lst[-1] - self.initial_asset)
                        / self.initial_asset
                    )
                ),
                (self.config.tradeDays_per_year / len(self.return_raw_lst)),
            )
            - 1
        )
        volatility_wocbf = np.sqrt(
            (
                np.sum(
                    np.power(
                        (dailyReturnRate_wocbf - np.mean(dailyReturnRate_wocbf)), 2
                    )
                )
                * self.config.tradeDays_per_year
                / (len(self.return_raw_lst) - 1)
            )
        )
        # Avoid division by zero
        if (
            volatility_wocbf == 0
            or np.isnan(volatility_wocbf)
            or np.isinf(volatility_wocbf)
        ):
            volatility_wocbf = 1e-6
        sharpeRatio_woCBF = (
            (annualReturn_wocbf_pct * 100) - self.config.mkt_rf[self.config.market_name]
        ) / (volatility_wocbf * 100)
        # Handle NaN/inf
        if np.isnan(sharpeRatio_woCBF) or np.isinf(sharpeRatio_woCBF):
            sharpeRatio_woCBF = 0.0
        sharpeRatio_woCBF = np.max([sharpeRatio_woCBF, 0])

        winRate = len(np.argwhere(diffPeriodAsset > 0)) / (len(diffPeriodAsset) + 1)

        # MDD
        repeat_asset_lst = np.tile(asset_lst, (len(asset_lst), 1))
        mdd_mtix = np.triu(1 - repeat_asset_lst / np.reshape(asset_lst, (-1, 1)), k=1)
        mddmaxidx = np.argmax(mdd_mtix)
        mdd_highidx = mddmaxidx // len(asset_lst)
        mdd_lowidx = mddmaxidx % len(asset_lst)
        self.mdd = np.max(mdd_mtix)
        self.mdd_high = (
            asset_lst[mdd_highidx] if mdd_highidx < len(asset_lst) else asset_lst[-1]
        )
        self.mdd_low = (
            asset_lst[mdd_lowidx] if mdd_lowidx < len(asset_lst) else asset_lst[-1]
        )
        # Handle index out of range for date_memory
        self.mdd_highTimepoint = (
            self.date_memory[mdd_highidx]
            if mdd_highidx < len(self.date_memory)
            else self.date_memory[-1]
            if self.date_memory
            else "N/A"
        )
        self.mdd_lowTimepoint = (
            self.date_memory[mdd_lowidx]
            if mdd_lowidx < len(self.date_memory)
            else self.date_memory[-1]
            if self.date_memory
            else "N/A"
        )

        # Strategy volatility during trading
        # Use actual profit_lst length instead of totalTradeDay to handle multi-epoch accumulation
        actual_trade_days = len(profit_lst)
        if actual_trade_days < 2:
            # Not enough data for volatility calculation
            cumsum_r = np.array([0.0])
            stg_vol_lst = np.array([0.0])
            vol_max = 0.0
            vol_min = 0.0
            vol_avg = 0.0
        else:
            cumsum_r = np.cumsum(profit_lst) / np.arange(
                1, actual_trade_days + 1
            )  # average cumulative returns rate
            repeat_profit_lst = np.tile(profit_lst, (len(profit_lst), 1))
            stg_vol_lst = np.sqrt(
                np.sum(
                    np.power(
                        np.tril(repeat_profit_lst - np.reshape(cumsum_r, (-1, 1)), k=0),
                        2,
                    ),
                    axis=1,
                )[1:]
                / np.arange(1, len(repeat_profit_lst))
                * self.config.tradeDays_per_year
            )
            stg_vol_lst = np.append([0], stg_vol_lst, axis=0)
            # stg_vol_lst  = np.sqrt((np.cumsum(np.power((self.profit_lst - cumsum_r), 2))/np.arange(1, self.totalTradeDay+1)) * self.config.tradeDays_per_year)

            vol_max = np.max(stg_vol_lst)
            stg_vol_array = np.array(stg_vol_lst)
            stg_vol_nonzero = stg_vol_array[stg_vol_array != 0]
            vol_min = np.min(stg_vol_nonzero) if len(stg_vol_nonzero) > 0 else 0.0
            vol_avg = np.mean(stg_vol_lst)

        # short-term risk
        risk_max = np.max(self.risk_cbf_lst)
        risk_cbf_array = np.array(self.risk_cbf_lst)
        risk_cbf_nonzero = risk_cbf_array[risk_cbf_array != 0]
        risk_min = np.min(risk_cbf_nonzero) if len(risk_cbf_nonzero) > 0 else 0.0
        risk_avg = np.mean(self.risk_cbf_lst)

        risk_raw_max = np.max(self.risk_raw_lst)
        risk_raw_array = np.array(self.risk_raw_lst)
        risk_raw_nonzero = risk_raw_array[risk_raw_array != 0]
        risk_raw_min = np.min(risk_raw_nonzero) if len(risk_raw_nonzero) > 0 else 0.0
        risk_raw_avg = np.mean(self.risk_raw_lst)

        # Downside risk at volatility
        if actual_trade_days < 2:
            risk_downsideAtVol = 0.0
            risk_downsideAtVol_daily = np.array([0.0])
            risk_downsideAtVol_daily_max = 0.0
            risk_downsideAtVol_daily_min = 0.0
            risk_downsideAtVol_daily_avg = 0.0
        else:
            risk_downsideAtVol_daily = np.sqrt(
                np.sum(
                    np.power(
                        np.tril(
                            (repeat_profit_lst - np.reshape(cumsum_r, (-1, 1)))
                            * (repeat_profit_lst < np.reshape(cumsum_r, (-1, 1))),
                            k=0,
                        ),
                        2,
                    ),
                    axis=1,
                )[1:]
                / np.arange(1, len(repeat_profit_lst))
                * self.config.tradeDays_per_year
            )
            risk_downsideAtVol_daily = np.append([0], risk_downsideAtVol_daily, axis=0)
            risk_downsideAtVol = risk_downsideAtVol_daily[-1]
            risk_downsideAtVol_daily_max = np.max(risk_downsideAtVol_daily)
            risk_downsideAtVol_daily_min = np.min(risk_downsideAtVol_daily)
            risk_downsideAtVol_daily_avg = np.mean(risk_downsideAtVol_daily)

        # Downside risk at value against initial capital
        risk_downsideAtValue_daily = (asset_lst / self.initial_asset) - 1
        risk_downsideAtValue_daily_max = np.max(risk_downsideAtValue_daily)
        risk_downsideAtValue_daily_min = np.min(risk_downsideAtValue_daily)
        risk_downsideAtValue_daily_avg = np.mean(risk_downsideAtValue_daily)

        # CVaR curve
        cvar_max = np.max(self.cvar_lst)
        cvar_array = np.array(self.cvar_lst)
        cvar_nonzero = cvar_array[cvar_array != 0]
        cvar_min = np.min(cvar_nonzero) if len(cvar_nonzero) > 0 else 0.0
        cvar_avg = np.mean(self.cvar_lst)

        cvar_raw_max = np.max(self.cvar_raw_lst)
        cvar_raw_array = np.array(self.cvar_raw_lst)
        cvar_raw_nonzero = cvar_raw_array[cvar_raw_array != 0]
        cvar_raw_min = np.min(cvar_raw_nonzero) if len(cvar_raw_nonzero) > 0 else 0.0
        cvar_raw_avg = np.mean(self.cvar_raw_lst)

        # Calmar ratio
        time_T = len(profit_lst)
        avg_return = netProfit_pct / time_T if time_T > 0 else 0.0
        variance_r = (
            np.sum(np.power((profit_lst - avg_dailyReturn_pct), 2))
            / (len(profit_lst) - 1)
            if len(profit_lst) > 1
            else 0.0
        )
        volatility_daily = np.sqrt(variance_r) if variance_r >= 0 else 0.0
        # Avoid division by zero
        if volatility_daily == 0:
            volatility_daily = 1e-6

        if netProfit_pct > 0:
            shrp = avg_return / volatility_daily if volatility_daily > 0 else 0.0
            if shrp > 0 and not (np.isnan(shrp) or np.isinf(shrp)):
                log_shrp = np.log(shrp)
                if not (np.isnan(log_shrp) or np.isinf(log_shrp)):
                    calmarRatio = (time_T * np.power(shrp, 2)) / (
                        0.63519 + 0.5 * np.log(time_T) + log_shrp
                    )
                else:
                    calmarRatio = 0.0
            else:
                calmarRatio = 0.0
        elif netProfit_pct == 0:
            calmarRatio = (
                (netProfit_pct) / (1.2533 * volatility_daily * np.sqrt(time_T))
                if volatility_daily > 0 and time_T > 0
                else 0.0
            )
        else:
            # netProfit_pct < 0
            if avg_return != 0 and not (np.isnan(avg_return) or np.isinf(avg_return)):
                calmarRatio = (netProfit_pct) / (
                    -(avg_return * time_T) - (variance_r / avg_return)
                )
            else:
                calmarRatio = 0.0

        # Final validation
        if np.isnan(calmarRatio) or np.isinf(calmarRatio):
            calmarRatio = 0.0

        # Sterling ratio
        move_mdd_mask = np.where(profit_lst < 0, 1, 0)
        moving_mdd = (
            np.sqrt(
                np.sum(np.power(profit_lst * move_mdd_mask, 2))
                * self.config.tradeDays_per_year
                / (len(profit_lst) - 1)
            )
            if len(profit_lst) > 1
            else 0.0
        )
        # Avoid division by zero
        if moving_mdd == 0 or np.isnan(moving_mdd) or np.isinf(moving_mdd):
            moving_mdd = 1e-6
        sterlingRatio = (
            (annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name]
        ) / (moving_mdd * 100)
        # Handle NaN/inf
        if np.isnan(sterlingRatio) or np.isinf(sterlingRatio):
            sterlingRatio = 0.0

        if self.mode == "train":
            cputime_use = self.end_cputime - self.start_cputime - self.exclusive_cputime
            systime_use = self.end_systime - self.start_systime - self.exclusive_systime
        else:
            cputime_use = self.end_cputime - self.start_cputime
            systime_use = self.end_systime - self.start_systime

        # Expand Top-K memories to full length for reporting without mutating raw history
        actions_memory_arr = np.array(self.actions_memory)
        rl_memory_full = self._expand_action_series(
            getattr(self, "action_rl_memory", [])
        )
        cbf_memory_full = self._expand_action_series(
            getattr(self, "action_cbf_memeory", [])
        )
        if actions_memory_arr.shape != (self.totalTradeDay, self.stock_num):
            actions_memory_arr = (
                np.ones((self.totalTradeDay, self.stock_num))
                * (1 / self.stock_num)
                * self.bound_flag
            )
        if rl_memory_full.shape != (self.totalTradeDay + 1, self.stock_num):
            rl_memory_full = (
                np.ones((self.totalTradeDay + 1, self.stock_num))
                * (1 / self.stock_num)
                * self.bound_flag
            )
        if cbf_memory_full.shape != (self.totalTradeDay + 1, self.stock_num):
            cbf_memory_full = np.zeros((self.totalTradeDay + 1, self.stock_num))
        if len(self.solvable_flag) == 0:
            self.solvable_flag = np.zeros(len(asset_lst))
        if len(self.risk_pred_lst) == 0:
            self.risk_pred_lst = np.zeros(len(asset_lst))

        cbf_abssum_contribution = np.sum(np.abs(cbf_memory_full[:-1]))

        target_len = len(asset_lst)

        def _align_array(arr, fill_value=0.0, target=target_len):
            if isinstance(arr, list):
                arr = np.array(arr)
            if arr is None:
                return np.full(target, fill_value, dtype=float)
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return np.full(target, float(arr), dtype=float)
            if len(arr) > target:
                return arr[:target]
            if len(arr) < target:
                pad_shape = (target - len(arr),) + arr.shape[1:]
                pad = np.full(pad_shape, fill_value, dtype=arr.dtype)
                return np.concatenate([arr, pad], axis=0)
            return arr

        final_action_abs = _align_array(np.sum(np.abs(actions_memory_arr), axis=1))
        rl_action_abs = _align_array(np.sum(np.abs(rl_memory_full), axis=1))
        cbf_action_abs = _align_array(np.sum(np.abs(cbf_memory_full), axis=1))

        # Compute aggregate reward statistics using absolute sum (not mean)
        if len(self.reward_lst) > 0:
            reward_arr = np.array(self.reward_lst, dtype=float)
            if np.all(np.isnan(reward_arr)):
                reward_total = 0.0
            else:
                reward_total = float(np.nansum(reward_arr))
        else:
            reward_total = 0.0

        info_dict = {
            "ep": self.epoch,
            "trading_days": self.totalTradeDay,
            "annualReturn_pct": annualReturn_pct,
            "volatility": volatility,
            "sharpeRatio": sharpeRatio,
            "sharpeRatio_wocbf": sharpeRatio_woCBF,
            "mdd": self.mdd * 100,  # Convert to percentage for consistency
            "calmarRatio": calmarRatio,
            "sterlingRatio": sterlingRatio,
            "netProfit": netProfit,
            "netProfit_pct": netProfit_pct,
            "winRate": winRate,
            "vol_max": vol_max,
            "vol_min": vol_min,
            "vol_avg": vol_avg,
            "risk_max": risk_max,
            "risk_min": risk_min,
            "risk_avg": risk_avg,
            "riskRaw_max": risk_raw_max,
            "riskRaw_min": risk_raw_min,
            "riskRaw_avg": risk_raw_avg,
            "dailySR_max": dailySR_max,
            "dailySR_min": dailySR_min,
            "dailySR_avg": dailySR_avg,
            "dailySR_wocbf_max": dailySR_wocbf_max,
            "dailySR_wocbf_min": dailySR_wocbf_min,
            "dailySR_wocbf_avg": dailySR_wocbf_avg,
            "dailyReturn_pct_max": dailyReturn_pct_max,
            "dailyReturn_pct_min": dailyReturn_pct_min,
            "dailyReturn_pct_avg": avg_dailyReturn_pct,
            "sigReturn_max": sigReturn_max,
            "sigReturn_min": sigReturn_min,
            "mdd_high": self.mdd_high,
            "mdd_low": self.mdd_low,
            "mdd_high_date": self.mdd_highTimepoint,
            "mdd_low_date": self.mdd_lowTimepoint,
            # Store reward as absolute sum for the episode
            "final_capital": self.cur_capital,
            "reward_sum": reward_total,
            "final_capital_wocbf": self.return_raw_lst[-1],
            "cbf_contribution": cbf_abssum_contribution,
            "risk_downsideAtVol": risk_downsideAtVol,
            "risk_downsideAtVol_daily_max": risk_downsideAtVol_daily_max,
            "risk_downsideAtVol_daily_min": risk_downsideAtVol_daily_min,
            "risk_downsideAtVol_daily_avg": risk_downsideAtVol_daily_avg,
            "risk_downsideAtValue_daily_max": risk_downsideAtValue_daily_max,
            "risk_downsideAtValue_daily_min": risk_downsideAtValue_daily_min,
            "risk_downsideAtValue_daily_avg": risk_downsideAtValue_daily_avg,
            "cvar_max": cvar_max,
            "cvar_min": cvar_min,
            "cvar_avg": cvar_avg,
            "cvar_raw_max": cvar_raw_max,
            "cvar_raw_min": cvar_raw_min,
            "cvar_raw_avg": cvar_raw_avg,
            "solver_solvable": self.solver_stat["solvable"],
            "solver_insolvable": self.solver_stat["insolvable"],
            "cputime": cputime_use,
            "systime": systime_use,
            "asset_lst": asset_lst.copy(),
            "daily_return_lst": profit_lst.copy(),
            "reward_lst": np.array(self.reward_lst).copy()
            if isinstance(self.reward_lst, list)
            else self.reward_lst.copy(),
            "stg_vol_lst": stg_vol_lst.copy()
            if isinstance(stg_vol_lst, np.ndarray)
            else np.array(stg_vol_lst).copy(),
            "risk_lst": np.array(self.risk_cbf_lst).copy()
            if isinstance(self.risk_cbf_lst, list)
            else self.risk_cbf_lst.copy(),
            "risk_wocbf_lst": np.array(self.risk_raw_lst).copy()
            if isinstance(self.risk_raw_lst, list)
            else self.risk_raw_lst.copy(),
            "capital_wocbf_lst": np.array(self.return_raw_lst).copy()
            if isinstance(self.return_raw_lst, list)
            else self.return_raw_lst.copy(),
            "daily_sr_lst": dailySR.copy()
            if isinstance(dailySR, np.ndarray)
            else np.array(dailySR).copy(),
            "daily_sr_wocbf_lst": dailySR_wocbf.copy()
            if isinstance(dailySR_wocbf, np.ndarray)
            else np.array(dailySR_wocbf).copy(),
            "risk_adj_lst": np.array(self.risk_adj_lst).copy()
            if isinstance(self.risk_adj_lst, list)
            else self.risk_adj_lst.copy(),
            "ctrl_weight_lst": np.array(self.ctrl_weight_lst).copy()
            if isinstance(self.ctrl_weight_lst, list)
            else self.ctrl_weight_lst.copy(),
            "solvable_flag": self.solvable_flag.copy()
            if isinstance(self.solvable_flag, np.ndarray)
            else np.array(self.solvable_flag).copy(),
            "risk_pred_lst": np.array(self.risk_pred_lst).copy()
            if isinstance(self.risk_pred_lst, list)
            else self.risk_pred_lst.copy(),
            "final_action_abssum_lst": final_action_abs.copy(),
            "rl_action_abssum_lst": rl_action_abs.copy(),
            "cbf_action_abssum_lst": cbf_action_abs.copy(),
            "daily_downsideAtVol_risk_lst": risk_downsideAtVol_daily.copy()
            if isinstance(risk_downsideAtVol_daily, np.ndarray)
            else np.array(risk_downsideAtVol_daily).copy(),
            "daily_downsideAtValue_risk_lst": risk_downsideAtValue_daily.copy()
            if isinstance(risk_downsideAtValue_daily, np.ndarray)
            else np.array(risk_downsideAtValue_daily).copy(),
            "cvar_lst": np.array(self.cvar_lst).copy()
            if isinstance(self.cvar_lst, list)
            else self.cvar_lst.copy(),
            "cvar_raw_lst": np.array(self.cvar_raw_lst).copy()
            if isinstance(self.cvar_raw_lst, list)
            else self.cvar_raw_lst.copy(),
        }

        return info_dict

    def _load_existing_profile_history(self):
        """
        Resume-friendly: load existing <mode>_profile.csv so new epochs append instead of overwriting.

        IMPORTANT: We load ALL existing epochs and rely on _profile_saved_epochs to prevent
        duplicate saves. This preserves historical data when:
        - Resuming from best_valid checkpoint to continue training
        - Walk-forward window chaining
        - Manual restarts

        Duplicates are prevented by checking (mode, epoch) in save_profile() before appending.
        """
        profile_path = os.path.join(self.config.res_dir, f"{self.mode}_profile.csv")
        if not os.path.exists(profile_path):
            return
        try:
            df = pd.read_csv(profile_path)
        except Exception as e:
            smart_print(
                f"[Profile Restore] Warning: failed to load existing profile at {profile_path}: {e}",
                flush=True,
            )
            return
        if df.empty:
            return

        # Load ALL existing epochs - do NOT filter based on resume_epoch
        # The save_profile() method uses _profile_saved_epochs to prevent duplicates
        # This ensures we preserve historical training data when resuming

        # Ensure all expected columns exist for backward compatibility
        for col in self.profile_hist_field_lst:
            if col not in df.columns:
                df[col] = None
        df = df[self.profile_hist_field_lst]
        for col in self.profile_hist_field_lst:
            self.profile_hist_ep[col] = df[col].tolist()
        if "ep" in df.columns:
            for raw_ep in df["ep"]:
                try:
                    ep_val = int(raw_ep)
                    self._profile_saved_epochs.add((self.mode, ep_val))
                except Exception:
                    continue
        smart_print(
            f"[Profile Restore] Loaded {len(df)} existing {self.mode} epochs from {profile_path}",
            flush=True,
        )

    def save_profile(self, invest_profile):
        # Prevent duplicate save when save_profile is called multiple times in same epoch/mode
        if not hasattr(self, "_profile_saved_epochs"):
            self._profile_saved_epochs = set()
        ep_key = (self.mode, invest_profile.get("ep", getattr(self, "epoch", None)))
        if ep_key in self._profile_saved_epochs:
            smart_print(f"[Profile Save] Skip duplicate save for {ep_key}", flush=True)
            return
        self._profile_saved_epochs.add(ep_key)

        # basic data
        missing_fields = []
        for fname in self.profile_hist_field_lst:
            if fname in list(invest_profile.keys()):
                self.profile_hist_ep[fname].append(invest_profile[fname])
            else:
                missing_fields.append(fname)
                # Use None as placeholder for missing fields instead of raising error
                self.profile_hist_ep[fname].append(None)
                smart_print(
                    f"Warning: Field '{fname}' not found in invest_profile, using None as placeholder",
                    flush=True,
                )

        if missing_fields:
            smart_print(
                f"Warning: Missing {len(missing_fields)} fields in invest_profile: {missing_fields}",
                flush=True,
            )
            smart_print(
                f"Available fields in invest_profile: {list(invest_profile.keys())}",
                flush=True,
            )

        try:
            phist_df = pd.DataFrame(
                self.profile_hist_ep, columns=self.profile_hist_field_lst
            )
            # Ensure numeric fields are proper dtype before computing best metrics
            numeric_fields = [
                "reward_sum",
                "final_capital",
                "sharpeRatio",
                "volatility",
                "mdd",
            ]
            for nf in numeric_fields:
                if nf in phist_df.columns:
                    phist_df[nf] = pd.to_numeric(phist_df[nf], errors="coerce")
            profile_path = os.path.join(
                self.config.res_dir, "{}_profile.csv".format(self.mode)
            )
            phist_df.to_csv(profile_path, index=False)
            smart_print(
                f"Profile saved to: {profile_path} (epoch {self.epoch}, {len(phist_df)} rows)",
                flush=True,
            )
        except Exception as e:
            smart_print(f"Error saving profile: {e}", flush=True)
            smart_print(
                f"profile_hist_ep keys: {list(self.profile_hist_ep.keys())}", flush=True
            )
            smart_print(
                f"profile_hist_field_lst: {self.profile_hist_field_lst}", flush=True
            )
            raise

        cputime_avg = np.mean(phist_df["cputime"])
        systime_avg = np.mean(phist_df["systime"])

        # Normalize best-model keys once so we reuse consistent names everywhere.
        # Train best is always by reward_sum (legacy behavior).
        # Valid/test: keep reward_sum for test; allow config override for valid.
        if self.mode == "train":
            # Train: always pick best by reward_sum
            metric_for_best = "reward_sum"
        elif self.mode in ["valid", "test"]:
            # Valid/Test: pick best by Sharpe ratio
            metric_for_best = "sharpeRatio"
        else:
            metric_for_best = (
                getattr(self.config, "early_stop_metric", None)
                or getattr(self.config, "trained_best_model_type", None)
                or "reward_sum"
            )
        best_key_ep = (
            "best_reward_sum_ep"
            if metric_for_best == "js_loss" or metric_for_best == "reward_sum"
            else f"{metric_for_best}_ep"
        )
        best_key_val = (
            "best_reward_sum"
            if metric_for_best == "js_loss" or metric_for_best == "reward_sum"
            else metric_for_best
        )

        bestmodel_dict = {}
        if metric_for_best == "max_capital":
            field_name = "final_capital"
            # Filter out NaN values before finding max
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.max(valid_values)
            else:
                v = np.nan
        elif metric_for_best in ["js_loss", "reward_sum"] or (
            "loss" in metric_for_best
        ):
            field_name = "reward_sum"
            # Filter out NaN values before finding max
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.max(valid_values)
            else:
                v = np.nan
        elif metric_for_best == "sharpeRatio":
            field_name = "sharpeRatio"
            # Filter out NaN values before finding max
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.max(valid_values)
            else:
                v = np.nan
        elif metric_for_best == "volatility":
            field_name = "volatility"
            # Filter out NaN values before finding min
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.min(valid_values)
            else:
                v = np.nan
        elif metric_for_best == "mdd":
            field_name = "mdd"
            # Filter out NaN values before finding min
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.min(valid_values)
            else:
                v = np.nan
        else:
            raise ValueError(
                "Unknown implementation with the best model type [{}]..".format(
                    self.config.trained_best_model_type
                )
            )

        # Handle NaN case: use current epoch value as fallback
        if np.isnan(v) or v is None:
            # Use current epoch's value as fallback
            current_value = invest_profile.get(field_name, None)
            if current_value is not None and not (
                isinstance(current_value, float) and np.isnan(current_value)
            ):
                v = current_value
                v_ep = self.epoch
                smart_print(
                    f"Warning: All values in {field_name} are NaN, using current epoch {v_ep} value {v} as fallback",
                    flush=True,
                )
            else:
                # Last resort: use current epoch number, value remains NaN (will be handled later)
                v_ep = self.epoch
                smart_print(
                    f"Warning: All values in {field_name} are NaN and current epoch value is also NaN, using epoch {v_ep} as fallback",
                    flush=True,
                )
        else:
            # Find epoch with the best value, handle floating point comparison and empty results
            filtered_df = phist_df[phist_df[field_name] == v]
            if len(filtered_df) == 0:
                # Fallback: use approximate comparison for floating point values
                if field_name in [
                    "final_capital",
                    "reward_sum",
                    "sharpeRatio",
                    "volatility",
                    "mdd",
                ]:
                    # Use np.isclose for floating point comparison
                    tolerance = 1e-6
                    filtered_df = phist_df[np.abs(phist_df[field_name] - v) < tolerance]

            if len(filtered_df) == 0:
                # If still empty, use current epoch as fallback
                v_ep = self.epoch
                smart_print(
                    f"Warning: Could not find epoch with {field_name}={v}, using current epoch {v_ep} as fallback",
                    flush=True,
                )
            else:
                v_ep = filtered_df["ep"].iloc[0]

        # Ensure v is not NaN before saving (use current epoch value if still NaN)
        if np.isnan(v) or v is None:
            # Try to get value from current epoch
            current_value = invest_profile.get(field_name, None)
            if current_value is not None and not (
                isinstance(current_value, float) and np.isnan(current_value)
            ):
                v = current_value
            else:
                # Use 0 as absolute fallback
                v = 0.0
                smart_print(
                    f"Warning: Using 0.0 as absolute fallback for {field_name}",
                    flush=True,
                )

        # Normalize key names: js_loss -> reward_sum naming
        bestmodel_dict[best_key_ep] = v_ep
        bestmodel_dict[best_key_val] = v

        # Ensure all values in bestmodel_dict are valid (not NaN) before saving
        for key, value in bestmodel_dict.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                # Replace NaN with appropriate default
                if "ep" in key:
                    bestmodel_dict[key] = self.epoch  # Use current epoch
                else:
                    # For value fields, try to get from current epoch
                    field_name = key
                    current_value = invest_profile.get(field_name, 0.0)
                    if current_value is not None and not (
                        isinstance(current_value, float) and np.isnan(current_value)
                    ):
                        bestmodel_dict[key] = current_value
                    else:
                        bestmodel_dict[key] = 0.0  # Absolute fallback
                smart_print(
                    f"Warning: Replaced NaN in bestmodel_dict['{key}'] with {bestmodel_dict[key]}",
                    flush=True,
                )

        if True:
            smart_print("-" * 30)
            # Use final_capital from invest_profile instead of self.cur_capital to avoid nan
            current_capital = invest_profile.get("final_capital", self.cur_capital)
            # Handle nan values
            if current_capital is None or (
                isinstance(current_capital, float) and np.isnan(current_capital)
            ):
                current_capital = self.initial_asset  # Fallback to initial asset
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = current_capital  # Fallback to current capital

            # Map trained_best_model_type to display name for log message
            field_display_name = {
                "max_capital": "capital",
                "js_loss": "reward_sum",
                "sharpeRatio": "sharpeRatio",
                "volatility": "volatility",
                "mdd": "mdd",
            }.get(metric_for_best, metric_for_best)

            # Print NaN statistics summary
            total_nan_events = (
                self.nan_stats["cur_close_nan"]
                + self.nan_stats["last_close_nan"]
                + self.nan_stats["poDayReturn_nan"]
                + self.nan_stats["updatePoValue_nan"]
            )
            if total_nan_events > 0:
                smart_print(
                    f"\n[NaN Statistics] Epoch {self.epoch} - Total NaN events: {total_nan_events}",
                    flush=True,
                )
                smart_print(
                    f"  - cur_close_price NaN/inf/<=0: {self.nan_stats['cur_close_nan']} times",
                    flush=True,
                )
                smart_print(
                    f"  - last_close_price NaN/inf/<=0: {self.nan_stats['last_close_nan']} times",
                    flush=True,
                )
                smart_print(
                    f"  - poDayReturn NaN/inf: {self.nan_stats['poDayReturn_nan']} times",
                    flush=True,
                )
                smart_print(
                    f"  - updatePoValue NaN/inf/<=0: {self.nan_stats['updatePoValue_nan']} times",
                    flush=True,
                )

                # Print sample details (first few occurrences)
                if len(self.nan_stats["cur_close_details"]) > 0:
                    smart_print(
                        f"  Sample cur_close NaN events (showing first {min(3, len(self.nan_stats['cur_close_details']))}):",
                        flush=True,
                    )
                    for detail in self.nan_stats["cur_close_details"][:3]:
                        smart_print(
                            f"    Day {detail['day']} ({detail['date']}): stock={detail['stock']}, value={detail['value']}, fallback={detail['fallback']}",
                            flush=True,
                        )
                if len(self.nan_stats["poDayReturn_details"]) > 0:
                    smart_print(
                        f"  Sample poDayReturn NaN events (showing first {min(3, len(self.nan_stats['poDayReturn_details']))}):",
                        flush=True,
                    )
                    for detail in self.nan_stats["poDayReturn_details"][:3]:
                        smart_print(
                            f"    Day {detail['day']} ({detail['date']}): value={detail['value']}",
                            flush=True,
                        )
                if len(self.nan_stats["updatePoValue_details"]) > 0:
                    smart_print(
                        f"  Sample updatePoValue NaN events (showing first {min(3, len(self.nan_stats['updatePoValue_details']))}):",
                        flush=True,
                    )
                    for detail in self.nan_stats["updatePoValue_details"][:3]:
                        smart_print(
                            f"    Day {detail['day']} ({detail['date']}): value={detail['value']}, prev_capital={detail['prev_capital']:.2f}, poDayReturn={detail['poDayReturn']}",
                            flush=True,
                        )
                smart_print("", flush=True)

            log_str = "Mode: {}, Ep: {}, Current epoch capital: {:.2f}, historical best {} ({} ep): {:.2f} | solvable: {}, insolvable: {} | step count: {} | cputime cur: {} s, avg: {} s, system time cur: {} s/ep, avg: {} s/ep..".format(
                self.mode,
                self.epoch,
                current_capital,
                field_display_name,
                v_ep,
                v,
                np.array(phist_df["solver_solvable"])[-1],
                np.array(phist_df["solver_insolvable"])[-1],
                self.stepcount,
                np.round(np.array(phist_df["cputime"])[-1], 2),
                np.round(cputime_avg, 2),
                np.round(np.array(phist_df["systime"])[-1], 2),
                np.round(systime_avg, 2),
            )
            smart_print(log_str)
            # Print top 10 stocks with weights for current epoch
            if (
                hasattr(self, "actions_memory")
                and len(self.actions_memory) > 0
                and hasattr(self, "stock_lst")
            ):
                sample_idx = min(4, len(self.actions_memory) - 1)
                if isinstance(self.actions_memory, list):
                    sample_weights = self.actions_memory[sample_idx]
                else:
                    sample_weights = self.actions_memory[sample_idx]
                if isinstance(sample_weights, np.ndarray) and len(
                    sample_weights
                ) == len(self.stock_lst):
                    top_k_log = getattr(self.config, "topK", 10)
                    # Use weight_symbol_mapper if available
                    if get_top_stocks is not None:
                        try:
                            top_stocks = get_top_stocks(
                                sample_weights, self.stock_lst, top_k=top_k_log
                            )
                            smart_print(f"  Top {top_k_log} stocks (day {sample_idx}):")
                            for i, (symbol, weight) in enumerate(top_stocks, 1):
                                smart_print(
                                    "    {}. {}: {:.4f} ({:.2f}%)".format(
                                        i, symbol, weight, weight * 100
                                    )
                                )
                        except Exception as e:
                            # Fallback to simple printing if mapping fails
                            smart_print(
                                f"  Top {top_k_log} stocks (day {sample_idx}): (mapping error: {e})"
                            )
                    else:
                        # Fallback: print top 10 by weight manually
                        weight_stock_pairs = list(zip(sample_weights, self.stock_lst))
                        weight_stock_pairs.sort(key=lambda x: x[0], reverse=True)
                        smart_print(f"  Top {top_k_log} stocks (day {sample_idx}):")
                        for i, (weight, symbol) in enumerate(
                            weight_stock_pairs[:top_k_log], 1
                        ):
                            smart_print(
                                "    {}. {}: {:.4f} ({:.2f}%)".format(
                                    i, symbol, weight, weight * 100
                                )
                            )
        # Create DataFrame and ensure no NaN values (we already handled NaN above, but double-check)
        bestmodel_df = pd.DataFrame([bestmodel_dict])
        # Final safety check: replace any remaining NaN
        for col in bestmodel_df.columns:
            if bestmodel_df[col].isna().any():
                if "ep" in col:
                    bestmodel_df[col] = bestmodel_df[col].fillna(self.epoch)
                else:
                    # Try to get from current invest_profile
                    field_name = col
                    current_value = invest_profile.get(field_name, 0.0)
                    if current_value is not None and not (
                        isinstance(current_value, float) and np.isnan(current_value)
                    ):
                        bestmodel_df[col] = bestmodel_df[col].fillna(current_value)
                    else:
                        bestmodel_df[col] = bestmodel_df[col].fillna(0.0)

        bestmodel_path = os.path.join(
            self.config.res_dir, "{}_bestmodel.csv".format(self.mode)
        )
        bestmodel_df.to_csv(bestmodel_path, index=False)

        # save data of each step in 1st/best/last model
        fpath = os.path.join(self.config.res_dir, "{}_stepdata.csv".format(self.mode))
        target_len = (
            len(invest_profile["asset_lst"])
            if "asset_lst" in invest_profile
            else len(self.asset_lst)
        )

        def _align_profile_array(arr, fill_value=0.0):
            if arr is None:
                return np.full(target_len, fill_value)
            if isinstance(arr, list):
                arr = np.asarray(arr)
            elif not isinstance(arr, np.ndarray):
                arr = np.full(target_len, arr)
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return np.full(target_len, float(arr))
            if len(arr) > target_len:
                return arr[:target_len]
            if len(arr) < target_len:
                pad = np.full(target_len - len(arr), fill_value)
                return np.concatenate([arr, pad], axis=0)
            return arr

        if not os.path.exists(fpath):
            # Check if we're resuming from a checkpoint (arrays may have inconsistent lengths)
            is_resuming = hasattr(self, "_is_resuming") and self._is_resuming
            if is_resuming:
                smart_print(
                    f"[SAVE_PROFILE] Skipping stepdata.csv creation during resume (arrays may have inconsistent lengths)"
                )
                # Create empty DataFrame with correct columns for future appends
                step_data = pd.DataFrame(
                    columns=[
                        "date",
                        "capital_policy_1",
                        "dailyReturn_policy_1",
                        "reward_policy_1",
                        "strategyVolatility_policy_1",
                        "risk_policy_1",
                        "risk_wocbf_policy_1",
                        "capital_wocbf_policy_1",
                        "dailySR_policy_1",
                        "dailySR_wocbf_policy_1",
                        "riskAccepted_policy_1",
                        "ctrlWeight_policy_1",
                        "solvable_flag_policy_1",
                        "risk_pred_policy_1",
                        "final_action_abssum_policy_1",
                        "rl_action_abssum_policy_1",
                        "cbf_action_abssum_policy_1",
                        "downsideAtVol_risk_policy_1",
                        "downsideAtValue_risk_policy_1",
                        "cvar_policy_1",
                        "cvar_raw_policy_1",
                    ]
                )
                step_data.to_csv(fpath, index=False)
            else:
                # Ensure all arrays have the same length for fresh training
                array_length = len(invest_profile["asset_lst"])
                smart_print(
                    f"[SAVE_PROFILE] Creating new stepdata.csv with {array_length} rows"
                )

                step_data = {
                    "date": self.date_memory[:array_length]
                    if len(self.date_memory) >= array_length
                    else self.date_memory
                    + [""] * (array_length - len(self.date_memory)),
                    "capital_policy_1": _align_profile_array(
                        invest_profile["asset_lst"]
                    ).tolist(),
                    "dailyReturn_policy_1": _align_profile_array(
                        invest_profile["daily_return_lst"]
                    ).tolist(),
                    "reward_policy_1": _align_profile_array(
                        invest_profile["reward_lst"]
                    ).tolist(),
                    "strategyVolatility_policy_1": _align_profile_array(
                        invest_profile["stg_vol_lst"]
                    ).tolist(),
                    "risk_policy_1": _align_profile_array(
                        invest_profile["risk_lst"]
                    ).tolist(),
                    "risk_wocbf_policy_1": _align_profile_array(
                        invest_profile["risk_wocbf_lst"]
                    ).tolist(),
                    "capital_wocbf_policy_1": _align_profile_array(
                        invest_profile["capital_wocbf_lst"]
                    ).tolist(),
                    "dailySR_policy_1": _align_profile_array(
                        invest_profile["daily_sr_lst"]
                    ).tolist(),
                    "dailySR_wocbf_policy_1": _align_profile_array(
                        invest_profile["daily_sr_wocbf_lst"]
                    ).tolist(),
                    "riskAccepted_policy_1": _align_profile_array(
                        invest_profile["risk_adj_lst"]
                    ).tolist(),
                    "ctrlWeight_policy_1": _align_profile_array(
                        invest_profile["ctrl_weight_lst"]
                    ).tolist(),
                    "solvable_flag_policy_1": _align_profile_array(
                        invest_profile["solvable_flag"]
                    ).tolist(),
                    "risk_pred_policy_1": _align_profile_array(
                        invest_profile["risk_pred_lst"]
                    ).tolist(),
                    "final_action_abssum_policy_1": _align_profile_array(
                        invest_profile["final_action_abssum_lst"]
                    ).tolist(),
                    "rl_action_abssum_policy_1": _align_profile_array(
                        invest_profile["rl_action_abssum_lst"]
                    ).tolist(),
                    "cbf_action_abssum_policy_1": _align_profile_array(
                        invest_profile["cbf_action_abssum_lst"]
                    ).tolist(),
                    "downsideAtVol_risk_policy_1": _align_profile_array(
                        invest_profile["daily_downsideAtVol_risk_lst"]
                    ).tolist(),
                    "downsideAtValue_risk_policy_1": _align_profile_array(
                        invest_profile["daily_downsideAtValue_risk_lst"]
                    ).tolist(),
                    "cvar_policy_1": _align_profile_array(
                        invest_profile["cvar_lst"]
                    ).tolist(),
                    "cvar_raw_policy_1": _align_profile_array(
                        invest_profile["cvar_raw_lst"]
                    ).tolist(),
                }

                # Verify all arrays have same length
                lengths = [len(v) for v in step_data.values()]
                if len(set(lengths)) != 1:
                    smart_print(
                        f"[SAVE_PROFILE] ERROR: Arrays have different lengths: {dict(zip(step_data.keys(), lengths))}"
                    )
                    raise ValueError(
                        "Cannot create stepdata.csv: arrays have inconsistent lengths"
                    )

                step_data = pd.DataFrame(step_data)
        else:
            step_data = pd.DataFrame(pd.read_csv(fpath, header=0))
            # If existing step_data has different length, align target_len to it to avoid assignment mismatch
            if len(step_data) != target_len:
                smart_print(
                    f"[SAVE_PROFILE] Existing stepdata length ({len(step_data)}) != target_len ({target_len}), aligning to existing length",
                    flush=True,
                )
                target_len = len(step_data)

        # Only update best model columns if this is the best model and arrays have consistent length
        best_epoch_in_record = bestmodel_dict.get(
            best_key_ep, bestmodel_dict.get(f"{self.config.trained_best_model_type}_ep")
        )
        if best_epoch_in_record == invest_profile["ep"]:
            # Check if step_data is empty (resume case) or has matching length
            if len(step_data) == 0:
                smart_print(
                    f"[SAVE_PROFILE] Skipping best model update for resume case (empty step_data)"
                )
            elif len(step_data) == len(invest_profile["asset_lst"]):
                step_data["capital_policy_best"] = _align_profile_array(
                    invest_profile["asset_lst"]
                ).tolist()
                step_data["dailyReturn_policy_best"] = _align_profile_array(
                    invest_profile["daily_return_lst"]
                ).tolist()
                step_data["reward_policy_best"] = _align_profile_array(
                    invest_profile["reward_lst"]
                ).tolist()
                step_data["strategyVolatility_policy_best"] = _align_profile_array(
                    invest_profile["stg_vol_lst"]
                ).tolist()
                step_data["risk_policy_best"] = _align_profile_array(
                    invest_profile["risk_lst"]
                ).tolist()
                step_data["risk_wocbf_policy_best"] = _align_profile_array(
                    invest_profile["risk_wocbf_lst"]
                ).tolist()
                step_data["capital_wocbf_policy_best"] = _align_profile_array(
                    invest_profile["capital_wocbf_lst"]
                ).tolist()
                step_data["dailySR_policy_best"] = _align_profile_array(
                    invest_profile["daily_sr_lst"]
                ).tolist()
                step_data["dailySR_wocbf_policy_best"] = _align_profile_array(
                    invest_profile["daily_sr_wocbf_lst"]
                ).tolist()
                step_data["riskAccepted_policy_best"] = _align_profile_array(
                    invest_profile["risk_adj_lst"]
                ).tolist()
                step_data["ctrlWeight_policy_best"] = _align_profile_array(
                    invest_profile["ctrl_weight_lst"]
                ).tolist()
                step_data["solvable_flag_policy_best"] = _align_profile_array(
                    invest_profile["solvable_flag"]
                ).tolist()
                step_data["risk_pred_policy_best"] = _align_profile_array(
                    invest_profile["risk_pred_lst"]
                ).tolist()
                step_data["final_action_abssum_policy_best"] = _align_profile_array(
                    invest_profile["final_action_abssum_lst"]
                ).tolist()
                step_data["rl_action_abssum_policy_best"] = _align_profile_array(
                    invest_profile["rl_action_abssum_lst"]
                ).tolist()
                step_data["cbf_action_abssum_policy_best"] = _align_profile_array(
                    invest_profile["cbf_action_abssum_lst"]
                ).tolist()
                step_data["downsideAtVol_risk_policy_best"] = _align_profile_array(
                    invest_profile["daily_downsideAtVol_risk_lst"]
                ).tolist()
                step_data["downsideAtValue_risk_policy_best"] = _align_profile_array(
                    invest_profile["daily_downsideAtValue_risk_lst"]
                ).tolist()
                step_data["cvar_policy_best"] = _align_profile_array(
                    invest_profile["cvar_lst"]
                ).tolist()
                step_data["cvar_raw_policy_best"] = _align_profile_array(
                    invest_profile["cvar_raw_lst"]
                ).tolist()
            else:
                smart_print(
                    f"[SAVE_PROFILE] Skipping best model update: step_data length ({len(step_data)}) != invest_profile arrays length ({len(invest_profile['asset_lst'])})"
                )
        # Record the test set performance on valid_best_policy
        if self.mode == "test":
            valid_fpath = os.path.join(self.config.res_dir, "valid_bestmodel.csv")
            if os.path.exists(valid_fpath):
                valid_records = pd.DataFrame(pd.read_csv(valid_fpath, header=0))
                valid_ep_col = (
                    best_key_ep
                    if best_key_ep in valid_records.columns
                    else f"{self.config.trained_best_model_type}_ep"
                )
                if valid_ep_col in valid_records.columns:
                    best_valid_ep = int(valid_records[valid_ep_col][0])
                    if best_valid_ep == invest_profile["ep"]:
                        # Align arrays to target_len to prevent length mismatches with existing step_data
                        step_data["capital_policy_validbest"] = _align_profile_array(
                            invest_profile["asset_lst"]
                        ).tolist()
                        step_data["dailyReturn_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["daily_return_lst"]
                            ).tolist()
                        )
                        step_data["reward_policy_validbest"] = _align_profile_array(
                            invest_profile["reward_lst"]
                        ).tolist()
                        step_data["strategyVolatility_policy_validbest"] = (
                            _align_profile_array(invest_profile["stg_vol_lst"]).tolist()
                        )
                        step_data["risk_policy_validbest"] = _align_profile_array(
                            invest_profile["risk_lst"]
                        ).tolist()
                        step_data["risk_wocbf_policy_validbest"] = _align_profile_array(
                            invest_profile["risk_wocbf_lst"]
                        ).tolist()
                        step_data["capital_wocbf_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["capital_wocbf_lst"]
                            ).tolist()
                        )
                        step_data["dailySR_policy_validbest"] = _align_profile_array(
                            invest_profile["daily_sr_lst"]
                        ).tolist()
                        step_data["dailySR_wocbf_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["daily_sr_wocbf_lst"]
                            ).tolist()
                        )
                        step_data["riskAccepted_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["risk_adj_lst"]
                            ).tolist()
                        )
                        step_data["ctrlWeight_policy_validbest"] = _align_profile_array(
                            invest_profile["ctrl_weight_lst"]
                        ).tolist()
                        step_data["solvable_flag_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["solvable_flag"]
                            ).tolist()
                        )
                        step_data["risk_pred_policy_validbest"] = _align_profile_array(
                            invest_profile["risk_pred_lst"]
                        ).tolist()
                        step_data["final_action_abssum_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["final_action_abssum_lst"]
                            ).tolist()
                        )
                        step_data["rl_action_abssum_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["rl_action_abssum_lst"]
                            ).tolist()
                        )
                        step_data["cbf_action_abssum_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["cbf_action_abssum_lst"]
                            ).tolist()
                        )
                        step_data["downsideAtVol_risk_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["daily_downsideAtVol_risk_lst"]
                            ).tolist()
                        )
                        step_data["downsideAtValue_risk_policy_validbest"] = (
                            _align_profile_array(
                                invest_profile["daily_downsideAtValue_risk_lst"]
                            ).tolist()
                        )
                        step_data["cvar_policy_validbest"] = _align_profile_array(
                            invest_profile["cvar_lst"]
                        ).tolist()
                        step_data["cvar_raw_policy_validbest"] = _align_profile_array(
                            invest_profile["cvar_raw_lst"]
                        ).tolist()
                        smart_print("-" * 30)
                        log_str = "Mode: Best-{}, Ep: {}, Capital (test set, by using the best validation model, {} ep): {} ".format(
                            self.mode,
                            self.epoch,
                            best_valid_ep,
                            np.array(step_data["capital_policy_validbest"])[-1],
                        )
                        smart_print(log_str)
                else:
                    smart_print(
                        f"[SAVE_PROFILE] valid_bestmodel.csv missing expected epoch column ({valid_ep_col}); available columns: {list(valid_records.columns)}"
                    )

        if invest_profile["ep"] == self.config.num_epochs:
            step_data["capital_policy_last"] = _align_profile_array(
                invest_profile["asset_lst"]
            ).tolist()
            step_data["dailyReturn_policy_last"] = _align_profile_array(
                invest_profile["daily_return_lst"]
            ).tolist()
            step_data["reward_policy_last"] = _align_profile_array(
                invest_profile["reward_lst"]
            ).tolist()
            step_data["strategyVolatility_policy_last"] = _align_profile_array(
                invest_profile["stg_vol_lst"]
            ).tolist()
            step_data["risk_policy_last"] = _align_profile_array(
                invest_profile["risk_lst"]
            ).tolist()
            step_data["risk_wocbf_policy_last"] = _align_profile_array(
                invest_profile["risk_wocbf_lst"]
            ).tolist()
            step_data["capital_wocbf_policy_last"] = _align_profile_array(
                invest_profile["capital_wocbf_lst"]
            ).tolist()
            step_data["dailySR_policy_last"] = _align_profile_array(
                invest_profile["daily_sr_lst"]
            ).tolist()
            step_data["dailySR_wocbf_policy_last"] = _align_profile_array(
                invest_profile["daily_sr_wocbf_lst"]
            ).tolist()
            step_data["riskAccepted_policy_last"] = _align_profile_array(
                invest_profile["risk_adj_lst"]
            ).tolist()
            step_data["ctrlWeight_policy_last"] = _align_profile_array(
                invest_profile["ctrl_weight_lst"]
            ).tolist()
            step_data["solvable_flag_policy_last"] = _align_profile_array(
                invest_profile["solvable_flag"]
            ).tolist()
            step_data["risk_pred_policy_last"] = _align_profile_array(
                invest_profile["risk_pred_lst"]
            ).tolist()
            step_data["final_action_abssum_policy_last"] = _align_profile_array(
                invest_profile["final_action_abssum_lst"]
            ).tolist()
            step_data["rl_action_abssum_policy_last"] = _align_profile_array(
                invest_profile["rl_action_abssum_lst"]
            ).tolist()
            step_data["cbf_action_abssum_policy_last"] = _align_profile_array(
                invest_profile["cbf_action_abssum_lst"]
            ).tolist()
            step_data["downsideAtVol_risk_policy_last"] = _align_profile_array(
                invest_profile["daily_downsideAtVol_risk_lst"]
            ).tolist()
            step_data["downsideAtValue_risk_policy_last"] = _align_profile_array(
                invest_profile["daily_downsideAtValue_risk_lst"]
            ).tolist()
            step_data["cvar_policy_last"] = _align_profile_array(
                invest_profile["cvar_lst"]
            ).tolist()
            step_data["cvar_raw_policy_last"] = _align_profile_array(
                invest_profile["cvar_raw_lst"]
            ).tolist()
        step_data.to_csv(fpath, index=False)

        # Save detailed portfolio weights (actions) to separate file
        if hasattr(self, "actions_memory") and len(self.actions_memory) > 0:
            action_df = self.save_action_memory()
            action_fpath = os.path.join(
                self.config.res_dir, "{}_actions.csv".format(self.mode)
            )
            action_df.to_csv(action_fpath, index=False)

    def run_mkt_observer(
        self,
        stage=None,
        rate_of_price_change=None,
        selection_trigger=False,
        selection_reason=None,
    ):
        cur_date = self.curData["date"].unique()[0]
        training_allowed = bool(
            getattr(self.config, "mafia_allow_observer_training", False)
        )
        self._last_selection_triggered = False
        self._last_selection_reason = None
        # Reset rebalance flags at start of each step (will be set if rebalance occurs)
        self.last_rebalance_scheduled = False
        self.last_rebalance_regime = False
        if not self.config.enable_market_observer or self.mkt_observer is None:
            default_eta = getattr(
                self.config,
                "risk_eta_default",
                getattr(self.config, "risk_default", 1.0),
            )
            return default_eta, None
        if self.config.enable_market_observer:
            if stage in ["reset", "init"] and (self.mode == "train"):
                self.mkt_observer.reset()

            # MAFIA needs raw OCHLV data with T_w=30 window
            raw_ochlv_data = self._extract_raw_ochlv_window(
                cur_date, window_size=self.config.mafia_T_w
            )
            # raw_ochlv_data shape: (N, M, T_w) where M=5 (OCHLV), T_w=30

            # Get market price for reward calculation
            finemkt_feat = self.extra_data["fine_market"]
            ma_close = finemkt_feat[finemkt_feat["date"] == cur_date][
                [
                    "mkt_{}_close".format(self.config.finefreq),
                    "mkt_{}_ma".format(self.config.finefreq),
                ]
            ].values[-1]
            mkt_cur_close_price = ma_close[0]
            mkt_ma_price = ma_close[1]
            # Index return for volatility/Regime detection
            idx_return = None
            if hasattr(
                self, "mkt_last_close_price"
            ) and self.mkt_last_close_price not in [None, 0]:
                try:
                    idx_return = (
                        mkt_cur_close_price / float(self.mkt_last_close_price)
                    ) - 1.0
                except Exception:
                    idx_return = None
            # Compute realized volatility using returns BEFORE current day (no look-ahead)
            vol_current = None
            vol_mu = None
            vol_sigma = None
            if len(self._index_return_buffer) >= 2:
                try:
                    vol_current = float(
                        np.std(np.array(self._index_return_buffer, dtype=float))
                    )
                except Exception:
                    vol_current = None
            if len(self._vol_history) >= 3:
                vols = np.array(self._vol_history, dtype=float)
                vol_mu = float(np.mean(vols))
                vol_sigma = float(np.std(vols))

            # Get stock MA price for controller
            finestock_feat = self.extra_data["fine_stock"]
            finestock_date_data = finestock_feat[finestock_feat["date"] == cur_date]

            # Ensure stock_ma_price has same stocks and order as curData
            stock_ma_price_dict = {}
            if len(finestock_date_data) > 0:
                for _, row in finestock_date_data.iterrows():
                    stock_ma_price_dict[row["stock"]] = row[
                        "stock_{}_ma".format(self.config.finefreq)
                    ]

            # Create stock_ma_price array aligned with curData stock order
            stock_ma_price = np.array(
                [
                    stock_ma_price_dict.get(
                        stock,
                        self.curData[self.curData["stock"] == stock]["close"].values[0]
                        if len(self.curData[self.curData["stock"] == stock]) > 0
                        else 1.0,
                    )
                    for stock in self.curData["stock"].values
                ]
            )
            if self.use_multibranch_state:
                full_ma_price = np.ones(self.stock_num, dtype=np.float32)
                for local_idx, stock in enumerate(self.curData["stock"].values):
                    idx = self.stock_index_map.get(stock)
                    if idx is None:
                        continue
                    full_ma_price[idx] = stock_ma_price[local_idx]
                self.latest_stock_ma_price = full_ma_price

            def _predict(
                pg_flag: bool, collect_pg: bool, collect_eta: bool, force_indices=None
            ):
                input_kwargs = {"mode": self.mode, "env": self}
                observer_mode = self.mode
                if observer_mode == "train" and not training_allowed:
                    observer_mode = "valid"
                input_kwargs["mode"] = observer_mode
                input_kwargs["pg_active"] = pg_flag
                input_kwargs["collect_pg"] = collect_pg
                input_kwargs["collect_eta"] = collect_eta
                input_kwargs["force_topk_indices"] = force_indices
                return self.mkt_observer.predict(
                    raw_ochlv_data=raw_ochlv_data, current_date=cur_date, **input_kwargs
                )

            scheduled_reason = None
            if selection_trigger:
                scheduled_reason = "schedule"
            force_indices = (
                None if selection_trigger else self._get_current_topk_indices()
            )
            (
                market_vector_np,
                risk_eta_np,
                market_scores_full_np,
                market_context_np,
                direction_logits_np,
                topk_indices_np,
                topk_embeddings_np,
                topk_scores_np,
                market_logits_np,
            ) = _predict(
                pg_flag=selection_trigger,
                collect_pg=selection_trigger,
                collect_eta=True,
                force_indices=force_indices,
            )

            # No gate_weights returned in optimized flow; keep zeros for compatibility
            self.gate_weights = np.zeros(4, dtype=np.float32)
            if market_context_np.ndim > 1:
                self.market_context_vec = market_context_np[-1].astype(np.float32)
            else:
                self.market_context_vec = market_context_np.astype(np.float32)

            # Suggested Top-K from current scores (used if regime forces rebalance)
            scores_vec = (
                market_scores_full_np[-1]
                if market_scores_full_np.ndim > 1
                else market_scores_full_np
            )
            scores_vec = scores_vec.flatten()
            k = min(self.rl_stock_num, len(scores_vec))
            suggested_topk = (
                np.argsort(scores_vec)[-k:][::-1] if k > 0 else np.array([], dtype=int)
            )

            # Continuous eta mapping from benchmark price z-score (spec)
            eta_window = int(getattr(self.config, "risk_eta_window", 50))
            eta_lambda = float(getattr(self.config, "risk_eta_lambda", 0.3))
            eta_k = float(getattr(self.config, "risk_eta_sensitivity", 1.0))

            # Maintain rolling buffer of market close prices
            try:
                if self._eta_price_buffer.maxlen != eta_window:
                    self._eta_price_buffer = deque(
                        self._eta_price_buffer, maxlen=eta_window
                    )
            except Exception:
                self._eta_price_buffer = deque(maxlen=eta_window)
            self._eta_price_buffer.append(float(mkt_cur_close_price))

            if len(self._eta_price_buffer) >= 2:
                prices = np.array(self._eta_price_buffer, dtype=float)
                ma = np.mean(prices)
                std = np.std(prices)
                if std < 1e-8:
                    std = 1e-6
                z_score = float((prices[-1] - ma) / std)
                adjustment = eta_lambda * np.tanh(eta_k * z_score)
                cur_risk_boundary = 1.0 + adjustment
                cur_risk_boundary = float(
                    np.clip(cur_risk_boundary, 1.0 - eta_lambda, 1.0 + eta_lambda)
                )
                self.last_trend_zscore = z_score
            else:
                cur_risk_boundary = float(getattr(self.config, "risk_eta_default", 1.0))
                self.last_trend_zscore = 0.0

            # Regime shift detection: update buffers and flag
            trend_z = getattr(self, "last_trend_zscore", None)
            # Direction prediction label (0=up/bull, 1=side, 2=down/bear)
            dir_label = 1  # default to sideways
            try:
                if direction_logits_np is not None:
                    logits = (
                        direction_logits_np[-1]
                        if hasattr(direction_logits_np, "__len__")
                        else direction_logits_np
                    )
                    dir_label = int(np.argmax(logits))
            except Exception:
                dir_label = 1
            self.last_mkt_direction_pred = dir_label
            self._direction_history.append(dir_label)
            # Update DC pivot for index
            dc_triggered = False
            if self._last_index_pivot is None:
                self._last_index_pivot = mkt_cur_close_price
            else:
                pct_change = (mkt_cur_close_price - self._last_index_pivot) / max(
                    self._last_index_pivot, 1e-8
                )
                dc_thresh = float(getattr(self.config, "regime_dc_threshold_pct", 0.05))
                if pct_change >= dc_thresh and self._last_dc_direction <= 0:
                    dc_triggered = True
                    self._last_dc_direction = 1
                    self._last_index_pivot = mkt_cur_close_price
                elif pct_change <= -dc_thresh and self._last_dc_direction >= 0:
                    dc_triggered = True
                    self._last_dc_direction = -1
                    self._last_index_pivot = mkt_cur_close_price

            # Check if regime shift detection is enabled (default: True)
            enable_regime_shift = getattr(self.config, "enable_regime_shift_detection", True)

            if enable_regime_shift:
                regime_shift, reason = self._detect_regime_shift(
                    dir_label=dir_label,
                    trend_z=trend_z,
                    vol_current=vol_current,
                    vol_mu=vol_mu,
                    vol_sigma=vol_sigma,
                    dc_triggered=dc_triggered,
                )
            else:
                regime_shift, reason = False, None

            self.regime_shift_detected = regime_shift
            self.regime_shift_reason = reason

            # Track regime shift count for summary
            if not hasattr(self, "regime_shift_count"):
                self.regime_shift_count = 0

            if regime_shift:
                self.regime_shift_count += 1
                # Descriptive reason mapping
                REGIME_SHIFT_REASONS = {
                    "dc_trigger_index": "Directional Change vượt ngưỡng (chỉ số thị trường thay đổi mạnh)",
                    "trend_reversal": "Đảo chiều xu hướng (Direction Head dự đoán thay đổi hướng)",
                    "vol_shock": "Shock biến động (Volatility vượt ngưỡng bất thường)",
                    "direction_change": "Thay đổi hướng thị trường ổn định (N ngày liên tục)",
                    "dc_trigger": "Directional Change kích hoạt",
                }
                MARKET_DIRECTION_NAMES = {
                    0: "🐻 Bear (Giảm)",
                    1: "➡️ Neutral (Đi ngang)",
                    2: "🐂 Bull (Tăng)",
                }
                reason_desc = REGIME_SHIFT_REASONS.get(reason, reason)
                dir_desc = MARKET_DIRECTION_NAMES.get(
                    dir_label, f"Unknown ({dir_label})"
                )

                # Format values safely (conditional inside f-string format specifier doesn't work)
                trend_z_str = f"{trend_z:.4f}" if trend_z is not None else "N/A"
                vol_current_str = (
                    f"{vol_current:.6f}" if vol_current is not None else "N/A"
                )
                vol_mu_str = (
                    f"{vol_mu:.6f}" if vol_mu is not None else "Chưa đủ dữ liệu"
                )
                vol_sigma_str = (
                    f"{vol_sigma:.6f}" if vol_sigma is not None else "Chưa đủ dữ liệu"
                )

                # Update LiveDisplay with regime shift (or fallback to print)
                _live_disp_on = (
                    LIVE_DISPLAY_AVAILABLE and get_display() and get_display().enabled
                )
                if _live_disp_on:
                    # Identify trigger type and extract from/to regimes
                    DIRECTION_NAMES = {0: "BEAR", 1: "FLAT", 2: "BULL"}
                    to_regime = DIRECTION_NAMES.get(dir_label, "FLAT")

                    if reason.startswith("direction_reversal"):
                        trigger_type = "direction_reversal"
                        # Parse from/to directly from reason string
                        # e.g., "direction_reversal_bear_to_bull_3d" -> from=BEAR, to=BULL
                        if "bear_to_bull" in reason:
                            from_regime = "BEAR"
                            to_regime = "BULL"
                        elif "bull_to_bear" in reason:
                            from_regime = "BULL"
                            to_regime = "BEAR"
                        else:
                            from_regime = to_regime  # fallback
                    elif reason.startswith("vol_shock"):
                        trigger_type = "vol_shock"
                        from_regime = to_regime  # vol_shock doesn't have from/to
                    elif reason.startswith("dc_trigger"):
                        trigger_type = "dc_trigger"
                        from_regime = to_regime  # dc_trigger doesn't have from/to
                    else:
                        trigger_type = ""
                        from_regime = to_regime

                    display_update_regime_shift(
                        str(cur_date),
                        reason_desc,
                        from_regime=from_regime,
                        to_regime=to_regime,
                        trigger_type=trigger_type,
                    )
                else:
                    # Use smart_print to route to log file when display is active
                    smart_print(
                        f"🚨 [REGIME SHIFT] {cur_date} | {reason_desc} | {dir_desc} | "
                        f"Z={trend_z_str} | vol={vol_current_str} | shift_count={self.regime_shift_count}"
                    )

            final_selection = bool(
                selection_trigger
                or (
                    regime_shift
                    and getattr(
                        self.config, "mafia_enable_regime_force_rebalance", True
                    )
                )
            )
            if final_selection and scheduled_reason is None and regime_shift:
                scheduled_reason = f"regime_shift:{reason}"

            # Set rebalance flags for TD3 state (reset first, then set if applicable)
            self.last_rebalance_scheduled = False
            self.last_rebalance_regime = False
            if final_selection:
                # Determine rebalance type
                if regime_shift and getattr(
                    self.config, "mafia_enable_regime_force_rebalance", True
                ):
                    self.last_rebalance_regime = True
                if selection_trigger and selection_reason in ("schedule", "init"):
                    self.last_rebalance_scheduled = True

            # If regime forced rebalance (or schedule was off), run observer again to collect PG on fresh Top-K
            if final_selection and not selection_trigger:
                (
                    market_vector_np,
                    risk_eta_np,
                    market_scores_full_np,
                    market_context_np,
                    direction_logits_np,
                    topk_indices_np,
                    topk_embeddings_np,
                    topk_scores_np,
                    market_logits_np,
                ) = _predict(
                    pg_flag=True,
                    collect_pg=True,
                    collect_eta=False,
                    force_indices=None,
                )

            # Update buffers AFTER detection to avoid using current-day vol as signal
            if idx_return is not None and np.isfinite(idx_return):
                self._index_return_buffer.append(idx_return)
            if len(self._index_return_buffer) >= 2:
                try:
                    vol_with_current = float(
                        np.std(np.array(self._index_return_buffer, dtype=float))
                    )
                    self._vol_history.append(vol_with_current)
                except Exception:
                    pass

            # Store Top-K scores/embeddings directly for RL state (aligned to active Top-K)
            prev_topk = self._get_current_topk_indices()

            # Track selection mask for cadence alignment training
            # 1 = rebalance day (train Stock Selection), 0 = holding day (skip)
            mask_val = 1 if final_selection else 0
            self._selection_mask_lst.append(mask_val)

            # Debug: Log mask accumulation every 10 steps
            if len(self._selection_mask_lst) % 10 == 0:
                rebalance_days = sum(self._selection_mask_lst)
                total_days = len(self._selection_mask_lst)
                pct = rebalance_days / total_days * 100 if total_days > 0 else 0
                smart_print(
                    f"[DEBUG CADENCE] Step {total_days}: mask_val={mask_val}, "
                    f"rebalance_days={rebalance_days}/{total_days} ({pct:.1f}%)"
                )

            if final_selection:
                active_topk = (
                    topk_indices_np[-1]
                    if topk_indices_np is not None
                    else suggested_topk
                )
                self._last_selection_triggered = True
                self._last_selection_reason = scheduled_reason or "unknown"
                
                # EXECUTION WORKFLOW (Spec 8.3 & 8.5): State Teleportation
                # Reset time decay immediately so the new state reflects fresh start
                self.days_since_rebalance = 0
                
                # Reset RL last action to uniform/neutral to break momentum from previous regime
                # (Spec 8.5.1 "rl_last_action to reset")
                if hasattr(self, "rl_stock_num") and self.rl_stock_num > 0:
                    self.rl_last_action = np.ones(self.rl_stock_num, dtype=np.float32) / self.rl_stock_num
            else:
                active_topk = prev_topk

            if active_topk is None or len(np.atleast_1d(active_topk)) == 0:
                active_topk = suggested_topk
            active_topk = np.array(active_topk, dtype=int).flatten()
            if active_topk.size > 0:
                active_topk = np.clip(active_topk, 0, self.stock_num - 1)
            self.observer_topk_indices = active_topk

            scores_full_vec = (
                market_scores_full_np[-1]
                if market_scores_full_np.ndim > 1
                else market_scores_full_np
            )
            scores_full_vec = np.nan_to_num(
                scores_full_vec, nan=0.0, posinf=0.0, neginf=0.0
            )
            topk_scores = (
                scores_full_vec[active_topk]
                if active_topk.size > 0
                else scores_full_vec[: self.rl_stock_num]
            )
            self.market_scores_full = topk_scores
            self.topk_indices = active_topk
            self.topk_embeddings = (
                topk_embeddings_np[-1] if topk_embeddings_np is not None else None
            )
            # Per-stock embedding for RL (K only)
            k = self.rl_stock_num
            self.per_stock_embedding = np.zeros(
                (k, self.config.mafia_D), dtype=np.float32
            )
            if self.topk_embeddings is not None:
                emb = self.topk_embeddings
                if emb.shape[0] < k:
                    pad = np.zeros(
                        (k - emb.shape[0], self.config.mafia_D), dtype=np.float32
                    )
                    emb = np.concatenate([emb, pad], axis=0)
                elif emb.shape[0] > k:
                    emb = emb[:k]
                self.per_stock_embedding = emb.astype(np.float32)

            # TOP-K REBALANCE: Observer chọn lại K cổ phiếu tốt nhất (thay đổi danh mục)
            if final_selection:
                prev_set = set(prev_topk.tolist()) if prev_topk is not None else set()
                new_set = set(active_topk.tolist())
                churn = len(prev_set.symmetric_difference(new_set))
                added_stocks = new_set - prev_set
                removed_stocks = prev_set - new_set

                # Get stock symbols if available
                def get_symbols(indices):
                    if hasattr(self, "stock_lst") and self.stock_lst is not None:
                        return [
                            self.stock_lst[i] if i < len(self.stock_lst) else f"idx_{i}"
                            for i in indices
                        ]
                    return [f"idx_{i}" for i in indices]

                prev_symbols = get_symbols(list(prev_set)[:5]) if prev_set else []
                new_symbols = get_symbols(list(new_set)[:5])
                added_symbols = get_symbols(list(added_stocks)[:3])
                removed_symbols = get_symbols(list(removed_stocks)[:3])

                # Reason description mapping
                SELECTION_REASONS = {
                    "scheduled_rebalance": "📅 Định kỳ rebalance (theo lịch)",
                    "regime_shift:dc_trigger_index": "🚨 Regime Shift - DC vượt ngưỡng",
                    "regime_shift:vol_shock": "🚨 Regime Shift - Shock biến động",
                    "regime_shift:trend_reversal": "🚨 Regime Shift - Đảo chiều xu hướng",
                    "regime_shift:direction_change": "🚨 Regime Shift - Đổi hướng thị trường",
                }
                reason_desc = SELECTION_REASONS.get(
                    self._last_selection_reason, self._last_selection_reason
                )

                # Update LiveDisplay with selection event (or fallback to print)
                _live_display_active = False
                if LIVE_DISPLAY_AVAILABLE:
                    _disp = get_display()
                    if _disp and _disp.enabled:
                        _live_display_active = True
                        # Store kept/added/removed for display
                        kept_stocks = prev_set & new_set
                        self._prev_topk_stocks = get_symbols(list(prev_set))
                        self._cur_topk_stocks = get_symbols(list(new_set))
                        display_update_selection(
                            step=self.curTradeDay,
                            date=str(cur_date),
                            trigger=self._last_selection_reason or "schedule",
                            kept=get_symbols(list(kept_stocks)),
                            added=get_symbols(list(added_stocks)),
                            removed=get_symbols(list(removed_stocks)),
                        )

                # Use smart_print to route to log file when display is active
                smart_print(
                    f"🔄 [TOP-K] {cur_date} day={self.curTradeDay} | {reason_desc} | K={len(active_topk)} | "
                    f"churn={churn} added={len(added_stocks)} removed={len(removed_stocks)}"
                )
            # SCHEDULER HOLD: Giữ nguyên danh mục Top-K hiện tại (không thay đổi)
            elif getattr(self.config, "mafia_log_scheduler", False):
                smart_print(
                    f"[SCHEDULER HOLD] Top-{len(active_topk)} | day={self.curTradeDay} date={cur_date}"
                )

            # Store info for realtime logging
            self._last_scheduler_info = {
                "rebalance": final_selection,
                "reason": self._last_selection_reason if final_selection else "hold",
                "k": len(active_topk),
            }

            # Build compact state for MAFIA optimized: [market_vector(K), portfolio_value, (optional) risk_boundary]
            # Removed market_index_state - rely on Observer's learned representation
            self._build_mafia_state(
                market_vector_np, finemkt_feat, cur_date, cur_risk_boundary
            )
            self.mkt_last_close_price = mkt_cur_close_price

            # Calculate eta delta for logging
            prev_eta = (
                self._last_eta_value
                if self._last_eta_value is not None
                else cur_risk_boundary
            )
            delta_eta = cur_risk_boundary - prev_eta
            self._last_eta_value = cur_risk_boundary

            # Store info for realtime logging
            self._last_eta_info = {
                "eta": cur_risk_boundary,
                "delta": delta_eta,
                "direction": "UP"
                if dir_label == 0
                else "DOWN"
                if dir_label == 2
                else "FLAT",
            }

            # Risk-Eta verbose log (separate lines) - optional
            if getattr(self.config, "mafia_log_eta", False):
                smart_print(
                    f"\n[RISK-ETA] Risk boundary: day={self.curTradeDay} date={cur_date} "
                    f"eta={cur_risk_boundary:.4f} (delta={delta_eta:+.4f}) "
                    f"market_direction={self._last_eta_info['direction']} "
                    f"rebalance={'YES' if final_selection else 'NO'}",
                    flush=True,
                )

            # ========== REALTIME SINGLE-LINE STATUS (recommended) ==========
            # Combines: day, date, eta, reward, scheduler action on ONE line
            if getattr(self.config, "mafia_log_realtime", True):
                reward_info = getattr(
                    self, "_last_reward_info", {"log_return": 0.0, "total_reward": 0.0}
                )
                sched_info = getattr(
                    self,
                    "_last_scheduler_info",
                    {"rebalance": False, "reason": "init", "k": 0},
                )
                eta_info = self._last_eta_info

                status_action = "REBAL" if sched_info["rebalance"] else "HOLD"
                # Use smart_print to route to log file when display is active
                smart_print(
                    f"[ENV] day={self.curTradeDay:>4} | {str(cur_date)[:10]} | "
                    f"η={eta_info['eta']:.3f}({eta_info['delta']:+.3f}) | "
                    f"mkt={eta_info['direction']:>4} | "
                    f"K={sched_info['k']:>2} {status_action:>5} | "
                    f"ret={reward_info['log_return']:+.4f} rew={reward_info['total_reward']:+.4f}"
                )
            if (
                (rate_of_price_change is not None)
                and (self.mode == "train")
                and training_allowed
            ):
                try:
                    last_close = getattr(self, "mkt_last_close_price", None)
                except Exception:
                    last_close = None
                if last_close is None or last_close <= 0:
                    market_return = 0.0
                else:
                    market_return = (mkt_cur_close_price / last_close) - 1.0
                if last_close is None:
                    mkt_direction = 1
                else:
                    if mkt_cur_close_price > last_close:
                        mkt_direction = 0
                    elif mkt_cur_close_price < last_close:
                        mkt_direction = 2
                    else:
                        mkt_direction = 1
                mkt_direction = np.array([mkt_direction])
                self.mkt_observer.update_hidden_vec_reward(
                    mode=self.mode,
                    rate_of_price_change=rate_of_price_change,
                    mkt_direction=mkt_direction,
                    market_return=np.array([market_return], dtype=np.float32),
                    current_date=cur_date,
                    env=self,
                    pg_active=final_selection,
                )
            # Reset emergency flags after applying selection decision
            self.force_rebalance = False
            self.emergency_rebalance = False

        return cur_risk_boundary, stock_ma_price

    def _extract_raw_ochlv_window(self, cur_date, window_size=30):
        """
        Extract raw OCHLV data for MAFIA.

        Args:
            cur_date: Current date
            window_size: Window size (T_w, default 30)

        Returns:
            ochlv_array: (N, M, T_w) where M=5 (open, close, high, low, volume), T_w=window_size
            N is fixed to len(self.stock_lst) to ensure consistency across windows
        """
        # Get current date index in rawdata
        date_mask = self.rawdata["date"] == cur_date
        if not date_mask.any():
            raise ValueError(f"Date {cur_date} not found in rawdata")

        date_indices = self.rawdata[date_mask].index
        if len(date_indices) == 0:
            raise ValueError(f"Date {cur_date} not found in rawdata")

        # Get the first occurrence (all stocks should have same date)
        date_idx = date_indices[0]

        # Get window: [date_idx - window_size + 1, date_idx]
        start_idx = max(0, date_idx - window_size + 1)
        end_idx = date_idx + 1

        # Extract window data
        window_data = self.rawdata.iloc[start_idx:end_idx].copy()

        # Use fixed stock list to ensure consistent number of stocks across windows
        stocks = self.stock_lst  # Fixed list of stocks
        N = len(stocks)  # Fixed number of stocks
        M = 5  # open, close, high, low, volume

        # Get all unique dates in window for proper time alignment
        window_dates = sorted(window_data["date"].unique())

        # Initialize output array
        ochlv_array = np.zeros((N, M, window_size))

        # Extract data for each stock
        for i, stock in enumerate(stocks):
            # Get stock data from window
            stock_data = window_data[window_data["stock"] == stock].sort_values("date")

            # Create DataFrame with all dates in window for proper alignment
            date_df = pd.DataFrame({"date": window_dates})
            stock_df = stock_data[
                ["date", "open", "close", "high", "low", "volume"]
            ].copy()

            # Merge to align dates (left join to keep all window_dates)
            merged_df = date_df.merge(stock_df, on="date", how="left")

            # Fill missing values: forward fill first, then backward fill
            # This handles cases where stock data is missing for some dates
            for col in ["open", "close", "high", "low", "volume"]:
                # Forward fill (use previous value)
                merged_df[col] = merged_df[col].ffill()
                # Backward fill (use next value for leading NaNs)
                merged_df[col] = merged_df[col].bfill()
                # If still NaN (shouldn't happen with proper data), use 0
                merged_df[col] = _fillna_infer(merged_df[col], value=0.0)

            # Ensure we have exactly window_size data points
            # Take the last window_size rows (most recent data)
            if len(merged_df) > window_size:
                merged_df = merged_df.iloc[-window_size:]
            elif len(merged_df) < window_size:
                # Pad with last row if needed
                last_row = (
                    merged_df.iloc[-1:]
                    if len(merged_df) > 0
                    else pd.DataFrame(
                        {
                            "date": [window_dates[-1]]
                            if len(window_dates) > 0
                            else [cur_date],
                            "open": [0.0],
                            "close": [0.0],
                            "high": [0.0],
                            "low": [0.0],
                            "volume": [0.0],
                        }
                    )
                )
                padding = pd.concat(
                    [last_row] * (window_size - len(merged_df)), ignore_index=True
                )
                merged_df = pd.concat([merged_df, padding], ignore_index=True)

            # Extract to array (take first window_size rows)
            ochlv_array[i, 0, :] = merged_df["open"].values[:window_size]  # Open
            ochlv_array[i, 1, :] = merged_df["close"].values[:window_size]  # Close
            ochlv_array[i, 2, :] = merged_df["high"].values[:window_size]  # High
            ochlv_array[i, 3, :] = merged_df["low"].values[:window_size]  # Low
            ochlv_array[i, 4, :] = merged_df["volume"].values[:window_size]  # Volume

        return ochlv_array

    def _extract_market_index_ochlv_window(self, cur_date, window_size=30):
        """
        Extract raw OCHLV data for Market-index Agent (VNINDEX).

        Args:
            cur_date: Current date
            window_size: Window size (T_w, default 30)

        Returns:
            ochlv_array: (1, 5, T_w) where 5 = [open, close, high, low, volume], T_w=window_size
            Single asset (VNINDEX) OCHLV window
        """
        import os
        from utils.data_validator import get_index_data_file

        # Ensure index data is loaded and augmented with regime features (past-only)
        if not hasattr(self, "_index_data_cache") or self._index_data_cache is None:
            fpath, error_msg = get_index_data_file(self.config, freq="1d")
            if fpath is None:
                raise ValueError(f"Cannot extract market index OCHLV: {error_msg}")
            index_data = pd.read_csv(fpath, header=0)
            index_data["date"] = pd.to_datetime(index_data["date"])
            if index_data["date"].dt.tz is not None:
                index_data["date"] = index_data["date"].dt.tz_localize(None)
            index_data = index_data.sort_values(
                "date", ascending=True, ignore_index=True
            )
            self._index_data_cache = index_data
        index_data = self._index_data_cache

        # Find current date in index data
        date_mask = index_data["date"] == cur_date
        if not date_mask.any():
            # Try to find closest date
            date_diffs = (index_data["date"] - cur_date).abs()
            closest_idx = date_diffs.idxmin()
            if date_diffs[closest_idx] > pd.Timedelta(days=7):
                raise ValueError(
                    f"Date {cur_date} not found in index data and no close date within 7 days"
                )
            date_idx = closest_idx
        else:
            date_idx = date_mask.idxmax()  # Get first occurrence

        # Get window: [date_idx - window_size + 1, date_idx]
        start_idx = max(0, date_idx - window_size + 1)
        end_idx = date_idx + 1

        # Extract window data
        window_data = index_data.iloc[start_idx:end_idx].copy()

        # Required columns: open, close, high, low, volume
        required_cols = ["open", "close", "high", "low", "volume"]
        missing_cols = [col for col in required_cols if col not in window_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in index data: {missing_cols}")

        # Initialize output array: (1, 5, window_size)
        ochlv_array = np.zeros((1, 5, window_size), dtype=np.float32)

        # Extract OCHLV data
        window_values = window_data[required_cols].values  # (actual_window_size, 5)

        # Ensure we have exactly window_size data points
        if len(window_values) > window_size:
            # Take the last window_size rows (most recent data)
            window_values = window_values[-window_size:]
        elif len(window_values) < window_size:
            # Pad with last row if needed
            if len(window_values) > 0:
                last_row = window_values[-1:]
                padding = np.tile(last_row, (window_size - len(window_values), 1))
                window_values = np.vstack([window_values, padding])
            else:
                # If no data, fill with zeros
                window_values = np.zeros((window_size, 5), dtype=np.float32)

        # Transpose and assign: (5, window_size) -> (1, 5, window_size)
        ochlv_array[0, :, :] = window_values.T

        return ochlv_array

    def _rolling_percentile(self, series: np.ndarray, window: int) -> np.ndarray:
        """
        Compute trailing percentile rank of the latest value within a rolling window.
        Uses only past data (inclusive) to avoid leakage.
        """
        n = len(series)
        out = np.zeros(n, dtype=np.float32)
        for i in range(n):
            start = max(0, i - window + 1)
            window_vals = series[start : i + 1]
            if len(window_vals) == 0:
                out[i] = 0.0
                continue
            current = window_vals[-1]
            rank = np.sum(window_vals <= current)
            out[i] = rank / float(len(window_vals))
        return out

    def _compute_adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Lightweight ADX computation (trailing, no future leakage).
        """
        plus_dm = np.zeros_like(close, dtype=np.float32)
        minus_dm = np.zeros_like(close, dtype=np.float32)
        tr = np.zeros_like(close, dtype=np.float32)
        for i in range(1, len(close)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        # Smooth with exponential moving average
        alpha = 1.0 / period
        atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (
            pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
            / atr.replace(0, np.nan)
        ).replace(np.nan, 0.0)
        minus_di = 100 * (
            pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
            / atr.replace(0, np.nan)
        ).replace(np.nan, 0.0)
        dx = 100 * (
            abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        ).replace(np.nan, 0.0)
        adx = dx.ewm(alpha=alpha, adjust=False).mean().values.astype(np.float32)
        return np.nan_to_num(adx, nan=0.0, posinf=0.0, neginf=0.0)

    def _augment_index_regime_features(self, index_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime-aware features to index_data using trailing windows only (no look-ahead).
        """
        df = index_data.copy()
        df["ret"] = df["close"].pct_change(fill_method=None).fillna(0.0)
        df["vol20"] = (
            df["ret"].rolling(window=20, min_periods=2).std(ddof=0).fillna(0.0)
        )
        # Drawdown vs trailing peak
        rolling_peak = df["close"].cummax().replace(0, np.nan)
        df["drawdown"] = ((rolling_peak - df["close"]) / rolling_peak).replace(
            np.nan, 0.0
        )
        # Percentiles using trailing windows
        df["vol_pct_252"] = self._rolling_percentile(df["vol20"].to_numpy(), window=252)
        df["vol_pct_756"] = self._rolling_percentile(df["vol20"].to_numpy(), window=756)
        df["dd_pct_252"] = self._rolling_percentile(
            df["drawdown"].to_numpy(), window=252
        )
        df["dd_pct_756"] = self._rolling_percentile(
            df["drawdown"].to_numpy(), window=756
        )
        # Trend proxies
        df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["ma200"] = df["close"].rolling(window=200, min_periods=1).mean()
        df["ma20_ma200_diff"] = (df["ma20"] - df["ma200"]).fillna(0.0)
        # ADX and percentile
        adx_vals = self._compute_adx(
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
            period=14,
        )
        df["adx14"] = adx_vals
        df["adx_pct_252"] = self._rolling_percentile(adx_vals, window=252)
        # Autocorr of returns (lag1) rolling
        df["autocorr_ret_60"] = (
            df["ret"]
            .rolling(window=60, min_periods=10)
            .apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
            .fillna(0.0)
        )
        df["autocorr_ret_120"] = (
            df["ret"]
            .rolling(window=120, min_periods=10)
            .apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
            .fillna(0.0)
        )
        # Tail shape
        df["skew20"] = df["ret"].rolling(window=20, min_periods=5).skew().fillna(0.0)
        df["kurt20"] = df["ret"].rolling(window=20, min_periods=5).kurt().fillna(0.0)
        # Clamp extremes to keep gating stable
        clamp_cols = [
            "vol_pct_252",
            "vol_pct_756",
            "dd_pct_252",
            "dd_pct_756",
            "adx_pct_252",
        ]
        for col in clamp_cols:
            df[col] = df[col].clip(0.0, 1.0)
        skew_kurt_cols = [
            "skew20",
            "kurt20",
            "autocorr_ret_60",
            "autocorr_ret_120",
            "ma20_ma200_diff",
        ]
        for col in skew_kurt_cols:
            df[col] = np.nan_to_num(df[col], nan=0.0, posinf=0.0, neginf=0.0)
        df[["vol20", "drawdown"]] = df[["vol20", "drawdown"]].fillna(0.0)
        return df

    def _get_market_regime_vector(self, cur_date):
        """
        Build regime feature vector for the current date using precomputed trailing stats.
        """
        if getattr(self.config, "mafia_regime_dim", 0) <= 0:
            return None
        if not hasattr(self, "_index_data_cache") or self._index_data_cache is None:
            return None
        idx_df = self._index_data_cache
        date_mask = idx_df["date"] == cur_date
        if not date_mask.any():
            date_diffs = (idx_df["date"] - cur_date).abs()
            closest_idx = date_diffs.idxmin()
            if date_diffs[closest_idx] > pd.Timedelta(days=7):
                return None
            row = idx_df.iloc[[closest_idx]]
        else:
            row = idx_df.loc[date_mask]
        cols = getattr(
            self.config,
            "mafia_regime_feature_names",
            [
                "vol_pct_252",
                "vol_pct_756",
                "dd_pct_252",
                "dd_pct_756",
                "ma20_ma200_diff",
                "adx_pct_252",
                "autocorr_ret_60",
                "autocorr_ret_120",
                "skew20",
                "kurt20",
            ],
        )
        missing = [c for c in cols if c not in row.columns]
        if missing:
            return None
        vec = row[cols].iloc[0].astype(np.float32).to_numpy()
        # Repeat across T_w when fed into observer
        return np.tile(vec.reshape(1, -1), (self.config.mafia_T_w, 1)).astype(
            np.float32
        )

    def _build_mafia_state(
        self, market_vector_np, finemkt_feat, cur_date, cur_risk_boundary
    ):
        """
        Build state for MAFIA optimized architecture.

        SIMPLIFIED: Removed market_index_state, relying only on Observer's learned representation.
        Spec (refactor_mafia.md): only compact Top-K mode is supported.

        Args:
            market_vector_np: (batch, N) or (N,) - Market vector from MAFIA (may have zeros for non-top-K)
            finemkt_feat: DataFrame - Fine market features (not used in simplified version)
            cur_date: Current date (not used in simplified version)
            cur_risk_boundary: Current risk boundary value
        """
        if self.use_multibranch_state:
            self._build_multibranch_state(cur_risk_boundary)
            return

        # Compact mode only: use Top-K from market_vector
        mv = market_vector_np[-1] if market_vector_np.ndim > 1 else market_vector_np
        k = min(self.config.mafia_top_k, len(mv))
        # Get top-K indices (non-zero positions)
        topk_idx = np.argpartition(mv, -k)[-k:]
        topk_sorted = topk_idx[np.argsort(mv[topk_idx])[::-1]]
        market_vector_state = mv[topk_sorted].astype(np.float32)

        # Portfolio value (log-normalized)
        pv = (
            np.log((self.cur_capital / self.initial_asset))
            if self.cur_capital > 0
            else 0.0
        )

        # Compose simplified state (no market_index_state)
        state_parts = [market_vector_state, np.array([pv], dtype=np.float32)]
        if getattr(self.config, "mafia_include_risk_boundary_in_state", False):
            state_parts.append(np.array([cur_risk_boundary], dtype=np.float32))

        self.state = np.concatenate(state_parts, axis=0).astype(np.float32)

    def _init_rl_multibranch_spec(self):
        self.rl_per_stock_feature_names = getattr(
            self.config, "rl_obs_per_stock_features", ["score", "embedding"]
        )
        # Dim = 1 (score) + mafia_D (embedding)
        self.rl_per_stock_feature_dim = 1 + self.config.mafia_D
        self.rl_obs_history_len = int(getattr(self.config, "rl_obs_history_len", 5))
        self.rl_history_feature_names = getattr(
            self.config,
            "rl_obs_history_features",
            [
                # Scalars (Internal State)
                "log_capital",
                "last_turnover_scalar", # renamed to avoid conflict with history turnover which might be duplicative? spec says "last_turnover" for scalar
                "drawdown",
                "time_decay",
                "relative_alpha",
                "risk_violation",
                # Sequence items
                "portfolio_return",
                "rl_last_action",
                "rl_last_turnover", # This is typically the same as last_turnover scalar but locally tracking history
                "cbf_adjustment",
            ],
        )
        self.rl_history_feature_dim = self._compute_history_feature_dim(
            self.rl_history_feature_names
        )
        # Keep essential portfolio scalars in global branch
        self.rl_global_scalar_items = getattr(
            self.config,
            "rl_obs_global_scalars",
            ["log_capital", "last_turnover", "drawdown"],
        )
        # global dim = market_context(D) + direction_logits(3) + selected scalars
        # regime features will be appended dynamically if available
        # Direction logits: one-hot vector (3,) for down/flat/up instead of scalar
        self._direction_dim = 3  # Per spec: direction_logits ∈ (3,)
        self._regime_dim = getattr(self.config, "mafia_regime_dim", None)
        if self._regime_dim is not None:
            try:
                self._regime_dim = max(0, int(self._regime_dim))
            except Exception:
                self._regime_dim = 0
        self.rl_global_dim = (
            self.config.mafia_D
            + self._direction_dim  # Changed from 1 to 3 per spec
            + len(self.rl_global_scalar_items)
            + (self._regime_dim or 0)
        )
        self.market_context_vec = np.zeros(self.config.mafia_D, dtype=np.float32)
        self._reset_rl_multibranch_buffers()

    def _reset_rl_multibranch_buffers(self):
        self.rl_history_buffer = deque(maxlen=self.rl_obs_history_len)
        for _ in range(self.rl_obs_history_len):
            self.rl_history_buffer.append(
                np.zeros(self.rl_history_feature_dim, dtype=np.float32)
            )
        self.rl_last_turnover = 0.0
        base_weight = 1.0 / max(1, self.rl_stock_num)
        self.rl_last_action = np.ones(self.rl_stock_num, dtype=np.float32) * base_weight
        self.peak_capital = self.initial_asset
        self.latest_stock_ma_price = np.ones(self.rl_stock_num, dtype=np.float32)
        self.per_stock_embedding = np.zeros(
            (self.rl_stock_num, self.config.mafia_D), dtype=np.float32
        )
        regime_dim = self._regime_dim or 0
        self.last_regime_vec = (
            np.zeros(regime_dim, dtype=np.float32) if regime_dim > 0 else None
        )

    def _log_reward_debug(
        self,
        j_return: float,
        scaled_profit_part: float,
        js_divergence: float,
        scaled_js_part: float,
        turnover_amt: float,
        turnover_penalty: float,
        change_penalty: float,
        cur_reward: float,
        show_penalty: bool = True,
    ):
        """Log reward components for early steps (controlled by config.reward_debug_steps)."""
        debug_steps = getattr(self.config, "reward_debug_steps", 0)
        if debug_steps <= 0 or self.mode != "train" or self.curTradeDay > debug_steps:
            return
        # Skip when LiveDisplay is active
        if _is_live_display_active():
            return
        try:
            # Clear any live status line before printing to avoid concatenation
            sys.stdout.write("\r\033[2K")
            if show_penalty:
                msg = (
                    "[REWARD-DEBUG] day={} j_return={:.6f} scaled_profit={:.4f} "
                    "js={:.6f} scaled_js={:.4f} turnover={:.4f} turnover_pen={:.4f} "
                    "change_pen={:.4f} reward={:.4f}"
                ).format(
                    self.curTradeDay,
                    float(j_return),
                    float(scaled_profit_part),
                    float(js_divergence),
                    float(scaled_js_part),
                    float(turnover_amt),
                    float(turnover_penalty),
                    float(change_penalty),
                    float(cur_reward),
                )
            else:
                msg = (
                    "[REWARD-DEBUG] day={} j_return={:.6f} scaled_profit={:.4f} "
                    "js={:.6f} scaled_js={:.4f} turnover={:.4f} reward={:.4f}"
                ).format(
                    self.curTradeDay,
                    float(j_return),
                    float(scaled_profit_part),
                    float(js_divergence),
                    float(scaled_js_part),
                    float(turnover_amt),
                    float(cur_reward),
                )
            smart_print(msg, flush=True)
        except Exception:
            pass

    def _build_multibranch_state(self, cur_risk_boundary):
        global_context = self._compose_global_context_vector(cur_risk_boundary)
        per_stock = self._compose_per_stock_features()
        history = self._compose_history_tensor(cur_risk_boundary)
        self.state = {
            "global_context": global_context,
            "per_stock": per_stock,
            "history": history,
        }

    def _detect_regime_shift(
        self, dir_label, trend_z, vol_current, vol_mu, vol_sigma, dc_triggered
    ):
        """
        Detect regime shift combining direction reversal, volatility shock, and DC trigger.
        dir_label: latest direction class (0 bull, 1 side, 2 bear)
        trend_z: trend z-score (float or None)
        vol_current: realized volatility (float or None)
        vol_mu/vol_sigma: baseline stats of volatility (float or None)
        dc_triggered: bool from DC-style reversal on index
        """
        reason = None
        # Direction reversal with confirmation window
        window = int(max(1, getattr(self.config, "regime_confirmation_window", 3)))
        trend_thr = float(getattr(self.config, "regime_trend_z_threshold", 1.0))
        hist = list(self._direction_history)
        if len(hist) >= window + 1:
            last_seq = hist[-window:]
            prev_regime = hist[-(window + 1)]
            # Standard MAFIA direction mapping: 0=Bear, 1=Side, 2=Bull
            bull, bear = 2, 0
            strong_trend = trend_z is None or abs(trend_z) >= trend_thr
            if (
                all(v == bull for v in last_seq)
                and prev_regime == bear
                and strong_trend
            ):
                reason = f"direction_reversal_bear_to_bull_{window}d"
            if (
                all(v == bear for v in last_seq)
                and prev_regime == bull
                and strong_trend
            ):
                reason = f"direction_reversal_bull_to_bear_{window}d"

        # Volatility shock
        if reason is None:
            k = float(getattr(self.config, "regime_vol_k", 3.0))
            if (
                vol_current is not None
                and vol_mu is not None
                and vol_sigma is not None
                and vol_sigma > 0
                and vol_current > vol_mu + k * vol_sigma
            ):
                reason = f"vol_shock_curr{vol_current:.4f}_mu{vol_mu:.4f}_sigma{vol_sigma:.4f}_k{k}"

        # DC trigger on index (Structural Break)
        # Spec 8.2.2: "Market DC 2.0% Event_Flag == 1.0 combined with State == -1 (Major Downward Reversal)"
        # Note: We only trigger regime shift on DOWNWARD DC (Crisis detection)
        if reason is None and dc_triggered:
            # Check direction: -1 indicates Downward DC in tradeEnv logic
            if getattr(self, "_last_dc_direction", 0) == -1:
                reason = "dc_trigger_index_bear"
        
        return (reason is not None), reason

    def _direction_to_onehot(self, direction):
        """
        Convert market direction scalar (0/1/2) to one-hot vector (3,).

        Args:
            direction: Scalar direction prediction (0=down, 1=flat, 2=up) or None

        Returns:
            np.ndarray: One-hot vector of shape (3,)
                - [1, 0, 0] for down (bear)
                - [0, 1, 0] for flat (sideways)
                - [0, 0, 1] for up (bull)
                - [0.33, 0.33, 0.33] if direction is None (uniform prior)
        """
        one_hot = np.zeros(3, dtype=np.float32)
        if direction is None:
            # Uniform prior when direction is unavailable
            one_hot[:] = 1.0 / 3.0
        else:
            try:
                idx = int(direction)
                if 0 <= idx < 3:
                    one_hot[idx] = 1.0
                else:
                    # Invalid index, use uniform
                    one_hot[:] = 1.0 / 3.0
            except (ValueError, TypeError):
                one_hot[:] = 1.0 / 3.0
        return one_hot

    def _compose_global_context_vector(self, cur_risk_boundary):
        market_context = getattr(self, "market_context_vec", None)
        if market_context is None or len(market_context) != self.config.mafia_D:
            market_context = np.zeros(self.config.mafia_D, dtype=np.float32)

        # Market direction: convert scalar (0/1/2) to one-hot vector (3,) per spec
        # 0 = down (bear), 1 = flat (sideways), 2 = up (bull)
        market_direction = getattr(self, "last_mkt_direction_pred", None)
        direction_logits = self._direction_to_onehot(market_direction)

        # Map scalar items to current values (Top-K aware via rl_last_turnover)
        # Includes Contextual Awareness Features (spec Section 3.4)
        # Plus Rebalance Signals for TD3 to learn distinct policies
        scalar_lookup = {
            "log_capital": np.log((self.cur_capital / self.initial_asset))
            if self.cur_capital > 0
            else 0.0,
            "last_turnover": self.rl_last_turnover,
            "drawdown": max(
                0.0, (self.peak_capital - self.cur_capital) / self.peak_capital
            )
            if self.peak_capital > 0
            else 0.0,
            # Contextual Awareness: time_decay = days_since_rebalance / rebalance_interval
            # τ → 0: Fresh signal (aggressive allocation), τ → 1: Stale signal (defensive)
            "time_decay": min(
                1.0,
                max(
                    0.0,
                    getattr(self, "days_since_rebalance", 0)
                    / max(
                        1,
                        getattr(
                            self, "topk_rebalance_interval", self.rebalance_interval
                        ),
                    ),
                ),
            ),
            # Contextual Awareness: relative_alpha = R_portfolio - R_topk_avg
            # Exoneration: RL outperforms in down market. Anti-Luck: RL underperforms in up market.
            "relative_alpha": getattr(self, "last_relative_alpha", 0.0),
            # Risk Violation Diagnostic: δ_risk = max(0, σ(a^RL) - σ_target)
            # δ_risk = 0: RL compliant (safe zone), δ_risk > 0: RL needs risk reduction
            "risk_violation": getattr(self, "last_risk_violation", 0.0),
            # Rebalance Signals: TD3 can learn different policies for rebalance vs hold
            # is_rebalance_scheduled: 1.0 on scheduled rebalance day (every N days), 0.0 otherwise
            "is_rebalance_scheduled": float(
                getattr(self, "last_rebalance_scheduled", False)
            ),
            # is_rebalance_regime: 1.0 when regime shift triggers early rebalance, 0.0 otherwise
            "is_rebalance_regime": float(getattr(self, "last_rebalance_regime", False)),
        }
        scalars = []
        for key in self.rl_global_scalar_items:
            scalars.append(float(scalar_lookup.get(key, 0.0)))
        padded_scalars = np.array(scalars, dtype=np.float32)

        # Optional regime features (trailing, no look-ahead)
        regime_vec = getattr(self, "last_regime_vec", None)
        if regime_vec is None and self._regime_dim:
            regime_vec = np.zeros(self._regime_dim, dtype=np.float32)
        regime_vec = (
            regime_vec.astype(np.float32)
            if regime_vec is not None
            else np.array([], dtype=np.float32)
        )
        full_vec = np.concatenate(
            [
                market_context.astype(np.float32),
                direction_logits,  # (3,) one-hot vector per spec
                padded_scalars,
                regime_vec,
            ],
            axis=0,
        )
        target_dim = int(self.rl_global_dim)
        if full_vec.shape[0] != target_dim:
            # Keep buffer shapes stable for DummyVecEnv by padding/trimming as needed
            if full_vec.shape[0] > target_dim:
                full_vec = full_vec[:target_dim]
            else:
                full_vec = np.concatenate(
                    [
                        full_vec,
                        np.zeros(target_dim - full_vec.shape[0], dtype=np.float32),
                    ],
                    axis=0,
                )
        return full_vec.astype(np.float32)

    def _compose_per_stock_features(self):
        target_k = self.rl_stock_num
        # Observer outputs are Top-K already; pad/truncate to K for stability
        scores = (
            np.array(self.market_scores_full, dtype=float)
            if self.market_scores_full is not None
            else np.zeros(target_k)
        )
        if scores.ndim > 1:
            scores = scores.flatten()
        if len(scores) < target_k:
            scores = np.concatenate([scores, np.zeros(target_k - len(scores))])
        elif len(scores) > target_k:
            scores = scores[:target_k]

        embeddings = (
            np.array(self.per_stock_embedding, dtype=float)
            if self.per_stock_embedding is not None
            else np.zeros((target_k, self.config.mafia_D))
        )
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[0] < target_k:
            pad_rows = target_k - embeddings.shape[0]
            embeddings = np.concatenate(
                [embeddings, np.zeros((pad_rows, self.config.mafia_D))], axis=0
            )
        elif embeddings.shape[0] > target_k:
            embeddings = embeddings[:target_k]
        if embeddings.shape[1] < self.config.mafia_D:
            pad_cols = self.config.mafia_D - embeddings.shape[1]
            embeddings = np.pad(embeddings, ((0, 0), (0, pad_cols)))
        elif embeddings.shape[1] > self.config.mafia_D:
            embeddings = embeddings[:, : self.config.mafia_D]

        features = np.zeros((target_k, self.rl_per_stock_feature_dim), dtype=np.float32)
        features[:, 0] = scores[:target_k]
        features[:, 1:] = embeddings[:target_k, : self.config.mafia_D]
        return features

    def _compose_history_tensor(self, cur_risk_boundary):
        self._update_history_buffer(cur_risk_boundary)
        stacked = np.stack(list(self.rl_history_buffer), axis=0)
        return stacked.astype(np.float32)

    def _update_history_buffer(self, cur_risk_boundary):
        portfolio_return = self.profit_lst[-1] if len(self.profit_lst) > 0 else 0.0
        raw_action_full = self._get_full_action_from_memory(self.action_rl_memory, -1)
        # Project action to Top-K space aligned with current observer indices
        if getattr(self, "observer_topk_indices", None) is not None:
            idx = np.array(self.observer_topk_indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < raw_action_full.shape[0])]
            current_action = (
                raw_action_full[idx]
                if idx.size > 0
                else raw_action_full[: self.rl_stock_num]
            )
        else:
            current_action = raw_action_full[: self.rl_stock_num]
        if current_action.size < self.rl_stock_num:
            pad = self.rl_stock_num - current_action.size
            current_action = np.concatenate(
                [current_action, np.zeros(pad, dtype=np.float32)]
            )
        current_action = current_action / (np.sum(np.abs(current_action)) + 1e-8)
        turnover = np.sum(
            np.abs(current_action - self.rl_last_action[: self.rl_stock_num])
        )
        self.rl_last_turnover = float(turnover)
        # Keep last action length aligned to Top-K
        if self.rl_last_action.shape[0] != self.rl_stock_num:
            self.rl_last_action = np.ones(self.rl_stock_num, dtype=np.float32) * (
                1.0 / max(1, self.rl_stock_num)
            )
        self.rl_last_action = current_action
        # Get CBF adjustment from previous step (a_final - a_alloc)
        cbf_adjustment = self._get_cbf_adjustment_topk()

        # Prepare scalar values for Internal State (Branch 3)
        scalar_values = {
            "log_capital": np.log((self.cur_capital / self.initial_asset))
            if self.cur_capital > 0
            else 0.0,
            "last_turnover_scalar": self.rl_last_turnover,
            "drawdown": max(
                0.0, (self.peak_capital - self.cur_capital) / self.peak_capital
            )
            if self.peak_capital > 0
            else 0.0,
            "time_decay": min(
                1.0,
                max(
                    0.0,
                    getattr(self, "days_since_rebalance", 0)
                    / max(
                        1,
                        getattr(
                            self, "topk_rebalance_interval", self.rebalance_interval
                        ),
                    ),
                ),
            ),
            "relative_alpha": getattr(self, "last_relative_alpha", 0.0),
            "risk_violation": getattr(self, "last_risk_violation", 0.0),
        }

        history_entry = []
        for name in self.rl_history_feature_names:
            if name == "portfolio_return":
                history_entry.append(portfolio_return)
            elif name == "rl_last_turnover":
                history_entry.append(self.rl_last_turnover)
            elif name == "rl_last_action":
                # Append full action vector (Top-K aligned)
                history_entry.extend(self.rl_last_action[: self.rl_stock_num].tolist())
            elif name == "cbf_adjustment":
                # Append CBF adjustment vector (Top-K aligned)
                history_entry.extend(cbf_adjustment[: self.rl_stock_num].tolist())
            elif name in scalar_values:
                history_entry.append(float(scalar_values[name]))
            else:
                history_entry.append(0.0)
        self.rl_history_buffer.append(np.array(history_entry, dtype=np.float32))

    def _get_cbf_adjustment_topk(self):
        """
        Get CBF adjustment vector (a_final - a_alloc) from previous step, aligned to Top-K.

        Returns:
            np.ndarray: CBF adjustment vector of shape (rl_stock_num,)
        """
        # Get the CBF adjustment from memory (stored as Top-K)
        if hasattr(self, "action_cbf_memeory") and len(self.action_cbf_memeory) > 0:
            cbf_adj = np.array(self.action_cbf_memeory[-1], dtype=np.float32).flatten()
        else:
            cbf_adj = np.zeros(self.rl_stock_num, dtype=np.float32)

        # Ensure correct length
        if cbf_adj.size < self.rl_stock_num:
            pad = self.rl_stock_num - cbf_adj.size
            cbf_adj = np.concatenate([cbf_adj, np.zeros(pad, dtype=np.float32)])
        elif cbf_adj.size > self.rl_stock_num:
            cbf_adj = cbf_adj[: self.rl_stock_num]

        return cbf_adj

    def _compute_history_feature_dim(self, feature_names):
        dim = 0
        for name in feature_names:
            if name == "rl_last_action":
                dim += self.rl_stock_num
            elif name == "cbf_adjustment":
                dim += self.rl_stock_num
            else:
                dim += 1
        return dim

    def _copy_observation(self, obs):
        if obs is None:
            return None
        if isinstance(obs, dict):
            return {k: np.array(v, copy=True) for k, v in obs.items()}
        if isinstance(obs, np.ndarray):
            return obs.copy()
        try:
            return copy.deepcopy(obs)
        except Exception:
            return obs


class StockPortfolioEnv_cash(StockPortfolioEnv):
    # Considering cash item
    def step(self, actions):
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        allow_observer_training = bool(
            getattr(self.config, "mafia_allow_observer_training", False)
        )
        if self.terminal:
            self.cur_capital = self.cur_capital * (
                1 - (np.sum(np.abs(self.actions_memory[-1])) * self.transaction_cost)
            )
            self.asset_lst[-1] = self.cur_capital
            self.profit_lst[-1] = (
                self.cur_capital - self.asset_lst[-2]
            ) / self.asset_lst[-2]
            if len(self.action_rl_memory) > 1:
                last_rl_full = self._get_full_action_from_memory(
                    self.action_rl_memory, -1
                )
                self.return_raw_lst[-1] = self.return_raw_lst[-1] * (
                    1 - (np.sum(np.abs(last_rl_full)) * self.transaction_cost)
                )

            if (
                (self.config.enable_market_observer)
                and (self.mkt_observer is not None)
                and (self.mode == "train")
                and allow_observer_training
            ):
                # Training at the end of epoch
                ori_profit_rate = np.append(
                    [1],
                    np.array(self.return_raw_lst)[1:]
                    / np.array(self.return_raw_lst)[:-1],
                    axis=0,
                )
                adj_profit_rate = np.array(self.profit_lst) + 1
                # Build selection mask for cadence alignment
                selection_mask = getattr(self, "_selection_mask_lst", None)
                if selection_mask is not None:
                    selection_mask = np.array(selection_mask, dtype=np.float32)
                label_kwargs = {
                    "mode": self.mode,
                    "ori_profit": ori_profit_rate,
                    "adj_profit": adj_profit_rate,
                    "ori_risk": np.array(self.risk_raw_lst),
                    "adj_risk": np.array(self.risk_cbf_lst),
                    "topk_selection_mask": selection_mask,
                }
                smart_print(
                    f"[MAFIA][OBSERVER] ⭐ Epoch-end train (cash env) @ epoch {self.epoch} | steps: {self.curTradeDay + 1}/{self.totalTradeDay}",
                    flush=True,
                )
                self.mkt_observer.train(**label_kwargs)
            elif (
                (self.config.enable_market_observer)
                and (self.mkt_observer is not None)
                and (self.mode == "train")
                and (not allow_observer_training)
            ):
                self.mkt_observer.reset()

            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True
            smart_print(
                f"[Profile Save] Epoch {self.epoch} complete (mode={self.mode}, cash), calling get_results() and save_profile()...",
                flush=True,
            )
            invest_profile = self.get_results()
            smart_print(
                f"[Profile Save] get_results() completed, calling save_profile()...",
                flush=True,
            )
            self.save_profile(invest_profile=invest_profile)
            smart_print(
                f"[Profile Save] save_profile() completed for epoch {self.epoch}",
                flush=True,
            )

            # Return format compatible with both gym and gymnasium
            # Always use gymnasium format (terminated, truncated, info) for consistency with VecEnv
            return (
                self.state,
                self.reward,
                self.terminal,
                False,
                self._build_step_info(),
            )
        else:
            actions = np.reshape(
                actions, (-1)
            )  # [1, num_of_stocks] or [num_of_stocks, ]
            actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            turnover_amt = 0.0
            if self.curTradeDay == 0:
                weights = self.weights_normalization(
                    actions=actions
                )  # Unnormalized weights -> normalized weights
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                self.actions_memory.append(weights[1:])
                self.cur_capital = self.cur_capital * (
                    1 - (1 - 1 / len(weights)) * self.transaction_cost
                )
                if not np.isfinite(self.cur_capital):
                    self.cur_capital = (
                        self.last_valid_capital
                        if np.isfinite(self.last_valid_capital)
                        else self.initial_asset
                    )
                if len(self.asset_lst) > 0:
                    self.asset_lst[-1] = self.cur_capital
                    self.profit_lst[-1] = 0.0
            else:
                is_rebalance_step = self.curTradeDay % self.rebalance_interval == 0
                # Ensure curData and lastDayData prices are aligned with stock_lst
                cur_close_prices = np.zeros(self.stock_num)
                last_close_prices = np.zeros(self.stock_num)
                for i, stock in enumerate(self.stock_lst):
                    if stock in self.curData["stock"].values:
                        close_val = self.curData[self.curData["stock"] == stock][
                            "close"
                        ].values[0]
                        # Validate and sanitize NaN/inf/<=0 to prevent NaN propagation
                        if pd.isna(close_val) or np.isinf(close_val) or close_val <= 0:
                            # Fallback: use last known price or 1.0
                            if (
                                self.lastDayData is not None
                                and stock in self.lastDayData["stock"].values
                            ):
                                last_close_val = self.lastDayData[
                                    self.lastDayData["stock"] == stock
                                ]["close"].values[0]
                                if not (
                                    pd.isna(last_close_val)
                                    or np.isinf(last_close_val)
                                    or last_close_val <= 0
                                ):
                                    cur_close_prices[i] = last_close_val
                                else:
                                    cur_close_prices[i] = 1.0
                            else:
                                cur_close_prices[i] = 1.0
                        else:
                            cur_close_prices[i] = close_val
                    else:
                        cur_close_prices[i] = 1.0  # Default to no change if missing

                    if (
                        self.lastDayData is not None
                        and stock in self.lastDayData["stock"].values
                    ):
                        last_close_val = self.lastDayData[
                            self.lastDayData["stock"] == stock
                        ]["close"].values[0]
                        # Validate and sanitize NaN/inf/<=0
                        if (
                            pd.isna(last_close_val)
                            or np.isinf(last_close_val)
                            or last_close_val <= 0
                        ):
                            # Fallback: use current price or 1.0
                            if not (
                                pd.isna(cur_close_prices[i])
                                or np.isinf(cur_close_prices[i])
                                or cur_close_prices[i] <= 0
                            ):
                                last_close_prices[i] = cur_close_prices[i]
                            else:
                                last_close_prices[i] = 1.0
                        else:
                            last_close_prices[i] = last_close_val
                    else:
                        # Validate cur_close_prices before using
                        if (
                            pd.isna(cur_close_prices[i])
                            or np.isinf(cur_close_prices[i])
                            or cur_close_prices[i] <= 0
                        ):
                            last_close_prices[i] = 1.0
                        else:
                            last_close_prices[i] = cur_close_prices[
                                i
                            ]  # Use current if last not available

        cur_p = cur_close_prices * (1 + self.cur_slippage_drift)
        last_p = last_close_prices * (1 + self.last_slippage_drift)
        x_p = cur_p / last_p
        last_action = np.array(self.actions_memory[-1])
        last_action = np.append(
            [1.0 - np.sum(np.abs(last_action))], last_action, axis=0
        )  # cash
        x_p_adj = np.where((x_p >= 2) & (last_action[1:] < 0), 2, x_p)
        x_p_adj = np.append([1.0], x_p_adj, axis=0)  # cash
        sgn = np.sign(last_action)
        sgn[0] = 1.0  # cash sign
        adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
        adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
        if (adj_cap <= 0) or np.all(adj_w_ay == 0):
            raise ValueError(
                "Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                    self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay
                )
            )
        last_w_adj = adj_w_ay / adj_cap
        if is_rebalance_step:
            weights = self.weights_normalization(
                actions=actions
            )  # Unnormalized weights -> normalized weights
            self.force_rebalance = False
            self.days_since_rebalance = 0
        else:
            weights = last_w_adj
            self.days_since_rebalance += 1
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        last_w_adj = np.nan_to_num(last_w_adj, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions_memory.append(weights[1:])
        turnover_amt = float(
            np.nan_to_num(
                np.sum(np.abs(weights[1:] - last_w_adj[1:])),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        self.cur_capital = self.cur_capital * (
            1 - (turnover_amt * self.transaction_cost)
        )
        if not np.isfinite(self.cur_capital):
            self.cur_capital = (
                self.last_valid_capital
                if np.isfinite(self.last_valid_capital)
                else self.initial_asset
            )
        self.asset_lst[-1] = self.cur_capital
        self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[
            -2
        ]
        if len(self.action_rl_memory) > 1:
            last_rl_action = self._get_full_action_from_memory(
                self.action_rl_memory, -2
            )
            last_rl_action = np.nan_to_num(
                last_rl_action, nan=0.0, posinf=0.0, neginf=0.0
            )
            last_rl_action = np.append(
                [1.0 - np.sum(np.abs(last_rl_action))], last_rl_action, axis=0
            )  # cash
            x_p_adjrl = np.where((x_p >= 2) & (last_rl_action[1:] < 0), 2, x_p)
            sgn_rl = np.sign(last_rl_action)
            sgn_rl[0] = 1.0  # cash sign
            prev_rl_cap = self.return_raw_lst[-1]
            adj_w_ay = sgn_rl * (
                last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action)
            )
            adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1
            if (adj_cap <= 0) or np.all(adj_w_ay == 0):
                smart_print(
                    "Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                        self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay
                    )
                )
                adj_w_ay = (
                    np.array([1 / (self.stock_num + 1)] * (self.stock_num + 1))
                    * self.bound_flag
                )
                adj_cap = 1
            last_rlw_adj = adj_w_ay / adj_cap
            cur_rl_full = self._get_full_action_from_memory(self.action_rl_memory, -1)
            return_raw = prev_rl_cap * (
                1
                - (
                    np.sum(np.abs(cur_rl_full - last_rlw_adj[1:]))
                    * self.transaction_cost
                )
            )
            self.return_raw_lst[-1] = return_raw

        # Jump to the next day
        self.curTradeDay = self.curTradeDay + 1
        # Save current data as lastDayData (ensure it has all stocks)
        self.lastDayData = self.curData.copy()
        self.last_slippage_drift = self.cur_slippage_drift
        self.curData = self._ensure_dataframe(
            self.rawdata.loc[self.curTradeDay, :].copy()
        )
        self.curData.sort_values(["stock"], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        self._ensure_minimal_features()

        self.ctl_state = self._build_ctl_state()
        cur_date = self.curData["date"].unique()[0]
        self.date_memory.append(cur_date)

        self.cur_slippage_drift = (
            np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
        )

        # Ensure curData and lastDayData have same stocks in same order
        cur_close_prices = np.zeros(self.stock_num)
        last_close_prices = np.zeros(self.stock_num)
        for i, stock in enumerate(self.stock_lst):
            if stock in self.curData["stock"].values:
                cur_close_prices[i] = self.curData[self.curData["stock"] == stock][
                    "close"
                ].values[0]
            else:
                cur_close_prices[i] = 1.0  # Default to no change if missing
            if (
                self.lastDayData is not None
                and stock in self.lastDayData["stock"].values
            ):
                last_close_prices[i] = self.lastDayData[
                    self.lastDayData["stock"] == stock
                ]["close"].values[0]
            else:
                last_close_prices[i] = cur_close_prices[
                    i
                ]  # Use current if last not available

        curDay_ClosePrice_withSlippage = cur_close_prices * (
            1 + self.cur_slippage_drift
        )
        lastDay_ClosePrice_withSlippage = last_close_prices * (
            1 + self.last_slippage_drift
        )
        rate_of_price_change = curDay_ClosePrice_withSlippage / (
            lastDay_ClosePrice_withSlippage + 1e-8
        )  # Avoid division by zero
        rate_of_price_change_adj = np.where(
            (rate_of_price_change >= 2) & (weights[1:] < 0), 2, rate_of_price_change
        )
        sigDayReturn = (rate_of_price_change_adj - 1) * weights[
            1:
        ]  # [s1_pct, s2_pct, .., px_pct_returns]
        poDayReturn = np.sum(sigDayReturn)
        if poDayReturn <= (-1):
            raise ValueError(
                "Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(
                    self.curTradeDay, self.date_memory[-1], poDayReturn
                )
            )

        updatePoValue = self.cur_capital * (
            (poDayReturn + 1 - np.abs(weights[0])) + np.abs(weights[0])
        )
        poDayReturn_withcost = (updatePoValue - self.cur_capital) / self.cur_capital
        self.cur_capital = updatePoValue

        self.profit_lst.append(poDayReturn_withcost)
        self.asset_lst.append(self.cur_capital)

        # Build MAFIA state via market observer
        # Pass selection_trigger for cadence alignment (mask PG loss on holding days)
        rate_of_price_change_withcash = np.append(
            [1.0], rate_of_price_change_adj, axis=0
        )
        selection_reason = "schedule" if is_rebalance_step else None
        cur_risk_boundary, stock_ma_price = self.run_mkt_observer(
            stage="run",
            rate_of_price_change=np.array([rate_of_price_change_withcash]),
            selection_trigger=is_rebalance_step,
            selection_reason=selection_reason,
        )
        if stock_ma_price is not None:
            self.ctl_state["MA-{}".format(self.config.otherRef_indicator_ma_window)] = (
                stock_ma_price
            )
        self.risk_adj_lst.append(cur_risk_boundary)
        self.ctrl_weight_lst.append(1.0)

        # For debugging
        daily_return_ay = _safe_array_from_values(
            self.curData[
                "DAILYRETURNS-{}".format(self.config.dailyRetun_lookback)
            ].values
        )
        cur_cov = np.cov(daily_return_ay)
        self.risk_cbf_lst.append(
            np.sqrt(np.matmul(np.matmul(weights[1:], cur_cov), weights[1:].T))
        )
        w_rl = self._get_full_action_from_memory(
            self.action_rl_memory, -1
        )  # Not applicable # weights[1:] - self.action_cbf_memeory[-1]
        w_rl = w_rl / np.sum(np.abs(w_rl))
        self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))

        if self.curTradeDay == 1:
            prev_rl_cap = self.return_raw_lst[-1] * (
                1 - (1 - 1 / len(weights)) * self.transaction_cost
            )
        else:
            prev_rl_cap = self.return_raw_lst[-1]

        rate_of_price_change_adj_rawrl = np.where(
            (rate_of_price_change >= 2) & (w_rl < 0), 2, rate_of_price_change
        )
        return_raw = prev_rl_cap * (
            (
                np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl)
                + 1
                - np.abs(weights[0])
            )
            + np.abs(weights[0])
        )
        if return_raw <= 0:
            raise ValueError(
                "Loss the whole capital if using RL actions only! [Day: {}, date: {}, return_raw: {}]".format(
                    self.curTradeDay, self.date_memory[-1], return_raw
                )
            )
        self.return_raw_lst.append(return_raw)

        # CVaR
        # daily_return_ay is 1D, take last 21 elements
        if daily_return_ay.ndim == 1:
            expected_r_series = (
                daily_return_ay[-21:] if len(daily_return_ay) >= 21 else daily_return_ay
            )
            expected_r_prev = (
                np.mean(expected_r_series[-1:]) if len(expected_r_series) > 0 else 0.0
            )
            # For 1D, expected_r_prev is scalar, convert to array matching weights[1:]
            expected_r_prev = np.full(len(weights[1:]), expected_r_prev)
            expected_r_prev = np.where(
                (expected_r_prev >= 1) & (weights[1:] < 0), 1, expected_r_prev
            )
            expected_r = np.sum(expected_r_prev * weights[1:])
            # For 1D series, covariance is scalar, create diagonal matrix
            if len(expected_r_series) < 2:
                expected_cov = np.eye(len(weights[1:])) * self.config.risk_market
            else:
                expected_cov_val = np.var(expected_r_series)
                expected_cov = np.eye(len(weights[1:])) * expected_cov_val
        else:
            expected_r_series = (
                daily_return_ay[:, -21:]
                if daily_return_ay.shape[1] >= 21
                else daily_return_ay
            )
            expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prev = np.where(
                (expected_r_prev >= 1) & (weights[1:] < 0), 1, expected_r_prev
            )
            expected_r = np.sum(
                np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights[1:], (-1, 1))
            )
            expected_cov = np.cov(expected_r_series)
            # Ensure expected_cov is 2D and matches weights[1:] shape
            if expected_cov.ndim == 0:
                expected_cov = np.eye(len(weights[1:])) * float(expected_cov)
            elif expected_cov.ndim == 1:
                expected_cov = np.diag(expected_cov)
            if expected_cov.shape[0] != len(weights[1:]) or expected_cov.shape[
                1
            ] != len(weights[1:]):
                if expected_cov.size == 1:
                    expected_cov = np.eye(len(weights[1:])) * float(
                        expected_cov.flat[0]
                    )
                else:
                    target_size = len(weights[1:])
                    if expected_cov.shape[0] < target_size:
                        pad_size = target_size - expected_cov.shape[0]
                        expected_cov = np.pad(
                            expected_cov,
                            ((0, pad_size), (0, pad_size)),
                            mode="constant",
                            constant_values=self.config.risk_market,
                        )
                    elif expected_cov.shape[0] > target_size:
                        expected_cov = expected_cov[:target_size, :target_size]

        # Calculate expected_std safely
        weights_1d = weights[1:].flatten() if weights[1:].ndim > 1 else weights[1:]
        try:
            temp = np.matmul(weights_1d, expected_cov)  # (N,) @ (N, N) -> (N,)
            expected_std = np.sqrt(np.matmul(temp, weights_1d))  # (N,) @ (N,) -> scalar
            if isinstance(expected_std, np.ndarray):
                expected_std = float(
                    expected_std.item()
                    if expected_std.size == 1
                    else expected_std.flat[0]
                )
        except (ValueError, TypeError, IndexError):
            expected_std = self.config.risk_market
        cvar_lz = spstats.norm.ppf(
            1 - 0.05
        )  # positive 1.65 for 95%(=1-alpha) confidence level.
        cvar_Z = np.exp(-0.5 * np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2 * np.pi)
        cvar_expected = -expected_r + expected_std * cvar_Z
        self.cvar_lst.append(cvar_expected)

        # CVaR without risk controller
        if expected_r_series.ndim == 1:
            expected_r_prevrl = (
                np.mean(expected_r_series[-1:]) if len(expected_r_series) > 0 else 0.0
            )
            expected_r_prevrl = np.full(len(w_rl), expected_r_prevrl)
            expected_r_prevrl = np.where(
                (expected_r_prevrl >= 1) & (w_rl < 0), 1, expected_r_prevrl
            )
            expected_r_raw = np.sum(expected_r_prevrl * w_rl)
        else:
            expected_r_prevrl = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prevrl = np.where(
                (expected_r_prevrl >= 1) & (w_rl < 0), 1, expected_r_prevrl
            )
            expected_r_raw = np.sum(
                np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1))
            )

        # Calculate expected_std_raw safely
        w_rl_1d = w_rl.flatten() if w_rl.ndim > 1 else w_rl
        try:
            temp = np.matmul(w_rl_1d, expected_cov)  # (N,) @ (N, N) -> (N,)
            expected_std_raw = np.sqrt(
                np.matmul(temp, w_rl_1d)
            )  # (N,) @ (N,) -> scalar
            if isinstance(expected_std_raw, np.ndarray):
                expected_std_raw = float(
                    expected_std_raw.item()
                    if expected_std_raw.size == 1
                    else expected_std_raw.flat[0]
                )
        except (ValueError, TypeError, IndexError):
            expected_std_raw = self.config.risk_market
        cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
        self.cvar_raw_lst.append(cvar_expected_raw)

        # ============ PORTFOLIO ALLOCATOR REWARD (NEW 2-COMPONENT) ============
        if self.use_pa_reward:
            # Extract a_alloc and a_final (cash env version)
            a_alloc = None
            a_final = None
            if len(self.action_rl_memory) > 0:
                a_alloc = self._get_full_action_from_memory(self.action_rl_memory, -1)
            if len(self.action_cbf_memeory) > 0:
                cbf_adj = self._get_full_action_from_memory(self.action_cbf_memeory, -1)
                if (
                    a_alloc is not None
                    and cbf_adj is not None
                    and len(cbf_adj) == len(a_alloc)
                ):
                    a_final = a_alloc + cbf_adj
                else:
                    a_final = a_alloc

            # Fallback
            if a_alloc is None:
                a_alloc = weights
            if a_final is None:
                a_final = weights

            # Compute reward
            reward_components = compute_pa_reward(
                portfolio_return=poDayReturn_withcost,
                a_alloc=a_alloc,
                a_final=a_final,
                w_return=self.pa_w_return,
                lambda_js=self.pa_lambda_js,
                reward_scale=self.pa_reward_scale,
            )

            cur_reward = reward_components["reward_total"]

            # Track components
            self.rl_reward_return_lst.append(reward_components["r_return"])
            self.rl_reward_divergence_lst.append(reward_components["r_divergence"])
            
            # Update Dashboard
            update_reward_components(
                reward_return=reward_components["r_return"],
                reward_js=-reward_components["r_divergence"], # Display as negative penalty
                reward_total=cur_reward,
                reward_unscaled=0.0, # Not calculated in PA mode
                js_divergence=reward_components.get("raw_divergence", 0.0), # Assuming available
                w_return=getattr(self, "pa_w_return", 1.0),
                lambda_js=getattr(self, "pa_lambda_js", 1.0),
                reward_scale=getattr(self, "pa_reward_scale", 1.0),
            )
        else:
            # ============ LEGACY REWARD (CASH ENV fallback) ============
            # Ensure poDayReturn_withcost is valid before taking log
            if np.isnan(poDayReturn_withcost) or np.isinf(poDayReturn_withcost):
                poDayReturn_withcost = 0.0
            # Avoid log(0) = -inf when poDayReturn_withcost = -1
            if poDayReturn_withcost <= -1:
                profit_part = -10.0  # Large negative value instead of -inf
            else:
                profit_part = np.log(poDayReturn_withcost + 1)
            if len(self.action_rl_memory) > 0:
                w_rl_latest = self._get_full_action_from_memory(
                    self.action_rl_memory, -1
                )
            else:
                w_rl_latest = weights
            weights_norm = _normalize_prob(weights)
            w_rl_norm = _normalize_prob(w_rl_latest)
            js_m = 0.5 * (w_rl_norm + weights_norm)
            js_divergence = 0.5 * entropy(
                pk=w_rl_norm, qk=js_m, base=2
            ) + 0.5 * entropy(pk=weights_norm, qk=js_m, base=2)
            if np.isnan(js_divergence) or np.isinf(js_divergence):
                js_divergence = 0.0
            j_return = profit_part
            scaled_profit_part = self.config.lambda_1 * j_return
            scaled_js_part = self.config.lambda_2 * js_divergence
            turnover_penalty = self.lambda_tc * turnover_amt
            change_penalty = 0.0  # cash env không dùng change penalty
            penalty_enabled = not getattr(
                self.config, "rl_reward_disable_turnover_change", True
            )
            total_penalty = turnover_penalty if penalty_enabled else 0.0
            cur_reward = scaled_profit_part - scaled_js_part - total_penalty

            self._log_reward_debug(
                j_return,
                scaled_profit_part,
                js_divergence,
                scaled_js_part,
                turnover_amt,
                turnover_penalty if penalty_enabled else 0.0,
                change_penalty,
                cur_reward,
                penalty_enabled,
            )

            self.rl_reward_risk_lst.append(scaled_js_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)

        # Ensure cur_reward is not NaN or inf
        if np.isnan(cur_reward) or np.isinf(cur_reward):
            smart_print(
                f"Warning: cur_reward is {cur_reward}, replacing with 0.0",
                flush=True,
            )
            cur_reward = 0.0
        self.reward = cur_reward
        self.reward_lst.append(self.reward)
        self.model_save_flag = False

        # Return format compatible with both gym and gymnasium
        # Always use gymnasium format (terminated, truncated, info) for consistency with VecEnv
        return self.state, self.reward, self.terminal, False, self._build_step_info()
