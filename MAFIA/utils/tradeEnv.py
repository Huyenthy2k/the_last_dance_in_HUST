#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: tradeEnv.py  
 Description: Define the trading environment for the trading agent.
--------------------------------
'''
import numpy as np
import os
import pandas as pd
import time
import copy
# Handle gym/gymnasium compatibility
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYMNASIUM_AVAILABLE = False
    except ImportError:
        raise ImportError("Neither gymnasium nor gym is installed")
try:
    from gym.utils import seeding
    GYM_SEEDING_AVAILABLE = True
except ImportError:
    GYM_SEEDING_AVAILABLE = False
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import scipy.stats as spstats
try:
    from utils.weight_symbol_mapper import get_top_stocks
except ImportError:
    # Fallback if weight_symbol_mapper is not available
    get_top_stocks = None

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
                elif hasattr(v, 'item'):
                    try:
                        result.append(float(v.item()))
                    except (ValueError, TypeError):
                        result.append(float(v) if isinstance(v, (int, float, np.number)) else 0.0)
                else:
                    result.append(float(v) if isinstance(v, (int, float, np.number)) else 0.0)
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
            elif hasattr(v, 'item'):
                try:
                    result.append(float(v.item()))
                except (ValueError, TypeError):
                    result.append(float(v) if isinstance(v, (int, float, np.number)) else 0.0)
            else:
                result.append(float(v) if isinstance(v, (int, float, np.number)) else 0.0)
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
                    cov,
                    ((0, pad), (0, pad)),
                    mode='constant',
                    constant_values=fallback
                )
                np.fill_diagonal(cov[-pad:, -pad:], fallback)
            else:
                cov = cov[:target_size, :target_size]

    cov = 0.5 * (cov + cov.T)
    cov += np.eye(target_size) * 1e-9
    return cov

class StockPortfolioEnv(gym.Env):

    def __init__(self, config, rawdata, mode, stock_num, action_dim, tech_indicator_lst=None, max_shares=None,
                 initial_asset=1000000, reward_scaling=1, norm_method='sum', transaction_cost=0.001, slippage=0.001, 
                 seed_num=2022, extra_data=None, mkt_observer=None):
        
        self.config = config
        self.rawdata = rawdata
        self.mode = mode # train, valid, test
        self.stock_num = stock_num # Number of stocks
        self.action_dim = action_dim # Number of assets
        self.validation_mode = False # Flag to indicate if environment is in validation mode
        
        # Handle tech_indicator_lst (optional for MAFIA, required for legacy)
        if tech_indicator_lst is None:
            tech_indicator_lst = []
        self.tech_indicator_lst = tech_indicator_lst
        self.tech_indicator_lst_wocov = copy.deepcopy(self.tech_indicator_lst) # without cov feature
        if 'cov' in self.tech_indicator_lst_wocov:
            self.tech_indicator_lst_wocov.remove('cov')
        self.max_shares = max_shares # Maximum number of shares
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

        self.initial_asset = initial_asset # Initial portfolio value
        self.reward_scaling = reward_scaling 
        self.norm_method = norm_method
        self.transaction_cost = transaction_cost # 0.001
        self.slippage = slippage # 0.001 for one-side, 0.002 for two-side
        self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
        if extra_data is not None:
            self.extra_data = extra_data
        else:
            self.extra_data = None

        if self.norm_method == 'softmax':
            self.weights_normalization = self.softmax_normalization
        elif self.norm_method == 'sum':
            self.weights_normalization = self.sum_normalization
        else:
            raise ValueError("Unexpected normalization method of stock weights: {}".format(self.norm_method))
        if self.config.enable_cov_features:        
            self.state_dim = ((len(self.tech_indicator_lst_wocov)+self.stock_num) * self.stock_num) + 1 # +1: current portfolio value 
        else:
            self.state_dim = (len(self.tech_indicator_lst_wocov) * self.stock_num) + 1 # +1: current portfolio value
        if self.config.enable_market_observer:
            self.mkt_observer = mkt_observer
            # If using MAFIA observer, override state dimension based on mode
            try:
                from RL_controller.mafia_observer import MAFIAObserver
                if isinstance(self.mkt_observer, MAFIAObserver):
                    state_mode = getattr(self.config, 'mafia_state_mode', 'compact')
                    risk_dim = 1 if getattr(self.config, 'mafia_include_risk_boundary_in_state', False) else 0
                    if state_mode == 'full-score':
                        # Full-score mode: state = [market_scores_full(N), portfolio_value, (optional) risk_boundary]
                        # Removed market_index_state - rely on Observer's learned representation
                        self.state_dim = self.stock_num + 1 + risk_dim
                    else:
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
        self.observer_topk_indices = None  # Top-K indices selected by Observer (for mode 'compact')
        
        # Validate MAFIA state mode if using MAFIA observer
        if self.config.enable_market_observer and self.mkt_observer is not None:
            try:
                from RL_controller.mafia_observer import MAFIAObserver
                if isinstance(self.mkt_observer, MAFIAObserver):
                    state_mode = getattr(self.config, 'mafia_state_mode', 'compact')
                    if state_mode not in ['compact', 'full-score']:
                        raise ValueError(f"Invalid mafia_state_mode: {state_mode}. Must be 'compact' or 'full-score'")
                    # Log current mode for debugging
                    if state_mode == 'full-score':
                        print(f"[MAFIA] State mode: FULL-SCORE (N={self.stock_num} stocks in state)")
                    else:
                        print(f"[MAFIA] State mode: COMPACT (K={self.config.mafia_top_k} stocks in state)")
            except Exception:
                pass
        
        if self.config.benchmark_algo in self.config.only_long_algo_lst:
            # Long only
            self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim, ), dtype=np.float32)
            self.bound_flag = 1 # 1 for long and long+short, -1 for short
        else:
            if self.config.trade_pattern == 1:
                # Long only
                self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim, ), dtype=np.float32)
                self.bound_flag = 1 # 1 for long and long+short, -1 for short
            else:
                raise ValueError("Unexpected trade pattern: {}".format(self.config.trade_pattern))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim, ), dtype=np.float32)

        self.rawdata.sort_values(['date', 'stock'], ascending=True, inplace=True)
        self.rawdata.index = self.rawdata.date.factorize()[0]
        self.totalTradeDay = len(self.rawdata['date'].unique())
        self.stock_lst = np.sort(self.rawdata['stock'].unique())

        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)

        # Initialize state (will be built in run_mkt_observer)
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst}
        self.terminal = False
        
        # Initialize capital before running market observer (needed for state building)
        self.cur_capital = self.initial_asset
        self.last_valid_capital = self.cur_capital
        self.last_valid_capital = self.cur_capital

        self.profit_lst = [0] # percentage of portfolio daily returns
        cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage='init') # after curData and state, before cur_risk_boundary
        if stock_ma_price is not None:
            self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
        
        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]

        self.asset_lst = [self.initial_asset] 
        self.date_memory = [self.curData['date'].unique()[0]]
        self.reward_lst = [0]
        self.action_cbf_memeory = [np.array([0] * self.stock_num)]

        self.actions_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag] 
        self.action_rl_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]

        self.risk_adj_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0] # For performance analysis. Record the risk without using risk controllrt during the validation/test period.
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset] 
        self.solver_stat = {'solvable': 0, 'insolvable': 0, 'stochastic_solvable': 0, 'stochastic_time': [], 'socp_solvable': 0, 'socp_time': []} 

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
        if self.mode == 'train':
            self.exclusive_cputime = 0
            self.exclusive_systime = 0
        # For saving profile
        self.profile_hist_field_lst = [            
            'ep', 'trading_days', 'annualReturn_pct', 'mdd', 'sharpeRatio', 'final_capital', 'volatility', 
            'calmarRatio', 'sterlingRatio',
            'netProfit', 'netProfit_pct', 'winRate',
            'vol_max', 'vol_min', 'vol_avg', 
            'risk_max', 'risk_min', 'risk_avg', 'riskRaw_max', 'riskRaw_min', 'riskRaw_avg',
            'dailySR_max', 'dailySR_min', 'dailySR_avg', 'dailySR_wocbf_max', 'dailySR_wocbf_min', 'dailySR_wocbf_avg',
            'dailyReturn_pct_max', 'dailyReturn_pct_min', 'dailyReturn_pct_avg',
            'sigReturn_max', 'sigReturn_min', 'mdd_high', 'mdd_low', 'mdd_high_date', 'mdd_low_date', 'sharpeRatio_wocbf',
            'reward_sum', 'final_capital_wocbf', 'cbf_contribution',
            'risk_downsideAtVol', 'risk_downsideAtVol_daily_max', 'risk_downsideAtVol_daily_min', 'risk_downsideAtVol_daily_avg',
            'risk_downsideAtValue_daily_max', 'risk_downsideAtValue_daily_min', 'risk_downsideAtValue_daily_avg',
            'cvar_max', 'cvar_min', 'cvar_avg', 'cvar_raw_max', 'cvar_raw_min', 'cvar_raw_avg',
            'solver_solvable', 'solver_insolvable', 'cputime', 'systime', 
        ]
        self.profile_hist_ep = {k: [] for k in self.profile_hist_field_lst}
    

    def step(self, actions):
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        # Debug: Log when episode ends
        if self.terminal:
            print(f"[ENV] ⚠️ Episode end detected! curTradeDay={self.curTradeDay}, totalTradeDay={self.totalTradeDay}, terminal={self.terminal}, epoch={self.epoch}", flush=True)
        if self.terminal:
            self.cur_capital = self.cur_capital * (1 - self.transaction_cost)
            self.asset_lst[-1] = self.cur_capital
            self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]   
            if len(self.action_rl_memory) > 1:
                self.return_raw_lst[-1] = self.return_raw_lst[-1] * (1 - self.transaction_cost)

            if (self.config.enable_market_observer) and (self.mode == 'train'):
                # Training at the end of epoch
                ori_profit_rate = np.append([1], np.array(self.return_raw_lst)[1:] / np.array(self.return_raw_lst)[:-1], axis=0)
                adj_profit_rate = np.array(self.profit_lst) + 1
                label_kwargs = {'mode': self.mode, 'ori_profit': ori_profit_rate, 'adj_profit': adj_profit_rate, 'ori_risk': np.array(self.risk_raw_lst), 'adj_risk': np.array(self.risk_cbf_lst)}
                self.mkt_observer.train(**label_kwargs)       
            
            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True

            # Skip save_profile during validation to prevent array length conflicts
            if not self.validation_mode:
                print(f"[Profile Save] Epoch {self.epoch} complete (mode={self.mode}), calling get_results() and save_profile()...", flush=True)
                invest_profile = self.get_results()
                print(f"[Profile Save] get_results() completed, calling save_profile()...", flush=True)
                self.save_profile(invest_profile=invest_profile)
                print(f"[Profile Save] save_profile() completed for epoch {self.epoch}", flush=True)
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

            # Return format compatible with both gym and gymnasium
            # Always use gymnasium format (terminated, truncated, info) for consistency with VecEnv
            # Debug: Log return value
            print(f"[ENV] Returning terminal={self.terminal} (will be converted to dones array by VecEnv)", flush=True)
            return self.state, self.reward, self.terminal, False, {}
        else:
            actions = np.reshape(actions, (-1)) # [1, num_of_stocks] or [num_of_stocks, ]
            weights = self.weights_normalization(actions=actions) # Unnormalized weights -> normalized weights
            self.actions_memory.append(weights)
            if self.curTradeDay == 0:
                self.cur_capital = self.cur_capital * (1 - self.transaction_cost)
            else:
                # Ensure curData and lastDayData prices are aligned with stock_lst
                cur_close_prices = np.zeros(self.stock_num)
                last_close_prices = np.zeros(self.stock_num)
                for i, stock in enumerate(self.stock_lst):
                    if stock in self.curData['stock'].values:
                        close_val = self.curData[self.curData['stock'] == stock]['close'].values[0]
                        # Validate and sanitize NaN/inf/<=0 to prevent NaN propagation
                        if pd.isna(close_val) or np.isinf(close_val) or close_val <= 0:
                            self.nan_stats['cur_close_nan'] += 1
                            cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else 'N/A'
                            fallback_used = None
                            # Fallback: use last known price or 1.0
                            if self.lastDayData is not None and stock in self.lastDayData['stock'].values:
                                last_close_val = self.lastDayData[self.lastDayData['stock'] == stock]['close'].values[0]
                                if not (pd.isna(last_close_val) or np.isinf(last_close_val) or last_close_val <= 0):
                                    cur_close_prices[i] = last_close_val
                                    fallback_used = f'last_known_price({last_close_val:.4f})'
                                else:
                                    cur_close_prices[i] = 1.0
                                    fallback_used = '1.0_default'
                            else:
                                cur_close_prices[i] = 1.0
                                fallback_used = '1.0_default'
                            # Log details (limit to first 10 per epoch to avoid spam)
                            if len(self.nan_stats['cur_close_details']) < 10:
                                self.nan_stats['cur_close_details'].append({
                                    'day': self.curTradeDay,
                                    'date': cur_date,
                                    'stock': stock,
                                    'value': close_val,
                                    'fallback': fallback_used
                                })
                        else:
                            cur_close_prices[i] = close_val
                    else:
                        cur_close_prices[i] = 1.0  # Default to no change if missing
                    
                    if self.lastDayData is not None and stock in self.lastDayData['stock'].values:
                        last_close_val = self.lastDayData[self.lastDayData['stock'] == stock]['close'].values[0]
                        # Validate and sanitize NaN/inf/<=0
                        if pd.isna(last_close_val) or np.isinf(last_close_val) or last_close_val <= 0:
                            self.nan_stats['last_close_nan'] += 1
                            cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else 'N/A'
                            fallback_used = None
                            # Fallback: use current price or 1.0
                            if not (pd.isna(cur_close_prices[i]) or np.isinf(cur_close_prices[i]) or cur_close_prices[i] <= 0):
                                last_close_prices[i] = cur_close_prices[i]
                                fallback_used = f'current_price({cur_close_prices[i]:.4f})'
                            else:
                                last_close_prices[i] = 1.0
                                fallback_used = '1.0_default'
                            # Log details (limit to first 10 per epoch to avoid spam)
                            if len(self.nan_stats['last_close_details']) < 10:
                                self.nan_stats['last_close_details'].append({
                                    'day': self.curTradeDay,
                                    'date': cur_date,
                                    'stock': stock,
                                    'value': last_close_val,
                                    'fallback': fallback_used
                                })
                        else:
                            last_close_prices[i] = last_close_val
                    else:
                        # Validate cur_close_prices before using
                        if pd.isna(cur_close_prices[i]) or np.isinf(cur_close_prices[i]) or cur_close_prices[i] <= 0:
                            last_close_prices[i] = 1.0
                        else:
                            last_close_prices[i] = cur_close_prices[i]  # Use current if last not available
                
                cur_p = cur_close_prices * (1 + self.cur_slippage_drift)
                last_p = last_close_prices * (1 + self.last_slippage_drift)
                x_p = cur_p / last_p
                last_action = np.array(self.actions_memory[-2])
                x_p_adj = np.where((x_p>=2)&(last_action<0), 2, x_p)
                sgn = np.sign(last_action)
                # Check if loss the whole capital
                adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
                adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                if (adj_cap <= 0) or np.all(adj_w_ay==0):
                    raise ValueError("Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                last_w_adj = adj_w_ay / adj_cap
                self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1] - last_w_adj)) * self.transaction_cost))
                self.asset_lst[-1] = self.cur_capital
                self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]
                if len(self.action_rl_memory) > 1:
                    last_rl_action = np.array(self.action_rl_memory[-2])
                    sgn_rl = np.sign(last_rl_action)
                    prev_rl_cap = self.return_raw_lst[-1]
                    x_p_adjrl = np.where((x_p>=2)&(last_rl_action<0), 2, x_p)
                    adj_w_ay = sgn_rl * (last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action))
                    adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1
                    if (adj_cap <= 0) or np.all(adj_w_ay==0):
                        print("Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                        adj_w_ay = np.array([1/self.stock_num]*self.stock_num) * self.bound_flag
                        adj_cap = 1
                    last_rlw_adj =  adj_w_ay / adj_cap 
                    return_raw = prev_rl_cap * (1 - (np.sum(np.abs(self.action_rl_memory[-1] - last_rlw_adj)) * self.transaction_cost))
                    self.return_raw_lst[-1] = return_raw

            # Jump to the next day
            self.curTradeDay = self.curTradeDay + 1
            # Save current data as lastDayData (ensure it has all stocks)
            self.lastDayData = self.curData.copy()
            self.last_slippage_drift = self.cur_slippage_drift
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.curData.sort_values(['stock'], ascending=True, inplace=True)
            self.curData.reset_index(drop=True, inplace=True)
            
            # Ensure curData has all expected stocks (from stock_lst)
            # If some stocks are missing, add them with NaN values (will be handled later)
            if len(self.curData) < len(self.stock_lst):
                missing_stocks = set(self.stock_lst) - set(self.curData['stock'].unique())
                if len(missing_stocks) > 0:
                    # Add missing stocks with NaN values (will use forward fill from lastDayData if available)
                    cur_date = self.curData['date'].unique()[0] if len(self.curData) > 0 else self.date_memory[-1] if len(self.date_memory) > 0 else None
                    if cur_date is not None:
                        for stock in missing_stocks:
                            # Try to get from lastDayData first
                            if self.lastDayData is not None and stock in self.lastDayData['stock'].values:
                                last_row = self.lastDayData[self.lastDayData['stock'] == stock].iloc[0].copy()
                                last_row['date'] = cur_date
                                self.curData = pd.concat([self.curData, last_row.to_frame().T], ignore_index=True)
                            else:
                                # Create row with NaN values (will be filled later)
                                new_row = pd.DataFrame({
                                    'stock': [stock],
                                    'date': [cur_date],
                                    'open': [np.nan], 'close': [np.nan], 'high': [np.nan], 'low': [np.nan], 'volume': [np.nan]
                                })
                                # Add other required columns if they exist
                                for col in self.curData.columns:
                                    if col not in new_row.columns:
                                        new_row[col] = np.nan
                                self.curData = pd.concat([self.curData, new_row], ignore_index=True)
                        # Re-sort after adding missing stocks
                        self.curData.sort_values(['stock'], ascending=True, inplace=True)
                        self.curData.reset_index(drop=True, inplace=True)
                        
                        # Fill NaN values with forward fill from lastDayData or use last valid value
                        if self.lastDayData is not None:
                            for col in ['open', 'close', 'high', 'low', 'volume']:
                                if col in self.curData.columns:
                                    # Fill NaN with corresponding value from lastDayData
                                    for idx, row in self.curData.iterrows():
                                        if pd.isna(row[col]) and row['stock'] in self.lastDayData['stock'].values:
                                            last_val = self.lastDayData[self.lastDayData['stock'] == row['stock']][col].values
                                            if len(last_val) > 0:
                                                self.curData.at[idx, col] = last_val[0]
                                    # If still NaN, fill with 0 (shouldn't happen with proper data)
                                    self.curData[col] = _fillna_infer(self.curData[col], value=0.0)
            
            self.ctl_state = {k:_safe_array_from_values(self.curData[k].values) for k in self.config.otherRef_indicator_lst}
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)

            self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
            
            # Ensure curData and lastDayData have same stocks in same order
            cur_close_prices = np.zeros(self.stock_num)
            last_close_prices = np.zeros(self.stock_num)
            for i, stock in enumerate(self.stock_lst):
                if stock in self.curData['stock'].values:
                    close_val = self.curData[self.curData['stock'] == stock]['close'].values[0]
                    # Validate and sanitize NaN/inf/<=0 to prevent NaN propagation
                    if pd.isna(close_val) or np.isinf(close_val) or close_val <= 0:
                        self.nan_stats['cur_close_nan'] += 1
                        cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else 'N/A'
                        fallback_used = None
                        # Fallback: use last known price or 1.0
                        if self.lastDayData is not None and stock in self.lastDayData['stock'].values:
                            last_close_val = self.lastDayData[self.lastDayData['stock'] == stock]['close'].values[0]
                            if not (pd.isna(last_close_val) or np.isinf(last_close_val) or last_close_val <= 0):
                                cur_close_prices[i] = last_close_val
                                fallback_used = f'last_known_price({last_close_val:.4f})'
                            else:
                                cur_close_prices[i] = 1.0
                                fallback_used = '1.0_default'
                        else:
                            cur_close_prices[i] = 1.0
                            fallback_used = '1.0_default'
                        # Log details (limit to first 10 per epoch to avoid spam)
                        if len(self.nan_stats['cur_close_details']) < 10:
                            self.nan_stats['cur_close_details'].append({
                                'day': self.curTradeDay,
                                'date': cur_date,
                                'stock': stock,
                                'value': close_val,
                                'fallback': fallback_used
                            })
                    else:
                        cur_close_prices[i] = close_val
                else:
                    cur_close_prices[i] = 1.0  # Default to no change if missing
                
                if self.lastDayData is not None and stock in self.lastDayData['stock'].values:
                    last_close_val = self.lastDayData[self.lastDayData['stock'] == stock]['close'].values[0]
                    # Validate and sanitize NaN/inf/<=0
                    if pd.isna(last_close_val) or np.isinf(last_close_val) or last_close_val <= 0:
                        self.nan_stats['last_close_nan'] += 1
                        cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else 'N/A'
                        fallback_used = None
                        # Fallback: use current price or 1.0
                        if not (pd.isna(cur_close_prices[i]) or np.isinf(cur_close_prices[i]) or cur_close_prices[i] <= 0):
                            last_close_prices[i] = cur_close_prices[i]
                            fallback_used = f'current_price({cur_close_prices[i]:.4f})'
                        else:
                            last_close_prices[i] = 1.0
                            fallback_used = '1.0_default'
                        # Log details (limit to first 10 per epoch to avoid spam)
                        if len(self.nan_stats['last_close_details']) < 10:
                            self.nan_stats['last_close_details'].append({
                                'day': self.curTradeDay,
                                'date': cur_date,
                                'stock': stock,
                                'value': last_close_val,
                                'fallback': fallback_used
                            })
                    else:
                        last_close_prices[i] = last_close_val
                else:
                    # Validate cur_close_prices before using
                    if pd.isna(cur_close_prices[i]) or np.isinf(cur_close_prices[i]) or cur_close_prices[i] <= 0:
                        last_close_prices[i] = 1.0
                    else:
                        last_close_prices[i] = cur_close_prices[i]  # Use current if last not available
            
            curDay_ClosePrice_withSlippage = cur_close_prices * (1 + self.cur_slippage_drift)
            lastDay_ClosePrice_withSlippage = last_close_prices * (1 + self.last_slippage_drift)
            rate_of_price_change = curDay_ClosePrice_withSlippage / (lastDay_ClosePrice_withSlippage + 1e-8)  # Avoid division by zero
            rate_of_price_change_adj = np.where((rate_of_price_change>=2)&(weights<0), 2, rate_of_price_change)
            sigDayReturn = (rate_of_price_change_adj - 1) * weights # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)
            
            # Validate poDayReturn to prevent NaN propagation
            if np.isnan(poDayReturn) or np.isinf(poDayReturn):
                self.nan_stats['poDayReturn_nan'] += 1
                cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else 'N/A'
                print(f"Warning: poDayReturn is {poDayReturn}, using 0.0 as fallback (Day: {self.curTradeDay}, date: {cur_date})", flush=True)
                # Log details (limit to first 10 per epoch to avoid spam)
                if len(self.nan_stats['poDayReturn_details']) < 10:
                    self.nan_stats['poDayReturn_details'].append({
                        'day': self.curTradeDay,
                        'date': cur_date,
                        'value': poDayReturn
                    })
                poDayReturn = 0.0
            
            if poDayReturn <= (-1):
                raise ValueError("Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(self.curTradeDay, self.date_memory[-1], poDayReturn))

            prev_cur_capital = self.cur_capital  # Save previous value for poDayReturn_withcost calculation
            updatePoValue = self.cur_capital * (poDayReturn + 1)
            # Validate updatePoValue to prevent NaN propagation
            if np.isnan(updatePoValue) or np.isinf(updatePoValue) or updatePoValue <= 0:
                self.nan_stats['updatePoValue_nan'] += 1
                cur_date = self.date_memory[-1] if len(self.date_memory) > 0 else 'N/A'
                fallback_cap = self.last_valid_capital if np.isfinite(self.last_valid_capital) else self.initial_asset
                print(f"Warning: updatePoValue is {updatePoValue}, using fallback capital {fallback_cap} (Day: {self.curTradeDay}, date: {cur_date})", flush=True)
                # Log details (limit to first 10 per epoch to avoid spam)
                if len(self.nan_stats['updatePoValue_details']) < 10:
                    self.nan_stats['updatePoValue_details'].append({
                        'day': self.curTradeDay,
                        'date': cur_date,
                        'value': updatePoValue,
                        'prev_capital': prev_cur_capital,
                        'poDayReturn': poDayReturn
                    })
                updatePoValue = fallback_cap  # Keep previous valid value
            
            self.cur_capital = updatePoValue
            if np.isfinite(self.cur_capital):
                self.last_valid_capital = self.cur_capital
            poDayReturn_withcost = (updatePoValue - prev_cur_capital) / (prev_cur_capital + 1e-8)  # Avoid division by zero
            
            self.profit_lst.append(poDayReturn_withcost)
            self.asset_lst.append(self.cur_capital)

            # Build MAFIA state via market observer
            cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage='run', rate_of_price_change=np.array([rate_of_price_change]))
            if stock_ma_price is not None:
                self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
            self.risk_adj_lst.append(cur_risk_boundary)
            self.ctrl_weight_lst.append(1.0)       

            daily_return_ay = self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values
            cur_cov = _build_cov_from_indicator(
                daily_return_ay,
                target_size=len(weights),
                fallback=self.config.risk_market
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
                    risk_cbf = float(risk_cbf.item() if risk_cbf.size == 1 else risk_cbf.flat[0])
                self.risk_cbf_lst.append(float(risk_cbf))
            except (ValueError, TypeError, IndexError) as e:
                self.risk_cbf_lst.append(self.config.risk_market)
            
            w_rl = self.action_rl_memory[-1] # weights - self.action_cbf_memeory[-1]
            # Ensure w_rl is 1D
            w_rl = w_rl.flatten() if w_rl.ndim > 1 else w_rl
            w_rl = w_rl / (np.sum(np.abs(w_rl)) + 1e-8)  # Avoid division by zero
            try:
                # w_rl @ cur_cov @ w_rl.T
                temp = np.matmul(w_rl, cur_cov)  # (N,) @ (N, N) -> (N,)
                risk_raw = np.sqrt(np.matmul(temp, w_rl))  # (N,) @ (N,) -> scalar
                if np.isnan(risk_raw) or np.isinf(risk_raw):
                    risk_raw = self.config.risk_market
                # Ensure scalar
                if isinstance(risk_raw, np.ndarray):
                    risk_raw = float(risk_raw.item() if risk_raw.size == 1 else risk_raw.flat[0])
                self.risk_raw_lst.append(float(risk_raw))
            except (ValueError, TypeError, IndexError) as e:
                self.risk_raw_lst.append(self.config.risk_market)

            if self.curTradeDay == 1:
                prev_rl_cap = self.return_raw_lst[-1] * (1 - self.transaction_cost)
            else:
                prev_rl_cap = self.return_raw_lst[-1]
            
            rate_of_price_change_adj_rawrl = np.where((rate_of_price_change>=2)&(w_rl<0), 2, rate_of_price_change)
            po_r_rl = np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl)
            if po_r_rl <= (-1):
                raise ValueError("Loss the whole capital if using RL actions only! [Day: {}, date: {}, po_r_rl: {}]".format(self.curTradeDay, self.date_memory[-1], po_r_rl))
            return_raw = prev_rl_cap * (po_r_rl + 1) 
            self.return_raw_lst.append(return_raw)

            # CVaR
            # daily_return_ay is 1D, take last 21 elements
            if daily_return_ay.ndim == 1:
                expected_r_series = daily_return_ay[-21:] if len(daily_return_ay) >= 21 else daily_return_ay
                expected_r_prev = np.mean(expected_r_series[-1:]) if len(expected_r_series) > 0 else 0.0
                # For 1D, expected_r_prev is scalar, convert to array matching weights
                expected_r_prev = np.full(len(weights), expected_r_prev)
                expected_r_prev = np.where((expected_r_prev>=1)&(weights<0), 1, expected_r_prev)
                expected_r = np.sum(expected_r_prev * weights)
                # For 1D series, covariance is scalar, create diagonal matrix
                if len(expected_r_series) < 2:
                    expected_cov = np.eye(len(weights)) * self.config.risk_market
                else:
                    expected_cov_val = np.var(expected_r_series)
                    expected_cov = np.eye(len(weights)) * expected_cov_val
            else:
                expected_r_series = daily_return_ay[:, -21:] if daily_return_ay.shape[1] >= 21 else daily_return_ay
                expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
                expected_r_prev = np.where((expected_r_prev>=1)&(weights<0), 1, expected_r_prev)
                expected_r = np.sum(np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights, (-1, 1)))
                expected_cov = np.cov(expected_r_series)
                # Ensure expected_cov is 2D and matches weights shape
                if expected_cov.ndim == 0:
                    expected_cov = np.eye(len(weights)) * float(expected_cov)
                elif expected_cov.ndim == 1:
                    expected_cov = np.diag(expected_cov)
                if expected_cov.shape[0] != len(weights) or expected_cov.shape[1] != len(weights):
                    if expected_cov.size == 1:
                        expected_cov = np.eye(len(weights)) * float(expected_cov.flat[0])
                    else:
                        target_size = len(weights)
                        if expected_cov.shape[0] < target_size:
                            pad_size = target_size - expected_cov.shape[0]
                            expected_cov = np.pad(expected_cov, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=self.config.risk_market)
                        elif expected_cov.shape[0] > target_size:
                            expected_cov = expected_cov[:target_size, :target_size]
            
            # Calculate expected_std safely
            weights_1d = weights.flatten() if weights.ndim > 1 else weights
            try:
                temp = np.matmul(weights_1d, expected_cov)  # (N,) @ (N, N) -> (N,)
                expected_std = np.sqrt(np.matmul(temp, weights_1d))  # (N,) @ (N,) -> scalar
                if isinstance(expected_std, np.ndarray):
                    expected_std = float(expected_std.item() if expected_std.size == 1 else expected_std.flat[0])
            except (ValueError, TypeError, IndexError):
                expected_std = self.config.risk_market
            cvar_lz = spstats.norm.ppf(1-0.05) # positive 1.65 for 95%(=1-alpha) confidence level.
            cvar_Z = np.exp(-0.5*np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2*np.pi)
            cvar_expected = -expected_r + expected_std * cvar_Z
            self.cvar_lst.append(cvar_expected)

            # CVaR without risk controller
            if expected_r_series.ndim == 1:
                expected_r_prevrl = np.mean(expected_r_series[-1:]) if len(expected_r_series) > 0 else 0.0
                expected_r_prevrl = np.full(len(w_rl), expected_r_prevrl)
                expected_r_prevrl = np.where((expected_r_prevrl>=1)&(w_rl<0), 1, expected_r_prevrl)
                expected_r_raw = np.sum(expected_r_prevrl * w_rl)
            else:
                expected_r_prevrl = np.mean(expected_r_series[:, -1:], axis=1)
                expected_r_prevrl = np.where((expected_r_prevrl>=1)&(w_rl<0), 1, expected_r_prevrl)
                expected_r_raw = np.sum(np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1)))
            
            # Calculate expected_std_raw safely
            w_rl_1d = w_rl.flatten() if w_rl.ndim > 1 else w_rl
            try:
                temp = np.matmul(w_rl_1d, expected_cov)  # (N,) @ (N, N) -> (N,)
                expected_std_raw = np.sqrt(np.matmul(temp, w_rl_1d))  # (N,) @ (N,) -> scalar
                if isinstance(expected_std_raw, np.ndarray):
                    expected_std_raw = float(expected_std_raw.item() if expected_std_raw.size == 1 else expected_std_raw.flat[0])
            except (ValueError, TypeError, IndexError):
                expected_std_raw = self.config.risk_market
            cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
            self.cvar_raw_lst.append(cvar_expected_raw)

            # Ensure poDayReturn_withcost is valid before taking log
            if np.isnan(poDayReturn_withcost) or np.isinf(poDayReturn_withcost):
                poDayReturn_withcost = 0.0
            # Avoid log(0) = -inf when poDayReturn_withcost = -1
            if poDayReturn_withcost <= -1:
                profit_part = -10.0  # Large negative value instead of -inf
            else:
                profit_part = np.log(poDayReturn_withcost+1)
            # Unified reward: log-return + Jensen-Shannon diversity
            if len(self.action_rl_memory) > 0:
                w_rl_latest = self.action_rl_memory[-1]
            else:
                w_rl_latest = weights
            weights_norm = _normalize_prob(weights)
            w_rl_norm = _normalize_prob(w_rl_latest)
            js_distance = jensenshannon(w_rl_norm, weights_norm, base=2) ** 2
            if np.isnan(js_distance) or np.isinf(js_distance):
                js_distance = 0.0
            j_return = profit_part  # log-return at current step
            scaled_profit_part = self.config.lambda_1 * j_return
            scaled_js_part = self.config.lambda_2 * js_distance
            cur_reward = scaled_profit_part + scaled_js_part

            self.rl_reward_risk_lst.append(scaled_js_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)
            # Ensure cur_reward is not NaN or inf (root cause fix)
            if np.isnan(cur_reward) or np.isinf(cur_reward):
                print(f"Warning: cur_reward is {cur_reward} (profit_part={profit_part}, scaled_js_part={scaled_js_part}), replacing with 0.0", flush=True)
                cur_reward = 0.0
            self.reward = cur_reward
            self.reward_lst.append(self.reward)
            self.model_save_flag = False
            # Return format compatible with both gym and gymnasium
            # Always use gymnasium format (terminated, truncated, info) for consistency with VecEnv
            return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        # Handle gymnasium compatibility (seed and options parameters)
        if seed is not None:
            self.seed_num = seed
            np.random.seed(seed)
        self.epoch = self.epoch + 1
        self.curTradeDay = 0
        
        # Reset NaN statistics for new epoch
        self.nan_stats = {
            'cur_close_nan': 0,
            'last_close_nan': 0,
            'poDayReturn_nan': 0,
            'updatePoValue_nan': 0,
            'cur_close_details': [],
            'last_close_details': [],
            'poDayReturn_details': [],
            'updatePoValue_details': []
        }

        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        
        # Initialize state (will be built in run_mkt_observer)
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst} 
        self.terminal = False

        self.profit_lst = [0] 
        cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage='reset')  
        if stock_ma_price is not None:
            self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price

        self.cur_capital = self.initial_asset
        self.last_valid_capital = self.cur_capital


        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]

        self.asset_lst = [self.initial_asset] 

        self.actions_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]
        self.date_memory = [self.curData['date'].unique()[0]]
        self.reward_lst = [0]
        self.action_cbf_memeory = [np.array([0] * self.stock_num)]
        self.action_rl_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]

        self.risk_adj_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0]
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset]
        self.solver_stat = {'solvable': 0, 'insolvable': 0, 'stochastic_solvable': 0, 'stochastic_time': [], 'socp_solvable': 0, 'socp_time': []} 

        self.ctrl_weight_lst = [1.0]
        self.solvable_flag = []
        self.risk_pred_lst = []

        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.cnt1 = 0
        self.cnt2 = 0
        self.stepcount = 0
        self.latest_invest_profile = None

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

    def render(self, mode='human'):
        return self.state
    

    def softmax_normalization(self, actions):
        if np.sum(np.abs(actions)) == 0:
            norm_weights = np.array([1/len(actions)]*len(actions)) * self.bound_flag
        else:
            # Proper softmax: exp(x_i) / sum(exp(x_j))
            # For short positions (bound_flag = -1), negate actions first
            if self.bound_flag < 0:
                actions = -actions  # Flip actions for short positions
            exp_actions = np.exp(actions - np.max(actions))  # Subtract max for numerical stability
            norm_weights = exp_actions / np.sum(exp_actions)
            norm_weights = norm_weights * self.bound_flag
        return norm_weights
    
    def sum_normalization(self, actions):
        if np.sum(np.abs(actions)) == 0:
            norm_weights = np.array([1/len(actions)]*len(actions)) * self.bound_flag
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

    def save_action_memory(self):

        action_arr = np.array(self.actions_memory)
        columns = list(self.stock_lst)
        if len(columns) != action_arr.shape[1]:
            if len(columns) > action_arr.shape[1]:
                columns = columns[:action_arr.shape[1]]
            else:
                columns = columns + [f'UNNAMED_{i}' for i in range(action_arr.shape[1] - len(columns))]
            print(f"[SAVE_PROFILE] Warning: stock list length mismatch (actions: {action_arr.shape[1]}, columns: {len(self.stock_lst)}). Adjusting columns for consistency.", flush=True)
        action_pd = pd.DataFrame(action_arr, columns=columns)
        action_pd['date'] = self.date_memory
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
            'epoch': self.epoch,
            'curTradeDay': self.curTradeDay,
            'cur_capital': self.cur_capital,
            'curData': self.curData.copy() if hasattr(self, 'curData') and self.curData is not None else None,
            'lastDayData': self.lastDayData.copy() if hasattr(self, 'lastDayData') and self.lastDayData is not None else None,
            'cur_slippage_drift': self.cur_slippage_drift.copy() if hasattr(self, 'cur_slippage_drift') else None,
            'last_slippage_drift': self.last_slippage_drift.copy() if hasattr(self, 'last_slippage_drift') else None,
            'state': self.state.copy() if hasattr(self, 'state') and self.state is not None else None,
            'ctl_state': {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.ctl_state.items()} if hasattr(self, 'ctl_state') else None,
            'terminal': self.terminal if hasattr(self, 'terminal') else False,
            'profit_lst': self.profit_lst.copy() if hasattr(self, 'profit_lst') else None,
            'asset_lst': self.asset_lst.copy() if hasattr(self, 'asset_lst') else None,
            'actions_memory': [a.copy() if isinstance(a, np.ndarray) else a for a in self.actions_memory] if hasattr(self, 'actions_memory') else None,
            'date_memory': self.date_memory.copy() if hasattr(self, 'date_memory') else None,
            'reward_lst': self.reward_lst.copy() if hasattr(self, 'reward_lst') else None,
            'action_cbf_memeory': [a.copy() if isinstance(a, np.ndarray) else a for a in self.action_cbf_memeory] if hasattr(self, 'action_cbf_memeory') else None,
            'action_rl_memory': [a.copy() if isinstance(a, np.ndarray) else a for a in self.action_rl_memory] if hasattr(self, 'action_rl_memory') else None,
            'risk_adj_lst': self.risk_adj_lst.copy() if hasattr(self, 'risk_adj_lst') else None,
            'risk_raw_lst': self.risk_raw_lst.copy() if hasattr(self, 'risk_raw_lst') else None,
            'risk_cbf_lst': self.risk_cbf_lst.copy() if hasattr(self, 'risk_cbf_lst') else None,
            'return_raw_lst': self.return_raw_lst.copy() if hasattr(self, 'return_raw_lst') else None,
            'cvar_lst': self.cvar_lst.copy() if hasattr(self, 'cvar_lst') else None,
            'cvar_raw_lst': self.cvar_raw_lst.copy() if hasattr(self, 'cvar_raw_lst') else None,
            'rl_reward_risk_lst': self.rl_reward_risk_lst.copy() if hasattr(self, 'rl_reward_risk_lst') else None,
            'rl_reward_profit_lst': self.rl_reward_profit_lst.copy() if hasattr(self, 'rl_reward_profit_lst') else None,
            'is_last_ctrl_solvable': self.is_last_ctrl_solvable if hasattr(self, 'is_last_ctrl_solvable') else False,
            'nan_stats': self.nan_stats.copy() if hasattr(self, 'nan_stats') else None,
            'latest_invest_profile': self.latest_invest_profile,
            'last_epoch_profile': self.last_epoch_profile
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"[ENV STATE] Saved environment state to {filepath}", flush=True)

    def restore_state(self, filepath):
        """
        Restore environment state from file for checkpoint resume.
        This allows resuming from mid-epoch checkpoints.
        """
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.epoch = state.get('epoch', 0)
        self.curTradeDay = state.get('curTradeDay', 0)
        self.cur_capital = state.get('cur_capital', self.initial_asset)
        self.last_valid_capital = self.cur_capital if np.isfinite(self.cur_capital) else self.initial_asset
        
        if state.get('curData') is not None:
            self.curData = state['curData'].copy()
        if state.get('lastDayData') is not None:
            self.lastDayData = state['lastDayData'].copy()
        if state.get('cur_slippage_drift') is not None:
            self.cur_slippage_drift = state['cur_slippage_drift'].copy()
        if state.get('last_slippage_drift') is not None:
            self.last_slippage_drift = state['last_slippage_drift'].copy()
        if state.get('state') is not None:
            self.state = state['state'].copy()
        if state.get('ctl_state') is not None:
            self.ctl_state = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state['ctl_state'].items()}
        if state.get('terminal') is not None:
            self.terminal = state['terminal']
        if state.get('profit_lst') is not None:
            self.profit_lst = state['profit_lst'].copy()
        if state.get('asset_lst') is not None:
            self.asset_lst = state['asset_lst'].copy()
        if state.get('actions_memory') is not None:
            self.actions_memory = [a.copy() if isinstance(a, np.ndarray) else a for a in state['actions_memory']]
        if state.get('date_memory') is not None:
            self.date_memory = state['date_memory'].copy()
        if state.get('reward_lst') is not None:
            self.reward_lst = state['reward_lst'].copy()
        if state.get('action_cbf_memeory') is not None:
            self.action_cbf_memeory = [a.copy() if isinstance(a, np.ndarray) else a for a in state['action_cbf_memeory']]
        if state.get('action_rl_memory') is not None:
            self.action_rl_memory = [a.copy() if isinstance(a, np.ndarray) else a for a in state['action_rl_memory']]
        if state.get('risk_adj_lst') is not None:
            self.risk_adj_lst = state['risk_adj_lst'].copy()
        if state.get('risk_raw_lst') is not None:
            self.risk_raw_lst = state['risk_raw_lst'].copy()
        if state.get('risk_cbf_lst') is not None:
            self.risk_cbf_lst = state['risk_cbf_lst'].copy()
        if state.get('return_raw_lst') is not None:
            self.return_raw_lst = state['return_raw_lst'].copy()
        if state.get('cvar_lst') is not None:
            self.cvar_lst = state['cvar_lst'].copy()
        if state.get('cvar_raw_lst') is not None:
            self.cvar_raw_lst = state['cvar_raw_lst'].copy()
        if state.get('rl_reward_risk_lst') is not None:
            self.rl_reward_risk_lst = state['rl_reward_risk_lst'].copy()
        if state.get('rl_reward_profit_lst') is not None:
            self.rl_reward_profit_lst = state['rl_reward_profit_lst'].copy()
        if state.get('is_last_ctrl_solvable') is not None:
            self.is_last_ctrl_solvable = state['is_last_ctrl_solvable']
        if state.get('nan_stats') is not None:
            self.nan_stats = state['nan_stats'].copy()
        self.latest_invest_profile = state.get('latest_invest_profile', None)
        self.last_epoch_profile = state.get('last_epoch_profile', self.latest_invest_profile)

        # Set flag to indicate we're resuming from checkpoint (affects save_profile logic)
        self._is_resuming = True

        print(f"[ENV STATE] Restored environment state from {filepath} | epoch={self.epoch}, day={self.curTradeDay}, capital={self.cur_capital:.2f}", flush=True)

    def get_results(self):
        # Create copies to avoid modifying original lists during ongoing episodes
        profit_lst = np.array(self.profit_lst)
        asset_lst = np.array(self.asset_lst)

        netProfit = self.cur_capital - self.initial_asset # Profits
        netProfit_pct = netProfit / self.initial_asset # Rate of overall returns

        diffPeriodAsset = np.diff(asset_lst)
        if len(diffPeriodAsset) > 0:
            sigReturn_max = np.max(diffPeriodAsset) # Maximal returns in a single transaction.
            sigReturn_min = np.min(diffPeriodAsset) # Minimal returns in a single transaction
        else:
            sigReturn_max = 0.0
            sigReturn_min = 0.0

        # Annual Returns
        annualReturn_pct = np.power((1 + netProfit_pct), (self.config.tradeDays_per_year/len(asset_lst))) - 1

        dailyReturn_pct_max = np.max(profit_lst)
        dailyReturn_pct_min = np.min(profit_lst)
        avg_dailyReturn_pct = np.mean(profit_lst)
        # strategy volatility
        volatility = np.sqrt(np.sum(np.power((profit_lst - avg_dailyReturn_pct), 2)) * self.config.tradeDays_per_year / (len(profit_lst) - 1))
        # Avoid division by zero
        if volatility == 0 or np.isnan(volatility) or np.isinf(volatility):
            volatility = 1e-6  # Small epsilon to avoid division by zero

        # SR_Vol, Long-term risk
        sharpeRatio = ((annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name])/ (volatility * 100)
        # Handle NaN/inf
        if np.isnan(sharpeRatio) or np.isinf(sharpeRatio):
            sharpeRatio = 0.0
        # sharpeRatio = np.max([sharpeRatio, 0])

        dailyAnnualReturn_lst = np.power((1+profit_lst), self.config.tradeDays_per_year) - 1
        dailyRisk_lst = np.array(self.risk_cbf_lst) * np.sqrt(self.config.tradeDays_per_year) # Daily Risk to Anuual Risk
        # Avoid division by zero
        dailyRisk_lst_safe = np.where(dailyRisk_lst == 0, 1e-6, dailyRisk_lst)
        dailySR = ((dailyAnnualReturn_lst[1:] * 100) - self.config.mkt_rf[self.config.market_name]) / (dailyRisk_lst_safe[1:] * 100)
        # Handle NaN/inf
        dailySR = np.where(np.isnan(dailySR) | np.isinf(dailySR), 0.0, dailySR)
        dailySR = np.append(0, dailySR)
        # dailySR = np.where(dailySR < 0, 0, dailySR)
        dailySR_max = np.max(dailySR)
        dailySR_nonzero = dailySR[dailySR!=0]
        dailySR_min = np.min(dailySR_nonzero) if len(dailySR_nonzero) > 0 else 0.0
        dailySR_avg = np.mean(dailySR)

        # For performance analysis
        # Avoid division by zero
        return_raw_array = np.array(self.return_raw_lst)
        return_raw_safe = np.where(return_raw_array == 0, 1e-6, return_raw_array)
        dailyReturnRate_wocbf = np.diff(return_raw_array) / return_raw_safe[:-1]
        dailyReturnRate_wocbf = np.append(0, dailyReturnRate_wocbf)
        # Handle NaN/inf
        dailyReturnRate_wocbf = np.where(np.isnan(dailyReturnRate_wocbf) | np.isinf(dailyReturnRate_wocbf), 0.0, dailyReturnRate_wocbf)
        dailyAnnualReturn_wocbf_lst = np.power((1+dailyReturnRate_wocbf), self.config.tradeDays_per_year) - 1
        dailyRisk_wocbf_lst = np.array(self.risk_raw_lst) * np.sqrt(self.config.tradeDays_per_year)
        # Avoid division by zero
        dailyRisk_wocbf_lst_safe = np.where(dailyRisk_wocbf_lst == 0, 1e-6, dailyRisk_wocbf_lst)
        dailySR_wocbf = ((dailyAnnualReturn_wocbf_lst[1:] * 100) - self.config.mkt_rf[self.config.market_name]) / (dailyRisk_wocbf_lst_safe[1:] * 100)
        # Handle NaN/inf
        dailySR_wocbf = np.where(np.isnan(dailySR_wocbf) | np.isinf(dailySR_wocbf), 0.0, dailySR_wocbf)
        dailySR_wocbf = np.append(0, dailySR_wocbf)
        # dailySR_wocbf = np.where(dailySR_wocbf < 0, 0, dailySR_wocbf)
        dailySR_wocbf_max = np.max(dailySR_wocbf)
        dailySR_wocbf_nonzero = dailySR_wocbf[dailySR_wocbf!=0]
        dailySR_wocbf_min = np.min(dailySR_wocbf_nonzero) if len(dailySR_wocbf_nonzero) > 0 else 0.0
        dailySR_wocbf_avg = np.mean(dailySR_wocbf)

        annualReturn_wocbf_pct = np.power((1 + ((self.return_raw_lst[-1] - self.initial_asset) / self.initial_asset)), (self.config.tradeDays_per_year/len(self.return_raw_lst))) - 1
        volatility_wocbf = np.sqrt((np.sum(np.power((dailyReturnRate_wocbf - np.mean(dailyReturnRate_wocbf)), 2)) * self.config.tradeDays_per_year / (len(self.return_raw_lst) - 1)))
        # Avoid division by zero
        if volatility_wocbf == 0 or np.isnan(volatility_wocbf) or np.isinf(volatility_wocbf):
            volatility_wocbf = 1e-6
        sharpeRatio_woCBF = ((annualReturn_wocbf_pct * 100) - self.config.mkt_rf[self.config.market_name])/ (volatility_wocbf * 100)
        # Handle NaN/inf
        if np.isnan(sharpeRatio_woCBF) or np.isinf(sharpeRatio_woCBF):
            sharpeRatio_woCBF = 0.0
        sharpeRatio_woCBF = np.max([sharpeRatio_woCBF, 0])

        winRate = len(np.argwhere(diffPeriodAsset>0))/(len(diffPeriodAsset) + 1)

        # MDD
        repeat_asset_lst = np.tile(asset_lst, (len(asset_lst), 1))
        mdd_mtix = np.triu(1 - repeat_asset_lst / np.reshape(asset_lst, (-1, 1)), k=1)
        mddmaxidx = np.argmax(mdd_mtix)
        mdd_highidx = mddmaxidx // len(asset_lst)
        mdd_lowidx = mddmaxidx % len(asset_lst)
        self.mdd = np.max(mdd_mtix)
        self.mdd_high = asset_lst[mdd_highidx]
        self.mdd_low = asset_lst[mdd_lowidx]
        self.mdd_highTimepoint = self.date_memory[mdd_highidx]
        self.mdd_lowTimepoint = self.date_memory[mdd_lowidx]

        # Strategy volatility during trading
        cumsum_r = np.cumsum(profit_lst)/np.arange(1, self.totalTradeDay+1) # average cumulative returns rate
        repeat_profit_lst = np.tile(profit_lst, (len(profit_lst), 1))
        stg_vol_lst = np.sqrt(np.sum(np.power(np.tril(repeat_profit_lst - np.reshape(cumsum_r, (-1,1)), k=0), 2), axis=1)[1:] / np.arange(1, len(repeat_profit_lst)) * self.config.tradeDays_per_year)
        stg_vol_lst = np.append([0], stg_vol_lst, axis=0)
        # stg_vol_lst  = np.sqrt((np.cumsum(np.power((self.profit_lst - cumsum_r), 2))/np.arange(1, self.totalTradeDay+1)) * self.config.tradeDays_per_year)

        vol_max = np.max(stg_vol_lst)
        stg_vol_array = np.array(stg_vol_lst)
        stg_vol_nonzero = stg_vol_array[stg_vol_array!=0]
        vol_min = np.min(stg_vol_nonzero) if len(stg_vol_nonzero) > 0 else 0.0
        vol_avg = np.mean(stg_vol_lst)

        # short-term risk
        risk_max = np.max(self.risk_cbf_lst)
        risk_cbf_array = np.array(self.risk_cbf_lst)
        risk_cbf_nonzero = risk_cbf_array[risk_cbf_array!=0]
        risk_min = np.min(risk_cbf_nonzero) if len(risk_cbf_nonzero) > 0 else 0.0
        risk_avg = np.mean(self.risk_cbf_lst)

        risk_raw_max = np.max(self.risk_raw_lst)
        risk_raw_array = np.array(self.risk_raw_lst)
        risk_raw_nonzero = risk_raw_array[risk_raw_array!=0]
        risk_raw_min = np.min(risk_raw_nonzero) if len(risk_raw_nonzero) > 0 else 0.0
        risk_raw_avg = np.mean(self.risk_raw_lst)

        # Downside risk at volatility
        risk_downsideAtVol_daily = np.sqrt(np.sum(np.power(np.tril((repeat_profit_lst - np.reshape(cumsum_r, (-1,1))) * (repeat_profit_lst<np.reshape(cumsum_r, (-1,1))), k=0), 2), axis=1)[1:] / np.arange(1, len(repeat_profit_lst)) * self.config.tradeDays_per_year)
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
        cvar_nonzero = cvar_array[cvar_array!=0]
        cvar_min = np.min(cvar_nonzero) if len(cvar_nonzero) > 0 else 0.0
        cvar_avg = np.mean(self.cvar_lst)

        cvar_raw_max = np.max(self.cvar_raw_lst)
        cvar_raw_array = np.array(self.cvar_raw_lst)
        cvar_raw_nonzero = cvar_raw_array[cvar_raw_array!=0]
        cvar_raw_min = np.min(cvar_raw_nonzero) if len(cvar_raw_nonzero) > 0 else 0.0
        cvar_raw_avg = np.mean(self.cvar_raw_lst)

        # Calmar ratio
        time_T = len(profit_lst)
        avg_return = netProfit_pct / time_T if time_T > 0 else 0.0
        variance_r = np.sum(np.power((profit_lst - avg_dailyReturn_pct), 2)) / (len(profit_lst) - 1) if len(profit_lst) > 1 else 0.0
        volatility_daily = np.sqrt(variance_r) if variance_r >= 0 else 0.0
        # Avoid division by zero
        if volatility_daily == 0:
            volatility_daily = 1e-6
 
        if netProfit_pct > 0:
            shrp = avg_return / volatility_daily if volatility_daily > 0 else 0.0
            if shrp > 0 and not (np.isnan(shrp) or np.isinf(shrp)):
                log_shrp = np.log(shrp)
                if not (np.isnan(log_shrp) or np.isinf(log_shrp)):
                    calmarRatio = (time_T * np.power(shrp, 2)) / (0.63519 + 0.5 * np.log(time_T) + log_shrp)
                else:
                    calmarRatio = 0.0
            else:
                calmarRatio = 0.0
        elif netProfit_pct == 0:
            calmarRatio = (netProfit_pct) / (1.2533 * volatility_daily * np.sqrt(time_T)) if volatility_daily > 0 and time_T > 0 else 0.0
        else:
            # netProfit_pct < 0
            if avg_return != 0 and not (np.isnan(avg_return) or np.isinf(avg_return)):
                calmarRatio = (netProfit_pct) / (-(avg_return * time_T) - (variance_r / avg_return))
            else:
                calmarRatio = 0.0
        
        # Final validation
        if np.isnan(calmarRatio) or np.isinf(calmarRatio):
            calmarRatio = 0.0

        # Sterling ratio
        move_mdd_mask = np.where(profit_lst<0, 1, 0)
        moving_mdd = np.sqrt(np.sum(np.power(profit_lst * move_mdd_mask, 2))  * self.config.tradeDays_per_year / (len(profit_lst) - 1)) if len(profit_lst) > 1 else 0.0
        # Avoid division by zero
        if moving_mdd == 0 or np.isnan(moving_mdd) or np.isinf(moving_mdd):
            moving_mdd = 1e-6
        sterlingRatio =  ((annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name]) / (moving_mdd * 100)
        # Handle NaN/inf
        if np.isnan(sterlingRatio) or np.isinf(sterlingRatio):
            sterlingRatio = 0.0

        if self.mode == 'train':
            cputime_use = self.end_cputime - self.start_cputime - self.exclusive_cputime
            systime_use = self.end_systime - self.start_systime - self.exclusive_systime
        else:
            cputime_use = self.end_cputime - self.start_cputime
            systime_use = self.end_systime - self.start_systime

        if np.shape(np.array(self.actions_memory)) != (self.totalTradeDay, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('actions_memory shape error in the RLcontroller mode')
            else:
                self.actions_memory = np.ones((self.totalTradeDay, self.stock_num)) * (1/self.stock_num) * self.bound_flag
        if np.shape(np.array(self.action_rl_memory)) != (self.totalTradeDay+1, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('action_rl_memory shape error in the RLcontroller mode')
            else:
                self.action_rl_memory = np.ones((self.totalTradeDay+1, self.stock_num)) * (1/self.stock_num) * self.bound_flag
        if np.shape(np.array(self.action_cbf_memeory)) != (self.totalTradeDay+1, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('action_cbf_memeory shape error in the RLcontroller mode')
            else:
                self.action_cbf_memeory = np.zeros((self.totalTradeDay+1, self.stock_num))
        if len(self.solvable_flag) == 0:
            self.solvable_flag = np.zeros(len(asset_lst))
        if len(self.risk_pred_lst) == 0:
            self.risk_pred_lst = np.zeros(len(asset_lst))
  
        cbf_abssum_contribution = np.sum(np.abs(self.action_cbf_memeory[:-1]))

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

        final_action_abs = _align_array(np.sum(np.abs(np.array(self.actions_memory)), axis=1))
        rl_action_abs = _align_array(np.sum(np.abs(np.array(self.action_rl_memory)), axis=1))
        cbf_action_abs = _align_array(np.sum(np.abs(np.array(self.action_cbf_memeory)), axis=1))

        info_dict = {
            'ep': self.epoch, 'trading_days': self.totalTradeDay, 'annualReturn_pct': annualReturn_pct, 'volatility': volatility, 'sharpeRatio': sharpeRatio, 'sharpeRatio_wocbf': sharpeRatio_woCBF,
            'mdd': self.mdd, 'calmarRatio': calmarRatio, 'sterlingRatio': sterlingRatio, 'netProfit': netProfit, 'netProfit_pct': netProfit_pct, 'winRate': winRate,
            'vol_max': vol_max, 'vol_min': vol_min, 'vol_avg': vol_avg,
            'risk_max': risk_max, 'risk_min': risk_min, 'risk_avg': risk_avg,
            'riskRaw_max': risk_raw_max, 'riskRaw_min': risk_raw_min, 'riskRaw_avg': risk_raw_avg,
            'dailySR_max': dailySR_max, 'dailySR_min': dailySR_min, 'dailySR_avg': dailySR_avg, 'dailySR_wocbf_max': dailySR_wocbf_max, 'dailySR_wocbf_min': dailySR_wocbf_min, 'dailySR_wocbf_avg': dailySR_wocbf_avg,
            'dailyReturn_pct_max': dailyReturn_pct_max, 'dailyReturn_pct_min': dailyReturn_pct_min, 'dailyReturn_pct_avg': avg_dailyReturn_pct,
            'sigReturn_max': sigReturn_max, 'sigReturn_min': sigReturn_min, 
            'mdd_high': self.mdd_high, 'mdd_low': self.mdd_low, 'mdd_high_date': self.mdd_highTimepoint, 'mdd_low_date': self.mdd_lowTimepoint, 
            'final_capital': self.cur_capital, 'reward_sum': np.nansum(self.reward_lst) if len(self.reward_lst) > 0 else 0.0,
            'final_capital_wocbf': self.return_raw_lst[-1], 
            'cbf_contribution': cbf_abssum_contribution,
            'risk_downsideAtVol': risk_downsideAtVol, 'risk_downsideAtVol_daily_max': risk_downsideAtVol_daily_max, 'risk_downsideAtVol_daily_min': risk_downsideAtVol_daily_min, 'risk_downsideAtVol_daily_avg': risk_downsideAtVol_daily_avg,
            'risk_downsideAtValue_daily_max': risk_downsideAtValue_daily_max, 'risk_downsideAtValue_daily_min': risk_downsideAtValue_daily_min, 'risk_downsideAtValue_daily_avg': risk_downsideAtValue_daily_avg,
            'cvar_max': cvar_max, 'cvar_min': cvar_min, 'cvar_avg': cvar_avg, 'cvar_raw_max': cvar_raw_max, 'cvar_raw_min': cvar_raw_min, 'cvar_raw_avg': cvar_raw_avg,
            'solver_solvable': self.solver_stat['solvable'], 'solver_insolvable': self.solver_stat['insolvable'], 'cputime': cputime_use, 'systime': systime_use,
            'asset_lst': asset_lst.copy(),
            'daily_return_lst': profit_lst.copy(), 
            'reward_lst': np.array(self.reward_lst).copy() if isinstance(self.reward_lst, list) else self.reward_lst.copy(), 
            'stg_vol_lst': stg_vol_lst.copy() if isinstance(stg_vol_lst, np.ndarray) else np.array(stg_vol_lst).copy(), 
            'risk_lst': np.array(self.risk_cbf_lst).copy() if isinstance(self.risk_cbf_lst, list) else self.risk_cbf_lst.copy(), 
            'risk_wocbf_lst': np.array(self.risk_raw_lst).copy() if isinstance(self.risk_raw_lst, list) else self.risk_raw_lst.copy(),
            'capital_wocbf_lst': np.array(self.return_raw_lst).copy() if isinstance(self.return_raw_lst, list) else self.return_raw_lst.copy(), 
            'daily_sr_lst': dailySR.copy() if isinstance(dailySR, np.ndarray) else np.array(dailySR).copy(), 
            'daily_sr_wocbf_lst': dailySR_wocbf.copy() if isinstance(dailySR_wocbf, np.ndarray) else np.array(dailySR_wocbf).copy(),
            'risk_adj_lst': np.array(self.risk_adj_lst).copy() if isinstance(self.risk_adj_lst, list) else self.risk_adj_lst.copy(), 
            'ctrl_weight_lst': np.array(self.ctrl_weight_lst).copy() if isinstance(self.ctrl_weight_lst, list) else self.ctrl_weight_lst.copy(), 
            'solvable_flag': self.solvable_flag.copy() if isinstance(self.solvable_flag, np.ndarray) else np.array(self.solvable_flag).copy(), 
            'risk_pred_lst': np.array(self.risk_pred_lst).copy() if isinstance(self.risk_pred_lst, list) else self.risk_pred_lst.copy(),
            'final_action_abssum_lst': final_action_abs.copy(),
            'rl_action_abssum_lst': rl_action_abs.copy(),
            'cbf_action_abssum_lst': cbf_action_abs.copy(), 
            'daily_downsideAtVol_risk_lst': risk_downsideAtVol_daily.copy() if isinstance(risk_downsideAtVol_daily, np.ndarray) else np.array(risk_downsideAtVol_daily).copy(), 
            'daily_downsideAtValue_risk_lst': risk_downsideAtValue_daily.copy() if isinstance(risk_downsideAtValue_daily, np.ndarray) else np.array(risk_downsideAtValue_daily).copy(),
            'cvar_lst': np.array(self.cvar_lst).copy() if isinstance(self.cvar_lst, list) else self.cvar_lst.copy(), 
            'cvar_raw_lst': np.array(self.cvar_raw_lst).copy() if isinstance(self.cvar_raw_lst, list) else self.cvar_raw_lst.copy(),
        }

        return info_dict

    def save_profile(self, invest_profile):
        # basic data
        missing_fields = []
        for fname in self.profile_hist_field_lst:
            if fname in list(invest_profile.keys()):
                self.profile_hist_ep[fname].append(invest_profile[fname])
            else:
                missing_fields.append(fname)
                # Use None as placeholder for missing fields instead of raising error
                self.profile_hist_ep[fname].append(None)
                print(f"Warning: Field '{fname}' not found in invest_profile, using None as placeholder", flush=True)
        
        if missing_fields:
            print(f"Warning: Missing {len(missing_fields)} fields in invest_profile: {missing_fields}", flush=True)
            print(f"Available fields in invest_profile: {list(invest_profile.keys())}", flush=True)
        
        try:
            phist_df = pd.DataFrame(self.profile_hist_ep, columns=self.profile_hist_field_lst)
            profile_path = os.path.join(self.config.res_dir, '{}_profile.csv'.format(self.mode))
            phist_df.to_csv(profile_path, index=False)
            print(f"Profile saved to: {profile_path} (epoch {self.epoch}, {len(phist_df)} rows)", flush=True)
        except Exception as e:
            print(f"Error saving profile: {e}", flush=True)
            print(f"profile_hist_ep keys: {list(self.profile_hist_ep.keys())}", flush=True)
            print(f"profile_hist_field_lst: {self.profile_hist_field_lst}", flush=True)
            raise

        cputime_avg = np.mean(phist_df['cputime'])
        systime_avg = np.mean(phist_df['systime'])

        bestmodel_dict = {}
        if self.config.trained_best_model_type == 'max_capital':
            field_name = 'final_capital'
            # Filter out NaN values before finding max
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.max(valid_values)
            else:
                v = np.nan
        elif 'loss' in self.config.trained_best_model_type:
            field_name = 'reward_sum'
            # Filter out NaN values before finding max
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.max(valid_values)
            else:
                v = np.nan
        elif self.config.trained_best_model_type == 'sharpeRatio':
            field_name = 'sharpeRatio'
            # Filter out NaN values before finding max
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.max(valid_values)
            else:
                v = np.nan
        elif self.config.trained_best_model_type == 'volatility':
            field_name = 'volatility'
            # Filter out NaN values before finding min
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.min(valid_values)
            else:
                v = np.nan
        elif self.config.trained_best_model_type == 'mdd':
            field_name = 'mdd'
            # Filter out NaN values before finding min
            valid_values = phist_df[field_name].dropna()
            if len(valid_values) > 0:
                v = np.min(valid_values)
            else:
                v = np.nan
        else:
            raise ValueError('Unknown implementation with the best model type [{}]..'.format(self.config.trained_best_model_type))
        
        # Handle NaN case: use current epoch value as fallback
        if np.isnan(v) or v is None:
            # Use current epoch's value as fallback
            current_value = invest_profile.get(field_name, None)
            if current_value is not None and not (isinstance(current_value, float) and np.isnan(current_value)):
                v = current_value
                v_ep = self.epoch
                print(f"Warning: All values in {field_name} are NaN, using current epoch {v_ep} value {v} as fallback", flush=True)
            else:
                # Last resort: use current epoch number, value remains NaN (will be handled later)
                v_ep = self.epoch
                print(f"Warning: All values in {field_name} are NaN and current epoch value is also NaN, using epoch {v_ep} as fallback", flush=True)
        else:
            # Find epoch with the best value, handle floating point comparison and empty results
            filtered_df = phist_df[phist_df[field_name] == v]
            if len(filtered_df) == 0:
                # Fallback: use approximate comparison for floating point values
                if field_name in ['final_capital', 'reward_sum', 'sharpeRatio', 'volatility', 'mdd']:
                    # Use np.isclose for floating point comparison
                    tolerance = 1e-6
                    filtered_df = phist_df[np.abs(phist_df[field_name] - v) < tolerance]
            
            if len(filtered_df) == 0:
                # If still empty, use current epoch as fallback
                v_ep = self.epoch
                print(f"Warning: Could not find epoch with {field_name}={v}, using current epoch {v_ep} as fallback", flush=True)
            else:
                v_ep = filtered_df['ep'].iloc[0]
        
        # Ensure v is not NaN before saving (use current epoch value if still NaN)
        if np.isnan(v) or v is None:
            # Try to get value from current epoch
            current_value = invest_profile.get(field_name, None)
            if current_value is not None and not (isinstance(current_value, float) and np.isnan(current_value)):
                v = current_value
            else:
                # Use 0 as absolute fallback
                v = 0.0
                print(f"Warning: Using 0.0 as absolute fallback for {field_name}", flush=True)
        
        bestmodel_dict['{}_ep'.format(self.config.trained_best_model_type)] = v_ep
        bestmodel_dict[self.config.trained_best_model_type] = v
        
        # Ensure all values in bestmodel_dict are valid (not NaN) before saving
        for key, value in bestmodel_dict.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                # Replace NaN with appropriate default
                if 'ep' in key:
                    bestmodel_dict[key] = self.epoch  # Use current epoch
                else:
                    # For value fields, try to get from current epoch
                    field_name = key
                    current_value = invest_profile.get(field_name, 0.0)
                    if current_value is not None and not (isinstance(current_value, float) and np.isnan(current_value)):
                        bestmodel_dict[key] = current_value
                    else:
                        bestmodel_dict[key] = 0.0  # Absolute fallback
                print(f"Warning: Replaced NaN in bestmodel_dict['{key}'] with {bestmodel_dict[key]}", flush=True)
        
        if True:
            print("-"*30)
            # Use final_capital from invest_profile instead of self.cur_capital to avoid nan
            current_capital = invest_profile.get('final_capital', self.cur_capital)
            # Handle nan values
            if current_capital is None or (isinstance(current_capital, float) and np.isnan(current_capital)):
                current_capital = self.initial_asset  # Fallback to initial asset
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = current_capital  # Fallback to current capital
            
            # Map trained_best_model_type to display name for log message
            field_display_name = {
                'max_capital': 'capital',
                'js_loss': 'reward_sum',
                'sharpeRatio': 'sharpeRatio',
                'volatility': 'volatility',
                'mdd': 'mdd'
            }.get(self.config.trained_best_model_type, self.config.trained_best_model_type)
            
            # Print NaN statistics summary
            total_nan_events = (self.nan_stats['cur_close_nan'] + self.nan_stats['last_close_nan'] + 
                              self.nan_stats['poDayReturn_nan'] + self.nan_stats['updatePoValue_nan'])
            if total_nan_events > 0:
                print(f"\n[NaN Statistics] Epoch {self.epoch} - Total NaN events: {total_nan_events}", flush=True)
                print(f"  - cur_close_price NaN/inf/<=0: {self.nan_stats['cur_close_nan']} times", flush=True)
                print(f"  - last_close_price NaN/inf/<=0: {self.nan_stats['last_close_nan']} times", flush=True)
                print(f"  - poDayReturn NaN/inf: {self.nan_stats['poDayReturn_nan']} times", flush=True)
                print(f"  - updatePoValue NaN/inf/<=0: {self.nan_stats['updatePoValue_nan']} times", flush=True)
                
                # Print sample details (first few occurrences)
                if len(self.nan_stats['cur_close_details']) > 0:
                    print(f"  Sample cur_close NaN events (showing first {min(3, len(self.nan_stats['cur_close_details']))}):", flush=True)
                    for detail in self.nan_stats['cur_close_details'][:3]:
                        print(f"    Day {detail['day']} ({detail['date']}): stock={detail['stock']}, value={detail['value']}, fallback={detail['fallback']}", flush=True)
                if len(self.nan_stats['poDayReturn_details']) > 0:
                    print(f"  Sample poDayReturn NaN events (showing first {min(3, len(self.nan_stats['poDayReturn_details']))}):", flush=True)
                    for detail in self.nan_stats['poDayReturn_details'][:3]:
                        print(f"    Day {detail['day']} ({detail['date']}): value={detail['value']}", flush=True)
                if len(self.nan_stats['updatePoValue_details']) > 0:
                    print(f"  Sample updatePoValue NaN events (showing first {min(3, len(self.nan_stats['updatePoValue_details']))}):", flush=True)
                    for detail in self.nan_stats['updatePoValue_details'][:3]:
                        print(f"    Day {detail['day']} ({detail['date']}): value={detail['value']}, prev_capital={detail['prev_capital']:.2f}, poDayReturn={detail['poDayReturn']}", flush=True)
                print("", flush=True)
            
            log_str = "Mode: {}, Ep: {}, Current epoch capital: {:.2f}, historical best {} ({} ep): {:.2f} | solvable: {}, insolvable: {} | step count: {} | cputime cur: {} s, avg: {} s, system time cur: {} s/ep, avg: {} s/ep..".format(
                self.mode, self.epoch, current_capital, field_display_name, v_ep, v, 
                np.array(phist_df['solver_solvable'])[-1], 
                np.array(phist_df['solver_insolvable'])[-1], 
                self.stepcount, 
                np.round(np.array(phist_df['cputime'])[-1], 2), 
                np.round(cputime_avg, 2), 
                np.round(np.array(phist_df['systime'])[-1], 2), 
                np.round(systime_avg, 2)
            )
            print(log_str)
            # Print top 10 stocks with weights for current epoch
            if hasattr(self, 'actions_memory') and len(self.actions_memory) > 0 and hasattr(self, 'stock_lst'):
                sample_idx = min(4, len(self.actions_memory) - 1)
                if isinstance(self.actions_memory, list):
                    sample_weights = self.actions_memory[sample_idx]
                else:
                    sample_weights = self.actions_memory[sample_idx]
                if isinstance(sample_weights, np.ndarray) and len(sample_weights) == len(self.stock_lst):
                    # Use weight_symbol_mapper if available
                    if get_top_stocks is not None:
                        try:
                            top_10_stocks = get_top_stocks(sample_weights, self.stock_lst, top_k=10)
                            print("  Top 10 stocks (day {}):".format(sample_idx))
                            for i, (symbol, weight) in enumerate(top_10_stocks, 1):
                                print("    {}. {}: {:.4f} ({:.2f}%)".format(i, symbol, weight, weight * 100))
                        except Exception as e:
                            # Fallback to simple printing if mapping fails
                            print("  Top 10 stocks (day {}): (mapping error: {})".format(sample_idx, str(e)))
                    else:
                        # Fallback: print top 10 by weight manually
                        weight_stock_pairs = list(zip(sample_weights, self.stock_lst))
                        weight_stock_pairs.sort(key=lambda x: x[0], reverse=True)
                        print("  Top 10 stocks (day {}):".format(sample_idx))
                        for i, (weight, symbol) in enumerate(weight_stock_pairs[:10], 1):
                            print("    {}. {}: {:.4f} ({:.2f}%)".format(i, symbol, weight, weight * 100))
        # Create DataFrame and ensure no NaN values (we already handled NaN above, but double-check)
        bestmodel_df = pd.DataFrame([bestmodel_dict])
        # Final safety check: replace any remaining NaN
        for col in bestmodel_df.columns:
            if bestmodel_df[col].isna().any():
                if 'ep' in col:
                    bestmodel_df[col] = bestmodel_df[col].fillna(self.epoch)
                else:
                    # Try to get from current invest_profile
                    field_name = col
                    current_value = invest_profile.get(field_name, 0.0)
                    if current_value is not None and not (isinstance(current_value, float) and np.isnan(current_value)):
                        bestmodel_df[col] = bestmodel_df[col].fillna(current_value)
                    else:
                        bestmodel_df[col] = bestmodel_df[col].fillna(0.0)
        
        bestmodel_path = os.path.join(self.config.res_dir, '{}_bestmodel.csv'.format(self.mode))
        bestmodel_df.to_csv(bestmodel_path, index=False)

        # save data of each step in 1st/best/last model
        fpath = os.path.join(self.config.res_dir, '{}_stepdata.csv'.format(self.mode))
        target_len = len(invest_profile['asset_lst']) if 'asset_lst' in invest_profile else len(self.asset_lst)

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
            is_resuming = hasattr(self, '_is_resuming') and self._is_resuming
            if is_resuming:
                print(f"[SAVE_PROFILE] Skipping stepdata.csv creation during resume (arrays may have inconsistent lengths)")
                # Create empty DataFrame with correct columns for future appends
                step_data = pd.DataFrame(columns=[
                    'date', 'capital_policy_1', 'dailyReturn_policy_1', 'reward_policy_1',
                    'strategyVolatility_policy_1', 'risk_policy_1', 'risk_wocbf_policy_1', 'capital_wocbf_policy_1',
                    'dailySR_policy_1', 'dailySR_wocbf_policy_1', 'riskAccepted_policy_1', 'ctrlWeight_policy_1',
                    'solvable_flag_policy_1', 'risk_pred_policy_1', 'final_action_abssum_policy_1',
                    'rl_action_abssum_policy_1', 'cbf_action_abssum_policy_1', 'downsideAtVol_risk_policy_1',
                    'downsideAtValue_risk_policy_1', 'cvar_policy_1', 'cvar_raw_policy_1'
                ])
                step_data.to_csv(fpath, index=False)
            else:
                # Ensure all arrays have the same length for fresh training
                array_length = len(invest_profile['asset_lst'])
                print(f"[SAVE_PROFILE] Creating new stepdata.csv with {array_length} rows")

                step_data = {
                    'date': self.date_memory[:array_length] if len(self.date_memory) >= array_length else self.date_memory + [''] * (array_length - len(self.date_memory)),
                    'capital_policy_1': _align_profile_array(invest_profile['asset_lst']).tolist(),
                    'dailyReturn_policy_1': _align_profile_array(invest_profile['daily_return_lst']).tolist(),
                    'reward_policy_1': _align_profile_array(invest_profile['reward_lst']).tolist(),
                    'strategyVolatility_policy_1': _align_profile_array(invest_profile['stg_vol_lst']).tolist(),
                    'risk_policy_1': _align_profile_array(invest_profile['risk_lst']).tolist(),
                    'risk_wocbf_policy_1': _align_profile_array(invest_profile['risk_wocbf_lst']).tolist(),
                    'capital_wocbf_policy_1': _align_profile_array(invest_profile['capital_wocbf_lst']).tolist(),
                    'dailySR_policy_1': _align_profile_array(invest_profile['daily_sr_lst']).tolist(),
                    'dailySR_wocbf_policy_1': _align_profile_array(invest_profile['daily_sr_wocbf_lst']).tolist(),
                    'riskAccepted_policy_1': _align_profile_array(invest_profile['risk_adj_lst']).tolist(),
                    'ctrlWeight_policy_1': _align_profile_array(invest_profile['ctrl_weight_lst']).tolist(),
                    'solvable_flag_policy_1': _align_profile_array(invest_profile['solvable_flag']).tolist(),
                    'risk_pred_policy_1': _align_profile_array(invest_profile['risk_pred_lst']).tolist(),
                    'final_action_abssum_policy_1': _align_profile_array(invest_profile['final_action_abssum_lst']).tolist(),
                    'rl_action_abssum_policy_1': _align_profile_array(invest_profile['rl_action_abssum_lst']).tolist(),
                    'cbf_action_abssum_policy_1': _align_profile_array(invest_profile['cbf_action_abssum_lst']).tolist(),
                    'downsideAtVol_risk_policy_1': _align_profile_array(invest_profile['daily_downsideAtVol_risk_lst']).tolist(),
                    'downsideAtValue_risk_policy_1': _align_profile_array(invest_profile['daily_downsideAtValue_risk_lst']).tolist(),
                    'cvar_policy_1': _align_profile_array(invest_profile['cvar_lst']).tolist(),
                    'cvar_raw_policy_1': _align_profile_array(invest_profile['cvar_raw_lst']).tolist(),
                }

                # Verify all arrays have same length
                lengths = [len(v) for v in step_data.values()]
                if len(set(lengths)) != 1:
                    print(f"[SAVE_PROFILE] ERROR: Arrays have different lengths: {dict(zip(step_data.keys(), lengths))}")
                    raise ValueError("Cannot create stepdata.csv: arrays have inconsistent lengths")

                step_data = pd.DataFrame(step_data)
        else:
            step_data = pd.DataFrame(pd.read_csv(fpath, header=0))

        # Only update best model columns if this is the best model and arrays have consistent length
        if bestmodel_dict['{}_ep'.format(self.config.trained_best_model_type)] == invest_profile['ep']:
            # Check if step_data is empty (resume case) or has matching length
            if len(step_data) == 0:
                print(f"[SAVE_PROFILE] Skipping best model update for resume case (empty step_data)")
            elif len(step_data) == len(invest_profile['asset_lst']):
                step_data['capital_policy_best'] = _align_profile_array(invest_profile['asset_lst']).tolist()
                step_data['dailyReturn_policy_best'] = _align_profile_array(invest_profile['daily_return_lst']).tolist()
                step_data['reward_policy_best'] = _align_profile_array(invest_profile['reward_lst']).tolist()
                step_data['strategyVolatility_policy_best'] = _align_profile_array(invest_profile['stg_vol_lst']).tolist()
                step_data['risk_policy_best'] = _align_profile_array(invest_profile['risk_lst']).tolist()
                step_data['risk_wocbf_policy_best'] = _align_profile_array(invest_profile['risk_wocbf_lst']).tolist()
                step_data['capital_wocbf_policy_best'] = _align_profile_array(invest_profile['capital_wocbf_lst']).tolist()
                step_data['dailySR_policy_best'] = _align_profile_array(invest_profile['daily_sr_lst']).tolist()
                step_data['dailySR_wocbf_policy_best'] = _align_profile_array(invest_profile['daily_sr_wocbf_lst']).tolist()
                step_data['riskAccepted_policy_best'] = _align_profile_array(invest_profile['risk_adj_lst']).tolist()
                step_data['ctrlWeight_policy_best'] = _align_profile_array(invest_profile['ctrl_weight_lst']).tolist()
                step_data['solvable_flag_policy_best'] = _align_profile_array(invest_profile['solvable_flag']).tolist()
                step_data['risk_pred_policy_best'] = _align_profile_array(invest_profile['risk_pred_lst']).tolist()
                step_data['final_action_abssum_policy_best'] = _align_profile_array(invest_profile['final_action_abssum_lst']).tolist()
                step_data['rl_action_abssum_policy_best'] = _align_profile_array(invest_profile['rl_action_abssum_lst']).tolist()
                step_data['cbf_action_abssum_policy_best'] = _align_profile_array(invest_profile['cbf_action_abssum_lst']).tolist()
                step_data['downsideAtVol_risk_policy_best'] = _align_profile_array(invest_profile['daily_downsideAtVol_risk_lst']).tolist()
                step_data['downsideAtValue_risk_policy_best'] = _align_profile_array(invest_profile['daily_downsideAtValue_risk_lst']).tolist()
                step_data['cvar_policy_best'] = _align_profile_array(invest_profile['cvar_lst']).tolist()
                step_data['cvar_raw_policy_best'] = _align_profile_array(invest_profile['cvar_raw_lst']).tolist()
            else:
                print(f"[SAVE_PROFILE] Skipping best model update: step_data length ({len(step_data)}) != invest_profile arrays length ({len(invest_profile['asset_lst'])})")
        # Record the test set performance on valid_best_policy
        if self.mode == 'test':
            valid_fpath = os.path.join(self.config.res_dir, 'valid_bestmodel.csv')
            if os.path.exists(valid_fpath):
                valid_records = pd.DataFrame(pd.read_csv(valid_fpath, header=0))
                if int(valid_records['{}_ep'.format(self.config.trained_best_model_type)][0]) == invest_profile['ep']:
                    step_data['capital_policy_validbest'] = invest_profile['asset_lst']
                    step_data['dailyReturn_policy_validbest'] = invest_profile['daily_return_lst'] # Plot the scatter plot of efficient frontier.
                    step_data['reward_policy_validbest'] = invest_profile['reward_lst']
                    step_data['strategyVolatility_policy_validbest'] = invest_profile['stg_vol_lst']
                    step_data['risk_policy_validbest'] = invest_profile['risk_lst'] # Plot the scatter plot of efficient frontier.
                    step_data['risk_wocbf_policy_validbest'] = invest_profile['risk_wocbf_lst']
                    step_data['capital_wocbf_policy_validbest'] = invest_profile['capital_wocbf_lst']
                    step_data['dailySR_policy_validbest'] = invest_profile['daily_sr_lst']
                    step_data['dailySR_wocbf_policy_validbest'] = invest_profile['daily_sr_wocbf_lst']
                    step_data['riskAccepted_policy_validbest'] = invest_profile['risk_adj_lst']
                    step_data['ctrlWeight_policy_validbest'] = invest_profile['ctrl_weight_lst']
                    step_data['solvable_flag_policy_validbest'] = invest_profile['solvable_flag']
                    step_data['risk_pred_policy_validbest'] = invest_profile['risk_pred_lst']
                    step_data['final_action_abssum_policy_validbest'] = invest_profile['final_action_abssum_lst']
                    step_data['rl_action_abssum_policy_validbest'] = invest_profile['rl_action_abssum_lst']
                    step_data['cbf_action_abssum_policy_validbest'] = invest_profile['cbf_action_abssum_lst']           
                    step_data['downsideAtVol_risk_policy_validbest'] = invest_profile['daily_downsideAtVol_risk_lst']
                    step_data['downsideAtValue_risk_policy_validbest'] = invest_profile['daily_downsideAtValue_risk_lst']
                    step_data['cvar_policy_validbest'] = invest_profile['cvar_lst']
                    step_data['cvar_raw_policy_validbest'] = invest_profile['cvar_raw_lst']
                print("-"*30)
                log_str = "Mode: Best-{}, Ep: {}, Capital (test set, by using the best validation model, {} ep): {} ".format(self.mode, self.epoch, int(valid_records['{}_ep'.format(self.config.trained_best_model_type)][0]), np.array(step_data['capital_policy_validbest'])[-1])
                print(log_str)

        if invest_profile['ep'] == self.config.num_epochs:
            step_data['capital_policy_last'] = _align_profile_array(invest_profile['asset_lst']).tolist()
            step_data['dailyReturn_policy_last'] = _align_profile_array(invest_profile['daily_return_lst']).tolist()
            step_data['reward_policy_last'] = _align_profile_array(invest_profile['reward_lst']).tolist()
            step_data['strategyVolatility_policy_last'] = _align_profile_array(invest_profile['stg_vol_lst']).tolist()
            step_data['risk_policy_last'] = _align_profile_array(invest_profile['risk_lst']).tolist()
            step_data['risk_wocbf_policy_last'] = _align_profile_array(invest_profile['risk_wocbf_lst']).tolist()
            step_data['capital_wocbf_policy_last'] = _align_profile_array(invest_profile['capital_wocbf_lst']).tolist()
            step_data['dailySR_policy_last'] = _align_profile_array(invest_profile['daily_sr_lst']).tolist()
            step_data['dailySR_wocbf_policy_last'] = _align_profile_array(invest_profile['daily_sr_wocbf_lst']).tolist()
            step_data['riskAccepted_policy_last'] = _align_profile_array(invest_profile['risk_adj_lst']).tolist()  
            step_data['ctrlWeight_policy_last'] = _align_profile_array(invest_profile['ctrl_weight_lst']).tolist()   
            step_data['solvable_flag_policy_last'] = _align_profile_array(invest_profile['solvable_flag']).tolist() 
            step_data['risk_pred_policy_last'] = _align_profile_array(invest_profile['risk_pred_lst']).tolist()
            step_data['final_action_abssum_policy_last'] = _align_profile_array(invest_profile['final_action_abssum_lst']).tolist()
            step_data['rl_action_abssum_policy_last'] = _align_profile_array(invest_profile['rl_action_abssum_lst']).tolist()
            step_data['cbf_action_abssum_policy_last'] = _align_profile_array(invest_profile['cbf_action_abssum_lst']).tolist()   
            step_data['downsideAtVol_risk_policy_last'] = _align_profile_array(invest_profile['daily_downsideAtVol_risk_lst']).tolist()
            step_data['downsideAtValue_risk_policy_last'] = _align_profile_array(invest_profile['daily_downsideAtValue_risk_lst']).tolist()
            step_data['cvar_policy_last'] = _align_profile_array(invest_profile['cvar_lst']).tolist()
            step_data['cvar_raw_policy_last'] = _align_profile_array(invest_profile['cvar_raw_lst']).tolist()
        step_data.to_csv(fpath, index=False)
        
        # Save detailed portfolio weights (actions) to separate file
        if hasattr(self, 'actions_memory') and len(self.actions_memory) > 0:
            action_df = self.save_action_memory()
            action_fpath = os.path.join(self.config.res_dir, '{}_actions.csv'.format(self.mode))
            action_df.to_csv(action_fpath, index=False)


    def run_mkt_observer(self, stage=None, rate_of_price_change=None):
        cur_date = self.curData['date'].unique()[0]
        if self.config.enable_market_observer:
            if stage in ['reset', 'init'] and (self.mode == 'train'):
                self.mkt_observer.reset()

            # MAFIA needs raw OCHLV data with T_w=30 window
            raw_ochlv_data = self._extract_raw_ochlv_window(cur_date, window_size=self.config.mafia_T_w)
            # raw_ochlv_data shape: (N, M, T_w) where M=5 (OCHLV), T_w=30
            
            # Get market price for reward calculation
            finemkt_feat = self.extra_data['fine_market']
            ma_close = finemkt_feat[finemkt_feat['date']==cur_date][['mkt_{}_close'.format(self.config.finefreq), 'mkt_{}_ma'.format(self.config.finefreq)]].values[-1]
            mkt_cur_close_price = ma_close[0]
            mkt_ma_price = ma_close[1]
            
            if (rate_of_price_change is not None) and (self.mode == 'train'):
                if mkt_cur_close_price > self.mkt_last_close_price:
                    mkt_direction = 0
                elif mkt_cur_close_price < self.mkt_last_close_price:
                    mkt_direction = 2
                else:
                    mkt_direction = 1
                mkt_direction = np.array([mkt_direction])
                self.mkt_observer.update_hidden_vec_reward(mode=self.mode, rate_of_price_change=rate_of_price_change, mkt_direction=mkt_direction)
            
            # Get stock MA price for controller
            finestock_feat = self.extra_data['fine_stock']
            finestock_date_data = finestock_feat[finestock_feat['date']==cur_date]
            
            # Ensure stock_ma_price has same stocks and order as curData
            stock_ma_price_dict = {}
            if len(finestock_date_data) > 0:
                for _, row in finestock_date_data.iterrows():
                    stock_ma_price_dict[row['stock']] = row['stock_{}_ma'.format(self.config.finefreq)]
            
            # Create stock_ma_price array aligned with curData stock order
            stock_ma_price = np.array([
                stock_ma_price_dict.get(stock, self.curData[self.curData['stock']==stock]['close'].values[0] if len(self.curData[self.curData['stock']==stock]) > 0 else 1.0)
                for stock in self.curData['stock'].values
            ])
            
            input_kwargs = {'mode': self.mode, 'env': self}  # Pass environment for market-index data extraction
            market_vector_np, lambda_val, boundary_risk_np, market_scores_full_np, gate_weights_np = self.mkt_observer.predict(
                raw_ochlv_data=raw_ochlv_data, 
                **input_kwargs
            )
            
            # Store gate_weights for potential analysis/visualization
            if gate_weights_np.ndim > 1:
                self.gate_weights = gate_weights_np[-1]  # (4,)
            else:
                self.gate_weights = gate_weights_np  # (4,)
            
            # Store market_scores_full in environment for RL/Solver to access
            # market_scores_full_np shape: (batch, N), extract last batch item
            if market_scores_full_np.ndim > 1:
                self.market_scores_full = market_scores_full_np[-1]  # (N,)
            else:
                self.market_scores_full = market_scores_full_np  # (N,)
            
            # Store topk_indices from Observer (for mode 'compact' - RL uses Observer's Top-K)
            if hasattr(self.mkt_observer, 'last_topk_indices') and self.mkt_observer.last_topk_indices is not None:
                # Extract last batch item if needed
                topk_indices = self.mkt_observer.last_topk_indices
                if topk_indices.ndim > 1:
                    self.observer_topk_indices = topk_indices[-1]  # (K,)
                else:
                    self.observer_topk_indices = topk_indices  # (K,)
                # Ensure indices are within valid range
                if len(self.observer_topk_indices) > 0:
                    self.observer_topk_indices = np.clip(self.observer_topk_indices, 0, self.stock_num - 1).astype(int)
            else:
                self.observer_topk_indices = None
            
            # Ensure market_scores_full has correct length (match stock_num)
            if len(self.market_scores_full) != self.stock_num:
                if len(self.market_scores_full) < self.stock_num:
                    # Pad with uniform distribution
                    padding = np.ones(self.stock_num - len(self.market_scores_full)) / self.stock_num
                    self.market_scores_full = np.concatenate([self.market_scores_full, padding])
                    # Renormalize
                    self.market_scores_full = self.market_scores_full / (np.sum(self.market_scores_full) + 1e-8)
                else:
                    # Truncate
                    self.market_scores_full = self.market_scores_full[:self.stock_num]
                    # Renormalize
                    self.market_scores_full = self.market_scores_full / (np.sum(self.market_scores_full) + 1e-8)
            
            # Handle continuous boundary_risk from MAFIA
            if self.config.is_enable_dynamic_risk_bound:
                # boundary_risk is continuous (ℝ^+)
                boundary_risk_raw = float(boundary_risk_np[-1])  # Get scalar value
                # Clip to reasonable range
                cur_risk_boundary = np.clip(
                    boundary_risk_raw,
                    self.config.risk_up_bound,
                    self.config.risk_down_bound
                )
            else:
                cur_risk_boundary = self.config.risk_default

            # Build compact state for MAFIA optimized: [market_vector(K), portfolio_value, (optional) risk_boundary]
            # Removed market_index_state - rely on Observer's learned representation
            self._build_mafia_state(market_vector_np, finemkt_feat, cur_date, cur_risk_boundary)
            self.mkt_last_close_price = mkt_cur_close_price
        else:
            cur_risk_boundary = self.config.risk_default
            if self.config.mode == 'RLcontroller':
                finestock_feat = self.extra_data['fine_stock']
                stock_cur_close_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_close'.format(self.config.finefreq)].values # (num_of_stock, )
                stock_ma_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_ma'.format(self.config.finefreq)].values # (num_of_stock, )
            else:
                stock_ma_price = None

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
        date_mask = self.rawdata['date'] == cur_date
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
        window_dates = sorted(window_data['date'].unique())
        
        # Initialize output array
        ochlv_array = np.zeros((N, M, window_size))
        
        # Extract data for each stock
        for i, stock in enumerate(stocks):
            # Get stock data from window
            stock_data = window_data[window_data['stock'] == stock].sort_values('date')
            
            # Create DataFrame with all dates in window for proper alignment
            date_df = pd.DataFrame({'date': window_dates})
            stock_df = stock_data[['date', 'open', 'close', 'high', 'low', 'volume']].copy()
            
            # Merge to align dates (left join to keep all window_dates)
            merged_df = date_df.merge(stock_df, on='date', how='left')
            
            # Fill missing values: forward fill first, then backward fill
            # This handles cases where stock data is missing for some dates
            for col in ['open', 'close', 'high', 'low', 'volume']:
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
                last_row = merged_df.iloc[-1:] if len(merged_df) > 0 else pd.DataFrame({
                    'date': [window_dates[-1]] if len(window_dates) > 0 else [cur_date],
                    'open': [0.0], 'close': [0.0], 'high': [0.0], 'low': [0.0], 'volume': [0.0]
                })
                padding = pd.concat([last_row] * (window_size - len(merged_df)), ignore_index=True)
                merged_df = pd.concat([merged_df, padding], ignore_index=True)
            
            # Extract to array (take first window_size rows)
            ochlv_array[i, 0, :] = merged_df['open'].values[:window_size]  # Open
            ochlv_array[i, 1, :] = merged_df['close'].values[:window_size]  # Close
            ochlv_array[i, 2, :] = merged_df['high'].values[:window_size]   # High
            ochlv_array[i, 3, :] = merged_df['low'].values[:window_size]    # Low
            ochlv_array[i, 4, :] = merged_df['volume'].values[:window_size] # Volume
        
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
        
        # Get index data file path
        fpath, error_msg = get_index_data_file(self.config, freq='1d')
        if fpath is None:
            raise ValueError(f"Cannot extract market index OCHLV: {error_msg}")
        
        # Load index data if not already cached
        if not hasattr(self, '_index_data_cache') or self._index_data_cache is None:
            index_data = pd.read_csv(fpath, header=0)
            index_data['date'] = pd.to_datetime(index_data['date'])
            # Normalize timezone: remove timezone info to match config dates (naive datetime)
            if index_data['date'].dt.tz is not None:
                index_data['date'] = index_data['date'].dt.tz_localize(None)
            index_data = index_data.sort_values('date', ascending=True, ignore_index=True)
            self._index_data_cache = index_data
        
        index_data = self._index_data_cache
        
        # Find current date in index data
        date_mask = index_data['date'] == cur_date
        if not date_mask.any():
            # Try to find closest date
            date_diffs = (index_data['date'] - cur_date).abs()
            closest_idx = date_diffs.idxmin()
            if date_diffs[closest_idx] > pd.Timedelta(days=7):
                raise ValueError(f"Date {cur_date} not found in index data and no close date within 7 days")
            date_idx = closest_idx
        else:
            date_idx = date_mask.idxmax()  # Get first occurrence
        
        # Get window: [date_idx - window_size + 1, date_idx]
        start_idx = max(0, date_idx - window_size + 1)
        end_idx = date_idx + 1
        
        # Extract window data
        window_data = index_data.iloc[start_idx:end_idx].copy()
        
        # Required columns: open, close, high, low, volume
        required_cols = ['open', 'close', 'high', 'low', 'volume']
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
    
    def _build_mafia_state(self, market_vector_np, finemkt_feat, cur_date, cur_risk_boundary):
        """
        Build state for MAFIA optimized architecture.
        
        SIMPLIFIED: Removed market_index_state, relying only on Observer's learned representation.
        
        Supports two modes:
        - 'compact': [market_vector(K), portfolio_value, (optional) risk_boundary]
        - 'full-score': [market_scores_full(N), portfolio_value, (optional) risk_boundary]
        
        Args:
            market_vector_np: (batch, N) or (N,) - Market vector from MAFIA (may have zeros for non-top-K)
            finemkt_feat: DataFrame - Fine market features (not used in simplified version)
            cur_date: Current date (not used in simplified version)
            cur_risk_boundary: Current risk boundary value
        """
        state_mode = getattr(self.config, 'mafia_state_mode', 'compact')
        
        if state_mode == 'full-score':
            # Full-score mode: Use market_scores_full (all N stocks)
            if self.market_scores_full is None:
                # Fallback to compact mode if market_scores_full not available
                state_mode = 'compact'
            else:
                market_scores = self.market_scores_full.copy()  # (N,)
                # Ensure correct length
                if len(market_scores) != self.stock_num:
                    if len(market_scores) < self.stock_num:
                        padding = np.ones(self.stock_num - len(market_scores)) / self.stock_num
                        market_scores = np.concatenate([market_scores, padding])
                        market_scores = market_scores / (np.sum(market_scores) + 1e-8)
                    else:
                        market_scores = market_scores[:self.stock_num]
                        market_scores = market_scores / (np.sum(market_scores) + 1e-8)
                market_vector_state = market_scores.astype(np.float32)
        
        if state_mode == 'compact':
            # Compact mode: Use Top-K from market_vector (default)
            mv = market_vector_np[-1] if market_vector_np.ndim > 1 else market_vector_np
            k = min(self.config.mafia_top_k, len(mv))
            # Get top-K indices (non-zero positions)
            topk_idx = np.argpartition(mv, -k)[-k:]
            topk_sorted = topk_idx[np.argsort(mv[topk_idx])[::-1]]
            market_vector_state = mv[topk_sorted].astype(np.float32)
        
        # Portfolio value (log-normalized)
        pv = np.log((self.cur_capital / self.initial_asset)) if self.cur_capital > 0 else 0.0
        
        # Compose simplified state (no market_index_state)
        state_parts = [
            market_vector_state,
            np.array([pv], dtype=np.float32)
        ]
        if getattr(self.config, 'mafia_include_risk_boundary_in_state', False):
            state_parts.append(np.array([cur_risk_boundary], dtype=np.float32))
        
        self.state = np.concatenate(state_parts, axis=0).astype(np.float32)

class StockPortfolioEnv_cash(StockPortfolioEnv):
    # Considering cash item
    def step(self, actions):
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        if self.terminal:
            self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1])) * self.transaction_cost))
            self.asset_lst[-1] = self.cur_capital
            self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]     
            if len(self.action_rl_memory) > 1:
                self.return_raw_lst[-1] = self.return_raw_lst[-1] * (1 - (np.sum(np.abs(self.action_rl_memory[-1])) * self.transaction_cost))

            if (self.config.enable_market_observer) and (self.mode == 'train'):
                # Training at the end of epoch
                ori_profit_rate = np.append([1], np.array(self.return_raw_lst)[1:] / np.array(self.return_raw_lst)[:-1], axis=0)
                adj_profit_rate = np.array(self.profit_lst) + 1
                label_kwargs = {'ori_profit': ori_profit_rate, 'adj_profit': adj_profit_rate, 'ori_risk': np.array(self.risk_raw_lst), 'adj_risk': np.array(self.risk_cbf_lst)}
                self.mkt_observer.train(**label_kwargs)

            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True
            print(f"[Profile Save] Epoch {self.epoch} complete (mode={self.mode}, cash), calling get_results() and save_profile()...", flush=True)
            invest_profile = self.get_results()
            print(f"[Profile Save] get_results() completed, calling save_profile()...", flush=True)
            self.save_profile(invest_profile=invest_profile)
            print(f"[Profile Save] save_profile() completed for epoch {self.epoch}", flush=True)

            # Return format compatible with both gym and gymnasium
            # Always use gymnasium format (terminated, truncated, info) for consistency with VecEnv
            return self.state, self.reward, self.terminal, False, {}
        else:
            actions = np.reshape(actions, (-1)) # [1, num_of_stocks] or [num_of_stocks, ]
            weights = self.weights_normalization(actions=actions) # Unnormalized weights -> normalized weights 
            self.actions_memory.append(weights[1:]) 
            if self.curTradeDay == 0:
                self.cur_capital = self.cur_capital * (1 - (1-1/len(weights)) * self.transaction_cost)
            else:
                # Ensure curData and lastDayData prices are aligned with stock_lst
                cur_close_prices = np.zeros(self.stock_num)
                last_close_prices = np.zeros(self.stock_num)
                for i, stock in enumerate(self.stock_lst):
                    if stock in self.curData['stock'].values:
                        close_val = self.curData[self.curData['stock'] == stock]['close'].values[0]
                        # Validate and sanitize NaN/inf/<=0 to prevent NaN propagation
                        if pd.isna(close_val) or np.isinf(close_val) or close_val <= 0:
                            # Fallback: use last known price or 1.0
                            if self.lastDayData is not None and stock in self.lastDayData['stock'].values:
                                last_close_val = self.lastDayData[self.lastDayData['stock'] == stock]['close'].values[0]
                                if not (pd.isna(last_close_val) or np.isinf(last_close_val) or last_close_val <= 0):
                                    cur_close_prices[i] = last_close_val
                                else:
                                    cur_close_prices[i] = 1.0
                            else:
                                cur_close_prices[i] = 1.0
                        else:
                            cur_close_prices[i] = close_val
                    else:
                        cur_close_prices[i] = 1.0  # Default to no change if missing
                    
                    if self.lastDayData is not None and stock in self.lastDayData['stock'].values:
                        last_close_val = self.lastDayData[self.lastDayData['stock'] == stock]['close'].values[0]
                        # Validate and sanitize NaN/inf/<=0
                        if pd.isna(last_close_val) or np.isinf(last_close_val) or last_close_val <= 0:
                            # Fallback: use current price or 1.0
                            if not (pd.isna(cur_close_prices[i]) or np.isinf(cur_close_prices[i]) or cur_close_prices[i] <= 0):
                                last_close_prices[i] = cur_close_prices[i]
                            else:
                                last_close_prices[i] = 1.0
                        else:
                            last_close_prices[i] = last_close_val
                    else:
                        # Validate cur_close_prices before using
                        if pd.isna(cur_close_prices[i]) or np.isinf(cur_close_prices[i]) or cur_close_prices[i] <= 0:
                            last_close_prices[i] = 1.0
                        else:
                            last_close_prices[i] = cur_close_prices[i]  # Use current if last not available
                
                cur_p = cur_close_prices * (1 + self.cur_slippage_drift)
                last_p = last_close_prices * (1 + self.last_slippage_drift)
                x_p = cur_p / last_p
                last_action = np.array(self.actions_memory[-2])
                last_action = np.append([1.0 - np.sum(np.abs(last_action))], last_action, axis=0) # cash
                x_p_adj = np.where((x_p>=2)&(last_action[1:]<0), 2, x_p)
                x_p_adj = np.append([1.0], x_p_adj, axis=0) # cash
                sgn = np.sign(last_action)
                sgn[0] = 1.0 # cash sign
                adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
                adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                if (adj_cap <= 0) or np.all(adj_w_ay==0):
                    raise ValueError("Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                last_w_adj = adj_w_ay / adj_cap
                self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1] - last_w_adj[1:])) * self.transaction_cost))
                self.asset_lst[-1] = self.cur_capital
                self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]
                if len(self.action_rl_memory) > 1:
                    last_rl_action = np.array(self.action_rl_memory[-2])
                    last_rl_action = np.append([1.0 - np.sum(np.abs(last_rl_action))], last_rl_action, axis=0) # cash
                    x_p_adjrl = np.where((x_p>=2)&(last_rl_action[1:]<0), 2, x_p)
                    sgn_rl = np.sign(last_rl_action)
                    sgn_rl[0] = 1.0 # cash sign
                    prev_rl_cap = self.return_raw_lst[-1]
                    adj_w_ay = sgn_rl * (last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action))
                    adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1
                    if (adj_cap <= 0) or np.all(adj_w_ay==0):
                        print("Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                        adj_w_ay = np.array([1/(self.stock_num+1)]*(self.stock_num+1)) * self.bound_flag
                        adj_cap = 1
                    last_rlw_adj =  adj_w_ay / adj_cap
                    return_raw = prev_rl_cap * (1 - (np.sum(np.abs(self.action_rl_memory[-1] - last_rlw_adj[1:])) * self.transaction_cost))
                    self.return_raw_lst[-1] = return_raw       
            
            # Jump to the next day
            self.curTradeDay = self.curTradeDay + 1
            # Save current data as lastDayData (ensure it has all stocks)
            self.lastDayData = self.curData.copy()
            self.last_slippage_drift = self.cur_slippage_drift     
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.curData.sort_values(['stock'], ascending=True, inplace=True)
            self.curData.reset_index(drop=True, inplace=True)
            
            self.ctl_state = {k:_safe_array_from_values(self.curData[k].values) for k in self.config.otherRef_indicator_lst}
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)

            self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
            
            # Ensure curData and lastDayData have same stocks in same order
            cur_close_prices = np.zeros(self.stock_num)
            last_close_prices = np.zeros(self.stock_num)
            for i, stock in enumerate(self.stock_lst):
                if stock in self.curData['stock'].values:
                    cur_close_prices[i] = self.curData[self.curData['stock'] == stock]['close'].values[0]
                else:
                    cur_close_prices[i] = 1.0  # Default to no change if missing
                if self.lastDayData is not None and stock in self.lastDayData['stock'].values:
                    last_close_prices[i] = self.lastDayData[self.lastDayData['stock'] == stock]['close'].values[0]
                else:
                    last_close_prices[i] = cur_close_prices[i]  # Use current if last not available
            
            curDay_ClosePrice_withSlippage = cur_close_prices * (1 + self.cur_slippage_drift)
            lastDay_ClosePrice_withSlippage = last_close_prices * (1 + self.last_slippage_drift)
            rate_of_price_change = curDay_ClosePrice_withSlippage / (lastDay_ClosePrice_withSlippage + 1e-8)  # Avoid division by zero
            rate_of_price_change_adj = np.where((rate_of_price_change>=2)&(weights[1:]<0), 2, rate_of_price_change)
            sigDayReturn = (rate_of_price_change_adj - 1) * weights[1:] # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)
            if poDayReturn <= (-1):
                raise ValueError("Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(self.curTradeDay, self.date_memory[-1], poDayReturn))
            
            updatePoValue = self.cur_capital * ((poDayReturn + 1 - np.abs(weights[0])) + np.abs(weights[0]))
            poDayReturn_withcost = (updatePoValue - self.cur_capital) / self.cur_capital
            self.cur_capital = updatePoValue
            
            self.profit_lst.append(poDayReturn_withcost)
            self.asset_lst.append(self.cur_capital)

            # Build MAFIA state via market observer
            rate_of_price_change_withcash = np.append([1.0], rate_of_price_change_adj, axis=0)
            cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage='run', rate_of_price_change=np.array([rate_of_price_change_withcash]))  
            if stock_ma_price is not None:
                self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
            self.risk_adj_lst.append(cur_risk_boundary)
            self.ctrl_weight_lst.append(1.0) 

            # For debugging
            daily_return_ay = _safe_array_from_values(self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values)
            cur_cov = np.cov(daily_return_ay) 
            self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights[1:], cur_cov), weights[1:].T)))
            w_rl = self.action_rl_memory[-1] # Not applicable # weights[1:] - self.action_cbf_memeory[-1]
            w_rl = w_rl / np.sum(np.abs(w_rl))
            self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))

            if self.curTradeDay == 1:
                prev_rl_cap = self.return_raw_lst[-1] * (1 - (1-1/len(weights)) * self.transaction_cost)
            else:
                prev_rl_cap = self.return_raw_lst[-1]

            rate_of_price_change_adj_rawrl = np.where((rate_of_price_change>=2)&(w_rl<0), 2, rate_of_price_change)
            return_raw = prev_rl_cap * ((np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl) + 1 - np.abs(weights[0])) + np.abs(weights[0]))
            if return_raw <= 0:
                raise ValueError("Loss the whole capital if using RL actions only! [Day: {}, date: {}, return_raw: {}]".format(self.curTradeDay, self.date_memory[-1], return_raw))
            self.return_raw_lst.append(return_raw)

            # CVaR
            # daily_return_ay is 1D, take last 21 elements
            if daily_return_ay.ndim == 1:
                expected_r_series = daily_return_ay[-21:] if len(daily_return_ay) >= 21 else daily_return_ay
                expected_r_prev = np.mean(expected_r_series[-1:]) if len(expected_r_series) > 0 else 0.0
                # For 1D, expected_r_prev is scalar, convert to array matching weights[1:]
                expected_r_prev = np.full(len(weights[1:]), expected_r_prev)
                expected_r_prev = np.where((expected_r_prev>=1)&(weights[1:]<0), 1, expected_r_prev)
                expected_r = np.sum(expected_r_prev * weights[1:])
                # For 1D series, covariance is scalar, create diagonal matrix
                if len(expected_r_series) < 2:
                    expected_cov = np.eye(len(weights[1:])) * self.config.risk_market
                else:
                    expected_cov_val = np.var(expected_r_series)
                    expected_cov = np.eye(len(weights[1:])) * expected_cov_val
            else:
                expected_r_series = daily_return_ay[:, -21:] if daily_return_ay.shape[1] >= 21 else daily_return_ay
                expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
                expected_r_prev = np.where((expected_r_prev>=1)&(weights[1:]<0), 1, expected_r_prev)
                expected_r = np.sum(np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights[1:], (-1, 1)))
                expected_cov = np.cov(expected_r_series)
                # Ensure expected_cov is 2D and matches weights[1:] shape
                if expected_cov.ndim == 0:
                    expected_cov = np.eye(len(weights[1:])) * float(expected_cov)
                elif expected_cov.ndim == 1:
                    expected_cov = np.diag(expected_cov)
                if expected_cov.shape[0] != len(weights[1:]) or expected_cov.shape[1] != len(weights[1:]):
                    if expected_cov.size == 1:
                        expected_cov = np.eye(len(weights[1:])) * float(expected_cov.flat[0])
                    else:
                        target_size = len(weights[1:])
                        if expected_cov.shape[0] < target_size:
                            pad_size = target_size - expected_cov.shape[0]
                            expected_cov = np.pad(expected_cov, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=self.config.risk_market)
                        elif expected_cov.shape[0] > target_size:
                            expected_cov = expected_cov[:target_size, :target_size]
            
            # Calculate expected_std safely
            weights_1d = weights[1:].flatten() if weights[1:].ndim > 1 else weights[1:]
            try:
                temp = np.matmul(weights_1d, expected_cov)  # (N,) @ (N, N) -> (N,)
                expected_std = np.sqrt(np.matmul(temp, weights_1d))  # (N,) @ (N,) -> scalar
                if isinstance(expected_std, np.ndarray):
                    expected_std = float(expected_std.item() if expected_std.size == 1 else expected_std.flat[0])
            except (ValueError, TypeError, IndexError):
                expected_std = self.config.risk_market
            cvar_lz = spstats.norm.ppf(1-0.05) # positive 1.65 for 95%(=1-alpha) confidence level.
            cvar_Z = np.exp(-0.5*np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2*np.pi)
            cvar_expected = -expected_r + expected_std * cvar_Z
            self.cvar_lst.append(cvar_expected)

            # CVaR without risk controller
            if expected_r_series.ndim == 1:
                expected_r_prevrl = np.mean(expected_r_series[-1:]) if len(expected_r_series) > 0 else 0.0
                expected_r_prevrl = np.full(len(w_rl), expected_r_prevrl)
                expected_r_prevrl = np.where((expected_r_prevrl>=1)&(w_rl<0), 1, expected_r_prevrl)
                expected_r_raw = np.sum(expected_r_prevrl * w_rl)
            else:
                expected_r_prevrl = np.mean(expected_r_series[:, -1:], axis=1)
                expected_r_prevrl = np.where((expected_r_prevrl>=1)&(w_rl<0), 1, expected_r_prevrl)
                expected_r_raw = np.sum(np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1)))
            
            # Calculate expected_std_raw safely
            w_rl_1d = w_rl.flatten() if w_rl.ndim > 1 else w_rl
            try:
                temp = np.matmul(w_rl_1d, expected_cov)  # (N,) @ (N, N) -> (N,)
                expected_std_raw = np.sqrt(np.matmul(temp, w_rl_1d))  # (N,) @ (N,) -> scalar
                if isinstance(expected_std_raw, np.ndarray):
                    expected_std_raw = float(expected_std_raw.item() if expected_std_raw.size == 1 else expected_std_raw.flat[0])
            except (ValueError, TypeError, IndexError):
                expected_std_raw = self.config.risk_market
            cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
            self.cvar_raw_lst.append(cvar_expected_raw)

            # Ensure poDayReturn_withcost is valid before taking log
            if np.isnan(poDayReturn_withcost) or np.isinf(poDayReturn_withcost):
                poDayReturn_withcost = 0.0
            # Avoid log(0) = -inf when poDayReturn_withcost = -1
            if poDayReturn_withcost <= -1:
                profit_part = -10.0  # Large negative value instead of -inf
            else:
                profit_part = np.log(poDayReturn_withcost+1)
            if len(self.action_rl_memory) > 0:
                w_rl_latest = self.action_rl_memory[-1]
            else:
                w_rl_latest = weights
            weights_norm = _normalize_prob(weights)
            w_rl_norm = _normalize_prob(w_rl_latest)
            js_distance = jensenshannon(w_rl_norm, weights_norm, base=2) ** 2
            if np.isnan(js_distance) or np.isinf(js_distance):
                js_distance = 0.0
            j_return = profit_part
            scaled_profit_part = self.config.lambda_1 * j_return
            scaled_js_part = self.config.lambda_2 * js_distance
            cur_reward = scaled_profit_part + scaled_js_part

            self.rl_reward_risk_lst.append(scaled_js_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)
            # Ensure cur_reward is not NaN or inf (root cause fix)
            if np.isnan(cur_reward) or np.isinf(cur_reward):
                print(f"Warning: cur_reward is {cur_reward} (profit_part={profit_part}, scaled_js_part={scaled_js_part}), replacing with 0.0", flush=True)
                cur_reward = 0.0
            self.reward = cur_reward
            self.reward_lst.append(self.reward)
            self.model_save_flag = False

            # Return format compatible with both gym and gymnasium
            # Always use gymnasium format (terminated, truncated, info) for consistency with VecEnv
            return self.state, self.reward, self.terminal, False, {}
