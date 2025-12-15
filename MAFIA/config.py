# ！/usr/bin/python
# -*- coding: utf-8 -*-#


import argparse
import datetime
import numpy as np
import os
import pandas as pd
import sys
import time
from stable_baselines3.td3.policies import MultiInputPolicy
from RL_controller.compressed_replay_buffer import (
    CompressedDictReplayBuffer,
    CompressedReplayBuffer,
)

# from RL_controller.TD3_controller import TD3PolicyOriginal
# from RL_controller.feature_extractors import (
#     MAFIAMultiBranchExtractor as MAFIASingleStageMLP,
# )
from stable_baselines3.common.utils import LinearSchedule


class Config:
    GATING_MODE_MAP = {
        "attentive": "attention_based_aggregation",
        "cnn": "temporal_convolution",
        "lstm": "lstm",
    }
    DEFAULT_GATING_MODE_ALIAS = "lstm"
    DEFAULT_GATING_ENCODER = GATING_MODE_MAP[DEFAULT_GATING_MODE_ALIAS]
    GATING_MODE_DESCRIPTIONS = {
        "attentive": "Self-attention gating that weighs temporal embeddings via multi-head attention.",
        "cnn": "Temporal convolution (CNN) gating that extracts short-term patterns before routing.",
        "lstm": "Unidirectional LSTM gating (causal, forward-only) that captures sequential dependencies.",
    }

    def __init__(self, seed_num=2022, current_date=None, create_dirs=True):
        """
        Initialize Config.

        Args:
            seed_num: Random seed (default: 2022)
            current_date: Override current date string (default: None, uses now())
            create_dirs: If True, auto-create result directories (default: True)
                        Set to False when using custom output paths (e.g., observer offline training)
        """
        self.notes = "MAFIA Implementation - MAFIA-only (Legacy models removed)"

        # MAFIA-only configuration (Legacy TD3-only and old MASA variants removed)
        self.benchmark_algo = "TD3-PR"  # TD3 Profit-Risk optimization
        self.market_name = "VNINDEX"  # Financial Index: 'DJIA', 'SP500', 'CSI300'
        self.topK = 10  # Number of assets in a portfolio (10, 20, 30)
        self.num_epochs = 50  # episodes for convergence

        # MAFIA configuration (fixed)
        self.rl_model_name = "TD3"  # RL agent implemented by TD3
        self.mode = "RLcontroller"  # MASA framework with MAFIA observer
        self.mktobs_algo = "mafia_1"  # MAFIA observer (only observer supported)
        self.trained_best_model_type = "js_loss"

        self.is_enable_dynamic_risk_bound = True  # True if enabling that the market observer sends info to solver-based agents.
        self.enable_controller = True  # Enable solver-based agent for risk correction
        self.enable_market_observer = True  # True if enabling the market observer.
        # Enable multibranch Dict observation (global + per_stock + history) when using observer
        self.rl_obs_use_multibranch_state = True
        # Optimized MAFIA toggles / hyperparameters
        self.mafia_use_gumbel_topk = (
            False  # Disable Gumbel-TopK, pass raw market vector to RL
        )
        self.mafia_top_k = 10
        self.mafia_gumbel_temperature = 1.0  # Training init temperature
        self.mafia_gumbel_temp_inference = (
            0.1  # Inference temperature (sharper decisions)
        )
        self.mafia_gumbel_temp_min = 0.1  # Minimum temperature after decay
        self.mafia_gumbel_temp_decay = (
            0.99  # Decay rate per episode: temp = max(min, init * decay^episode)
        )
        self.mafia_hard_topk_inference = True  # use hard Top-K at eval
        self.mafia_include_risk_boundary_in_state = True
        # Live display (can be re-enabled with env: MAFIA_USE_LIVE_DISPLAY=1)
        env_live_display = os.environ.get("MAFIA_USE_LIVE_DISPLAY")
        if env_live_display is None:
            self.use_live_display = True  # Default True (can be disabled via env var)
        else:
            self.use_live_display = env_live_display.strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
        # Observer Pretrain Configuration
        # Pretrain uses mini-epochs (each = observer_mini_epoch_steps = 126 steps)
        # After pretrain, Observer is ALWAYS frozen (Static Expert mode)
        # Note: 3 mini-epochs = 378 steps > observer_warmup_samples (300)
        # This ensures at least one training happens during pretrain
        self.mafia_pretrain_mini_epochs = (
            3  # Number of observer-only mini-epochs before TD3 (window 0 only)
        )
        # Observer training flag
        # - True: Only during pretrain phase (set automatically by entrance.py)
        # - False: After pretrain, Observer frozen as "Static Expert"
        self.mafia_allow_observer_training = False
        # Warmup: Minimum samples to collect before first Observer training update
        # Observer will skip training until buffer has >= this many samples
        self.observer_warmup_samples = 300
        # Observer sequence sampling: train on contiguous mini-batch with random start
        # Set >0 to sample a window of this length from observer buffers at each train()
        self.observer_seq_len = 128  # window length T_m (days); 0 = use full buffer
        self.observer_seq_sampling = "random"  # options: "random", "recent"
        # Trajectory configuration (Spec §6)
        self.mafia_trajectory_length = 128  # T_m for on-policy batch segments
        self.mafia_sampling_strategy = "random_trajectory"  # or "recent_trajectory"
        # MAFIA state mode: 'compact' (Top-K market_vector) or 'full-score' (full market_scores_full)
        # Top-K only: Observer chọn Top-K → State có Top-K → RL tối ưu trên Top-K (full-universe mode bị vô hiệu hóa)
        self.mafia_state_mode = "compact"

        # RL Top-K selection method (only used in 'full-score' mode)
        # In 'compact' mode: RL automatically uses Top-K from Observer (no selection needed)
        # In 'full-score' mode: RL automatically self-selects Top-K from market_scores_full
        self.mafia_rl_topk_selection_method = (
            "topk"  # Options: 'topk' (select top K), 'threshold' (scores > threshold)
        )
        self.mafia_rl_topk_threshold = 0.02  # Threshold for 'threshold' method (only stocks with score > threshold)

        # CBF Controller: Use market_scores_full as prior distribution (works in both modes)
        self.mafia_cbf_use_prior = (
            False  # If True, CBF uses market_scores_full as prior
        )
        self.mafia_cbf_prior_weight = 0.3  # Weight for prior distribution (0.0-1.0)
        # Legacy flag (disabled): RL full-universe weights không còn được hỗ trợ, luôn dùng Top-K từ Observer
        self.mafia_action_full_universe = False
        # Solver: in 'full-score' mode, restrict optimization to RL-selected Top-K only (zero others)
        # Default True to keep optimization aligned to RL Top-K
        self.mafia_solver_use_rl_topk_only = True
        # Risk bound handling
        self.risk_bound_warmup_steps = 30
        self.risk_bound_is_annualized = False  # Set True if observer outputs annualized sigma; will be scaled to daily
        self.risk_bound_minvar_eps = 0.0  # Safety margin when kẹp bound to min_var_K (risk_bound = max(bound, min_var_K*(1+eps)))

        # Solver boost behavior (only active in 'full-score' mode when RL has selected Top-K)
        # In 'full-score' mode: Solver can boost stocks already selected by RL, or keep original logic
        self.mafia_solver_boost_enabled = False  # If True, Solver boosts stocks selected by RL (only in 'full-score' mode)
        self.mafia_solver_boost_method = (
            "blend"  # Options: 'blend' or 'proportional' (same as boost methods)
        )
        self.mafia_solver_boost_factor = 0.3  # Weight for blending/boosting (0.0-1.0)
        # Solve-Based Agent L1 limits by market regime (Solve spec §4.4)
        self.solver_alpha_up = 0.05
        self.solver_alpha_hold = 0.12
        self.solver_alpha_down = 0.15
        # Auto-relax L1 cap when risk bound is infeasible under current alpha (keeps controller from giving up)
        self.solver_alpha_relax_factor = 1.05

        # Market-index Agent and Self-Attention configuration
        self.mafia_use_market_index_agent = (
            True  # If True, enable Market-index agent (VNINDEX)
        )
        self.mafia_attention_agg = "weighted"  # Aggregation method for ST-Fusion embeddings in attention: 'mean', 'weighted', 'max'

        self.trade_pattern = 1  # 1: Long only, 2: Long and short (Not applicable), 3: short only (Not applicable)
        # Reward weights (tuned via quick Optuna on mini window)
        self.lambda_1 = 500  # return reward weight
        self.lambda_2 = 30  # Tăng λ₂ nghĩa là phạt nặng hơn khi RL khác xa controller → khuyến khích RL bám sát/quy phục hành động của controller (ít lệch, ít “liều” theo hướng riêng), thường dẫn đến phân bổ ổn định hơn và ít turnover thay đổi mạnh.
        # Nếu True: bỏ phạt turnover/change khỏi reward (chỉ dùng khi muốn RL thuần lợi nhuận + JS)
        self.rl_reward_disable_turnover_change = True
        # Encourage diversified actions (entropy regularizer on policy output)
        self.entropy_coef = 0.2  # loss_actor = − E_s [ Q(s, π(s)) ] − entropy_coef * entropy -> entropy_coef * entropy: thưởng entropy để hành động đa dạng/khám phá; entropy_coef càng lớn, actor càng “spread” phân phối hành động.
        # Policy Gradient horizon for observer PG loss (Top-K compounding window)
        self.mafia_pg_reward_horizon = (
            14  # Increased from 1 to learn longer-term patterns
        )
        # PG reward shaping penalties (Top-K turnover & membership change)
        # Penalty coefficients (stronger to stand against reward scaling=100)
        self.mafia_pg_alpha_turnover = 1.25  # Turnover penalty coefficient
        self.mafia_pg_alpha_change = 1.5  # Membership change penalty coefficient
        self.mafia_reward_alpha_hold = 1.5  # Trend Holding Bonus (Reward for holding profitable stocks)
        # Direction label generation (future-based)
        self.direction_label_lookahead = 14  # k days ahead for R_fut
        self.direction_label_delta = (
            0.025  # symmetric threshold δ (default 2.5%) - used in online mode
        )

        # Dynamic Threshold for Direction Labeling (Spec §5.1.3)
        # Used in offline batch training for adaptive thresholding based on market volatility
        # δ_t = max(δ_min, k_atr * ATR_14 / P_t)
        # Optimized based on VNINDEX analysis: ATR/Price mean=1.4%, IntraDD P10=-8.5%
        self.direction_label_atr_period = 14  # ATR window (days)
        self.direction_label_atr_multiplier = 2.0  # k_atr: ATR multiplier (2σ margin)
        self.direction_label_delta_min = 0.020  # δ_min: minimum threshold (2.0%)
        self.direction_label_stop_loss = -0.07  # IntraDD threshold (-7%, P10)

        # ============================================================
        # Regime Shift Detection Thresholds (Spec Section 8)
        # Calibrated from VNINDEX statistics (2015-2025, 2709 trading days)
        # ============================================================
        # Enable regime shift detection in training & inference
        self.enable_regime_shift_detection = (
            True  # Master switch for regime shift logic
        )

        # Trigger 1: Direction Reversal (Bull <-> Bear confirmed over N days)
        self.regime_confirmation_window = 3  # N-day confirmation for direction reversal
        self.regime_trend_z_threshold = 1.0  # minimum |Z_trend| to confirm reversal

        # Trigger 2: Volatility Shock (V_curr > mu + k*sigma)
        self.regime_vol_window = 20  # window for realized volatility
        self.regime_vol_k = (
            3.0  # k * std for volatility shock (captures ~1% extreme events)
        )

        # Trigger 3: Structural Break (Downward DC at major threshold)
        self.regime_dc_threshold_pct = (
            0.02  # DC trigger threshold on index (2%) - Spec 8.2 Major Reversal
        )

        # Legacy threshold (for backward compatibility)
        self.regime_shift_threshold = 0.2  # General regime shift detection threshold
        self.controller_reg_lambda = 1.5  # λ_reg: controller regularization weight ||x - a_RL||^2,  λ_reg lớn → bám sát RL, nhỏ → cho solver chỉnh mạnh hơn
        self.controller_observer_bias_weight = (
            0.3  # α: scales observer signal when forming linear bias q
        )
        # TD3 learning-rate schedule: 'linear', 'linear_per_epoch', 'cyclic_run_decay', or None
        # cyclic_run_decay: LR decays across the full run and also tapers in the tail of each epoch (no per-epoch reset)
        self.td3_lr_schedule = "cyclic_run_decay"
        self.td3_lr_end_factor = 0.2  # end_lr = start_lr * end_factor
        self.td3_lr_end_fraction = (
            0.5  # fraction of each epoch where end_lr reached (for linear_per_epoch)
        )
        # Turnover and membership-change penalties (turnover uses raw sum |w_t - w_{t-1}| )
        self.lambda_tc = 0.001
        self.lambda_change = (
            0.001  # Penalty weight for membership change (Top-K symmetric difference)
        )
        # Debug: log reward components for first N train steps (0 = disable)
        self.reward_debug_steps = 30
        self.train_freq = [1, "step"]  # Update every trading step
        # Risk tolerance factors (eta): relative multipliers on baseline sigma (Top-K)
        self.risk_eta_default = 1.0
        self.risk_eta_hold = 1.0  # sideways
        self.risk_eta_up = 1.3  # bull
        self.risk_eta_down = 0.7  # bear
        # Supervised eta label generation (lookback/lookahead)
        self.risk_eta_label_lookback = 60
        self.risk_eta_lookahead = 14  # Default 14 days lookahead for eta target
        self.risk_eta_label_epsilon = 1e-6
        # S_risk: Scaling factor for MSE (amplify gradient)
        # L_risk (MSE) ≈ 0.01, need to scale up to match L_selection (~1.0) and L_direction (~1.0)
        # With scale=10.0: L_risk contribution ≈ 0.01 × 10 = 0.1 (balanced)
        self.scale_factor_risk = 10.0
        # S_reward: Scaling factor for PG Reward (amplify Advantage gradient)
        # Raw compounding returns ~0.01, scale to ~1.0 for stable gradients
        self.scale_factor_reward = 100.0
        # Continuous eta mapping (tanh over market index z-score)
        self.risk_eta_window = 30
        self.risk_eta_lambda = 0.3  # eta in [1-λ, 1+λ] => [0.7, 1.3]
        self.risk_eta_sensitivity = 1.0
        # MaxDD Penalty for Hybrid eta_target (spec 5.1.2)
        self.risk_eta_lambda_dd = 0.5  # Drawdown penalty weight
        self.risk_eta_dd_ref = 0.10  # Reference drawdown (10%)
        # Observer eta scaling (tanh-based, clipped)
        # FIX: Expanded range to match target eta range [0.1, 2.0]
        # Target eta formula: eta = 1.0 + lambda_val*tanh(z) - lambda_dd*clip(dd/ref,0,1)
        # With lambda_val=0.3, lambda_dd=0.5 => raw range [0.2, 1.3], clipped to [0.1, 2.0]
        # Predicted must cover same range for proper correlation
        self.mafia_eta_base = 1.05  # Center of [0.1, 2.0]
        self.mafia_eta_amplitude = 0.95  # (2.0 - 0.1) / 2 = 0.95
        self.mafia_eta_min = 0.1
        self.mafia_eta_max = 2.0
        # Legacy aliases maintained for backward compatibility
        self.risk_default = self.risk_eta_default
        self.risk_hold_bound = self.risk_eta_hold
        self.risk_up_bound = self.risk_eta_up
        self.risk_down_bound = self.risk_eta_down
        # Smooth risk_bound over time to avoid abrupt jumps that cause large controller moves (0 = no smoothing)
        self.risk_bound_smoothing_alpha = 0.0

        # ============================================================
        # Portfolio Allocator (TD3) Configuration
        # Per spec: Porfolio_allocator_spec.md Section 11
        # ============================================================

        # Enable new 2-component reward function (log return + JS divergence)
        self.use_portfolio_allocator_reward = True

        # Reward weights
        self.allocator_return_weight = 1.0  # w_return: weight for log return component
        self.allocator_lambda_js = (
            0.1  # λ_js: Jensen-Shannon divergence penalty [0.05, 0.2]
        )
        # Reward scaling: amplifies reward signal for stronger gradients
        # Daily log returns are tiny (~0.001), scaling helps TD3 learn faster
        # Recommended range: [10, 100], default 100 to bring rewards to ~0.1 scale
        self.allocator_reward_scale = 100.0
        self.allocator_reward_norm_alpha = (
            0.01  # EMA coefficient for reward normalization
        )

        # Learning rates (spec defaults)
        self.allocator_learning_rate_actor = 1e-4
        self.allocator_learning_rate_critic = 1e-3

        # TD3 hyperparameters (aligned with SB3 defaults per spec)
        self.allocator_discount_gamma = 0.99  # γ: discount factor
        self.allocator_polyak_tau = 0.005  # τ: target network update rate
        self.allocator_policy_delay = 2  # d: update actor every d critic updates

        # Training parameters
        self.allocator_batch_size = 128  # Mini-batch size (spec: 64-256)
        self.allocator_replay_buffer_size = 100000  # Replay buffer capacity
        self.allocator_warmup_steps = (
            5000  # Steps before training begins (spec: 1000-5000)
        )
        self.allocator_exploration_noise_std = 0.1  # Exploration noise std

        # Architecture dimensions
        self.allocator_feature_hidden_dim = 128  # Feature extractor hidden dim
        self.allocator_actor_hidden_dim = 256  # Actor network hidden dim
        self.allocator_critic_hidden_dim = 256  # Critic network hidden dim
        self.allocator_gradient_clip = 1.0  # Gradient clipping threshold

        # Feature extractor output dimension (D_hidden per spec Section 3.4)
        self.rl_features_dim = 256

        # State configuration (per spec Section 3)
        # History buffer features: portfolio_return, rl_last_action, rl_last_turnover, cbf_adjustment
        self.rl_obs_history_features = [
            "portfolio_return",
            "rl_last_action",
            "rl_last_turnover",
            "cbf_adjustment",
        ]
        self.rl_obs_history_len = 5  # Number of historical steps in state

        # Global context scalar features (per spec Section 3.4)
        # Includes Contextual Awareness Features: time_decay, relative_alpha, risk_violation
        # Plus Rebalance Signals for TD3 to learn distinct policies
        self.rl_obs_global_scalars = [
            "log_capital",
            "last_turnover",
            "drawdown",
            "time_decay",  # τ_decay: days_since_rebalance / rebalance_interval ∈ [0, 1]
            "relative_alpha",  # α_rel: R_portfolio - R_topk_avg (Credit Assignment)
            "risk_violation",  # δ_risk: max(0, σ(a^RL) - σ_target) (Risk Violation Diagnostic)
            # Rebalance Signals: TD3 can learn different policies for each type
            "is_rebalance_scheduled",  # 1.0 = scheduled rebalance (every N days), 0.0 = hold
            "is_rebalance_regime",  # 1.0 = regime shift triggered rebalance, 0.0 = no regime shift
        ]

        # ============================================================
        # Walk-Forward Training Configuration
        # Separates Observer pre-training from RL training
        # Phase 1: Train Observer → Save observer_best.pth
        # Phase 2: RL Training with frozen Observer ("Static Expert")
        # NOTE: NO COMBINED MODE! Observer and TD3 NEVER train together!
        # ============================================================

        # Training phase indicator (1 = Observer, 2 = TD3)
        # Phase 1: Observer training, TD3 frozen (uniform weights)
        # Phase 2: TD3 training, Observer frozen (Static Expert)
        self.training_phase = 2  # Default: Phase 2 (TD3 training)

        # Training mode for display (derived from training_phase)
        # "OBSERVER_ONLY" for Phase 1, "RL_ONLY" for Phase 2
        self.training_mode = "RL_ONLY"  # Default: Phase 2

        # Observer-only training mode (Phase 1)
        # Set to True only when running Phase 1 (train_separated.py --phase 1)
        self.observer_only_training = False  # Default: Phase 2 (TD3 training)

        # Pre-trained observer checkpoint path (Phase 2)
        # REQUIRED for Phase 2 - loads this checkpoint and freezes observer
        self.observer_pretrained_path = None  # Set path to observer_best.pth

        # Freeze observer during RL training (Phase 2) - DEFAULT ENABLED
        # Observer acts as "Static Expert" - only provides Top-K, risk_eta, direction
        self.freeze_observer_during_rl = True  # Separated training by default

        # Walk-forward iteration settings
        self.walkforward_iteration = (
            0  # 0 = base training (from scratch), 1+ = finetune
        )
        self.walkforward_train_start_year = 2015  # Expanding window start
        self.walkforward_train_end_year = None  # Set dynamically per iteration
        self.walkforward_valid_year = None  # Validation year (same as train_end)
        self.walkforward_infer_year = None  # Inference year (train_end + 1)

        # Epoch schedule (varies by iteration)
        self.walkforward_base_epochs = 50  # Iteration 0: train from scratch
        self.walkforward_finetune_epochs = 20  # Iteration 1+: finetune
        self.walkforward_base_lr = 1e-4  # Base learning rate
        self.walkforward_finetune_lr = 1e-5  # Finetune learning rate
        self.walkforward_base_patience = 15  # Early stopping patience (base)
        self.walkforward_finetune_patience = 5  # Early stopping patience (finetune)

        # Composite score weights for checkpoint selection
        # Score = w_sharpe × SR + w_f1 × F1 + w_risk × (1 - MSE)
        self.walkforward_score_w_sharpe = 0.5  # Top-K Sharpe Ratio weight
        self.walkforward_score_w_f1 = 0.3  # Direction F1-Macro weight (Increased)
        self.walkforward_score_w_risk = 0.2  # Risk MSE weight (Inverted)

        # ============================================================
        # Expanding Window Mode Configuration (New Training Paradigm)
        # ============================================================
        # Timeline per iteration:
        #   Iter 1: Train[01/2015 → 06/2017] → Valid[07/2017 → 12/2017] → Infer[2018]
        #   Iter 2: Train[01/2015 → 06/2018] → Valid[07/2018 → 12/2018] → Infer[2019]
        #   ...
        # Key differences from Sliding Window:
        #   - Train start is FIXED (expanding_train_start)
        #   - Train end EXPANDS each iteration
        #   - Each iteration loads checkpoint from previous iteration (finetune)
        # ============================================================

        # Enable Expanding Window mode (False = use traditional Sliding Window)
        self.expanding_window_mode = True

        # Fixed training start date (Expanding Window only)
        self.expanding_train_start = "2015-01-01"

        # Validation period: 6 months at end of train year (07-12)
        self.expanding_valid_months = 6

        # Save inference states for downstream RL training
        self.save_inference_states = True

        # Finetune settings (applied when loading checkpoint from previous iteration)
        # Iter 1 trains from scratch, Iter 2+ finetunes from Ckpt_Best_{Year-1}
        self.finetune_epoch_factor = (
            0.4  # epochs = base_epochs * factor (e.g., 50 * 0.4 = 20)
        )
        self.finetune_lr_factor = 0.1  # lr = base_lr * factor (e.g., 1e-4 * 0.1 = 1e-5)

        # Rebalance interval for selection head training (days)
        self.topk_rebalance_interval = 14  # Selection Head update every 14 days

        # Output directories for walk-forward artifacts
        self.walkforward_checkpoint_dir = None  # Directory for observer checkpoints
        self.walkforward_states_dir = None  # Directory for pre-computed RL states

        self.period_mode = 1
        self.tmp_name = "Cls3_{}_{}_K{}_M{}_{}_{}".format(
            self.mode,
            self.mktobs_algo,
            self.topK,
            self.period_mode,
            self.market_name,
            self.trained_best_model_type,
        )
        # Allow overriding data root for container/host mounts
        self.dataDir = os.path.abspath(os.environ.get("MAFIA_DATA_DIR", "./data"))

        # Data file configuration - can specify custom file names or use None for auto-detection
        # If None, will use pattern: {market_name}_{topK}_{freq}.csv
        # Set to a specific filename to use that file from the data directory
        # Available filtered datasets:
        #   - stock_data_top23.csv (23 blue-chip stocks, ADV > 300B, recommended for 16GB RAM)
        #   - stock_data_top143.csv or stock_data_dynamic143.csv (143 stocks, ADV > 5B, needs 32GB+ RAM)
        #   - stock_prices_all_20251108_234851.csv (235 stocks, full dataset, needs 64GB+ RAM)
        self.stock_data_file = (
            "stock_data_dynamic143.csv"  # Use top 23 stocks for Mac 16GB
        )
        # Data quality guards
        self.mafia_price_floor = 1e-6  # Replace zero/negative prices with ffill/bfill
        self.mafia_extreme_return_threshold = (
            10.0  # |return| > threshold is treated as anomaly before clipping
        )
        self.mafia_return_clip_min = (
            -0.5
        )  # Post-clean clip bounds (matches PG reward safety)
        self.mafia_return_clip_max = 1.0
        self.index_data_file = "VNINDEX_1d_index.csv"  # Optional: 'DJIA_1d_index.csv' or None. If None, market features will be generated from stock data
        self.pricePredModel = "MA"
        self.cov_lookback = 30
        self.norm_method = "sum"
        self.max_zero_volume_days = (
            100  # Drop stocks with > this number of zero-volume days
        )
        self.rebalance_interval = 1  # Days between portfolio rebalances
        # Scheduler: decouple Top-K refresh cadence from daily eta updates
        # Top-K/market-direction refreshed on rebalance window or regime-shift trigger
        self.mafia_topk_rebalance_interval = int(
            os.environ.get(
                "MAFIA_TOPK_REBALANCE_INTERVAL", 10
            )  # Changed from 15 to 10 days
        )
        # Eta/risk calibration tick (env step granularity); keep daily by default
        self.mafia_eta_update_interval = 1
        # Allow regime detector to force early rebalance outside fixed cadence
        self.mafia_enable_regime_force_rebalance = True
        # Verbose scheduler/regime logs on terminal during training
        # Set to True for debugging, False for clean output
        self.mafia_log_scheduler = (
            False  # [SCHEDULER] hold status logs (separate lines)
        )
        self.mafia_log_eta = False  # [RISK-ETA] risk boundary logs (separate lines)
        self.mafia_log_reward = (
            False  # [PA-REWARD] reward component logs (separate lines)
        )
        # Single-line realtime status (recommended for monitoring)
        self.mafia_log_realtime = (
            True  # [ENV-STATUS] all-in-one realtime status on 1 line
        )
        
        # Trajectory Logging: Default to logging only representative sample (idx=0)
        # Set to True (via config or CLI --log-details) to log ALL trajectories in batch.
        self.log_trajectory_details = False

        if self.mode == "Benchmark":
            self.trained_best_model_type = "max_capital"
        if self.mode == "RLonly":
            if self.trained_best_model_type not in ["max_capital", "js_loss"]:
                raise ValueError(
                    "The trained_best_model_type[{}] of {} should be in ['max_capital', 'js_loss'].".format(
                        self.trained_best_model_type, self.mode
                    )
                )

        # Default fallback for market risk (\Sigma_beta): 18% annual vol -> daily sigma
        self.default_risk_market = 0.18 / np.sqrt(252)
        self.cbf_gamma = 0.7
        # Blend factor for observer risk bound vs min-variance bound in CBF solver
        self.risk_bound_alpha = 0.7
        # Observer mini-epochs: train observer more frequently and reset its buffers to save memory
        self.observer_mini_epoch_steps = (
            126  # 0.5 epoch mini-epochs (126 steps for 252 trading days/year)
        )
        # TD3 config
        self.reward_scaling = 1
        self.learning_rate = 0.0001
        # Warm-up at least as long as observer mini-epoch (but not too large to delay TD3)
        self.learning_starts = max(self.observer_mini_epoch_steps, 300)
        self.batch_size = 256
        # Number of mini-batches the TD3 learner runs after each train_freq chunk.
        # Higher value ensures TD3 actually updates parameters frequently.
        self.gradient_steps = 1
        # Use SB3's memory-optimized replay buffer (safe when using VecEnv/DummyVecEnv)
        self.optimize_memory_usage = True
        # Compress observations inside the replay buffer to reduce memory footprint
        self.compress_obs = True
        self.replay_buffer_dtype = np.float16
        self.action_noise_sigma = (
            0.1328873248989795  # Std-dev for Gaussian action noise
        )
        self.ars_trial = 10
        self.last_td3_actor_loss = None
        self.last_td3_critic_loss = None
        self.last_td3_mean_reward = None
        self.last_td3_updates = 0
        self.last_mafia_loss = None
        self.last_mafia_direction_loss = None
        self.last_mafia_eta_loss = None
        self.last_mafia_eta_mae = None
        self.last_mafia_pg_loss = None
        self.last_mafia_eta_pred = 1.0
        self.last_mafia_dir_pred = "FLAT"
        self.last_mafia_dir_conf = 0.0

        if current_date is None:
            self.cur_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            self.cur_datetime = current_date
        res_root = os.path.abspath(os.environ.get("MAFIA_RES_ROOT", "./res"))
        self.res_root = res_root
        self.res_dir = os.path.join(
            res_root,
            self.mode,
            self.rl_model_name,
            "{}-{}".format(self.market_name, self.topK),
            self.cur_datetime,
        )
        self.res_model_dir = os.path.join(self.res_dir, "model")
        self.res_img_dir = os.path.join(self.res_dir, "graph")
        self.metrics_history_path = os.path.join(self.res_dir, "metrics_history.csv")
        self.run_manifest_path = os.path.join(self.res_dir, "run_manifest.json")

        # Checkpoint configuration
        self.checkpoint_dir = os.path.join(self.res_dir, "checkpoints")

        # Only create directories if requested (avoid creating unwanted TD3 dirs during observer training)
        if create_dirs:
            os.makedirs(self.res_dir, exist_ok=True)
            os.makedirs(self.res_model_dir, exist_ok=True)
            os.makedirs(self.res_img_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_freq = 1  # Save checkpoint every N epochs (0 to disable)
        self.validation_freq = (
            1  # Run validation every N epochs (test runs only at final epoch)
        )
        # Control whether to auto-generate Top-K CSVs during train/valid/test. Keep False to run Top-K separately.
        self.enable_topk_postprocess = False
        # Step-based checkpointing (0 to disable, >0 saves every N timesteps)
        # Recommended: 500-1000 for frequent saves, or 0 to disable
        # Checkpoint/save configuration
        self.partial_checkpoint_steps = (
            0  # Disable step-based checkpoints; only save per epoch/best_valid
        )
        self.resume_from_checkpoint = None  # Path to checkpoint to resume from (None to start fresh or use auto_resume)
        self.auto_resume_from_latest = True  # Auto-resume from latest checkpoint if exists (when resume_from_checkpoint is None)
        self.enable_checkpoint_cleanup = (
            True  # Automatically delete old checkpoints to control disk usage
        )
        self.max_checkpoints_to_keep = (
            1  # Applies to both epoch and step checkpoints when cleanup enabled
        )
        # On resume: only load model weights/observer; skip replay buffer to avoid stale/overlapping samples
        self.reset_replay_buffer_on_resume = False

        # Walk-forward buffer filtering: filter buffer to keep only experiences
        # within the new window's train period instead of full reset.
        # This retains overlapping experiences (e.g., with step_years=1, 2/3 of experiences
        # from previous window may still be valid for new window's train period).
        # Set to True to enable smart filtering; False to use reset_replay_buffer_on_resume behavior.
        self.filter_replay_buffer_on_resume = True

        # Priority sampling for regime shift experiences in replay buffer.
        # Fraction of each batch to sample from regime shift experiences (if available).
        # Default 0.20 = 20% of batch. Set to 0 to disable priority sampling.
        # This prevents "catastrophic forgetting" of rare crisis patterns.
        self.regime_shift_sample_ratio = 0.20

        # Skip observer pretrain when resuming from checkpoint (walk-forward chaining).
        # When True: Observer already has learned weights from previous window, skip warmup.
        # When False: Always run pretrain epochs even when resuming.
        self.skip_pretrain_on_resume = True

        # Reset LR scheduler when resuming from checkpoint for new window.
        # When True: LR starts fresh from initial learning rate (recommended for walk-forward).
        # When False: LR continues from where it left off in previous window.
        self.reset_lr_scheduler_on_resume = True

        self.save_replay_buffer_on_step_checkpoints = (
            False  # Skip 3GB+ replay buffer for frequent step checkpoints
        )
        self.save_replay_buffer_on_epoch_checkpoints = (
            True  # Save replay buffer on epoch checkpoints (uses float16 compression)
        )
        # Early stopping (validation-based): stop if no improvement for N epochs
        self.early_stop_patience = 10  # Stop after 10 epochs without improvement
        self.early_stop_min_delta = 0.01  # Minimum improvement threshold
        self.early_stop_warmup = 2  # Start checking after 2 epochs
        self.early_stop_metric = (
            "sharpeRatio"  # Metric to monitor (fallback to reward_sum if missing)
        )
        self.tradeDays_per_year = 252
        self.tradeDays_per_month = 21
        self.seed_num = seed_num
        self._market_risk_warned_insufficient = False

        # LiveDisplay configuration (real-time terminal updates)
        # Configured at top of file (defaults to True, override via MAFIA_USE_LIVE_DISPLAY)

        # Web Dashboard configuration (replaces terminal display for stable layout)
        # Set to True to enable web-based dashboard at http://localhost:5050
        # This provides a stable layout without terminal ANSI code issues
        self.use_web_dashboard = True  # Default disabled, set True to enable
        self.dashboard_port = 5050  # Web dashboard port
        self.dashboard_open_browser = True  # Auto-open browser when dashboard starts

        # TD3 Training Window Configuration (Phase 2)
        # Train: 2018-2022 (5 years single window)
        # Valid: 2023-2024 (2 years)
        # Test: 2025 (1 year - final evaluation)
        date_split_dict = {
            1: {
                "train_date_start": "2018-01-02 00:00:00",
                "train_date_end": "2022-12-30 23:59:59",
                "valid_date_start": "2023-01-02 00:00:00",
                "valid_date_end": "2024-12-31 23:59:59",
                "test_date_start": "2025-01-02 00:00:00",
                "test_date_end": "2025-12-31 23:59:59",
            },
        }

        self.train_date_start = pd.Timestamp(
            date_split_dict[self.period_mode]["train_date_start"]
        )
        self.train_date_end = pd.Timestamp(
            date_split_dict[self.period_mode]["train_date_end"]
        )
        if (date_split_dict[self.period_mode]["valid_date_start"] is not None) and (
            date_split_dict[self.period_mode]["valid_date_end"] is not None
        ):
            self.valid_date_start = pd.Timestamp(
                date_split_dict[self.period_mode]["valid_date_start"]
            )
            self.valid_date_end = pd.Timestamp(
                date_split_dict[self.period_mode]["valid_date_end"]
            )
        else:
            self.valid_date_start = None
            self.valid_date_end = None
        if (date_split_dict[self.period_mode]["test_date_start"] is not None) and (
            date_split_dict[self.period_mode]["test_date_end"] is not None
        ):
            self.test_date_start = pd.Timestamp(
                date_split_dict[self.period_mode]["test_date_start"]
            )
            self.test_date_end = pd.Timestamp(
                date_split_dict[self.period_mode]["test_date_end"]
            )
        else:
            self.test_date_start = None
            self.test_date_end = None

        # Use a fixed fallback for market risk (observer will emit dynamic bounds)
        self.risk_market = self.default_risk_market
        self._calibrate_risk_bounds()

        self.tech_indicator_talib_lst = ["SMA", "RSI", "ATR"]
        self.tech_indicator_extra_lst = ["CHANGE"]
        self.tech_indicator_input_lst = (
            self.tech_indicator_talib_lst + self.tech_indicator_extra_lst
        )
        self.dailyRetun_lookback = self.cov_lookback
        self.otherRef_indicator_ma_window = 5
        self.enable_cov_features = (
            True  # enable using the cov features in RL-based agent
        )

        self.otherRef_indicator_lst = [
            "MA-{}".format(self.otherRef_indicator_ma_window),
            "DAILYRETURNS-{}".format(self.dailyRetun_lookback),
        ]

        self.mkt_rf = {
            "VNINDEX": 3.0,  # Risk-free rate for Vietnam market (based on 10-year government bond yield, ~3.0% as of 2022). Adjust based on your data period.
        }

        self.market_close_time = {
            "VNINDEX": "15:00:00",  # Vietnam stock market close time
        }
        self.invest_env_para = {
            "max_shares": 100,
            "initial_asset": 1000000,
            "reward_scaling": self.reward_scaling,
            "norm_method": self.norm_method,
            "transaction_cost": 0.0003,
            "slippage": 0.001,
            "seed_num": self.seed_num,
        }

        # Legacy algorithm lists (kept for reference, not used)
        self.only_long_algo_lst = []  # Removed legacy algorithms
        self.use_cash_algo_lst = []  # Removed legacy algorithms
        # MAFIA always uses RLcontroller mode with controller enabled
        # (removed legacy branching that disabled controller)

        # MAFIA uses its own DC feature generation
        if self.mktobs_algo == "mafia_1":
            self.is_gen_dc_feat = False
        else:
            self.is_gen_dc_feat = False

        self.load_para()
        self.load_model_config()
        self.load_market_observer_config()

    def load_model_config(self):
        # Order matters for CHANGE features (Spec §II.2 Technical Agent)
        self.use_features = ["close", "open", "high", "low", "volume"]
        # Note: window_size removed - MAFIA uses mafia_T_w instead (on-the-fly computation)

    def load_market_observer_config(self):
        self.freq = "1d"
        self.finefreq = "1d"
        self.fine_window_size = 30
        self.feat_scaler = 10

        # MAFIA Hyperparameters (must be defined first as they're used below)
        self.mafia_T_w = 30  # Observation window size
        self.mafia_DC_thresholds = [0.005, 0.01, 0.02]  # DC thresholds for 3 DC agents
        self.mafia_D = 64  # Embedding dimension
        self.mafia_D_h = 128  # Hidden layer dimension
        self.mafia_encoder_layers = 2  # Number of transformer encoder layers
        self.mafia_encoder_heads = 4  # Number of attention heads
        self.mafia_M_tech = 8  # Features for Technical agent (5 OCHLV + 3 indicators)
        self.mafia_M_dc = 5  # Features for DC agents
        # Market-index agent features: only 19 base kênh (ΔOHLCV + indicator set), không dùng regime append
        self.mafia_M_mkt = 19
        # Regime features removed from market-index agent
        self.mafia_regime_feature_names = []
        self.mafia_regime_dim = 0
        self.mafia_learning_rate = 1e-4  # Learning rate for MAFIA training
        # Match TD3-style LR schedule for observer (linear decay)
        self.mafia_lr_schedule = (
            "linear_per_epoch"  # "linear_per_epoch", "linear", or "step"
        )
        self.mafia_lr_end_factor = 0.2  # end_lr = start_lr * end_factor
        self.mafia_lr_end_fraction = (
            0.5  # fraction of epoch/cycle where end_lr is reached
        )
        self.mafia_weight_decay = 0.003  # Weight decay for optimizer
        self.hidden_vec_loss_weight = (
            1.0  # Weight for market_vector loss in Policy Gradient training
        )
        self.market_direction_loss_weight = (
            1.0  # Weight for market direction classification loss
        )

        # ===== Offline Batch Training Configuration (Spec §6) =====
        # Training Duration per Iteration
        self.mafia_observer_base_epochs = 50  # Base training (iter 0) epochs
        self.mafia_observer_finetune_epochs = 5  # Finetune (iter > 0) epochs
        # Steps per epoch: auto-computed as ceil((Len(Data) - T_m - h) / Batch_Size)
        # No manual override needed; calculated dynamically per dataset

        # Trajectory Configuration (Spec §6)
        self.mafia_trajectory_length = 128  # T_m: trajectory length
        self.mafia_batch_size = 64  # B: batch size for random trajectory sampling
        self.mafia_sampling_strategy = (
            "random_trajectory"  # "random_trajectory" or "recent_trajectory"
        )
        self.mafia_use_class_balanced_sampling = (
            False  # Disabled: Focal Loss α handles class imbalance per Spec
        )

        # ===== Memory Optimization Configuration =====
        # Mixed Precision Training (FP16)
        # Reduces VRAM usage by ~50% with minimal accuracy impact
        self.use_mixed_precision = True  # Enable automatic mixed precision (AMP)

        # Gradient Accumulation
        # Accumulate gradients over N microbatches before optimizer step
        # Allows using smaller batch sizes while maintaining effective large batch training
        # Reduces RAM usage by ~70% (can use batch_size=8 instead of 32)
        # NOTE: Set to 1 for faster iteration. Increase to 4 if training is unstable.
        self.gradient_accumulation_steps = (
            1  # Effective batch = batch_size * accumulation_steps = 32
        )

        # Gradient Checkpointing
        # Trade compute for memory by recomputing activations during backward pass
        # Reduces VRAM for activations by ~30-40%
        self.use_gradient_checkpointing = (
            False  # Enable for Router LSTM and Expert Transformers
        )

        # ===== TensorBoard Configuration =====
        # Real-time visualization of training metrics via web dashboard
        self.use_tensorboard = True  # Enable/disable TensorBoard logging
        self.tensorboard_log_dir = "tensorboard"  # Subdirectory for TensorBoard logs
        self.tensorboard_log_images = True  # Log training charts as images
        self.tensorboard_log_histograms = True  # Log weight/gradient histograms
        self.tensorboard_histogram_freq = 10  # Log histograms every N batches

        # Loss Weights (Spec §6)
        self.mafia_lambda_pg = 1.0  # Weight for L_PG (policy gradient loss)
        self.mafia_lambda_risk = 0.3  # Weight for L_Risk (risk calibration loss)
        self.mafia_lambda_dir = 0.5  # Weight for L_Dir (direction classification loss)

        # Entropy Bonus for L_PG (spec 5.1.1)
        self.mafia_beta_entropy = 0.01  # β_ent: entropy bonus coefficient

        # Policy Gradient Reward Shaping
        self.mafia_pg_reward_horizon = (
            14  # h: lookahead horizon for reward accumulation (spec §5.1.1)
        )
        # At time t, when Observer selects a portfolio, h determines how many days forward
        # to accumulate returns for evaluating that decision
        # Stronger portfolio churn penalties (penalty ~25–30% reward at λ=1 with typical turnover/symdiff)
        # Old override removed to respect lines 177-178

        # SOFT LANDING for Resume at Epoch 7:
        # Start ramp-up from Epoch 6 (Index 6), so Warmup = 5.
        # Epoch 7: (7 - 5)/5 = 0.4 (40% penalty).
        # Turnover: 0.6 | Change: 0.8 (Higher than old 0.4/0.5)
        self.curriculum_warmup_epochs = 0  # Start ramp-up immediately (Epoch 0)

        self.curriculum_penalty_rampup = 5  # Rampup over 5 epochs

        # Risk/Direction Head Config
        self.mafia_explicit_dim = 6  # [Vol20, DC, Breadth, Div, VPI, DD60]
        self.direction_head_dropout = 0.2
        self.mafia_direction_threshold = (
            0.02  # δ: threshold for bull/bear classification
        )

        # Direction Loss (Focal Loss) Class Weights
        # Optimized based on VNINDEX ground truth distribution: Bear=22.1%, Side=43.8%, Bull=34.1%
        # α = [α_bear, α_side, α_bull] - balances gradient contribution across classes
        #
        # IMPORTANT: These weights compensate for class imbalance AND focal modulation
        # With γ=2.0, easy (majority) samples get ~0.1x gradient vs hard samples
        # Direction Loss (Focal Loss) Class Weights: [Bear, Side, Bull]
        # Based on VNINDEX distribution: Bear=22%, Side=44%, Bull=34%
        self.mafia_focal_alpha = [1.7, 0.70, 1.0]
        self.mafia_focal_gamma = (
            2.0  # Focusing parameter γ: reduces loss for confident (easy) predictions
        )
        # Label Smoothing (Spec 5.1.3): Converts [0,1,0] → [0.033, 0.933, 0.033]
        # Helps model converge stably, avoids overconfidence on noisy labels
        self.mafia_direction_label_smoothing = 0.05  # ε: smoothing factor (reduced)
        # Temperature scaling for direction logits (T<1 sharpens, T>1 flattens)
        self.mafia_direction_temperature = 1

        # Gradient Clipping
        self.mafia_max_grad_norm = 1.0  # Max gradient norm for clipping

        # Top-K Selection
        self.mafia_top_k = 10  # Number of assets to select

        # Rebalance settings
        self.mafia_topk_rebalance_interval = 14  # Rebalance every 14 days (User Intent)
        self.mafia_hard_topk_inference = True  # Use hard TopK at inference

        # Dense MoE Gating Configuration
        self.mafia_gating_encoder_type = (
            self.DEFAULT_GATING_ENCODER
        )  # Options: 'attention_based_aggregation', 'temporal_convolution', 'lstm'
        self.mafia_gating_num_heads = 4  # For attention-based encoder
        self.mafia_gating_dropout = 0.2  # Dropout for gating networks
        self.mafia_gating_lstm_layers = 2  # For lstm encoder
        self.mafia_gating_conv_kernels = [3, 5, 7]  # For temporal_convolution encoder

        # Temporal Context Augmentation for Gating Router (Spec 3.6)
        # Solves distribution shift: Router updated per rebalance, but market_context updated daily
        # Augmented input: C_aug = [C_mkt^(t), C_bar_mkt, Delta_C_mkt] ∈ R^{3D}
        self.router_context_window = int(
            os.environ.get("ROUTER_CONTEXT_WINDOW", 14)
        )  # Lookback window W for rolling mean/drift (align with topk_rebalance_interval)
        self.router_use_temporal_augmentation = (
            True  # Enable C_aug = [C, C_bar, Delta_C]
        )

        self.finestock_feat_cols_lst = []
        self.finemkt_feat_cols_lst = []
        # Fine stock windows keep using fine_window_size
        for ifeat in self.use_features:
            for iwin in range(1, self.fine_window_size + 1):
                self.finestock_feat_cols_lst.append(
                    "stock_{}_{}_w{}".format(self.finefreq, ifeat, iwin)
                )
        # Market index windows use mafia_T_w (e.g., 30)
        for ifeat in self.use_features:
            for iwin in range(
                1, self.mafia_T_w + 1 if hasattr(self, "mafia_T_w") else 31
            ):
                self.finemkt_feat_cols_lst.append(
                    "mkt_{}_{}_w{}".format(self.finefreq, ifeat, iwin)
                )

        # Add market technical indicators to fine market feature list
        self.finemkt_indicator_cols = [
            "mkt_{}_sma20".format(self.finefreq),
            "mkt_{}_rsi14".format(self.finefreq),
            "mkt_{}_atr14".format(self.finefreq),
            # Extended indicators (all causal, computed with past-only data)
            "mkt_{}_macd_hist".format(self.finefreq),
            "mkt_{}_bb_width".format(self.finefreq),
            "mkt_{}_stoch_k".format(self.finefreq),
            "mkt_{}_stoch_d".format(self.finefreq),
            "mkt_{}_adx14".format(self.finefreq),
            "mkt_{}_obv".format(self.finefreq),
            "mkt_{}_mfi14".format(self.finefreq),
            "mkt_{}_cci20".format(self.finefreq),
            "mkt_{}_vol_std20".format(self.finefreq),
            "mkt_{}_drawdown60".format(self.finefreq),
            "mkt_{}_regime_sma20_60".format(self.finefreq),
        ]
        for col in self.finemkt_indicator_cols:
            if col not in self.finemkt_feat_cols_lst:
                self.finemkt_feat_cols_lst.append(col)

        # Market index feature selection (DEPRECATED - not used in simplified state):
        # Previous architecture used these explicit features in the state
        # Now we rely on Observer's Market-Index Agent for learned representation
        # Kept for backward compatibility or future experimentation
        self.market_index_feature_names = []
        # ΔOHLCV windows for all OHLCV features, across mafia_T_w (30)
        for change_feat in ["open", "close", "high", "low", "volume"]:
            for iwin in range(1, self.mafia_T_w + 1):
                feat_name = "mkt_{}_{}_w{}".format(self.finefreq, change_feat, iwin)
                if feat_name in self.finemkt_feat_cols_lst:
                    self.market_index_feature_names.append(feat_name)
        # Basic indicators
        basic_indicators = [
            "mkt_{}_sma20".format(self.finefreq),
            "mkt_{}_rsi14".format(self.finefreq),
            "mkt_{}_atr14".format(self.finefreq),
        ]
        for col in basic_indicators:
            if col in self.finemkt_feat_cols_lst:
                self.market_index_feature_names.append(col)
        # Extended indicators (predefined in finemkt_indicator_cols)
        for col in self.finemkt_indicator_cols:
            if col in self.finemkt_feat_cols_lst:
                self.market_index_feature_names.append(col)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for name in self.market_index_feature_names:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        self.market_index_feature_names = deduped
        # DEPRECATED CODE BELOW - No longer used in state building
        # State architecture now only uses Observer's learned representation (market_vector/market_scores_full)
        # This dimension adjustment code is kept for backward compatibility only
        # Commented out as it's not needed for current simplified state architecture
        # if len(self.market_index_feature_names) > self.market_index_state_dim:
        #     self.market_index_feature_names = self.market_index_feature_names[:self.market_index_state_dim]
        # elif len(self.market_index_feature_names) < self.market_index_state_dim:
        #     # Fill with None placeholders to be handled downstream
        #     self.market_index_feature_names.extend([None] * (self.market_index_state_dim - len(self.market_index_feature_names)))

    def load_para(self):
        use_multibranch = (
            getattr(self, "rl_obs_use_multibranch_state", False)
            and self.enable_market_observer
        )

        # Select replay buffer implementation based on observation space (Dict vs Box)
        if use_multibranch:
            buffer_class = CompressedDictReplayBuffer
            # DictReplayBuffer does not support optimize_memory_usage
            optimize_memory_usage = False
        else:
            buffer_class = CompressedReplayBuffer
            optimize_memory_usage = self.optimize_memory_usage

        replay_buffer_kwargs = {
            "handle_timeout_termination": False,
            "compress_obs": self.compress_obs,
            "storage_dtype": self.replay_buffer_dtype,
            "regime_shift_sample_ratio": self.regime_shift_sample_ratio,
        }

        if self.enable_market_observer:
            if self.rl_model_name == "TD3":
                policy_name = MultiInputPolicy if use_multibranch else "TD3PolicyAdj"
            else:
                raise ValueError(
                    "Cannot specify the {} policy name when enabling market observer.".format(
                        self.rl_model_name
                    )
                )
        else:
            if self.rl_model_name == "TD3":
                from RL_controller.TD3_controller import TD3PolicyOriginal

                policy_name = TD3PolicyOriginal
            else:
                if self.mode in ["RLonly", "RLcontroller"]:
                    raise ValueError(
                        "Cannot specify the {} policy name when using stable-baseline.".format(
                            self.rl_model_name
                        )
                    )
                else:
                    policy_name = "MlpPolicy"
        start_lr = self.learning_rate
        lr_value = start_lr
        schedule_mode = getattr(self, "td3_lr_schedule", None)
        if schedule_mode in ("linear", "linear_per_epoch"):
            end_lr = start_lr * getattr(self, "td3_lr_end_factor", 0.2)
            frac = getattr(self, "td3_lr_end_fraction", 0.5)
            if schedule_mode == "linear":
                lr_value = LinearSchedule(start=start_lr, end=end_lr, end_fraction=frac)
            else:
                # Repeat linear decay every epoch instead of over the whole run.
                # progress_remaining goes from 1.0 -> 0.0 over the run; map it into per-epoch phase.
                total_epochs = max(1, getattr(self, "num_epochs", 1))

                def _per_epoch_linear(progress_remaining: float):
                    progress = 1.0 - progress_remaining  # 0..1 over full run
                    phase = (progress * total_epochs) % 1.0  # 0..1 within current epoch
                    if phase > frac:
                        return end_lr
                    return start_lr + (phase / frac) * (end_lr - start_lr)

                lr_value = _per_epoch_linear
        elif schedule_mode == "cyclic_run_decay":
            # Global decay across the full run AND per-epoch tail decay (no reset each epoch)
            end_factor = getattr(self, "td3_lr_end_factor", 0.2)
            frac = getattr(self, "td3_lr_end_fraction", 0.5)
            total_epochs = max(1, getattr(self, "num_epochs", 1))
            frac = max(1e-6, min(1.0, frac))

            def _cyclic_decay(progress_remaining: float) -> float:
                # progress: 0 at start -> 1 at end of run
                progress = 1.0 - progress_remaining
                # Global decay over the whole run
                global_scale = 1.0 - (1.0 - end_factor) * progress
                # Epoch-phase decay near the tail of each epoch
                epoch_phase = (
                    progress * total_epochs
                ) % 1.0  # 0..1 within current epoch
                tail_phase = min(epoch_phase / frac, 1.0)
                epoch_scale = 1.0 - (1.0 - end_factor) * tail_phase
                lr_curr = start_lr * max(0.0, global_scale * epoch_scale)
                return lr_curr

            lr_value = _cyclic_decay

        base_para = {
            "policy": policy_name,
            "learning_rate": lr_value,
            "buffer_size": int(1.56 * 1e5),
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (self.train_freq[0], self.train_freq[1]),
            "verbose": 1,
            "gradient_steps": self.gradient_steps,
            "action_noise": None,
            "replay_buffer_class": buffer_class,
            "replay_buffer_kwargs": replay_buffer_kwargs,
            "optimize_memory_usage": optimize_memory_usage,
            "entropy_coef": self.entropy_coef,
            "tensorboard_log": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "tb_logs"
            ),
            "policy_kwargs": None,
            "seed": self.seed_num,
            "device": "auto",
            "_init_setup_model": True,
        }
        algo_para = {
            "TD3": {
                "policy_delay": 2,
                "target_policy_noise": 0.15,
                "target_noise_clip": 0.5,
            },
            "SAC": {
                "ent_coef": "auto",
                "target_update_interval": 1,
                "target_entropy": "auto",
                "use_sde": False,
                "sde_sample_freq": -1,
                "use_sde_at_warmup": False,
            },
            "PPO": {"n_steps": 100},
        }
        if (self.topK == 20) or (self.topK == 30):
            algo_para["TD3"]["policy_kwargs"] = {
                "net_arch": [1024, 512, 128],  # [400, 300]
            }
        algo_para_rm_from_base = {
            "PPO": [
                "buffer_size",
                "learning_starts",
                "tau",
                "train_freq",
                "gradient_steps",
                "action_noise",
                "replay_buffer_class",
                "replay_buffer_kwargs",
                "optimize_memory_usage",
            ]
        }
        if self.rl_model_name in algo_para.keys():
            self.model_para = {**base_para, **algo_para[self.rl_model_name]}
        else:
            self.model_para = base_para
        # Inject multi-input policy kwargs when using observer+dict obs
        if use_multibranch and self.rl_model_name == "TD3":
            extractor_kwargs = {"config": self}
            from RL_controller.feature_extractors import (
                MAFIAMultiBranchExtractor as MAFIASingleStageMLP,
            )

            mi_policy_kwargs = {
                "features_extractor_class": MAFIASingleStageMLP,
                "features_extractor_kwargs": extractor_kwargs,
            }
            if (
                "policy_kwargs" in self.model_para
                and self.model_para["policy_kwargs"] is not None
            ):
                merged = {**self.model_para["policy_kwargs"], **mi_policy_kwargs}
                self.model_para["policy_kwargs"] = merged
            else:
                self.model_para["policy_kwargs"] = mi_policy_kwargs
        if self.rl_model_name in algo_para_rm_from_base.keys():
            for rm_field in algo_para_rm_from_base[self.rl_model_name]:
                del self.model_para[rm_field]

    def configure_gating_mode(self, gating_mode: str):
        """Update the gating encoder using a friendly alias (attentive/cnn/lstm)."""
        if gating_mode is None:
            return self.mafia_gating_encoder_type

        alias = gating_mode.lower().strip()
        if alias not in self.GATING_MODE_MAP:
            valid = ", ".join(self.GATING_MODE_MAP.keys())
            raise ValueError(
                f"Unsupported gating mode '{gating_mode}'. Valid options: {valid}."
            )

        encoder_type = self.GATING_MODE_MAP[alias]
        self.mafia_gating_encoder_type = encoder_type
        return encoder_type

    @classmethod
    def iter_gating_modes(cls):
        """Yield (alias, encoder_name, description) tuples for the three supported modes."""
        for alias, encoder in cls.GATING_MODE_MAP.items():
            description = cls.GATING_MODE_DESCRIPTIONS.get(alias, "")
            yield alias, encoder, description

    def _compute_market_risk(self, cutoff_date=None):
        """
        Estimate market-wide risk (sigma_beta) from benchmark index daily returns
        using the latest self.cov_lookback window. Falls back to default value if
        data is missing or invalid.

        Args:
            cutoff_date: Optional datetime-like upper bound for index data (ex-ante). If None,
                         falls back to self.train_date_end when available.
        """
        fallback = getattr(self, "default_risk_market", 0.001)
        index_file = self.index_data_file
        if index_file is None:
            return fallback

        # Resolve candidate paths
        candidate_paths = []
        if os.path.isabs(index_file):
            candidate_paths.append(index_file)
        else:
            # Primary: relative to configured data directory
            candidate_paths.append(os.path.join(self.dataDir, index_file))
            # Secondary: alongside this config file (agents/MAFIA/data)
            config_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_paths.append(os.path.join(config_dir, "data", index_file))

        data_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                data_path = path
                break

        if data_path is None:
            print(
                f"[Config] Warning: Cannot locate index data file {index_file}, using default risk_market={fallback}",
                flush=True,
            )
            return fallback

        try:
            index_df = pd.read_csv(data_path)
            if "close" not in index_df.columns:
                raise ValueError("close column missing")
            index_df = index_df.dropna(subset=["close"])
            if "date" in index_df.columns:
                index_df["date"] = pd.to_datetime(index_df["date"], errors="coerce")
                index_df = index_df.sort_values("date")
                effective_cutoff = (
                    cutoff_date
                    if cutoff_date is not None
                    else getattr(self, "train_date_end", None)
                )
                if effective_cutoff is not None:
                    cutoff_ts = pd.to_datetime(effective_cutoff)
                    # Normalize timezone information to avoid comparison errors
                    try:
                        index_df["date"] = index_df["date"].dt.tz_localize(None)
                    except Exception:
                        pass
                    try:
                        cutoff_ts = cutoff_ts.tz_localize(None)
                    except Exception:
                        pass
                    index_df = index_df[index_df["date"] <= cutoff_ts]
            if len(index_df) < 2:
                raise ValueError("Not enough rows in index data after filtering")
            closes = pd.to_numeric(index_df["close"], errors="coerce").dropna()
            if len(closes) < 2:
                raise ValueError("Not enough close prices for returns")
            returns = closes.pct_change(fill_method=None).dropna()
            if len(returns) == 0:
                raise ValueError("Empty returns series")
            window = int(max(2, self.cov_lookback))
            window = min(window, len(returns))
            if window < 2:
                raise ValueError("Not enough returns to compute std (window < 2)")
            recent_returns = returns.iloc[-window:]
            market_risk = recent_returns.std(ddof=1)
            if np.isnan(market_risk) or np.isinf(market_risk):
                raise ValueError("Invalid market risk value")
            return float(market_risk)
        except Exception as e:
            # Avoid spamming when the only issue is insufficient history at very early dates
            msg = f"[Config] Warning: Failed to compute market risk from {data_path}: {e}. Using default {fallback}"
            if "Not enough" in str(e) or "Empty returns" in str(e):
                if not getattr(self, "_market_risk_warned_insufficient", False):
                    print(msg, flush=True)
                    self._market_risk_warned_insufficient = True
            else:
                print(msg, flush=True)
            return fallback

    def _calibrate_risk_bounds(self):
        """
        Ensure risk tolerance factors remain ordered and positive.
        """
        eps = 1e-6
        self.risk_default = max(self.risk_default, eps)
        self.risk_hold_bound = max(self.risk_hold_bound, eps)
        self.risk_up_bound = max(self.risk_up_bound, eps)
        self.risk_down_bound = max(self.risk_down_bound, eps)

        # Enforce ordering: down <= hold <= up, and default within [down, up]
        if self.risk_down_bound > self.risk_hold_bound:
            self.risk_down_bound = self.risk_hold_bound
        if self.risk_up_bound < self.risk_hold_bound:
            self.risk_up_bound = self.risk_hold_bound
        self.risk_default = min(
            max(self.risk_default, self.risk_down_bound), self.risk_up_bound
        )

        # Keep eta aliases in sync
        self.risk_eta_default = self.risk_default
        self.risk_eta_hold = self.risk_hold_bound
        self.risk_eta_up = self.risk_up_bound
        self.risk_eta_down = self.risk_down_bound

    def rebuild_result_paths(self, new_res_dir: str, create_dirs: bool = True):
        """
        Rebuild all result-related paths when res_dir changes.

        This is necessary when redirecting outputs to a different directory
        (e.g., for Observer walk-forward training vs TD3 training).

        Args:
            new_res_dir: New base directory for results
            create_dirs: Whether to create directories (default: True)

        Example:
            config.rebuild_result_paths("./observer_walkforward/iter_0_valid_2017")
            # This updates:
            # - res_dir, res_root
            # - res_model_dir, res_img_dir, checkpoint_dir
            # - metrics_history_path, run_manifest_path
        """
        self.res_dir = os.path.abspath(new_res_dir)
        self.res_root = self.res_dir

        # Rebuild subdirectories
        self.res_model_dir = os.path.join(self.res_dir, "model")
        self.res_img_dir = os.path.join(self.res_dir, "graph")
        self.checkpoint_dir = os.path.join(self.res_dir, "checkpoints")

        # Rebuild file paths
        self.metrics_history_path = os.path.join(self.res_dir, "metrics_history.csv")
        self.run_manifest_path = os.path.join(self.res_dir, "run_manifest.json")

        if create_dirs:
            os.makedirs(self.res_dir, exist_ok=True)
            os.makedirs(self.res_model_dir, exist_ok=True)
            os.makedirs(self.res_img_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        return self

    def print_config(self):
        log_str = "=" * 30 + "\n"
        para_str = "{} \n".format(self.notes)
        log_str = log_str + para_str
        para_str = "mode: {}, rl_model_name: {}, market_name: {}, topK: {}, dataDir: {}, enable_controller: {}, \n".format(
            self.mode,
            self.rl_model_name,
            self.market_name,
            self.topK,
            self.dataDir,
            self.enable_controller,
        )
        log_str = log_str + para_str
        para_str = "trade_pattern: {} \n".format(self.trade_pattern)
        log_str = log_str + para_str
        para_str = "period_mode: {}, num_epochs: {}, cov_lookback: {}, norm_method: {}, benchmark_algo: {}, trained_best_model_type: {}, pricePredModel: {}, \n".format(
            self.period_mode,
            self.num_epochs,
            self.cov_lookback,
            self.norm_method,
            self.benchmark_algo,
            self.trained_best_model_type,
            self.pricePredModel,
        )
        log_str = log_str + para_str
        para_str = "is_enable_dynamic_risk_bound: {}, risk_market: {}, risk_default: {}, cbf_gamma: {}, ars_trial: {} \n".format(
            self.is_enable_dynamic_risk_bound,
            self.risk_market,
            self.risk_default,
            self.cbf_gamma,
            self.ars_trial,
        )
        log_str = log_str + para_str
        para_str = "cur_datetime: {}, res_dir: {}, tradeDays_per_year: {}, tradeDays_per_month: {}, seed_num: {}, \n".format(
            self.cur_datetime,
            self.res_dir,
            self.tradeDays_per_year,
            self.tradeDays_per_month,
            self.seed_num,
        )
        log_str = log_str + para_str
        para_str = "train_date_start: {}, train_date_end: {}, valid_date_start: {}, valid_date_end: {}, test_date_start: {}, test_date_end: {}, \n".format(
            self.train_date_start,
            self.train_date_end,
            self.valid_date_start,
            self.valid_date_end,
            self.test_date_start,
            self.test_date_end,
        )
        log_str = log_str + para_str
        para_str = "tech_indicator_input_lst: {}, \n".format(
            self.tech_indicator_input_lst
        )
        log_str = log_str + para_str
        para_str = "otherRef_indicator_lst: {}, enable_cov_features: {} \n".format(
            self.otherRef_indicator_lst, self.enable_cov_features
        )
        log_str = log_str + para_str
        para_str = "tmp_name: {}, mkt_rf: {} \n".format(self.tmp_name, self.mkt_rf)
        log_str = log_str + para_str
        para_str = "invest_env_para: {}, \n".format(self.invest_env_para)
        log_str = log_str + para_str
        para_str = "model_para: {}, \n".format(self.model_para)
        log_str = log_str + para_str
        para_str = "only_long_algo_lst: {}, \n".format(self.only_long_algo_lst)
        log_str = log_str + para_str
        para_str = "checkpoint_freq: {}, partial_checkpoint_steps: {}, \n".format(
            self.checkpoint_freq, self.partial_checkpoint_steps
        )
        log_str = log_str + para_str
        para_str = "lstr para: use_features: {}, freq: {}, finefreq: {}, fine_window_size: {}, mafia_T_w: {}\n".format(
            self.use_features,
            self.freq,
            self.finefreq,
            self.fine_window_size,
            self.mafia_T_w,
        )
        log_str = log_str + para_str
        para_str = (
            "enable_market_observer: {}, mktobs_algo: {}, feat_scaler: {} \n".format(
                self.enable_market_observer, self.mktobs_algo, self.feat_scaler
            )
        )
        log_str = log_str + para_str
        log_str = log_str + "=" * 30 + "\n"

        print(log_str, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect or update the MAFIA gating mode configuration."
    )
    parser.add_argument(
        "--gating-mode",
        choices=list(Config.GATING_MODE_MAP.keys()),
        default=None,
        help="Select one of the supported gating modes (attentive, cnn, lstm).",
    )
    args = parser.parse_args()

    print("Available gating modes:", flush=True)
    for alias, encoder_name, description in Config.iter_gating_modes():
        print(f" - {alias:<9} -> {encoder_name}: {description}", flush=True)

    if args.gating_mode is not None:
        encoder = Config.GATING_MODE_MAP[args.gating_mode]
        print(
            f"[Config] Selected gating mode '{args.gating_mode}' maps to encoder '{encoder}'.",
            flush=True,
        )
        print(
            "Update 'mafia_gating_encoder_type' or call Config.configure_gating_mode() with this alias in your pipeline.",
            flush=True,
        )
    else:
        print(
            f"[Config] Default gating mode is '{Config.DEFAULT_GATING_MODE_ALIAS}' "
            f"({Config.DEFAULT_GATING_ENCODER}).",
            flush=True,
        )
