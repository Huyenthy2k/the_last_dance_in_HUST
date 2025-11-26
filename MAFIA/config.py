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
from RL_controller.TD3_controller import TD3PolicyOriginal
from RL_controller.feature_extractors import MAFIAMultiModalExtractor
from stable_baselines3.common.utils import LinearSchedule


class Config:
    GATING_MODE_MAP = {
        "attentive": "attention_based_aggregation",
        "cnn": "temporal_convolution",
        "lstm": "bidirectional_lstm",
    }
    DEFAULT_GATING_MODE_ALIAS = "cnn"
    DEFAULT_GATING_ENCODER = GATING_MODE_MAP[DEFAULT_GATING_MODE_ALIAS]
    GATING_MODE_DESCRIPTIONS = {
        "attentive": "Self-attention gating that weighs temporal embeddings via multi-head attention.",
        "cnn": "Temporal convolution (CNN) gating that extracts short-term patterns before routing.",
        "lstm": "Bidirectional LSTM gating that captures sequential dependencies.",
    }

    def __init__(self, seed_num=2022, current_date=None):
        self.notes = "MAFIA Implementation - MAFIA-only (Legacy models removed)"

        # MAFIA-only configuration (Legacy TD3-only and old MASA variants removed)
        self.benchmark_algo = "TD3-PR"  # TD3 Profit-Risk optimization
        self.market_name = "VNINDEX"  # Financial Index: 'DJIA', 'SP500', 'CSI300'
        self.topK = 10  # Number of assets in a portfolio (10, 20, 30)
        self.num_epochs = 100  # episodes for convergence

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
        self.mafia_gumbel_temperature = 1.0
        self.mafia_hard_topk_inference = True  # use hard Top-K at eval
        self.mafia_include_risk_boundary_in_state = True
        # MAFIA state mode: 'compact' (Top-K market_vector) or 'full-score' (full market_scores_full)
        # 'compact': Observer chọn Top-K → State có Top-K → RL tự động nhận Top-K từ Observer (state)
        # 'full-score': Observer đưa ra Full N stocks → State có market_scores_full → RL tự động tự quyết Top-K từ market_scores_full
        self.mafia_state_mode = (
            "full-score"  # Pass full market scores directly to RL agent
        )

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

        # Solver boost behavior (only active in 'full-score' mode when RL has selected Top-K)
        # In 'full-score' mode: Solver can boost stocks already selected by RL, or keep original logic
        self.mafia_solver_boost_enabled = False  # If True, Solver boosts stocks selected by RL (only in 'full-score' mode)
        self.mafia_solver_boost_method = (
            "blend"  # Options: 'blend' or 'proportional' (same as boost methods)
        )
        self.mafia_solver_boost_factor = 0.3  # Weight for blending/boosting (0.0-1.0)

        # Market-index Agent and Self-Attention configuration
        self.mafia_use_market_index_agent = (
            True  # If True, enable Market-index agent (VNINDEX)
        )
        self.mafia_attention_agg = "weighted"  # Aggregation method for ST-Fusion embeddings in attention: 'mean', 'weighted', 'max'

        self.trade_pattern = 1  # 1: Long only, 2: Long and short (Not applicable), 3: short only (Not applicable)
        # Reward weights (tuned via quick Optuna on mini window)
        self.lambda_1 = 800  # return reward weight
        self.lambda_2 = 22.001620711836445  # JS penalty weight (controller adherence)
        # Encourage diversified actions (entropy regularizer on policy output)
        self.entropy_coef = 0.0013865911407901592
        self.controller_reg_lambda = (
            1.0  # λ_reg: controller regularization weight ||x - a_RL||^2
        )
        self.controller_observer_bias_weight = (
            0.3  # α: scales observer signal when forming linear bias q
        )
        # TD3 learning-rate schedule: 'linear', 'linear_per_epoch', or None
        self.td3_lr_schedule = "linear_per_epoch"
        self.td3_lr_end_factor = 0.2  # end_lr = start_lr * end_factor
        self.td3_lr_end_fraction = (
            0.5  # fraction of each epoch where end_lr reached (for linear_per_epoch)
        )
        # Turnover and membership-change penalties (turnover uses raw sum |w_t - w_{t-1}| )
        self.lambda_tc = 0.05
        self.lambda_change = (
            0.05  # Penalty weight for membership change (Top-K symmetric difference)
        )
        # Debug: log reward components for first N train steps (0 = disable)
        self.reward_debug_steps = 20
        self.train_freq = [1, "step"]  # Update every trading step
        self.risk_default = 0.015
        self.risk_up_bound = 0.025  # bull market
        self.risk_down_bound = 0.009  # bear market
        self.risk_hold_bound = 0.013  # sideways

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
        # Example: self.stock_data_file = 'stock_prices_all_20251108_234851.csv'
        self.stock_data_file = (
            "stock_prices_all_20251108_234851.csv"  # Set to None for auto-detection
        )
        self.index_data_file = "VNINDEX_1d_index.csv"  # Optional: 'DJIA_1d_index.csv' or None. If None, market features will be generated from stock data
        self.pricePredModel = "MA"
        self.cov_lookback = 30
        self.norm_method = "sum"
        self.max_zero_volume_days = (
            100  # Drop stocks with > this number of zero-volume days
        )
        self.rebalance_interval = 1  # Days between portfolio rebalances

        if self.mode == "Benchmark":
            self.trained_best_model_type = "max_capital"
        if self.mode == "RLonly":
            if self.trained_best_model_type not in ["max_capital", "js_loss"]:
                raise ValueError(
                    "The trained_best_model_type[{}] of {} should be in ['max_capital', 'js_loss'].".format(
                        self.trained_best_model_type, self.mode
                    )
                )

        self.default_risk_market = (
            0.001  # Default fallback for market risk (\Sigma_beta)
        )
        self.cbf_gamma = 0.7
        # Observer mini-epochs: train observer more frequently and reset its buffers to save memory
        self.observer_mini_epoch_steps = (
            252  # Set to 0 to disable mid-epoch observer training
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
        os.makedirs(self.res_dir, exist_ok=True)
        self.res_model_dir = os.path.join(self.res_dir, "model")
        os.makedirs(self.res_model_dir, exist_ok=True)
        self.res_img_dir = os.path.join(self.res_dir, "graph")
        os.makedirs(self.res_img_dir, exist_ok=True)
        self.metrics_history_path = os.path.join(self.res_dir, "metrics_history.csv")
        self.run_manifest_path = os.path.join(self.res_dir, "run_manifest.json")

        # Checkpoint configuration
        self.checkpoint_dir = os.path.join(self.res_dir, "checkpoints")
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
        # Early stopping based on validation Sharpe (patience in epochs)
        self.early_stop_patience = None
        self.resume_from_checkpoint = None  # Path to checkpoint to resume from (None to start fresh or use auto_resume)
        self.auto_resume_from_latest = True  # Auto-resume from latest checkpoint if exists (when resume_from_checkpoint is None)
        self.enable_checkpoint_cleanup = (
            True  # Automatically delete old checkpoints to control disk usage
        )
        self.max_checkpoints_to_keep = (
            1  # Applies to both epoch and step checkpoints when cleanup enabled
        )
        self.save_replay_buffer_on_step_checkpoints = (
            False  # Skip 3GB+ replay buffer for frequent step checkpoints
        )
        self.save_replay_buffer_on_epoch_checkpoints = (
            True  # Save replay buffer on epoch checkpoints (uses float16 compression)
        )
        self.tradeDays_per_year = 252
        self.tradeDays_per_month = 21
        self.seed_num = seed_num
        self._market_risk_warned_insufficient = False
        date_split_dict = {
            1: {
                "train_date_start": "2017-01-03 00:00:00",
                "train_date_end": "2021-12-31 23:59:59",
                "valid_date_start": "2022-01-01 00:00:00",
                "valid_date_end": "2023-12-31 23:59:59",
                "test_date_start": "2024-01-01 00:00:00",
                "test_date_end": "2025-11-06 23:59:59",
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

        # Compute market risk using data available up to the end of the training period to avoid look-ahead bias
        self.risk_market = self._compute_market_risk(cutoff_date=self.train_date_end)
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
        self.mafia_M_mkt = 19  # Features for Market-index agent (5 change + 3 basic + 11 extended indicators)
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

        # Dense MoE Gating Configuration
        self.mafia_gating_encoder_type = self.DEFAULT_GATING_ENCODER  # Options: 'attention_based_aggregation', 'temporal_convolution', 'bidirectional_lstm'
        self.mafia_gating_num_heads = 4  # For attention-based encoder
        self.mafia_gating_dropout = 0.2  # Dropout for gating networks
        self.mafia_gating_lstm_layers = 2  # For bidirectional_lstm encoder
        self.mafia_gating_conv_kernels = [3, 5, 7]  # For temporal_convolution encoder

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

        base_para = {
            "policy": policy_name,
            "learning_rate": lr_value,
            "buffer_size": int(2.6 * 1e5),
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
            "tensorboard_log": "./tb_logs",
            "policy_kwargs": None,
            "seed": self.seed_num,
            "device": "auto",
            "_init_setup_model": True,
        }
        algo_para = {
            "TD3": {
                "policy_delay": 2,
                "target_policy_noise": 0.2,
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
            mi_policy_kwargs = {
                "features_extractor_class": MAFIAMultiModalExtractor,
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
            returns = closes.pct_change().dropna()
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
        Ensure risk boundaries remain ordered relative to dynamic market risk.
        Automatically nudges values instead of raising hard errors.
        """
        margin = max(0.001, self.risk_market * 0.15)

        if self.risk_default <= self.risk_market:
            new_value = self.risk_market + margin
            print(
                f"[Config] Adjusting risk_default from {self.risk_default} to {new_value} (market risk={self.risk_market})",
                flush=True,
            )
            self.risk_default = new_value

        if self.risk_hold_bound >= self.risk_default:
            new_value = max(self.risk_market, self.risk_default - margin * 0.5)
            print(
                f"[Config] Adjusting risk_hold_bound from {self.risk_hold_bound} to {new_value} to keep below risk_default",
                flush=True,
            )
            self.risk_hold_bound = new_value

        if self.risk_down_bound >= self.risk_hold_bound:
            new_value = max(self.risk_market * 0.8, self.risk_hold_bound - margin * 0.5)
            print(
                f"[Config] Adjusting risk_down_bound from {self.risk_down_bound} to {new_value} to keep ordering",
                flush=True,
            )
            self.risk_down_bound = new_value

        if self.risk_up_bound <= self.risk_default:
            new_value = self.risk_default + margin * 0.5
            print(
                f"[Config] Adjusting risk_up_bound from {self.risk_up_bound} to {new_value} (above risk_default)",
                flush=True,
            )
            self.risk_up_bound = new_value

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
