#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observer Offline Batch Trainer - Implements Spec §7 Inner Training Loop

This module implements the offline batch training mode for MAFIA Observer
following the "Collect → Train → Discard" pattern described in the spec.

Key Features per Spec:
1. Random trajectory sampling: Choose random t_s from available data
2. Fresh hidden state: Reset BiLSTM hidden state at start of each batch
3. Sequential forward pass: Maintain hidden state within trajectory
4. Cadence-aware loss masking: L_PG only on rebalance days
5. Transient buffer: Discard after each training step

Spec Reference: refactor_mafia.md §7 "Observer Inner Training Loop"

Usage:
    trainer = ObserverOfflineBatchTrainer(config, observer, data_loader)
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_data)
        valid_metrics = trainer.validate_epoch(valid_data)
"""

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import smart_print
except ImportError:
    smart_print = print  # Fallback


@dataclass
class TrajectoryBatch:
    """Container for a batch of trajectories."""

    # Raw OCHLV data: (B, T_m, N, 5) - Batch, Time, Stocks, Features
    stock_ochlv: th.Tensor

    # Market index OCHLV: (B, T_m, 1, 5)
    market_ochlv: Optional[th.Tensor]

    # Price returns for reward calculation: (B, T_m, N)
    price_returns: th.Tensor

    # Market returns for baseline: (B, T_m)
    market_returns: th.Tensor

    # Rebalance mask: (B, T_m) - 1 if rebalance day, 0 if holding
    rebalance_mask: th.Tensor

    # Direction labels (future-based): (B, T_m) - 0=bear, 1=side, 2=bull
    direction_labels: th.Tensor

    # Risk targets (future volatility): (B, T_m)
    risk_targets: th.Tensor

    # Trajectory start indices for debugging: (B,)
    start_indices: th.Tensor

    # Dates for each step: (B, T_m)
    dates: Optional[np.ndarray] = None

    # Stock symbols list for display: List[str] of length N
    stock_list: Optional[list] = None

    # Volatility Shock Mask: (B, T_m) - 1 if vol shock triggered, 0 otherwise
    vol_shock_mask: Optional[th.Tensor] = None

    # Structural Break (DC) Mask: (B, T_m) - 1 if DC Event confimed, 0 otherwise
    dc_mask: Optional[th.Tensor] = None


from RL_controller.mafia_modules import MAFIAModel

# =============================================================================
# Focal Loss (Spec 5.1.3)
# =============================================================================
class FocalLoss(th.nn.Module):
    """
    Multi-class Focal Loss for handling class imbalance.
    L = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, gamma=2.0, alpha=None, device=None, reduction='mean', label_smoothing=0.0):
        """
        Args:
            gamma (float): Focusing parameter. Default 2.0.
            alpha (list/float/tensor): Balance factor for each class.
                                       If list, must have len=num_classes.
            device: Device to place alpha tensor on.
            reduction (str): 'mean', 'sum' or 'none'.
            label_smoothing (float): Label smoothing factor (Spec 5.1.3). Default 0.0.
                                     Converts [0,1,0] → [ε/C, 1-ε+ε/C, ε/C] where C=3.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = th.tensor(alpha, dtype=th.float32)
            elif isinstance(alpha, (float, int)):
                self.alpha = th.tensor([alpha], dtype=th.float32)
            else:
                self.alpha = alpha
            
            if device:
                self.alpha = self.alpha.to(device)
        else:
            self.alpha = None

    def forward(self, input, target):
        """
        Args:
            input: Logits (N, C) or (N, C, ...)
            target: Ground truth labels (N) or (N, ...)
        """
        # Ensure alpha is on correct device if not already
        if self.alpha is not None and self.alpha.device != input.device:
            self.alpha = self.alpha.to(input.device)

        # CrossEntropyLoss computes log_softmax internally
        # Label smoothing (Spec 5.1.3): [0,1,0] → [ε/C, 1-ε+ε/C, ε/C]
        ce_loss = th.nn.functional.cross_entropy(
            input, target, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = th.exp(-ce_loss)  # probability of correct class
        
        # Focal component: (1-pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            # Gather alpha for target classes
            # Support multi-dimensional target (flatten first)
            if target.dim() > 1:
                target_flat = target.view(-1)
                losses_flat = focal_loss.view(-1)
                alpha_t = self.alpha[target_flat]
                focal_loss = losses_flat * alpha_t
                focal_loss = focal_loss.view_as(target)
            else:
                alpha_t = self.alpha[target]
                focal_loss = focal_loss * alpha_t

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ObserverOfflineBatchTrainer:
    """
    Offline batch trainer for MAFIA Observer.

    Implements the Collect → Train → Discard loop per Spec §7.
    """

    def __init__(
        self,
        config,
        observer,  # MAFIAObserver instance
        device: Optional[th.device] = None,
        tensorboard_logger=None,  # Optional TensorBoard logger
    ):
        """
        Initialize offline trainer.

        Args:
            config: Configuration object with training params
            observer: MAFIAObserver instance (contains mafia_model)
            device: Torch device
            tensorboard_logger: Optional TensorBoard logger for real-time visualization
        """
        self.config = config
        self.observer = observer
        # Device selection: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if device is not None:
            self.device = device
        elif th.backends.mps.is_available():
            self.device = th.device("mps")
        elif th.cuda.is_available():
            self.device = th.device("cuda")
        else:
            self.device = th.device("cpu")
        self.optimizer = observer.optimizer # Use observer's optimizer
        self.tb_logger = tensorboard_logger  # Store TensorBoard logger

        # Training parameters from config
        self.T_m = int(
            getattr(config, "mafia_trajectory_length", 128)
        )  # Trajectory length
        self.batch_size = int(getattr(config, "mafia_batch_size", 32))  # Batch size B
        self.T_w = int(getattr(config, "mafia_T_w", 30))  # Window size for features
        
        # Verify sampling strategy
        sampling_strategy = getattr(config, "mafia_sampling_strategy", "random_trajectory")
        if sampling_strategy != "random_trajectory":
            print(f"[WARN] ObserverOfflineBatchTrainer enforces 'random_trajectory' but config is '{sampling_strategy}'")

        self.K = int(getattr(config, "mafia_top_k", 10))  # Top-K stocks
        self.horizon = int(
            getattr(config, "mafia_pg_reward_horizon", 5)
        )  # h: Look-ahead horizon for rewards (default: 5 days)

        # Rebalance interval
        self.rebalance_interval = int(
            getattr(
                config,
                "mafia_topk_rebalance_interval",
                getattr(config, "rebalance_interval", 14),
            )
        )

        # Loss weights (Spec §5 Hyperparameters)
        self.lambda_pg = float(getattr(config, "mafia_lambda_pg", 1.0))    # λ_pg = 1.0 (default)
        self.lambda_risk = float(getattr(config, "mafia_lambda_risk", 0.3))  # λ_risk = 0.3 per Spec §5
        self.lambda_dir = float(getattr(config, "mafia_lambda_dir", 0.5))   # λ_dir = 0.5 per Spec §5

        # Penalty coefficients for PG
        self.alpha_turnover = float(getattr(config, "mafia_pg_alpha_turnover", 0.08))
        self.alpha_change = float(getattr(config, "mafia_pg_alpha_change", 0.08))

        # Risk loss scaling factor (Spec §5.1.2: S_risk)
        # L_risk (MSE) ≈ 0.01, scale up to match L_PG (~1.0) and L_dir (~1.0)
        self.risk_scaling_factor = float(getattr(config, "scale_factor_risk", 10.0))

        # Reward scaling factor (Spec §5.1.1: S_reward)
        # Raw compounding returns ~0.01, scale to ~1.0 for stable gradients
        self.scale_factor_reward = float(getattr(config, "scale_factor_reward", 100.0))

        # Curriculum Learning parameters (Spec §7.1)
        self.curriculum_warmup_epochs = int(getattr(config, "curriculum_warmup_epochs", 5))
        self.curriculum_penalty_rampup = int(getattr(config, "curriculum_penalty_rampup", 5))
        self._current_lambda_epoch = 1.0  # Default: full penalty (will be updated per epoch)

        # Entropy bonus coefficient for PG (Spec §5.1.1: β_ent = 0.01)
        self.beta_entropy = float(getattr(config, "mafia_beta_entropy", 0.01))

        # Focal Loss parameters for Direction (Spec §5.1.3)
        # Optimized α based on VNINDEX distribution: Bear=19.1%, Side=46.0%, Bull=34.9%
        self.focal_gamma = float(getattr(config, "mafia_focal_gamma", 2.0))       # Focusing parameter
        self.focal_alpha = list(getattr(config, "mafia_focal_alpha", [1.5, 0.4, 1.0]))  # Class weights [bear, side, bull]
        # Temperature scaling for direction logits (T<1 sharpens distribution)
        self.direction_temperature = float(getattr(config, "mafia_direction_temperature", 1.0))
        
        # Initialize Loss Criteria
        # Initialize Loss Criteria
        self.risk_criterion = th.nn.MSELoss()
        
        # Direction Loss: Focal Loss (Spec 5.1.3)
        # Handles class imbalance (Side >> Bear/Bull) and hard examples
        # Label smoothing ε=0.1: [0,1,0] → [0.033, 0.933, 0.033] - avoids overconfidence
        self.direction_label_smoothing = float(
            getattr(config, "mafia_direction_label_smoothing", 0.1)
        )
        self.dir_criterion = FocalLoss(
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            device=self.device,
            label_smoothing=self.direction_label_smoothing  # Spec 5.1.3
        )
        smart_print(f"[LOSS] Direction Loss: FocalLoss(γ={self.focal_gamma}, α={self.focal_alpha}, ε_smooth={self.direction_label_smoothing})")

        # Direction threshold for labeling
        self.direction_threshold = float(
            getattr(config, "mafia_direction_threshold", 0.02)
        )

        # Regime Shift parameters
        self.vol_k = float(getattr(config, "regime_vol_k", 3.0))
        self.vol_window_size = int(getattr(config, "regime_vol_window", 60))

        # Gradient clipping
        self.max_grad_norm = float(getattr(config, "mafia_max_grad_norm", 1.0))

        # Log device being used
        smart_print(f"[DEVICE] Using device: {self.device}")

        # Memory Optimization: Mixed Precision Training
        self.use_mixed_precision = getattr(config, "use_mixed_precision", True)
        self.scaler = None
        if self.use_mixed_precision:
            # Only create scaler if CUDA is available (AMP requires CUDA)
            if self.device.type == "cuda":
                self.scaler = th.cuda.amp.GradScaler()
                smart_print("[OPTIMIZER] Mixed Precision (FP16) enabled with GradScaler")
            else:
                self.use_mixed_precision = False  # Disable on non-CUDA (MPS/CPU)
                device_name = "MPS" if self.device.type == "mps" else "CPU"
                smart_print(f"[OPTIMIZER] Mixed Precision disabled ({device_name} mode - AMP requires CUDA)")
        
        # Memory Optimization: Gradient Accumulation
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        if self.gradient_accumulation_steps > 1:
            smart_print(f"[OPTIMIZER] Gradient Accumulation enabled: {self.gradient_accumulation_steps} steps")
            smart_print(f"            Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")

        # Training tracking
        self._epoch = 0
        self._step = 0
        self._total_loss_accum = 0.0
        self._batch_count = 0

    def _clean_price_anomalies(
        self,
        df: pd.DataFrame,
        price_floor: Optional[float] = None,
        extreme_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Guard against bad price rows and pathological returns before tensorization.

        - Replace non-positive O/H/L/C with ffill/bfill (per stock)
        - Cap extreme single-day returns by adjusting prices to clipped bounds
        """
        price_cols = ["open", "high", "low", "close"]
        floor = (
            float(getattr(self.config, "mafia_price_floor", 1e-6))
            if price_floor is None
            else price_floor
        )
        extreme = (
            float(getattr(self.config, "mafia_extreme_return_threshold", 0.8))
            if extreme_threshold is None
            else extreme_threshold
        )
        clip_min = float(getattr(self.config, "mafia_return_clip_min", -0.5))
        clip_max = float(getattr(self.config, "mafia_return_clip_max", 1.0))

        cleaned = df.sort_values(["stock", "date"], ascending=True, ignore_index=True).copy()

        # Step 1: replace non-positive prices
        nonpos_mask = (cleaned[price_cols] <= floor).any(axis=1)
        nonpos_count = int(nonpos_mask.sum())
        if nonpos_count > 0:
            smart_print(
                f"[DATA-CLEAN] Replacing {nonpos_count} rows with non-positive prices "
                f"via ffill/bfill (floor={floor})"
            )
            for col in price_cols:
                cleaned[col] = (
                    cleaned.groupby("stock")[col]
                    .transform(lambda s: s.mask(s <= floor).ffill().bfill())
                    .fillna(floor)
                )

        # Step 2: cap extreme returns by adjusting prices
        cleaned["_ret_tmp"] = cleaned.groupby("stock")["close"].pct_change()
        extreme_mask = cleaned["_ret_tmp"].abs() > extreme
        extreme_count = int(extreme_mask.sum())
        if extreme_count > 0:
            max_ret = float(cleaned.loc[extreme_mask, "_ret_tmp"].abs().max())
            sample_rows = cleaned.loc[extreme_mask, ["date", "stock", "_ret_tmp"]].head(3)
            smart_print(
                f"[DATA-CLEAN] Capping {extreme_count} extreme returns "
                f"(max |r|={max_ret:.2f}, threshold={extreme}) "
                f"to [{clip_min}, {clip_max}]"
            )
            for _, row in sample_rows.iterrows():
                smart_print(
                    f"  • {row['date'].date()} {row['stock']} return={row['_ret_tmp']:.2f}"
                )

            for stock, stock_df in cleaned.groupby("stock", sort=False):
                returns = stock_df["_ret_tmp"].values
                idx_list = stock_df.index.tolist()
                bad_pos = np.where(np.abs(returns) > extreme)[0]
                if len(bad_pos) == 0:
                    continue

                for pos in bad_pos:
                    if pos == 0:
                        continue  # no previous price to anchor
                    prev_idx = idx_list[pos - 1]
                    cur_idx = idx_list[pos]
                    prev_close = cleaned.at[prev_idx, "close"]
                    clipped_ret = float(np.clip(returns[pos], clip_min, clip_max))
                    new_close = prev_close * (1.0 + clipped_ret)
                    cleaned.loc[cur_idx, ["close", "open", "high", "low"]] = new_close

            # Recompute returns to purge capped values from helper column
            cleaned["_ret_tmp"] = cleaned.groupby("stock")["close"].pct_change()

        cleaned.drop(columns=["_ret_tmp"], inplace=True)
        return cleaned

    def _compute_lambda_epoch(self, current_epoch: int) -> float:
        """
        Curriculum Learning: Compute penalty weight λ_epoch (Spec §7.1).

        Phase 1 (epoch < warmup): λ = 0.0 (pure alpha learning)
        Phase 2 (warmup <= epoch < warmup+rampup): λ ramps 0→1 linearly
        Phase 3 (epoch >= warmup+rampup): λ = 1.0 (full discipline)

        Args:
            current_epoch: Current training epoch (0-indexed)

        Returns:
            λ_epoch in [0.0, 1.0]
        """
        if current_epoch < self.curriculum_warmup_epochs:
            return 0.0

        ramp_progress = current_epoch - self.curriculum_warmup_epochs
        if ramp_progress >= self.curriculum_penalty_rampup:
            return 1.0

        return ramp_progress / self.curriculum_penalty_rampup

    def prepare_data_tensors(
        self,
        data: pd.DataFrame,
        stock_list: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, th.Tensor]:
        """
        Prepare data tensors for trajectory sampling.
        
        OPTIMIZED (Memory-Safe): Processes stock-by-stock to avoid Peak RAM spikes.
        """
        import gc
        
        smart_print("[OFFLINE] Filtering data by date range...")
        # 1. First, just filter date to reduce size, but keep as DataFrame
        mask = (data["date"] >= start_date) & (data["date"] <= end_date)
        # Keep only necessary columns + stock/date for indexing
        filtered = data.loc[mask, ["date", "stock", "open", "close", "high", "low", "volume"]].copy()
        filtered = self._clean_price_anomalies(filtered)
        
        # Free original huge dataframe reference
        del data
        gc.collect()

        # Get unique dates
        dates = np.sort(filtered["date"].unique())
        T_total = len(dates)
        N = len(stock_list)

        # Relaxed check: Allow smaller datasets for EVAL mode (min 1 step)
        min_required = self.T_w + self.horizon + 1
        if T_total < min_required:
            raise ValueError(
                f"Insufficient data: {T_total} days. Need at least "
                f"T_w({self.T_w}) + h({self.horizon}) + 1 = {min_required} days."
            )
        
        if T_total < self.T_m + self.horizon + self.T_w:
            smart_print(
                f"[WARN] Data length ({T_total}) is smaller than standard training trajectory "
                f"(T_m={self.T_m} + T_w={self.T_w} + h={self.horizon} = {self.T_m + self.horizon + self.T_w}). "
                f"This is fine for EVAL but will fail in TRAIN mode if not handled."
            )

        smart_print(f"[OFFLINE] Processing {N} stocks over {T_total} days (Chunked Mode)...")
        
        # Date mapping for fast lookup: Timestamp -> Index
        date_map = {pd.Timestamp(d): i for i, d in enumerate(dates)}
        
        # Preallocate arrays: (T_total, N, 5) for OCHLV
        # Initialize with NaN
        ochlv_array = np.full((T_total, N, 5), np.nan, dtype=np.float32)

        # 2. Iterate stock-by-stock to fill array (Low Peak Memory)
        # Group by stock first to avoid repeated filtering
        grouped = filtered.groupby("stock")
        
        for n, stock_ticker in enumerate(stock_list):
            if n % 50 == 0:
                smart_print(f"[OFFLINE]   Processing stock {n}/{N}: {stock_ticker}...")
                gc.collect() # Garbage collect regularly
            
            if stock_ticker not in grouped.groups:
                continue
                
            # Get slice for this stock
            stock_df = grouped.get_group(stock_ticker)
            
            # Map valid dates
            # We can vectorize this part per stock (small enough for RAM)
            valid_dates = stock_df["date"].map(date_map)
            valid_mask = valid_dates.notna()
            
            if not valid_mask.any():
                continue
                
            indices = valid_dates[valid_mask].astype(int).values
            
            # Fill One Stock Slice
            # (Indices, n, 0..4)
            ochlv_array[indices, n, 0] = stock_df.loc[valid_mask, "open"].values
            ochlv_array[indices, n, 1] = stock_df.loc[valid_mask, "close"].values
            ochlv_array[indices, n, 2] = stock_df.loc[valid_mask, "high"].values
            ochlv_array[indices, n, 3] = stock_df.loc[valid_mask, "low"].values
            ochlv_array[indices, n, 4] = stock_df.loc[valid_mask, "volume"].values
            
            # Handle NaN for this stock immediately (Forward/Backward Fill)
            for f in range(5):
                col = ochlv_array[:, n, f]
                mask_nan = np.isnan(col)
                if mask_nan.all():
                    col[:] = 1.0 if f < 4 else 0.0
                elif mask_nan.any():
                    # Forward fill
                    idx = np.where(~mask_nan, np.arange(len(col)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    col[:] = col[idx]
                    # Backward fill
                    mask_nan = np.isnan(col)
                    if mask_nan.any():
                        idx = np.where(~mask_nan, np.arange(len(col)), len(col) - 1)
                        idx = np.minimum.accumulate(idx[::-1])[::-1]
                        col[:] = col[idx]

        # Free source dataframe completely
        del filtered
        del grouped
        gc.collect()

        smart_print("[OFFLINE] Computing returns...")
        # Compute returns: (T_total, N)
        close_prices = ochlv_array[:, :, 1]
        returns = np.zeros_like(close_prices)
        returns[1:] = (close_prices[1:] - close_prices[:-1]) / np.clip(close_prices[:-1], 1e-8, None)
        clip_min = float(getattr(self.config, "mafia_return_clip_min", -0.5))
        clip_max = float(getattr(self.config, "mafia_return_clip_max", 1.0))
        clip_threshold = max(abs(clip_min), abs(clip_max))
        
        # 🚨 DATA QUALITY CHECK: Detect extreme returns that cause compounding overflow
        extreme_mask = np.abs(returns) > clip_threshold
        if extreme_mask.any():
            num_extreme = extreme_mask.sum()
            max_return = np.abs(returns).max()
            percentile_99 = np.percentile(np.abs(returns), 99)
            smart_print(
                f"\n{'='*80}\n"
                f"🚨 WARNING: Extreme returns detected!\n"
                f"{'='*80}\n"
                f"  Count:              {num_extreme:,} / {returns.size:,} ({num_extreme/returns.size*100:.3f}%)\n"
                f"  Max absolute:       {max_return:.2f} ({max_return*100:.0f}%)\n"
                f"  99th percentile:   {percentile_99:.4f}\n"
                f"  This indicates data quality issues:\n"
                f"    - Stock splits not adjusted\n"
                f"    - Bad price data\n"
                f"    - Corporate actions\n"
                f"  → Clipping to [{clip_min}, {clip_max}] to prevent overflow\n"
                f"{'='*80}\n"
            )
            # Clip to reasonable range
            returns = np.clip(returns, clip_min, clip_max)


        # Process Market Data if available (Vectorized for single asset is fine)
        market_ochlv_array = np.zeros((T_total, 1, 5), dtype=np.float32)
        market_returns = np.zeros((T_total, 1), dtype=np.float32)

        if market_data is not None:
            smart_print("[OFFLINE] Processing market data...")
            req_mkt_cols = ["date", "open", "high", "low", "close", "volume"]
            if all(c in market_data.columns for c in req_mkt_cols):
                mkt_copy = market_data[req_mkt_cols].copy()
                mkt_copy["date"] = pd.to_datetime(mkt_copy["date"])
                if mkt_copy["date"].dt.tz is not None:
                    mkt_copy["date"] = mkt_copy["date"].dt.tz_localize(None)

                mkt_filtered = mkt_copy[(mkt_copy["date"] >= start_date) & (mkt_copy["date"] <= end_date)].copy()
                del mkt_copy
                
                # Build date map for market data
                mkt_filtered["date_idx"] = mkt_filtered["date"].map(date_map)
                mkt_filtered = mkt_filtered[mkt_filtered["date_idx"].notna()].copy()
                mkt_filtered["date_idx"] = mkt_filtered["date_idx"].astype(int)
                
                # Vectorized fill
                mkt_date_indices = mkt_filtered["date_idx"].values
                market_ochlv_array[mkt_date_indices, 0, 0] = mkt_filtered["open"].values.astype(np.float32)
                market_ochlv_array[mkt_date_indices, 0, 1] = mkt_filtered["close"].values.astype(np.float32)
                market_ochlv_array[mkt_date_indices, 0, 2] = mkt_filtered["high"].values.astype(np.float32)
                market_ochlv_array[mkt_date_indices, 0, 3] = mkt_filtered["low"].values.astype(np.float32)
                market_ochlv_array[mkt_date_indices, 0, 4] = mkt_filtered["volume"].values.astype(np.float32)
                
                del mkt_filtered
                gc.collect()

                # Fill Nan
                for f in range(5):
                    col = market_ochlv_array[:, 0, f]
                    mask_zero = col == 0
                    if mask_zero.all():
                        col[:] = 1.0 if f < 4 else 0.0
                    elif mask_zero.any():
                        idx = np.where(~mask_zero, np.arange(len(col)), 0)
                        np.maximum.accumulate(idx, out=idx)
                        col[:] = col[idx]

                # Compute market returns
                mkt_close = market_ochlv_array[:, 0, 1]
                market_returns[1:, 0] = (mkt_close[1:] - mkt_close[:-1]) / np.clip(mkt_close[:-1], 1e-8, None)
        
        smart_print("[OFFLINE] Converting to tensors (on CPU first)...")
        # Convert to tensors
        result = {
            "ochlv": th.from_numpy(ochlv_array),  # (T_total, N, 5)
            "returns": th.from_numpy(returns),  # (T_total, N)
            "market_ochlv": th.from_numpy(market_ochlv_array),  # (T_total, 1, 5)
            "market_returns": th.from_numpy(market_returns),  # (T_total, 1)
            "dates": dates,
            "stock_list": stock_list,
            "T_total": T_total,
            "N": N,
        }
        
        # Free numpy arrays
        del ochlv_array, returns, market_ochlv_array, market_returns
        gc.collect()

        return result

    def _build_class_indices(
        self,
        data_tensors: Dict[str, th.Tensor],
        min_start: int,
        max_start: int,
        dir_lookahead: int,
    ) -> Dict[int, np.ndarray]:
        """
        Pre-compute direction labels for all valid start indices and group by class.

        Returns:
            Dict mapping class label (0=Bear, 1=Side, 2=Bull) to array of valid start indices
        """
        market_ochlv = data_tensors.get("market_ochlv")
        if market_ochlv is None:
            # Fallback to uniform sampling if no market data
            return None

        # Convert to numpy for faster computation
        market_close = market_ochlv[:, 0, 1].cpu().numpy()  # (T_total,)
        market_high = market_ochlv[:, 0, 2].cpu().numpy()
        market_low = market_ochlv[:, 0, 3].cpu().numpy()

        # Compute ATR for dynamic threshold (Spec 5.1.3)
        high = market_high
        low = market_low
        close = market_close
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[tr[0]], tr])  # Pad first element
        atr_period = max(1, int(getattr(self.config, "direction_label_atr_period", 14)))
        atr = np.convolve(tr, np.ones(atr_period) / atr_period, mode="same")  # SMA ATR

        # Direction labeling params (Spec 5.1.3)
        k_atr = float(getattr(self.config, "direction_label_atr_multiplier", 2.0))
        delta_min = float(
            getattr(
                self.config,
                "direction_label_delta_min",
                getattr(self.config, "mafia_direction_threshold", 0.02),
            )
        )
        stop_loss = float(getattr(self.config, "direction_label_stop_loss", -0.07))

        # Compute labels for all valid indices
        class_indices = {0: [], 1: [], 2: []}  # Bear, Side, Bull

        for t_s in range(min_start, max_start + 1):
            future_idx = min(t_s + dir_lookahead, len(market_close) - 1)

            # Dynamic threshold
            atr_ratio = atr[t_s] / market_close[t_s] if market_close[t_s] > 0 else 0
            delta_t = max(delta_min, k_atr * atr_ratio)

            # Future return
            r_fut = (market_close[future_idx] / market_close[t_s]) - 1.0 if market_close[t_s] > 0 else 0

            # Intra-period drawdown
            future_lows = market_low[t_s+1:future_idx+1]
            if len(future_lows) > 0:
                min_price = np.min(future_lows)
                intra_dd = (min_price / market_close[t_s]) - 1.0 if market_close[t_s] > 0 else 0
            else:
                intra_dd = 0

            # Label assignment
            if r_fut < -delta_t or intra_dd < stop_loss:
                label = 0  # Bear
            elif r_fut > delta_t and intra_dd >= stop_loss:
                label = 2  # Bull
            else:
                label = 1  # Side

            class_indices[label].append(t_s)

        # Convert to numpy arrays
        for k in class_indices:
            class_indices[k] = np.array(class_indices[k])

        return class_indices

    def sample_trajectory_batch(
        self,
        data_tensors: Dict[str, th.Tensor],
        market_tensors: Optional[Dict[str, th.Tensor]] = None,
        mode: str = "TRAIN",
    ) -> TrajectoryBatch:
        """
        Sample a batch of random trajectories per Spec §6.

        Each trajectory:
        1. Has random start index t_s
        2. Spans [t_s, t_s + T_m) consecutive days
        3. Is independent of other trajectories in batch

        Args:
            data_tensors: Dict from prepare_data_tensors()
            market_tensors: Optional market index tensors

        Returns:
            TrajectoryBatch with B independent trajectories
        """
        T_total = data_tensors["T_total"]
        ochlv = data_tensors["ochlv"]
        returns = data_tensors["returns"]
        dates = data_tensors["dates"]

        # Market data (if valid, otherwise fallback to stock avg)
        market_ochlv = data_tensors.get("market_ochlv")  # (T_total, 1, 5)
        market_returns = data_tensors.get("market_returns")  # (T_total, 1)
        use_market_index = (
            (market_ochlv is not None)
            and (market_returns is not None)
            and (market_ochlv.sum() != 0)
        )

        # Valid start range: need T_w lookback + T_m trajectory + horizon lookahead
        # Valid start range: need T_w lookback + T_m trajectory + horizon lookahead
        # But first, determine T_actual based on Mode and Available Data
        
        # L_avail = Len(Data) - T_w - h
        L_avail = T_total - self.T_w - self.horizon
        
        if L_avail <= 0:
            raise ValueError(
                f"Data too short (L_avail={L_avail} <= 0). "
                f"T_total={T_total}, T_w={self.T_w}, h={self.horizon}."
            )

        if mode == "TRAIN":
            # Strict consistency for training
            T_actual = self.T_m
            if L_avail < T_actual:
                 # Data shorter than required for one full training trajectory
                 raise ValueError(
                     f"[TRAIN] Data length ({T_total}) insufficient for target T_m={self.T_m}. "
                     f"L_avail={L_avail} < {self.T_m}. Training requires consistent batch sizes."
                 )
        else: # EVAL or VALIDATION
            # Flexible length: utilize what's available
            # "Mô hình chạy một mạch (Single Pass) hoặc chia nhỏ Sequential Chunks với độ dài tối đa là L_avail"
            # We cap at self.T_m to avoid memory issues, but allow smaller.
            T_actual = min(self.T_m, L_avail)
            # If T_actual is significantly smaller, that's fine for Eval.

        # Calculate max_start
        # If we use T_actual, we need [t_s, t_s + T_actual)
        # Plus lookahead h: [t_s, t_s + T_actual + h)
        # Plus lookback T_w: [t_s - T_w, ...]
        # So last index needed is t_s + T_actual + h - 1?
        # Actually: max_start is the last valid t_s.
        # Valid range of data is [0, T_total).
        # We need data up to index `t_s + T_actual + h - 1` to exist.
        # So `t_s + T_actual + h <= T_total` => `t_s <= T_total - T_actual - h`.
        max_start = T_total - T_actual - self.horizon
        min_start = self.T_w  # Need T_w days of history for features
        
        # In EVAL mode with minimal data (L_avail small), max_start might equal min_start.
        if max_start < min_start:
             # Should be caught by L_avail check, but double check
             raise ValueError(
                f"Invalid sampling range: [{min_start}, {max_start}]. "
                f"T_total={T_total}, T_actual={T_actual}, T_w={self.T_w}, h={self.horizon}"
             )

        # Sample B random start indices
        # Two strategies: uniform random OR class-balanced stratified sampling
        use_class_balanced = getattr(self.config, "mafia_use_class_balanced_sampling", False)

        if use_class_balanced and mode == "TRAIN":
            # Class-balanced sampling: sample equally from Bear, Side, Bull classes
            dir_lookahead = int(getattr(self.config, "direction_label_lookahead", 14))
            class_indices = self._build_class_indices(data_tensors, min_start, max_start, dir_lookahead)

            if class_indices is not None:
                # Calculate samples per class (round robin to handle batch_size % 3 != 0)
                samples_per_class = self.batch_size // 3
                remainder = self.batch_size % 3

                start_indices = []
                for class_id in [0, 1, 2]:  # Bear, Side, Bull
                    n_samples = samples_per_class + (1 if class_id < remainder else 0)
                    indices = class_indices[class_id]

                    if len(indices) == 0:
                        # Fallback to uniform if class is empty
                        sampled = np.random.randint(min_start, max_start + 1, size=n_samples)
                    elif len(indices) < n_samples:
                        # Sample with replacement if not enough samples
                        sampled = np.random.choice(indices, size=n_samples, replace=True)
                    else:
                        # Sample without replacement
                        sampled = np.random.choice(indices, size=n_samples, replace=False)

                    start_indices.extend(sampled)

                start_indices = np.array(start_indices)
                np.random.shuffle(start_indices)  # Shuffle to avoid class order bias
            else:
                # Fallback to uniform random if class indices couldn't be built
                start_indices = np.random.randint(min_start, max_start + 1, size=self.batch_size)
        else:
            # Uniform random sampling (default)
            start_indices = np.random.randint(min_start, max_start + 1, size=self.batch_size)

        # Extract trajectory windows
        # For features, we need [t - T_w, t] for each step t in trajectory
        # So for trajectory [t_s, t_s + T_m), feature windows are [t_s - T_w, t_s + T_m)

        batch_stock_ochlv = []  # Will be (B, T_m, N, 5)
        batch_price_returns = []  # (B, T_m, N)
        batch_market_returns = []  # (B, T_m)
        batch_rebalance_mask = []  # (B, T_m)
        batch_direction_labels = []  # (B, T_m)
        batch_risk_targets = []  # (B, T_m)
        batch_dates = []

        # Risk params from config or defaults
        risk_lambda = float(getattr(self.config, "risk_eta_range", 0.3))
        risk_kappa = float(getattr(self.config, "risk_eta_sensitivity", 1.0))
        risk_lookahead = int(getattr(self.config, "risk_eta_lookahead", 20))

        # Direction params
        dir_lookahead = int(getattr(self.config, "direction_label_lookahead", 14))

        for b, t_s in enumerate(start_indices):
            # Extract trajectory slice [t_s, t_s + T_actual)
            traj_slice = slice(t_s, t_s + T_actual)

            # Stock OCHLV for this trajectory
            # For each step t in trajectory, we need window [t - T_w + 1, t + 1]
            stock_ochlv_traj = ochlv[traj_slice]  # (T_m, N, 5)
            batch_stock_ochlv.append(stock_ochlv_traj)

            # Returns for reward calculation
            # Returns for reward calculation
            # Need T_m + horizon for lookahead compounding
            # Slice from t_s to t_s + T_m + horizon
            # Ensure we don't go out of bounds (should be handled by max_start check)
            traj_end_extended = min(t_s + self.T_m + self.horizon, T_total)
            traj_slice_extended = slice(t_s, traj_end_extended)
            
            returns_traj = returns[traj_slice_extended]  # (T_m + h, N) or shorter if end of data
            
            # Pad if shorter (should not happen given max_start, but for safety)
            req_len = T_actual + self.horizon
            if returns_traj.shape[0] < req_len:
                 pad_len = req_len - returns_traj.shape[0]
                 # Pad with zeros (neutral return) - use same device as source tensor
                 pad = th.zeros((pad_len, returns_traj.shape[1]), device=returns_traj.device)
                 returns_traj = th.cat([returns_traj, pad], dim=0)

            batch_price_returns.append(returns_traj)

            # Market returns for Baseline
            if use_market_index:
                mkt_ret_traj = market_returns[traj_slice_extended].squeeze(-1)  # (T_m + h,)
                if mkt_ret_traj.shape[0] < req_len:
                     pad_len = req_len - mkt_ret_traj.shape[0]
                     # Use same device as source tensor
                     pad = th.zeros(pad_len, device=mkt_ret_traj.device)
                     mkt_ret_traj = th.cat([mkt_ret_traj, pad], dim=0)
                batch_market_returns.append(mkt_ret_traj)
            else:
                # Fallback to mean stock return if no market index
                batch_market_returns.append(returns_traj.mean(dim=-1))

            # Rebalance mask: 1 every rebalance_interval days, starting from day 0
            rebal_mask = th.zeros(T_actual, device=self.device)
            rebal_mask[0] = 1.0  # First day is always rebalance
            for t in range(self.rebalance_interval, T_actual, self.rebalance_interval):
                rebal_mask[t] = 1.0
            batch_rebalance_mask.append(rebal_mask)

            # -------------------------------------------------------------------------
            # Direction labels (Spec 5.1.3: Dynamic Threshold + Path Dependency)
            # -------------------------------------------------------------------------
            # 1. Calc ATR_14(t) for Dynamic Threshold
            # We need high/low/close from history [t-14, t]
            # Since we iterate t, we can slice from stock_ochlv_traj or market
            
            # Helper to get price series for this trajectory
            if use_market_index:
                # (T_m, 5)
                traj_open = market_ochlv[traj_slice, 0, 0]
                traj_high = market_ochlv[traj_slice, 0, 2]
                traj_low  = market_ochlv[traj_slice, 0, 3]
                traj_close= market_ochlv[traj_slice, 0, 1]
            else:
                # Fallback: Mean of stocks
                traj_open = ochlv[traj_slice, :, 0].mean(dim=1)
                traj_high = ochlv[traj_slice, :, 2].mean(dim=1)
                traj_low  = ochlv[traj_slice, :, 3].mean(dim=1)
                traj_close= ochlv[traj_slice, :, 1].mean(dim=1)

            # Pre-calc ATR for the whole trajectory (approximate, should ideally look back further before t_s)
            # To do this correctly without fetching outside traj_slice inside loop, 
            # we should access data_tensors directly with indices.
            
            dir_labels = th.zeros(T_actual, dtype=th.long, device=self.device)
            # Dynamic Threshold parameters from config (Spec §5.1.3)
            # Optimized for VNINDEX: k_atr=2.0, δ_min=2%, SL=-7%
            atr_period = int(getattr(self.config, "direction_label_atr_period", 14))
            k_atr = float(getattr(self.config, "direction_label_atr_multiplier", 2.0))
            delta_min = float(getattr(self.config, "direction_label_delta_min", 0.020))
            stop_loss_limit = float(getattr(self.config, "direction_label_stop_loss", -0.07))

            for t in range(T_actual):
                current_idx = t_s + t
                future_idx = min(current_idx + dir_lookahead, T_total - 1)
                
                # 1. Get ATR(14)
                # Need atr_period TRs: grab atr_period+1 prices to compute TR.
                atr_window_start = max(0, current_idx - (atr_period + 1))
                
                # Extract simple window for ATR calc
                if use_market_index:
                     h_slice = market_ochlv[atr_window_start:current_idx+1, 0, 2]
                     l_slice = market_ochlv[atr_window_start:current_idx+1, 0, 3]
                     c_slice = market_ochlv[atr_window_start:current_idx+1, 0, 1]
                     p_t = market_ochlv[current_idx, 0, 1].item()
                else:
                     h_slice = ochlv[atr_window_start:current_idx+1, :, 2].mean(dim=1)
                     l_slice = ochlv[atr_window_start:current_idx+1, :, 3].mean(dim=1)
                     c_slice = ochlv[atr_window_start:current_idx+1, :, 1].mean(dim=1)
                     p_t = ochlv[current_idx, :, 1].mean().item()
                
                if len(c_slice) < 2:
                    current_atr = 0.0
                else:
                    # TR = max(H-L, |H-Cp|, |L-Cp|)
                    # Simple valid TR calc on tensor
                    tr_h_l = h_slice[1:] - l_slice[1:]
                    tr_h_cp = (h_slice[1:] - c_slice[:-1]).abs()
                    tr_l_cp = (l_slice[1:] - c_slice[:-1]).abs()
                    tr = th.max(tr_h_l, th.max(tr_h_cp, tr_l_cp))
                    # Simple avg for ATR (EMA is standard but SMA is fine for approx)
                    current_atr = tr.mean().item()

                # Dynamic Threshold
                delta_t = max(delta_min, k_atr * (current_atr / (p_t + 1e-8)))

                # 2. Get Future Price & Intra-DD
                # Window [t+1, t+k]
                future_slice_idx = slice(current_idx + 1, min(current_idx + dir_lookahead + 1, T_total))
                
                if use_market_index:
                    fut_lows = market_ochlv[future_slice_idx, 0, 3]
                    p_fut = market_ochlv[future_idx, 0, 1].item()
                else:
                    fut_lows = ochlv[future_slice_idx, :, 3].mean(dim=1)
                    p_fut = ochlv[future_idx, :, 1].mean().item()

                if p_t > 0:
                    r_fut = (p_fut - p_t) / p_t
                    # IntraDD = min(Low_{t+1..t+k}) / P_t - 1
                    if len(fut_lows) > 0:
                        min_low = fut_lows.min().item()
                        intra_dd = (min_low / p_t) - 1.0
                    else:
                        intra_dd = 0.0
                else:
                    r_fut = 0.0
                    intra_dd = 0.0

                # 3. Labeling Logic
                # Bear (0): r_fut < -delta_t OR intra_dd < stop_loss
                if r_fut < -delta_t or intra_dd < stop_loss_limit:
                    dir_labels[t] = 0 # Bear
                # Bull (2): r_fut > delta_t AND intra_dd >= stop_loss
                elif r_fut > delta_t and intra_dd >= stop_loss_limit:
                    dir_labels[t] = 2 # Bull
                # Side (1): Otherwise
                else:
                    dir_labels[t] = 1 # Side
            
            batch_direction_labels.append(dir_labels)

            # -------------------------------------------------------------------------
            # Risk targets (Spec 5.1.2: Hybrid Target with Drawdown Penalty)
            # -------------------------------------------------------------------------
            # eta = 1 + lambda_val * tanh(Z_fut) - lambda_dd * Clip(MaxDD/ref, 0, 1)

            risk_targets = th.ones(T_actual, device=self.device)
            # Hybrid eta_target parameters from config
            lambda_val = float(getattr(self.config, "risk_eta_lambda", 0.3))
            lambda_dd = float(getattr(self.config, "risk_eta_lambda_dd", 0.5))
            dd_ref = float(getattr(self.config, "risk_eta_dd_ref", 0.10))  # 10%
            
            for t in range(T_actual):
                current_idx = t_s + t
                future_end = min(current_idx + risk_lookahead + 1, T_total)
                future_slice = range(current_idx + 1, future_end)

                if not future_slice or list(future_slice) == []:
                    risk_targets[t] = 1.0
                    continue

                if use_market_index:
                    prices = market_ochlv[future_slice, 0, 1]  # Future close prices
                    p_t = market_ochlv[current_idx, 0, 1].item()
                    # For MaxDD, we usually look at Highs and Lows in the window, 
                    # but spec says P_{t..t+tau} which implies Close or generic price. 
                    # Let's use Close series for MaxDD calc to match Z-score basis.
                    price_series = th.cat([
                        market_ochlv[current_idx:current_idx+1, 0, 1], 
                        prices
                    ]) # Includes P_t at index 0
                else:
                    # Fallback
                    prices = ochlv[future_slice, :, 1].mean(dim=1)
                    p_t = ochlv[current_idx, :, 1].mean().item()
                    price_series = th.cat([
                         ochlv[current_idx:current_idx+1, :, 1].mean(dim=1),
                         prices
                    ])

                if len(prices) > 0:
                    # 1. Component 1: Relative Valuation (Z-score)
                    mu_fut = prices.mean().item()
                    std_fut = prices.std().item()
                    eps = 1e-6
                    z_score = (mu_fut - p_t) / (std_fut + eps)
                    term_val = lambda_val * np.tanh(z_score) # Note: Removing kappa per strict spec eq 1025? No, spec 5.1.2 simply says tanh(Z_fut).
                    
                    # 2. Component 2: MaxDD Penalty
                    # MaxDD_fut = max_{tau} ( (max(P_{t..tau}) - P_{tau}) / max(P...) )
                    # This is standard "DD from peak" within the window
                    
                    # Convert to numpy for easier rolling max
                    p_series_np = price_series.cpu().numpy()
                    # Accumulate max
                    rolling_max = np.maximum.accumulate(p_series_np)
                    drawdowns = (rolling_max - p_series_np) / (rolling_max + 1e-8)
                    # We care about MAX drawdown observed in the future window
                    max_dd_fut = drawdowns.max()
                    
                    term_dd = lambda_dd * np.clip(max_dd_fut / dd_ref, 0.0, 1.0)
                    
                    # specific formula: eta = 1.0 + term_val - term_dd
                    eta = 1.0 + term_val - term_dd
                    # Spec bound: keep η within adaptive tolerance window [eta_min, eta_max]
                    eta_base = getattr(self.config, "mafia_eta_base", 1.0)
                    eta_amp = getattr(self.config, "mafia_eta_amplitude", 0.3)
                    eta_min = getattr(self.config, "mafia_eta_min", eta_base - eta_amp)
                    eta_max = getattr(self.config, "mafia_eta_max", eta_base + eta_amp)
                    eta = max(eta_min, min(eta_max, eta))
                    
                    if np.isnan(eta) or np.isinf(eta):
                        eta = 1.0
                    
                    risk_targets[t] = eta

            batch_risk_targets.append(th.nan_to_num(risk_targets, nan=1.0, posinf=1.0, neginf=1.0))

            # Store dates
            batch_dates.append(dates[traj_slice])

        # Pre-calculate Volatility Shock Mask (Spec §8.2.3)
        # Logic: V_curr > μ_vol + k × σ_vol
        # Where:
        #   - V_curr = current 20-day rolling volatility (std of recent returns)
        #   - μ_vol = mean of historical volatilities
        #   - σ_vol = std of historical volatilities
        batch_vol_shock_mask = []
        vol_history_window = max(self.vol_window_size, 20)  # Window for volatility history

        for b, t_s in enumerate(start_indices):
            # We need extra history before t_s to build volatility history
            # Need at least vol_history_window + vol_window_size days before trajectory
            full_window_start = max(0, t_s - vol_history_window - self.vol_window_size)
            full_window_end = min(T_total, t_s + T_actual)

            # Use market returns if available, else stock mean returns
            if use_market_index:
                raw_returns = market_returns[full_window_start:full_window_end].squeeze(-1)  # (L,)
            else:
                raw_returns = returns[full_window_start:full_window_end].mean(dim=-1)  # (L,)

            raw_returns_np = raw_returns.cpu().numpy()
            traj_offset_in_window = t_s - full_window_start

            mask_traj = np.zeros(T_actual, dtype=np.float32)

            # Pre-compute rolling volatilities for the entire available window
            # rolling_vols[i] = std(returns[i - vol_window : i])
            rolling_vols = []
            for i in range(self.vol_window_size, len(raw_returns_np)):
                window_slice = raw_returns_np[i - self.vol_window_size : i]
                vol = np.std(window_slice)
                rolling_vols.append(vol)

            # Now for each timestep in trajectory, check volatility shock
            for t in range(T_actual):
                current_idx_in_window = traj_offset_in_window + t
                vol_idx = current_idx_in_window - self.vol_window_size  # Index in rolling_vols

                # Need enough history to compute both current vol and vol distribution
                if vol_idx < vol_history_window:
                    continue

                # Current volatility (V_curr)
                vol_current = rolling_vols[vol_idx]

                # Historical volatilities for computing μ_vol and σ_vol
                # Use last vol_history_window volatilities (excluding current)
                vol_history = rolling_vols[max(0, vol_idx - vol_history_window) : vol_idx]

                if len(vol_history) < 3:
                    continue

                vol_mu = np.mean(vol_history)
                vol_sigma = np.std(vol_history)

                # Volatility Shock: V_curr > μ_vol + k × σ_vol
                if vol_sigma > 1e-8 and vol_current > vol_mu + self.vol_k * vol_sigma:
                    mask_traj[t] = 1.0

            batch_vol_shock_mask.append(th.from_numpy(mask_traj).to(self.device))

        # Pre-calculate Structural Break (DC) Mask (Spec §8.1.3)
        # Threshold: mafia_DC_threshold (default 0.02)
        dc_threshold = float(getattr(self.config, "mafia_DC_threshold", 0.02))
        batch_dc_mask = []
        
        for b, t_s in enumerate(start_indices):
            # Use Market Prices for DC detection
            traj_slice_idx = slice(t_s, t_s + T_actual)
            if use_market_index:
                # Market Close Price: (T_m,)
                prices = market_ochlv[traj_slice_idx, 0, 1].cpu().numpy()
            else:
                # Fallback: Mean Stock Close Price
                prices = ochlv[traj_slice_idx, :, 1].mean(dim=-1).cpu().numpy()
                
            mask_dc = np.zeros(T_actual, dtype=np.float32)
            
            # Simple DC Event Detection (Offline)
            # We iterate through the trajectory window only.
            # To be precise, we should track state from BEFORE t_s, but for offline training
            # a local DC detection within the window is sufficient/safe approximation,
            # or we can assume a reset at t=0.
            # Let's simple use the first price as reference.
            
            # DC Algorithm:
            # Mode: Up or Down
            # P_ext: Extreme price since last DC
            
            # Initial state
            p_ext = prices[0]
            mode = "up" # arbitrary start, will adjust
            
            for t in range(1, T_actual):
                price = prices[t]
                var = (price - p_ext) / p_ext
                
                dc_confirmed = False
                
                if mode == "up":
                    if var < -dc_threshold:
                        # Downward DC confirmed (CRASH TRIGGER)
                        dc_confirmed = True
                        mode = "down"
                        p_ext = price
                    elif price > p_ext:
                        p_ext = price
                else: # mode == "down"
                    if var > dc_threshold:
                        # Upward DC confirmed (RALLY - Update state but NO TRIGGER per Spec 8.2.2)
                        # We only want to force rebalance on CRASH/Downward break for defensive purposes.
                        dc_confirmed = False 
                        mode = "up"
                        p_ext = price
                    elif price < p_ext:
                        p_ext = price
                        
                if dc_confirmed:
                    mask_dc[t] = 1.0
                    
            batch_dc_mask.append(th.from_numpy(mask_dc).to(self.device))


        # Stack into batch tensors and move to device
        batch = TrajectoryBatch(
            stock_ochlv=th.stack(batch_stock_ochlv, dim=0).to(self.device),  # (B, T_m, N, 5)
            market_ochlv=None,  # Not used in batch forward pass directly yet
            price_returns=th.stack(batch_price_returns, dim=0).to(self.device),  # (B, T_m, N)
            market_returns=th.stack(batch_market_returns, dim=0).to(self.device),  # (B, T_m)
            rebalance_mask=th.stack(batch_rebalance_mask, dim=0).to(self.device),  # (B, T_m)
            direction_labels=th.stack(batch_direction_labels, dim=0).to(self.device),  # (B, T_m)
            risk_targets=th.stack(batch_risk_targets, dim=0).to(self.device),  # (B, T_m)
            start_indices=th.tensor(start_indices, device=self.device),  # (B,)
            dates=np.array(batch_dates),  # (B, T_m)
            stock_list=data_tensors.get("stock_list"),  # List[str] of N symbols
            vol_shock_mask=th.stack(batch_vol_shock_mask, dim=0).to(self.device), # (B, T_m)
            dc_mask=th.stack(batch_dc_mask, dim=0).to(self.device), # (B, T_m)
        )

        return batch

    def collect_and_train_step(
        self,
        batch: TrajectoryBatch,
        data_tensors: Dict[str, th.Tensor],
        batch_idx: int = 0,
        epoch_idx: int = 0,
    ) -> Dict[str, float]:
        """
        Execute one Collect → Train → Discard cycle per Spec §7.

        This is the core offline training loop:
        1. Reset hidden state (fresh start for random t_s)
        2. Sequential forward pass through trajectory
        3. Compute masked losses
        4. Backprop and update
        5. Discard trajectory buffer

        Args:
            batch: TrajectoryBatch from sample_trajectory_batch()
            data_tensors: Full data tensors for feature window extraction

        Returns:
            Dict of loss values and metrics
        """
        B = batch.stock_ochlv.size(0)
        T_m = batch.stock_ochlv.size(1)
        _N = batch.stock_ochlv.size(
            2
        )  # N stocks (unused but kept for shape documentation)
        
        self.observer.mafia_model.train()

        # ============================================================
        # FIX: Clear gradients at START before accumulating via backward()
        # ============================================================
        self.optimizer.zero_grad()

        # ============================================================
        # STEP 1: RESET - Fresh hidden state for new random trajectories
        # Per Spec §6: "Reset trạng thái hidden_state ở đầu mỗi Batch"
        # ============================================================
        if hasattr(self.observer.mafia_model, "reset_temporal_state"):
            self.observer.mafia_model.reset_temporal_state()
        
        # Reset dashboard tracking state for fresh batch
        self._last_rebalance_portfolio = {}

        # ============================================================
        # STEP 2: COLLECT Phase - Sequential forward through trajectory
        # ============================================================

        # Transient buffer (will be discarded after training)
        collected_topk_scores = []  # List of (B, K) tensors with grad
        collected_topk_indices = []  # List of (B, K) int tensors
        collected_risk_eta = []  # List of (B,) tensors with grad
        collected_direction_logits = []  # List of (B, 3) tensors
        
        # Initialize 'prev_indices' for turnover calculation (held portfolio)
        # At t=0, previous portfolio is empty set.
        prev_indices = [
            th.tensor([], dtype=th.long, device=self.device) 
            for _ in range(B)
        ] 

        # Initialize Context Buffer for Temporal Augmentation (Spec 3.6)
        # Cold start with zeros for each batch item
        W_route = int(getattr(self.config, "router_context_window", 5))
        # Get D from model or config
        if hasattr(self.observer.mafia_model, 'D'):
             mafia_D = self.observer.mafia_model.D
        else:
             mafia_D = int(getattr(self.config, "mafia_D", 128))

        batch_context_buffer = th.zeros(B, W_route, mafia_D, device=self.device)

        # Get full OCHLV data for feature window extraction
        full_ochlv = data_tensors["ochlv"]  # (T_total, N, 5)
        full_market_ochlv = data_tensors.get("market_ochlv")

        # Track dynamic Regime Shift Logic
        # 1. Volatility Shock: Pre-calculated in batch.vol_shock_mask
        # 2. Direction Reversal: Calculated online with N_confirm confirmation window (Spec §8.2.1)
        N_confirm = int(getattr(self.config, "regime_confirmation_window", 3))
        # Direction history buffer: (B, N_confirm+1) - stores last N_confirm+1 predictions
        # Initialize with Side(1) as neutral state
        direction_history = th.ones(B, N_confirm + 1, dtype=th.long, device=self.device)
        effective_rebalance_masks = [] # List of (B,) tensors

        # Spec 5.2: Track actual rebalance event count for PG normalization
        # L_PG normalized by: Σ m_t + ε (actual event count, not expected)
        actual_rebal_count = 0.0  # Running count of rebalance events across batch

        # FIX: Pre-compute EXPECTED rebalance count from static masks for stable normalization
        # This is used for loss normalization DURING the loop (avoids running count bias).
        # Static triggers: schedule (rebalance_mask) + vol_shock + dc_mask
        # Note: Direction reversal triggers are computed online and not included here,
        # but they are relatively rare compared to scheduled/event triggers.
        static_trigger_mask = (
            (batch.rebalance_mask > 0.5) |  # Schedule triggers
            (batch.vol_shock_mask > 0.5) |  # Volatility shock triggers
            (batch.dc_mask > 0.5)           # Structural break triggers
        )  # (B, T_m) Boolean
        expected_rebal_count = static_trigger_mask.float().sum().item()  # Scalar
        # Ensure non-zero for normalization
        expected_rebal_count = max(expected_rebal_count, 1.0)
        collected_market_context = [] # List of (B, D) tensors
        
        # Reward breakdown collection
        collected_raw_returns = []
        collected_turnovers = []
        collected_symdiffs = []
        collected_rewards = []
        collected_baselines = []
        collected_advantages = []
        
        # Loss collections (needed for backward pass later)
        pg_losses = []
        risk_losses = []
        dir_losses = []
        
        # Dynamic Schedule Tracking (Resets on ANY trigger)
        # Initialize to 0. At t=0, we force rebalance.
        days_since_last_rebal = th.zeros(B, dtype=th.long, device=self.device)
        rebal_interval = self.rebalance_interval

        for t in range(T_m):
            
            # Get feature window [t_s + t - T_w + 1, t_s + t + 1] for each batch item
            # Since we need T_w history, we need to index into full data
            batch_windows = []
            batch_windows_mkt = []
            for b in range(B):
                t_s = batch.start_indices[b].item()
                window_start = max(0, t_s + t - self.T_w + 1)
                window_end = t_s + t + 1
                window = full_ochlv[window_start:window_end]  # (T_w or less, N, 5)

                # Pad if window is shorter than T_w
                if window.size(0) < self.T_w:
                    pad_size = self.T_w - window.size(0)
                    pad = window[0:1].expand(pad_size, -1, -1)
                    window = th.cat([pad, window], dim=0)

                # Reshape to (N, 5, T_w) for MAFIA model
                window = window.permute(1, 2, 0)  # (N, 5, T_w)
                batch_windows.append(window)

                # --- Market Data Window Extraction ---
                if full_market_ochlv is not None:
                    # Extract window for market (same indices)
                    # full_market_ochlv is (T_total, 1, 5)
                    window_mkt = full_market_ochlv[window_start:window_end]
                    
                    if window_mkt.size(0) < self.T_w:
                        pad_size = self.T_w - window_mkt.size(0)
                        pad = window_mkt[0:1].expand(pad_size, -1, -1)
                        window_mkt = th.cat([pad, window_mkt], dim=0)
                    
                    # Reshape to (1, 5, T_w)
                    window_mkt = window_mkt.permute(1, 2, 0)
                    batch_windows_mkt.append(window_mkt)

            # Stack batch: (B, N, 5, T_w) and move to device
            ochlv_batch = th.stack(batch_windows, dim=0).to(self.device)

            market_ochlv_batch = None
            if batch_windows_mkt:
                market_ochlv_batch = th.stack(batch_windows_mkt, dim=0).to(self.device) # (B, 1, 5, T_w)

            # --- Check Regular Schedule Mask (Dynamic) ---
            # sched_mask = batch.rebalance_mask[:, t]  # OLD: Static mask
            
            if t == 0:
                # Force rebalance at start of trajectory
                sched_trigger = th.ones(B, dtype=th.bool, device=self.device)
                days_since_last_rebal[:] = 0
            else:
                days_since_last_rebal += 1
                sched_trigger = (days_since_last_rebal >= rebal_interval)

            # --- Forward Pass through MAFIA model ---
            # Model expects: ochlv_data (B, N, 5, T_w)
            with th.set_grad_enabled(True):
                # Temporary forward to get direction logits for trigger check (if needed)
                # But here we do full forward because we need outputs anyway.
                # However, force_topk_indices depends on whether we rebalance.
                # Logic:
                # 1. Determine trigger based on *previous* step info (not possible for current step outputs yet) 
                # OR trigger based on current input data (Vol) + current output direction (Reversal).
                # The spec says "when direction reverses". This implies we see the new direction.
                # But to select stocks, we need to know IF we rebalance.
                # If Rebalance -> force_indices = None. Else -> force_indices = prev.
                
                # To handle "Reversal Trigger" which depends on CURRENT prediction:
                # We essentially accept that if reversal happens at T, we trigger rebalance at T.
                # BUT `force_topk_indices` must be decided lazily? No, model forward needs it.
                # Solution: Run forward with force_topk_indices=None (optimistic)?
                # No, that breaks the "holding" logic simulation.
                # Better Proxy: Use the direction prediction from this step to trigger NEXT step?
                # Or, assume Rebalance decision happens AFTER observing market state (Direction).
                # In Spec 8: "Trigger: Direction Reversal... When triggered -> Force Rebalance"
                # This implies we see the reversal, THEN we select new Top-K.
                # So we might need a 2-pass or rely on the fact that if we rebalance, we ignore prev indices.
                
                # For efficient batch training, let's look at triggers available NOW:
                # 1. Schedule (Fixed)
                # 2. Vol Shock (Pre-calculated from Input)
                # 3. Direction Reversal (Needs current output).
                
                # We will perform the forward pass *optimistically* assuming NO rebalance (using prev indices),
                # UNLESS schedule or vol shock is active.
                # Then check direction reversal. If reversal found, we might need to *re-run* forward without forcing?
                # Actually, MAFIA model returns `market_scores_full` and `logits` regardless of forcing.
                # `force_topk_indices` only affects `topk_indices` and `topk_scores` output.
                # WE CAN RE-SAMPLE if trigger activates!
                
                # 1. Base Rebalance needed?
                vol_trigger = batch.vol_shock_mask[:, t] > 0.5 # (B,)
                dc_trigger = batch.dc_mask[:, t] > 0.5 # (B,) - Structural Break (Downward)
                # sched_trigger is already boolean (B,) computed above
                
                # Preliminary trigger (Explicit triggers known at input)
                pre_trigger = vol_trigger | sched_trigger | dc_trigger # (B,)
                
                # Prepare force indices for this step (initially based on pre_trigger)
                # If trigger -> None (fresh selection). Else -> prev indices.
                current_force_indices = None
                if t > 0 and collected_topk_indices:
                     prev_indices = collected_topk_indices[-1] # (B, K)
                     # We want to force indices WHERE NOT pre_trigger
                     # If pre_trigger[b] is True, we want fresh (None equivalent for that batch item).
                     # MAFIA model `force_topk_indices` argument usually takes a batch of indices.
                     # If we pass indices, it uses them. If we pass None, it selects fresh.
                     # Partial forcing isn't supported by standard MAFIA `predict` usually, let's check.
                     # Checking `mafia_model.forward`:
                     # ... `if force_topk_indices is not None: selected_indices = force_topk_indices`
                     # It applies to the WHOLE batch usually or per-item?
                     # Standard implementation applies `gather` using provided indices.
                     # If we want mixed behavior (some rebalance, some hold), we need `force_topk_indices` 
                     # to contain:
                     # - Prev indices for Hold items
                     # - New selected indices for Rebalance items (Wait, we don't have them yet!)
                     
                     # Implementation constraint: The model architecture usually supports either "All Force" or "All Select"
                     # or "Force with provided matrix".
                     # To support "Select New" for some batch items, we conceptually need to let the model 
                     # generate logits, sample indices, AND THEN overwrite the indices for "Hold" items with prev ones.
                     
                     # Correct approach for Mixed Batch:
                     # 1. Run Forward with force_topk_indices=None (Always generate fresh Top-K candidates)
                     # 2. Get `new_indices`
                     # 3. For items where `rebalance=False`: Overwrite `new_indices` with `prev_indices`.
                     # 4. This ensures `topk_scores` etc match the actual portfolio held.
                     pass 

                # Build explicit signals for Direction Head (Spec 3.5)
                # explicit_signals = [vol_shock_flag, dc_event_flag] for timestep t
                vol_flag = batch.vol_shock_mask[:, t:t+1]  # (B, 1)
                dc_flag = batch.dc_mask[:, t:t+1]          # (B, 1)
                explicit_signals = th.cat([vol_flag, dc_flag], dim=-1)  # (B, 2)

                # Pass buffer to model (Spec 3.6)
                outputs = self.observer.mafia_model(
                    ochlv_data=ochlv_batch,
                    market_index_ochlv_data=market_ochlv_batch,
                    force_topk_indices=None, # Always fresh selection first
                    router_context_buffer=batch_context_buffer, # Batched buffer for temporal augmentation
                    explicit_signals=explicit_signals, # Direction Head signals (Spec 3.5)
                )

                (
                    market_vector,  # (B, N)
                    risk_eta,  # (B,)
                    market_scores_full,  # (B, N)
                    direction_logits,  # (B, 3)
                    market_context,  # (B, D)
                    topk_indices,  # (B, K) - These are FRESH candidates
                    topk_embeddings,  # (B, K, D)
                    topk_scores,  # (B, K)
                ) = outputs

                # Temperature scaling for direction head (sharpen/flatten)
                if self.direction_temperature != 1.0:
                    direction_logits = direction_logits / self.direction_temperature
                
                # Save FRESH candidates before holding logic may overwrite
                fresh_topk_indices = topk_indices.clone()
                collected_market_context.append(market_context) # (B, D)
                
                # ============================================================
                # SPEC 3.6.3: Explicit detach after each timestep (Training Collection)
                # Purpose: Truncate gradient graph while preserving state values
                # ============================================================
                if hasattr(self.observer.mafia_model, "detach_temporal_state"):
                    self.observer.mafia_model.detach_temporal_state()
                # SPEC 3.6.3.2: Also detach context_buffer contents
                batch_context_buffer = batch_context_buffer.detach()
                
                # ============================================================
                # (Context logging removed - now handled by realtime dashboard)
                # ============================================================
                
                # --- Check Direction Reversal Trigger with Confirmation Window (Spec §8.2.1) ---
                # Requirement: Bull↔Bear transition maintained for N_confirm consecutive sessions
                pred_state = th.argmax(direction_logits, dim=-1)  # (B,)
                # Clamp to valid range [0, 2] for safety (handles NaN/unexpected values)
                pred_state = th.clamp(pred_state, 0, 2)

                # Update direction history buffer (FIFO: shift left, append new)
                direction_history = th.cat([direction_history[:, 1:], pred_state.unsqueeze(1)], dim=1)

                # Extract previous regime for logging (before confirmation window)
                prev_pred_state = direction_history[:, 0]  # (B,) - oldest regime in buffer

                # Check reversal with confirmation window
                # Need at least N_confirm+1 predictions to detect reversal
                if t < N_confirm:
                    dir_trigger = th.zeros(B, dtype=th.bool, device=self.device)
                else:
                    # last_seq = direction_history[:, -N_confirm:]  # Last N_confirm predictions
                    # prev_regime = direction_history[:, -(N_confirm+1)]  # Prediction before confirmation window
                    last_seq = direction_history[:, 1:]  # (B, N_confirm) - last N_confirm predictions
                    prev_regime = direction_history[:, 0]  # (B,) - prediction before window

                    BULL, BEAR = 2, 0

                    # Bear → Bull: prev_regime was Bear, all last N_confirm are Bull
                    all_bull = (last_seq == BULL).all(dim=1)  # (B,)
                    was_bear = (prev_regime == BEAR)  # (B,)
                    bear_to_bull = all_bull & was_bear

                    # Bull → Bear: prev_regime was Bull, all last N_confirm are Bear
                    all_bear = (last_seq == BEAR).all(dim=1)  # (B,)
                    was_bull = (prev_regime == BULL)  # (B,)
                    bull_to_bear = all_bear & was_bull

                    dir_trigger = bear_to_bull | bull_to_bear
                
                # --- Final Effective Rebalance Mask ---
                effective_mask = pre_trigger | dir_trigger # (B,) Boolean
                effective_mask_float = effective_mask.float() # (B,) 1.0 or 0.0
                effective_rebalance_masks.append(effective_mask_float)

                # Spec 5.2: Accumulate actual rebalance event count
                # Sum across batch dimension for this timestep
                actual_rebal_count += effective_mask_float.sum().item()

                # Reset Schedule Counter where rebalance occurred
                days_since_last_rebal[effective_mask] = 0
                
                # --- Apply Holding Logic ---
                # If effective_mask[b] is 0 -> We MUST hold previous portfolio.
                if t > 0 and collected_topk_indices:
                    prev_idx = collected_topk_indices[-1] # (B, K)
                    prev_sc = collected_topk_scores[-1] # (B, K)
                    
                    # Overwrite where mask is 0
                    # We need to broadcast mask to (B, K)
                    mask_k = effective_mask.unsqueeze(1).expand(-1, self.K) # (B, K)
                    
                    final_indices = th.where(mask_k, topk_indices, prev_idx)
                    
                    # For scores, we ideally want the scores associated with the *held* indices at current time.
                    # market_scores_full contains current logits.
                    # We can gather new scores for old indices.
                    #   new_scores_for_old_idx = gather(market_scores_full, prev_idx)
                    #   BUT topk_scores usually implies "active choice".
                    #   Simpler: Use new scores for the held indices (passive evolution of weights? No, usually fixed shares or fixed weights?)
                    #   Spec says: "Hold portfolio". Usually means weights drift or consistent logic.
                    #   Here, let's keep it simple: gather current probabilities for the held indices.
                    
                    # Re-gather scores if we overwrote indices
                    if (~effective_mask).any():
                         # (B, N) log_probs or scores -> gather(final_indices)
                         # topk_scores output from model is already gathered for `topk_indices`.
                         # We need to gather for `final_indices`.
                         # Note: `market_scores_full` is (B, N) softmaxed probabilities
                         final_scores = th.gather(market_scores_full, 1, final_indices)
                    else:
                         final_scores = topk_scores
                         
                    topk_indices = final_indices
                    topk_scores = final_scores

            # Store in transient buffer
            risk_eta = th.nan_to_num(risk_eta, nan=1.0, posinf=1.0, neginf=1.0)
            collected_topk_scores.append(topk_scores)
            collected_topk_indices.append(topk_indices)
            collected_risk_eta.append(risk_eta)
            collected_direction_logits.append(direction_logits)

            # ============================================================
            # DETAILED TRAJECTORY LOGGING - ALL TIMESTEPS
            # ============================================================
            if getattr(self.config, "log_trajectory_details", False):
                # Log EVERY timestep for complete trajectory visibility
                for b in range(B):
                    trigger_info = []
                    if sched_trigger[b].item():
                        trigger_info.append("SCHEDULE")
                    if vol_trigger[b].item():
                        trigger_info.append("VOL_SHOCK")
                # ============================================================
                # REAL-TIME METRICS & LOGGING (Moved Inside Loop)
                # ============================================================
                
                # 1. Calculate Rewards (R_t, Baseline) immediately
                start_idx = t + 1
                end_idx = t + 1 + self.horizon
                future_returns_slice = batch.price_returns[:, start_idx:end_idx, :] # (B, h, N)
            # ============================================================
            # REAL-TIME METRICS & LOGGING (Moved Inside Loop)
            # ============================================================
            
            # 1. Calculate Rewards (R_t, Baseline) immediately
            start_idx = t + 1
            end_idx = t + 1 + self.horizon
            future_returns_slice = batch.price_returns[:, start_idx:end_idx, :] # (B, h, N)
            market_returns_slice = batch.market_returns[:, start_idx:end_idx]   # (B, h)
            h_len = future_returns_slice.size(1)
            
            # Use final_indices which accounts for holding logic
            # We need to compute penalties (Turnover, SymDiff) too.
            # Need access to PREVIOUS final indices. 
            # We store collected_topk_indices, but that's for stacking.
            # Let's use `collected_topk_indices[-1]` BEFORE appending current `final_indices`?
            # Wait, we haven't appended `final_indices` to `collected_topk_indices` yet at this point in code?
            # Check lines 1005ish: `collected_topk_indices.append(final_indices)` is needed.
            
            if h_len == 0:
                R_net = th.zeros(B, device=self.device)
                baseline = th.zeros(B, device=self.device)
                R_raw = th.zeros(B, device=self.device)
            else:
                topk_idx = topk_indices # (B, K)
                gather_idx = topk_idx.unsqueeze(1).expand(-1, h_len, -1)
                selected_returns = th.gather(future_returns_slice, 2, gather_idx)
                compounded_stock_returns = th.prod(1 + selected_returns, dim=1) - 1
                # Apply S_reward scaling (Spec §5.1.1) to amplify gradient signal
                R_raw = self.scale_factor_reward * compounded_stock_returns.mean(dim=1)
                baseline = self.scale_factor_reward * (th.prod(1 + market_returns_slice, dim=1) - 1)
            
            # Penalties
            turnover = th.zeros(B, device=self.device)
            symdiff = th.zeros(B, device=self.device)
            
            if t > 0:
                # Compute Turnover & SymDiff
                for b_i in range(B):
                     curr_set = set(topk_indices[b_i].cpu().tolist())
                     prev_set = set(prev_indices[b_i].cpu().tolist()) 
                     
                     held_count = len(curr_set.intersection(prev_set))
                     t_val = (self.K - held_count) / self.K
                     turnover[b_i] = t_val
                     s_val = (2 * (self.K - held_count)) / self.K
                     symdiff[b_i] = s_val

            # Advantage (Spec §5.1.1)
            # A_raw = (R_t - λ_epoch × Penalties) - b_t
            # λ_epoch: Curriculum penalty weight (0 in warmup, ramps to 1)
            lambda_epoch = self._current_lambda_epoch
            penalty = lambda_epoch * (self.alpha_turnover * turnover + self.alpha_change * symdiff)
            R_net = R_raw - penalty
            A_t_raw = R_net - baseline
            
            # Normalize Advantage (Batch statistics)
            A_mean = A_t_raw.mean()
            eps = 1e-8
            if B > 1:
                A_std = A_t_raw.std() + eps
            else:
                A_std = 1.0
            A_t = (A_t_raw - A_mean) / A_std

            # Store for Post-Loop Loss Calc (still needed for backprop)
            collected_raw_returns.append(R_raw)
            collected_turnovers.append(turnover)
            collected_symdiffs.append(symdiff)
            collected_rewards.append(R_net)
            collected_baselines.append(baseline)
            collected_advantages.append(A_t)
            
            # -----------------------------------------------------------
            # REAL-TIME LOGGING
            # -----------------------------------------------------------
            b = 0
            is_rebalance_b0 = effective_mask[b].item() > 0.5
            target_dir_idx = int(min(max(batch.direction_labels[b, t].item(), 0), 2))
            pred_state_b = int(min(max(pred_state[b].item(), 0), 2))

            correct_dir = "✅" if pred_state_b == target_dir_idx else "❌"
            dir_pred_str = ["Bear", "Side", "Bull"][pred_state_b]
            target_pred_str = ["Bear", "Side", "Bull"][target_dir_idx]
            
            if correct_dir == "❌":
                dir_pred_str += f" (GT: {target_pred_str})"
            
            # Triggers (Active for Batch 0?)
            # NOTE: We want to know WHY it held or rebalanced.
            # Triggers might exist even if rebalance is masked (e.g. suppressed by cool-off?)
            # But here `trigger_str` is derived from masks.
            # If `sched_mask` is 1, it IS a trigger.
            
            triggers = []
            if sched_trigger[b].item():
                 triggers.append("Schedule")
            if batch.vol_shock_mask[b, t] > 0.5:
                 triggers.append("Vol Shock")
            if batch.dc_mask[b, t] > 0.5:
                 triggers.append("Struct Break (DC)")
            if dir_trigger[b].item():
                 prev_s_idx = int(min(max(prev_pred_state[b].item(), 0), 2))
                 prev_s = ["Bear", "Side", "Bull"][prev_s_idx]
                 triggers.append(f"Regime Shift ({prev_s}->{dir_pred_str})")
            elif t==0:
                 triggers.append("Regime Shift (Initial)")
                 
            trigger_str = " | ".join(triggers) if triggers else "None"
            trigger_desc = trigger_str # Always show triggers
            


            if is_rebalance_b0:
                 action_str = "🔄 REBALANCE"
            else:
                 action_str = "⏸  HOLD"

            # Context Difference (Dynamic Check)
            curr_c_mkt = market_context[b]
            if not hasattr(self, '_prev_c_mkt_b0'): self._prev_c_mkt_b0 = None
            
            if self._prev_c_mkt_b0 is not None:
                c_mkt_diff = (curr_c_mkt - self._prev_c_mkt_b0).norm().item()
            else:
                c_mkt_diff = 0.0
            
            # Update prev for next step
            self._prev_c_mkt_b0 = curr_c_mkt.detach() # Detach to be safe

            # Portfolio
            sel_stocks = topk_indices[b].cpu().tolist()
            
            # Hold count for display
            if not hasattr(self, '_last_rebalance_portfolio'): self._last_rebalance_portfolio = {}
            if is_rebalance_b0:
                prev_port = self._last_rebalance_portfolio.get(b, set())
                curr_set = set(sel_stocks)
                if prev_port:
                    held = len(curr_set.intersection(prev_port))
                    changed = len(curr_set) - held
                else:
                    held, changed = 0, len(sel_stocks)
                self._last_rebalance_portfolio[b] = curr_set
            else:
                held = len(sel_stocks)
                changed = 0

            # Symbols
            stock_list = batch.stock_list if batch.stock_list else None
            # Prepare stock string but maybe don't print it in quiet mode
            if stock_list:
                syms = [stock_list[idx] for idx in sel_stocks]
                if len(syms) > 5:
                     stock_str = ", ".join(syms[:5]) + f", ... (+{len(syms)-5})"
                else:
                     stock_str = ", ".join(syms)
            else:
                stock_str = ", ".join([f"S{i}" for i in sel_stocks[:5]])

            c_mkt_norm = market_context[b].norm().item()
            eta_val = risk_eta[b].item()
            
            # PG Loss Estimate (Pend)
            log_probs = th.log(th.gather(market_scores_full, 1, topk_indices) + eps) # (B, K)
            imp_weights = th.ones_like(log_probs)
            # Spec §5.1.1: Advantage must be detached to prevent gradient flow through it
            # This ensures model learns to change π(a|s), not "hack" the reward
            pg_term = -(log_probs * imp_weights * A_t.detach().unsqueeze(1)).mean(dim=1)  # (B,)

            # Entropy bonus on full distribution (spec 5.1.1)
            # H(π) = -Σ p_i × log(p_i) - encourages exploration over N stocks
            entropy = -(market_scores_full * th.log(market_scores_full + eps)).sum(dim=1)  # (B,)

            # L_PG = pg_term - β_ent × H(π) (subtract entropy to bonus exploration)
            step_loss_val = pg_term - self.beta_entropy * entropy  # (B,)
            
            # Auxiliary Losses
            risk_target = batch.risk_targets[:, t]
            risk_loss = self.risk_criterion(risk_eta, risk_target) * self.risk_scaling_factor
            
            dir_target = batch.direction_labels[:, t]
            dir_loss = self.dir_criterion(direction_logits, dir_target)

            pg_val = step_loss_val[b].item() if is_rebalance_b0 else 0.0

            # QUIET MODE LOGIC
            # If HOLD and no specific tricky triggers (schedule is boring if just daily check),
            # but here "Schedule" means REBALANCE interval hit.
            # If Action is HOLD, generally we are quiet.
            # Exception: Maybe if a Vol Shock happened but we logic-ed into holding?
            # For now: Quiet if HOLD. Verbose if REBALANCE.
            
            import sys
            


            # Universal Verbose Update (Block)
            smart_print("-" * 100)
            
            # --- Hold Counter Logic ---
            if not hasattr(self, '_hold_counters'):
                self._hold_counters = {}
            
            if triggers:
                # Reset counter on ANY trigger (Schedule, Vol Shock, DC, Regime Shift)
                self._hold_counters[b] = 0
                display_trigger = trigger_desc
            else:
                # Increment counter on Hold
                current_hold = self._hold_counters.get(b, 0) + 1
                self._hold_counters[b] = current_hold
                # Append Hold Day info
                # Cycle length is usually rebalance_interval (14)
                # But we just show count.
                display_trigger = f"None (Hold Day {current_hold})"
                
            header = f"  ⏱  [Epoch {epoch_idx}] Batch {batch_idx+1} | Step {t+1:3d}/{T_m} | {action_str}"
            # Show sample trajectory info (trajectory 0 of B)
            smart_print(f"{header}  [Sample 1/{B} trajectories]")
            smart_print(f"     Trigger: {display_trigger}")
            
            # Spec: R_t = RawReturn. A_t = (R_net) - Baseline.
            # To align with user mental model, we show Raw -> Costs -> Net -> Adv
            
            # 1. Raw Performance
            raw_str = f"RawReturn={R_raw[b].item():+.4f}"
            
            # 2. Costs/Penalties (scaled by curriculum λ_epoch)
            turn_pen = turnover[b].item() * self.alpha_turnover * lambda_epoch
            diff_pen = symdiff[b].item() * self.alpha_change * lambda_epoch
            cost_str = f"TurnPen={turn_pen:.4f} | SymDiffPen={diff_pen:.4f}"
            
            # 3. Net & Advantage
            # R_net = Raw - Costs
            r_net_val = R_net[b].item() 
            net_str = f"NetReward={r_net_val:+.4f}"
            
            adv_str = (
                f"Baseline={baseline[b].item():+.4f} | "
                f"Advantage={A_t_raw[b].item():+.4f}"
            )
            
            smart_print(f"     📈 Market:    {raw_str} | CtxDiff={c_mkt_diff:.3f}")
            smart_print(f"     💸 Costs:     {cost_str}")
            smart_print(f"     ⚖️  Outcome:   {net_str} | {adv_str}")
            smart_print(f"     📉 Losses:    L_PG={pg_val:.4f} | L_risk={risk_loss.item():.4f} | L_dir={dir_loss.item():.4f}")
            
            # Portfolio & State
            smart_print(f"     📊 Portfolio: Held={held}, Added={changed} | {stock_str}")
            
            # Predictions
            probs = th.nn.functional.softmax(direction_logits[b], dim=-1)
            logits_str = f"Bear={probs[0]:.2f}, Side={probs[1]:.2f}, Bull={probs[2]:.2f}"
            smart_print(f"     🔮 Forecast:  Risk(η)={eta_val:.2f}(Tg={risk_target[b].item():.2f}) | Dir={dir_pred_str}{correct_dir} [{logits_str}]")

            if t == T_m - 1:
                # End of trajectory - show batch-wide statistics
                smart_print("-" * 100)
                smart_print(f"  📊 BATCH SUMMARY (All {B} trajectories processed in PARALLEL):")

                # Compute batch-wide stats from the accumulated tensors
                all_returns = th.stack(collected_raw_returns, dim=1)  # (B, T_m)
                all_rewards = th.stack(collected_rewards, dim=1)  # (B, T_m) - Net rewards
                all_baselines = th.stack(collected_baselines, dim=1)  # (B, T_m)

                # Compute RAW advantage (before normalization) for meaningful stats
                all_raw_advantages = all_rewards - all_baselines  # (B, T_m)

                # Mean across all trajectories
                mean_return = all_returns.mean().item()
                std_return = all_returns.std().item()
                mean_raw_advantage = all_raw_advantages.mean().item()

                # Direction accuracy across all B trajectories at last step
                dir_preds = pred_state  # (B,) current predictions
                dir_targets = batch.direction_labels[:, t]  # (B,) targets
                dir_accuracy = (dir_preds == dir_targets).float().mean().item() * 100

                # Risk prediction stats across all B
                mean_risk_eta = risk_eta.mean().item()

                # Determine if portfolio is beating market
                adv_status = "🟢 Beating Market" if mean_raw_advantage > 0 else "🔴 Underperforming"

                smart_print(f"     📈 Returns:    Mean={mean_return:+.4f} | Std={std_return:.4f}")
                smart_print(f"     ⚖️  Advantage:  Mean={mean_raw_advantage:+.4f} ({adv_status})")
                smart_print(f"     🎯 Direction:  Accuracy={dir_accuracy:.1f}% (across {B} trajectories)")
                smart_print(f"     📉 Risk η:     Mean={mean_risk_eta:.3f}")
                smart_print("=" * 100)
        
            # ===========================================================
            # GRADIENT ACCUMULATION (O(1) Memory)
            # ===========================================================
            # Instead of stacking graphs for 128 steps (O(T) memory), we backward()
            # at each step and accumulate gradients in the parameters.

            # Loss Normalization Factors (Spec 5.2 compliant)
            # Risk/Dir Loss is averaged over *all days* (B × T_m)
            norm_risk = 1.0 / T_m
            norm_dir = 1.0 / T_m

            # Step Loss Components
            # step_loss_val is (B,) tensor of PG losses
            # effective_mask_float is (B,) 1.0/0.0 mask
            # FIX: Use .sum() NOT .mean() to match Spec §5.2 formula:
            # L_PG_norm = Σ(m×L) / (Σm + ε)  -- NOT  Σ(m×L) / (B × Σm)
            # Using .mean() would add extra factor B in denominator!
            step_pg_loss = (step_loss_val * effective_mask_float).sum()

            # FIX: Use pre-computed expected_rebal_count for stable PG normalization (Spec 5.2)
            # Problem: actual_rebal_count is running count, causing early timesteps to be over-weighted.
            # Solution: Use expected count from static masks (schedule + vol_shock + dc) computed BEFORE loop.
            # This is stable and doesn't change during the loop.
            eps_norm = 1e-8
            norm_pg = 1.0 / (expected_rebal_count + eps_norm)

            # Weighted Sum with proper normalization
            # Note: We must retain graph for backward()
            step_total_loss = (
                (step_pg_loss * norm_pg * self.lambda_pg) +
                (risk_loss * norm_risk * self.lambda_risk) +
                (dir_loss * norm_dir * self.lambda_dir)
            )
            
            # Backward Pass (Per-Step)
            # This accumulates gradients in params. Graph is freed immediately after.
            if step_total_loss.item() != 0.0: # Optimization: skip if zero
                 try:
                     step_total_loss.backward()
                 except RuntimeError as e:
                     if "nan" in str(e).lower() or "inf" in str(e).lower():
                         smart_print(f"[WARN] NaN/Inf in backward pass (t={t}). Skipping step.")
                     else:
                         raise e

            # Store Metrics (Float only - NO GRAPH)
            # We multiply back by T_m / effective_count to reconstruct "epoch average" for logging
            pg_losses.append(step_pg_loss.item()) # Raw value
            risk_losses.append(risk_loss.item())
            dir_losses.append(dir_loss.item())
            
            # Update previous indices for next timestep turnover calculation
            prev_indices = topk_indices

            # Store for CSV export
            if not hasattr(self, '_trajectory_log'):
                self._trajectory_log = []
            
            self._trajectory_log.append({
                'batch_idx': b,
                'timestep': t,
                'trigger': trigger_str,
                'rebalanced': is_rebalance_b0,
                'selected_stocks': ",".join([str(idx) for idx in sel_stocks]),
                'risk_eta': risk_eta[b].item(),
                'dir_logit_bear': direction_logits[b, 0].item(),
                'dir_logit_side': direction_logits[b, 1].item(),
                'dir_logit_bull': direction_logits[b, 2].item(),
                'c_mkt_norm': market_context[b].norm().item(),
                'c_mkt_diff': c_mkt_diff,
                'raw_return': R_raw[b].item(),
                'turnover_penalty': turnover[b].item() * self.alpha_turnover,
                'symdiff_penalty': symdiff[b].item() * self.alpha_change,
                'net_reward': R_net[b].item(),
                'baseline': baseline[b].item(),
                'advantage': A_t_raw[b].item(),
                'advantage_norm': A_t[b].item(),
                'pg_loss': pg_val, # Note: pg_val is tensor, need item()
                'risk_loss': risk_loss.item(),
                'dir_loss': dir_loss.item(),
            })

            # Note: Hidden state is automatically maintained inside BiLSTM
            # via _cached_state (detached for TBPTT)
            
            # Update Context Buffer via FIFO (Spec 3.6.2.2)
            # Remove oldest ([:, 0, :]), append new (market_context)
            # market_context is (B, D). Need (B, 1, D)
            # CAUTION: Detach market_context before adding to buffer to prevent growing graph through time
            # beyond the relevant window or intended BPTT.
            # (TBPTT implies we cut gradients. Buffer acts like state memory. Standard is to detach unless BPTT desired through it.)
            
            history_part = batch_context_buffer[:, 1:, :] # (B, W-1, D)
            new_part = market_context.detach().unsqueeze(1) # (B, 1, D)
            batch_context_buffer = th.cat([history_part, new_part], dim=1)

        # ============================================================
        # STEP 3: OPTIMIZER STEP (After accumulating all T_m gradients)
        # ============================================================
        
        # ===== Optimizer Step (gradients already accumulated via backward() in loop) =====
        try:
            # Clip gradients
            grad_norm = th.nn.utils.clip_grad_norm_(self.observer.mafia_model.parameters(), max_norm=1.0)
            if th.is_tensor(grad_norm):
                grad_norm = grad_norm.item()
            
            # Update weights
            self.optimizer.step()
            # If an LR scheduler is used, step it here. Assuming it's self.lr_scheduler
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                self.lr_scheduler.step()
        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                smart_print(f"[WARN] NaN/Inf detected during optimizer step. Skipping update. Error: {e}")
                # Optionally, you might want to re-zero gradients if the step was skipped
                self.optimizer.zero_grad() 
                skipped_update = True
                grad_norm = -1.0 # Indicator for failure
            else:
                raise e
        else:
            skipped_update = False
        
        # Calculate Average Metrics for Log
        # Note: pg_losses list contains 0s for non-rebal days. 
        # We want avg over rebal days only.
        pg_values = [v for v in pg_losses if v != 0.0]
        avg_pg = sum(pg_values) / len(pg_values) if pg_values else 0.0
        
        avg_risk = sum(risk_losses) / len(risk_losses)
        avg_dir = sum(dir_losses) / len(dir_losses)
        
        # Reconstruct approximate total loss for display
        # (This won't exactly match the backwarded loss due to estimator norm, but close enough for logs)
        total_disp = (
            self.lambda_pg * avg_pg +
            self.lambda_risk * avg_risk +
            self.lambda_dir * avg_dir
        )
        
        metrics = {
            "loss_total": total_disp,
            "loss_pg": avg_pg,
            "loss_risk": avg_risk,
            "loss_dir": avg_dir,
            "loss_dir": avg_dir,
            "loss_dir": avg_dir,
            "rebalance_ratio": sum([m.float().mean().item() for m in effective_rebalance_masks]) / T_m,
            "mean_risk_eta": th.stack(collected_risk_eta).mean().item() if collected_risk_eta else 0.0,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "skipped_update": 1.0 if skipped_update else 0.0
        }

        if hasattr(self, '_trajectory_log') and self._trajectory_log:
            for entry in self._trajectory_log:
                entry['grad_norm'] = grad_norm
                entry['skipped_update'] = 1.0 if skipped_update else 0.0

        # ============================================================
        # TENSORBOARD LOGGING (Batch-level)
        # ============================================================
        if self.tb_logger is not None:
            # Calculate global step (epoch * batches_per_epoch + batch_idx)
            global_step = epoch_idx * 1000 + batch_idx  # Approximate
            
            # Log scalar losses
            self.tb_logger.log_scalar("loss/total", total_disp, global_step, phase="train")
            self.tb_logger.log_scalar("loss/pg", avg_pg, global_step, phase="train")
            self.tb_logger.log_scalar("loss/risk", avg_risk, global_step, phase="train")
            self.tb_logger.log_scalar("loss/direction", avg_dir, global_step, phase="train")
            
            # Log gradient metrics
            self.tb_logger.log_scalar("gradients/norm", grad_norm, global_step, phase="train")
            self.tb_logger.log_scalar("gradients/skipped_update", 1.0 if skipped_update else 0.0, global_step, phase="train")
            
            # Log learning rate
            self.tb_logger.log_scalar("optimizer/learning_rate", self.optimizer.param_groups[0]["lr"], global_step, phase="train")
            
            # Log portfolio metrics
            self.tb_logger.log_scalar("portfolio/rebalance_ratio", metrics["rebalance_ratio"], global_step, phase="train")
            self.tb_logger.log_scalar("portfolio/mean_risk_eta", metrics["mean_risk_eta"], global_step, phase="train")
            
            # Log curriculum learning weight
            self.tb_logger.log_scalar("curriculum/lambda_epoch", self._current_lambda_epoch, global_step, phase="train")
            
            # Log histograms every N batches
            if batch_idx % self.tb_logger.histogram_freq == 0:
                # Log model weights and gradients
                self.tb_logger.log_model_weights(self.observer.mafia_model, global_step, phase="train")
                self.tb_logger.log_model_gradients(self.observer.mafia_model, global_step, phase="train")
                
                # Log action distribution (portfolio weights)
                if collected_topk_scores:
                    topk_scores_stacked = th.stack(collected_topk_scores, dim=1)  # (B, T_m, K)
                    self.tb_logger.log_histogram("actions/portfolio_weights", topk_scores_stacked, global_step, phase="train")

        # Clear memory
        del ochlv_batch, market_ochlv_batch
        th.cuda.empty_cache() if th.cuda.is_available() else None

        return metrics

    
    def _save_trajectory_log(self):
        """
        Save collected trajectory details to CSV file.
        Called after each training step if log_trajectory_details is enabled.
        """
        if not hasattr(self, '_trajectory_log') or not self._trajectory_log:
            return
        
        import pandas as pd
        import os
        
        # Determine output directory
        # Determine output directory
        # Use config.res_root if set (from script), else default to ./observer_offline (not results)
        output_dir = getattr(self.config, "res_root", "./observer_offline")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(self._trajectory_log)
        
        # Save to CSV (append mode)
        csv_path = os.path.join(output_dir, "trajectory_details.csv")
        mode = 'a' if os.path.exists(csv_path) else 'w'
        header = not os.path.exists(csv_path)
        
        df.to_csv(csv_path, mode=mode, header=header, index=False)
        
        # Clear log to free memory
        self._trajectory_log = []
        
        if header:  # First write
            smart_print(f"[TRAJ-LOG] Created trajectory details at: {csv_path}")

    def train_epoch(
        self,
        data_tensors: Dict[str, th.Tensor],
        steps_per_epoch: Optional[int] = None,
        writer: Optional[Any] = None,
        global_step_offset: int = 0,
        on_batch_done: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            data_tensors: Prepared data tensors from prepare_data_tensors()
            steps_per_epoch: Number of training steps (default: auto-computed)

        Returns:
            Dict of epoch-averaged metrics
        """
        T_total = data_tensors["T_total"]

        # Compute steps per epoch if not specified
        # Per Spec §6: steps = ceil((Len(Data) - T_m - h) / Batch_Size)
        # Note: We also subtract T_w for lookback window requirement
        if steps_per_epoch is None:
            # Available trajectory start positions
            available_starts = T_total - self.T_m - self.horizon - self.T_w
            # Use ceiling division to ensure full data coverage
            steps_per_epoch = max(1, int(np.ceil(available_starts / self.batch_size)))

        self._epoch += 1
        # Update curriculum penalty weight based on current epoch (Spec §7.1)
        self._current_lambda_epoch = self._compute_lambda_epoch(self._epoch)

        epoch_metrics = {
            "loss_total": 0.0,
            "loss_pg": 0.0,
            "loss_risk": 0.0,
            "loss_dir": 0.0,
            "rebalance_ratio": 0.0,
            "mean_risk_eta": 0.0,
        }
        
        # ============================================================
        # EPOCH HEADER
        # ============================================================
        smart_print("\n" + "=" * 100)
        smart_print(f"EPOCH {self._epoch} TRAINING")
        smart_print(f"  Batches per epoch: {steps_per_epoch}")
        smart_print(f"  Trajectories per batch: {self.batch_size} (processed in PARALLEL on {self.device})")
        smart_print(f"  Timesteps per trajectory: {self.T_m}")
        smart_print(f"  Total training steps this epoch: {steps_per_epoch * self.batch_size * self.T_m:,}")
        smart_print(f"  Curriculum λ_epoch: {self._current_lambda_epoch:.2f} (penalty weight)")
        smart_print("=" * 100 + "\n")

        for step in range(steps_per_epoch):
            # ============================================================
            # BATCH HEADER
            # ============================================================
            print(f"\n[BATCH {step + 1}/{steps_per_epoch}] 🚀 Processing {self.batch_size} trajectories in PARALLEL...", flush=True)

            # Sample batch
            batch = self.sample_trajectory_batch(data_tensors, mode="TRAIN")
            # -------------------------------------------------------------------
            # Collect and train step
            # -------------------------------------------------------------------
            step_metrics = self.collect_and_train_step(
                batch,
                data_tensors,
                batch_idx=step,
                epoch_idx=self._epoch
            )
            print(f"[BATCH {step + 1}/{steps_per_epoch}] ✅ Completed - {self.batch_size} trajectories trained in parallel", flush=True)
            
            # Save trajectory logs if enabled
            self._save_trajectory_log()
            
            # Real-time visualization callback (if provided)
            if on_batch_done is not None:
                try:
                    on_batch_done(step, steps_per_epoch)
                except Exception as viz_err:
                    print(f"      [WARN] Batch visualization failed: {viz_err}")
            
            # (Unified Logging removed in favor of adaptive realtime logging)

            # Accumulate
            for k, v in step_metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k] += v

            # TensorBoard logging (Step-level)
            if writer:
                global_step = global_step_offset + step
                writer.add_scalar("Train/Loss/Total", step_metrics["loss_total"], global_step)
                writer.add_scalar("Train/Loss/PG", step_metrics["loss_pg"], global_step)
                writer.add_scalar("Train/Loss/Risk", step_metrics["loss_risk"], global_step)
                writer.add_scalar("Train/Loss/Direction", step_metrics["loss_dir"], global_step)
                writer.add_scalar("Train/Risk_Eta/Mean", step_metrics["mean_risk_eta"], global_step)
                writer.add_scalar("Train/LR", step_metrics["lr"], global_step)

            # Progress logging (every 10% of epoch)
            if (step + 1) % max(1, steps_per_epoch // 10) == 0:
                progress = (step + 1) / steps_per_epoch * 100
                smart_print(
                    f"[OFFLINE-TRAIN] Epoch {self._epoch} | "
                    f"Batch {step + 1}/{steps_per_epoch} ({progress:.0f}%) | "
                    f"L_total: {step_metrics['loss_total']:.4f} | "
                    f"L_pg: {step_metrics['loss_pg']:.4f} | "
                    f"L_risk: {step_metrics['loss_risk']:.4f}"
                )

        # Average
        for k in epoch_metrics:
            epoch_metrics[k] /= steps_per_epoch

        epoch_metrics["epoch"] = self._epoch
        epoch_metrics["lr"] = self.observer.optimizer.param_groups[0]["lr"]

        # TensorBoard logging (Epoch-level)
        if writer:
            writer.add_scalar("Train/Epoch/Loss_Total", epoch_metrics["loss_total"], self._epoch)
            writer.add_scalar("Train/Epoch/Loss_PG", epoch_metrics["loss_pg"], self._epoch)
            writer.add_scalar("Train/Epoch/Loss_Risk", epoch_metrics["loss_risk"], self._epoch)
            writer.add_scalar("Train/Epoch/Loss_Dir", epoch_metrics["loss_dir"], self._epoch)
            writer.add_scalar("Train/Epoch/LR", epoch_metrics["lr"], self._epoch)

        # ============================================================
        # EPOCH SUMMARY
        # ============================================================
        smart_print("\n" + "=" * 100)
        smart_print(f"EPOCH {self._epoch} COMPLETE")
        smart_print("-" * 100)
        smart_print(f"  Total Loss (Combined):      {epoch_metrics['loss_total']:.6f}")
        smart_print(f"    ├─ Policy Gradient Loss:  {epoch_metrics['loss_pg']:.6f}")
        smart_print(f"    ├─ Risk Calibration Loss: {epoch_metrics['loss_risk']:.6f}")
        smart_print(f"    └─ Direction Loss:        {epoch_metrics['loss_dir']:.6f}")
        smart_print("-" * 100)
        smart_print(f"  Rebalance Ratio:            {epoch_metrics['rebalance_ratio']:.2%} (of all timesteps)")
        smart_print(f"  Mean Risk Tolerance (η):    {epoch_metrics['mean_risk_eta']:.4f}")
        smart_print(f"  Learning Rate:              {epoch_metrics['lr']:.6f}")
        smart_print("=" * 100 + "\n")

        return epoch_metrics

    @th.no_grad()
    def compute_selection_metrics(
        self,
        topk_indices: th.Tensor,  # (B, T_m, K)
        topk_scores: th.Tensor,  # (B, T_m, K) - kept for possible weighted portfolio
        price_returns: th.Tensor,  # (B, T_m, N)
    ) -> Dict[str, float]:
        """
        Compute Top-K portfolio selection performance metrics.
        
        Args:
            topk_indices: Selected Top-K stock indices for each timestep
            topk_scores: Softmax scores for Top-K stocks (for weighted portfolio, currently unused)
            price_returns: Returns for all N stocks
            
        Returns:
            Dict with:
                - sharpe_ratio: Sharpe Ratio of equal-weighted Top-K portfolio
                - mean_return: Daily mean return
                - volatility: Daily return volatility (std dev)
                - turnover: Average turnover rate (% of portfolio changed per timestep)
        """
        B, T_m, K = topk_indices.shape
        N = price_returns.shape[2]
        
        # Convert to numpy for easier computation
        topk_indices_np = topk_indices.cpu().numpy()  # (B, T_m, K)
        returns_np = price_returns[:, :T_m, :].cpu().numpy()  # (B, T_m, N)
        
        # Compute portfolio returns (equal-weighted Top-K)
        portfolio_returns = []
        turnover_rates = []
        
        for b in range(B):
            # For each timestep, compute portfolio return as mean of Top-K returns
            traj_returns = []
            traj_turnover = []
            
            prev_set = None
            for t in range(T_m):
                selected_idx = topk_indices_np[b, t]  # (K,)
                selected_returns = returns_np[b, t, selected_idx]  # (K,)
                
                # Equal-weighted portfolio return
                port_ret = selected_returns.mean()
                traj_returns.append(port_ret)
                
                # Turnover: % of stocks changed from previous timestep
                if prev_set is not None:
                    current_set = set(selected_idx)
                    unchanged = len(current_set.intersection(prev_set))
                    turnover = 1.0 - (unchanged / K)
                    traj_turnover.append(turnover)
                
                prev_set = set(selected_idx)
            
            portfolio_returns.extend(traj_returns)
            if traj_turnover:
                turnover_rates.extend(traj_turnover)
        
        # Convert to array
        portfolio_returns = np.array(portfolio_returns)
        
        # Compute metrics
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        # Sharpe Ratio (annualized, assuming daily returns)
        # Sharpe = (mean * 252) / (std * sqrt(252)) = mean * sqrt(252) / std
        if volatility > 1e-8:
            sharpe_ratio = mean_return * np.sqrt(252) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Average turnover
        avg_turnover = np.mean(turnover_rates) if turnover_rates else 0.0
        
        return {
            "sharpe_ratio": float(sharpe_ratio),
            "mean_return": float(mean_return),
            "volatility": float(volatility),
            "turnover": float(avg_turnover),
        }
    
    @th.no_grad()
    def compute_direction_metrics(
        self,
        direction_logits: th.Tensor,  # (B, T_m, 3)
        direction_labels: th.Tensor,  # (B, T_m)
    ) -> Dict[str, float]:
        """
        Compute direction classification metrics for 3-class Bull/Side/Bear.
        
        Args:
            direction_logits: Raw logits for 3 classes (0=Bear, 1=Side, 2=Bull)
            direction_labels: Ground truth labels
            
        Returns:
            Dict with:
                - accuracy: Overall accuracy
                - f1_bear: F1-score for Bear class
                - f1_side: F1-score for Side class
                - f1_bull: F1-score for Bull class
                - f1_macro: Macro-averaged F1 score
        """
        # Get predictions
        preds = th.argmax(direction_logits, dim=-1)  # (B, T_m)
        
        # Flatten
        preds_flat = preds.view(-1).cpu().numpy()
        labels_flat = direction_labels.view(-1).cpu().numpy()
        
        # Accuracy
        accuracy = (preds_flat == labels_flat).mean()
        
        # Per-class F1 scores
        f1_scores = {}
        class_names = ["bear", "side", "bull"]
        
        for class_id, class_name in enumerate(class_names):
            # Binary: This class vs rest
            pred_binary = (preds_flat == class_id).astype(int)
            label_binary = (labels_flat == class_id).astype(int)
            
            # TP, FP, FN
            tp = ((pred_binary == 1) & (label_binary == 1)).sum()
            fp = ((pred_binary == 1) & (label_binary == 0)).sum()
            fn = ((pred_binary == 0) & (label_binary == 1)).sum()
            
            # Precision and Recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            f1_scores[f"f1_{class_name}"] = float(f1)
        
        # Macro F1
        f1_macro = np.mean([f1_scores[f"f1_{name}"] for name in class_names])
        
        return {
            "accuracy": float(accuracy),
            **f1_scores,
            "f1_macro": float(f1_macro),
        }
    
    @th.no_grad()
    def compute_risk_metrics(
        self,
        risk_pred: th.Tensor,  # (B, T_m)
        risk_target: th.Tensor,  # (B, T_m)
    ) -> Dict[str, float]:
        """
        Compute risk prediction quality metrics.
        
        Args:
            risk_pred: Predicted risk tolerance eta
            risk_target: Target risk tolerance eta
            
        Returns:
            Dict with:
                - mse: Mean Squared Error
                - mae: Mean Absolute Error
                - correlation: Pearson correlation coefficient
        """
        # Flatten
        pred_flat = risk_pred.view(-1).cpu().numpy()
        target_flat = risk_target.view(-1).cpu().numpy()
        
        # MSE
        mse = np.mean((pred_flat - target_flat) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_flat - target_flat))
        
        # Pearson Correlation
        if len(pred_flat) > 1:
            correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "correlation": float(correlation),
        }

    @th.no_grad()
    def validate_epoch(
        self,
        data_tensors: Dict[str, th.Tensor],
        steps: Optional[int] = None,
    ):
        """
        Validate model on held-out data and return comprehensive metrics.

        Args:
            data_tensors: Prepared validation data tensors
            steps: Number of validation steps (default: 10)

        Returns:
            ObserverValidationResult with all validation metrics
        """
        from RL_controller.observer_validation_metrics import ObserverValidationResult
        
        self.observer.mafia_model.eval()

        if steps is None:
            T_total = data_tensors["T_total"]
            available_starts = T_total - self.T_m - self.horizon - self.T_w
            steps = max(1, min(10, available_starts // self.batch_size))

        # Accumulators for losses
        total_loss_total = 0.0
        total_loss_pg = 0.0
        total_loss_risk = 0.0
        total_loss_dir = 0.0
        
        # Accumulators for metric computation
        all_topk_indices = []
        all_topk_scores = []
        all_price_returns = []
        all_risk_pred = []
        all_risk_target = []
        all_direction_logits = []
        all_direction_labels = []

        for step in range(steps):
            batch = self.sample_trajectory_batch(data_tensors, mode="EVAL")

            with th.no_grad():
                B = batch.stock_ochlv.size(0)
                T_m = batch.stock_ochlv.size(1)

                # Reset hidden state
                if hasattr(self.observer.mafia_model, "reset_temporal_state"):
                    self.observer.mafia_model.reset_temporal_state()

                # Initialize context buffer (FIFO) to mirror training
                W_route = int(getattr(self.config, "router_context_window", 5))
                if hasattr(self.observer.mafia_model, 'D'):
                    mafia_D = self.observer.mafia_model.D
                else:
                    mafia_D = int(getattr(self.config, "mafia_D", 128))
                batch_context_buffer = th.zeros(B, W_route, mafia_D, device=self.device)

                full_ochlv = data_tensors["ochlv"]
                full_market_ochlv = data_tensors.get("market_ochlv")

                # Direction reversal tracking
                N_confirm = int(getattr(self.config, "regime_confirmation_window", 3))
                direction_history = th.ones(B, N_confirm + 1, dtype=th.long, device=self.device)

                days_since_last_rebal = th.zeros(B, dtype=th.long, device=self.device)
                rebal_interval = self.rebalance_interval

                # Static trigger count for normalization
                static_trigger_mask = (
                    (batch.rebalance_mask > 0.5) |
                    (batch.vol_shock_mask > 0.5) |
                    (batch.dc_mask > 0.5)
                )
                expected_rebal_count = static_trigger_mask.float().sum().item()
                expected_rebal_count = max(expected_rebal_count, 1.0)

                # Collections
                collected_topk_scores = []
                collected_topk_indices = []
                collected_risk_eta = []
                collected_direction_logits = []

                pg_losses = []
                risk_losses = []
                dir_losses = []

                prev_indices_tensor = None

                for t in range(T_m):
                    # Build windows
                    batch_windows = []
                    batch_windows_mkt = []
                    for b in range(B):
                        t_s = batch.start_indices[b].item()
                        window_start = max(0, t_s + t - self.T_w + 1)
                        window_end = t_s + t + 1
                        window = full_ochlv[window_start:window_end]
                        if window.size(0) < self.T_w:
                            pad_size = self.T_w - window.size(0)
                            pad = window[0:1].expand(pad_size, -1, -1)
                            window = th.cat([pad, window], dim=0)
                        window = window.permute(1, 2, 0)
                        batch_windows.append(window)

                        if full_market_ochlv is not None:
                            window_mkt = full_market_ochlv[window_start:window_end]
                            if window_mkt.size(0) < self.T_w:
                                pad_size = self.T_w - window_mkt.size(0)
                                pad = window_mkt[0:1].expand(pad_size, -1, -1)
                                window_mkt = th.cat([pad, window_mkt], dim=0)
                            window_mkt = window_mkt.permute(1, 2, 0)
                            batch_windows_mkt.append(window_mkt)

                    ochlv_batch = th.stack(batch_windows, dim=0).to(self.device)
                    market_ochlv_batch = None
                    if batch_windows_mkt:
                        market_ochlv_batch = th.stack(batch_windows_mkt, dim=0).to(self.device)

                    # Rebalance schedule trigger
                    if t == 0:
                        sched_trigger = th.ones(B, dtype=th.bool, device=self.device)
                        days_since_last_rebal[:] = 0
                    else:
                        days_since_last_rebal += 1
                        sched_trigger = (days_since_last_rebal >= rebal_interval)

                    vol_trigger = batch.vol_shock_mask[:, t] > 0.5
                    dc_trigger = batch.dc_mask[:, t] > 0.5
                    pre_trigger = vol_trigger | sched_trigger | dc_trigger

                    vol_flag = batch.vol_shock_mask[:, t:t+1]
                    dc_flag = batch.dc_mask[:, t:t+1]
                    explicit_signals = th.cat([vol_flag, dc_flag], dim=-1)

                    outputs = self.observer.mafia_model(
                        ochlv_data=ochlv_batch,
                        market_index_ochlv_data=market_ochlv_batch,
                        force_topk_indices=None,
                        router_context_buffer=batch_context_buffer,
                        explicit_signals=explicit_signals,
                    )

                    (
                        market_vector,
                        risk_eta,
                        market_scores_full,
                        direction_logits,
                        market_context,
                        topk_indices,
                        topk_embeddings,
                        topk_scores,
                    ) = outputs

                    # Temperature scaling for direction head (sharpen/flatten)
                    if self.direction_temperature != 1.0:
                        direction_logits = direction_logits / self.direction_temperature

                    fresh_topk_indices = topk_indices.clone()
                    fresh_topk_scores = topk_scores.clone()

                    # Detach temporal state and buffer
                    if hasattr(self.observer.mafia_model, "detach_temporal_state"):
                        self.observer.mafia_model.detach_temporal_state()
                    batch_context_buffer = batch_context_buffer.detach()

                    # Direction reversal trigger
                    pred_state = th.argmax(direction_logits, dim=-1)
                    pred_state = th.clamp(pred_state, 0, 2)
                    direction_history = th.cat([direction_history[:, 1:], pred_state.unsqueeze(1)], dim=1)

                    if t < N_confirm:
                        dir_trigger = th.zeros(B, dtype=th.bool, device=self.device)
                    else:
                        last_seq = direction_history[:, 1:]
                        prev_regime = direction_history[:, 0]
                        BULL, BEAR = 2, 0
                        all_bull = (last_seq == BULL).all(dim=1)
                        was_bear = (prev_regime == BEAR)
                        bear_to_bull = all_bull & was_bear
                        all_bear = (last_seq == BEAR).all(dim=1)
                        was_bull = (prev_regime == BULL)
                        bull_to_bear = all_bear & was_bull
                        dir_trigger = bear_to_bull | bull_to_bear

                    effective_mask = pre_trigger | dir_trigger
                    effective_mask_float = effective_mask.float()
                    days_since_last_rebal[effective_mask] = 0

                    # Hold logic
                    if t > 0 and collected_topk_indices:
                        prev_idx = collected_topk_indices[-1]
                        prev_sc = collected_topk_scores[-1]
                        mask_k = effective_mask.unsqueeze(1).expand(-1, self.K)
                        final_indices = th.where(mask_k, topk_indices, prev_idx)
                        final_scores = th.where(mask_k, topk_scores, prev_sc)
                    else:
                        final_indices = fresh_topk_indices
                        final_scores = fresh_topk_scores

                    # Reward calc
                    start_idx = t + 1
                    end_idx = t + 1 + self.horizon
                    future_returns_slice = batch.price_returns[:, start_idx:end_idx, :]
                    market_returns_slice = batch.market_returns[:, start_idx:end_idx]
                    h_len = future_returns_slice.size(1)
                    eps = 1e-8

                    if h_len == 0:
                        R_raw = th.zeros(B, device=self.device)
                        baseline = th.zeros(B, device=self.device)
                    else:
                        gather_idx = final_indices.unsqueeze(1).expand(-1, h_len, -1)
                        selected_returns = th.gather(future_returns_slice, 2, gather_idx)
                        compounded_stock_returns = th.prod(1 + selected_returns, dim=1) - 1
                        R_raw = self.scale_factor_reward * compounded_stock_returns.mean(dim=1)
                        baseline = self.scale_factor_reward * (th.prod(1 + market_returns_slice, dim=1) - 1)

                    turnover = th.zeros(B, device=self.device)
                    symdiff = th.zeros(B, device=self.device)
                    if prev_indices_tensor is not None:
                        for b_i in range(B):
                            curr_set = set(final_indices[b_i].cpu().tolist())
                            prev_set = set(prev_indices_tensor[b_i].cpu().tolist())
                            held_count = len(curr_set.intersection(prev_set))
                            t_val = (self.K - held_count) / self.K
                            turnover[b_i] = t_val
                            s_val = (2 * (self.K - held_count)) / self.K
                            symdiff[b_i] = s_val

                    lambda_epoch = self._current_lambda_epoch
                    penalty = lambda_epoch * (self.alpha_turnover * turnover + self.alpha_change * symdiff)
                    R_net = R_raw - penalty
                    A_t_raw = R_net - baseline
                    A_mean = A_t_raw.mean()
                    A_std = A_t_raw.std() + eps if B > 1 else 1.0
                    A_t = (A_t_raw - A_mean) / A_std

                    log_probs = th.log(th.gather(market_scores_full, 1, final_indices) + eps)
                    pg_term = -(log_probs * A_t.detach().unsqueeze(1)).mean(dim=1)
                    entropy = -(market_scores_full * th.log(market_scores_full + eps)).sum(dim=1)
                    step_loss_val = pg_term - self.beta_entropy * entropy

                    risk_loss = self.risk_criterion(risk_eta, batch.risk_targets[:, t]) * self.risk_scaling_factor
                    dir_loss = self.dir_criterion(direction_logits, batch.direction_labels[:, t])

                    step_pg_loss = (step_loss_val * effective_mask_float).sum()
                    pg_losses.append(step_pg_loss.item())
                    risk_losses.append(risk_loss.item())
                    dir_losses.append(dir_loss.item())

                    # Collect outputs
                    collected_topk_scores.append(final_scores)
                    collected_topk_indices.append(final_indices)
                    collected_risk_eta.append(risk_eta)
                    collected_direction_logits.append(direction_logits)

                    # Update context buffer FIFO
                    history_part = batch_context_buffer[:, 1:, :]
                    new_part = market_context.detach().unsqueeze(1)
                    batch_context_buffer = th.cat([history_part, new_part], dim=1)

                    prev_indices_tensor = final_indices

            # Stack collected outputs
            topk_scores_stack = th.stack(collected_topk_scores, dim=1)
            topk_indices_stack = th.stack(collected_topk_indices, dim=1)
            risk_eta_stack = th.stack(collected_risk_eta, dim=1)
            direction_logits_stack = th.stack(collected_direction_logits, dim=1)

            # Loss averages (normalized like training spec)
            norm_pg = 1.0 / (expected_rebal_count + 1e-8)
            pg_values = [v for v in pg_losses if v != 0.0]
            avg_pg = (sum(pg_values) / len(pg_values) if pg_values else 0.0) * norm_pg
            avg_risk = sum(risk_losses) / len(risk_losses) if risk_losses else 0.0
            avg_dir = sum(dir_losses) / len(dir_losses) if dir_losses else 0.0
            total_disp = (
                self.lambda_pg * avg_pg +
                self.lambda_risk * avg_risk +
                self.lambda_dir * avg_dir
            )

            total_loss_total += total_disp
            total_loss_pg += avg_pg
            total_loss_risk += avg_risk
            total_loss_dir += avg_dir

            # Store for metric computation
            all_topk_indices.append(topk_indices_stack)
            all_topk_scores.append(topk_scores_stack)
            all_price_returns.append(batch.price_returns[:, :T_m, :])
            all_risk_pred.append(risk_eta_stack)
            all_risk_target.append(batch.risk_targets)
            all_direction_logits.append(direction_logits_stack)
            all_direction_labels.append(batch.direction_labels)

        # Average losses
        avg_loss_total = total_loss_total / steps
        avg_loss_pg = total_loss_pg / steps
        avg_loss_risk = total_loss_risk / steps
        avg_loss_dir = total_loss_dir / steps

        # Concatenate all batches for metric computation
        all_topk_indices_cat = th.cat(all_topk_indices, dim=0)
        all_topk_scores_cat = th.cat(all_topk_scores, dim=0)
        all_price_returns_cat = th.cat(all_price_returns, dim=0)
        all_risk_pred_cat = th.cat(all_risk_pred, dim=0)
        all_risk_target_cat = th.cat(all_risk_target, dim=0)
        all_direction_logits_cat = th.cat(all_direction_logits, dim=0)
        all_direction_labels_cat = th.cat(all_direction_labels, dim=0)

        # Compute metrics
        selection_metrics = self.compute_selection_metrics(
            all_topk_indices_cat,
            all_topk_scores_cat,
            all_price_returns_cat,
        )
        
        direction_metrics = self.compute_direction_metrics(
            all_direction_logits_cat,
            all_direction_labels_cat,
        )
        
        risk_metrics = self.compute_risk_metrics(
            all_risk_pred_cat,
            all_risk_target_cat,
        )

        result = ObserverValidationResult(
            epoch=self._epoch,
            loss_total=avg_loss_total,
            loss_pg=avg_loss_pg,
            loss_risk=avg_loss_risk,
            loss_dir=avg_loss_dir,
            topk_sharpe_ratio=selection_metrics["sharpe_ratio"],
            topk_mean_return=selection_metrics["mean_return"],
            topk_volatility=selection_metrics["volatility"],
            topk_turnover=selection_metrics["turnover"],
            direction_accuracy=direction_metrics["accuracy"],
            direction_f1_bear=direction_metrics["f1_bear"],
            direction_f1_side=direction_metrics["f1_side"],
            direction_f1_bull=direction_metrics["f1_bull"],
            direction_f1_macro=direction_metrics["f1_macro"],
            risk_mse=risk_metrics["mse"],
            risk_mae=risk_metrics["mae"],
            risk_correlation=risk_metrics["correlation"],
        )

        self.observer.mafia_model.train()

        return result



def create_offline_trainer(config, observer) -> ObserverOfflineBatchTrainer:
    """
    Factory function to create offline trainer.

    Args:
        config: Configuration object
        observer: MAFIAObserver instance

    Returns:
        ObserverOfflineBatchTrainer instance
    """
    return ObserverOfflineBatchTrainer(
        config=config,
        observer=observer,
    )
