#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observer Offline Batch Trainer - Implements Spec §7 Inner Training Loop

This module implements the offline batch training mode for MAFIA Observer
following the "Collect → Train → Discard" pattern described in the spec.

Key Features per Spec:
1. Random trajectory sampling: Choose random t_s from available data
2. Fresh hidden state: Reset LSTM hidden state at start of each batch
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
from dataclasses import dataclass, asdict

from RL_controller.observer_validation_metrics import ObserverValidationResult

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

    # === Volatility for Rebalancing Logic (Spec §8.2.3) ===
    # NOT part of Direction Head explicit signals, but used for vol shock detection
    vol_std20: Optional[th.Tensor] = None  # (B, T_m) - 20-day rolling volatility

    # === Explicit Signals for Direction Head (Spec §3.5.1 v2.1 - Wide Path) ===
    # 4-dim vector: [DC_Event_Flag, Breadth_Gap, Div_Signal, Signed_VPI_Zscore]

    # 1. DC_Event_Flag: (B, T_m) - Structural break signal from Market-DC Agent
    # Binary flag (1.0 if DC event, 0.0 otherwise) or magnitude
    dc_event_flag: Optional[th.Tensor] = None

    # 2. Breadth_Gap: (B, T_m) - Market Breadth Gap
    # = avg(RSI_14 of all stocks) - RSI_14(Index)
    # Detects "Xanh vỏ đỏ lòng" (green shell red core): Index up but Gap < 0 = Bull Trap warning
    breadth_gap: Optional[th.Tensor] = None

    # 3. Div_Signal: (B, T_m) - Directional Divergence Signal
    # = Sign(RSI_Slope) if trái dấu với Sign(Price_Slope), else 0
    # Detects reversal divergence: Follow the Momentum rule
    div_signal: Optional[th.Tensor] = None

    # 4. Signed_VPI_Zscore: (B, T_m) - Signed Volume-Price Index (Z-scored)
    # VPI = Sign(ΔP) × |ΔP| / (Vol / Vol_20_avg), then z-score normalized
    # Measures money flow efficiency: Price up + moderate Vol = Strong Bull
    # High Vol + flat Price = Distribution warning
    signed_vpi_zscore: Optional[th.Tensor] = None

    # 5. Drawdown60: (B, T_m) - 60-day Max Drawdown (Sign inverted or magnitude)
    # Measures how far current price is from 60-day high.
    # DD = (Close - MaxHigh60) / MaxHigh60 (Negative value)
    drawdown60: Optional[th.Tensor] = None

    # 5. Drawdown60: (B, T_m) - Rolling Max Drawdown (Pain Index)
    # Drawdown = (Price - Rolling_Peak_60) / Rolling_Peak_60
    # Measures market stress and "Pain".
    # - Low Drawdown (~0): Bull market / High confidence.
    # - High Drawdown (<-10%): Correction / Bear market.
    drawdown60: Optional[th.Tensor] = None


from RL_controller.mafia_modules import MAFIAModel


# =============================================================================
# Focal Loss (Spec 5.1.3)
# =============================================================================
class FocalLoss(th.nn.Module):
    """
    Multi-class Focal Loss for handling class imbalance.
    L = -alpha * (1-p)^gamma * log(p)
    """

    def __init__(
        self, gamma=2.0, alpha=None, device=None, reduction="mean", label_smoothing=0.0
    ):
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

        # Stable Log-Space Calculation
        # log_softmax is more stable than explicit softmax + log
        log_pt = th.nn.functional.log_softmax(input, dim=-1)
        
        # Gather log_prob for the target class
        # Target shape adjust if needed
        if target.dim() == input.dim() - 1:
             target = target.unsqueeze(-1)
             
        log_pt_target = log_pt.gather(-1, target)
        log_pt_target = log_pt_target.view(-1) # Flatten
        pt = log_pt_target.exp()
        
        # Stability fix: Clamp pt for focal term
        # Avoid exactly 0 or 1 for power calc, though (1-pt) is generally safe for pt in [0,1]
        pt_clamped = pt.clamp(min=1e-8, max=1.0 - 1e-8)
        
        # Focal Term: (1 - pt)^gamma
        focal_term = (1.0 - pt_clamped).pow(self.gamma)

        # Loss = -alpha * focal_term * log_pt
        loss = -1 * focal_term * log_pt_target

        if self.alpha is not None:
             target_flat = target.view(-1)
             alpha_t = self.alpha[target_flat]
             loss = loss * alpha_t
             
        # Label Smoothing Adjustment (Approximate or skip if strict focal is needed)
        # Standard Focal Loss doesn't always mix identically with LS.
        # But user tuning explicitly requested LS = 0.08.
        # If LS is ON, we should perhaps fall back to the CE implementation 
        # OR add the smoothing term: loss = (1-eps)*Focal + eps*UniformFocal
        # For Robustness now, let's keep it simple. If LS>0, the simple CE-based impl was actually cleaner...
        # Wait, the PREVIOUS implementation handled LS correctly via F.cross_entropy.
        # The crash was `(1-pt)**gamma`.
        # Let's revert to wrapping F.cross_entropy BUT perform the math safely.
        
        # REVISED PLAN: Wrapper around CE is cleaner for Label Smoothing support.
        # We just need to safeguard the `pt` tensor before power.
        
        # 1. Compute CE (with LS support) -> This gives -log(pt_smoothed)
        ce_loss = th.nn.functional.cross_entropy(
            input, target.view_as(target_flat) if target.dim()!=input.dim()-1 else target, 
            reduction="none", 
            label_smoothing=self.label_smoothing
        )
        
        # 2. Extract pt safely
        # pt = exp(-ce)
        pt = th.exp(-ce_loss)
        
        # 3. CRITICAL STABILITY: Stop gradients on `pt` for the focal term?
        # Usually we want gradients through focal term.
        # The issue is `pow` derivative at base=0 (if gamma < 1) or huge base.
        # Clamp pt to [0.0001, 0.9999] just for the focal term calculation base.
        pt_safe = pt.clamp(min=1e-6, max=1.0-1e-6)
        
        # 4. Focal term
        focal_term = (1.0 - pt_safe).pow(self.gamma)
        
        # 5. Combine
        focal_loss = focal_term * ce_loss
        
        # 6. Alpha
        if self.alpha is not None:
            if target.dim() > 1: # If input was [N, C] target [N], it's handled. If target [N, ...], flatten
                 alpha_t = self.alpha[target.view(-1)]
            else:
                 alpha_t = self.alpha[target]
            
            # Ensure shape match
            if alpha_t.shape != focal_loss.shape:
                 alpha_t = alpha_t.view_as(focal_loss)
                 
            focal_loss = focal_loss * alpha_t

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
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
        elif th.cuda.is_available():
            self.device = th.device("cuda")
        elif th.backends.mps.is_available():
            self.device = th.device("mps")
        else:
            self.device = th.device("cpu")
        self.optimizer = observer.optimizer  # Use observer's optimizer
        self.tb_logger = tensorboard_logger  # Store TensorBoard logger

        # Training parameters from config
        self.T_m = int(
            getattr(config, "mafia_trajectory_length", 128)
        )  # Trajectory length
        self.batch_size = int(getattr(config, "mafia_batch_size", 32))  # Batch size B
        self.T_w = int(getattr(config, "mafia_T_w", 30))  # Window size for features

        # Verify sampling strategy
        sampling_strategy = getattr(
            config, "mafia_sampling_strategy", "random_trajectory"
        )
        if sampling_strategy != "random_trajectory":
            print(
                f"[WARN] ObserverOfflineBatchTrainer enforces 'random_trajectory' but config is '{sampling_strategy}'"
            )

        self.K = int(getattr(config, "mafia_top_k", 10))  # Top-K stocks
        self.horizon = int(
            getattr(config, "mafia_pg_reward_horizon", 5)
        )  # h: Look-ahead horizon for        # Spec 3.4: Rebalance Interval T_r
        # Fixed interval for rebalancing events (e.g. 14 days)
        self.rebalance_interval = int(
            getattr(
                config,
                "mafia_topk_rebalance_interval",
                getattr(config, "rebalance_interval", 14),
            )
        )

        # Holding Reward Config
        self.r_hold_alpha = float(getattr(config, "mafia_reward_alpha_hold", 0.0))  # Trend bonus

        # Loss weights (Spec §5 Hyperparameters)
        self.lambda_pg = float(
            getattr(config, "mafia_lambda_pg", 1.0)
        )  # λ_pg = 1.0 (default)
        self.lambda_risk = float(
            getattr(config, "mafia_lambda_risk", 0.3)
        )  # λ_risk = 0.3 per Spec §5
        self.lambda_dir = float(
            getattr(config, "mafia_lambda_dir", 0.5)
        )  # λ_dir = 0.5 per Spec §5
        self.lambda_balance = float(
            getattr(config, "mafia_lambda_balance", 0.1)
        ) # Load Balancing Loss Weight

        # Penalty coefficients for PG
        self.alpha_turnover = float(getattr(config, "mafia_pg_alpha_turnover", 0.50))
        self.alpha_change = float(getattr(config, "mafia_pg_alpha_change", 0.50))

        # Risk loss scaling factor (Spec §5.1.2: S_risk)
        # L_risk (MSE) ≈ 0.01, scale up to match L_PG (~1.0) and L_dir (~1.0)
        self.risk_scaling_factor = float(getattr(config, "scale_factor_risk", 10.0))

        # Reward scaling factor (Spec §5.1.1: S_reward)
        # Raw compounding returns ~0.01, scale to ~1.0 for stable gradients
        self.scale_factor_reward = float(getattr(config, "scale_factor_reward", 100.0))

        # Curriculum Learning parameters (Spec §7.1)
        self.curriculum_warmup_epochs = int(
            getattr(config, "curriculum_warmup_epochs", 0)
        )
        self.curriculum_penalty_rampup = int(
            getattr(config, "curriculum_penalty_rampup", 5)
        )
        self._current_lambda_epoch = (
            1.0  # Default: full penalty (will be updated per epoch)
        )

        # Entropy bonus coefficient for PG (Spec §5.1.1: β_ent = 0.01)
        self.beta_entropy = float(getattr(config, "mafia_beta_entropy", 0.01))

        # Focal Loss parameters for Direction (Spec §5.1.3)
        # Optimized α based on VNINDEX distribution: Bear=19.1%, Side=46.0%, Bull=34.9%
        self.focal_gamma = float(
            getattr(config, "mafia_focal_gamma", 2.0)
        )  # Focusing parameter
        self.focal_alpha = list(
            getattr(config, "mafia_focal_alpha", [1.5, 0.4, 1.0])
        )  # Class weights [bear, side, bull]
        # Temperature scaling for direction logits (T<1 sharpens distribution)
        self.direction_temperature = float(
            getattr(config, "mafia_direction_temperature", 1.0)
        )

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
            label_smoothing=self.direction_label_smoothing,  # Spec 5.1.3
        )
        smart_print(
            f"[LOSS] Direction Loss: FocalLoss(γ={self.focal_gamma}, α={self.focal_alpha}, ε_smooth={self.direction_label_smoothing})"
        )

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
        self.scaler = th.amp.GradScaler(enabled=self.use_mixed_precision)

        # Hook for Consistency Check (Non-intrusive weight inspection)
        self._latest_gate_weights = None
        self._register_gate_hook()
        self.scaler = None
        if self.use_mixed_precision:
            # Only create scaler if CUDA is available (AMP requires CUDA)
            if self.device.type == "cuda":
                self.scaler = th.cuda.amp.GradScaler()
                smart_print(
                    "[OPTIMIZER] Mixed Precision (FP16) enabled with GradScaler"
                )
            else:
                self.use_mixed_precision = False  # Disable on non-CUDA (MPS/CPU)
                device_name = "MPS" if self.device.type == "mps" else "CPU"
                smart_print(
                    f"[OPTIMIZER] Mixed Precision disabled ({device_name} mode - AMP requires CUDA)"
                )

        # Memory Optimization: Gradient Accumulation
        self.gradient_accumulation_steps = getattr(
            config, "gradient_accumulation_steps", 1
        )
        if self.gradient_accumulation_steps > 1:
            smart_print(
                f"[OPTIMIZER] Gradient Accumulation enabled: {self.gradient_accumulation_steps} steps"
            )
            smart_print(
                f"            Effective batch size: {self.batch_size * self.gradient_accumulation_steps}"
            )

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

        cleaned = df.sort_values(
            ["stock", "date"], ascending=True, ignore_index=True
        ).copy()

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
            sample_rows = cleaned.loc[extreme_mask, ["date", "stock", "_ret_tmp"]].head(
                3
            )
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
        # Updated logic: Ramp up from 40% (approx old_penalty/new_penalty)
        start_fraction = 0.4  # Start at ~1.2 (0.4 * 3.0)

        if current_epoch < self.curriculum_warmup_epochs:
            # Return base penalty (start_fraction) instead of 0.0
            # This ensures pre-ramp epochs (like Epoch 18) are calculated with 1.25, not 0.
            return start_fraction
        
        ramp_progress = current_epoch - self.curriculum_warmup_epochs
        if ramp_progress >= self.curriculum_penalty_rampup:
            return 1.0
            
        fraction_progress = ramp_progress / self.curriculum_penalty_rampup
        # Linearly interpolate from start_fraction to 1.0
        return start_fraction + (1.0 - start_fraction) * fraction_progress

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
        filtered = data.loc[
            mask, ["date", "stock", "open", "close", "high", "low", "volume"]
        ].copy()
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

        smart_print(
            f"[OFFLINE] Processing {N} stocks over {T_total} days (Chunked Mode)..."
        )

        # Date mapping for fast lookup: Timestamp -> Index
        date_map = {pd.Timestamp(d): i for i, d in enumerate(dates)}

        # Preallocate arrays: (T_total, N, 5) for OCHLV
        # Initialize with NaN
        ochlv_array = np.full((T_total, N, 5), np.nan, dtype=np.float32)

        # 2. Iterate stock-by-stock to fill array (Low Peak Memory)
        # Group by stock first to avoid repeated filtering
        grouped = filtered.groupby("stock")
    
    # [NAN-FIX] Ensure no Infs exist before filling
    # Since we fill stock-by-stock, we handle it inside loop or pre-check filtered?
    # Filtered is DataFrame.
    # ochlv_array is initialized with NaN.
    # We copy values from DF to Array.
    # So we should valid_mask &= ~np.isinf(...) inside the loop?
    # Or just replace Inf with NaN in the array slice.

        for n, stock_ticker in enumerate(stock_list):
            if n % 50 == 0:
                smart_print(f"[OFFLINE]   Processing stock {n}/{N}: {stock_ticker}...")
                gc.collect()  # Garbage collect regularly

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

            # [NAN-FIX] Replace Inf with NaN ensuring fill logic works for them too
            slice_view = ochlv_array[:, n, :]
            if np.isinf(slice_view).any():
                 slice_view[np.isinf(slice_view)] = np.nan

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

        # [NAN-FIX] Final pass for stocks with NO data (skipped in loop)
        # Fill Prices (0-3) with 1.0
        for f in range(4):
             col_view = ochlv_array[:, :, f]
             if np.isnan(col_view).any():
                 col_view[np.isnan(col_view)] = 1.0
                 
        # Fill Volume (4) with 0.0
        col_vol = ochlv_array[:, :, 4]
        if np.isnan(col_vol).any():
            col_vol[np.isnan(col_vol)] = 0.0

        smart_print(f"[OFFLINE] Fixed missing stocks. NaNs remaining: {np.isnan(ochlv_array).sum()}")

        smart_print("[OFFLINE] Computing returns...")
        # Compute returns: (T_total, N)
        close_prices = ochlv_array[:, :, 1]
        returns = np.zeros_like(close_prices)
        returns[1:] = (close_prices[1:] - close_prices[:-1]) / np.clip(
            close_prices[:-1], 1e-8, None
        )
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
                f"\n{'=' * 80}\n"
                f"🚨 WARNING: Extreme returns detected!\n"
                f"{'=' * 80}\n"
                f"  Count:              {num_extreme:,} / {returns.size:,} ({num_extreme / returns.size * 100:.3f}%)\n"
                f"  Max absolute:       {max_return:.2f} ({max_return * 100:.0f}%)\n"
                f"  99th percentile:   {percentile_99:.4f}\n"
                f"  This indicates data quality issues:\n"
                f"    - Stock splits not adjusted\n"
                f"    - Bad price data\n"
                f"    - Corporate actions\n"
                f"  → Clipping to [{clip_min}, {clip_max}] to prevent overflow\n"
                f"{'=' * 80}\n"
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

                mkt_filtered = mkt_copy[
                    (mkt_copy["date"] >= start_date) & (mkt_copy["date"] <= end_date)
                ].copy()
                del mkt_copy

                # Build date map for market data
                mkt_filtered["date_idx"] = mkt_filtered["date"].map(date_map)
                mkt_filtered = mkt_filtered[mkt_filtered["date_idx"].notna()].copy()
                mkt_filtered["date_idx"] = mkt_filtered["date_idx"].astype(int)

                # Vectorized fill
                mkt_date_indices = mkt_filtered["date_idx"].values
                market_ochlv_array[mkt_date_indices, 0, 0] = mkt_filtered[
                    "open"
                ].values.astype(np.float32)
                market_ochlv_array[mkt_date_indices, 0, 1] = mkt_filtered[
                    "close"
                ].values.astype(np.float32)
                market_ochlv_array[mkt_date_indices, 0, 2] = mkt_filtered[
                    "high"
                ].values.astype(np.float32)
                market_ochlv_array[mkt_date_indices, 0, 3] = mkt_filtered[
                    "low"
                ].values.astype(np.float32)
                market_ochlv_array[mkt_date_indices, 0, 4] = mkt_filtered[
                    "volume"
                ].values.astype(np.float32)

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
                market_returns[1:, 0] = (mkt_close[1:] - mkt_close[:-1]) / np.clip(
                    mkt_close[:-1], 1e-8, None
                )

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

    def _register_gate_hook(self):
        """
        Register forward hook on Gating Router to inspect expert weights
        without modifying model architecture.
        """
        if hasattr(self.observer.mafia_model, "signal_generator"):
            router = self.observer.mafia_model.signal_generator.gating_router
            router.register_forward_hook(self._capture_gate_weights_hook)
            
    def _capture_gate_weights_hook(self, module, input, output):
        """
        Hook callback to capture gate weights from Router output.
        Router returns: (gate_weights, raw_context)
        """
        # Capture WITH GRAD for loss calculation
        self._gate_weights_with_grad = output[0]
        # Detach for logging/monitoring (keeps legacy behavior safe)
        self._latest_gate_weights = output[0].detach()

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
            r_fut = (
                (market_close[future_idx] / market_close[t_s]) - 1.0
                if market_close[t_s] > 0
                else 0
            )

            # Intra-period drawdown
            future_lows = market_low[t_s + 1 : future_idx + 1]
            if len(future_lows) > 0:
                min_price = np.min(future_lows)
                intra_dd = (
                    (min_price / market_close[t_s]) - 1.0
                    if market_close[t_s] > 0
                    else 0
                )
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
        else:  # EVAL or VALIDATION
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
        use_class_balanced = getattr(
            self.config, "mafia_use_class_balanced_sampling", False
        )

        if use_class_balanced and mode == "TRAIN":
            # Class-balanced sampling: sample equally from Bear, Side, Bull classes
            dir_lookahead = int(getattr(self.config, "direction_label_lookahead", 14))
            class_indices = self._build_class_indices(
                data_tensors, min_start, max_start, dir_lookahead
            )

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
                        sampled = np.random.randint(
                            min_start, max_start + 1, size=n_samples
                        )
                    elif len(indices) < n_samples:
                        # Sample with replacement if not enough samples
                        sampled = np.random.choice(
                            indices, size=n_samples, replace=True
                        )
                    else:
                        # Sample without replacement
                        sampled = np.random.choice(
                            indices, size=n_samples, replace=False
                        )

                    start_indices.extend(sampled)

                start_indices = np.array(start_indices)
                np.random.shuffle(start_indices)  # Shuffle to avoid class order bias
            else:
                # Fallback to uniform random if class indices couldn't be built
                start_indices = np.random.randint(
                    min_start, max_start + 1, size=self.batch_size
                )
        else:
            # Uniform random sampling (default)
            start_indices = np.random.randint(
                min_start, max_start + 1, size=self.batch_size
            )

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
        # === Vol_Std20 for Rebalancing Logic (NOT for Direction Head) ===
        batch_vol_std20 = []  # (B, T_m) - 20-day rolling volatility

        # === Wide Path Explicit Signals (Spec §3.5.1 v2.1) ===
        batch_dc_event_flag = []      # (B, T_m) - DC structural break flag
        batch_breadth_gap = []        # (B, T_m) - avg(RSI_stocks) - RSI_index
        batch_div_signal = []         # (B, T_m) - RSI vs Price divergence
        batch_signed_vpi_zscore = []  # (B, T_m) - Volume-price efficiency
        batch_drawdown60 = []         # (B, T_m) - Rolling Drawdown 60 (New)

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

            returns_traj = returns[
                traj_slice_extended
            ]  # (T_m + h, N) or shorter if end of data

            # Pad if shorter (should not happen given max_start, but for safety)
            req_len = T_actual + self.horizon
            if returns_traj.shape[0] < req_len:
                pad_len = req_len - returns_traj.shape[0]
                # Pad with zeros (neutral return) - use same device as source tensor
                pad = th.zeros(
                    (pad_len, returns_traj.shape[1]), device=returns_traj.device
                )
                returns_traj = th.cat([returns_traj, pad], dim=0)

            batch_price_returns.append(returns_traj)

            # Market returns for Baseline
            if use_market_index:
                mkt_ret_traj = market_returns[traj_slice_extended].squeeze(
                    -1
                )  # (T_m + h,)
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
                traj_low = market_ochlv[traj_slice, 0, 3]
                traj_close = market_ochlv[traj_slice, 0, 1]
            else:
                # Fallback: Mean of stocks
                traj_open = ochlv[traj_slice, :, 0].mean(dim=1)
                traj_high = ochlv[traj_slice, :, 2].mean(dim=1)
                traj_low = ochlv[traj_slice, :, 3].mean(dim=1)
                traj_close = ochlv[traj_slice, :, 1].mean(dim=1)

            # Pre-calc ATR for the whole trajectory (approximate, should ideally look back further before t_s)
            # To do this correctly without fetching outside traj_slice inside loop,
            # we should access data_tensors directly with indices.

            dir_labels = th.zeros(T_actual, dtype=th.long, device=self.device)
            # Dynamic Threshold parameters from config (Spec §5.1.3)
            # Optimized for VNINDEX: k_atr=2.0, δ_min=2%, SL=-7%
            atr_period = int(getattr(self.config, "direction_label_atr_period", 14))
            k_atr = float(getattr(self.config, "direction_label_atr_multiplier", 2.0))
            delta_min = float(getattr(self.config, "direction_label_delta_min", 0.020))
            stop_loss_limit = float(
                getattr(self.config, "direction_label_stop_loss", -0.07)
            )

            for t in range(T_actual):
                current_idx = t_s + t
                future_idx = min(current_idx + dir_lookahead, T_total - 1)

                # 1. Get ATR(14)
                # Need atr_period TRs: grab atr_period+1 prices to compute TR.
                atr_window_start = max(0, current_idx - (atr_period + 1))

                # Extract simple window for ATR calc
                if use_market_index:
                    h_slice = market_ochlv[atr_window_start : current_idx + 1, 0, 2]
                    l_slice = market_ochlv[atr_window_start : current_idx + 1, 0, 3]
                    c_slice = market_ochlv[atr_window_start : current_idx + 1, 0, 1]
                    p_t = market_ochlv[current_idx, 0, 1].item()
                else:
                    h_slice = ochlv[atr_window_start : current_idx + 1, :, 2].mean(
                        dim=1
                    )
                    l_slice = ochlv[atr_window_start : current_idx + 1, :, 3].mean(
                        dim=1
                    )
                    c_slice = ochlv[atr_window_start : current_idx + 1, :, 1].mean(
                        dim=1
                    )
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
                future_slice_idx = slice(
                    current_idx + 1, min(current_idx + dir_lookahead + 1, T_total)
                )

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
                    dir_labels[t] = 0  # Bear
                # Bull (2): r_fut > delta_t AND intra_dd >= stop_loss
                elif r_fut > delta_t and intra_dd >= stop_loss_limit:
                    dir_labels[t] = 2  # Bull
                # Side (1): Otherwise
                else:
                    dir_labels[t] = 1  # Side

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
                    price_series = th.cat(
                        [market_ochlv[current_idx : current_idx + 1, 0, 1], prices]
                    )  # Includes P_t at index 0
                else:
                    # Fallback
                    prices = ochlv[future_slice, :, 1].mean(dim=1)
                    p_t = ochlv[current_idx, :, 1].mean().item()
                    price_series = th.cat(
                        [ochlv[current_idx : current_idx + 1, :, 1].mean(dim=1), prices]
                    )

                if len(prices) > 0:
                    # 1. Component 1: Relative Valuation (Z-score)
                    mu_fut = prices.mean().item()
                    std_fut = prices.std().item()
                    eps = 1e-6
                    # Stabilized Z-score: Prevent explosion when vol is near zero
                    # Min denominator: 0.5% of price or eps
                    denom = max(std_fut, max(p_t * 0.005, eps))
                    z_score = (mu_fut - p_t) / denom
                    term_val = (
                        lambda_val * np.tanh(z_score)
                    )  # Note: Removing kappa per strict spec eq 1025? No, spec 5.1.2 simply says tanh(Z_fut).

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

            batch_risk_targets.append(
                th.nan_to_num(risk_targets, nan=1.0, posinf=1.0, neginf=1.0)
            )

            # Store dates
            batch_dates.append(dates[traj_slice])

        # =================================================================
        # Pre-calculate Wide Path Explicit Signals (Spec §3.5.1 v2.1)
        # 4 signals: DC_Event_Flag, Breadth_Gap, Div_Signal, Signed_VPI_Zscore
        # =================================================================

        # Helper: Compute RSI for a price series
        def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
            """Compute RSI for a price series. Returns array of same length."""
            rsi = np.zeros(len(prices), dtype=np.float32)
            if len(prices) < period + 1:
                return rsi + 50.0  # Neutral RSI
            deltas = np.diff(prices)
            for i in range(period, len(prices)):
                window = deltas[i - period : i]
                gains = np.maximum(window, 0)
                losses = np.abs(np.minimum(window, 0))
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
                rs = avg_gain / (avg_loss + 1e-8)
                rsi[i] = 100 - (100 / (1 + rs))
            return rsi

        dc_threshold = float(getattr(self.config, "mafia_DC_threshold", 0.02))

        for b, t_s in enumerate(start_indices):
            # Need extra history for indicators
            history_needed = 25  # RSI_14 + divergence lookback(5)
            full_start = max(0, t_s - history_needed)
            full_end = t_s + T_actual

            # Stock data: close prices (L, N) and volumes (L, N)
            stock_closes = ochlv[full_start:full_end, :, 1].cpu().numpy()
            stock_volumes = ochlv[full_start:full_end, :, 4].cpu().numpy()

            # Market data
            if use_market_index:
                market_closes = market_ochlv[full_start:full_end, 0, 1].cpu().numpy()
                market_volumes = market_ochlv[full_start:full_end, 0, 4].cpu().numpy()
            else:
                market_closes = stock_closes.mean(axis=1)
                market_volumes = stock_volumes.mean(axis=1)

            traj_offset = t_s - full_start
            N_stocks = stock_closes.shape[1]

            # Initialize trajectory arrays
            vol_std20_traj = np.zeros(T_actual, dtype=np.float32)
            dc_event_flag_traj = np.zeros(T_actual, dtype=np.float32)
            breadth_gap_traj = np.zeros(T_actual, dtype=np.float32)
            div_signal_traj = np.zeros(T_actual, dtype=np.float32)
            signed_vpi_zscore_traj = np.zeros(T_actual, dtype=np.float32)
            drawdown60_traj = np.zeros(T_actual, dtype=np.float32)

            # === 0. Vol_Std20: 20-day rolling volatility (for rebalancing, NOT Direction Head) ===
            # Compute returns from market closes
            market_returns_np = np.zeros(len(market_closes), dtype=np.float32)
            if len(market_closes) > 1:
                market_returns_np[1:] = np.diff(market_closes) / (market_closes[:-1] + 1e-8)

            # Compute 20-day rolling volatility
            vol_std20_full = np.zeros(len(market_closes), dtype=np.float32)
            for i in range(20, len(market_closes)):
                vol_std20_full[i] = np.std(market_returns_np[i - 20:i])

            # === 1. DC_Event_Flag: Structural break detection ===
            # Binary flag (1.0) when DC event occurs
            p_ext = market_closes[0] if len(market_closes) > 0 else 1.0
            mode = "up"

            dc_flag_full = np.zeros(len(market_closes), dtype=np.float32)
            for i in range(1, len(market_closes)):
                price = market_closes[i]
                var = (price - p_ext) / (p_ext + 1e-8)

                if mode == "up":
                    if var < -dc_threshold:
                        dc_flag_full[i] = 1.0  # Downward DC (Crash)
                        mode = "down"
                        p_ext = price
                    elif price > p_ext:
                        p_ext = price
                else:  # mode == "down"
                    if var > dc_threshold:
                        dc_flag_full[i] = 1.0  # Upward DC (Rally)
                        mode = "up"
                        p_ext = price
                    elif price < p_ext:
                        p_ext = price

            # === 2. Breadth_Gap: avg(RSI_stocks) - RSI_index ===
            # Detects "Xanh vỏ đỏ lòng" (Index up but stocks weak)
            rsi_index_full = compute_rsi(market_closes, period=14)

            # Compute RSI for each stock
            rsi_stocks_full = np.zeros((len(market_closes), N_stocks), dtype=np.float32)
            for n in range(N_stocks):
                rsi_stocks_full[:, n] = compute_rsi(stock_closes[:, n], period=14)

            # Breadth_Gap = avg(RSI_stocks) - RSI_index
            breadth_gap_full = np.mean(rsi_stocks_full, axis=1) - rsi_index_full
            # Normalize to [-1, 1] (divide by 100 since RSI is 0-100)
            breadth_gap_full = breadth_gap_full / 100.0

            # === 3. Div_Signal: RSI slope vs Price slope divergence ===
            # Signal = Sign(RSI_slope) if opposite to Sign(Price_slope), else 0
            div_signal_full = np.zeros(len(market_closes), dtype=np.float32)
            lookback = 5  # Compare over 5 days

            for i in range(lookback, len(market_closes)):
                # Price slope (percentage change over lookback)
                price_slope = (market_closes[i] - market_closes[i - lookback]) / (market_closes[i - lookback] + 1e-8)
                # RSI slope
                rsi_slope = rsi_index_full[i] - rsi_index_full[i - lookback]

                # Divergence: RSI and Price moving in opposite directions
                if np.sign(price_slope) != np.sign(rsi_slope) and abs(rsi_slope) > 5:
                    # Signal = -Sign(Price_slope) indicates potential reversal
                    div_signal_full[i] = -np.sign(price_slope)

            # === 4. Signed_VPI_Zscore: Volume-Price Efficiency Index ===
            # VPI = Sign(ΔP) × |ΔP| / (Vol / Vol_20_avg)
            # Measures money flow efficiency
            vpi_full = np.zeros(len(market_closes), dtype=np.float32)

            for i in range(20, len(market_closes)):
                # Price change
                delta_p = (market_closes[i] - market_closes[i - 1]) / (market_closes[i - 1] + 1e-8)

                # Volume ratio
                vol_20_avg = np.mean(market_volumes[i - 20:i]) + 1e-8
                vol_ratio = market_volumes[i] / vol_20_avg

                # VPI = Sign(ΔP) × |ΔP| / Vol_ratio
                # High price move with moderate volume = efficient (strong trend)
                # Small price move with high volume = inefficient (distribution/accumulation)
                vpi_full[i] = np.sign(delta_p) * abs(delta_p) / (vol_ratio + 1e-8)

            # Z-score normalize VPI
            vpi_mean = np.mean(vpi_full[20:]) if len(vpi_full) > 20 else 0
            vpi_std = np.std(vpi_full[20:]) + 1e-8 if len(vpi_full) > 20 else 1.0
            vpi_zscore_full = (vpi_full - vpi_mean) / vpi_std
            vpi_zscore_full = np.clip(vpi_zscore_full, -3, 3)

            # === 5. Drawdown60: (P_t - Max_P_{t-60..t}) / Max_P ===
            # Calculates "depth" from recent 60-day peak.
            dd60_full = np.zeros(len(market_closes), dtype=np.float32)
            for i in range(1, len(market_closes)):
                start_win = max(0, i - 60)
                # Max High in window (using closes as proxy or Highs if available)
                # Using Closes for simplicity and consistency with index data shape
                window_max = np.max(market_closes[start_win : i+1])
                curr_price = market_closes[i]
                dd = (curr_price - window_max) / (window_max + 1e-8)
                dd60_full[i] = dd

            # Fill trajectory arrays
            for t in range(T_actual):
                idx = traj_offset + t
                if idx < len(vol_std20_full):
                    vol_std20_traj[t] = vol_std20_full[idx]
                if idx < len(dc_flag_full):
                    dc_event_flag_traj[t] = dc_flag_full[idx]
                if idx < len(breadth_gap_full):
                    breadth_gap_traj[t] = breadth_gap_full[idx]
                if idx < len(div_signal_full):
                    div_signal_traj[t] = div_signal_full[idx]
                if idx < len(vpi_zscore_full):
                    signed_vpi_zscore_traj[t] = vpi_zscore_full[idx]

            # === 5. Drawdown60: Rolling Max Drawdown (Pain Index) ===
            # Relative to 60-day rolling peak
            dd60_traj = np.zeros(T_actual, dtype=np.float32)
            
            # Calculate rolling peak over past 60 days
            rolling_peak = np.zeros(len(market_closes), dtype=np.float32)
            for i in range(len(market_closes)):
                start_w = max(0, i - 60 + 1)
                rolling_peak[i] = np.max(market_closes[start_w : i + 1])
                
            # Drawdown = (Price - Peak) / Peak
            drawdowns_full = (market_closes - rolling_peak) / (rolling_peak + 1e-8)
            
            # Fill trajectory
            for t in range(T_actual):
                idx = traj_offset + t
                if idx < len(drawdowns_full):
                    drawdown60_traj[t] = drawdowns_full[idx]

            batch_vol_std20.append(th.from_numpy(vol_std20_traj).to(self.device))
            batch_dc_event_flag.append(th.from_numpy(dc_event_flag_traj).to(self.device))
            batch_breadth_gap.append(th.from_numpy(breadth_gap_traj).to(self.device))
            batch_div_signal.append(th.from_numpy(div_signal_traj).to(self.device))
            batch_signed_vpi_zscore.append(th.from_numpy(signed_vpi_zscore_traj).to(self.device))
            # New field
            batch_drawdown60.append(th.from_numpy(drawdown60_traj).to(self.device))

        # Stack into batch tensors and move to device
        batch = TrajectoryBatch(
            stock_ochlv=th.stack(batch_stock_ochlv, dim=0).to(
                self.device
            ),  # (B, T_m, N, 5)
            market_ochlv=None,  # Not used in batch forward pass directly yet
            price_returns=th.stack(batch_price_returns, dim=0).to(
                self.device
            ),  # (B, T_m, N)
            market_returns=th.stack(batch_market_returns, dim=0).to(
                self.device
            ),  # (B, T_m)
            rebalance_mask=th.stack(batch_rebalance_mask, dim=0).to(
                self.device
            ),  # (B, T_m)
            direction_labels=th.stack(batch_direction_labels, dim=0).to(
                self.device
            ),  # (B, T_m)
            risk_targets=th.stack(batch_risk_targets, dim=0).to(
                self.device
            ),  # (B, T_m)
            start_indices=th.tensor(start_indices, device=self.device),  # (B,)
            dates=np.array(batch_dates),  # (B, T_m)
            stock_list=data_tensors.get("stock_list"),  # List[str] of N symbols
            # Vol_Std20 for rebalancing logic (NOT for Direction Head)
            vol_std20=th.stack(batch_vol_std20, dim=0).to(self.device),  # (B, T_m)
            # Wide Path Explicit Signals (Spec §3.5.1 v2.1)
            dc_event_flag=th.stack(batch_dc_event_flag, dim=0).to(self.device),  # (B, T_m)
            breadth_gap=th.stack(batch_breadth_gap, dim=0).to(self.device),  # (B, T_m)
            div_signal=th.stack(batch_div_signal, dim=0).to(self.device),
            signed_vpi_zscore=th.stack(batch_signed_vpi_zscore, dim=0).to(self.device),
            drawdown60=th.stack(batch_drawdown60, dim=0).to(self.device),
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
        # Step-wise Gumbel Annealing (Smoother Decay)
        # Update temp every step based on fractional epoch progress
        if hasattr(self, "steps_per_epoch") and self.steps_per_epoch > 0:
            fractional_epoch = epoch_idx + (batch_idx / float(self.steps_per_epoch))
            self._current_temp = self.observer.update_temperature(fractional_epoch)

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

        # [METRIC FIX] Collect unscaled returns for financial metrics
        collected_unscaled_returns = []

        # Initialize 'prev_indices' for turnover calculation (held portfolio)
        # At t=0, previous portfolio is empty set.
        prev_indices = [
            th.tensor([], dtype=th.long, device=self.device) for _ in range(B)
        ]

        # Initialize 'prev_holdings' Binary Mask for Memory Injection (Spec "Memory Injection")
        # Shape: (B, N) - 1.0 if held, 0.0 otherwise
        N_action = self.observer.action_dim
        prev_holdings = th.zeros(B, N_action, device=self.device)

        # Initialize Context Buffer for Temporal Augmentation (Spec 3.6)
        # Cold start with zeros for each batch item
        W_route = int(getattr(self.config, "router_context_window", 5))
        # Get D from model or config
        if hasattr(self.observer.mafia_model, "D"):
            mafia_D = self.observer.mafia_model.D
        else:
            mafia_D = int(getattr(self.config, "mafia_D", 128))

        batch_context_buffer = th.zeros(B, W_route, mafia_D, device=self.device)

        # Get full OCHLV data for feature window extraction
        full_ochlv = data_tensors["ochlv"]  # (T_total, N, 5)
        full_market_ochlv = data_tensors.get("market_ochlv")

        # Track dynamic Regime Shift Logic
        # 1. Volatility Shock: Detected from batch.vol_std20 using threshold
        # 2. DC Events: Detected from batch.dc_event_flag > 0.5
        # 3. Direction Reversal: Calculated online with N_confirm confirmation window (Spec §8.2.1)
        N_confirm = int(getattr(self.config, "regime_confirmation_window", 3))
        # Direction history buffer: (B, N_confirm+1) - stores last N_confirm+1 predictions
        # Initialize with Side(1) as neutral state
        direction_history = th.ones(B, N_confirm + 1, dtype=th.long, device=self.device)
        effective_rebalance_masks = []  # List of (B,) tensors

        # Spec 5.2: Track actual rebalance event count for PG normalization
        # L_PG normalized by: Σ m_t + ε (actual event count, not expected)
        actual_rebal_count = 0.0  # Running count of rebalance events across batch

        # Compute volatility threshold for shock detection (Spec §8.1.2)
        # V_shock = μ_vol + k × σ_vol where k = vol_k (default 2.0)
        vol_mean = batch.vol_std20.mean()
        vol_std = batch.vol_std20.std()
        vol_shock_threshold = vol_mean + self.vol_k * vol_std

        # FIX: Pre-compute EXPECTED rebalance count from static masks for stable normalization
        # This is used for loss normalization DURING the loop (avoids running count bias).
        # Static triggers: schedule (rebalance_mask) + vol_shock + dc_events
        # Note: Direction reversal triggers are computed online and not included here,
        # but they are relatively rare compared to scheduled/event triggers.
        static_trigger_mask = (
            (batch.rebalance_mask > 0.5)  # Schedule triggers
            | (batch.vol_std20 > vol_shock_threshold)  # Volatility shock triggers
            | (batch.dc_event_flag > 0.5)  # DC event triggers (binary flag)
        )  # (B, T_m) Boolean
        expected_rebal_count = static_trigger_mask.float().sum().item()  # Scalar
        # Ensure non-zero for normalization
        expected_rebal_count = max(expected_rebal_count, 1.0)
        collected_market_context = []  # List of (B, D) tensors

        # Reward breakdown collection
        collected_raw_returns = []
        collected_turnovers = []
        collected_symdiffs = []
        collected_rewards = []
        collected_baselines = []
        collected_advantages = []
        collected_hold_rewards = []  # New: Holding scores

        # Loss collections (needed for backward pass later)
        pg_losses = []
        risk_losses = []
        dir_losses = []
        balance_losses = [] # [NEW] Track balance loss per step

        # Dynamic Schedule Tracking (Resets on ANY trigger)
        # Initialize to 0. At t=0, we force rebalance.
        days_since_last_rebal = th.zeros(B, dtype=th.long, device=self.device)
        rebal_interval = self.rebalance_interval

        # Track start index for retrospective log update
        if not hasattr(self, "_trajectory_log"):
             self._trajectory_log = []
        start_log_idx = len(self._trajectory_log)

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
                market_ochlv_batch = th.stack(batch_windows_mkt, dim=0).to(
                    self.device
                )  # (B, 1, 5, T_w)
            
            # [DEBUG] Check for NaNs in inputs (Critical input validation)
            if th.isnan(ochlv_batch).any() or th.isinf(ochlv_batch).any():
                smart_print(f"     ⚠️  [CRITICAL] NaN/Inf detected in ochlv_batch at t={t}")
                raise ValueError(f"NaN in ochlv_batch at t={t}")
            
            if market_ochlv_batch is not None and (th.isnan(market_ochlv_batch).any() or th.isinf(market_ochlv_batch).any()):
                smart_print(f"     ⚠️  [CRITICAL] NaN/Inf detected in market_ochlv_batch at t={t}")
                raise ValueError(f"NaN in market_ochlv_batch at t={t}")

            # --- Check Regular Schedule Mask (Dynamic) ---
            # sched_mask = batch.rebalance_mask[:, t]  # OLD: Static mask

            if t == 0:
                # Force rebalance at start of trajectory
                sched_trigger = th.ones(B, dtype=th.bool, device=self.device)
                days_since_last_rebal[:] = 0
            else:
                days_since_last_rebal += 1
                sched_trigger = days_since_last_rebal >= rebal_interval

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
                vol_trigger = batch.vol_std20[:, t] > vol_shock_threshold  # (B,)
                dc_trigger = (
                    batch.dc_event_flag[:, t] > 0.5
                )  # (B,) - DC Event confirmed (binary flag)
                # sched_trigger is already boolean (B,) computed above

                # Preliminary trigger (Explicit triggers known at input)
                pre_trigger = vol_trigger | sched_trigger | dc_trigger  # (B,)

                # Prepare force indices for this step (initially based on pre_trigger)
                # If trigger -> None (fresh selection). Else -> prev indices.
                current_force_indices = None
                if t > 0 and collected_topk_indices:
                    prev_indices = collected_topk_indices[-1]  # (B, K)
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

                    # Correct approach for Mixed Batch:
                    # 1. Run Forward with force_topk_indices=None (Always generate fresh Top-K candidates)
                    # 2. Get `new_indices`
                    # 3. For items where `rebalance=False`: Overwrite `new_indices` with `prev_indices`.
                    # 4. This ensures `topk_scores` etc match the actual portfolio held.
                    pass

                # Build explicit signals for Direction Head (Spec 3.5.1 v2.1 - Wide Path)
                # 6 signals: Vol_Std20, DC_Event_Flag, Breadth_Gap, Div_Signal, Signed_VPI_Zscore, Drawdown60
                vol_std20_value = batch.vol_std20[:, t : t + 1] # (B, 1)
                dc_flag_value = batch.dc_event_flag[:, t : t + 1]  # (B, 1) - already binary [0, 1]
                breadth_gap_value = batch.breadth_gap[:, t : t + 1]  # (B, 1) - already normalized [-1, 1]
                div_signal_value = batch.div_signal[:, t : t + 1]  # (B, 1) - already {-1, 0, 1}
                vpi_zscore_value = batch.signed_vpi_zscore[:, t : t + 1]  # (B, 1) - already z-scored [-3, 3]
                drawdown60_value = batch.drawdown60[:, t : t + 1] # (B, 1)

                # Signals are already normalized during computation, concat directly
                explicit_signals = th.cat(
                    [
                        vol_std20_value,
                        dc_flag_value,
                        breadth_gap_value,
                        div_signal_value,
                        vpi_zscore_value,
                        drawdown60_value,
                    ], dim=-1
                )  # (B, 6)

                # Pass buffer to model (Spec 3.6)
                outputs = self.observer.mafia_model(
                    ochlv_data=ochlv_batch,
                    market_index_ochlv_data=market_ochlv_batch,
                    force_topk_indices=None,  # Always fresh selection first
                    router_context_buffer=batch_context_buffer,  # Batched buffer for temporal augmentation
                    explicit_signals=explicit_signals,  # Direction Head signals (Spec 3.5)
                    prev_holdings=prev_holdings,  # Memory Injection: Bias current logits with last holdings
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
                    market_logits,  # (B, N) - RAW LOGITS
                ) = outputs

                # Temperature scaling for direction head (sharpen/flatten)
                if self.direction_temperature != 1.0:
                    direction_logits = direction_logits / self.direction_temperature

                # Save FRESH candidates before holding logic may overwrite
                fresh_topk_indices = topk_indices.clone()
                collected_market_context.append(market_context)  # (B, D)

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
                direction_history = th.cat(
                    [direction_history[:, 1:], pred_state.unsqueeze(1)], dim=1
                )

                # Extract previous regime for logging (before confirmation window)
                prev_pred_state = direction_history[
                    :, 0
                ]  # (B,) - oldest regime in buffer

                # Check reversal with confirmation window
                # Need at least N_confirm+1 predictions to detect reversal
                if t < N_confirm:
                    dir_trigger = th.zeros(B, dtype=th.bool, device=self.device)
                else:
                    # last_seq = direction_history[:, -N_confirm:]  # Last N_confirm predictions
                    # prev_regime = direction_history[:, -(N_confirm+1)]  # Prediction before confirmation window
                    last_seq = direction_history[
                        :, 1:
                    ]  # (B, N_confirm) - last N_confirm predictions
                    prev_regime = direction_history[
                        :, 0
                    ]  # (B,) - prediction before window

                    BULL, BEAR = 2, 0

                    # Bear → Bull: prev_regime was Bear, all last N_confirm are Bull
                    all_bull = (last_seq == BULL).all(dim=1)  # (B,)
                    was_bear = prev_regime == BEAR  # (B,)
                    bear_to_bull = all_bull & was_bear

                    # Bull → Bear: prev_regime was Bull, all last N_confirm are Bear
                    all_bear = (last_seq == BEAR).all(dim=1)  # (B,)
                    was_bull = prev_regime == BULL  # (B,)
                    bull_to_bear = all_bear & was_bull

                    dir_trigger = bear_to_bull | bull_to_bear

                # --- Final Effective Rebalance Mask ---
                effective_mask = pre_trigger | dir_trigger  # (B,) Boolean
                effective_mask_float = effective_mask.float()  # (B,) 1.0 or 0.0
                effective_rebalance_masks.append(effective_mask_float)

                # Spec 5.2: Accumulate actual rebalance event count
                # Sum across batch dimension for this timestep
                actual_rebal_count += effective_mask_float.sum().item()

                # Reset Schedule Counter where rebalance occurred
                days_since_last_rebal[effective_mask] = 0

                # --- Apply Holding Logic ---
                # If effective_mask[b] is 0 -> We MUST hold previous portfolio.
                if t > 0 and collected_topk_indices:
                    prev_idx = collected_topk_indices[-1]  # (B, K)
                    prev_sc = collected_topk_scores[-1]  # (B, K)

                    # Overwrite where mask is 0
                    # We need to broadcast mask to (B, K)
                    mask_k = effective_mask.unsqueeze(1).expand(-1, self.K)  # (B, K)

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
            # Sanitize topk_scores to prevent NaN propagation in turnover calculation
            topk_scores = th.nan_to_num(topk_scores, nan=0.0, posinf=10.0, neginf=-10.0)
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
                future_returns_slice = batch.price_returns[
                    :, start_idx:end_idx, :
                ]  # (B, h, N)
            # ============================================================
            # REAL-TIME METRICS & LOGGING (Moved Inside Loop)
            # ============================================================

            # 1. Calculate Rewards (R_t, Baseline) immediately
            start_idx = t + 1
            end_idx = t + 1 + self.horizon
            future_returns_slice = batch.price_returns[
                :, start_idx:end_idx, :
            ]  # (B, h, N)
            market_returns_slice = batch.market_returns[:, start_idx:end_idx]  # (B, h)
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
                R_unscaled = th.zeros(B, device=self.device)
            else:
                topk_idx = topk_indices  # (B, K)
                gather_idx = topk_idx.unsqueeze(1).expand(-1, h_len, -1)
                selected_returns = th.gather(future_returns_slice, 2, gather_idx)
                compounded_stock_returns = th.prod(1 + selected_returns, dim=1) - 1
                # Apply S_reward scaling (Spec §5.1.1) to amplify gradient signal
                R_unscaled = compounded_stock_returns.mean(dim=1)
                R_raw = self.scale_factor_reward * R_unscaled
                baseline = self.scale_factor_reward * (
                    th.prod(1 + market_returns_slice, dim=1) - 1
                )

            # Penalties
            turnover = th.zeros(B, device=self.device)
            symdiff = th.zeros(B, device=self.device)

            if t > 0:
                # Compute Penalties
                # 1. SymDiff (Membership Change Cost) - Set Based
                # SymDiff = 2 * (K - Intersection) / K
                for b_i in range(B):
                    curr_set = set(topk_indices[b_i].cpu().tolist())
                    prev_set = set(prev_indices[b_i].cpu().tolist())

                    held_count = len(curr_set.intersection(prev_set))
                    s_val = (2 * (self.K - held_count)) / self.K
                    symdiff[b_i] = s_val
                
                # 2. Turnover (Re-weighting Cost) - Weight Based (L1 Distance)
                # Turnover = 0.5 * sum(|w_t - w_{t-1}|)
                # Reconstruct full weight vectors (B, N)
                # Weights are derived from scores (normalized). Assuming Softmax was applied to raw logits.
                # However, topk_scores are typically log_probs or logits.
                # For fairness, we should use the implied portfolio weights.
                # If we assume equal weight on Top-K, Turnover == SymDiff/2.
                # But here we use `topk_scores` (softmax probs).
                
                # Normalizing scores to sum to 1 (Portfolio Weights)
                # curr_weights (B, K) -> Scatter to (B, N)
                # prev_weights (B, K) -> Scatter to (B, N)
                
                N_stocks = batch.stock_ochlv.shape[2]
                
                curr_w_vector = th.zeros(B, N_stocks, device=self.device)
                curr_local_w = F.softmax(topk_scores, dim=1) # Normalize selected scores to legitimate weights
                curr_w_vector.scatter_(1, topk_indices, curr_local_w)
                
                prev_w_vector = th.zeros(B, N_stocks, device=self.device)
                
                # BUGFIX: We appended current scores to `collected_topk_scores` at line 1980.
                # So `collected[-1]` is CURRENT step. `collected[-2]` is PREVIOUS.
                if len(collected_topk_scores) >= 2:
                     p_scores = collected_topk_scores[-2]
                     p_idx = collected_topk_indices[-2]
                     prev_local_w = F.softmax(p_scores, dim=1)
                     prev_w_vector.scatter_(1, p_idx, prev_local_w)
                
                # L1 Distance
                weight_diff = (curr_w_vector - prev_w_vector).abs().sum(dim=1) # (B,)
                if th.isnan(weight_diff).any():
                     smart_print(f"     ⚠️  [WARN] NaN in turnover weight_diff at t={t}. (Should not happen with stable logits).")
                     # weight_diff = th.nan_to_num(weight_diff, nan=0.0) # Removed mask
                turnover = 0.5 * weight_diff
                turnover = th.clamp(turnover, 0.0, 1.0) # Ensure range [0, 1]


            # Trend Holding Reward (R_hold)
            # Bonus for holding profitable stocks (Trend Following)
            # FIX: Must intersect prev_holdings with CURRENT holdings (mask_k indicates whether we kept indices)
            # Actually, `topk_indices` represents the ACTUAL portfolio for step T (after logic).
            # So we just need to check if `topk_indices` overlaps with `prev_holdings`.
            
            r_hold = th.zeros(B, device=self.device)
            if self.r_hold_alpha > 0 and h_len > 0:
                # OPTIMIZED HOLD REWARD: "Adaptive Duration Winner"
                # Instead of 1-day check, we check consistency over the full horizon (h_len).
                # Score = (Count of Days where R_stock > R_mkt AND R_stock > 0) / h_len
                
                # 1. Identify winning days (B, h, N)
                # Expand market returns for broadcast: (B, h) -> (B, h, 1)
                mkt_ret_expanded = market_returns_slice.unsqueeze(-1)
                
                # Check condition: Stock > Market AND Stock > 0
                win_day_mask = (future_returns_slice > mkt_ret_expanded) & (future_returns_slice > 0)
                
                # 2. Compute Consistency Score per Stock (B, N) -> range [0.0, 1.0]
                stock_win_consistency = win_day_mask.float().sum(dim=1) / float(h_len)
                
                # 3. Construct Binary Mask of Currently Held Portfolio (B, N)
                curr_holdings = th.zeros(B, N_action, device=self.device)
                curr_holdings.scatter_(1, topk_indices, 1.0)
                
                # 4. Filter only Held Stocks (Intersection: Prev & Curr)
                held_mask = prev_holdings * curr_holdings # (B, N)
                
                # 5. Sum Scores for Held Stocks
                # If we hold a stock that wins 100% of days, we get full points.
                # If we hold a stock that wins 0% of days, we get 0 points.
                portfolio_win_score = (stock_win_consistency * held_mask).sum(dim=1) # (B,)
                
                # 6. Apply Reward Scale
                # r_hold = alpha * (Sum_Scores / K)
                # Max potential r_hold = alpha (if we hold K stocks that win every day)
                r_hold = self.r_hold_alpha * (portfolio_win_score / self.K)

            # Penalties

            # Advantage (Spec §5.1.1)
            # A_raw = (R_t - λ_epoch × Penalties) - b_t
            # λ_epoch: Curriculum penalty weight (0 in warmup, ramps to 1)
            # NOTE: Curriculum learning ONLY applies in TRAINING mode
            # Validation/Inference always use λ_epoch = 1.0 (full penalty)
            lambda_epoch = self._current_lambda_epoch
            penalty = lambda_epoch * (
                self.alpha_turnover * turnover + self.alpha_change * symdiff
            )
            # R_net = R_raw + R_hold - penalty
            # Note: R_hold is added to R_net but usually penalties are subtracted.
            # R_hold is a positive incentive.
            R_net = R_raw + r_hold - penalty
            A_t_raw = R_net - baseline

            # Normalize Advantage (Batch statistics)
            A_mean = A_t_raw.mean()
            eps = 1e-5  # Increased from 1e-8 for small batch stability
            if B > 1:
                A_std = A_t_raw.std() + eps
            else:
                A_std = 1.0
            A_t = (A_t_raw - A_mean) / A_std

            # Store for Post-Loop Loss Calc (still needed for backprop)
            collected_raw_returns.append(R_raw)
            collected_unscaled_returns.append(R_unscaled) # [METRIC FIX]
            collected_turnovers.append(turnover)
            collected_symdiffs.append(symdiff)
            collected_rewards.append(R_net)
            collected_hold_rewards.append(r_hold)
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
            if batch.vol_std20[b, t] > vol_shock_threshold:
                triggers.append("Vol Shock")
            if batch.dc_event_flag[b, t] > 0.5:
                triggers.append("Struct Break (DC)")
            if dir_trigger[b].item():
                prev_s_idx = int(min(max(prev_pred_state[b].item(), 0), 2))
                prev_s = ["Bear", "Side", "Bull"][prev_s_idx]
                triggers.append(f"Regime Shift ({prev_s}->{dir_pred_str})")
            elif t == 0:
                triggers.append("Regime Shift (Initial)")

            trigger_str = " | ".join(triggers) if triggers else "None"
            trigger_desc = trigger_str  # Always show triggers

            if is_rebalance_b0:
                action_str = "🔄 REBALANCE"
            else:
                action_str = "⏸  HOLD"

            # Context Difference (Dynamic Check)
            curr_c_mkt = market_context[b]
            if not hasattr(self, "_prev_c_mkt_b0"):
                self._prev_c_mkt_b0 = None

            if self._prev_c_mkt_b0 is not None:
                c_mkt_diff = (curr_c_mkt - self._prev_c_mkt_b0).norm().item()
            else:
                c_mkt_diff = 0.0

            # Update prev for next step
            self._prev_c_mkt_b0 = curr_c_mkt.detach()  # Detach to be safe

            # Portfolio
            sel_stocks = topk_indices[b].cpu().tolist()

            # Hold count for display
            if not hasattr(self, "_last_rebalance_portfolio"):
                self._last_rebalance_portfolio = {}
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
                    stock_str = ", ".join(syms[:5]) + f", ... (+{len(syms) - 5})"
                else:
                    stock_str = ", ".join(syms)
            else:
                stock_str = ", ".join([f"S{i}" for i in sel_stocks[:5]])

            c_mkt_norm = market_context[b].norm().item()
            eta_val = risk_eta[b].item()

            # PG Loss Estimate (Pend)
            # Sanitize market_logits to prevent NaN propagation
            if th.isnan(market_logits).any() or th.isinf(market_logits).any():
                market_logits = th.nan_to_num(market_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            # Use stable log_softmax on logits
            log_market_probs = F.log_softmax(market_logits, dim=-1)  # (B, N)
            
            log_probs = th.gather(log_market_probs, 1, topk_indices) # (B, K)

            imp_weights = th.ones_like(log_probs)
            # Spec §5.1.1: Advantage must be detached to prevent gradient flow through it
            # This ensures model learns to change π(a|s), not "hack" the reward
            pg_term = -(log_probs * imp_weights * A_t.detach().unsqueeze(1)).mean(
                dim=1
            )  # (B,)

            # Entropy bonus on full distribution (spec 5.1.1)
            # H(π) = -Σ p_i × log(p_i)
            # Use probs * log_probs for stability (0 * -inf handled by torch usually? No, but probs will be 0)
            # Actually, log_softmax output is safe. 
            entropy = -(market_scores_full * log_market_probs).sum(
                dim=1
            )  # (B,)

            # L_PG = pg_term - β_ent × H(π) (subtract entropy to bonus exploration)
            step_loss_val = pg_term - self.beta_entropy * entropy  # (B,)

            # Auxiliary Losses
            risk_target = batch.risk_targets[:, t]
            risk_loss = (
                self.risk_criterion(risk_eta, risk_target) * self.risk_scaling_factor
            )

            # [NAN-FIX] Input Guard for Direction Loss
            dir_target = batch.direction_labels[:, t]
            
            if th.isnan(direction_logits).any():
                smart_print(f"     ⚠️  [WARN] NaN detected in direction_logits at t={t}. Skipping DirLoss.")
                dir_loss = th.tensor(0.0, device=self.device, requires_grad=True)
                # Sample loss also undefined
                dir_loss_sample = th.zeros(1, device=self.device)
            else:
                dir_loss = self.dir_criterion(direction_logits, dir_target)
                
                # Sample-specific Losses (for logging only)
                # Compute sample loss here to match logic
                dir_loss_sample = self.dir_criterion(
                    direction_logits[b : b + 1], dir_target[b : b + 1]
                )

            # Sample-specific Losses (Risk is fine usually unless eta is NaN)
            risk_loss_sample = (
                self.risk_criterion(
                    risk_eta[b : b + 1], risk_target[b : b + 1]
                )
                * self.risk_scaling_factor
            )
            # dir_loss_sample is computed above or dummy


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
            if not hasattr(self, "_hold_counters"):
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

            header = f"  ⏱  [Epoch {epoch_idx}] Batch {batch_idx + 1} | Step {t + 1:3d}/{T_m} | {action_str}"
            # Show sample trajectory info (trajectory 0 of B)
            smart_print(f"{header}  [Sample 1/{B} trajectories]")
            smart_print(f"     Trigger: {display_trigger}")

            # Spec: R_t = RawReturn. A_t = (R_net) - Baseline.
            # To align with user mental model, we show Raw -> Costs -> Net -> Adv

            # 1. Raw Performance
            raw_str = f"RawReturn={R_raw[b].item():+.4f}"

            # 2. Costs/Penalties (scaled by curriculum λ_epoch)
            # CLARITY FIX: Only show actual penalty values when REBALANCING
            # During HOLD, penalties are computed but NOT applied to loss (masked out)
            if is_rebalance_b0:
                turn_pen = turnover[b].item() * self.alpha_turnover * lambda_epoch
                diff_pen = symdiff[b].item() * self.alpha_change * lambda_epoch
                cost_str = f"TurnPen={turn_pen:.4f} | SymDiffPen={diff_pen:.4f}"
            else:
                cost_str = f"TurnPen=N/A (HOLD) | SymDiffPen=N/A (HOLD)"

            # 3. Net & Advantage
            r_net_val = R_net[b].item()
            net_str = f"R_total={r_net_val:+.4f}"
            
            # Hold Bonus Logging
            hold_val = r_hold[b].item()
            # User request: Don't show (Masked) if reward is active
            # The reward is always added to R_net, so it's always "active" for evaluation
            mask_suffix = "" 
            
            # Always display HoldBonus, even if 0.0, to keep log format consistent
            # [LOGGING] Add Consistency/WinRate details if available
            if "portfolio_win_score" in locals():
                score_val = portfolio_win_score[b].item()
                # Score is sum of consistency scores. Avg Consistency = Score / Held_Count
                if held > 0:
                    avg_consistency = score_val / held
                else:
                    avg_consistency = 0.0
                consist_str = f" | Consistency={avg_consistency:.2f}(Sum={score_val:.1f})"
            else:
                consist_str = ""

            hold_str = f"R_hold={hold_val:+.4f}{mask_suffix}{consist_str}"

            adv_str = (
                f"R_baseline={baseline[b].item():+.4f} | "
                f"Advantage={A_t_raw[b].item():+.4f}"
            )

            smart_print(f"     📈 Market:    {raw_str} | CtxDiff={c_mkt_diff:.3f}")
            smart_print(f"     💸 Costs:     {cost_str}")
            smart_print(f"     ⚖️  Outcome:   {net_str} | {adv_str} | {hold_str}")
            smart_print(
                f"     📉 Losses:    L_PG={pg_val:.4f} | "
                f"L_risk(B)={risk_loss.item():.4f} | L_risk(S)={risk_loss_sample.item():.4f} | "
                f"L_dir(B)={dir_loss.item():.4f} | L_dir(S)={dir_loss_sample.item():.4f} | "
                f"L_bal={loss_balance.item() if 'loss_balance' in locals() else 0.0:.4f}"
            )

            # Portfolio & State
            smart_print(
                f"     📊 Portfolio: Held={held}, Added={changed} | {stock_str}"
            )

            # Predictions
            probs = th.nn.functional.softmax(direction_logits[b], dim=-1)
            logits_str = (
                f"Bear={probs[0]:.2f}, Side={probs[1]:.2f}, Bull={probs[2]:.2f}"
            )
            smart_print(
                f"     🔮 Forecast:  Risk(η)={eta_val:.2f}(Tg={risk_target[b].item():.2f}) | Dir={dir_pred_str}{correct_dir} [{logits_str}]"
            )

            # --- Consistency Check Logging (Hook Data) ---
            if self._latest_gate_weights is not None:
                # Get weights for this batch item (take last captured if multiple passes, usually fine for step-by-step)
                # Ensure shape match: (B, 4)
                if self._latest_gate_weights.size(0) > b:
                    w_tech = self._latest_gate_weights[b, 0].item() * 100
                    w_dc1 = self._latest_gate_weights[b, 1].item() * 100
                    w_dc2 = self._latest_gate_weights[b, 2].item() * 100
                    w_dc3 = self._latest_gate_weights[b, 3].item() * 100
                    
                    gate_str = f"Tech={w_tech:.1f}%, DC1={w_dc1:.1f}%, DC2={w_dc2:.1f}%, DC3={w_dc3:.1f}%"
                    smart_print(f"     🧩 Gate:      {gate_str}")
                    
                    # Detection Logic
                    # Inconsistency 1: Bear Forecast but High Tech Allocation
                    if pred_state_b == 0 and w_tech > 50.0: # Bear
                        smart_print(f"     ⚠️  [DETECT] Inconsistency: Bearish forecast but relying heavily on Tech Expert ({w_tech:.1f}%)")
                    
                    # Inconsistency 2: Bull Forecast but Low Tech Allocation (Avoiding Trend)
                    if pred_state_b == 2 and w_tech < 20.0: # Bull
                        smart_print(f"     ⚠️  [DETECT] Inconsistency: Bullish forecast but avoiding Tech Expert ({w_tech:.1f}%)")


            if t == T_m - 1:
                # End of trajectory - show batch-wide statistics
                smart_print("-" * 100)
                smart_print(
                    f"  📊 BATCH SUMMARY (All {B} trajectories processed in PARALLEL):"
                )

                # Compute batch-wide stats from the accumulated tensors
                all_returns = th.stack(collected_raw_returns, dim=1)  # (B, T_m)
                all_rewards = th.stack(
                    collected_rewards, dim=1
                )  # (B, T_m) - Net rewards
                all_baselines = th.stack(collected_baselines, dim=1)  # (B, T_m)

                # Compute RAW advantage (before normalization) for meaningful stats
                all_raw_advantages = all_rewards - all_baselines  # (B, T_m)

                # Compute turnover and symdiff penalties
                all_turnovers = th.stack(collected_turnovers, dim=1)  # (B, T_m)
                all_symdiffs = th.stack(collected_symdiffs, dim=1)  # (B, T_m)

                # Mean across all trajectories
                mean_return = all_returns.mean().item()
                std_return = all_returns.std().item()
                mean_raw_advantage = all_raw_advantages.mean().item()
                mean_net_reward = all_rewards.mean().item()
                mean_turn_pen = (all_turnovers * self.alpha_turnover).mean().item()
                mean_symdiff_pen = (all_symdiffs * self.alpha_change).mean().item()

                # Direction accuracy across ALL timesteps (B × T_m predictions)
                all_dir_logits = th.stack(collected_direction_logits, dim=1)  # (B, T_m, 3)
                all_dir_preds = th.argmax(all_dir_logits, dim=-1)  # (B, T_m)
                all_dir_targets = batch.direction_labels[:, :T_m]  # (B, T_m)
                dir_accuracy = (all_dir_preds == all_dir_targets).float().mean().item() * 100
                total_predictions = B * T_m

                # Prediction distribution analysis (detect bias) - across ALL timesteps
                dir_preds_flat = all_dir_preds.flatten()  # (B * T_m,)
                dir_targets_flat = all_dir_targets.flatten()  # (B * T_m,)
                pred_bear_pct = (dir_preds_flat == 0).float().mean().item() * 100
                pred_side_pct = (dir_preds_flat == 1).float().mean().item() * 100
                pred_bull_pct = (dir_preds_flat == 2).float().mean().item() * 100
                target_bear_pct = (dir_targets_flat == 0).float().mean().item() * 100
                target_side_pct = (dir_targets_flat == 1).float().mean().item() * 100
                target_bull_pct = (dir_targets_flat == 2).float().mean().item() * 100

                # Risk prediction stats across all B
                mean_risk_eta = risk_eta.mean().item()

                # Determine if portfolio is beating market
                adv_status = (
                    "🟢 Beating Market"
                    if mean_raw_advantage > 0
                    else "🔴 Underperforming"
                )

                smart_print(
                    f"     📈 Returns:    Mean={mean_return:+.4f} | Std={std_return:.4f}"
                )
                smart_print(
                    f"     💰 Net Reward: Mean={mean_net_reward:+.4f} | TurnPen={mean_turn_pen:.4f} | SymDiffPen={mean_symdiff_pen:.4f}"
                )
                smart_print(
                    f"     ⚖️  Advantage:  Mean={mean_raw_advantage:+.4f} ({adv_status})"
                )
                smart_print(
                    f"     🎯 Direction:  Accuracy={dir_accuracy:.1f}% (across {total_predictions} predictions = {B} traj × {T_m} steps)"
                )
                smart_print(
                    f"        Preds:   Bear={pred_bear_pct:4.1f}% | Side={pred_side_pct:4.1f}% | Bull={pred_bull_pct:4.1f}%"
                )
                smart_print(
                    f"        Target:  Bear={target_bear_pct:4.1f}% | Side={target_side_pct:4.1f}% | Bull={target_bull_pct:4.1f}%"
                )
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
                (step_pg_loss * norm_pg * self.lambda_pg)
                + (risk_loss * norm_risk * self.lambda_risk)
                + (dir_loss * norm_dir * self.lambda_dir)
            )
            
            # [NEW] Load Balancing Loss (Expert Collapse Prevention)
            loss_balance = th.tensor(0.0, device=self.device)
            if hasattr(self, "_gate_weights_with_grad") and self._gate_weights_with_grad is not None:
                # 1. Get Average Gate distribution across batch: (num_experts,)
                avg_gate = self._gate_weights_with_grad.mean(dim=0)
                # 2. Compute MSE from Uniform (0.25): sum((w - 0.25)^2)
                # or CV^2 (Variance / Mean^2)
                num_experts = avg_gate.size(0)
                target_uniform = 1.0 / num_experts
                loss_balance = (avg_gate - target_uniform).pow(2).sum() * 100.0 # Scale up for magnitude
                
                # Add to total loss
                bal_term = (loss_balance * self.lambda_balance)
                step_total_loss = step_total_loss + bal_term
                
                # Add to list for averaging
                balance_losses.append(loss_balance.item())
                
                # Cleanup reference to free graph after backward
                self._gate_weights_with_grad = None

            # Backward Pass (Per-Step)
            # This accumulates gradients in params. Graph is freed immediately after.
            if step_total_loss.item() != 0.0:  # Optimization: skip if zero
                try:
                    step_total_loss.backward()
                except RuntimeError as e:
                    if "nan" in str(e).lower() or "inf" in str(e).lower():
                        smart_print(
                            f"[WARN] NaN/Inf in backward pass (t={t}). Skipping step."
                        )
                    else:
                        raise e

            # Store Metrics (Float only - NO GRAPH)
            # We multiply back by T_m / effective_count to reconstruct "epoch average" for logging
            pg_losses.append(step_pg_loss.item())  # Raw value
            risk_losses.append(risk_loss.item())
            dir_losses.append(dir_loss.item())

            # Update previous indices for next timestep turnover calculation
            prev_indices = topk_indices

            # --- Update Memory Injection State (prev_holdings) for NEXT step ---
            # This must reflect the ACTUAL portfolio held after this step (whether rebalanced or held)
            prev_holdings = th.zeros(B, N_action, device=self.device)
            prev_holdings.scatter_(1, topk_indices, 1.0)

            # Store for CSV export
            if not hasattr(self, "_trajectory_log"):
                self._trajectory_log = []

            # Determine whether to log ALL trajectories or just the representative one (b=0)
            log_all = getattr(self.config, "log_trajectory_details", False)
            log_indices = range(B) if log_all else [0]

            for b_log in log_indices:
                # Reconstruct triggers for this sample
                triggers_b = []
                if sched_trigger[b_log].item():
                    triggers_b.append("Schedule")
                if batch.vol_std20[b_log, t] > vol_shock_threshold:
                    triggers_b.append("Vol Shock")
                if batch.dc_event_flag[b_log, t] > 0.5:
                    triggers_b.append("Struct Break (DC)")
                
                # Regime Shift logic
                if dir_trigger[b_log].item():
                    prev_s_idx_b = int(min(max(prev_pred_state[b_log].item(), 0), 2))
                    curr_s_idx_b = int(min(max(pred_state[b_log].item(), 0), 2))
                    prev_s_str = ["Bear", "Side", "Bull"][prev_s_idx_b]
                    curr_s_str = ["Bear", "Side", "Bull"][curr_s_idx_b]
                    triggers_b.append(f"Regime Shift ({prev_s_str}->{curr_s_str})")
                elif t == 0:
                    triggers_b.append("Regime Shift (Initial)")
                
                trigger_str_b = " | ".join(triggers_b) if triggers_b else "None"
                
                # Context Diff (Only available for b=0 or just put 0.0)
                # Re-computing it for log is expensive/complex without state tracking.
                c_mkt_diff_b = c_mkt_diff if b_log == 0 else 0.0

                self._trajectory_log.append(
                    {
                        "batch_idx": batch_idx,
                        "sample_idx": b_log,
                        "timestep": t,
                        "trigger": trigger_str_b,
                        "rebalanced": effective_mask[b_log].item() > 0.5,
                        "selected_stocks": ",".join([str(idx) for idx in topk_indices[b_log].cpu().tolist()]),
                        "risk_eta": risk_eta[b_log].item(),
                        "dir_logit_bear": direction_logits[b_log, 0].item(),
                        "dir_logit_side": direction_logits[b_log, 1].item(),
                        "dir_logit_bull": direction_logits[b_log, 2].item(),
                        "c_mkt_norm": market_context[b_log].norm().item(),
                        "c_mkt_diff": c_mkt_diff_b,
                        "raw_return": R_raw[b_log].item(),
                        "turnover_penalty": turnover[b_log].item() * self.alpha_turnover * lambda_epoch, # Log EFFECTIVE penalty
                        "symdiff_penalty": symdiff[b_log].item() * self.alpha_change * lambda_epoch, # Log EFFECTIVE penalty
                        "net_reward": R_net[b_log].item(),
                        "baseline": baseline[b_log].item(),
                        "advantage": A_t_raw[b_log].item(),
                        "advantage_norm": A_t[b_log].item(),
                        "pg_loss": step_pg_loss.item(),  # Log aggregated PG loss (scalar)
                        "risk_loss": risk_loss.item(),   # Log aggregated Risk loss
                        "dir_loss": dir_loss.item(),     # Log aggregated Dir loss
                    }
                )

            # Note: Hidden state is automatically maintained inside the LSTM
            # via _cached_state (detached for TBPTT)

            # Update Context Buffer via FIFO (Spec 3.6.2.2)
            # Remove oldest ([:, 0, :]), append new (market_context)
            # market_context is (B, D). Need (B, 1, D)
            # CAUTION: Detach market_context before adding to buffer to prevent growing graph through time
            # beyond the relevant window or intended BPTT.
            # (TBPTT implies we cut gradients. Buffer acts like state memory. Standard is to detach unless BPTT desired through it.)

            history_part = batch_context_buffer[:, 1:, :]  # (B, W-1, D)
            new_part = market_context.detach().unsqueeze(1)  # (B, 1, D)
            batch_context_buffer = th.cat([history_part, new_part], dim=1)

        # ============================================================
        # STEP 3: OPTIMIZER STEP (After accumulating all T_m gradients)
        # ============================================================

        # ===== Optimizer Step (gradients already accumulated via backward() in loop) =====
        try:
            # Clip gradients
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.observer.mafia_model.parameters(), max_norm=1.0
            )
            if th.is_tensor(grad_norm):
                grad_norm = grad_norm.item()

            # Retrospective Log Update
            if hasattr(self, "_trajectory_log"):
                for i in range(start_log_idx, len(self._trajectory_log)):
                    self._trajectory_log[i]["grad_norm"] = grad_norm

            # Update weights
            self.optimizer.step()
            # If an LR scheduler is used, step it here. Assuming it's self.lr_scheduler
            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.step()
        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                smart_print(
                    f"[WARN] NaN/Inf detected during optimizer step. Skipping update. Error: {e}"
                )
                # Optionally, you might want to re-zero gradients if the step was skipped
                self.optimizer.zero_grad()
                skipped_update = True
                grad_norm = -1.0  # Indicator for failure

                if start_log_idx != -1 and hasattr(self, "_trajectory_log"):
                    for i in range(start_log_idx, len(self._trajectory_log)):
                        self._trajectory_log[i]["grad_norm"] = -1.0
            else:
                raise e
        else:
            skipped_update = False

        # Calculate Average Metrics for Log
        # Note: pg_losses list contains 0s for non-rebal days.
        # We want avg over rebal days only.
        pg_values = [v for v in pg_losses if v != 0.0]
        avg_pg = sum(pg_values) / len(pg_values) if pg_values else 0.0

        avg_risk = sum(risk_losses) / len(risk_losses) if risk_losses else 0.0
        avg_dir = sum(dir_losses) / len(dir_losses) if dir_losses else 0.0
        avg_loss_balance = sum(balance_losses) / len(balance_losses) if balance_losses else 0.0 # [NEW]

        # Reconstruct approximate total loss for display
        # (This won't exactly match the backwarded loss due to estimator norm, but close enough for logs)
        total_disp = (
            self.lambda_pg * avg_pg
            + self.lambda_risk * avg_risk
            + self.lambda_dir * avg_dir
            + self.lambda_balance * avg_loss_balance # [NEW]
        )

        # Compute direction stats for epoch accumulation
        # Use ALL timesteps predictions and targets (B × T_m) for this batch
        if collected_direction_logits:
            all_dir_logits = th.stack(collected_direction_logits, dim=1)  # (B, T_m, 3)
            all_dir_preds = th.argmax(all_dir_logits, dim=-1)  # (B, T_m)
            batch_dir_preds = all_dir_preds.flatten().cpu().numpy()  # (B * T_m,)
        else:
            batch_dir_preds = np.array([])

        if hasattr(batch, 'direction_labels'):
            batch_dir_targets = batch.direction_labels[:, :T_m].flatten().cpu().numpy()  # (B * T_m,)
        else:
            batch_dir_targets = np.array([])

        # Compute returns, advantages, and penalties for epoch summary
        if collected_raw_returns and collected_baselines and collected_rewards:
            all_return_ratios = th.stack(collected_unscaled_returns, dim=1) # [METRIC FIX] (B, T_m)
            all_returns = th.stack(collected_raw_returns, dim=1)  # (B, T_m)
            all_rewards = th.stack(collected_rewards, dim=1)  # (B, T_m)
            all_baselines = th.stack(collected_baselines, dim=1)  # (B, T_m)
            all_raw_advantages = all_rewards - all_baselines  # (B, T_m)
            all_turnovers = th.stack(collected_turnovers, dim=1)  # (B, T_m)
            all_symdiffs = th.stack(collected_symdiffs, dim=1)  # (B, T_m)
            all_hold_rewards = th.stack(collected_hold_rewards, dim=1) if collected_hold_rewards else None

            # Mean across time for each trajectory
            batch_returns = all_returns.detach().mean(dim=1).cpu().numpy()  # (B,)
            batch_advantages = all_raw_advantages.detach().mean(dim=1).cpu().numpy()  # (B,)
            batch_net_rewards = all_rewards.detach().mean(dim=1).cpu().numpy()  # (B,)
            batch_turn_pens = (all_turnovers.detach() * self.alpha_turnover).mean(dim=1).cpu().numpy()  # (B,)
            batch_symdiff_pens = (all_symdiffs.detach() * self.alpha_change).mean(dim=1).cpu().numpy()  # (B,)
        else:
            batch_returns = np.array([])
            batch_advantages = np.array([])
            batch_net_rewards = np.array([])
            batch_turn_pens = np.array([])
            batch_symdiff_pens = np.array([])
            all_hold_rewards = None
            all_returns = None
            all_turnovers = None
            all_raw_advantages = None

        # Compute Selection Metrics (Sharpe, Turnover, etc.) for this batch
        # We need (B, T_m, K) indices.
        if collected_topk_indices and all_returns is not None:
            # batch_topk_indices = th.stack(collected_topk_indices, dim=1)  # Unused for current metric calc
            
            selection_metrics = self.compute_selection_metrics(
                returns=all_return_ratios.detach(), # [METRIC FIX] Use unscaled return ratios
                turnover=all_turnovers.detach(),
                advantages=all_raw_advantages.detach(),
                hold_rewards=all_hold_rewards.detach() if all_hold_rewards is not None else None
            )
        else:
            selection_metrics = {
                "sharpe_ratio": 0.0,
                "mean_return": 0.0,
                "volatility": 0.0,
                "turnover": 0.0,
                "topk_advantage": 0.0,
            }

        # Compute Direction Metrics (F1, etc.) for this batch
        # We need (B, T_m, 3) logits and (B, T_m) labels
        if collected_direction_logits:
            batch_dir_logits = th.stack(collected_direction_logits, dim=1)  # (B, T_m, 3)
            # Ensure labels are same length (T_m)
            batch_dir_labels = batch.direction_labels[:, :T_m]
            
            direction_metrics = self.compute_direction_metrics(
                direction_logits=batch_dir_logits.detach(),
                direction_labels=batch_dir_labels
            )
        else:
            direction_metrics = {
                "accuracy": 0.0,
                "f1_bear": 0.0,
                "f1_side": 0.0,
                "f1_bull": 0.0,
                "f1_macro": 0.0,
            }

        # Compute Risk Metrics (MSE, MAE) for this batch
        if collected_risk_eta:
             batch_risk_pred = th.stack(collected_risk_eta, dim=1) # (B, T_m) which was actually (B,) in loop list -> (B, T_m)?
             # collected_risk_eta is list of (B,) tensors?
             # Let's check loop: collected_risk_eta.append(risk_eta) where risk_eta is (B,)
             # So stack dim=1 gives (B, T_m)
             # batch.risk_targets is (B, T_m)
             
             batch_risk_pred = th.stack(collected_risk_eta, dim=1)
             batch_risk_target = batch.risk_targets[:, :T_m]
             
             risk_metrics_val = self.compute_risk_metrics(
                 risk_pred=batch_risk_pred.detach(),
                 risk_target=batch_risk_target
             )
        else:
             risk_metrics_val = {
                 "mse": 0.0,
                 "mae": 0.0,
                 "correlation": 0.0,
             }

        # Merge all metrics
        metrics = {
            "loss_total": total_disp,
            "loss_pg": avg_pg,
            "loss_risk": avg_risk,
            "loss_dir": avg_dir,
            "loss_bal": avg_loss_balance, # [NEW]
            "rebalance_ratio": sum(
                [m.float().mean().item() for m in effective_rebalance_masks]
            )
            / T_m,
            "mean_risk_eta": th.stack(collected_risk_eta).mean().item()
            if collected_risk_eta
            else 0.0,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "skipped_update": 1.0 if skipped_update else 0.0,
            # Direction stats for epoch accumulation (keeping raw for fallback)
            "dir_preds": batch_dir_preds,
            "dir_targets": batch_dir_targets,
            "batch_returns": batch_returns,
            "batch_advantages": batch_advantages,
            "batch_net_rewards": batch_net_rewards,
            "batch_turn_pens": batch_turn_pens,
            "batch_symdiff_pens": batch_symdiff_pens,
            
            # Detailed Selection Metrics
            "topk_sharpe_ratio": selection_metrics["sharpe_ratio"],
            "topk_mean_return": selection_metrics["mean_return"],
            "topk_volatility": selection_metrics["volatility"],
            "topk_turnover": selection_metrics["turnover"],
            "topk_advantage": selection_metrics["topk_advantage"],
            "topk_hold_reward": selection_metrics["topk_hold_reward"],
            
            # Detailed Direction Metrics
            "direction_accuracy": direction_metrics["accuracy"],
            "direction_f1_bear": direction_metrics["f1_bear"],
            "direction_f1_side": direction_metrics["f1_side"],
            "direction_f1_bull": direction_metrics["f1_bull"],
            "direction_f1_macro": direction_metrics["f1_macro"],
            
            # Detailed Risk Metrics
            "risk_mse": risk_metrics_val["mse"],
            "risk_mae": risk_metrics_val["mae"],
            "risk_correlation": risk_metrics_val["correlation"],
            
            # Raw data for global epoch aggregation
            "dir_logits": batch_dir_logits.detach().cpu() if collected_direction_logits else None,
            "risk_pred_raw": batch_risk_pred.detach().cpu() if collected_risk_eta else None,
            "risk_pred_raw": batch_risk_pred.detach().cpu() if collected_risk_eta else None,
            "risk_target_raw": batch_risk_target.detach().cpu() if collected_risk_eta else None,
            
            # Learnable Bias Param (Spec 5.3)
            "mafia_holding_bias": self.observer.mafia_model.signal_generator.holding_bias.item() 
            if hasattr(self.observer.mafia_model.signal_generator, "holding_bias") else 0.0
        }

        # Clear memory
        del ochlv_batch, market_ochlv_batch
        th.cuda.empty_cache() if th.cuda.is_available() else None

        return metrics

    def _save_trajectory_log(self):
        """
        Save collected trajectory details to CSV file.
        Called after each training step if log_trajectory_details is enabled.
        """
        if not hasattr(self, "_trajectory_log") or not self._trajectory_log:
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
        mode = "a" if os.path.exists(csv_path) else "w"
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
        log_file: Optional[str] = None,  # NEW: Path to save detailed trajectory log
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            data_tensors: Prepared data tensors from prepare_data_tensors()
            steps_per_epoch: Number of training steps (default: auto-computed)
            log_file: Optional path to append detailed step metrics (CSV)

        Returns:
            ObserverValidationResult: containing all training metrics
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
        
        # Store for step-wise annealing
        self.steps_per_epoch = steps_per_epoch

        self._epoch += 1
        # Update curriculum penalty weight based on current epoch (Spec §7.1)
        self._current_lambda_epoch = self._compute_lambda_epoch(self._epoch)

        # Removed epoch-wise update in favor of step-wise in collect_and_train_step
        # self._current_temp = self.observer.update_temperature(self._epoch)
        # smart_print(f"🔥 Epoch {self._epoch} | Lambda: {self._current_lambda_epoch:.2f}") - We will log temp in step or just rely on existing logs

        epoch_metrics = {
            "loss_total": 0.0,
            "loss_pg": 0.0,
            "loss_risk": 0.0,
            "loss_dir": 0.0,
            "loss_bal": 0.0, # [NEW] Accumulator
            "rebalance_ratio": 0.0,
            "mean_risk_eta": 0.0,
            
            # Selection Metrics
            "topk_sharpe_ratio": 0.0,
            "topk_mean_return": 0.0,
            "topk_volatility": 0.0,
            "topk_turnover": 0.0,
            "topk_advantage": 0.0,
            "topk_hold_reward": 0.0,
            
            # Direction Metrics
            "direction_accuracy": 0.0,
            "direction_f1_bear": 0.0,
            "direction_f1_side": 0.0,
            "direction_f1_bull": 0.0,
            "direction_f1_macro": 0.0,
            
            # Risk Metrics
            "risk_mse": 0.0,
            "risk_mae": 0.0,
            "risk_correlation": 0.0,
            
            # Additional Reward components for manual accumulation if needed
            "reward": 0.0, # Will be sum(net_reward)? or mean? 
            # Note: valid_metrics has 'reward' and 'net_reward'. 
            # In validation result class:
            # reward: float = 0.0
            # net_reward: float = 0.0
            # turnover_penalty: float = 0.0
            # symdiff_penalty: float = 0.0
        }
        
        # Accumulators for reward info to match ObserverValidationResult additional fields
        acc_reward_raw = 0.0
        acc_reward_net = 0.0
        acc_turnover_pen = 0.0
        acc_symdiff_pen = 0.0
        
        # [DEBUG] Check Weights
        dir_head = self.observer.mafia_model.signal_generator.direction_head
        if th.isnan(dir_head.classifier.weight).any() or th.isnan(dir_head.classifier.bias).any():
             smart_print("     🔥 [CRITICAL] DirectionHead weights/bias contain NaN at START of epoch!")
        else:
             smart_print("     ✅ DirectionHead weights legitimate at START.")


        # Tracking for epoch-level direction summary
        epoch_dir_preds = []  # List of prediction arrays
        epoch_dir_targets = []  # List of target arrays
        epoch_returns = []  # List of return values
        epoch_advantages = []  # List of advantage values
        epoch_net_rewards = []  # List of net reward values
        epoch_turn_pens = []  # List of turnover penalty values
        epoch_symdiff_pens = []  # List of symdiff penalty values
        
        # Accumulators for Global Metrics (to match validation logic)
        epoch_dir_logits = []
        epoch_dir_labels = []
        epoch_risk_preds = []
        epoch_risk_targets = []
        
        # NEW: Log details container
        detailed_log_data = []

        # ============================================================
        # EPOCH HEADER
        # ============================================================
        smart_print("\n" + "=" * 100)
        smart_print(f"EPOCH {self._epoch} TRAINING")
        smart_print(f"  Batches per epoch: {steps_per_epoch}")
        smart_print(
            f"  Trajectories per batch: {self.batch_size} (processed in PARALLEL on {self.device})"
        )
        smart_print(f"  Timesteps per trajectory: {self.T_m}")
        smart_print(
            f"  Total training steps this epoch: {steps_per_epoch * self.batch_size * self.T_m:,}"
        )
        smart_print(
            f"  Curriculum λ_epoch: {self._current_lambda_epoch:.2f} (penalty weight)"
        )
        smart_print("=" * 100 + "\n")

        for step in range(steps_per_epoch):
            # ============================================================
            # BATCH HEADER
            # ============================================================
            print(
                f"\n[BATCH {step + 1}/{steps_per_epoch}] 🚀 Processing {self.batch_size} trajectories in PARALLEL...",
                flush=True,
            )

            # Sample batch
            batch = self.sample_trajectory_batch(data_tensors, mode="TRAIN")
            # -------------------------------------------------------------------
            # Collect and train step
            # -------------------------------------------------------------------
            step_metrics = self.collect_and_train_step(
                batch, data_tensors, batch_idx=step, epoch_idx=self._epoch
            )
            print(
                f"[BATCH {step + 1}/{steps_per_epoch}] ✅ Completed - {self.batch_size} trajectories trained in parallel",
                flush=True,
            )
            
            # --- Collect Detailed Log Data ---
            if log_file:
                # We have step_metrics for the BATCH. To be perfectly accurate for visualization,
                # we might want sub-step dynamics (time t=0..T_m). 
                # But `collect_and_train_step` aggregates losses over T_m.
                # However, it runs a loop t=0..T_m internally. 
                # The returned `step_metrics` is averaged/aggregated.
                # If we want T_m level dynamics, we need `collect_and_train_step` to return trace.
                # For now, let's log the batch-level average as a simplified "step".
                # User wants "Training Stability (Gradient Norm)" -> This IS at update step level (Batch).
                # User wants "Loss Dynamics" -> This also varies per batch update.
                # So Batch-level logging is correct for "Trajectory Dynamics" in terms of training steps.
                
                log_row = {
                    "epoch": self._epoch,
                    "batch": step,
                    "global_step": global_step_offset + step,
                    "total_loss": step_metrics["loss_total"],
                    "pg_loss": step_metrics["loss_pg"],
                    "risk_loss": step_metrics["loss_risk"],
                    "dir_loss": step_metrics["loss_dir"],
                    "bal_loss": step_metrics.get("loss_bal", 0.0), # [NEW]
                    "grad_norm": step_metrics["grad_norm"],
                    "skipped_update": step_metrics["skipped_update"],
                    "risk_eta": step_metrics["mean_risk_eta"],
                    # "c_mkt_norm": ??? (Not returned by step_metrics currently, but useful)
                }
                detailed_log_data.append(log_row)

            # Accumulate epoch-level direction stats
            if "dir_preds" in step_metrics and len(step_metrics["dir_preds"]) > 0:
                epoch_dir_preds.append(step_metrics["dir_preds"])
            if "dir_targets" in step_metrics and len(step_metrics["dir_targets"]) > 0:
                epoch_dir_targets.append(step_metrics["dir_targets"])
            if "batch_returns" in step_metrics and len(step_metrics["batch_returns"]) > 0:
                epoch_returns.append(step_metrics["batch_returns"])
            if "batch_advantages" in step_metrics and len(step_metrics["batch_advantages"]) > 0:
                epoch_advantages.append(step_metrics["batch_advantages"])
            if "batch_net_rewards" in step_metrics and len(step_metrics["batch_net_rewards"]) > 0:
                epoch_net_rewards.append(step_metrics["batch_net_rewards"])
            if "batch_turn_pens" in step_metrics and len(step_metrics["batch_turn_pens"]) > 0:
                epoch_turn_pens.append(step_metrics["batch_turn_pens"])
            if "batch_symdiff_pens" in step_metrics and len(step_metrics["batch_symdiff_pens"]) > 0:
                epoch_symdiff_pens.append(step_metrics["batch_symdiff_pens"])
                
            # Collect raw data for global metrics
            if step_metrics.get("dir_logits") is not None:
                epoch_dir_logits.append(step_metrics["dir_logits"])
                epoch_dir_labels.append(step_metrics["dir_targets"]) # Using existing targets (numpy)
            
            if step_metrics.get("risk_pred_raw") is not None:
                epoch_risk_preds.append(step_metrics["risk_pred_raw"])
                epoch_risk_targets.append(step_metrics["risk_target_raw"])

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
                writer.add_scalar(
                    "Train/Loss/Total", step_metrics["loss_total"], global_step
                )
                writer.add_scalar("Train/Loss/PG", step_metrics["loss_pg"], global_step)
                writer.add_scalar(
                    "Train/Loss/Risk", step_metrics["loss_risk"], global_step
                )
                writer.add_scalar(
                    "Train/Loss/Direction", step_metrics["loss_dir"], global_step
                )
                writer.add_scalar(
                    "Train/Risk_Eta/Mean", step_metrics["mean_risk_eta"], global_step
                )
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
        epoch_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        
        # Accumulate reward stuff manually from step lists if available, or just take from step_metrics average if we added them there?
        # We didn't add "reward", "net_reward" etc to step_metrics explicitly above, but we have "batch_net_rewards" list.
        # Let's use the epoch-level lists we collected to compute these accurately for the Result object.
        
        # We have epoch_returns, epoch_net_rewards, etc. computed below for printing.
        # Let's reuse them or just use the averaged values in epoch_metrics if we added them? 
        # Actually I added "topk_mean_return" etc to epoch_metrics, which are averaged from batch metrics.
        
        # For 'reward', 'net_reward' scalar fields in ObserverValidationResult:
        # These are usually epoch-averaged values.
        
        if epoch_returns:
            all_rets = np.concatenate(epoch_returns)
            all_nets = np.concatenate(epoch_net_rewards)
            all_turn = np.concatenate(epoch_turn_pens)
            all_sym = np.concatenate(epoch_symdiff_pens)
            # ObserverValidationResult expects floats
            final_reward_raw = all_rets.mean()
            final_reward_net = all_nets.mean()
            final_turn_pen = all_turn.mean()
            final_sym_pen = all_sym.mean()
        else:
            final_reward_raw = 0.0
            final_reward_net = 0.0
            final_turn_pen = 0.0
            final_sym_pen = 0.0
            
        # ============================================================
        # GLOBAL METRIC COMPUTATION (Synchronized with Validation)
        # ============================================================
        
        # 1. Global Direction Metrics
        if epoch_dir_logits and epoch_dir_labels:
            # Concatenate
            all_logits = th.cat([t if isinstance(t, th.Tensor) else th.from_numpy(t) for t in epoch_dir_logits], dim=0)
            all_labels = th.cat([t if isinstance(t, th.Tensor) else th.from_numpy(t) for t in epoch_dir_labels], dim=0)
            
            # Compute global metrics using same function as validation
            global_dir_metrics = self.compute_direction_metrics(all_logits, all_labels)
            
            # Overwrite averaged metrics with global ones
            epoch_metrics["direction_accuracy"] = global_dir_metrics["accuracy"]
            epoch_metrics["direction_f1_bear"] = global_dir_metrics["f1_bear"]
            epoch_metrics["direction_f1_side"] = global_dir_metrics["f1_side"]
            epoch_metrics["direction_f1_bull"] = global_dir_metrics["f1_bull"]
            epoch_metrics["direction_f1_macro"] = global_dir_metrics["f1_macro"]
            
        # 2. Global Risk Metrics
        if epoch_risk_preds and epoch_risk_targets:
            # Concatenate
            all_risk_p = th.cat([t if isinstance(t, th.Tensor) else th.from_numpy(t) for t in epoch_risk_preds], dim=0)
            all_risk_t = th.cat([t if isinstance(t, th.Tensor) else th.from_numpy(t) for t in epoch_risk_targets], dim=0)
            
            # Compute global metrics
            global_risk_metrics = self.compute_risk_metrics(all_risk_p, all_risk_t)
            
            # Overwrite averaged metrics
            epoch_metrics["risk_mse"] = global_risk_metrics["mse"]
            epoch_metrics["risk_mae"] = global_risk_metrics["mae"]
            epoch_metrics["risk_correlation"] = global_risk_metrics["correlation"]

        # TensorBoard logging (Epoch-level)
        if writer:
            writer.add_scalar(
                "Train/Epoch/Loss_Total", epoch_metrics["loss_total"], self._epoch
            )
            writer.add_scalar(
                "Train/Epoch/Loss_PG", epoch_metrics["loss_pg"], self._epoch
            )
            writer.add_scalar(
                "Train/Epoch/Loss_Risk", epoch_metrics["loss_risk"], self._epoch
            )
            writer.add_scalar(
                "Train/Epoch/Loss_Dir", epoch_metrics["loss_dir"], self._epoch
            )
            writer.add_scalar("Train/Epoch/LR", epoch_metrics["lr"], self._epoch)
            # Add new metrics
            writer.add_scalar("Train/Epoch/Sharpe", epoch_metrics["topk_sharpe_ratio"], self._epoch)
            writer.add_scalar("Train/Epoch/Dir_F1", epoch_metrics["direction_f1_macro"], self._epoch)
            writer.add_scalar("Train/Epoch/Risk_MSE", epoch_metrics["risk_mse"], self._epoch)

        # ============================================================
        # EPOCH SUMMARY
        # ============================================================
        smart_print("\n" + "=" * 100)
        smart_print(f"📊 EPOCH {self._epoch} SUMMARY")
        smart_print("-" * 100)

        # Compute epoch-level direction stats
        total_trajectories = 0
        if epoch_dir_preds and epoch_dir_targets:
            all_preds = np.concatenate(epoch_dir_preds)
            all_targets = np.concatenate(epoch_dir_targets)
            total_trajectories = len(all_preds)

            # Direction accuracy
            epoch_dir_accuracy = (all_preds == all_targets).mean() * 100

            # Prediction distribution
            epoch_pred_bear = (all_preds == 0).mean() * 100
            epoch_pred_side = (all_preds == 1).mean() * 100
            epoch_pred_bull = (all_preds == 2).mean() * 100

            # Target distribution
            epoch_target_bear = (all_targets == 0).mean() * 100
            epoch_target_side = (all_targets == 1).mean() * 100
            epoch_target_bull = (all_targets == 2).mean() * 100
        else:
            epoch_dir_accuracy = 0.0
            epoch_pred_bear = epoch_pred_side = epoch_pred_bull = 0.0
            epoch_target_bear = epoch_target_side = epoch_target_bull = 0.0

        # Compute epoch-level returns and advantages
        if epoch_returns and epoch_advantages:
            all_returns = np.concatenate(epoch_returns)
            all_advantages = np.concatenate(epoch_advantages)
            epoch_mean_return = all_returns.mean()
            epoch_std_return = all_returns.std()
            epoch_mean_advantage = all_advantages.mean()
        else:
            epoch_mean_return = 0.0
            epoch_std_return = 0.0
            epoch_mean_advantage = 0.0

        # Compute epoch-level net rewards and penalties
        if epoch_net_rewards and epoch_turn_pens and epoch_symdiff_pens:
            all_net_rewards = np.concatenate(epoch_net_rewards)
            all_turn_pens = np.concatenate(epoch_turn_pens)
            all_symdiff_pens = np.concatenate(epoch_symdiff_pens)
            epoch_mean_net_reward = all_net_rewards.mean()
            epoch_mean_turn_pen = all_turn_pens.mean()
            epoch_mean_symdiff_pen = all_symdiff_pens.mean()
        else:
            epoch_mean_net_reward = 0.0
            epoch_mean_turn_pen = 0.0
            epoch_mean_symdiff_pen = 0.0

        # Advantage status
        adv_status = "🟢 Beating Market" if epoch_mean_advantage > 0 else "🔴 Underperforming"

        # Display portfolio performance
        smart_print(f"  📈 Returns:     Mean={epoch_mean_return:+.4f} | Std={epoch_std_return:.4f}")
        smart_print(f"  💰 Net Reward:  Mean={epoch_mean_net_reward:+.4f} | TurnPen={epoch_mean_turn_pen:.4f} | SymDiffPen={epoch_mean_symdiff_pen:.4f}")
        smart_print(f"  ⚖️  Advantage:   Mean={epoch_mean_advantage:+.4f} ({adv_status})")
        smart_print(f"  🎯 Direction:   Accuracy={epoch_dir_accuracy:.1f}% (across {total_trajectories} predictions)")
        smart_print(f"     Preds:    Bear={epoch_pred_bear:4.1f}% | Side={epoch_pred_side:4.1f}% | Bull={epoch_pred_bull:4.1f}%")
        smart_print(f"     Target:   Bear={epoch_target_bear:4.1f}% | Side={epoch_target_side:4.1f}% | Bull={epoch_target_bull:4.1f}%")
        smart_print(f"  📉 Risk η:      Mean={epoch_metrics['mean_risk_eta']:.3f}")
        smart_print("-" * 100)
        smart_print("  📉 Losses:")
        smart_print(f"    ├─ Total:     {epoch_metrics['loss_total']:.6f}")
        smart_print(f"    ├─ PG:        {epoch_metrics['loss_pg']:.6f}")
        smart_print(f"    ├─ Risk:      {epoch_metrics['loss_risk']:.6f}")
        smart_print(f"    ├─ Direction: {epoch_metrics['loss_dir']:.6f}")
        smart_print(f"    └─ Balance:   {epoch_metrics['loss_bal']:.6f}")
        smart_print("-" * 100)
        smart_print(
            f"  Rebalance Ratio:  {epoch_metrics['rebalance_ratio']:.2%} (of all timesteps)"
        )
        smart_print(f"  Learning Rate:    {epoch_metrics['lr']:.6f}")
        smart_print("=" * 100 + "\n")
        
        # --- Save Detailed Log Data to CSV ---
        if log_file and detailed_log_data:
            try:
                import os
                import pandas as pd
                df_log = pd.DataFrame(detailed_log_data)
                header = not os.path.exists(log_file)
                df_log.to_csv(log_file, mode='a', header=header, index=False)
                # print(f"📝 Appended {len(df_log)} rows to detailed log: {log_file}")
            except Exception as e:
                smart_print(f"[WARN] Failed to write detailed trajectory log: {e}")

        # Convert to ObserverValidationResult
        result = ObserverValidationResult(
            epoch=self._epoch,
            # Loss
            loss_total=epoch_metrics["loss_total"],
            loss_pg=epoch_metrics["loss_pg"],
            loss_risk=epoch_metrics["loss_risk"],
            loss_dir=epoch_metrics["loss_dir"],
            loss_bal=epoch_metrics.get("loss_bal", 0.0),  # [FIX] Add missing loss_bal

            # Selection
            topk_sharpe_ratio=epoch_metrics["topk_sharpe_ratio"],
            topk_mean_return=epoch_metrics["topk_mean_return"],
            topk_volatility=epoch_metrics["topk_volatility"],
            topk_turnover=epoch_metrics["topk_turnover"],
            
            # Direction
            direction_accuracy=epoch_dir_accuracy, # Use epoch aggregated accuracy (0-100)
            direction_f1_bear=epoch_metrics.get("direction_f1_bear", 0.0),
            direction_f1_side=epoch_metrics.get("direction_f1_side", 0.0),
            direction_f1_bull=epoch_metrics.get("direction_f1_bull", 0.0),
            direction_f1_macro=epoch_metrics.get("direction_f1_macro", 0.0),
            
            # Risk
            risk_mse=epoch_metrics.get("risk_mse", 0.0),
            risk_mae=epoch_metrics.get("risk_mae", 0.0),
            risk_correlation=epoch_metrics.get("risk_correlation", 0.0),
            
            # Additional Metrics
            topk_advantage=epoch_metrics.get("topk_advantage", 0.0),
            topk_hold_reward=epoch_metrics.get("topk_hold_reward", 0.0),
            

            
            # Raw Components
            # Approximate gross reward for Logging
            reward=epoch_mean_net_reward + epoch_mean_turn_pen + epoch_mean_symdiff_pen,
            net_reward=epoch_mean_net_reward,
            turnover_penalty=epoch_mean_turn_pen,
            symdiff_penalty=epoch_mean_symdiff_pen,
            
            # CES placeholders
            ces_score=0.0
        )

        return result

    @th.no_grad()
    def compute_selection_metrics(
        self,
        returns: th.Tensor,
        turnover: th.Tensor,
        advantages: th.Tensor = None,
        hold_rewards: th.Tensor = None,
    ) -> Dict[str, float]:
        """
        Compute financial metrics for the selected portfolios.

        Args:
            returns: (B, T) tensor of portfolio returns
            turnover: (B, T) tensor of turnover
            advantages: (B, T) tensor of advantage values (optional)
            hold_rewards: (B, T) tensor of hold duration/profit bonuses (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}
        # Annualization factor: 252 trading days per year
        # Fix: returns are NOT daily, they are cumulative over `pg_reward_horizon` (e.g. 14 days)
        # So we must scale by sqrt(252 / horizon) instead of sqrt(252)
        horizon = getattr(self, "pg_reward_horizon", 14)  # Default to 14 if not found
        annual_factor = 252.0 / max(horizon, 1)

        mean_return = returns.mean().item()
        std_return = returns.std().item()

        # Get Risk-Free Rate (Annual %)
        # Default to 4.2% (Vietnam 10Y Bond Yield) if not specified
        rf_annual_percent = self.config.mkt_rf.get(self.config.market_name, 4.2)
        rf_annual = rf_annual_percent / 100.0
        
        # Convert to period (horizon) Rf using Geometric Formula
        # R_daily = (1 + R_annual)^(1/252) - 1
        # R_period = (1 + R_annual)^(horizon/252) - 1
        horizon = max(getattr(self, "pg_reward_horizon", 14), 1)
        rf_period = (1 + rf_annual) ** (horizon / 252.0) - 1
        
        # Excess Return for Sharpe Calculation
        # [METRIC FIX] returns are Raw Ratios (e.g. 0.015), not Percent
        # So rf_period must also remain Raw Ratio.
        rf_period_percent = rf_period  # No x100 scaling
        
        excess_mean_return = mean_return - rf_period_percent
        
        # Sharpe Ratio (Ann.) = (Mean - Rf) / Std * sqrt(252/h)
        if std_return > 1e-6:
            sharpe = (excess_mean_return / std_return) * (annual_factor**0.5)
        else:
            sharpe = 0.0

        # Review confirmed: Turnover, Advantage, HoldReward, Direction, Risk are OK.
        # Fix: Annualize Return and Volatility for consistency with Sharpe Ratio
        # Output as Percent for readability (0.27 -> 27.0)
        metrics["mean_return"] = mean_return * annual_factor * 100.0
        metrics["volatility"] = std_return * (annual_factor**0.5) * 100.0
        metrics["sharpe_ratio"] = sharpe
        metrics["turnover"] = turnover.mean().item()
        metrics["topk_advantage"] = advantages.mean().item() if advantages is not None else 0.0
        metrics["topk_hold_reward"] = hold_rewards.mean().item() if hold_rewards is not None else 0.0

        return metrics

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
        data_tensors: Dict[str, th.Tensor], # Changed from dataloader to data_tensors to match usage
        steps: Optional[int] = None,
        compute_loss: bool = False,
    ) -> ObserverValidationResult:
        """
        Validate one epoch.
        Args:
            data_tensors: Prepared validation data tensors
        Returns:
            ObserverValidationResult with all validation metrics
        """
        print(f"[DEBUG] validate_epoch called with compute_loss={compute_loss}", flush=True)
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
        total_loss_balance = 0.0 # [NEW] Accumulator
        
        # Accumulators for rewards & penalties
        total_reward = 0.0
        total_net_reward = 0.0
        total_turnover_penalty = 0.0
        total_symdiff_penalty = 0.0
        total_hold_reward = 0.0  # [NEW] Accumulator

        # Accumulators for metric computation
        # Tensors for metric computation (for compute_selection_metrics)
        all_return_tensors = []
        all_turnover_tensors = []
        all_advantage_tensors = []

        all_topk_indices = []
        all_topk_scores = []
        all_price_returns = []
        all_market_returns = [] # NEW
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
                if hasattr(self.observer.mafia_model, "D"):
                    mafia_D = self.observer.mafia_model.D
                else:
                    mafia_D = int(getattr(self.config, "mafia_D", 128))
                batch_context_buffer = th.zeros(B, W_route, mafia_D, device=self.device)

                full_ochlv = data_tensors["ochlv"]
                full_market_ochlv = data_tensors.get("market_ochlv")
                
                # Direction reversal tracking
                N_confirm = int(getattr(self.config, "regime_confirmation_window", 3))
                direction_history = th.ones(
                    B, N_confirm + 1, dtype=th.long, device=self.device
                )

                days_since_last_rebal = th.zeros(B, dtype=th.long, device=self.device)
                rebal_interval = self.rebalance_interval

                # Compute volatility threshold for validation mode
                vol_mean_val = batch.vol_std20.mean()
                vol_std_val = batch.vol_std20.std()
                vol_shock_threshold_val = vol_mean_val + self.vol_k * vol_std_val

                # Static trigger count for normalization
                static_trigger_mask = (
                    (batch.rebalance_mask > 0.5)
                    | (batch.vol_std20 > vol_shock_threshold_val)
                    | (batch.dc_event_flag > 0.5)  # Binary flag
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
                balance_losses = [] # [NEW] Track balance loss per step
                
                # Reward collections
                collected_raw_returns = []
                collected_baselines = []
                collected_unscaled_returns = []
                
                curr_batch_rewards = []
                curr_batch_net = []
                curr_batch_turnover = []
                curr_batch_symdiff = []
                curr_batch_hold_rewards = []  # [NEW]

                prev_indices_tensor = None
                prev_scores_tensor = None

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
                        market_ochlv_batch = th.stack(batch_windows_mkt, dim=0).to(
                            self.device
                        )

                    # Rebalance schedule trigger
                    if t == 0:
                        sched_trigger = th.ones(B, dtype=th.bool, device=self.device)
                        days_since_last_rebal[:] = 0
                    else:
                        days_since_last_rebal += 1
                        sched_trigger = days_since_last_rebal >= rebal_interval

                    vol_trigger = batch.vol_std20[:, t] > vol_shock_threshold_val
                    dc_trigger = batch.dc_event_flag[:, t] > 0.5  # Binary flag
                    pre_trigger = vol_trigger | sched_trigger | dc_trigger

                    # Build explicit signals for Direction Head (Spec 3.5.1 v2.1 - Wide Path)
                    # 6 signals: Vol_Std20, DC_Event_Flag, Breadth_Gap, Div_Signal, Signed_VPI_Zscore, Drawdown60
                    vol_std20_value = batch.vol_std20[:, t : t + 1] # (B, 1)
                    dc_flag_value = batch.dc_event_flag[:, t : t + 1]  # (B, 1) - already binary [0, 1]
                    breadth_gap_value = batch.breadth_gap[:, t : t + 1]  # (B, 1) - already normalized [-1, 1]
                    div_signal_value = batch.div_signal[:, t : t + 1]  # (B, 1) - already {-1, 0, 1}
                    vpi_zscore_value = batch.signed_vpi_zscore[:, t : t + 1] # (B, 1)
                    drawdown60_value = batch.drawdown60[:, t : t + 1] # (B, 1)
                    # Signals are already normalized during computation, concat directly
                    explicit_signals = th.cat(
                        [
                            vol_std20_value,
                            dc_flag_value,
                            breadth_gap_value,
                            div_signal_value,
                            vpi_zscore_value,
                            drawdown60_value,
                        ], dim=-1
                    )  # (B, 6)
                    
                    # [NAN-FIX] Sanitize Explicit Signals
                    # Force sanitization to prevent Inf/NaN from crashing DirectionHead/RiskHead
                    # (Conditional check proved unreliable on MPS)
                    explicit_signals = th.nan_to_num(explicit_signals, nan=0.0, posinf=5.0, neginf=-5.0)
                    explicit_signals = th.clamp(explicit_signals, min=-10.0, max=10.0)

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
                        market_logits,
                    ) = outputs
                    
                    # [DEBUG] Check Backbone Output
                    if th.isnan(market_context).any() or th.isinf(market_context).any():
                         smart_print(f"     [DEBUG] NaN/Inf in market_context (Backbone) at t={t}")
                    if th.isnan(direction_logits).any() or th.isinf(direction_logits).any():
                         smart_print(f"     [DEBUG] NaN/Inf in direction_logits (Head) at t={t}")

                    # Temperature scaling for direction head (sharpen/flatten)
                    if self.direction_temperature != 1.0:
                        direction_logits = direction_logits / self.direction_temperature

                    fresh_topk_indices = topk_indices.clone()
                    fresh_topk_scores = topk_scores.clone()

                    # [NEW] Load Balancing Loss (Validation)
                    loss_balance_step = 0.0
                    if self._latest_gate_weights is not None:
                        # _latest_gate_weights is (B, num_experts) - detached in hook
                        avg_gate = self._latest_gate_weights.mean(dim=0)
                        num_experts = avg_gate.size(0)
                        target_uniform = 1.0 / num_experts
                        loss_balance_step = (avg_gate - target_uniform).pow(2).sum().item() * 100.0
                    balance_losses.append(loss_balance_step)

                    # Detach temporal state and buffer
                    if hasattr(self.observer.mafia_model, "detach_temporal_state"):
                        self.observer.mafia_model.detach_temporal_state()
                    batch_context_buffer = batch_context_buffer.detach()

                    # Direction reversal trigger
                    pred_state = th.argmax(direction_logits, dim=-1)
                    pred_state = th.clamp(pred_state, 0, 2)
                    direction_history = th.cat(
                        [direction_history[:, 1:], pred_state.unsqueeze(1)], dim=1
                    )

                    if t < N_confirm:
                        dir_trigger = th.zeros(B, dtype=th.bool, device=self.device)
                    else:
                        last_seq = direction_history[:, 1:]
                        prev_regime = direction_history[:, 0]
                        BULL, BEAR = 2, 0
                        all_bull = (last_seq == BULL).all(dim=1)
                        was_bear = prev_regime == BEAR
                        bear_to_bull = all_bull & was_bear
                        all_bear = (last_seq == BEAR).all(dim=1)
                        was_bull = prev_regime == BULL
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
                        R_unscaled = th.zeros(B, device=self.device) # [METRIC FIX]
                    else:
                        gather_idx = final_indices.unsqueeze(1).expand(-1, h_len, -1)
                        selected_returns = th.gather(
                            future_returns_slice, 2, gather_idx
                        )
                        compounded_stock_returns = (
                            th.prod(1 + selected_returns, dim=1) - 1
                        )
                        R_raw = (
                            self.scale_factor_reward
                            * compounded_stock_returns.mean(dim=1)
                        )
                        baseline = self.scale_factor_reward * (
                            th.prod(1 + market_returns_slice, dim=1) - 1
                        )
                        # [METRIC FIX] Store UN-SCALED return for metrics
                        R_unscaled = compounded_stock_returns.mean(dim=1)

                    collected_raw_returns.append(R_raw)
                    collected_baselines.append(baseline)
                    collected_unscaled_returns.append(R_unscaled) # [METRIC FIX]

                    turnover = th.zeros(B, device=self.device)
                    symdiff = th.zeros(B, device=self.device)
                    if prev_indices_tensor is not None:
                        for b_i in range(B):
                            curr_set = set(final_indices[b_i].cpu().tolist())
                            prev_set = set(prev_indices_tensor[b_i].cpu().tolist())
                            held_count = len(curr_set.intersection(prev_set))
                            s_val = (2 * (self.K - held_count)) / self.K
                            symdiff[b_i] = s_val
                            
                        # Turnover (Weight Based)
                        N_stocks = batch.stock_ochlv.shape[2]
                        
                        curr_w_vector = th.zeros(B, N_stocks, device=self.device)
                        curr_local_w = F.softmax(final_scores, dim=1)
                        curr_w_vector.scatter_(1, final_indices, curr_local_w)
                        
                        prev_w_vector = th.zeros(B, N_stocks, device=self.device)
                        if prev_scores_tensor is not None:
                             prev_local_w = F.softmax(prev_scores_tensor, dim=1)
                             prev_w_vector.scatter_(1, prev_indices_tensor, prev_local_w)
                        
                        
                        turnover = 0.5 * (curr_w_vector - prev_w_vector).abs().sum(dim=1)
                        turnover = th.clamp(turnover, 0.0, 1.0) # Ensure range [0, 1]

                    # VALIDATION MODE: Always use full penalty (no curriculum learning)
                    lambda_epoch = 1.0
                    penalty_sum = (
                        self.alpha_turnover * turnover + self.alpha_change * symdiff
                    )
                    penalty = lambda_epoch * penalty_sum
                    
                    # [NEW] R_hold Calculation (Validation)
                    # Consistency Score = (Count of Days where R_stock > R_mkt AND R_stock > 0) / h_len
                    if h_len > 0:
                        # Expand market returns for broadcast: (B, h) -> (B, h, 1)
                        mkt_ret_expanded = market_returns_slice.unsqueeze(-1)
                        # Check condition: Stock > Market AND Stock > 0
                        win_day_mask = (future_returns_slice > mkt_ret_expanded) & (future_returns_slice > 0)
                        # Score per Stock (B, N)
                        stock_consistency = win_day_mask.float().sum(dim=1) / float(h_len)
                        
                        # Current Holdings Mask (B, N)
                        curr_holdings = th.zeros(B, self.observer.action_dim, device=self.device)
                        curr_holdings.scatter_(1, final_indices, 1.0)
                        
                        # Previous Holdings Mask (for continuity check - optional, but logical given intent)
                        # However, for R_hold, do we require holding from t-1? The spec says "Held Stocks".
                        # In training, we use `prev_holdings * curr_holdings`.
                        # Here, `prev_indices_tensor` is available.
                        if prev_indices_tensor is not None:
                            prev_h = th.zeros(B, self.observer.action_dim, device=self.device)
                            prev_h.scatter_(1, prev_indices_tensor, 1.0)
                            held_mask = prev_h * curr_holdings
                        else:
                            held_mask = curr_holdings # First step, treat current as held? Or 0? Usually 0 if no prev.
                        
                        # Sum Scores
                        portfolio_win_score = (stock_consistency * held_mask).sum(dim=1)
                        
                        # Scale
                        r_hold = self.r_hold_alpha * (portfolio_win_score / self.K)
                    else:
                        r_hold = th.zeros(B, device=self.device)

                    R_net = R_raw + r_hold - penalty
                    A_t_raw = R_net - baseline
                    A_mean = A_t_raw.mean()
                    A_std = A_t_raw.std() + eps if B > 1 else 1.0
                    A_t = (A_t_raw - A_mean) / A_std

                    if compute_loss:
                        # [Restored] Loss calculation for training metrics or if requested
                        log_probs = th.log(
                            th.gather(market_scores_full, 1, final_indices) + eps
                        )
                        pg_term = -(log_probs * A_t.detach().unsqueeze(1)).mean(dim=1)
                        entropy = -(
                            market_scores_full * th.log(market_scores_full + eps)
                        ).sum(dim=1)
                        step_loss_val = pg_term - self.beta_entropy * entropy

                        risk_loss = (
                            self.risk_criterion(risk_eta, batch.risk_targets[:, t])
                            * self.risk_scaling_factor
                        )
                        dir_loss = self.dir_criterion(
                            direction_logits, batch.direction_labels[:, t]
                        )
                    else:
                        step_loss_val = th.tensor(0.0, device=self.device)
                        risk_loss = th.tensor(0.0, device=self.device)
                        dir_loss = th.tensor(0.0, device=self.device)

                    step_pg_loss = (step_loss_val * effective_mask_float).sum()
                    pg_losses.append(step_pg_loss.item())
                    risk_losses.append(risk_loss.item())
                    dir_losses.append(dir_loss.item())
                    
                    if compute_loss:
                         print(f"[DEBUG-LOSS] t={t} | PG={step_pg_loss.item():.4f} | Risk={risk_loss.item():.4f} | Dir={dir_loss.item():.4f}", flush=True)
                    
                    # Store rewards ONLY for rebalance events (matching effective_mask)
                    # Or do we want average per step? Spec usually implies rebalance events?
                    # But for overall validation metrics, we can just sum up everything or mask it.
                    # Since losses are sparse, let's follow the mask logic for accurate counts.
                    # R_raw/penalty are defined at every step but only meaningful at rebalance.
                    # But if we hold, turnover=0, R_raw is still computed (future return).
                    # Actually spec says PG considers rewards from action a_t.
                    # If we held, action was 'hold'.
                    # Let's collect ALL for simple averaging, or masked?
                    # Usually metrics like "Turnover" should be avg per step or avg per rebalance?
                    # Standard practice: Average over all steps to see load, or average per event.
                    # Let's collect raw values and normalized later if needed.
                    # Actually, let's use the effective mask to only count "decisions".
                    # But "Reward" exists even if we hold.
                    # Let's average over the whole trajectory for now, consistent with losses.
                    # NOTE: R_net is used for Advantage.
                    
                    curr_batch_rewards.append(R_raw.mean().item()) # Mean across batch
                    curr_batch_net.append(R_net.mean().item())
                    curr_batch_hold_rewards.append(r_hold.mean().item()) # [NEW]
                    
                    # For penalties, meaningful only if rebalance occurred? 
                    # If we hold, turnover=0. So summing 0s works fine for average.
                    curr_batch_turnover.append((turnover * lambda_epoch * self.alpha_turnover).mean().item())
                    curr_batch_symdiff.append((symdiff * lambda_epoch * self.alpha_change).mean().item())

                    # Collect tensors for compute_selection_metrics
                    # [METRIC FIX] Use UN-SCALED returns for financial metrics
                    # R_raw is scaled by scale_factor_reward (100.0) for RL stability
                    all_return_tensors.append(compounded_stock_returns.mean(dim=1)) 
                    all_turnover_tensors.append(turnover)
                    all_advantage_tensors.append(A_t_raw)

                    # Collect outputs (sanitize to prevent NaN propagation)
                    final_scores = th.nan_to_num(final_scores, nan=0.0, posinf=10.0, neginf=-10.0)
                    risk_eta = th.nan_to_num(risk_eta, nan=1.0, posinf=1.0, neginf=1.0)
                    collected_topk_scores.append(final_scores)
                    collected_topk_indices.append(final_indices)
                    collected_risk_eta.append(risk_eta)
                    collected_direction_logits.append(direction_logits)

                    # Update context buffer FIFO
                    history_part = batch_context_buffer[:, 1:, :]
                    new_part = market_context.detach().unsqueeze(1)
                    batch_context_buffer = th.cat([history_part, new_part], dim=1)

                    prev_indices_tensor = final_indices
                    prev_scores_tensor = final_scores

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
            avg_risk = sum(risk_losses) / len(risk_losses) if risk_losses else 0.0
            avg_dir = sum(dir_losses) / len(dir_losses) if dir_losses else 0.0
            avg_loss_balance = sum(balance_losses) / len(balance_losses) if balance_losses else 0.0 # [NEW]
            total_disp = (
                self.lambda_pg * avg_pg
                + self.lambda_risk * avg_risk
                + self.lambda_dir * avg_dir
                + self.lambda_balance * avg_loss_balance # [NEW] Include in reported total
            )

            total_loss_total += total_disp
            total_loss_pg += avg_pg
            total_loss_risk += avg_risk
            total_loss_dir += avg_dir
            total_loss_balance += avg_loss_balance # [FIX] Accumulate balance loss
            
            print(f"[DEBUG-ACCUM] Step | Total={total_loss_total:.4f} | Bal={total_loss_balance:.4f} (InnerAvg={avg_loss_balance:.4f}) | Dir={total_loss_dir:.4f}", flush=True)
            
            # Note: total_loss_total will be updated with L_bal in the subsequent block where L_bal is computed
            # actually total_disp calculated above now includes it.
            
            # Rewards and Hold Bonus
            
            # Rewards and Hold Bonus
            avg_rew = sum(curr_batch_rewards) / len(curr_batch_rewards) if curr_batch_rewards else 0.0
            avg_net = sum(curr_batch_net) / len(curr_batch_net) if curr_batch_net else 0.0
            avg_turn = sum(curr_batch_turnover) / len(curr_batch_turnover) if curr_batch_turnover else 0.0
            avg_sym = sum(curr_batch_symdiff) / len(curr_batch_symdiff) if curr_batch_symdiff else 0.0
            
            # [NEW] Average Hold Reward
            # Note: collected_hold_rewards might be empty if horizon=0 or other edge cases
            # We need to collect it in the loop first!
            avg_hold_reward = sum(curr_batch_hold_rewards) / len(curr_batch_hold_rewards) if curr_batch_hold_rewards else 0.0
            
            total_reward += avg_rew
            total_net_reward += avg_net
            total_turnover_penalty += avg_turn
            total_symdiff_penalty += avg_sym
            total_hold_reward += avg_hold_reward # Need this accumulator initialized

            # Store for metric computation
            all_topk_indices.append(topk_indices_stack)
            all_topk_scores.append(topk_scores_stack)
            all_price_returns.append(batch.price_returns[:, :T_m, :])
            all_market_returns.append(batch.market_returns[:, :T_m]) # NEW
            all_risk_pred.append(risk_eta_stack)
            all_risk_target.append(batch.risk_targets)
            all_direction_logits.append(direction_logits_stack)
            all_direction_labels.append(batch.direction_labels)

        # Average losses
        avg_loss_total = total_loss_total / steps
        avg_loss_pg = total_loss_pg / steps
        avg_loss_risk = total_loss_risk / steps
        avg_loss_dir = total_loss_dir / steps
        avg_loss_balance = total_loss_balance / steps
        
        # Average rewards
        avg_reward = total_reward / steps
        avg_net_reward = total_net_reward / steps
        avg_turnover_penalty = total_turnover_penalty / steps
        avg_symdiff_penalty = total_symdiff_penalty / steps
        avg_hold_reward = total_hold_reward / steps # [NEW]

        # Concatenate all batches for metric computation
        all_topk_indices_cat = th.cat(all_topk_indices, dim=0)
        all_topk_scores_cat = th.cat(all_topk_scores, dim=0)
        all_price_returns_cat = th.cat(all_price_returns, dim=0)
        all_market_returns_cat = th.cat(all_market_returns, dim=0) # NEW
        all_risk_pred_cat = th.cat(all_risk_pred, dim=0)
        all_risk_target_cat = th.cat(all_risk_target, dim=0)
        all_direction_logits_cat = th.cat(all_direction_logits, dim=0)
        all_direction_labels_cat = th.cat(all_direction_labels, dim=0)

        all_direction_labels_cat = th.cat(all_direction_labels, dim=0)

        # Consolidate standard metric tensors
        all_returns_cat = th.cat(all_return_tensors, dim=0) if all_return_tensors else th.tensor([], device=self.device)
        all_turnovers_cat = th.cat(all_turnover_tensors, dim=0) if all_turnover_tensors else th.tensor([], device=self.device)
        all_advantages_cat = th.cat(all_advantage_tensors, dim=0) if all_advantage_tensors else th.tensor([], device=self.device)

        # Compute metrics
        selection_metrics = self.compute_selection_metrics(
            returns=all_returns_cat,
            turnover=all_turnovers_cat,
            advantages=all_advantages_cat,
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
            loss_bal=avg_loss_balance, # [NEW]
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
            # NEW fields
            topk_advantage=selection_metrics.get("topk_advantage", 0.0),
            topk_hold_reward=avg_hold_reward, # [NEW]
            reward=avg_reward,
            net_reward=avg_net_reward,
            turnover_penalty=avg_turnover_penalty,
            symdiff_penalty=avg_symdiff_penalty,
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
