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
from config import MafiaTrainMode

# LiveDisplay smart_print for terminal-safe logging
# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import smart_print, get_display
except ImportError:
    smart_print = print  # Fallback
    get_display = lambda: None


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

    def forward(self, input, target, alpha_override: th.Tensor = None):
        """
        Args:
            input: Logits (N, C) or (N, C, ...)
            target: Ground truth labels (N) or (N, ...)
            alpha_override: Optional per-call class weights (C,). If provided,
                           overrides self.alpha for this call only. Used for
                           dynamic per-batch class weighting.
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
        log_pt_target = log_pt_target.view(-1)  # Flatten
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
            input,
            target.view_as(target_flat) if target.dim() != input.dim() - 1 else target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # 2. Extract pt safely
        # pt = exp(-ce)
        pt = th.exp(-ce_loss)

        # 3. CRITICAL STABILITY: Stop gradients on `pt` for the focal term?
        # Usually we want gradients through focal term.
        # The issue is `pow` derivative at base=0 (if gamma < 1) or huge base.
        # Clamp pt to [0.0001, 0.9999] just for the focal term calculation base.
        pt_safe = pt.clamp(min=1e-6, max=1.0 - 1e-6)

        # 4. Focal term
        focal_term = (1.0 - pt_safe).pow(self.gamma)

        # 5. Combine
        focal_loss = focal_term * ce_loss

        # 6. Alpha (with dynamic override support for per-batch weighting)
        # Use alpha_override if provided, otherwise fall back to self.alpha
        effective_alpha = alpha_override if alpha_override is not None else self.alpha

        if effective_alpha is not None:
            # Ensure alpha is on correct device
            if effective_alpha.device != input.device:
                effective_alpha = effective_alpha.to(input.device)

            if (
                target.dim() > 1
            ):  # If input was [N, C] target [N], it's handled. If target [N, ...], flatten
                alpha_t = effective_alpha[target.view(-1)]
            else:
                alpha_t = effective_alpha[target]

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


# =============================================================================
# Hybrid Risk Loss (MSE + Correlation)
# =============================================================================
class HybridRiskLoss(th.nn.Module):
    """
    Hybrid Risk Loss combining MSE and Correlation for better pattern learning.

    L_hybrid = (1 - α) × MSE + α × (1 - ρ)

    where:
        - MSE: Mean Squared Error (accuracy-focused, prone to overfitting)
        - ρ: Pearson correlation (pattern-focused, regularization effect)
        - α: Balance factor (0.0 = pure MSE, 1.0 = pure correlation)

    Benefits:
        - MSE ensures numerical accuracy
        - Correlation focuses on pattern/trend learning
        - Scale-invariant correlation acts as natural regularization
        - Reduces overfitting compared to pure MSE
    """

    def __init__(self, alpha: float = 0.7, eps: float = 1e-8, reduction: str = "mean"):
        """
        Args:
            alpha: Weight for correlation component [0.0, 1.0]
                   0.0 = pure MSE (original behavior)
                   0.5 = balanced
                   0.7 = focus on pattern learning (recommended)
            eps: Numerical stability constant
            reduction: 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
        self.mse_criterion = th.nn.MSELoss(reduction=reduction)

    def _compute_correlation_loss(
        self, pred: th.Tensor, target: th.Tensor
    ) -> th.Tensor:
        """
        Compute correlation-based loss: L = 1 - ρ(pred, target)

        Range: [0, 2] where 0 = perfect correlation, 2 = perfect anti-correlation
        """
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        n = pred_flat.shape[0]

        if n < 2:
            # Can't compute correlation with less than 2 samples
            return th.tensor(1.0, device=pred.device, dtype=pred.dtype)

        # Center the tensors
        pred_centered = pred_flat - pred_flat.mean()
        target_centered = target_flat - target_flat.mean()

        # Compute standard deviations
        pred_std = pred_centered.std(unbiased=False)
        target_std = target_centered.std(unbiased=False)

        # Check for constant predictions (std = 0)
        if pred_std < self.eps or target_std < self.eps:
            # If either is constant, correlation is undefined
            # Return 1.0 + small penalty based on variance to encourage diversity
            # This maintains gradient flow through pred_std
            variance_penalty = th.relu(self.eps - pred_std) * 10.0  # Encourage variance
            return 1.0 + variance_penalty

        # Pearson correlation
        covariance = (pred_centered * target_centered).mean()
        correlation = covariance / (pred_std * target_std + self.eps)

        # Clamp correlation to [-1, 1] for numerical stability
        correlation = th.clamp(correlation, -1.0 + self.eps, 1.0 - self.eps)

        # Loss = 1 - correlation
        # Range: [0, 2] where 0 = perfect positive correlation
        return 1.0 - correlation

    def forward(
        self, pred: th.Tensor, target: th.Tensor, return_components: bool = False
    ):
        """
        Compute hybrid loss.

        Args:
            pred: Predicted risk eta values
            target: Target risk eta values
            return_components: If True, also return component breakdown

        Returns:
            If return_components=False: Hybrid loss tensor
            If return_components=True: (hybrid_loss, {"mse": mse_loss, "corr": corr_loss})
        """
        # MSE component
        mse_loss = self.mse_criterion(pred, target)

        # Correlation component
        corr_loss = self._compute_correlation_loss(pred, target)

        # Hybrid combination
        hybrid_loss = (1.0 - self.alpha) * mse_loss + self.alpha * corr_loss

        if return_components:
            return hybrid_loss, {"mse": mse_loss, "corr": corr_loss}
        return hybrid_loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, eps={self.eps}, reduction={self.reduction}"


def compute_dynamic_class_weights(
    targets: th.Tensor,
    num_classes: int = 3,
    min_count: int = 1,
    max_weight: float = 5.0,
    smoothing: float = 0.1,
    device: th.device = None,
) -> th.Tensor:
    """
    Compute inverse-frequency class weights from batch targets.

    This addresses mode collapse by dynamically weighting classes based on
    their frequency in the current batch, rather than using static dataset-level weights.

    Args:
        targets: Ground truth labels (B,) with values in {0, 1, ..., num_classes-1}
        num_classes: Number of classes (default 3 for Bear/Side/Bull)
        min_count: Minimum samples per class to avoid extreme weights (default 1)
        max_weight: Maximum weight cap to prevent instability (default 5.0)
        smoothing: Smoothing factor [0, 1]. 0=pure inverse-freq, 1=uniform (default 0.1)
        device: Target device for output tensor

    Returns:
        weights: Tensor of shape (num_classes,) with class weights

    Example:
        If batch has: Bear=5, Side=15, Bull=30 (total=50)
        Frequencies: [0.1, 0.3, 0.6]
        Inverse-freq normalized (Bull=1.0): [6.0, 2.0, 1.0]
        Capped at max_weight=5.0: [5.0, 2.0, 1.0]
    """
    if device is None:
        device = targets.device

    # Count samples per class (bincount ensures all classes represented)
    counts = th.bincount(targets, minlength=num_classes).float()

    # Apply minimum count floor (prevents division by zero)
    counts = th.clamp(counts, min=float(min_count))

    # Total samples
    total = counts.sum()

    # Compute frequencies
    frequencies = counts / total

    # Inverse frequency weights
    inverse_freq = 1.0 / frequencies

    # Normalize so the most frequent class (highest count) has weight 1.0
    weights = inverse_freq / inverse_freq.min()

    # Apply smoothing: weights = (1 - smoothing) * weights + smoothing * uniform
    uniform_weight = th.ones(num_classes, device=device)
    weights = (1.0 - smoothing) * weights + smoothing * uniform_weight

    # Cap at max_weight to prevent extreme gradients
    weights = th.clamp(weights, max=max_weight)

    return weights.to(device)


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

        # MPS Memory Optimization Flag
        self._is_mps = self.device.type == "mps"
        if self._is_mps:
            smart_print(
                "[MPS] Apple Silicon GPU detected - enabling memory optimizations"
            )

        # Training parameters from config
        self.T_m = int(
            getattr(config, "mafia_trajectory_length", 128)
        )  # Trajectory length

        # Batch size with MPS-aware reduction
        _base_batch_size = int(getattr(config, "mafia_batch_size", 32))
        if self._is_mps:
            # MPS has ~18GB limit shared with system - reduce batch size to prevent OOM
            # Empirically: batch_size=16 with T_m=128 is safe for 18GB MPS
            _mps_max_batch = int(getattr(config, "mps_max_batch_size", 16))
            if _base_batch_size > _mps_max_batch:
                smart_print(
                    f"[MPS] Reducing batch_size from {_base_batch_size} to {_mps_max_batch} to prevent OOM"
                )
                _base_batch_size = _mps_max_batch
        self.batch_size = _base_batch_size  # Batch size B
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
        self.r_hold_alpha = float(
            getattr(config, "mafia_reward_alpha_hold", 0.0)
        )  # Trend bonus

        # Loss weights (Spec §5 Hyperparameters)
        self.lambda_pg = float(
            getattr(config, "mafia_lambda_pg", 1.0)
        )  # λ_pg = 1.0 (default)
        self.lambda_risk = float(
            getattr(config, "mafia_lambda_risk", 1.0)
        )  # λ_risk = 0.3 per Spec §5
        self.lambda_dir = float(
            getattr(config, "mafia_lambda_dir", 1.0)
        )  # λ_dir = 0.5 per Spec §5
        self.lambda_balance = float(
            getattr(config, "mafia_lambda_balance", 0.1)
        )  # Load Balancing Loss Weight
        self.balance_loss_scale = float(
            getattr(config, "mafia_balance_loss_scale", 100.0)
        )  # Scale factor to bring balance loss to similar magnitude as other losses

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
        # warmup=1, rampup=5 → total 6 epochs curriculum for 10-epoch finetune
        self.curriculum_warmup_epochs = int(
            getattr(config, "curriculum_warmup_epochs", 1)
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
            getattr(config, "mafia_focal_alpha", [1.0, 1.0, 1.0])
        )  # Class weights [bear, side, bull]
        # Temperature scaling for direction logits (T<1 sharpens distribution)
        self.direction_temperature = float(
            getattr(config, "mafia_direction_temperature", 1.0)
        )

        # Dynamic class weighting config (Per-Step)
        self.use_dynamic_class_weights = bool(
            getattr(config, "mafia_use_dynamic_class_weights", True)
        )
        self.dynamic_weight_min_count = int(
            getattr(config, "mafia_dynamic_weight_min_count", 1)
        )
        self.dynamic_weight_max = float(
            getattr(config, "mafia_dynamic_weight_max", 5.0)
        )
        self.dynamic_weight_smoothing = float(
            getattr(config, "mafia_dynamic_weight_smoothing", 0.1)
        )

        # Initialize Loss Criteria
        # Risk Loss: Hybrid MSE + Correlation (reduces overfitting)
        self.risk_corr_alpha = float(
            getattr(config, "risk_loss_correlation_alpha", 0.7)
        )
        self.risk_corr_eps = float(
            getattr(config, "risk_loss_correlation_eps", 1e-8)
        )
        self.risk_criterion = HybridRiskLoss(
            alpha=self.risk_corr_alpha,
            eps=self.risk_corr_eps,
            reduction="mean"
        )
        smart_print(
            f"[LOSS] Risk Loss: HybridRiskLoss(α={self.risk_corr_alpha}, "
            f"MSE={(1-self.risk_corr_alpha):.0%}, Corr={self.risk_corr_alpha:.0%})"
        )

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
        if self.use_dynamic_class_weights:
            smart_print(
                f"[LOSS] Dynamic Class Weights ENABLED: "
                f"min_count={self.dynamic_weight_min_count}, "
                f"max={self.dynamic_weight_max}, "
                f"smoothing={self.dynamic_weight_smoothing}"
            )
        else:
            smart_print(
                f"[LOSS] Using STATIC class weights: α={self.focal_alpha}"
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

    def _clear_memory_cache(self, force_gc: bool = False) -> None:
        """
        Clear GPU memory cache for MPS/CUDA devices.

        MPS has a hard memory limit (~18GB shared with system). This method helps
        prevent OOM by clearing unused cached memory after intensive operations.

        Args:
            force_gc: If True, also trigger Python garbage collection (slower but thorough)
        """
        if self._is_mps:
            # MPS-specific memory cleanup
            if hasattr(th.mps, "empty_cache"):
                th.mps.empty_cache()
            if hasattr(th.mps, "synchronize"):
                th.mps.synchronize()  # Ensure all operations complete before cleanup
        elif self.device.type == "cuda":
            th.cuda.empty_cache()
            th.cuda.synchronize()

        if force_gc:
            import gc

            gc.collect()

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

    def _precompute_ochlv_windows(
        self,
        full_ochlv: th.Tensor,
        start_indices: th.Tensor,
        T_m: int,
        full_market_ochlv: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Pre-compute all OCHLV windows for a batch BEFORE the time loop.

        This is a CRITICAL OPTIMIZATION: Instead of slicing tensors inside
        nested Python loops (O(B * T_m) iterations), we pre-compute all windows
        vectorized, then transfer to GPU ONCE.

        Args:
            full_ochlv: (T_total, N, 5) - Full OCHLV data
            start_indices: (B,) - Start index for each trajectory in batch
            T_m: int - Trajectory length (number of timesteps)
            full_market_ochlv: Optional (T_total, 1, 5) - Market index OCHLV

        Returns:
            precomputed_windows: (B, T_m, N, 5, T_w) - All windows pre-computed
            precomputed_market_windows: Optional (B, T_m, 1, 5, T_w)
        """
        B = start_indices.size(0)
        T_w = self.T_w
        N = full_ochlv.size(1)
        T_total = full_ochlv.size(0)

        # Pre-allocate output tensors on CPU first
        precomputed = th.zeros(B, T_m, N, 5, T_w, dtype=full_ochlv.dtype)
        precomputed_mkt = None
        if full_market_ochlv is not None:
            precomputed_mkt = th.zeros(B, T_m, 1, 5, T_w, dtype=full_market_ochlv.dtype)

        # Vectorized window computation
        # For each (b, t), window spans [max(0, t_s[b] + t - T_w + 1), t_s[b] + t + 1)
        for b in range(B):
            t_s = start_indices[b].item()
            for t in range(T_m):
                window_end = t_s + t + 1
                window_start = max(0, window_end - T_w)
                actual_len = window_end - window_start

                # Extract window
                window = full_ochlv[window_start:window_end]  # (actual_len, N, 5)

                # Handle padding if window is shorter than T_w
                if actual_len < T_w:
                    pad_size = T_w - actual_len
                    pad = window[0:1].expand(pad_size, -1, -1)
                    window = th.cat([pad, window], dim=0)  # (T_w, N, 5)

                # Permute to (N, 5, T_w) and store
                precomputed[b, t] = window.permute(1, 2, 0)

                # Same for market data
                if full_market_ochlv is not None:
                    window_mkt = full_market_ochlv[window_start:window_end]
                    if actual_len < T_w:
                        pad_mkt = window_mkt[0:1].expand(pad_size, -1, -1)
                        window_mkt = th.cat([pad_mkt, window_mkt], dim=0)
                    precomputed_mkt[b, t] = window_mkt.permute(1, 2, 0)

        # Move to device ONCE (instead of T_m times in the loop)
        precomputed = precomputed.to(self.device)
        if precomputed_mkt is not None:
            precomputed_mkt = precomputed_mkt.to(self.device)

        return precomputed, precomputed_mkt

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

    def scale_curriculum_for_finetune(self, num_epochs: int, is_finetune: bool = False):
        """
        Scale curriculum warmup/rampup for finetune iterations.

        Default curriculum: warmup=1, rampup=5 → total 6 epochs.
        For 10-epoch finetune: fits perfectly within 2/3 limit (6 epochs).
        For shorter iterations: scales down proportionally.

        Solution: Use original values if they fit, otherwise scale proportionally.
        - If warmup + rampup <= num_epochs * 2/3: use original values
        - Otherwise: scale down to fit within 2/3 of training

        Args:
            num_epochs: Total epochs for this iteration
            is_finetune: Whether this is a finetune iteration (not base training)
        """
        if not is_finetune:
            # Base training: use default config values
            self.curriculum_warmup_epochs = int(
                getattr(self.config, "curriculum_warmup_epochs", 1)
            )
            self.curriculum_penalty_rampup = int(
                getattr(self.config, "curriculum_penalty_rampup", 5)
            )
            return

        # Finetune: Scale curriculum to fit within num_epochs
        # Default: warmup=1, rampup=5 → total 6 epochs for 10-epoch finetune
        original_warmup = int(getattr(self.config, "curriculum_warmup_epochs", 1))
        original_rampup = int(getattr(self.config, "curriculum_penalty_rampup", 5))
        total_original = original_warmup + original_rampup

        # Target: lambda reaches 1.0 by epoch num_epochs * 2/3
        max_curriculum_epochs = max(1, (num_epochs * 2) // 3)

        if total_original <= max_curriculum_epochs:
            # Fits within limit, use original values directly
            scaled_warmup = original_warmup
            scaled_rampup = original_rampup
        else:
            # Need to scale down proportionally
            ratio = max_curriculum_epochs / total_original
            scaled_warmup = max(0, int(original_warmup * ratio))
            scaled_rampup = max(1, max_curriculum_epochs - scaled_warmup)

        self.curriculum_warmup_epochs = scaled_warmup
        self.curriculum_penalty_rampup = scaled_rampup

        smart_print(f"[CURRICULUM] Scaled for finetune ({num_epochs} epochs):")
        smart_print(f"  - Warmup: {original_warmup} → {scaled_warmup}")
        smart_print(f"  - Rampup: {original_rampup} → {scaled_rampup}")
        smart_print(f"  - Lambda=1.0 by epoch: {scaled_warmup + scaled_rampup}")

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

        smart_print(
            f"[OFFLINE] Fixed missing stocks. NaNs remaining: {np.isnan(ochlv_array).sum()}"
        )

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

    def _build_regime_indices(
        self,
        data_tensors: Dict[str, th.Tensor],
        min_start: int,
        max_start: int,
    ) -> np.ndarray:
        """
        Identify indices corresponding to market regime shifts (Crisis/Vol Shock).
        Spec 8: Volatility Shock or Downward DC Event.

        Args:
            data_tensors: Dictionary containing market data
            min_start: Minimum valid start index
            max_start: Maximum valid start index

        Returns:
            np.ndarray: Array of indices marked as regime shifts
        """
        market_ochlv = data_tensors.get("market_ochlv")
        if market_ochlv is None:
            return np.array([])
        
        # Move to CPU/Numpy for vector ops
        market_close = market_ochlv[:, 0, 1].cpu().numpy()
        
        T_total = len(market_close)
        regime_indices_list = []
        
        # 1. Volatility Shock Detection (Spec 8.1.3)
        # V_curr > mu_vol + k * sigma_vol
        vol_window = int(getattr(self.config, "regime_vol_window", 20))
        vol_k = float(getattr(self.config, "regime_vol_k", 3.0))
        
        # Calc Rolling Volatility
        log_ret = np.zeros_like(market_close)
        log_ret[1:] = np.log(market_close[1:] / (market_close[:-1] + 1e-8))
        
        # Use simple rolling std for speed
        rolling_std = pd.Series(log_ret).rolling(window=vol_window).std().values
        # Fill NaN
        rolling_std[:vol_window] = 0.0
        
        # Calc Volatility Z-Score (Mean/Std of Vol itself over long window e.g. 252)
        # Spec says: V_curr > mu + k*sigma. Mu/Sigma here refer to distribution of Volatility itself?
        # Or simplistic: Price change > 3 sigma?
        # "Biến động giá vượt quá biên độ chuẩn (Bollinger Band breakout/Sigma shock)"
        # Logic: V_curr > mean_vol_252 + k * std_vol_252
        
        long_window = 252
        vol_mean_long = pd.Series(rolling_std).rolling(window=long_window).mean().values
        vol_std_long = pd.Series(rolling_std).rolling(window=long_window).std().values
        
        # Detect Shocks
        threshold = vol_mean_long + vol_k * vol_std_long
        vol_shock_mask = (rolling_std > threshold) & (rolling_std > 0)
        
        # 2. Structural Break (Major DC Event) (Spec 8.1.2)
        # Downward DC at 2.0% threshold
        dc_threshold = float(getattr(self.config, "regime_dc_threshold_pct", 0.02))
        
        # Simplified DC Logic for Indexing (offline pre-scan)
        # We assume any sharp drop > 2% from recent high is a candidate
        # For strict DC, we need the event loop. Here we approximate with MaxDD_20d
        
        # Rolling Max (20d)
        roll_max = pd.Series(market_close).rolling(window=20).max().values
        dd_20 = (market_close / (roll_max + 1e-8)) - 1.0
        dc_crash_mask = dd_20 < -dc_threshold
        
        # Combine Masks
        regime_mask = vol_shock_mask | dc_crash_mask
        
        # Filter valid range [min_start, max_start]
        valid_range_mask = np.zeros(T_total, dtype=bool)
        valid_range_mask[min_start : max_start + 1] = True
        
        final_mask = regime_mask & valid_range_mask
        
        return np.where(final_mask)[0]

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
        batch_idx: Optional[int] = None,
    ) -> Optional[TrajectoryBatch]:
        """
        Sample a batch of trajectories per Spec §6.

        For TRAIN mode: Random sampling (hybrid weighted or uniform)
        For EVAL mode with batch_idx: Sequential sweep (deterministic, full coverage)
        For EVAL mode without batch_idx: Random sampling (legacy behavior)

        Each trajectory:
        1. Has start index t_s (random for TRAIN, sequential for EVAL+batch_idx)
        2. Spans [t_s, t_s + T_m) consecutive days
        3. Is independent of other trajectories in batch

        Args:
            data_tensors: Dict from prepare_data_tensors()
            market_tensors: Optional market index tensors
            mode: "TRAIN" or "EVAL"
            batch_idx: For EVAL mode, which batch index (0, 1, 2, ...) for sequential sweep.
                       If provided, returns non-overlapping sequential trajectories.
                       Returns None if batch_idx exceeds available data.

        Returns:
            TrajectoryBatch with B trajectories, or None if no more data (EVAL mode)
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

        # Direction label lookahead (needed for all sampling strategies and labels)
        dir_lookahead = int(getattr(self.config, "direction_label_lookahead", 14))

        # ========== EVAL MODE: Sequential Sweep (when batch_idx provided) ==========
        # This ensures 100% coverage of validation data with no overlap
        if mode == "EVAL" and batch_idx is not None:
            available_starts = max_start - min_start + 1
            start = min_start + batch_idx * self.batch_size

            # Check if we've exhausted all data
            if start > max_start:
                return None  # No more data to process

            # Calculate end index for this batch (may be smaller than batch_size for last batch)
            end = min(start + self.batch_size, max_start + 1)
            start_indices = np.arange(start, end)

        # ========== TRAIN MODE or EVAL without batch_idx: Random Sampling ==========
        else:
            # Sample B random start indices
            # Two strategies: uniform random OR class-balanced stratified sampling
            use_class_balanced = getattr(
                self.config, "mafia_use_class_balanced_sampling", False
            )

            if use_class_balanced and mode == "TRAIN":
                # Class-balanced sampling: sample equally from Bear, Side, Bull classes
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

            elif self.config.mafia_sampling_strategy == "hybrid_weighted" and mode == "TRAIN":
                # Hybrid Sampling: Recency (50%) + Regime (30%) + Random (20%)
                # Get ratios from config
                r_recent = float(getattr(self.config, "mafia_sampling_recency_ratio", 0.5))
                r_regime = float(getattr(self.config, "mafia_sampling_regime_ratio", 0.3))
                r_random = 1.0 - r_recent - r_regime

                n_recent = int(self.batch_size * r_recent)
                n_regime = int(self.batch_size * r_regime)
                n_random = self.batch_size - n_recent - n_regime  # Remainder

                # ========== NO DUPLICATE SAMPLING MECHANISM ==========
                # Track all sampled indices to ensure no duplicates across components
                sampled_set = set()
                start_indices = []

                # Define data ranges for sampling
                data_range = max_start - min_start + 1
                older_cutoff = min_start + int(data_range * 0.7)  # First 70% = older
                recent_start = older_cutoff + 1  # Last 30% = recent

                # Helper function: weighted sampling without replacement, excluding already sampled
                def weighted_sample_no_dup(pool, n_samples, weights, exclude_set):
                    """Sample n_samples from pool with weights, excluding indices in exclude_set."""
                    available = np.array([i for i in pool if i not in exclude_set])
                    if len(available) == 0:
                        return []

                    # Recompute weights for available indices
                    if weights is not None:
                        # Map original weights to available indices
                        pool_list = list(pool)
                        avail_weights = np.array([weights[pool_list.index(i)] for i in available])
                        avail_weights = avail_weights / (avail_weights.sum() + 1e-8)
                    else:
                        avail_weights = None

                    actual_n = min(n_samples, len(available))
                    if actual_n == 0:
                        return []

                    if avail_weights is not None:
                        samples = np.random.choice(available, size=actual_n, p=avail_weights, replace=False)
                    else:
                        samples = np.random.choice(available, size=actual_n, replace=False)
                    return list(samples)

                # Helper function: uniform sampling without replacement, excluding already sampled
                def uniform_sample_no_dup(low, high, n_samples, exclude_set):
                    """Sample n_samples uniformly from [low, high], excluding indices in exclude_set."""
                    available = np.array([i for i in range(low, high + 1) if i not in exclude_set])
                    if len(available) == 0:
                        return []
                    actual_n = min(n_samples, len(available))
                    if actual_n == 0:
                        return []
                    samples = np.random.choice(available, size=actual_n, replace=False)
                    return list(samples)

                # ========== 1. RECENCY SAMPLING (last 30%, fallback to historical) ==========
                recent_indices = np.arange(recent_start, max_start + 1)
                n_recent_available = len(recent_indices)
                p_power = float(getattr(self.config, "mafia_sampling_recency_power", 1.0))

                if n_recent_available > 0:
                    # Compute weights for recent portion
                    weights = np.linspace(0, 1, n_recent_available)
                    weights = weights ** p_power
                    weights = weights / (weights.sum() + 1e-8)

                    # Sample from recent with weights (no duplicates)
                    recent_samples = weighted_sample_no_dup(
                        recent_indices, n_recent, weights, sampled_set
                    )
                    start_indices.extend(recent_samples)
                    sampled_set.update(recent_samples)

                # If not enough from recent, fill from historical (older)
                shortfall_recent = n_recent - len([s for s in start_indices])
                if shortfall_recent > 0:
                    historical_samples = uniform_sample_no_dup(
                        min_start, older_cutoff, shortfall_recent, sampled_set
                    )
                    start_indices.extend(historical_samples)
                    sampled_set.update(historical_samples)

                # ========== 2. REGIME SAMPLING (crisis events, fallback to older) ==========
                regime_idx_pool = self._build_regime_indices(
                    data_tensors, min_start, max_start
                )

                if n_regime > 0:
                    # Filter out already sampled indices from regime pool
                    available_regime = [i for i in regime_idx_pool if i not in sampled_set]

                    if len(available_regime) > 0:
                        actual_regime = min(n_regime, len(available_regime))
                        regime_samples = np.random.choice(
                            available_regime, size=actual_regime, replace=False
                        )
                        start_indices.extend(regime_samples)
                        sampled_set.update(regime_samples)

                        # Fill shortfall from older data
                        shortfall_regime = n_regime - actual_regime
                        if shortfall_regime > 0:
                            older_samples = uniform_sample_no_dup(
                                min_start, older_cutoff, shortfall_regime, sampled_set
                            )
                            start_indices.extend(older_samples)
                            sampled_set.update(older_samples)
                    else:
                        # No regime events available - sample from older data
                        older_samples = uniform_sample_no_dup(
                            min_start, older_cutoff, n_regime, sampled_set
                        )
                        start_indices.extend(older_samples)
                        sampled_set.update(older_samples)

                # ========== 3. RANDOM SAMPLING (uniform from entire range) ==========
                if n_random > 0:
                    random_samples = uniform_sample_no_dup(
                        min_start, max_start, n_random, sampled_set
                    )
                    start_indices.extend(random_samples)
                    sampled_set.update(random_samples)

                # ========== FINAL: Handle edge case where we still need more samples ==========
                # This can happen if total available indices < batch_size
                total_sampled = len(start_indices)
                if total_sampled < self.batch_size:
                    # Allow duplicates only as last resort
                    all_indices = np.arange(min_start, max_start + 1)
                    extra_needed = self.batch_size - total_sampled
                    extra_samples = np.random.choice(all_indices, size=extra_needed, replace=True)
                    start_indices.extend(extra_samples)

                start_indices = np.array(start_indices)
                np.random.shuffle(start_indices)

            else:
                # Uniform random sampling (default for EVAL without batch_idx)
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
        batch_dc_event_flag = []  # (B, T_m) - DC structural break flag
        batch_breadth_gap = []  # (B, T_m) - avg(RSI_stocks) - RSI_index
        batch_div_signal = []  # (B, T_m) - RSI vs Price divergence
        batch_signed_vpi_zscore = []  # (B, T_m) - Volume-price efficiency
        batch_drawdown60 = []  # (B, T_m) - Rolling Drawdown 60 (New)

        # Risk params from config or defaults
        risk_lambda = float(getattr(self.config, "risk_eta_range", 0.3))
        risk_kappa = float(getattr(self.config, "risk_eta_sensitivity", 1.0))
        risk_lookahead = int(getattr(self.config, "risk_eta_lookahead", 20))

        # Rebalance Triggers
        # Vol shock now relative (e.g. 2.0x normal vol). Default 2.0.
        vol_shock_threshold = float(getattr(self.config, "mafia_rebalance_vol_threshold", 2.0))
        rebal_interval = int(getattr(self.config, "mafia_rebalance_interval", 20))

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

        # Dynamic DC Threshold Params (Spec 3.5.1 v2.2)
        # Use multipliers instead of fixed threshold
        dc_k_atr = float(getattr(self.config, "mafia_DC_multipliers", [1.0])[1]) # Use medium sensitivity (1.0)
        atr_period = 14

        for b, t_s in enumerate(start_indices):
            # Need extra history for indicators
            history_needed = 60  # RSI_14 + lookback + Vol30
            full_start = max(0, t_s - history_needed)
            full_end = t_s + T_actual

            # Stock data: close prices (L, N) and volumes (L, N)
            stock_closes = ochlv[full_start:full_end, :, 1].cpu().numpy()
            stock_volumes = ochlv[full_start:full_end, :, 4].cpu().numpy()

            # Market data
            if use_market_index:
                market_closes = market_ochlv[full_start:full_end, 0, 1].cpu().numpy()
                market_volumes = market_ochlv[full_start:full_end, 0, 4].cpu().numpy()
                market_highs = market_ochlv[full_start:full_end, 0, 2].cpu().numpy()
                market_lows = market_ochlv[full_start:full_end, 0, 3].cpu().numpy()
            else:
                market_closes = stock_closes.mean(axis=1)
                market_volumes = stock_volumes.mean(axis=1)
                # Approx High/Low for fallback ATR
                market_highs = ochlv[full_start:full_end, :, 2].mean(dim=1).cpu().numpy()
                market_lows = ochlv[full_start:full_end, :, 3].mean(dim=1).cpu().numpy()

            traj_offset = t_s - full_start
            N_stocks = stock_closes.shape[1]

            # Initialize trajectory arrays
            vol_std20_traj = np.zeros(T_actual, dtype=np.float32)
            dc_event_flag_traj = np.zeros(T_actual, dtype=np.float32)
            breadth_gap_traj = np.zeros(T_actual, dtype=np.float32)
            div_signal_traj = np.zeros(T_actual, dtype=np.float32)
            signed_vpi_zscore_traj = np.zeros(T_actual, dtype=np.float32)
            drawdown60_traj = np.zeros(T_actual, dtype=np.float32)

            # === 0. Volatility: Relative Volatility (Robustness Fix) ===
            # Previously: vol_std20 (Raw). Now: Vol10 / Vol30 (Relative).
            # Compute returns from market closes
            market_returns_np = np.zeros(len(market_closes), dtype=np.float32)
            if len(market_closes) > 1:
                market_returns_np[1:] = np.diff(market_closes) / (
                    market_closes[:-1] + 1e-8
                )

            # Compute Rolling Volatility
            vol_std10_full = np.zeros(len(market_closes), dtype=np.float32)
            vol_std30_full = np.zeros(len(market_closes), dtype=np.float32)
            
            # Use pandas rolling for efficiency and safety (matches FeatureProcessor)
            s_ret = pd.Series(market_returns_np)
            vol_std10_full = s_ret.rolling(window=10, min_periods=1).std(ddof=0).fillna(0.0).values
            vol_std30_full = s_ret.rolling(window=30, min_periods=1).std(ddof=0).fillna(0.0).values
            
            # Compute Relative Volatility (Vol10 / Vol30)
            vol_relative_full = np.divide(
                vol_std10_full, vol_std30_full, 
                out=np.ones_like(vol_std10_full), 
                where=vol_std30_full > 1e-8
            )
            
            # === 1. DC_Event_Flag: Adaptive Threshold (ATR-based) ===
            # Compute ATR for the whole window
            tr_full = np.zeros(len(market_closes), dtype=np.float32)
            if len(market_closes) > 1:
                h = market_highs
                l = market_lows
                c_prev = np.roll(market_closes, 1)
                c_prev[0] = c_prev[1]
                
                tr1 = h - l
                tr2 = np.abs(h - c_prev)
                tr3 = np.abs(l - c_prev)
                tr_full = np.maximum(tr1, np.maximum(tr2, tr3))
                
            # Smoothed ATR
            atr_full = pd.Series(tr_full).rolling(window=atr_period, min_periods=1).mean().fillna(0.0).values
            
            # Dynamic Threshold Series
            dc_thresholds = dc_k_atr * (atr_full / (market_closes + 1e-8))
            # Clamp min threshold to 0.5% to avoid noise in flat markets
            dc_thresholds = np.maximum(dc_thresholds, 0.005)

            p_ext = market_closes[0] if len(market_closes) > 0 else 1.0
            mode = "up"
            dc_flag_full = np.zeros(len(market_closes), dtype=np.float32)
            
            for i in range(1, len(market_closes)):
                price = market_closes[i]
                threshold = dc_thresholds[i]
                var = (price - p_ext) / (p_ext + 1e-8)

                if mode == "up":
                    if var < -threshold:
                        dc_flag_full[i] = 1.0  # Downward DC (Crash)
                        mode = "down"
                        p_ext = price
                    elif price > p_ext:
                        p_ext = price
                else:  # mode == "down"
                    if var > threshold:
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
                price_slope = (market_closes[i] - market_closes[i - lookback]) / (
                    market_closes[i - lookback] + 1e-8
                )
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
            
            # Compute VPI values
            if len(market_volumes) > 20:
                vol_20_avg = pd.Series(market_volumes).rolling(window=20, min_periods=1).mean().values
                for i in range(1, len(market_closes)):
                    delta_p = market_returns_np[i]  # Already computed price return
                    vol_ratio = market_volumes[i] / (vol_20_avg[i] + 1e-8)
                    # VPI = sign(ΔP) * |ΔP| / vol_ratio (adjusted formula)
                    vpi_full[i] = np.sign(delta_p) * abs(delta_p) / (vol_ratio + 1e-8)
            
            # Z-score normalization (Expanding Window for causality)
            # Use cumulative stats to avoid lookahead bias
            vpi_zscore_full = np.zeros(len(market_closes), dtype=np.float32)
            for i in range(1, len(market_closes)):
                window = vpi_full[:i+1]
                mean_val = np.mean(window)
                std_val = np.std(window)
                if std_val > 1e-8:
                    vpi_zscore_full[i] = (vpi_full[i] - mean_val) / std_val
                else:
                    vpi_zscore_full[i] = 0.0

            # Fill trajectory arrays
            for t in range(T_actual):
                idx = traj_offset + t
                if idx < len(vol_relative_full):
                    vol_std20_traj[t] = vol_relative_full[idx] # Relative Vol
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
            # (Logic maintained as is, just context for surrounding lines)
            dd60_traj = np.zeros(T_actual, dtype=np.float32)

            # Calculate rolling peak over past 60 days
            rolling_peak = np.zeros(len(market_closes), dtype=np.float32)
            # Efficient Vectorized Rolling Max (numpy trick for simple rolling window)
            # or loop as before
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
            for t in range(T_actual):
                idx = traj_offset + t
                if idx < len(vol_relative_full):
                    vol_std20_traj[t] = vol_relative_full[idx]
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
            batch_dc_event_flag.append(
                th.from_numpy(dc_event_flag_traj).to(self.device)
            )
            batch_breadth_gap.append(th.from_numpy(breadth_gap_traj).to(self.device))
            batch_div_signal.append(th.from_numpy(div_signal_traj).to(self.device))
            batch_signed_vpi_zscore.append(
                th.from_numpy(signed_vpi_zscore_traj).to(self.device)
            )
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
            dc_event_flag=th.stack(batch_dc_event_flag, dim=0).to(
                self.device
            ),  # (B, T_m)
            breadth_gap=th.stack(batch_breadth_gap, dim=0).to(self.device),  # (B, T_m)
            div_signal=th.stack(batch_div_signal, dim=0).to(self.device),
            signed_vpi_zscore=th.stack(batch_signed_vpi_zscore, dim=0).to(self.device),
            drawdown60=th.stack(batch_drawdown60, dim=0).to(self.device),
        )

        # [MPS-FIX] Clear temporary lists to free CPU memory before returning
        del batch_stock_ochlv, batch_price_returns, batch_market_returns
        del batch_rebalance_mask, batch_direction_labels, batch_risk_targets
        del batch_vol_std20, batch_dc_event_flag, batch_breadth_gap
        del batch_div_signal, batch_signed_vpi_zscore, batch_drawdown60

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

        # [OPTIMIZATION] Pre-compute ALL windows BEFORE the time loop
        # This moves O(B * T_m) slicing operations OUTSIDE the loop,
        # and transfers to GPU only ONCE instead of T_m times.
        precomputed_windows, precomputed_mkt_windows = self._precompute_ochlv_windows(
            full_ochlv=full_ochlv,
            start_indices=batch.start_indices,
            T_m=T_m,
            full_market_ochlv=full_market_ochlv,
        )
        # precomputed_windows: (B, T_m, N, 5, T_w) on device
        # precomputed_mkt_windows: (B, T_m, 1, 5, T_w) on device or None

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
        
        # [OPTIMIZATION] Accumulators for metrics (Tensor) to avoid .item() sync inside loop
        acc_loss_pg = th.tensor(0.0, device=self.device)
        acc_loss_risk = th.tensor(0.0, device=self.device)
        acc_loss_dir = th.tensor(0.0, device=self.device)
        acc_loss_bal = th.tensor(0.0, device=self.device)
        
        # Debug flag for detailed logging
        debug_logging = getattr(self.config, "mafia_debug_logging", False)


        # Reward breakdown collection
        collected_raw_returns = []
        collected_turnovers = []
        collected_symdiffs = []
        collected_rewards = []
        collected_baselines = []
        collected_advantages = []
        collected_hold_rewards = []  # New: Holding scores
        collected_risk_etas = []  # [MACRO_ONLY] Collect risk eta for min/max stats

        # Loss collections (needed for backward pass later)
        pg_losses = []
        risk_losses = []
        dir_losses = []
        balance_losses = []  # [NEW] Track balance loss per step

        # Dynamic Schedule Tracking (Resets on ANY trigger)
        # Initialize to 0. At t=0, we force rebalance.
        days_since_last_rebal = th.zeros(B, dtype=th.long, device=self.device)
        rebal_interval = self.rebalance_interval

        # Track start index for retrospective log update
        if not hasattr(self, "_trajectory_log"):
            self._trajectory_log = []
        start_log_idx = len(self._trajectory_log)

        # Initialize accumulated loss for the trajectory
        # trajectory_total_loss = th.tensor(0.0, device=self.device) # REMOVED: Reverting to per-step backward

        # [MPS-FIX] Log first timestep progress (JIT compilation is slow)
        _first_step_logged = False
        _progress_interval = max(1, T_m // 4)  # [SPEED-FIX] Log every 25% (was 10%)

        # [TIMING] Measure timestep performance
        import time
        _batch_start_time = time.perf_counter()
        _timestep_times = []

        # [MPS-FIX] Initialize loss variables to avoid UnboundLocalError
        step_pg_loss = th.tensor(0.0, device=self.device)
        risk_loss = th.tensor(0.0, device=self.device)
        dir_loss = th.tensor(0.0, device=self.device)

        for t in range(T_m):
            _step_start = time.perf_counter()

            # Progress logging for MPS (first step is slow due to JIT)
            if self._is_mps:
                if t == 0 and not _first_step_logged:
                    print(
                        f"      [MPS] Starting timestep loop (first step may be slow due to JIT)...",
                        flush=True,
                    )
                    _first_step_logged = True
                elif t == 1:
                    print(f"      [MPS] JIT warmup complete, continuing...", flush=True)
                elif t > 0 and t % _progress_interval == 0:
                    print(
                        f"      [MPS] Timestep {t}/{T_m} ({100 * t // T_m}%)",
                        flush=True,
                    )

            # [OPTIMIZED] Use pre-computed windows instead of nested loop slicing
            # Simply index into precomputed tensors (already on device)
            ochlv_batch = precomputed_windows[:, t]  # (B, N, 5, T_w)
            market_ochlv_batch = None
            if precomputed_mkt_windows is not None:
                market_ochlv_batch = precomputed_mkt_windows[:, t]  # (B, 1, 5, T_w)

            # [DEBUG] Check for NaNs in inputs (Critical input validation)
            if th.isnan(ochlv_batch).any() or th.isinf(ochlv_batch).any():
                smart_print(
                    f"     ⚠️  [CRITICAL] NaN/Inf detected in ochlv_batch at t={t}"
                )
                raise ValueError(f"NaN in ochlv_batch at t={t}")

            if market_ochlv_batch is not None and (
                th.isnan(market_ochlv_batch).any() or th.isinf(market_ochlv_batch).any()
            ):
                smart_print(
                    f"     ⚠️  [CRITICAL] NaN/Inf detected in market_ochlv_batch at t={t}"
                )
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
                vol_std20_value = batch.vol_std20[:, t : t + 1]  # (B, 1)
                dc_flag_value = batch.dc_event_flag[
                    :, t : t + 1
                ]  # (B, 1) - already binary [0, 1]
                breadth_gap_value = batch.breadth_gap[
                    :, t : t + 1
                ]  # (B, 1) - already normalized [-1, 1]
                div_signal_value = batch.div_signal[
                    :, t : t + 1
                ]  # (B, 1) - already {-1, 0, 1}
                vpi_zscore_value = batch.signed_vpi_zscore[
                    :, t : t + 1
                ]  # (B, 1) - already z-scored [-3, 3]
                drawdown60_value = batch.drawdown60[:, t : t + 1]  # (B, 1)

                # Signals are already normalized during computation, concat directly
                explicit_signals = th.cat(
                    [
                        vol_std20_value,
                        dc_flag_value,
                        breadth_gap_value,
                        div_signal_value,
                        vpi_zscore_value,
                        drawdown60_value,
                    ],
                    dim=-1,
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

                # [MPS-FIX] In MACRO_ONLY mode, immediately detach selection outputs
                # to free computation graph memory (selection gradients not needed)
                if self.config.mafia_train_mode == MafiaTrainMode.MACRO_ONLY:
                    market_vector = market_vector.detach()
                    market_scores_full = market_scores_full.detach()
                    topk_embeddings = topk_embeddings.detach()
                    topk_scores = topk_scores.detach()
                    market_logits = market_logits.detach()
                    # Note: topk_indices is already integer tensor (no grad)
                    # risk_eta and direction_logits need gradients for Macro loss

                # [MPS-FIX] In SELECTION_ONLY mode, immediately detach macro outputs
                # to free computation graph memory (macro gradients not needed)
                elif self.config.mafia_train_mode == MafiaTrainMode.SELECTION_ONLY:
                    risk_eta = risk_eta.detach()
                    direction_logits = direction_logits.detach()
                    # Note: selection outputs (market_logits, topk_scores, etc.) need gradients for PG loss

                # Temperature scaling for direction head (sharpen/flatten)
                if self.direction_temperature != 1.0:
                    direction_logits = direction_logits / self.direction_temperature

                # Save FRESH candidates before holding logic may overwrite
                fresh_topk_indices = topk_indices.clone()
                # [MPS-FIX] Detach market_context to prevent gradient accumulation through time
                collected_market_context.append(market_context.detach())  # (B, D)

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

            # [MPS-FIX] ALWAYS detach tensors when appending to collected lists
            # Backward happens from the original variables at each step, not from collected lists
            # Collected lists are only used for metrics AFTER the loop completes
            collected_topk_scores.append(topk_scores.detach())
            collected_topk_indices.append(topk_indices)  # Already no grad (indices)
            collected_risk_eta.append(risk_eta.detach())
            collected_direction_logits.append(direction_logits.detach())

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

            # [MACRO_ONLY OPTIMIZATION] Skip reward/penalty computation in MACRO_ONLY mode
            # MACRO_ONLY only needs Direction and Risk losses, not PG/Selection losses
            _is_macro_only = self.config.mafia_train_mode == MafiaTrainMode.MACRO_ONLY

            # lambda_epoch for penalty scaling (needed by logging even in MACRO_ONLY)
            lambda_epoch = self._current_lambda_epoch

            if _is_macro_only:
                # Skip expensive reward computation - use zeros for metrics
                R_net = th.zeros(B, device=self.device)
                baseline = th.zeros(B, device=self.device)
                R_raw = th.zeros(B, device=self.device)
                R_unscaled = th.zeros(B, device=self.device)
                turnover = th.zeros(B, device=self.device)
                symdiff = th.zeros(B, device=self.device)
                r_hold = th.zeros(B, device=self.device)
                A_t = th.zeros(B, device=self.device)
                A_t_raw = th.zeros(B, device=self.device)
            else:
                # FULL/SELECTION_ONLY: Compute rewards and penalties
                # 1. Calculate Rewards (R_t, Baseline) immediately
                start_idx = t + 1
                end_idx = t + 1 + self.horizon
                future_returns_slice = batch.price_returns[
                    :, start_idx:end_idx, :
                ]  # (B, h, N)
                market_returns_slice = batch.market_returns[
                    :, start_idx:end_idx
                ]  # (B, h)
                h_len = future_returns_slice.size(1)

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
                    N_stocks = batch.stock_ochlv.shape[2]

                    # [OPTIMIZED] 1. SymDiff on GPU (no CPU transfer!)
                    # Instead of Python set operations, use vectorized comparison
                    curr_mask = th.zeros(B, N_stocks, device=self.device)
                    curr_mask.scatter_(1, topk_indices, 1.0)

                    prev_mask = th.zeros(B, N_stocks, device=self.device)
                    # prev_indices is (B, K) tensor from collected_topk_indices[-1]
                    if isinstance(prev_indices, th.Tensor) and prev_indices.numel() > 0:
                        prev_mask.scatter_(1, prev_indices, 1.0)

                    # Intersection count = sum of (curr AND prev)
                    held_count = (curr_mask * prev_mask).sum(dim=1)  # (B,)

                    # Symmetric difference: 2 * (K - held) / K
                    symdiff = (2 * (self.K - held_count)) / self.K

                    # 2. Turnover (Re-weighting Cost) - Weight Based (L1 Distance)
                    curr_w_vector = th.zeros(B, N_stocks, device=self.device)
                    curr_local_w = F.softmax(topk_scores, dim=1)
                    curr_w_vector.scatter_(1, topk_indices, curr_local_w)

                    prev_w_vector = th.zeros(B, N_stocks, device=self.device)

                    if len(collected_topk_scores) >= 2:
                        p_scores = collected_topk_scores[-2]
                        p_idx = collected_topk_indices[-2]
                        prev_local_w = F.softmax(p_scores.detach(), dim=1)
                        prev_w_vector.scatter_(1, p_idx, prev_local_w)

                    weight_diff = (curr_w_vector - prev_w_vector).abs().sum(dim=1)
                    if th.isnan(weight_diff).any():
                        smart_print(
                            f"     ⚠️  [WARN] NaN in turnover weight_diff at t={t}."
                        )
                    turnover = 0.5 * weight_diff
                    turnover = th.clamp(turnover, 0.0, 1.0)

                # Trend Holding Reward (R_hold)
                r_hold = th.zeros(B, device=self.device)
                if self.r_hold_alpha > 0 and h_len > 0:
                    mkt_ret_expanded = market_returns_slice.unsqueeze(-1)
                    win_day_mask = (future_returns_slice > mkt_ret_expanded) & (
                        future_returns_slice > 0
                    )
                    stock_win_consistency = win_day_mask.float().sum(dim=1) / float(
                        h_len
                    )
                    curr_holdings = th.zeros(B, N_action, device=self.device)
                    curr_holdings.scatter_(1, topk_indices, 1.0)
                    held_mask = prev_holdings * curr_holdings
                    portfolio_win_score = (stock_win_consistency * held_mask).sum(dim=1)
                    r_hold = self.r_hold_alpha * (portfolio_win_score / self.K)

                # Advantage (Spec §5.1.1)
                # lambda_epoch already defined at top of block
                penalty = lambda_epoch * (
                    self.alpha_turnover * turnover + self.alpha_change * symdiff
                )
                R_net = R_raw + r_hold - penalty
                A_t_raw = R_net - baseline

                # Normalize Advantage (Batch statistics)
                A_mean = A_t_raw.mean()
                eps = 1e-5
                if B > 1:
                    A_std = A_t_raw.std() + eps
                else:
                    A_std = 1.0
                A_t = (A_t_raw - A_mean) / A_std

            # Store for Post-Loop Loss Calc (still needed for backprop)
            # [MPS-FIX] In MACRO_ONLY mode, detach all reward-related tensors
            # because PG loss is not computed, so gradients are not needed
            if self.config.mafia_train_mode == MafiaTrainMode.MACRO_ONLY:
                collected_raw_returns.append(R_raw.detach())
                collected_unscaled_returns.append(R_unscaled.detach())
                collected_turnovers.append(turnover.detach())
                collected_symdiffs.append(symdiff.detach())
                collected_rewards.append(R_net.detach())
                collected_hold_rewards.append(
                    r_hold.detach() if isinstance(r_hold, th.Tensor) else r_hold
                )
                collected_baselines.append(baseline.detach())
                collected_advantages.append(A_t.detach())
            else:
                collected_raw_returns.append(R_raw)
                collected_unscaled_returns.append(R_unscaled)
                collected_turnovers.append(turnover)
                collected_symdiffs.append(symdiff)
                collected_rewards.append(R_net)
                collected_hold_rewards.append(r_hold)
                collected_baselines.append(baseline)
                collected_advantages.append(A_t)

            # [MACRO_ONLY] Collect risk eta for batch/epoch summary min/max stats
            collected_risk_etas.append(risk_eta.detach())

            # -----------------------------------------------------------
            # LOSS CALCULATION (Restored)
            # -----------------------------------------------------------
            # Check if SELECTION_ONLY mode
            _is_selection_only = (
                self.config.mafia_train_mode == MafiaTrainMode.SELECTION_ONLY
            )

            # [FIX] Skip PG computation entirely in MACRO_ONLY mode to prevent
            # gradient graph accumulation from unused tensors (memory leak fix)
            if not _is_macro_only:
                # Sanitize market_logits to prevent NaN propagation
                if th.isnan(market_logits).any() or th.isinf(market_logits).any():
                    market_logits = th.nan_to_num(
                        market_logits, nan=0.0, posinf=10.0, neginf=-10.0
                    )
                # Use stable log_softmax on logits
                log_market_probs = F.log_softmax(market_logits, dim=-1)  # (B, N)

                log_probs = th.gather(log_market_probs, 1, topk_indices)  # (B, K)

                imp_weights = th.ones_like(log_probs)
                # Spec §5.1.1: Advantage must be detached to prevent gradient flow through it
                # This ensures model learns to change π(a|s), not "hack" the reward
                pg_term = -(log_probs * imp_weights * A_t.detach().unsqueeze(1)).mean(
                    dim=1
                )  # (B,)

                # Entropy bonus on full distribution (spec 5.1.1)
                # H(π) = -Σ p_i × log(p_i)
                # Use probs * log_probs for stability
                entropy = -(market_scores_full * log_market_probs).sum(dim=1)  # (B,)

                # L_PG = pg_term - β_ent × H(π) (subtract entropy to bonus exploration)
                step_loss_val = pg_term - self.beta_entropy * entropy  # (B,)
            else:
                # MACRO_ONLY: Zero tensors without gradient graph to save memory
                pg_term = th.tensor(0.0, device=self.device)
                entropy = th.tensor(0.0, device=self.device)
                step_loss_val = th.tensor(0.0, device=self.device)

            # Auxiliary Losses (skip in SELECTION_ONLY mode - macro heads are frozen)
            risk_target = batch.risk_targets[:, t]
            dir_target = batch.direction_labels[:, t]

            if _is_selection_only:
                # Skip macro loss computation - use zeros
                risk_loss = th.tensor(0.0, device=self.device)
                dir_loss = th.tensor(0.0, device=self.device)
                risk_loss_sample = th.tensor(0.0, device=self.device)
                dir_loss_sample = th.tensor(0.0, device=self.device)
            else:
                # FULL/MACRO_ONLY: Compute macro losses
                risk_loss = (
                    self.risk_criterion(risk_eta, risk_target)
                    * self.risk_scaling_factor
                )

                # [NAN-FIX] Input Guard for Direction Loss
                if th.isnan(direction_logits).any():
                    smart_print(
                        f"     ⚠️  [WARN] NaN detected in direction_logits at t={t}. Skipping DirLoss."
                    )
                    dir_loss = th.tensor(0.0, device=self.device, requires_grad=True)
                    # Sample loss also undefined
                    dir_loss_sample = th.zeros(1, device=self.device)
                else:
                    # Compute dynamic class weights if enabled
                    if self.use_dynamic_class_weights:
                        dynamic_alpha = compute_dynamic_class_weights(
                            targets=dir_target,
                            num_classes=3,
                            min_count=self.dynamic_weight_min_count,
                            max_weight=self.dynamic_weight_max,
                            smoothing=self.dynamic_weight_smoothing,
                            device=self.device,
                        )

                        dir_loss = self.dir_criterion(
                            direction_logits, dir_target, alpha_override=dynamic_alpha
                        )
                        # Sample-specific Losses (for logging only)
                        dir_loss_sample = self.dir_criterion(
                            direction_logits[0:1], dir_target[0:1], alpha_override=dynamic_alpha
                        )
                    else:
                        dir_loss = self.dir_criterion(direction_logits, dir_target)
                        # Sample-specific Losses (for logging only)
                        # Compute for sample 0 to match display (which shows sample 0)
                        dir_loss_sample = self.dir_criterion(
                            direction_logits[0:1], dir_target[0:1]
                        )

                # Sample-specific Losses (Risk) - use sample 0 to match display
                risk_loss_sample = (
                    self.risk_criterion(risk_eta[0:1], risk_target[0:1])
                    * self.risk_scaling_factor
                )

            # -----------------------------------------------------------
            # [MACRO_ONLY] Per-timestep detailed logging (ALL timesteps)
            # -----------------------------------------------------------
            # -----------------------------------------------------------
            # [MACRO_ONLY] Per-timestep detailed logging (ALL timesteps)
            # -----------------------------------------------------------
            if _is_macro_only and debug_logging:

                # Get direction prediction and target for sample b=0
                dir_names = ["Bear", "Side", "Bull"]
                pred_dir_idx = int(pred_state[0].item())
                target_dir_idx = int(batch.direction_labels[0, t].item())
                pred_dir_name = dir_names[min(max(pred_dir_idx, 0), 2)]
                target_dir_name = dir_names[min(max(target_dir_idx, 0), 2)]
                dir_match = "✅" if pred_dir_idx == target_dir_idx else "❌"

                # Direction probabilities for sample 0
                dir_probs = F.softmax(direction_logits[0], dim=-1)
                prob_bear = dir_probs[0].item()
                prob_side = dir_probs[1].item()
                prob_bull = dir_probs[2].item()

                # Risk values
                eta_val = risk_eta[0].item()
                eta_target = risk_target[0].item()

                smart_print(
                    f"  ⏱  [Epoch {self._epoch}] Batch {batch_idx + 1} | Step {t + 1}/{T_m}  [Sample 1/{B} trajectories]"
                )
                smart_print(
                    f"     📉 Losses:  L_risk(B)={risk_loss.item():.4f} | L_risk(S)={risk_loss_sample.item():.4f} | "
                    f"L_dir(B)={dir_loss.item():.4f} | L_dir(S)={dir_loss_sample.item():.4f}"
                )
                smart_print(
                    f"     🔮 Forecast:  Risk(η)={eta_val:.2f}(Tg={eta_target:.2f}) | "
                    f"Dir={pred_dir_name} (GT: {target_dir_name}){dir_match} "
                    f"[Bear={prob_bear:.2f}, Side={prob_side:.2f}, Bull={prob_bull:.2f}]"
                )

            # -----------------------------------------------------------
            # [SELECTION_ONLY] Per-timestep detailed logging (ALL timesteps)
            # -----------------------------------------------------------
            # -----------------------------------------------------------
            # [SELECTION_ONLY] Per-timestep detailed logging (ALL timesteps)
            # -----------------------------------------------------------
            elif _is_selection_only and debug_logging:

                # PG loss for this step
                pg_loss_step = step_loss_val.mean().item()

                # Selection info
                is_rebal = effective_mask[0].item() > 0.5
                rebal_status = "REBAL" if is_rebal else "HOLD"

                # Turnover/SymDiff from collected values (current step)
                turn_val = turnover[0].item() if t > 0 else 0.0
                sym_val = symdiff[0].item() if t > 0 else 0.0

                # Return and Advantage
                ret_val = R_raw[0].item() if "R_raw" in dir() else 0.0
                adv_val = A_t[0].item() if "A_t" in dir() else 0.0

                smart_print(
                    f"  ⏱  [Epoch {self._epoch}] Batch {batch_idx + 1} | Step {t + 1}/{T_m}  [Sample 1/{B} trajectories]"
                )
                smart_print(
                    f"     📉 Losses:  L_pg(B)={pg_loss_step:.4f} | Status={rebal_status}"
                )
                smart_print(
                    f"     💰 Selection:  Return={ret_val:+.4f} | Advantage={adv_val:+.4f} | "
                    f"Turnover={turn_val:.4f} | SymDiff={sym_val:.4f}"
                )

            # -----------------------------------------------------------
            # REAL-TIME LOGGING (UnifiedLogger Integration)
            # -----------------------------------------------------------
            b = 0
            # Construct Sample Details for Visualization
            curr_c_mkt = market_context[b]
            is_rebalance_b0 = effective_mask[b].item() > 0.5
            target_dir_idx = int(min(max(batch.direction_labels[b, t].item(), 0), 2))
            pred_state_b = int(min(max(pred_state[b].item(), 0), 2))

            # Triggers logic
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
                curr_s = ["Bear", "Side", "Bull"][pred_state_b]
                triggers.append(f"Regime Shift ({prev_s}->{curr_s})")
            elif t == 0:
                triggers.append("Regime Shift (Initial)")

            trigger_str = " | ".join(triggers) if triggers else "None"
            trigger_details = trigger_str

            # Costs (Conditional formatting logic moved to Logger, passing raw values)
            turn_pen_disp = (
                turnover[b].item() * self.alpha_turnover * lambda_epoch
                if is_rebalance_b0
                else 0.0
            )
            symdiff_pen_disp = (
                symdiff[b].item() * self.alpha_change * lambda_epoch
                if is_rebalance_b0
                else 0.0
            )

            # Symbols for Portfolio
            stock_list = batch.stock_list if batch.stock_list else None
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

            if stock_list:
                syms = [stock_list[idx] for idx in sel_stocks]
                # Pass list to logger
                portfolio_tickers = syms
            else:
                portfolio_tickers = [f"S{i}" for i in sel_stocks]

            # Gate Weights for Logging details
            gate_w_dict = {}
            if (
                self._latest_gate_weights is not None
                and self._latest_gate_weights.size(0) > b
            ):
                # Assuming shape (B, 4) -> Tech, DC1, DC2, DC3
                gw = self._latest_gate_weights[b]
                gate_w_dict = {
                    "Tech": gw[0].item(),
                    "DC1": gw[1].item(),
                    "DC2": gw[2].item(),
                    "DC3": gw[3].item(),
                }

            # Forecast probabilities
            probs = F.softmax(direction_logits[b], dim=-1).tolist()

            c_mkt_diff = (
                (curr_c_mkt - getattr(self, "_prev_c_mkt_b0", None)).norm().item()
                if getattr(self, "_prev_c_mkt_b0", None) is not None
                else 0.0
            )

            # Real-time display update (conditioned on config AND debug_logging)
            # Default to False to avoid verbose per-timestep logging in batch mode
            display = get_display()
            if display and getattr(self.config, "realtime_display", False) and debug_logging:
                display.update(
                    step=batch_idx * T_m + t,  # Unique step for batch
                    sample_details={
                        "step_t": t,
                        "total_steps_in_traj": T_m,
                        "trigger": trigger_str,
                        "trigger_details": trigger_details,
                        "market_raw_return": R_raw[b].item(),
                        "ctx_diff": c_mkt_diff,
                        "cost_turn_pen": turn_pen_disp,
                        "cost_symdiff_pen": symdiff_pen_disp,
                        "score_r_total": R_net[b].item(),
                        "score_r_baseline": baseline[b].item(),
                        "score_advantage": A_t_raw[b].item(),
                        "score_r_hold": r_hold[b].item(),
                        # [MPS-FIX] Use initialized loss variables (computed later in loop)
                        "loss_pg": step_loss_val[b].item()
                        if (
                            is_rebalance_b0
                            and self.config.mafia_train_mode
                            != MafiaTrainMode.MACRO_ONLY
                            and "step_loss_val" in dir()
                        )
                        else 0.0,
                        "loss_risk": risk_loss.item()
                        if (
                            self.config.mafia_train_mode
                            != MafiaTrainMode.SELECTION_ONLY
                            and risk_loss.abs().item() > 0
                        )
                        else 0.0,
                        "loss_dir": dir_loss.item()
                        if (
                            self.config.mafia_train_mode
                            != MafiaTrainMode.SELECTION_ONLY
                            and dir_loss.abs().item() > 0
                        )
                        else 0.0,
                        "loss_bal": loss_balance.item()
                        if ("loss_balance" in dir() and hasattr(loss_balance, "item"))
                        else 0.0,
                        "portfolio_held_count": float(held),  # held is calc before
                        "portfolio_tickers": portfolio_tickers,
                        "forecast_dir_pred": pred_state_b,
                        "forecast_dir_gt": target_dir_idx,
                        "forecast_dir_probs": probs,
                        "forecast_risk_eta": risk_eta[b].item(),
                        "forecast_risk_target": risk_target[b].item(),
                        "gate_weights": gate_w_dict,
                    },
                )


            # Update prev for next step (keep existing logic)
            self._prev_c_mkt_b0 = curr_c_mkt.detach()

            if t == T_m - 1:
                # End of trajectory - show batch-wide statistics
                smart_print("-" * 100)

                # [MACRO_ONLY OPTIMIZATION] Customize summary based on training mode
                if _is_macro_only:
                    # MACRO_ONLY mode: Only show Direction and Risk metrics (no PG/Selection)
                    smart_print(
                        f"  📊 BATCH SUMMARY [MACRO_ONLY] ({B} trajectories × {T_m} steps):"
                    )

                    # Direction accuracy across ALL timesteps (B × T_m predictions)
                    all_dir_logits = th.stack(
                        collected_direction_logits, dim=1
                    )  # (B, T_m, 3)
                    all_dir_preds = th.argmax(all_dir_logits, dim=-1)  # (B, T_m)
                    all_dir_targets = batch.direction_labels[:, :T_m]  # (B, T_m)
                    dir_accuracy = (
                        all_dir_preds == all_dir_targets
                    ).float().mean().item() * 100
                    total_predictions = B * T_m

                    # Prediction distribution analysis (detect bias)
                    dir_preds_flat = all_dir_preds.flatten()
                    dir_targets_flat = all_dir_targets.flatten()
                    pred_bear_pct = (dir_preds_flat == 0).float().mean().item() * 100
                    pred_side_pct = (dir_preds_flat == 1).float().mean().item() * 100
                    pred_bull_pct = (dir_preds_flat == 2).float().mean().item() * 100
                    target_bear_pct = (
                        dir_targets_flat == 0
                    ).float().mean().item() * 100
                    target_side_pct = (
                        dir_targets_flat == 1
                    ).float().mean().item() * 100
                    target_bull_pct = (
                        dir_targets_flat == 2
                    ).float().mean().item() * 100

                    # Risk prediction stats - use collected values for full trajectory stats
                    all_risk_etas = th.stack(collected_risk_etas, dim=1)  # (B, T_m)
                    mean_risk_eta = all_risk_etas.mean().item()
                    min_risk_eta = all_risk_etas.min().item()
                    max_risk_eta = all_risk_etas.max().item()
                    risk_target_mean = batch.risk_targets[:, :T_m].mean().item()

                    smart_print(
                        f"     🎯 Direction:  Accuracy={dir_accuracy:.1f}% ({total_predictions} predictions)"
                    )
                    smart_print(
                        f"        Preds:   Bear={pred_bear_pct:4.1f}% | Side={pred_side_pct:4.1f}% | Bull={pred_bull_pct:4.1f}%"
                    )
                    smart_print(
                        f"        Target:  Bear={target_bear_pct:4.1f}% | Side={target_side_pct:4.1f}% | Bull={target_bull_pct:4.1f}%"
                    )
                    smart_print(
                        f"     📉 Risk η:     Mean={mean_risk_eta:.3f}, Min={min_risk_eta:.3f}, Max={max_risk_eta:.3f} (Target={risk_target_mean:.3f})"
                    )
                    smart_print("=" * 100)
                elif _is_selection_only:
                    # SELECTION_ONLY mode: Show only Selection metrics (no Direction/Risk)
                    smart_print(
                        f"  📊 BATCH SUMMARY [SELECTION_ONLY] ({B} trajectories × {T_m} steps):"
                    )

                    # Compute batch-wide stats from the accumulated tensors
                    all_returns = th.stack(collected_raw_returns, dim=1)  # (B, T_m)
                    all_rewards = th.stack(collected_rewards, dim=1)  # (B, T_m)
                    all_baselines = th.stack(collected_baselines, dim=1)  # (B, T_m)
                    all_raw_advantages = all_rewards - all_baselines  # (B, T_m)
                    all_turnovers = th.stack(collected_turnovers, dim=1)  # (B, T_m)
                    all_symdiffs = th.stack(collected_symdiffs, dim=1)  # (B, T_m)

                    # Mean across all trajectories
                    mean_return = all_returns.mean().item()
                    std_return = all_returns.std().item()
                    min_return = all_returns.min().item()
                    max_return = all_returns.max().item()
                    mean_raw_advantage = all_raw_advantages.mean().item()
                    mean_net_reward = all_rewards.mean().item()
                    mean_turn_pen = (all_turnovers * self.alpha_turnover).mean().item()
                    mean_symdiff_pen = (all_symdiffs * self.alpha_change).mean().item()

                    # Determine if portfolio is beating market
                    adv_status = (
                        "🟢 Beating Market"
                        if mean_raw_advantage > 0
                        else "🔴 Underperforming"
                    )

                    smart_print(
                        f"     📈 Returns:    Mean={mean_return:+.4f} | Std={std_return:.4f} | Min={min_return:+.4f} | Max={max_return:+.4f}"
                    )
                    smart_print(
                        f"     💰 Net Reward: Mean={mean_net_reward:+.4f} | TurnPen={mean_turn_pen:.4f} | SymDiffPen={mean_symdiff_pen:.4f}"
                    )
                    smart_print(
                        f"     ⚖️  Advantage:  Mean={mean_raw_advantage:+.4f} ({adv_status})"
                    )
                    smart_print("=" * 100)
                else:
                    # FULL mode: Show all metrics including PG/Selection and Direction/Risk
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
                    all_dir_logits = th.stack(
                        collected_direction_logits, dim=1
                    )  # (B, T_m, 3)
                    all_dir_preds = th.argmax(all_dir_logits, dim=-1)  # (B, T_m)
                    all_dir_targets = batch.direction_labels[:, :T_m]  # (B, T_m)
                    dir_accuracy = (
                        all_dir_preds == all_dir_targets
                    ).float().mean().item() * 100
                    total_predictions = B * T_m

                    # Prediction distribution analysis (detect bias) - across ALL timesteps
                    dir_preds_flat = all_dir_preds.flatten()  # (B * T_m,)
                    dir_targets_flat = all_dir_targets.flatten()  # (B * T_m,)
                    pred_bear_pct = (dir_preds_flat == 0).float().mean().item() * 100
                    pred_side_pct = (dir_preds_flat == 1).float().mean().item() * 100
                    pred_bull_pct = (dir_preds_flat == 2).float().mean().item() * 100
                    target_bear_pct = (
                        dir_targets_flat == 0
                    ).float().mean().item() * 100
                    target_side_pct = (
                        dir_targets_flat == 1
                    ).float().mean().item() * 100
                    target_bull_pct = (
                        dir_targets_flat == 2
                    ).float().mean().item() * 100

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
            # Loss Masking and Conditional Computation (Spec §3.7 Refinement)
            # Avoid computing unused terms to prevent persistent graphs (MPS OOM prevention)

            step_total_loss = th.tensor(0.0, device=self.device)
            # Pre-compute normalization
            eps_norm = 1e-8
            norm_pg = 1.0 / (expected_rebal_count + eps_norm)

            # 1. PG Loss (Selection Phase)
            if self.config.mafia_train_mode != MafiaTrainMode.MACRO_ONLY:
                # Only compute PG term if we are NOT in Macro Only mode
                # step_pg_loss depends on the graph, effectively keeping it alive if computed
                step_pg_loss = (step_loss_val * effective_mask_float).sum()
                term_pg = step_pg_loss * norm_pg * self.lambda_pg
                step_total_loss = step_total_loss + term_pg
            else:
                step_pg_loss = th.tensor(0.0, device=self.device)
                term_pg = th.tensor(0.0, device=self.device)  # For logging

            # 2. Macro Losses (Macro Phase)
            if self.config.mafia_train_mode != MafiaTrainMode.SELECTION_ONLY:
                # Only compute Macro terms if we are NOT in Selection Only mode
                term_risk = risk_loss * norm_risk * self.lambda_risk
                term_dir = dir_loss * norm_dir * self.lambda_dir
                step_total_loss = step_total_loss + term_risk + term_dir
            else:
                term_risk = th.tensor(0.0, device=self.device)
                term_dir = th.tensor(0.0, device=self.device)

            # [NEW] Load Balancing Loss (Expert Collapse Prevention)
            # [MPS-FIX] Only compute in modes where Gate Network is being trained
            # MACRO_ONLY mode freezes Gate Network, so skip balance loss to save memory
            loss_balance = th.tensor(0.0, device=self.device)
            if (
                self.config.mafia_train_mode != MafiaTrainMode.MACRO_ONLY
                and hasattr(self, "_gate_weights_with_grad")
                and self._gate_weights_with_grad is not None
            ):
                # 1. Get Average Gate distribution across batch: (num_experts,)
                avg_gate = self._gate_weights_with_grad.mean(dim=0)
                # 2. Compute MSE from Uniform (0.25): sum((w - 0.25)^2)
                # or CV^2 (Variance / Mean^2)
                num_experts = avg_gate.size(0)
                target_uniform = 1.0 / num_experts
                loss_balance = (avg_gate - target_uniform).pow(
                    2
                ).sum() * self.balance_loss_scale

                # Add to total loss
                bal_term = loss_balance * self.lambda_balance
                step_total_loss = step_total_loss + bal_term

                # Add to list for averaging
                balance_losses.append(loss_balance.item())

            # [NEW] L1 Regularization (Lasso) - GLOBAL
            # Apply sparsity penalty to ALL trainable weights
            l1_lambda = getattr(self.config, "mafia_l1_lambda", 0.0)
            if l1_lambda > 0:
                l1_norm = th.tensor(0.0, device=self.device)
                for param in self.observer.mafia_model.parameters():
                    if param.requires_grad:
                         # Use th.norm(p=1) or .abs().sum()
                        l1_norm += param.abs().sum()
                
                l1_loss = l1_norm * l1_lambda
                step_total_loss = step_total_loss + l1_loss
            
            # Cleanup reference to free graph after backward (always clear to prevent leak)
            self._gate_weights_with_grad = None

            # ===========================================================
            # BACKWARD PASS (Per-Step) - RESTORED
            # ===========================================================
            # Truncated BPTT (k1=1): Backward immediate step only.
            # Requires graph not to be linked to previous steps (fixed via detach in turnover).
            # [SPEED-FIX] Always backward - checking zero adds CPU-GPU sync overhead
            # Zero loss just produces zero gradients (no side effects)
            if step_total_loss.requires_grad:
                try:
                    step_total_loss.backward()
                except RuntimeError as e:
                    if "nan" in str(e).lower() or "inf" in str(e).lower():
                        smart_print(
                            f"[WARN] NaN/Inf in backward pass (t={t}). Skipping step."
                        )
                    else:
                        raise e

            # [MPS-FIX] Free computation graph memory immediately after backward
            del step_total_loss

            # Store Metrics BEFORE cleanup (Float only - NO GRAPH)
            # [OPTIMIZATION] Accumulate tensors instead of .item() 
            if self.config.mafia_train_mode != MafiaTrainMode.MACRO_ONLY:
                acc_loss_pg += step_pg_loss.detach()
            
            if self.config.mafia_train_mode != MafiaTrainMode.SELECTION_ONLY:
                acc_loss_risk += risk_loss.detach()
                acc_loss_dir += dir_loss.detach()

            # [SPEED-FIX] Removed .item() check - just add detached tensor (zero adds zero)
            if loss_balance.numel() > 0:
                acc_loss_bal += loss_balance.detach()


            # [MPS-FIX] Cleanup block moved to end of loop


            # Periodic MPS cache cleanup to prevent memory buildup
            # [FIX] More aggressive cleanup for MACRO_ONLY (every 4 steps) to compensate
            # for any residual memory from detached tensors
            cleanup_interval = 64
            if self._is_mps and t > 0 and t % cleanup_interval == 0:
                self._clear_memory_cache(force_gc=False)

            # Update previous indices for next timestep turnover calculation
            prev_indices = topk_indices

            # --- Update Memory Injection State (prev_holdings) for NEXT step ---
            # This must reflect the ACTUAL portfolio held after this step (whether rebalanced or held)
            prev_holdings = th.zeros(B, N_action, device=self.device)
            prev_holdings.scatter_(1, topk_indices, 1.0)

            # [SPEED-FIX] Only do expensive logging when explicitly enabled
            # This block has many .item() calls that cause CPU-GPU sync
            if getattr(self.config, "log_trajectory_details", False):
                # Store for CSV export
                if not hasattr(self, "_trajectory_log"):
                    self._trajectory_log = []

                # Log all trajectories when enabled
                for b_log in range(B):
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
                    c_mkt_diff_b = c_mkt_diff if b_log == 0 else 0.0

                    # Build trajectory log entry based on training phase
                    log_entry = {
                        "epoch": self._epoch,
                        "batch_idx": batch_idx,
                        "sample_idx": b_log,
                        "timestep": t,
                        "trigger": trigger_str_b,
                        "rebalanced": effective_mask[b_log].item() > 0.5,
                        "c_mkt_norm": market_context[b_log].norm().item(),
                        "c_mkt_diff": c_mkt_diff_b,
                    }

                    if self.config.mafia_train_mode == MafiaTrainMode.MACRO_ONLY:
                        log_entry.update(
                            {
                                "risk_eta": risk_eta[b_log].item(),
                                "dir_logit_bear": direction_logits[b_log, 0].item(),
                                "dir_logit_side": direction_logits[b_log, 1].item(),
                                "dir_logit_bull": direction_logits[b_log, 2].item(),
                                "risk_loss": risk_loss.item(),
                                "dir_loss": dir_loss.item(),
                            }
                        )
                    else:  # SELECTION_ONLY or FULL
                        log_entry.update(
                            {
                                "selected_stocks": ",".join(
                                    [str(idx) for idx in topk_indices[b_log].cpu().tolist()]
                                ),
                                "raw_return": R_raw[b_log].item(),
                                "turnover_penalty": turnover[b_log].item()
                                * self.alpha_turnover
                                * lambda_epoch,
                                "symdiff_penalty": symdiff[b_log].item()
                                * self.alpha_change
                                * lambda_epoch,
                                "net_reward": R_net[b_log].item(),
                                "baseline": baseline[b_log].item(),
                                "advantage": A_t_raw[b_log].item(),
                                "advantage_norm": A_t[b_log].item(),
                                "pg_loss": step_pg_loss.item(),
                            }
                        )

                    self._trajectory_log.append(log_entry)

            # [REALTIME] Save trajectory logs periodically (every 32 timesteps) for real-time progress
            if (
                t > 0
                and t % 32 == 0
                and getattr(self.config, "log_trajectory_details", False)
            ):
                self._save_trajectory_log()

            # Note: Hidden state is automatically maintained inside the LSTM
            # via _cached_state (detached for TBPTT)
            
            # [MPS-FIX] Aggressive cleanup - delete intermediate tensors no longer needed
            # (Moved here to allow logging access before deletion)
            try:
                del pg_term, entropy, step_loss_val
                del term_pg, term_risk, term_dir, loss_balance
            except NameError:
                pass
            if self.config.mafia_train_mode != MafiaTrainMode.MACRO_ONLY:
                try:
                    del step_pg_loss
                except NameError:
                    pass
            if self.config.mafia_train_mode != MafiaTrainMode.SELECTION_ONLY:
                try:
                    del risk_loss, dir_loss
                except NameError:
                    pass


            # Update Context Buffer via FIFO (Spec 3.6.2.2)
            # Remove oldest ([:, 0, :]), append new (market_context)
            # market_context is (B, D). Need (B, 1, D)
            # CAUTION: Detach market_context before adding to buffer to prevent growing graph through time
            # beyond the relevant window or intended BPTT.
            # (TBPTT implies we cut gradients. Buffer acts like state memory. Standard is to detach unless BPTT desired through it.)

            history_part = batch_context_buffer[:, 1:, :]  # (B, W-1, D)
            new_part = market_context.detach().unsqueeze(1)  # (B, 1, D)
            batch_context_buffer = th.cat([history_part, new_part], dim=1)

            # [TIMING] Record timestep duration
            _timestep_times.append(time.perf_counter() - _step_start)

        # [TIMING] Report batch timing
        _batch_duration = time.perf_counter() - _batch_start_time
        if len(_timestep_times) > 2:  # Skip first 2 (JIT warmup)
            _avg_step_ms = 1000 * sum(_timestep_times[2:]) / len(_timestep_times[2:])
            print(f"      ⏱️  [TIMING] Batch: {_batch_duration:.1f}s | Avg step: {_avg_step_ms:.1f}ms (excl. JIT warmup)", flush=True)

        # ===========================================================
        # BACKWARD PASS (Per-Step) already done in loop
        # ===========================================================

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

            # [MPS-FIX] Clear gradient memory after optimizer step
            self.optimizer.zero_grad(
                set_to_none=True
            )  # More memory efficient than zero_grad()

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
        # [OPTIMIZATION] Sync once per batch
        avg_pg = (acc_loss_pg / expected_rebal_count).item() if expected_rebal_count > 0 else 0.0
        
        # Risk/Dir are averaged over T_m steps
        avg_risk = (acc_loss_risk / T_m).item()
        avg_dir = (acc_loss_dir / T_m).item()
        
        # Balance over T_m
        avg_loss_balance = (acc_loss_bal / T_m).item()


        # Reconstruct approximate total loss for display
        # (This won't exactly match the backwarded loss due to estimator norm, but close enough for logs)
        total_disp = (
            self.lambda_pg * avg_pg
            + self.lambda_risk * avg_risk
            + self.lambda_dir * avg_dir
            + self.lambda_balance * avg_loss_balance  # [NEW]
        )

        # Compute direction stats for epoch accumulation
        # Use ALL timesteps predictions and targets (B × T_m) for this batch
        if collected_direction_logits:
            all_dir_logits = th.stack(collected_direction_logits, dim=1)  # (B, T_m, 3)
            all_dir_preds = th.argmax(all_dir_logits, dim=-1)  # (B, T_m)
            batch_dir_preds = all_dir_preds.flatten().cpu().numpy()  # (B * T_m,)
        else:
            batch_dir_preds = np.array([])

        if hasattr(batch, "direction_labels"):
            batch_dir_targets = (
                batch.direction_labels[:, :T_m].flatten().cpu().numpy()
            )  # (B * T_m,)
        else:
            batch_dir_targets = np.array([])

        # Compute returns, advantages, and penalties for epoch summary
        if collected_raw_returns and collected_baselines and collected_rewards:
            all_return_ratios = th.stack(
                collected_unscaled_returns, dim=1
            )  # [METRIC FIX] (B, T_m)
            all_returns = th.stack(collected_raw_returns, dim=1)  # (B, T_m)
            all_rewards = th.stack(collected_rewards, dim=1)  # (B, T_m)
            all_baselines = th.stack(collected_baselines, dim=1)  # (B, T_m)
            all_raw_advantages = all_rewards - all_baselines  # (B, T_m)
            all_turnovers = th.stack(collected_turnovers, dim=1)  # (B, T_m)
            all_symdiffs = th.stack(collected_symdiffs, dim=1)  # (B, T_m)
            all_hold_rewards = (
                th.stack(collected_hold_rewards, dim=1)
                if collected_hold_rewards
                else None
            )

            # Mean across time for each trajectory
            batch_returns = all_returns.detach().mean(dim=1).cpu().numpy()  # (B,)
            batch_advantages = (
                all_raw_advantages.detach().mean(dim=1).cpu().numpy()
            )  # (B,)
            batch_net_rewards = all_rewards.detach().mean(dim=1).cpu().numpy()  # (B,)
            batch_turn_pens = (
                (all_turnovers.detach() * self.alpha_turnover).mean(dim=1).cpu().numpy()
            )  # (B,)
            batch_symdiff_pens = (
                (all_symdiffs.detach() * self.alpha_change).mean(dim=1).cpu().numpy()
            )  # (B,)
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
                returns=all_return_ratios.detach(),  # [METRIC FIX] Use unscaled return ratios
                turnover=all_turnovers.detach(),
                advantages=all_raw_advantages.detach(),
                hold_rewards=all_hold_rewards.detach()
                if all_hold_rewards is not None
                else None,
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
            batch_dir_logits = th.stack(
                collected_direction_logits, dim=1
            )  # (B, T_m, 3)
            # Ensure labels are same length (T_m)
            batch_dir_labels = batch.direction_labels[:, :T_m]

            direction_metrics = self.compute_direction_metrics(
                direction_logits=batch_dir_logits.detach(),
                direction_labels=batch_dir_labels,
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
            batch_risk_pred = th.stack(
                collected_risk_eta, dim=1
            )  # (B, T_m) which was actually (B,) in loop list -> (B, T_m)?
            # collected_risk_eta is list of (B,) tensors?
            # Let's check loop: collected_risk_eta.append(risk_eta) where risk_eta is (B,)
            # So stack dim=1 gives (B, T_m)
            # batch.risk_targets is (B, T_m)

            batch_risk_pred = th.stack(collected_risk_eta, dim=1)
            batch_risk_target = batch.risk_targets[:, :T_m]

            risk_metrics_val = self.compute_risk_metrics(
                risk_pred=batch_risk_pred.detach(), risk_target=batch_risk_target
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
            "loss_bal": avg_loss_balance,  # [NEW]
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
            "topk_hit_rate": selection_metrics.get("hit_rate", 0.5),
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
            "dir_logits": batch_dir_logits.detach().cpu()
            if collected_direction_logits
            else None,
            "risk_pred_raw": batch_risk_pred.detach().cpu()
            if collected_risk_eta
            else None,
            "risk_pred_raw": batch_risk_pred.detach().cpu()
            if collected_risk_eta
            else None,
            "risk_target_raw": batch_risk_target.detach().cpu()
            if collected_risk_eta
            else None,
            # Learnable Bias Param (Spec 5.3)
            "mafia_holding_bias": self.observer.mafia_model.signal_generator.holding_bias.item()
            if hasattr(self.observer.mafia_model.signal_generator, "holding_bias")
            else 0.0,
        }

        # [MPS-FIX] Clear collected tensors and GPU memory
        del (
            collected_topk_scores,
            collected_topk_indices,
            collected_risk_eta,
            collected_direction_logits,
        )
        del collected_raw_returns, collected_baselines, collected_unscaled_returns
        del (
            collected_turnovers,
            collected_symdiffs,
            collected_rewards,
            collected_advantages,
        )

        # Clear batch tensors
        if "ochlv_batch" in dir():
            del ochlv_batch
        if "market_ochlv_batch" in dir():
            del market_ochlv_batch

        # Device-aware memory cleanup
        self._clear_memory_cache(force_gc=True)

        return metrics

    def _save_trajectory_log(self):
        """
        Save collected trajectory details to CSV file.
        Called periodically during training and after each batch completion.
        """
        if not hasattr(self, "_trajectory_log") or not self._trajectory_log:
            return

        import pandas as pd
        import os

        # Determine output directory
        # Use config.res_root if set (from script)
        # IMPORTANT: Avoid hardcoded fallback to './observer_offline' to prevent accidental
        # overwrites when running with different --output-dir paths
        output_dir = getattr(self.config, "res_root", None)
        if output_dir is None:
            # Warn user about missing res_root - this indicates a setup issue
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./trajectory_logs_{timestamp}"
            smart_print(f"[WARN] config.res_root not set! Using fallback: {output_dir}")
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
        verbose: bool = True,  # NEW: Verbose print flag
        limit_batches: Optional[int] = None,
        training_mode: str = "FULL",  # NEW: Explicit Training Mode
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            data_tensors: Prepared data tensors from prepare_data_tensors()
            steps_per_epoch: Number of training steps (default: auto-computed)
            log_file: Optional path to append detailed step metrics (CSV)
            verbose: Verbose print flag
            training_mode: Explicit Training Mode

        Returns:
            ObserverValidationResult: containing all training metrics
        """
        # Sync config for phase-specific loss masking
        mode_enum = MafiaTrainMode.FULL
        clean_mode = training_mode.strip()
        if clean_mode == "MACRO_ONLY":
            mode_enum = MafiaTrainMode.MACRO_ONLY
        elif clean_mode == "SELECTION_ONLY":
            mode_enum = MafiaTrainMode.SELECTION_ONLY

        self.config.mafia_train_mode = mode_enum

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
        # Only compute for SELECTION_ONLY mode (MACRO_ONLY doesn't use penalties)
        if self.config.mafia_train_mode == MafiaTrainMode.SELECTION_ONLY:
            self._current_lambda_epoch = self._compute_lambda_epoch(self._epoch)
        else:
            self._current_lambda_epoch = 0.0  # Not used in MACRO_ONLY

        # Removed epoch-wise update in favor of step-wise in collect_and_train_step
        # self._current_temp = self.observer.update_temperature(self._epoch)
        # smart_print(f"🔥 Epoch {self._epoch} | Lambda: {self._current_lambda_epoch:.2f}") - We will log temp in step or just rely on existing logs

        epoch_metrics = {
            "loss_total": 0.0,
            "loss_pg": 0.0,
            "loss_risk": 0.0,
            "loss_dir": 0.0,
            "loss_bal": 0.0,  # [NEW] Accumulator
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
            "reward": 0.0,  # Will be sum(net_reward)? or mean?
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
        if (
            th.isnan(dir_head.classifier.weight).any()
            or th.isnan(dir_head.classifier.bias).any()
        ):
            smart_print(
                "     🔥 [CRITICAL] DirectionHead weights/bias contain NaN at START of epoch!"
            )
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
        # Only show curriculum lambda for SELECTION_ONLY mode (MACRO_ONLY doesn't use penalties)
        if self.config.mafia_train_mode == MafiaTrainMode.SELECTION_ONLY:
            smart_print(
                f"  Curriculum λ_epoch: {self._current_lambda_epoch:.2f} (penalty weight)"
            )
        smart_print("=" * 100 + "\n")

        for step in range(steps_per_epoch):
            if limit_batches is not None and step >= limit_batches:
                smart_print(f"[TRAIN] Early stopping (limit_batches={limit_batches})")
                break

            # ============================================================
            # BATCH HEADER
            # ============================================================
            # Consolidated Logging

            # Sample batch
            batch = self.sample_trajectory_batch(data_tensors, mode="TRAIN")
            # -------------------------------------------------------------------
            # Collect and train step
            # -------------------------------------------------------------------
            step_metrics = self.collect_and_train_step(
                batch, data_tensors, batch_idx=step, epoch_idx=self._epoch
            )

            if verbose:
                loss_bal_val = step_metrics.get("loss_bal", 0.0)
                bal_str = f", Bal: {loss_bal_val:.4f}" if loss_bal_val != 0 else ""

                print(
                    f"[Batch {step + 1}/{steps_per_epoch}] "
                    f"Loss: {step_metrics['loss_total']:.4f} "
                    f"(PG: {step_metrics['loss_pg']:.4f}, Risk: {step_metrics['loss_risk']:.4f}, "
                    f"Dir: {step_metrics['loss_dir']:.4f}{bal_str}) | "
                    f"Grad: {step_metrics['grad_norm']:.4f}",
                    flush=True,
                )
            elif (step + 1) % max(1, steps_per_epoch // 10) == 0:
                # Only show periodic update if NOT verbose
                progress = (step + 1) / steps_per_epoch * 100
                smart_print(
                    f"[TRAIN] Epoch {self._epoch} | "
                    f"Batch {step + 1}/{steps_per_epoch} ({progress:.0f}%) | "
                    f"L_total: {step_metrics['loss_total']:.4f}"
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
                    "bal_loss": step_metrics.get("loss_bal", 0.0),  # [NEW]
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
            if (
                "batch_returns" in step_metrics
                and len(step_metrics["batch_returns"]) > 0
            ):
                epoch_returns.append(step_metrics["batch_returns"])
            if (
                "batch_advantages" in step_metrics
                and len(step_metrics["batch_advantages"]) > 0
            ):
                epoch_advantages.append(step_metrics["batch_advantages"])
            if (
                "batch_net_rewards" in step_metrics
                and len(step_metrics["batch_net_rewards"]) > 0
            ):
                epoch_net_rewards.append(step_metrics["batch_net_rewards"])
            if (
                "batch_turn_pens" in step_metrics
                and len(step_metrics["batch_turn_pens"]) > 0
            ):
                epoch_turn_pens.append(step_metrics["batch_turn_pens"])
            if (
                "batch_symdiff_pens" in step_metrics
                and len(step_metrics["batch_symdiff_pens"]) > 0
            ):
                epoch_symdiff_pens.append(step_metrics["batch_symdiff_pens"])

            # Collect raw data for global metrics
            if step_metrics.get("dir_logits") is not None:
                epoch_dir_logits.append(step_metrics["dir_logits"])
                epoch_dir_labels.append(
                    step_metrics["dir_targets"]
                )  # Using existing targets (numpy)

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

            # [MPS-FIX] Periodic memory cleanup to prevent OOM during long epochs
            # Clean every 4 batches on MPS to balance performance vs memory
            if self._is_mps and (step + 1) % 4 == 0:
                self._clear_memory_cache(force_gc=False)

            # Clean up batch reference
            del batch

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
            all_logits = th.cat(
                [
                    t if isinstance(t, th.Tensor) else th.from_numpy(t)
                    for t in epoch_dir_logits
                ],
                dim=0,
            )
            all_labels = th.cat(
                [
                    t if isinstance(t, th.Tensor) else th.from_numpy(t)
                    for t in epoch_dir_labels
                ],
                dim=0,
            )

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
            all_risk_p = th.cat(
                [
                    t if isinstance(t, th.Tensor) else th.from_numpy(t)
                    for t in epoch_risk_preds
                ],
                dim=0,
            )
            all_risk_t = th.cat(
                [
                    t if isinstance(t, th.Tensor) else th.from_numpy(t)
                    for t in epoch_risk_targets
                ],
                dim=0,
            )

            # Compute global metrics
            global_risk_metrics = self.compute_risk_metrics(all_risk_p, all_risk_t)

            # Overwrite averaged metrics
            epoch_metrics["risk_mse"] = global_risk_metrics["mse"]
            epoch_metrics["risk_mae"] = global_risk_metrics["mae"]
            epoch_metrics["risk_correlation"] = global_risk_metrics["correlation"]

            # [MACRO_ONLY] Add min/max for Risk η display
            epoch_metrics["min_risk_eta"] = all_risk_p.min().item()
            epoch_metrics["max_risk_eta"] = all_risk_p.max().item()

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
            writer.add_scalar(
                "Train/Epoch/Sharpe", epoch_metrics["topk_sharpe_ratio"], self._epoch
            )
            writer.add_scalar(
                "Train/Epoch/Dir_F1", epoch_metrics["direction_f1_macro"], self._epoch
            )
            writer.add_scalar(
                "Train/Epoch/Risk_MSE", epoch_metrics["risk_mse"], self._epoch
            )

            # [USER-REQ] Detailed metrics
            writer.add_scalar(
                "Train/Epoch/Dir_Acc",
                epoch_metrics.get("direction_accuracy", 0.0),
                self._epoch,
            )
            writer.add_scalar(
                "Train/Epoch/Dir_F1_Bear",
                epoch_metrics.get("direction_f1_bear", 0.0),
                self._epoch,
            )

            writer.add_scalar(
                "Train/Epoch/Dir_F1_Side",
                epoch_metrics.get("direction_f1_side", 0.0),
                self._epoch,
            )
            writer.add_scalar(
                "Train/Epoch/Dir_F1_Bull",
                epoch_metrics.get("direction_f1_bull", 0.0),
                self._epoch,
            )

            writer.add_scalar(
                "Train/Epoch/Risk_MAE", epoch_metrics.get("risk_mae", 0.0), self._epoch
            )
            writer.add_scalar(
                "Train/Epoch/Risk_Corr",
                epoch_metrics.get("risk_correlation", 0.0),
                self._epoch,
            )

            # HybridRiskLoss configuration tracking
            writer.add_scalar(
                "Train/Config/Risk_Alpha",
                self.risk_corr_alpha,
                self._epoch,
            )
            # Correlation loss component = 1 - correlation (for reference)
            writer.add_scalar(
                "Train/Epoch/Risk_Corr_Loss",
                1.0 - epoch_metrics.get("risk_correlation", 0.0),
                self._epoch,
            )

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

        # Check training mode for customized display
        is_macro_only = self.config.mafia_train_mode == MafiaTrainMode.MACRO_ONLY
        is_selection_only = (
            self.config.mafia_train_mode == MafiaTrainMode.SELECTION_ONLY
        )

        if is_macro_only:
            # [MACRO_ONLY] Only show Direction and Risk metrics (no PG/Selection)
            smart_print(
                f"  🎯 Direction:   Accuracy={epoch_dir_accuracy:.1f}% (across {total_trajectories} predictions)"
            )
            smart_print(
                f"     Preds:    Bear={epoch_pred_bear:4.1f}% | Side={epoch_pred_side:4.1f}% | Bull={epoch_pred_bull:4.1f}%"
            )
            smart_print(
                f"     Target:   Bear={epoch_target_bear:4.1f}% | Side={epoch_target_side:4.1f}% | Bull={epoch_target_bull:4.1f}%"
            )
            # Risk η with min/max
            min_eta = epoch_metrics.get("min_risk_eta", 0.0)
            max_eta = epoch_metrics.get("max_risk_eta", 0.0)
            smart_print(
                f"  📉 Risk η:      Mean={epoch_metrics['mean_risk_eta']:.3f}, Min={min_eta:.3f}, Max={max_eta:.3f}"
            )
            smart_print("-" * 100)
            smart_print("  📉 Losses:")
            smart_print(f"    ├─ Total:     {epoch_metrics['loss_total']:.6f}")
            smart_print(f"    ├─ Risk:      {epoch_metrics['loss_risk']:.6f}")
            smart_print(f"    └─ Direction: {epoch_metrics['loss_dir']:.6f}")
        elif is_selection_only:
            # [SELECTION_ONLY] Only show PG/Selection metrics (no Direction/Risk)
            adv_status = (
                "🟢 Beating Market"
                if epoch_mean_advantage > 0
                else "🔴 Underperforming"
            )

            # Display portfolio performance
            smart_print(
                f"  📈 Returns:     Mean={epoch_mean_return:+.4f} | Std={epoch_std_return:.4f}"
            )
            smart_print(
                f"  💰 Net Reward:  Mean={epoch_mean_net_reward:+.4f} | TurnPen={epoch_mean_turn_pen:.4f} | SymDiffPen={epoch_mean_symdiff_pen:.4f}"
            )
            smart_print(
                f"  ⚖️  Advantage:   Mean={epoch_mean_advantage:+.4f} ({adv_status})"
            )
            smart_print("-" * 100)
            smart_print("  📉 Losses:")
            smart_print(f"    ├─ Total:     {epoch_metrics['loss_total']:.6f}")
            smart_print(f"    ├─ PG:        {epoch_metrics['loss_pg']:.6f}")
            smart_print(f"    └─ Balance:   {epoch_metrics['loss_bal']:.6f}")
        else:
            # FULL mode: Show all metrics including PG/Selection and Direction/Risk
            adv_status = (
                "🟢 Beating Market"
                if epoch_mean_advantage > 0
                else "🔴 Underperforming"
            )

            # Display portfolio performance
            smart_print(
                f"  📈 Returns:     Mean={epoch_mean_return:+.4f} | Std={epoch_std_return:.4f}"
            )
            smart_print(
                f"  💰 Net Reward:  Mean={epoch_mean_net_reward:+.4f} | TurnPen={epoch_mean_turn_pen:.4f} | SymDiffPen={epoch_mean_symdiff_pen:.4f}"
            )
            smart_print(
                f"  ⚖️  Advantage:   Mean={epoch_mean_advantage:+.4f} ({adv_status})"
            )
            smart_print(
                f"  🎯 Direction:   Accuracy={epoch_dir_accuracy:.1f}% (across {total_trajectories} predictions)"
            )
            smart_print(
                f"     Preds:    Bear={epoch_pred_bear:4.1f}% | Side={epoch_pred_side:4.1f}% | Bull={epoch_pred_bull:4.1f}%"
            )
            smart_print(
                f"     Target:   Bear={epoch_target_bear:4.1f}% | Side={epoch_target_side:4.1f}% | Bull={epoch_target_bull:4.1f}%"
            )
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
                df_log.to_csv(log_file, mode="a", header=header, index=False)
                # print(f"📝 Appended {len(df_log)} rows to detailed log: {log_file}")
            except Exception as e:
                smart_print(f"[WARN] Failed to write detailed trajectory log: {e}")

        # Convert to ObserverValidationResult
        result = ObserverValidationResult(
            epoch=self._epoch,
            training_mode=self.config.mafia_train_mode,  # MACRO_ONLY or SELECTION_ONLY
            # Loss
            loss_total=epoch_metrics["loss_total"],
            loss_pg=epoch_metrics["loss_pg"],
            loss_risk=epoch_metrics["loss_risk"],
            loss_dir=epoch_metrics["loss_dir"],
            loss_bal=epoch_metrics.get("loss_bal", 0.0),  # [FIX] Add missing loss_bal
            # Selection
            topk_sharpe_ratio=epoch_metrics["topk_sharpe_ratio"],
            topk_hit_rate=epoch_metrics.get("topk_hit_rate", 0.5),
            topk_mean_return=epoch_metrics["topk_mean_return"],
            topk_volatility=epoch_metrics["topk_volatility"],
            topk_turnover=epoch_metrics["topk_turnover"],
            # Direction
            direction_accuracy=epoch_dir_accuracy,  # Use epoch aggregated accuracy (0-100)
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
            ces_score=0.0,
        )

        # Compute phase_score for training metrics (same logic as ValidationTracker)
        if self.config.mafia_train_mode == MafiaTrainMode.MACRO_ONLY:
            # Component scores
            result.direction_score = result.direction_f1_macro  # Already in [0, 1]
            # HybridRiskLoss: risk_score = (1-α)*mse_score + α*corr_score
            alpha = self.risk_corr_alpha  # 0.7
            MAX_MSE = 1.0
            mse_norm = min(result.risk_mse / MAX_MSE, 1.0)  # Clamp to [0, 1]
            mse_score = 1.0 - mse_norm  # Higher = better
            corr_score = max(0.0, result.risk_correlation)  # Clamp negative to 0
            result.risk_score = (1.0 - alpha) * mse_score + alpha * corr_score
            # Equal weights
            result.phase_score = 0.5 * result.direction_score + 0.5 * result.risk_score
        elif self.config.mafia_train_mode == MafiaTrainMode.SELECTION_ONLY:
            # phase_score = 0.6 * sharpe_score + 0.4 * hit_rate
            # Normalize Sharpe to [0, 1]: clamp to [0, 3] then divide by 3
            sharpe_norm = min(max(result.topk_sharpe_ratio, 0.0), 3.0) / 3.0
            result.phase_score = 0.6 * sharpe_norm + 0.4 * result.topk_hit_rate

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
        metrics["topk_advantage"] = (
            advantages.mean().item() if advantages is not None else 0.0
        )
        metrics["topk_hold_reward"] = (
            hold_rewards.mean().item() if hold_rewards is not None else 0.0
        )

        # Hit Rate: % of periods where portfolio outperforms market (advantage > 0)
        if advantages is not None and advantages.numel() > 0:
            metrics["hit_rate"] = (advantages > 0).float().mean().item()
        else:
            metrics["hit_rate"] = 0.5  # Default to 50% if no data

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

        # Accuracy (0-1 scale for readability)
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
        data_tensors: Dict[
            str, th.Tensor
        ],  # Changed from dataloader to data_tensors to match usage
        steps: Optional[int] = None,
        compute_loss: bool = False,
        limit_batches: Optional[int] = None,
    ) -> ObserverValidationResult:
        """
        Validate one epoch.
        Args:
            data_tensors: Prepared validation data tensors
        Returns:
            ObserverValidationResult with all validation metrics
        """
        from RL_controller.observer_validation_metrics import ObserverValidationResult

        # The following code block seems to be intended for a `train_epoch` function
        # as it references `training_mode`, `epoch`, `verbose`, and `MafiaTrainMode`
        # which are not defined in `validate_epoch`.
        # To maintain syntactic correctness as per instructions, this block is commented out.
        # If this logic is truly needed in `validate_epoch`, the necessary variables
        # (`training_mode`, `verbose`, `epoch`, `MafiaTrainMode` import) must be provided.
        #
        # # Convert string mode to Enum if needed (or assume it matches)
        # # We need to sync self.config.mafia_train_mode for collect_and_train_step logic
        # # Map string to Enum because config expects Enum usually (or check definition)
        # # Actually config.mafia_train_mode might be string or Enum. Let's check Utils or Config.
        # # Given "MafiaTrainMode" import, we should map.
        #
        # mode_enum = MafiaTrainMode.FULL
        # if training_mode == "MACRO_ONLY":
        #     mode_enum = MafiaTrainMode.MACRO_ONLY
        # elif training_mode == "SELECTION_ONLY":
        #     mode_enum = MafiaTrainMode.SELECTION_ONLY
        #
        # self.config.mafia_train_mode = mode_enum
        # self.observer.set_train_mode(mode_enum)  # Ensure Model & SignalGen get it too
        #
        # # We also need to make sure the observer's config is consistent if it holds a reference
        # # self.observer.config.mafia_train_mode = mode_enum # Usually shared ref
        #
        # if verbose:
        #     smart_print(f"\n[EPOCH] Starting Epoch {epoch} | Mode: {training_mode}")

        self.observer.mafia_model.eval()

        if steps is None:
            T_total = data_tensors["T_total"]
            available_starts = T_total - self.T_m - self.horizon - self.T_w
            # Full coverage: use ceiling division to ensure ALL validation data is covered
            # Removed cap of 10 batches - now sweeps entire validation set
            steps = max(1, int(np.ceil(available_starts / self.batch_size)))

            # Accumulators for losses
        total_loss_total = 0.0
        total_loss_pg = 0.0
        total_loss_risk = 0.0
        total_loss_dir = 0.0
        total_loss_balance = 0.0  # [NEW] Accumulator

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
        all_market_returns = []  # NEW
        all_risk_pred = []
        all_risk_target = []
        all_direction_logits = []
        all_direction_labels = []

        # [DRY-RUN] Limit steps if requested
        if limit_batches is not None:
            if steps is None:
                steps = limit_batches
            else:
                steps = min(steps, limit_batches)

        for step in range(steps):
            # Sequential sweep: pass batch_idx for deterministic, non-overlapping coverage
            batch = self.sample_trajectory_batch(data_tensors, mode="EVAL", batch_idx=step)

            # Check if we've exhausted all data (last batch may be smaller)
            if batch is None:
                break

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

                # [OPTIMIZATION] Pre-compute ALL windows BEFORE the time loop
                # This moves O(B * T_m) slicing operations OUTSIDE the loop,
                # and transfers to GPU only ONCE instead of T_m times.
                precomputed_windows, precomputed_mkt_windows = self._precompute_ochlv_windows(
                    full_ochlv=full_ochlv,
                    start_indices=batch.start_indices,
                    T_m=T_m,
                    full_market_ochlv=full_market_ochlv,
                )
                # precomputed_windows: (B, T_m, N, 5, T_w) on device
                # precomputed_mkt_windows: (B, T_m, 1, 5, T_w) on device or None

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
                balance_losses = []  # [NEW] Track balance loss per step

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
                    # [OPTIMIZED] Use pre-computed windows instead of nested loop slicing
                    # Simply index into precomputed tensors (already on device)
                    ochlv_batch = precomputed_windows[:, t]  # (B, N, 5, T_w)
                    market_ochlv_batch = None
                    if precomputed_mkt_windows is not None:
                        market_ochlv_batch = precomputed_mkt_windows[:, t]  # (B, 1, 5, T_w)

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
                    vol_std20_value = batch.vol_std20[:, t : t + 1]  # (B, 1)
                    dc_flag_value = batch.dc_event_flag[
                        :, t : t + 1
                    ]  # (B, 1) - already binary [0, 1]
                    breadth_gap_value = batch.breadth_gap[
                        :, t : t + 1
                    ]  # (B, 1) - already normalized [-1, 1]
                    div_signal_value = batch.div_signal[
                        :, t : t + 1
                    ]  # (B, 1) - already {-1, 0, 1}
                    vpi_zscore_value = batch.signed_vpi_zscore[:, t : t + 1]  # (B, 1)
                    drawdown60_value = batch.drawdown60[:, t : t + 1]  # (B, 1)
                    # Signals are already normalized during computation, concat directly
                    explicit_signals = th.cat(
                        [
                            vol_std20_value,
                            dc_flag_value,
                            breadth_gap_value,
                            div_signal_value,
                            vpi_zscore_value,
                            drawdown60_value,
                        ],
                        dim=-1,
                    )  # (B, 6)

                    # [NAN-FIX] Sanitize Explicit Signals
                    # Force sanitization to prevent Inf/NaN from crashing DirectionHead/RiskHead
                    # (Conditional check proved unreliable on MPS)
                    explicit_signals = th.nan_to_num(
                        explicit_signals, nan=0.0, posinf=5.0, neginf=-5.0
                    )
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
                        smart_print(
                            f"     [DEBUG] NaN/Inf in market_context (Backbone) at t={t}"
                        )
                    if (
                        th.isnan(direction_logits).any()
                        or th.isinf(direction_logits).any()
                    ):
                        smart_print(
                            f"     [DEBUG] NaN/Inf in direction_logits (Head) at t={t}"
                        )

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
                        loss_balance_step = (avg_gate - target_uniform).pow(
                            2
                        ).sum().item() * self.balance_loss_scale
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
                        R_unscaled = th.zeros(B, device=self.device)  # [METRIC FIX]
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
                    collected_unscaled_returns.append(R_unscaled)  # [METRIC FIX]

                    turnover = th.zeros(B, device=self.device)
                    symdiff = th.zeros(B, device=self.device)
                    if prev_indices_tensor is not None:
                        # [OPTIMIZED] Compute symmetric difference on GPU (no CPU transfer!)
                        # Instead of Python set operations, use vectorized comparison
                        N_stocks = batch.stock_ochlv.shape[2]

                        # Create binary masks for current and previous holdings
                        curr_mask = th.zeros(B, N_stocks, device=self.device)
                        curr_mask.scatter_(1, final_indices, 1.0)

                        prev_mask = th.zeros(B, N_stocks, device=self.device)
                        prev_mask.scatter_(1, prev_indices_tensor, 1.0)

                        # Intersection count = sum of (curr AND prev)
                        held_count = (curr_mask * prev_mask).sum(dim=1)  # (B,)

                        # Symmetric difference: 2 * (K - held) / K
                        symdiff = (2 * (self.K - held_count)) / self.K

                        # Turnover (Weight Based)
                        curr_w_vector = th.zeros(B, N_stocks, device=self.device)
                        curr_local_w = F.softmax(final_scores, dim=1)
                        curr_w_vector.scatter_(1, final_indices, curr_local_w)

                        prev_w_vector = th.zeros(B, N_stocks, device=self.device)
                        if prev_scores_tensor is not None:
                            prev_local_w = F.softmax(prev_scores_tensor, dim=1)
                            prev_w_vector.scatter_(1, prev_indices_tensor, prev_local_w)

                        turnover = 0.5 * (curr_w_vector - prev_w_vector).abs().sum(
                            dim=1
                        )
                        turnover = th.clamp(turnover, 0.0, 1.0)  # Ensure range [0, 1]

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
                        win_day_mask = (future_returns_slice > mkt_ret_expanded) & (
                            future_returns_slice > 0
                        )
                        # Score per Stock (B, N)
                        stock_consistency = win_day_mask.float().sum(dim=1) / float(
                            h_len
                        )

                        # Current Holdings Mask (B, N)
                        curr_holdings = th.zeros(
                            B, self.observer.action_dim, device=self.device
                        )
                        curr_holdings.scatter_(1, final_indices, 1.0)

                        # Previous Holdings Mask (for continuity check - optional, but logical given intent)
                        # However, for R_hold, do we require holding from t-1? The spec says "Held Stocks".
                        # In training, we use `prev_holdings * curr_holdings`.
                        # Here, `prev_indices_tensor` is available.
                        if prev_indices_tensor is not None:
                            prev_h = th.zeros(
                                B, self.observer.action_dim, device=self.device
                            )
                            prev_h.scatter_(1, prev_indices_tensor, 1.0)
                            held_mask = prev_h * curr_holdings
                        else:
                            held_mask = curr_holdings  # First step, treat current as held? Or 0? Usually 0 if no prev.

                        # Sum Scores
                        # [FIX] Handle dimension mismatch during fast validation (limit_stocks < full_universe)
                        # stock_consistency has shape (B, N_data) e.g. (B, 10)
                        # held_mask has shape (B, N_model) e.g. (B, 122)
                        N_data = stock_consistency.shape[1]
                        N_model = held_mask.shape[1]
                        
                        if N_data < N_model:
                             # Align dimension by padding consistency with 0 (un-calcuated stocks don't contribute)
                             consistency_padded = th.zeros(B, N_model, device=self.device)
                             consistency_padded[:, :N_data] = stock_consistency
                             portfolio_win_score = (consistency_padded * held_mask).sum(dim=1)
                        else:
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
                        # [VALIDATION] Use unweighted standard CE for fair comparison
                        # (not focal loss or dynamic weights, which are training-specific)
                        dir_loss = F.cross_entropy(
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

                    curr_batch_rewards.append(R_raw.mean().item())  # Mean across batch
                    curr_batch_net.append(R_net.mean().item())
                    curr_batch_hold_rewards.append(r_hold.mean().item())  # [NEW]

                    # For penalties, meaningful only if rebalance occurred?
                    # If we hold, turnover=0. So summing 0s works fine for average.
                    curr_batch_turnover.append(
                        (turnover * lambda_epoch * self.alpha_turnover).mean().item()
                    )
                    curr_batch_symdiff.append(
                        (symdiff * lambda_epoch * self.alpha_change).mean().item()
                    )

                    # Collect tensors for compute_selection_metrics
                    # [METRIC FIX] Use UN-SCALED returns for financial metrics
                    # R_raw is scaled by scale_factor_reward (100.0) for RL stability
                    all_return_tensors.append(compounded_stock_returns.mean(dim=1))
                    all_turnover_tensors.append(turnover)
                    all_advantage_tensors.append(A_t_raw)

                    # Collect outputs (sanitize to prevent NaN propagation)
                    final_scores = th.nan_to_num(
                        final_scores, nan=0.0, posinf=10.0, neginf=-10.0
                    )
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
            avg_loss_balance = (
                sum(balance_losses) / len(balance_losses) if balance_losses else 0.0
            )  # [NEW]
            total_disp = (
                self.lambda_pg * avg_pg
                + self.lambda_risk * avg_risk
                + self.lambda_dir * avg_dir
                + self.lambda_balance
                * avg_loss_balance  # [NEW] Include in reported total
            )

            total_loss_total += total_disp
            total_loss_pg += avg_pg
            total_loss_risk += avg_risk
            total_loss_dir += avg_dir
            total_loss_balance += avg_loss_balance

            # Rewards and Hold Bonus
            avg_rew = (
                sum(curr_batch_rewards) / len(curr_batch_rewards)
                if curr_batch_rewards
                else 0.0
            )
            avg_net = (
                sum(curr_batch_net) / len(curr_batch_net) if curr_batch_net else 0.0
            )
            avg_turn = (
                sum(curr_batch_turnover) / len(curr_batch_turnover)
                if curr_batch_turnover
                else 0.0
            )
            avg_sym = (
                sum(curr_batch_symdiff) / len(curr_batch_symdiff)
                if curr_batch_symdiff
                else 0.0
            )

            # [NEW] Average Hold Reward
            # Note: collected_hold_rewards might be empty if horizon=0 or other edge cases
            # We need to collect it in the loop first!
            avg_hold_reward = (
                sum(curr_batch_hold_rewards) / len(curr_batch_hold_rewards)
                if curr_batch_hold_rewards
                else 0.0
            )

            total_reward += avg_rew
            total_net_reward += avg_net
            total_turnover_penalty += avg_turn
            total_symdiff_penalty += avg_sym
            total_hold_reward += avg_hold_reward  # Need this accumulator initialized

            # Store for metric computation
            all_topk_indices.append(topk_indices_stack)
            all_topk_scores.append(topk_scores_stack)
            all_price_returns.append(batch.price_returns[:, :T_m, :])
            all_market_returns.append(batch.market_returns[:, :T_m])  # NEW
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
        avg_hold_reward = total_hold_reward / steps  # [NEW]

        # Concatenate all batches for metric computation
        all_topk_indices_cat = th.cat(all_topk_indices, dim=0)
        all_topk_scores_cat = th.cat(all_topk_scores, dim=0)
        all_price_returns_cat = th.cat(all_price_returns, dim=0)
        all_market_returns_cat = th.cat(all_market_returns, dim=0)  # NEW
        all_risk_pred_cat = th.cat(all_risk_pred, dim=0)
        all_risk_target_cat = th.cat(all_risk_target, dim=0)
        all_direction_logits_cat = th.cat(all_direction_logits, dim=0)
        all_direction_labels_cat = th.cat(all_direction_labels, dim=0)

        all_direction_labels_cat = th.cat(all_direction_labels, dim=0)

        # Consolidate standard metric tensors
        all_returns_cat = (
            th.cat(all_return_tensors, dim=0)
            if all_return_tensors
            else th.tensor([], device=self.device)
        )
        all_turnovers_cat = (
            th.cat(all_turnover_tensors, dim=0)
            if all_turnover_tensors
            else th.tensor([], device=self.device)
        )
        all_advantages_cat = (
            th.cat(all_advantage_tensors, dim=0)
            if all_advantage_tensors
            else th.tensor([], device=self.device)
        )

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
            training_mode=self.config.mafia_train_mode,  # MACRO_ONLY or SELECTION_ONLY
            loss_total=avg_loss_total,
            loss_pg=avg_loss_pg,
            loss_risk=avg_loss_risk,
            loss_dir=avg_loss_dir,
            loss_bal=avg_loss_balance,  # [NEW]
            topk_sharpe_ratio=selection_metrics["sharpe_ratio"],
            topk_hit_rate=selection_metrics.get("hit_rate", 0.5),
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
            topk_hold_reward=avg_hold_reward,  # [NEW]
            reward=avg_reward,
            net_reward=avg_net_reward,
            turnover_penalty=avg_turnover_penalty,
            symdiff_penalty=avg_symdiff_penalty,
        )

        # Compute phase_score (same logic as ValidationTracker)
        if self.config.mafia_train_mode == MafiaTrainMode.MACRO_ONLY:
            # Component scores
            result.direction_score = result.direction_f1_macro  # Already in [0, 1]
            # HybridRiskLoss: risk_score = (1-α)*mse_score + α*corr_score
            alpha = self.risk_corr_alpha  # 0.7
            MAX_MSE = 1.0
            mse_norm = min(result.risk_mse / MAX_MSE, 1.0)  # Clamp to [0, 1]
            mse_score = 1.0 - mse_norm  # Higher = better
            corr_score = max(0.0, result.risk_correlation)  # Clamp negative to 0
            result.risk_score = (1.0 - alpha) * mse_score + alpha * corr_score
            # Equal weights
            result.phase_score = 0.5 * result.direction_score + 0.5 * result.risk_score
        elif self.config.mafia_train_mode == MafiaTrainMode.SELECTION_ONLY:
            # phase_score = 0.6 * sharpe_score + 0.4 * hit_rate
            # Normalize Sharpe to [0, 1]: clamp to [0, 3] then divide by 3
            sharpe_norm = min(max(result.topk_sharpe_ratio, 0.0), 3.0) / 3.0
            result.phase_score = 0.6 * sharpe_norm + 0.4 * result.topk_hit_rate

        self.observer.mafia_model.train()

        return result

    @th.no_grad()
    def generate_inference_state(
        self, data_tensors: Dict[str, th.Tensor], desc: str = "Inference"
    ) -> Dict[str, th.Tensor]:
        """
        [Phase 1 Data Factory]
        Run Observer in EVAL mode to generate 'Vintage States' (RL_State) for downstream TD3.

        Outputs (Time-Sequential):
        - embeddings: [T, N, D] - Fused embeddings for all stocks
        - top_k_indices: [T, K] - Selected stock indices
        - risk_eta: [T, 1] - Risk scalar
        - dates: [T] - Timestamps

        Args:
            data_tensors: Dictionary containing inputs (price_ochlv, stock_ochlv, etc.)
            desc: Description for progress bar

        Returns:
            Dict containing the 'Vintage State' tensors.
        """
        self.observer.mafia_model.eval()

        # 1. Unpack Data
        dates = data_tensors["dates"]
        # Ensure tensors are on device
        stock_ochlv = data_tensors["ochlv"].to(self.device).float()  # (T_total, N, 5)

        T_total, N_stocks, _ = stock_ochlv.shape
        T_start = self.T_w  # Start after lookback window
        T_end = T_total  # Run until end

        results = {
            "embeddings": [],
            "top_k_indices": [],
            "top_k_embeddings": [],
            "risk_eta": [],
            "dates": [],
        }

        # Buffers for Stateful Processing
        direction_history = th.zeros(
            (1, self.config.mafia_direction_window_size),  # Batch=1
            dtype=th.long,
            device=self.device,
        )
        direction_history.fill_(1)  # Start 'Sideway'

        days_since_regime_shift = th.zeros(1, dtype=th.long, device=self.device)
        days_since_last_rebal = th.zeros(1, dtype=th.long, device=self.device)

        # Context Buffer for Temporal Augmentation
        W_route = int(getattr(self.config, "router_context_window", 5))
        if hasattr(self.observer.mafia_model, "D"):
            mafia_D = self.observer.mafia_model.D
        else:
            mafia_D = int(getattr(self.config, "mafia_D", 128))

        batch_context_buffer = th.zeros(1, W_route, mafia_D, device=self.device)

        if hasattr(self.observer.mafia_model, "init_temporal_state"):
            self.observer.mafia_model.init_temporal_state(B=1)

        print(
            f"[{desc}] Generating Inference State from t={T_start} to {T_end} (N={N_stocks} stocks)..."
        )

        # 2. Sequential Inference Loop
        for t in range(T_start, T_end):
            # 2.1 Prepare Batch (Size 1)
            s_idx = t - self.T_w
            e_idx = t

            # stock_ochlv is (T_total, N, 5), we need (B, N, 5, T_w)
            window = stock_ochlv[s_idx:e_idx]  # (T_w, N, 5)
            batch_stock_ochlv = window.permute(1, 2, 0).unsqueeze(0)  # (1, N, 5, T_w)

            # Handle potential NaNs in input
            if th.isnan(batch_stock_ochlv).any():
                batch_stock_ochlv = th.nan_to_num(batch_stock_ochlv, nan=0.0)

            # 2.2 Forward Pass
            # MAFIAModel.forward() returns tuple, not namedtuple
            # Create dummy explicit_signals (B=1, 6 signals) for Direction/Risk Heads
            dummy_signals = th.zeros(1, 6, device=self.device)

            outputs = self.observer.mafia_model.forward(
                ochlv_data=batch_stock_ochlv,
                market_index_ochlv_data=None,  # Auto-derived inside model
                force_topk_indices=None,
                router_context_buffer=batch_context_buffer,
                explicit_signals=dummy_signals,
                prev_holdings=None,
            )

            # Unpack tuple: (market_vector, eta, market_scores_full, sigma_logits,
            #                market_context, topk_indices, topk_embeddings, topk_scores, market_logits)
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

            # 2.3 Extract Outputs
            results["embeddings"].append(market_context.detach().cpu())
            results["top_k_indices"].append(topk_indices.detach().cpu())
            results["top_k_embeddings"].append(topk_embeddings.detach().cpu())
            results["risk_eta"].append(
                risk_eta.detach().cpu().unsqueeze(-1)
            )  # (B,) -> (B, 1)
            results["dates"].append(dates[t])

            # 2.4 Update States
            pred_dir = th.argmax(direction_logits, dim=-1)
            direction_history = th.cat(
                [direction_history[:, 1:], pred_dir.unsqueeze(1)], dim=1
            )
            days_since_last_rebal += 1

            # Update context buffer (rolling window update)
            # batch_context_buffer is (1, W, D), market_context is (1, D)
            batch_context_buffer = th.cat(
                [batch_context_buffer[:, 1:, :], market_context.unsqueeze(1)], dim=1
            )

            if hasattr(self.observer.mafia_model, "detach_temporal_state"):
                self.observer.mafia_model.detach_temporal_state()

        # 3. Concatenate Results
        final_state = {
            "embeddings": th.cat(results["embeddings"], dim=0),  # (T, D)
            "top_k_indices": th.cat(results["top_k_indices"], dim=0),  # (T, K)
            "top_k_embeddings": th.cat(results["top_k_embeddings"], dim=0),  # (T, K, D)
            "risk_eta": th.cat(results["risk_eta"], dim=0),  # (T, 1)
            "dates": results["dates"],
        }

        print(f"[{desc}] Generated: {final_state['embeddings'].shape[0]} samples.")
        return final_state


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
