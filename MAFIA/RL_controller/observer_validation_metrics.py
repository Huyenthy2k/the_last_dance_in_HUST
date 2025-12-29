#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observer Validation Metrics - Output structure for Observer on-policy validation

This module defines the validation result structure for MAFIA Observer training
per Spec §7. Observer training is ON-POLICY (Collect → Train → Discard) and
outputs Observer-specific metrics, NOT TD3 off-policy metrics.

Validation Metrics Categories:
    1. Loss Components: L_PG, L_Risk, L_Dir, L_Bal
    2. Selection Performance: Top-K Sharpe Ratio, returns, volatility, turnover
    3. Direction Performance: F1-Score for Bull/Side/Bear classification
    4. Risk Performance: MSE/MAE/correlation for eta prediction
    5. CES Score: Composite Efficiency Score with rank normalization
"""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ObserverValidationResult:
    """
    Validation result for one epoch of Observer training.
    
    This structure contains ONLY Observer-specific metrics.
    NO TD3 metrics (policy returns, actions, etc.) should be included.
    
    Attributes:
        epoch: Epoch number (0-indexed)
        
        # Loss Components (Spec §5)
        loss_total: Total weighted loss
        loss_pg: Policy Gradient loss (selection task)
        loss_risk: MSE loss for risk eta prediction
        loss_risk: MSE loss for risk eta prediction
        loss_dir: Cross-Entropy loss for direction classification
        loss_bal: Load balancing loss (auxiliary)
        
        # Selection Performance Metrics (Spec §7.1168)
        topk_sharpe_ratio: Sharpe Ratio of equal-weighted Top-K portfolio
        topk_mean_return: Average daily return of Top-K
        topk_volatility: Std dev of Top-K returns
        topk_turnover: Average % of portfolio changed per rebalance
        
        # Direction Classification Metrics (Spec §7.1169)
        direction_accuracy: Overall accuracy
        direction_f1_bear: F1-score for Bear class
        direction_f1_side: F1-score for Side class
        direction_f1_bull: F1-score for Bull class
        direction_f1_macro: Macro-averaged F1 across 3 classes
        
        # Risk Prediction Metrics (Spec §7.1170)
        risk_mse: Mean Squared Error for eta prediction
        risk_mae: Mean Absolute Error for eta prediction
        risk_correlation: Pearson correlation between pred and target eta
        
        # Composite Efficiency Score (Spec §7.1164-1178)
        ces_score: Final CES = 0.5*Sharpe + 0.3*Dir_F1 + 0.2*(1-Risk_MSE)
        ces_rank_sharpe: Rank-normalized Sharpe (0 to 1)
        ces_rank_dir_f1: Rank-normalized Direction F1 (0 to 1)
        ces_rank_risk_mse: Rank-normalized Risk MSE (0 to 1)
        
        # Additional Metrics (Requested)
        topk_advantage: Advantage of Top-K over baseline (Top-K Return - Baseline Return)
        reward: Raw reward sum
        net_reward: Net reward (Reward - Penalties)
        turnover_penalty: Turnover penalty value
        symdiff_penalty: Symdiff penalty value
    """
    
    epoch: int

    # Training mode (MACRO_ONLY or SELECTION_ONLY)
    training_mode: str = ""

    # Phase-specific score for checkpoint selection
    # MACRO_ONLY: 0.5 * direction_score + 0.5 * risk_score
    # SELECTION_ONLY: 0.6 * sharpe_score + 0.4 * hit_rate
    phase_score: float = 0.0

    # Component scores for MACRO_ONLY (displayed in CSV)
    # direction_score: Direction F1 Macro (range 0-1)
    # risk_score: (1-α)*mse_score + α*corr_score (HybridLoss, α=0.7)
    #   mse_score = 1 - mse_norm (higher = better)
    #   corr_score = max(0, correlation) (clamp negative to 0)
    direction_score: float = 0.0
    risk_score: float = 0.0

    # Composite Efficiency Score (legacy, kept for SELECTION_ONLY compatibility)
    ces_score: float = 0.0
    ces_rank_sharpe: float = 0.0
    ces_rank_dir_f1: float = 0.0
    ces_rank_risk_mse: float = 0.0

    # Loss components
    loss_total: float = 0.0
    loss_pg: float = 0.0
    loss_risk: float = 0.0
    loss_dir: float = 0.0
    loss_bal: float = 0.0
    
    # Selection performance
    topk_sharpe_ratio: float = 0.0
    topk_hit_rate: float = 0.0  # Win rate: % of days portfolio outperforms index
    topk_mean_return: float = 0.0
    topk_volatility: float = 0.0
    topk_turnover: float = 0.0
    
    # Direction performance
    direction_accuracy: float = 0.0
    direction_f1_bear: float = 0.0
    direction_f1_side: float = 0.0
    direction_f1_bull: float = 0.0
    direction_f1_macro: float = 0.0
    
    # Risk performance
    risk_mse: float = 0.0
    risk_mae: float = 0.0
    risk_correlation: float = 0.0
    
    # Additional metrics
    topk_advantage: float = 0.0
    topk_hold_reward: float = 0.0
    reward: float = 0.0
    net_reward: float = 0.0
    turnover_penalty: float = 0.0
    symdiff_penalty: float = 0.0
    

    
    # Removed RL metrics (always 0 in validation)

    # Fields specific to each training phase
    # MACRO_ONLY: Direction + Risk heads (no PG, no Balance loss)
    MACRO_ONLY_FIELDS = {
        "epoch", "training_mode", "phase_score",
        "direction_score", "risk_score",  # Component scores
        "loss_total", "loss_risk", "loss_dir",  # NO loss_pg, NO loss_bal
        "direction_accuracy", "direction_f1_bear", "direction_f1_side",
        "direction_f1_bull", "direction_f1_macro",
        "risk_mse", "risk_mae", "risk_correlation",
    }

    # SELECTION_ONLY: PG + Gate Network (no Risk, no Direction loss)
    SELECTION_ONLY_FIELDS = {
        "epoch", "training_mode", "phase_score",
        "loss_total", "loss_pg", "loss_bal",  # NO loss_risk, NO loss_dir
        "topk_sharpe_ratio", "topk_hit_rate", "topk_mean_return", "topk_volatility", "topk_turnover",
        "topk_advantage", "topk_hold_reward",
        "reward", "net_reward", "turnover_penalty", "symdiff_penalty",
    }

    def to_dict(self, filter_by_phase: bool = True) -> Dict:
        """
        Convert to dictionary for CSV export.

        Args:
            filter_by_phase: If True, only include fields relevant to training_mode

        Returns:
            Dictionary with filtered or all fields
        """
        all_fields = asdict(self)

        if not filter_by_phase or not self.training_mode:
            return all_fields

        # Filter based on training mode
        if self.training_mode == "MACRO_ONLY":
            return {k: v for k, v in all_fields.items() if k in self.MACRO_ONLY_FIELDS}
        elif self.training_mode == "SELECTION_ONLY":
            return {k: v for k, v in all_fields.items() if k in self.SELECTION_ONLY_FIELDS}

        return all_fields
    
    def summary_str(self) -> str:
        """Return one-line summary for logging (phase-aware)."""
        if self.training_mode == "MACRO_ONLY":
            return (
                f"Epoch {self.epoch} | "
                f"Score: {self.phase_score:.4f} (Dir: {self.direction_score:.3f}, Risk: {self.risk_score:.3f}) | "
                f"Dir_F1: {self.direction_f1_macro:.3f} | "
                f"Risk_MSE: {self.risk_mse:.4f} | "
                f"Loss: {self.loss_total:.4f}"
            )
        else:  # SELECTION_ONLY
            return (
                f"Epoch {self.epoch} | "
                f"Sharpe: {self.topk_sharpe_ratio:.3f} | "
                f"Turnover: {self.topk_turnover:.3f} | "
                f"Loss: {self.loss_total:.4f}"
            )
