#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observer Validation Metrics - Output structure for Observer on-policy validation

This module defines the validation result structure for MAFIA Observer training
per Spec §7. Observer training is ON-POLICY (Collect → Train → Discard) and
outputs Observer-specific metrics, NOT TD3 off-policy metrics.

Validation Metrics Categories:
    1. Loss Components: L_PG, L_Risk, L_Dir
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
        loss_dir: Cross-Entropy loss for direction classification
        
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
        ces_score: Final CES = 0.6*Sharpe + 0.2*Dir_F1 + 0.2*(1-Risk_MSE)
        ces_rank_sharpe: Rank-normalized Sharpe (0 to 1)
        ces_rank_dir_f1: Rank-normalized Direction F1 (0 to 1)
        ces_rank_risk_mse: Rank-normalized Risk MSE (0 to 1)
    """
    
    epoch: int
    
    # Loss components
    loss_total: float
    loss_pg: float
    loss_risk: float
    loss_dir: float
    
    # Selection performance
    topk_sharpe_ratio: float
    topk_mean_return: float
    topk_volatility: float
    topk_turnover: float
    
    # Direction performance
    direction_accuracy: float
    direction_f1_bear: float
    direction_f1_side: float
    direction_f1_bull: float
    direction_f1_macro: float
    
    # Risk performance
    risk_mse: float
    risk_mae: float
    risk_correlation: float
    
    # CES components (updated by ValidationMetricsTracker)
    ces_score: float = 0.0
    ces_rank_sharpe: float = 0.0
    ces_rank_dir_f1: float = 0.0
    ces_rank_risk_mse: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return asdict(self)
    
    def summary_str(self) -> str:
        """Return one-line summary for logging."""
        return (
            f"Epoch {self.epoch} | "
            f"CES: {self.ces_score:.4f} | "            
            f"Sharpe: {self.topk_sharpe_ratio:.3f} | "
            f"Dir_F1: {self.direction_f1_macro:.3f} | "
            f"Risk_MSE: {self.risk_mse:.4f} | "
            f"Loss: {self.loss_total:.4f}"
        )
