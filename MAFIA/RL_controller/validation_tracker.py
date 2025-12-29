#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Metrics Tracker - Manages validation history and CES computation

This module implements rank-based normalization for CES (Composite Efficiency Score)
computation per Spec §7.1164-1178.

CES Formula:
    CES = 0.5 × S_sharpe + 0.3 × S_dir_f1 + 0.2 × (1 - S_risk_mse)
    
Where S_* are rank-normalized scores:
    1. Rank all checkpoints from worst (1) to best (N)
    2. Normalize: S_hat = (Rank(M) - 1) / (N - 1)
    3. Result ∈ [0, 1] reflects percentile performance
"""

import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from RL_controller.observer_validation_metrics import ObserverValidationResult


class ValidationMetricsTracker:
    """
    Tracks validation metrics across epochs and computes phase-specific scores.

    Phase-specific scoring:
    - MACRO_ONLY: 0.5 * Dir_F1 + 0.5 * risk_score
      where risk_score = (1-α)*mse_score + α*corr_score (HybridRiskLoss, α=0.7)
    - SELECTION_ONLY: 0.6 * sharpe_score + 0.4 * hit_rate

    Responsibilities:
    - Accumulate validation results per epoch
    - Compute phase-appropriate scores for checkpoint selection
    - Identify best checkpoint based on phase_score
    - Save validation history to CSV

    Attributes:
        output_dir: Directory to save validation metrics CSV
        training_mode: "MACRO_ONLY" or "SELECTION_ONLY"
        history: List of all validation results
        best_epoch: Epoch index with highest phase_score
        best_score: Best phase_score achieved
    """

    # Fixed N for rank-based normalization to ensure stable scores
    RANK_N_FIXED = 14

    # HybridRiskLoss alpha: weight for correlation component
    # Matches config.py risk_loss_correlation_alpha default
    RISK_CORR_ALPHA = 0.7

    def __init__(self, output_dir: str, training_mode: str = "MACRO_ONLY", rank_n: int = None):
        """
        Initialize tracker.

        Args:
            output_dir: Directory to save valid_metrics.csv
            training_mode: "MACRO_ONLY" or "SELECTION_ONLY"
            rank_n: Fixed N for rank normalization (default: 14)
        """
        self.output_dir = output_dir
        self.training_mode = training_mode
        self.history: List[ObserverValidationResult] = []
        self.best_epoch: Optional[int] = None
        self.best_score: float = -np.inf
        self.rank_n = rank_n if rank_n is not None else self.RANK_N_FIXED

        # Legacy alias for backward compatibility
        self.best_ces = self.best_score
        self.ces_rank_n = self.rank_n
        
    def add_epoch(self, result: ObserverValidationResult, allow_best_update: bool = True) -> bool:
        """
        Add validation result for an epoch and update phase scores.
        If epoch already exists (from resume), replace it instead of duplicating.

        Args:
            result: Validation result to add
            allow_best_update: If False, this epoch cannot be selected as new best even if score is higher.

        Returns:
            True if this is a new best checkpoint
        """
        # Check if epoch already exists (resume scenario) - replace instead of duplicate
        existing_idx = None
        for idx, h in enumerate(self.history):
            if h.epoch == result.epoch:
                existing_idx = idx
                break

        if existing_idx is not None:
            self.history[existing_idx] = result
        else:
            self.history.append(result)

        self._compute_phase_scores()
        is_best = self._update_best(allow_best_update=allow_best_update)
        return is_best
    
    def _compute_phase_scores(self):
        """
        Compute phase-specific scores for checkpoint selection.

        MACRO_ONLY:
            direction_score = Dir_F1_macro (range 0-1)
            risk_score = (1-α)*mse_score + α*corr_score (HybridRiskLoss, α=0.7)
                mse_score = 1 - mse_norm (higher = better)
                corr_score = max(0, correlation) (clamp negative to 0)
            phase_score = 0.5 * direction_score + 0.5 * risk_score (equal weights)

        SELECTION_ONLY:
            phase_score = 0.6 * sharpe_score + 0.4 * hit_rate
            where sharpe_score = clamp(Sharpe, 0, 3) / 3 (normalized to [0, 1])
        """
        N_actual = len(self.history)
        if N_actual == 0:
            return

        if self.training_mode == "MACRO_ONLY":
            # HybridRiskLoss: risk_score = (1-α)*mse_score + α*corr_score
            alpha = self.RISK_CORR_ALPHA  # 0.7
            MAX_MSE = 1.0
            for hist in self.history:
                # Component scores
                hist.direction_score = hist.direction_f1_macro  # Already in [0, 1]
                # MSE component: normalize and invert (higher = better)
                mse_norm = min(hist.risk_mse / MAX_MSE, 1.0)  # Clamp to [0, 1]
                mse_score = 1.0 - mse_norm
                # Correlation component: clamp negative to 0
                corr_score = max(0.0, hist.risk_correlation)
                # HybridRiskLoss formula
                hist.risk_score = (1.0 - alpha) * mse_score + alpha * corr_score
                # Equal weights
                hist.phase_score = 0.5 * hist.direction_score + 0.5 * hist.risk_score

        elif self.training_mode == "SELECTION_ONLY":
            # phase_score = 0.6 * sharpe_score + 0.4 * hit_rate
            for hist in self.history:
                # Normalize Sharpe to [0, 1]: clamp to [0, 3] then divide by 3
                sharpe_norm = min(max(hist.topk_sharpe_ratio, 0.0), 3.0) / 3.0
                hit_rate = getattr(hist, 'topk_hit_rate', 0.5)
                hist.phase_score = 0.6 * sharpe_norm + 0.4 * hit_rate

        # Update legacy best_ces alias
        self.best_ces = self.best_score

    def _recompute_ces_scores(self):
        """Legacy method - calls _compute_phase_scores for backward compatibility."""
        self._compute_phase_scores()
    
    def _update_best(self, allow_best_update: bool = True) -> bool:
        """
        Update best checkpoint tracker based on phase_score.

        Args:
            allow_best_update: If False, block update even if score is higher.

        Returns:
            True if last added result is new best
        """
        if not self.history:
            return False

        last_idx = len(self.history) - 1
        last_score = self.history[last_idx].phase_score

        # Only update "best" if allowed AND score improved
        if allow_best_update and last_score > self.best_score:
            self.best_score = last_score
            self.best_epoch = last_idx
            # Update legacy alias
            self.best_ces = self.best_score
            return True

        return False
    
    def get_best_checkpoint_info(self) -> Dict:
        """
        Get information about best checkpoint.

        Returns:
            Dict with epoch, phase_score, and phase-relevant metrics
        """
        if self.best_epoch is None or not self.history:
            return {
                "epoch": -1,
                "phase_score": 0.0,
                "ces_score": 0.0,  # Legacy
                "sharpe_ratio": 0.0,
                "direction_f1": 0.0,
                "risk_mse": 999.0,
            }

        best = self.history[self.best_epoch]
        return {
            "epoch": best.epoch,
            "phase_score": best.phase_score,
            "ces_score": best.phase_score,  # Legacy alias
            "sharpe_ratio": best.topk_sharpe_ratio,
            "direction_f1": best.direction_f1_macro,
            "risk_mse": best.risk_mse,
        }
    
    def save_validation_history(self) -> str:
        """
        Save validation metrics to CSV file using atomic write.
        
        Writing strategy:
        1. Write to temp file
        2. Flush and sync
        3. Rename temp file to target file (atomic on POSIX)
        
        Returns:
            Path to saved CSV file
        """
        if not self.history:
            return ""
        
        # Convert all results to dicts
        records = [h.to_dict() for h in self.history]
        df = pd.DataFrame(records)
        
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "valid_metrics.csv")
        temp_path = csv_path + ".tmp"
        
        try:
            # Format epoch as integer
            if "epoch" in df.columns:
                 df["epoch"] = df["epoch"].astype(int)
                 
            # Write to temp file first (force 5 decimal places)
            df.to_csv(temp_path, index=False, float_format='%.5f')
            
            # Atomic rename
            os.replace(temp_path, csv_path)
            

                
        except Exception as e:
            print(f"[TRACKER] Failed to save validation history: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            # Don't crash training on save fail, but warn loudly
            pass
        
        return csv_path
    
    def get_latest_result(self) -> Optional[ObserverValidationResult]:
        """Get most recent validation result."""
        return self.history[-1] if self.history else None

    def load_history_from_csv(self, csv_path: str):
        """
        Load validation history from CSV file.
        
        Args:
            csv_path: Path to valid_metrics.csv
            
        Raises:
            ValueError: If file content is invalid/corrupt (to prevent functional data loss)
            IOError: If file cannot be read
        """
        if not os.path.exists(csv_path):
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"[TRACKER] CSV file exists but is empty, starting fresh: {csv_path}")
                return

            # Check if required columns exist before parsing (at least epoch matches)
            if "epoch" not in df.columns:
                print(f"[TRACKER] CRITICAL: 'epoch' column missing in {csv_path}. Treating as corrupt.")
                # We decide to abort loading to force user intervention or explicit restart
                raise ValueError(f"Corrupt CSV: Missing 'epoch' column in {csv_path}")

            loaded_history = []

            for _, row in df.iterrows():
                # Convert row keys to match ObserverValidationResult fields
                # Filter out any extra keys from CSV that might not match the dataclass fields if any
                data = row.to_dict()
                
                # Ensure types (handle string fields like training_mode)
                STRING_FIELDS = {"training_mode"}
                try:
                    data["epoch"] = int(data["epoch"])
                    for k, v in data.items():
                        if k == "epoch" or k in STRING_FIELDS:
                            continue  # Already handled or string field
                        data[k] = float(v)
                except ValueError as ve:
                     raise ValueError(f"Invalid data type in CSV row: {row}") from ve
                
                # Robustly create object, ignoring unknown fields if schema evolved
                # Filter data to only valid fields of ObserverValidationResult
                valid_fields = ObserverValidationResult.__dataclass_fields__.keys()
                filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                
                # Check for critical missing fields if any (optional)
                
                # Create object
                result = ObserverValidationResult(**filtered_data)
                loaded_history.append(result)
            
            self.history = loaded_history

            # Recompute phase scores
            self._compute_phase_scores()

            # Restore best tracker state
            self.best_score = -np.inf
            self.best_epoch = None
            for idx, res in enumerate(self.history):
                if res.phase_score > self.best_score:
                    self.best_score = res.phase_score
                    self.best_epoch = idx

            # Update legacy alias
            self.best_ces = self.best_score

            if self.best_epoch is not None:
                print(f"[TRACKER] Loaded {len(self.history)} validation records. Best Phase Score: {self.best_score:.4f} (Epoch {self.history[self.best_epoch].epoch})")
            else:
                print(f"[TRACKER] Loaded {len(self.history)} validation records (no valid score found yet).")
            
        except Exception as e:
            print(f"[TRACKER] 🛑 FATAL: Failed to load history from {csv_path}: {e}")
            print(f"[TRACKER] Aborting load to prevent overwriting valid data with empty state.")
            raise  # Re-raise to let caller handle (likely abort)
