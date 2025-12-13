#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Metrics Tracker - Manages validation history and CES computation

This module implements rank-based normalization for CES (Composite Efficiency Score)
computation per Spec §7.1164-1178.

CES Formula:
    CES = 1.0 × S_sharpe + 0.5 × S_dir_f1 + 0.5 × (1 - S_risk_mse)
    
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
    Tracks validation metrics across epochs and computes CES score.
    
    Responsibilities:
    - Accumulate validation results per epoch
    - Perform rank-based normalization for CES computation
    - Identify best checkpoint based on CES score
    - Save validation history to CSV
    
    Attributes:
        output_dir: Directory to save validation metrics CSV
        history: List of all validation results
        best_epoch: Epoch index with highest CES score
        best_ces: Best CES score achieved
    """
    
    # Fixed N for rank-based normalization to ensure stable CES scores
    # Using N=14 provides consistent scale regardless of current epoch count
    CES_RANK_N_FIXED = 14

    def __init__(self, output_dir: str, ces_rank_n: int = None):
        """
        Initialize tracker.

        Args:
            output_dir: Directory to save valid_metrics.csv
            ces_rank_n: Fixed N for CES rank normalization (default: 14)
        """
        self.output_dir = output_dir
        self.history: List[ObserverValidationResult] = []
        self.best_epoch: Optional[int] = None
        self.best_ces: float = -np.inf
        self.ces_rank_n = ces_rank_n if ces_rank_n is not None else self.CES_RANK_N_FIXED
        
    def add_epoch(self, result: ObserverValidationResult) -> bool:
        """
        Add validation result for an epoch and update CES scores.
        If epoch already exists (from resume), replace it instead of duplicating.

        Args:
            result: Validation result to add

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

        self._recompute_ces_scores()
        is_best = self._update_best()
        return is_best
    
    def _recompute_ces_scores(self):
        """
        Recompute CES scores with rank-based normalization per Spec §7.1164-1178.

        Algorithm:
        1. For each metric M (Sharpe, Dir_F1, Risk_MSE):
            - Rank all checkpoints 1..N_actual (worst to best)
            - Normalize: S_hat = (Rank(M) - 1) / (N_fixed - 1)
        2. Compute CES = 1.0 × S_sharpe + 0.5 × S_dir_f1 + 0.5 × (1 - S_risk_mse)

        Note:
        - Risk_MSE is inverted (lower is better)
        - Uses fixed N (default 14) for stable normalization across training
        """
        N_actual = len(self.history)
        if N_actual == 0:
            return

        # Use fixed N for normalization to ensure stable CES scores
        # This prevents early epochs from having inflated/deflated scores
        N_norm = self.ces_rank_n

        # Extract raw metrics
        sharpes = np.array([h.topk_sharpe_ratio for h in self.history])
        dir_f1s = np.array([h.direction_f1_macro for h in self.history])
        risk_mses = np.array([h.risk_mse for h in self.history])

        def avg_rank(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
            """Compute tie-aware average ranks (1=worst) for a 1D array."""
            if higher_is_better:
                sort_order = np.argsort(values)  # ascending => worst to best
            else:
                sort_order = np.argsort(-values)  # invert: higher value = worse

            ranks = np.zeros_like(values, dtype=float)
            i = 0
            N = len(values)
            while i < N:
                j = i
                # group ties
                while j + 1 < N and values[sort_order[j + 1]] == values[sort_order[i]]:
                    j += 1
                # average rank for ties (1-indexed)
                avg = (i + 1 + j + 1) / 2.0
                ranks[sort_order[i:j+1]] = avg
                i = j + 1
            return ranks

        rank_sharpe_all = avg_rank(sharpes, higher_is_better=True)
        rank_dir_all = avg_rank(dir_f1s, higher_is_better=True)
        rank_risk_all = avg_rank(risk_mses, higher_is_better=False)  # lower MSE is better

            # Normalize to [0, 1] using FIXED N for stable scaling
            # Per Spec §7.1210: Ŝ_i = (Rank(M_i) - 1) / (N_fixed - 1)
            # This ensures consistent scale regardless of current epoch count
        for hist, r_s, r_d, r_r in zip(self.history, rank_sharpe_all, rank_dir_all, rank_risk_all):
            hist.ces_rank_sharpe = (r_s - 1) / max(1, N_norm - 1)
            hist.ces_rank_dir_f1 = (r_d - 1) / max(1, N_norm - 1)
            hist.ces_rank_risk_mse = (r_r - 1) / max(1, N_norm - 1)

            # CES formula from Spec §7.1200
            # CES = 1.0 × Ŝ_Sharpe + 0.5 × Ŝ_Dir_F1 + 0.5 × (1 - Ŝ_Risk_MSE)
            # Note: (1 - Ŝ_Risk_MSE) inverts the score since lower MSE is better
            hist.ces_score = (
                1.0 * hist.ces_rank_sharpe +
                0.5 * hist.ces_rank_dir_f1 +
                0.5 * (1 - hist.ces_rank_risk_mse)
            )
    
    def _update_best(self) -> bool:
        """
        Update best checkpoint tracker.
        
        Returns:
            True if last added result is new best
        """
        if not self.history:
            return False
        
        last_idx = len(self.history) - 1
        last_ces = self.history[last_idx].ces_score
        
        if last_ces > self.best_ces:
            self.best_ces = last_ces
            self.best_epoch = last_idx
            return True
        
        return False
    
    def get_best_checkpoint_info(self) -> Dict:
        """
        Get information about best checkpoint.
        
        Returns:
            Dict with epoch, CES score, and component metrics
        """
        if self.best_epoch is None or not self.history:
            return {
                "epoch": -1,
                "ces_score": 0.0,
                "sharpe_ratio": 0.0,
                "direction_f1": 0.0,
                "risk_mse": 999.0,
            }
        
        best = self.history[self.best_epoch]
        return {
            "epoch": best.epoch,
            "ces_score": best.ces_score,
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
            # Write to temp file first
            df.to_csv(temp_path, index=False)
            
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
                
                # Ensure types
                try:
                    data["epoch"] = int(data["epoch"])
                    for k, v in data.items():
                        if k != "epoch":
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
            
            # Recompute global bests
            self._recompute_ces_scores()
            # Restore best tracker state
            self.best_ces = -np.inf
            self.best_epoch = None
            for idx, res in enumerate(self.history):
                if res.ces_score > self.best_ces:
                    self.best_ces = res.ces_score
                    self.best_epoch = idx
            
            if self.best_epoch is not None:
                print(f"[TRACKER] Loaded {len(self.history)} validation records. Best CES: {self.best_ces:.4f} (Epoch {self.history[self.best_epoch].epoch})")
            else:
                print(f"[TRACKER] Loaded {len(self.history)} validation records (no valid CES found yet).")
            
        except Exception as e:
            print(f"[TRACKER] 🛑 FATAL: Failed to load history from {csv_path}: {e}")
            print(f"[TRACKER] Aborting load to prevent overwriting valid data with empty state.")
            raise  # Re-raise to let caller handle (likely abort)
