# ！/usr/bin/python
# -*- coding: utf-8 -*-#

"""
MAFIA (Multi-Agent Fusion and Intelligent Adaptation) Market Observer

This module implements the MAFIA Market Observer, which replaces the original
Market Observer in the MASA framework. MAFIA consists of:
- 1 Technical Agent: Analyzes raw prices + technical indicators
- 3 DC Agents: Analyze Directional Change events at different thresholds

Each agent has CSA, TA, and ST-Fusion modules. Outputs are aggregated to
generate market_vector and risk_eta.
See: Đặc Tả Kỹ Thuật (Technical Specification) - MAFIA.md
"""

import os
import math
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Callable, Any, Optional

# Import HybridRiskLoss for consistent risk loss computation
from RL_controller.observer_offline_trainer import HybridRiskLoss

# NOTE: Disabled for production - only enable for debugging
# th.autograd.set_detect_anomaly(True)
th.autograd.set_detect_anomaly(False)

# LiveDisplay integration for warmup logging
try:
    from utils.display_integration import (
        update_observer_warmup as display_update_observer_warmup,
        get_display,
        smart_print,
    )

    LIVE_DISPLAY_AVAILABLE = True
except ImportError:
    LIVE_DISPLAY_AVAILABLE = False
    display_update_observer_warmup = None
    get_display = lambda: None
    smart_print = print  # Fallback to normal print

# TrainingLogger integration for phase transitions
try:
    from utils.training_logger import get_logger, TrainingPhase

    TRAINING_LOGGER_AVAILABLE = True
except ImportError:
    TRAINING_LOGGER_AVAILABLE = False
    get_logger = None
    TrainingPhase = None


# ============================================================
# WatchlistManager: Manages stock screening and tracking
# ============================================================

class WatchlistManager:
    """
    Watchlist Manager for MAFIA stock screening and tracking.

    Responsibilities:
    1. Screening: Filter universe (122 stocks) to Watchlist (40 stocks)
       - Uses Technical Score + Model Logits
       - ADD/REMOVE with hysteresis thresholds

    2. Watching: Track embeddings of Watchlist stocks daily
       - Buffer of shape (context_window, watchlist_size, D)

    3. Holdings Protection: Protect profitable holdings from removal
       - Holdings removed only when: Tech < floor AND r_hold < -10%
       - Or: Not selected in Top-K during rebalance (ĐK B)

    4. Index Mapping: Map Watchlist indices to Universe indices
    """

    def __init__(self, config, universe_size: int, embedding_dim: int, device: th.device):
        """
        Initialize WatchlistManager.

        Args:
            config: Configuration object with watchlist parameters
            universe_size: Total number of stocks in universe (e.g., 122)
            embedding_dim: Dimension of stock embeddings (D)
            device: Torch device
        """
        self.config = config
        self.universe_size = universe_size
        self.embedding_dim = embedding_dim
        self.device = device

        # Watchlist configuration
        self.watchlist_size = getattr(config, 'watchlist_size', 40)
        self.min_size = getattr(config, 'watchlist_min_size', 35)
        self.max_size = getattr(config, 'watchlist_max_size', 45)
        self.context_window = getattr(config, 'watchlist_context_window', 30)

        # Screening thresholds
        self.tech_add_threshold = getattr(config, 'watchlist_tech_add_threshold', 0.7)
        self.tech_remove_threshold = getattr(config, 'watchlist_tech_remove_threshold', 0.3)
        self.model_add_percentile = getattr(config, 'watchlist_model_add_percentile', 50.0)
        self.model_remove_percentile = getattr(config, 'watchlist_model_remove_percentile', 30.0)

        # Holdings protection
        self.holdings_tech_floor = getattr(config, 'watchlist_holdings_tech_floor', 0.3)
        self.holdings_loss_threshold = getattr(config, 'watchlist_holdings_loss_threshold', -0.10)

        # Screening schedule
        self.screening_period_days = getattr(config, 'watchlist_screening_period_days', 10)

        # Combined Score Weights (Phương án C - Configurable)
        # Score = w_mom × Momentum + w_vol × Volume + w_ma20 × MA20 + w_model × Model
        self.weight_momentum = getattr(config, 'watchlist_weight_momentum', 0.20)
        self.weight_volume = getattr(config, 'watchlist_weight_volume', 0.20)
        self.weight_ma20 = getattr(config, 'watchlist_weight_ma20', 0.20)
        self.weight_model = getattr(config, 'watchlist_weight_model', 0.40)

        # State
        self._watchlist_indices = None  # (watchlist_size,) - indices into universe
        self._screening_scores = None  # (watchlist_size,) - screening scores for weights
        self._portfolio_weights = None  # (watchlist_size,) - allocation weights for Portfolio Context
        self._embeddings_buffer = None  # (context_window, watchlist_size, D)
        self._holdings_mask = None  # (universe_size,) - which stocks are held
        self._entry_prices = None  # (universe_size,) - entry prices for held stocks
        self._days_since_screening = 0
        self._initialized = False
        self._buffer_idx = 0  # FIFO buffer index

        # Model trust: Only use model_scores after sufficient training
        # When False: screening uses Technical Score only (model is random/untrained)
        # When True: screening uses Combined Score (Technical + Model)
        self._use_model_scores = False

    def initialize(self, initial_indices: Optional[np.ndarray] = None):
        """
        Initialize Watchlist with initial indices or cold start.

        Args:
            initial_indices: Optional (watchlist_size,) indices for cold start
        """
        if initial_indices is not None:
            self._watchlist_indices = initial_indices.copy()
        else:
            # Cold start: use first watchlist_size stocks
            self._watchlist_indices = np.arange(min(self.watchlist_size, self.universe_size))

        n_stocks = len(self._watchlist_indices)
        self._screening_scores = np.ones(n_stocks, dtype=np.float32) / n_stocks
        self._portfolio_weights = np.zeros(n_stocks, dtype=np.float32)  # No holdings initially
        self._embeddings_buffer = th.zeros(
            (self.context_window, self.watchlist_size, self.embedding_dim),
            dtype=th.float32,
            device=self.device
        )
        self._holdings_mask = np.zeros(self.universe_size, dtype=bool)
        self._entry_prices = np.zeros(self.universe_size, dtype=np.float32)
        self._days_since_screening = 0
        self._buffer_idx = 0
        self._initialized = True

    def reset(self):
        """Reset manager state (for new training episode)."""
        self._watchlist_indices = None
        self._screening_scores = None
        self._portfolio_weights = None
        self._embeddings_buffer = None
        self._holdings_mask = None
        self._entry_prices = None
        self._days_since_screening = 0
        self._buffer_idx = 0
        self._initialized = False
        # Note: Don't reset _use_model_scores - that persists across episodes

    def set_use_model_scores(self, use_model: bool):
        """
        Set whether to use model scores for screening.

        Call this after model has been trained (e.g., after first iteration).
        Before this is called, screening uses Technical Score only.

        Args:
            use_model: True to use Combined Score (Technical + Model),
                      False for Technical-only screening
        """
        self._use_model_scores = use_model

    def update_portfolio_weights(self, universe_weights: np.ndarray):
        """
        Update portfolio weights for Watchlist stocks.

        Args:
            universe_weights: (universe_size,) allocation weights for all stocks
        """
        if self._watchlist_indices is None:
            return

        # Extract weights for Watchlist stocks only
        self._portfolio_weights = universe_weights[self._watchlist_indices].astype(np.float32)

    def get_portfolio_context_weights(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get holdings mask and portfolio weights relative to Watchlist.

        Returns:
            Tuple of:
                - watchlist_holdings_mask: (watchlist_size,) boolean mask
                - watchlist_portfolio_weights: (watchlist_size,) allocation weights
        """
        if self._watchlist_indices is None:
            return None, None

        # Get holdings mask relative to Watchlist
        watchlist_holdings_mask = self._holdings_mask[self._watchlist_indices]
        watchlist_portfolio_weights = self._portfolio_weights if self._portfolio_weights is not None else np.zeros(len(self._watchlist_indices), dtype=np.float32)

        return watchlist_holdings_mask, watchlist_portfolio_weights

    @property
    def watchlist_indices(self) -> Optional[np.ndarray]:
        """Get current Watchlist indices (universe indices)."""
        return self._watchlist_indices

    @property
    def current_size(self) -> int:
        """Get current Watchlist size."""
        return len(self._watchlist_indices) if self._watchlist_indices is not None else 0

    def get_watchlist_mask(self) -> np.ndarray:
        """Get boolean mask (universe_size,) of stocks in Watchlist."""
        mask = np.zeros(self.universe_size, dtype=bool)
        if self._watchlist_indices is not None:
            mask[self._watchlist_indices] = True
        return mask

    def update_holdings(
        self,
        new_holdings_mask: np.ndarray,
        current_prices: np.ndarray
    ):
        """
        Update holdings information for protection logic.

        Args:
            new_holdings_mask: (universe_size,) boolean mask of current holdings
            current_prices: (universe_size,) current close prices
        """
        if self._holdings_mask is None:
            self._holdings_mask = np.zeros(self.universe_size, dtype=bool)
        if self._entry_prices is None:
            self._entry_prices = np.zeros(self.universe_size, dtype=np.float32)

        # Find new entries (just bought)
        new_entries = new_holdings_mask & (~self._holdings_mask)
        self._entry_prices[new_entries] = current_prices[new_entries]

        # Update holdings mask
        self._holdings_mask = new_holdings_mask.copy()

    def compute_holdings_returns(self, current_prices: np.ndarray) -> np.ndarray:
        """
        Compute returns since entry for held stocks.

        Args:
            current_prices: (universe_size,) current close prices

        Returns:
            holdings_returns: (universe_size,) returns since entry (0 for non-holdings)
        """
        returns = np.zeros(self.universe_size, dtype=np.float32)
        if self._holdings_mask is not None and self._entry_prices is not None:
            held_mask = self._holdings_mask & (self._entry_prices > 0)
            returns[held_mask] = (
                current_prices[held_mask] / self._entry_prices[held_mask] - 1.0
            )
        return returns

    def update_embeddings_buffer(self, embeddings: th.Tensor):
        """
        Update embeddings buffer with current day's Watchlist embeddings.

        Args:
            embeddings: (watchlist_size, D) embeddings for Watchlist stocks
        """
        if self._embeddings_buffer is None:
            return

        # FIFO update
        self._embeddings_buffer[self._buffer_idx] = embeddings.detach()
        self._buffer_idx = (self._buffer_idx + 1) % self.context_window

    def get_watchlist_context(self) -> Optional[th.Tensor]:
        """
        Compute Watchlist Context from embeddings buffer.

        C_watchlist = WeightedSum(embeddings[current_day], screening_scores)

        Returns:
            watchlist_context: (D,) tensor or None if not initialized
        """
        if self._embeddings_buffer is None or self._screening_scores is None:
            return None

        # Get current day embeddings (most recent in buffer)
        current_idx = (self._buffer_idx - 1) % self.context_window
        current_embeddings = self._embeddings_buffer[current_idx]  # (watchlist_size, D)

        # Weighted sum using screening scores as weights
        weights = th.tensor(self._screening_scores, dtype=th.float32, device=self.device)
        weights = weights / (weights.sum() + 1e-8)  # Normalize
        weights = weights.unsqueeze(-1)  # (watchlist_size, 1)

        context = (current_embeddings * weights).sum(dim=0)  # (D,)
        return context

    def _compute_combined_score(
        self,
        technical_scores: np.ndarray,
        model_percentiles: np.ndarray,
        tech_components: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Compute combined score using configurable weights.

        Iteration 0 (model not trusted):
            Score = w_mom × Momentum + w_vol × Volume + w_ma20 × MA20 (normalized)

        Iteration 1+ (model trusted):
            Score = w_mom × Momentum + w_vol × Volume + w_ma20 × MA20 + w_model × Model

        Args:
            technical_scores: (N,) Technical Scores (fallback if components not provided)
            model_percentiles: (N,) Model percentiles [0-100]
            tech_components: Optional dict with 'momentum_score', 'volume_score', 'ma20_score'

        Returns:
            combined_scores: (N,) Combined scores for ranking
        """
        N = len(technical_scores)

        if tech_components is not None:
            # Use individual components with configurable weights
            momentum = tech_components.get('momentum_score', np.full(N, 0.5))
            volume = tech_components.get('volume_score', np.full(N, 0.5))
            ma20 = tech_components.get('ma20_score', np.full(N, 0.5))

            if self._use_model_scores:
                # Iteration 1+: Use all 4 components
                # Score = w_mom × Momentum + w_vol × Volume + w_ma20 × MA20 + w_model × Model
                model_score = model_percentiles / 100.0  # Normalize to [0, 1]
                combined = (
                    self.weight_momentum * momentum +
                    self.weight_volume * volume +
                    self.weight_ma20 * ma20 +
                    self.weight_model * model_score
                )
            else:
                # Iteration 0: Technical only (normalize weights to sum to 1)
                tech_total = self.weight_momentum + self.weight_volume + self.weight_ma20
                if tech_total > 0:
                    combined = (
                        (self.weight_momentum / tech_total) * momentum +
                        (self.weight_volume / tech_total) * volume +
                        (self.weight_ma20 / tech_total) * ma20
                    )
                else:
                    combined = (momentum + volume + ma20) / 3.0
        else:
            # Fallback: use technical_scores directly (backward compatibility)
            if self._use_model_scores:
                tech_weight = self.weight_momentum + self.weight_volume + self.weight_ma20
                model_score = model_percentiles / 100.0
                combined = tech_weight * technical_scores + self.weight_model * model_score
            else:
                combined = technical_scores

        return combined.astype(np.float32)

    def run_screening(
        self,
        technical_scores: np.ndarray,
        model_percentiles: np.ndarray,
        current_prices: np.ndarray,
        ma20_prices: Optional[np.ndarray] = None,
        ma50_prices: Optional[np.ndarray] = None,
        return_20d: Optional[np.ndarray] = None,
        force_review: bool = False,
        tech_components: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Run daily screening to update Watchlist.

        Args:
            technical_scores: (universe_size,) Technical Scores (average of components)
            model_percentiles: (universe_size,) Model percentiles [0-100]
            current_prices: (universe_size,) current prices
            ma20_prices: (universe_size,) SMA(20) prices for uptrend check
            ma50_prices: (universe_size,) SMA(50) prices for emergency removal
            return_20d: (universe_size,) 20-day returns for emergency removal
            force_review: If True, force full Watchlist review
            tech_components: Optional dict with individual components:
                - 'momentum_score': (N,) momentum scores [0-1]
                - 'volume_score': (N,) volume scores [0-1]
                - 'ma20_score': (N,) MA20 scores [0-1]
                If provided, uses configurable weights; otherwise uses technical_scores directly.

        Returns:
            Dict with screening results and changes
        """
        self._days_since_screening += 1
        force_review = force_review or (self._days_since_screening >= self.screening_period_days)

        # Compute combined score using configurable weights
        combined_scores = self._compute_combined_score(
            technical_scores=technical_scores,
            model_percentiles=model_percentiles,
            tech_components=tech_components,
        )

        if not self._initialized:
            # Cold start: select top stocks by combined score
            top_indices = np.argsort(combined_scores)[-self.watchlist_size:]
            self.initialize(top_indices)
            return {
                'action': 'cold_start',
                'added': len(top_indices),
                'removed': 0,
                'new_size': self.current_size,
                'threshold_adjusted': False,
                'model_used': self._use_model_scores,
            }

        # Compute holdings returns for protection
        holdings_returns = self.compute_holdings_returns(current_prices)

        # Create screening decision
        current_watchlist_mask = self.get_watchlist_mask()
        N = len(technical_scores)

        # === DYNAMIC THRESHOLD ADJUSTMENT (Spec line 77-81) ===
        # Adjust thresholds to maintain target size
        # FIX: Always check size constraints, not just during force_review
        effective_add_threshold = self.tech_add_threshold
        effective_remove_threshold = self.tech_remove_threshold
        threshold_adjusted = False
        force_replenish = False

        current_size = self.current_size

        # === FIX: Aggressively relax thresholds when watchlist is too small ===
        if current_size < self.min_size:
            # Calculate how much to relax based on severity
            deficit = self.min_size - current_size
            severity = deficit / self.min_size  # 0 to 1

            # Relax ADD threshold: 0.7 -> 0.5 -> 0.3 based on severity
            relax_amount = 0.1 + (0.3 * severity)  # 0.1 to 0.4
            effective_add_threshold = max(0.3, self.tech_add_threshold - relax_amount)

            # If critically low (< min_size/2), force replenish mode
            if current_size < self.min_size // 2:
                force_replenish = True
                effective_add_threshold = 0.2  # Very aggressive

            threshold_adjusted = True
        elif current_size > self.max_size and force_review:
            # Too many stocks: tighten REMOVE threshold
            effective_remove_threshold = min(0.5, self.tech_remove_threshold + 0.1)
            threshold_adjusted = True

        # === ADD CONDITIONS ===
        # Model not trusted: Technical only (all 3 components pass)
        # Model trusted: Technical + Model must both pass
        tech_pass_add = technical_scores > effective_add_threshold
        if self._use_model_scores:
            model_pass_add = model_percentiles > self.model_add_percentile
            add_mask = tech_pass_add & model_pass_add & (~current_watchlist_mask)
        else:
            # Model not trusted: Technical pass is sufficient
            add_mask = tech_pass_add & (~current_watchlist_mask)

        # === REMOVE CONDITIONS ===
        # Model not trusted: Technical fail triggers removal
        # Model trusted: Technical OR Model fail triggers removal
        tech_fail = technical_scores < effective_remove_threshold
        if self._use_model_scores:
            model_fail = model_percentiles < self.model_remove_percentile
            base_remove = (tech_fail | model_fail) & current_watchlist_mask
        else:
            # Model not trusted: Only Technical fail triggers removal
            base_remove = tech_fail & current_watchlist_mask

        # === HOLDINGS PROTECTION (Spec line 339-341) ===
        # Holdings NOT removed if: Price > MA20 (uptrend) OR r_hold > 0 (profitable)
        # Only remove if: Tech < 0.3 AND r_hold < -10%
        if ma20_prices is not None:
            uptrend_mask = current_prices > ma20_prices
        else:
            uptrend_mask = np.zeros(N, dtype=bool)

        holdings_protected = self._holdings_mask & (
            uptrend_mask |  # Price > MA20 (uptrend)
            (holdings_returns >= 0) |  # r_hold > 0 (profitable)
            (  # NOT (Tech < floor AND loss > threshold)
                ~((technical_scores < self.holdings_tech_floor) &
                  (holdings_returns < self.holdings_loss_threshold))
            )
        )
        remove_mask = base_remove & (~holdings_protected)

        # === EMERGENCY REMOVAL (Spec Quy tắc 4, line 461-465) ===
        # Remove immediately if: Price < SMA(50) AND Return_20d < -20%
        emergency_removed = 0
        if getattr(self.config, 'watchlist_emergency_removal_enabled', False):
            if ma50_prices is not None and return_20d is not None:
                emergency_threshold = getattr(self.config, 'watchlist_emergency_return_threshold', -0.20)
                emergency_mask = (
                    current_watchlist_mask &
                    (current_prices < ma50_prices) &
                    (return_20d < emergency_threshold)
                )
                # Emergency removal overrides holdings protection
                remove_mask = remove_mask | emergency_mask
                emergency_removed = emergency_mask.sum()

        # === APPLY CHANGES ===
        stocks_to_add = np.where(add_mask)[0]
        stocks_to_remove = np.where(remove_mask)[0]

        # Remove stocks (but maintain min_size constraint)
        if len(stocks_to_remove) > 0:
            # FIX: Only remove if we'll stay above min_size
            projected_size = len(self._watchlist_indices) - len(stocks_to_remove)
            if projected_size < self.min_size:
                # Limit removals to stay at min_size
                max_to_remove = max(0, len(self._watchlist_indices) - self.min_size)
                stocks_to_remove = stocks_to_remove[:max_to_remove]

            if len(stocks_to_remove) > 0:
                keep_mask = ~np.isin(self._watchlist_indices, stocks_to_remove)
                self._watchlist_indices = self._watchlist_indices[keep_mask]
                self._screening_scores = self._screening_scores[keep_mask]
                # Also update portfolio weights if tracked
                if self._portfolio_weights is not None:
                    self._portfolio_weights = self._portfolio_weights[keep_mask]

        # === FIX: Force replenish if below min_size ===
        n_added = 0
        current_watchlist_after_remove = len(self._watchlist_indices)

        if force_replenish or current_watchlist_after_remove < self.min_size:
            # Need to add stocks to reach at least min_size
            needed = self.min_size - current_watchlist_after_remove
            available_slots = self.max_size - current_watchlist_after_remove

            if needed > 0 and available_slots > 0:
                # Get all non-watchlist stocks, sorted by combined_scores
                not_in_watchlist = ~current_watchlist_mask
                candidate_indices = np.where(not_in_watchlist)[0]

                if len(candidate_indices) > 0:
                    # Sort by combined score (highest first)
                    add_order = np.argsort(combined_scores[candidate_indices])[::-1]
                    n_to_add = min(max(needed, len(stocks_to_add)), available_slots, len(candidate_indices))
                    new_indices = candidate_indices[add_order[:n_to_add]]

                    self._watchlist_indices = np.concatenate([self._watchlist_indices, new_indices])
                    new_scores = combined_scores[new_indices]
                    self._screening_scores = np.concatenate([self._screening_scores, new_scores])
                    if self._portfolio_weights is not None:
                        self._portfolio_weights = np.concatenate([
                            self._portfolio_weights,
                            np.zeros(len(new_indices), dtype=np.float32)
                        ])
                    n_added = len(new_indices)
        else:
            # Normal add (threshold-based)
            available_slots = self.max_size - len(self._watchlist_indices)
            if len(stocks_to_add) > 0 and available_slots > 0:
                n_add = min(len(stocks_to_add), available_slots)
                # Sort by combined_scores (already computed with configurable weights)
                add_order = np.argsort(combined_scores[stocks_to_add])[::-1][:n_add]
                new_indices = stocks_to_add[add_order]

                self._watchlist_indices = np.concatenate([self._watchlist_indices, new_indices])
                new_scores = technical_scores[new_indices]
                self._screening_scores = np.concatenate([self._screening_scores, new_scores])
                # Add zero portfolio weights for new stocks
                if self._portfolio_weights is not None:
                    self._portfolio_weights = np.concatenate([
                        self._portfolio_weights,
                        np.zeros(len(new_indices), dtype=np.float32)
                    ])
                n_added = len(new_indices)

        # Normalize screening scores
        if len(self._screening_scores) > 0:
            self._screening_scores = self._screening_scores / (self._screening_scores.sum() + 1e-8)

        if force_review:
            self._days_since_screening = 0

        return {
            'action': 'screening',
            'added': n_added,
            'removed': len(stocks_to_remove),
            'emergency_removed': emergency_removed,
            'new_size': self.current_size,
            'holdings_protected': holdings_protected.sum(),
            'threshold_adjusted': threshold_adjusted,
            'effective_add_threshold': effective_add_threshold,
            'effective_remove_threshold': effective_remove_threshold,
            'model_used': self._use_model_scores,
        }

    def post_selection_cleanup(
        self,
        prev_holdings_indices: np.ndarray,
        new_holdings_indices: np.ndarray,
    ) -> int:
        """
        Post-selection cleanup: Remove holdings not selected in new Top-K (ĐK B).

        IMPORTANT: Maintains min_size constraint to prevent watchlist from shrinking too much.
        Only removes stocks if current_size > min_size after removal.

        Args:
            prev_holdings_indices: (K,) universe indices of previous holdings
            new_holdings_indices: (K,) universe indices of new Top-K selection

        Returns:
            Number of stocks removed
        """
        if self._watchlist_indices is None:
            return 0

        # Find holdings that were NOT re-selected
        not_reselected = np.setdiff1d(prev_holdings_indices, new_holdings_indices)

        # Filter to only stocks currently in watchlist
        candidates_to_remove = [idx for idx in not_reselected if idx in self._watchlist_indices]

        if len(candidates_to_remove) == 0:
            return 0

        # === FIX: Maintain min_size constraint ===
        # Calculate how many we can safely remove while staying above min_size
        current_size = len(self._watchlist_indices)
        max_removable = max(0, current_size - self.min_size)

        if max_removable == 0:
            # Already at or below min_size, don't remove anything
            return 0

        # Limit removals to maintain min_size
        n_to_remove = min(len(candidates_to_remove), max_removable)
        stocks_to_remove = candidates_to_remove[:n_to_remove]

        # Remove from Watchlist
        removed = 0
        for idx in stocks_to_remove:
            mask = self._watchlist_indices != idx
            self._watchlist_indices = self._watchlist_indices[mask]
            self._screening_scores = self._screening_scores[mask]
            if self._portfolio_weights is not None:
                self._portfolio_weights = self._portfolio_weights[mask]
            removed += 1

        # Normalize screening scores
        if len(self._screening_scores) > 0:
            self._screening_scores = self._screening_scores / (self._screening_scores.sum() + 1e-8)

        return removed

    def map_to_universe_indices(self, watchlist_indices: np.ndarray) -> np.ndarray:
        """
        Map Watchlist indices (0 to watchlist_size-1) to Universe indices.

        Args:
            watchlist_indices: (K,) indices within Watchlist

        Returns:
            universe_indices: (K,) indices within Universe
        """
        if self._watchlist_indices is None:
            return watchlist_indices  # Fallback: assume already universe indices

        return self._watchlist_indices[watchlist_indices]

    def filter_logits_to_watchlist(self, expert_logits: th.Tensor) -> th.Tensor:
        """
        Filter full universe logits to only Watchlist stocks.

        Args:
            expert_logits: (batch, universe_size) full logits

        Returns:
            watchlist_logits: (batch, watchlist_size) filtered logits
        """
        if self._watchlist_indices is None:
            return expert_logits[:, :self.watchlist_size]

        return expert_logits[:, self._watchlist_indices]

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'watchlist_indices': self._watchlist_indices,
            'screening_scores': self._screening_scores,
            'portfolio_weights': self._portfolio_weights,
            'embeddings_buffer': self._embeddings_buffer.cpu() if self._embeddings_buffer is not None else None,
            'holdings_mask': self._holdings_mask,
            'entry_prices': self._entry_prices,
            'days_since_screening': self._days_since_screening,
            'buffer_idx': self._buffer_idx,
            'initialized': self._initialized,
            'use_model_scores': self._use_model_scores,  # Persist model trust state
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self._watchlist_indices = state.get('watchlist_indices')
        self._screening_scores = state.get('screening_scores')
        self._portfolio_weights = state.get('portfolio_weights')
        buffer = state.get('embeddings_buffer')
        self._embeddings_buffer = buffer.to(self.device) if buffer is not None else None
        self._holdings_mask = state.get('holdings_mask')
        self._entry_prices = state.get('entry_prices')
        self._days_since_screening = state.get('days_since_screening', 0)
        self._buffer_idx = state.get('buffer_idx', 0)
        self._initialized = state.get('initialized', False)
        self._use_model_scores = state.get('use_model_scores', False)  # Restore model trust state


class MAFIAObserver:
    """
    MAFIA Market Observer implementing the same interface as MarketObserver.

    MAFIA consists of:
    - 1 Technical Agent (analyzes raw prices + technical indicators)
    - 3 DC Agents (analyze Directional Change events at different thresholds)

    Each agent has:
    - Cross-Sectional Analysis (CSA) module
    - Temporal Analysis (TA) module
    - Spatial-Temporal Fusion (ST-Fusion) module

    Outputs:
    - market_vector: (batch, N) - Market trend vector for N assets
    - risk_eta: (batch,) - Risk tolerance factor eta
    """

    def __init__(self, config, action_dim):
        """
        Initialize MAFIA Observer.

        Args:
            config: Configuration object with MAFIA hyperparameters
            action_dim: Number of assets (N)
        """
        self.config = config
        self.action_dim = action_dim  # N (number of stocks)
        self.last_topk_indices = None
        self.last_gate_weights = None
        self.last_market_context = None

        # Temporal Context Augmentation buffer (Spec 3.6)
        # FIFO buffer of shape (W, D) storing historical market_context for router augmentation
        self.router_context_window = getattr(config, "router_context_window", 14)
        self.context_buffer = None  # Lazily initialized as (W, D) tensor
        self._context_buffer_initialized = False

        # Price history buffer for explicit signals computation (Spec §3.5)
        # Store recent close prices for Vol_Std20 and DC event detection
        self.price_history_window = 50  # Need ~30+ days for Vol_Std20 + history
        self.price_history_buffer = None  # (history_len,) - market close prices
        self.dc_state = {"mode": "up", "p_ext": None}  # DC algorithm state
        self.dc_threshold = float(getattr(config, "mafia_DC_threshold", 0.02))

        # Validate MAFIA hyperparameters exist
        self._validate_config()

        # Setup device first: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if th.cuda.is_available():
            self.device = th.device("cuda:0")
        elif th.backends.mps.is_available():
            self.device = th.device("mps")
        else:
            self.device = th.device("cpu")

        # Initialize MAFIA model
        from RL_controller.mafia_modules import MAFIAModel

        self.mafia_model = MAFIAModel(config=config, action_dim=action_dim)
        self.mafia_model = self.mafia_model.to(self.device)

        # === SEPARATED TRAINING: Set Train Mode & Filter Optimizer ===
        # Apply mode-specific freezing (Macro vs Selection)
        self.mafia_model.set_train_mode(config.mafia_train_mode)

        # === DIFFERENTIAL LEARNING RATES (Per-Head) ===
        # Create parameter groups with different learning rates
        lr_direction = getattr(config, "mafia_lr_direction_head", 5e-4)
        lr_risk = getattr(config, "mafia_lr_risk_head", 1e-4)
        lr_backbone = getattr(config, "mafia_lr_backbone", 3e-4)

        # Check L2 regularization enable flag (for ablation studies)
        l2_enabled = getattr(config, "macro_enable_l2_regularization", True)
        if l2_enabled:
            weight_decay = config.mafia_weight_decay
            weight_decay_risk = getattr(config, "mafia_weight_decay_risk_head", 0.01)
        else:
            weight_decay = 0.0
            weight_decay_risk = 0.0
            smart_print("[MAFIAObserver] L2 Regularization DISABLED (weight_decay=0)")

        # Initialize HybridRiskLoss for consistent risk loss computation
        # Same as offline trainer for consistency
        risk_corr_alpha = float(getattr(config, "risk_loss_correlation_alpha", 0.7))
        risk_corr_eps = float(getattr(config, "risk_loss_correlation_eps", 1e-8))
        self.risk_criterion = HybridRiskLoss(
            alpha=risk_corr_alpha,
            eps=risk_corr_eps,
            reduction="mean"
        )
        smart_print(
            f"[MAFIAObserver] Risk Loss: HybridRiskLoss(α={risk_corr_alpha}, "
            f"MSE={int((1-risk_corr_alpha)*100)}%, Corr={int(risk_corr_alpha*100)}%)"
        )

        # Access heads via signal_generator (DenseMoESignalGenerator)
        signal_gen = self.mafia_model.signal_generator

        # Collect parameter IDs for each head to avoid duplicates
        direction_head_params = set(id(p) for p in signal_gen.direction_head.parameters())
        risk_head_params = set(id(p) for p in signal_gen.risk_head.parameters())

        # Build parameter groups
        param_groups = []

        # Group 1: Direction Head (higher LR - was learning slow)
        direction_params = [
            p for p in signal_gen.direction_head.parameters()
            if p.requires_grad
        ]
        if direction_params:
            param_groups.append({
                "params": direction_params,
                "lr": lr_direction,
                "name": "direction_head"
            })

        # Group 2: Risk Head (lower LR + higher weight decay - prevent overfit)
        risk_params = [
            p for p in signal_gen.risk_head.parameters()
            if p.requires_grad
        ]
        if risk_params:
            param_groups.append({
                "params": risk_params,
                "lr": lr_risk,
                "weight_decay": weight_decay_risk,  # Higher regularization
                "name": "risk_head"
            })

        # Group 3: Backbone (all other parameters)
        backbone_params = [
            p for p in self.mafia_model.parameters()
            if p.requires_grad and id(p) not in direction_head_params and id(p) not in risk_head_params
        ]
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": lr_backbone,
                "name": "backbone"
            })

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
        )

        # Log parameter group info
        print(f"[MAFIA Observer] Differential LR initialized:")
        print(f"  - Direction Head: lr={lr_direction:.1e} ({len(direction_params)} params)")
        print(f"  - Risk Head:      lr={lr_risk:.1e}, wd={weight_decay_risk:.1e} ({len(risk_params)} params)")
        print(f"  - Backbone:       lr={lr_backbone:.1e} ({len(backbone_params)} params)")

        # === LEARNING RATE SCHEDULE ===
        schedule_mode = getattr(config, "mafia_lr_schedule", "cosine")
        warmup_epochs = getattr(config, "mafia_lr_warmup_epochs", 2)
        min_lr = getattr(config, "mafia_lr_min", 1e-6)
        total_epochs = max(1, config.num_epochs)

        if schedule_mode == "cosine":
            # Cosine Annealing with Warmup (recommended for stable training)
            def _lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup: scale from min_lr to full lr
                    return max(min_lr / lr_backbone, (epoch + 1) / warmup_epochs)
                # Cosine decay after warmup
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr / lr_backbone, cosine_decay)

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
            print(f"  - Schedule: Cosine with {warmup_epochs} warmup epochs, min_lr={min_lr:.1e}")

        elif schedule_mode == "linear":
            # Legacy linear decay
            end_factor = getattr(config, "mafia_lr_end_factor", 0.2)
            frac = getattr(config, "mafia_lr_end_fraction", 0.5)
            decay_epochs = max(1, int(total_epochs * frac))

            def _lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                adj_epoch = epoch - warmup_epochs
                adj_decay = max(1, decay_epochs - warmup_epochs)
                if adj_epoch >= adj_decay:
                    return end_factor
                return 1.0 - (1.0 - end_factor) * (adj_epoch / float(adj_decay))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
            print(f"  - Schedule: Linear decay to {end_factor}x over {frac*100:.0f}% epochs")

        elif schedule_mode == "linear_per_epoch":
            # Repeat linear decay every epoch: use a cycle on scheduler steps
            cycle_steps = max(1, int(getattr(config, "observer_mini_epoch_steps", 1)))
            end_factor = getattr(config, "mafia_lr_end_factor", 0.2)
            frac = getattr(config, "mafia_lr_end_fraction", 0.5)
            decay_steps = max(1, int(cycle_steps * frac))

            def _lr_lambda(step_idx):
                step_in_cycle = step_idx % cycle_steps
                if step_in_cycle >= decay_steps:
                    return end_factor
                return 1.0 + (end_factor - 1.0) * (step_in_cycle / float(decay_steps))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
            print(f"  - Schedule: Linear per-epoch decay")

        else:
            # Step decay (fallback)
            decay_steps = max(1, total_epochs // 3)
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=decay_steps, gamma=0.1
            )
            print(f"  - Schedule: Step decay every {decay_steps} epochs")

        # Training buffers (for Policy Gradient)
        self.market_vector_lst = []  # Store market_vector for each step (with gradient)
        self.risk_eta_lst = []  # Store risk eta for each step (detached for logging)
        self.risk_eta_pred_tensor_lst = []  # Store risk eta tensors (with grad) for supervised loss
        self.risk_eta_target_tensor_lst = []  # Store eta targets (no grad)
        self.rate_of_price_change_lst = []  # Store rate_of_price_change for reward calculation
        self.mkt_direction_lst = []  # Store market direction labels
        self.sigma_log_p_lst = []  # Store log-probabilities for market direction classification
        self.last_sigma_log_p: Optional[th.Tensor] = None
        self.last_sigma_pred: Optional[th.Tensor] = None
        # Buffers for Policy Gradient on Top-K
        self.topk_scores_lst = []  # (batch, K) with grad
        self.topk_indices_lst = []  # (batch, K) int
        self.baseline_return_lst = []  # (batch,) baseline (e.g., market index)
        self.last_action_dim = action_dim
        self.last_pg_turnover = 0.0
        self.last_pg_change = 0.0
        # PG reward components for display
        self.last_pg_mean_return = 0.0  # Mean portfolio return
        self.last_pg_turnover_penalty = 0.0  # α_turnover × turnover
        self.last_pg_change_penalty = 0.0  # α_change × symdiff
        self.last_pg_shaped_return = 0.0  # mean_return - penalties (R_t)

        # === GUMBEL TEMPERATURE ANNEALING ===
        self._gumbel_temp_init = getattr(config, "mafia_gumbel_temperature", 1.0)
        self._gumbel_temp_min = getattr(config, "mafia_gumbel_temp_min", 0.1)
        self._gumbel_temp_decay = getattr(config, "mafia_gumbel_temp_decay", 0.99)
        self._gumbel_temp_inference = getattr(
            config, "mafia_gumbel_temp_inference", 0.1
        )
        self._current_gumbel_temp = self._gumbel_temp_init
        self._current_episode = 0

        # === WARMUP TRACKING ===
        # Track if warmup phase is complete (collected enough samples for first training)
        self._warmup_complete = False

    def update_temperature(self, episode: int) -> float:
        """
        Update Gumbel temperature with decay based on episode.
        Formula: temp = max(temp_min, temp_init * (decay ** episode))

        Args:
            episode: Current episode number

        Returns:
            Updated temperature value
        """
        self._current_episode = episode
        self._current_gumbel_temp = max(
            self._gumbel_temp_min,
            self._gumbel_temp_init * (self._gumbel_temp_decay**episode),
        )
        # Update the model's temperature parameter
        if hasattr(self.mafia_model, "signal_generator") and hasattr(
            self.mafia_model.signal_generator, "tau_gumbel"
        ):
            self.mafia_model.signal_generator.tau_gumbel = self._current_gumbel_temp
        return self._current_gumbel_temp

    def get_temperature(self, training: bool = True) -> float:
        """
        Get appropriate temperature for training or inference.

        Args:
            training: True for training mode (use annealed temp), False for inference (use fixed low temp)

        Returns:
            Temperature value
        """
        if training:
            return self._current_gumbel_temp
        else:
            return self._gumbel_temp_inference

    def enable_feature_caching(self, rawdata, stock_list):
        """
        Enable feature caching for the underlying MAFIAModel.

        This pre-computes technical indicators (SMA, RSI, ATR) for the entire
        time series once, avoiding redundant computation on every forward pass.
        Should be called once during setup with the full rawdata.

        Args:
            rawdata: DataFrame with columns [stock, date, open, high, low, close, volume]
            stock_list: List of stock symbols to pre-compute for
        """
        if hasattr(self, 'mafia_model') and self.mafia_model is not None:
            self.mafia_model.enable_feature_caching(rawdata, stock_list)
        else:
            print("[MAFIAObserver] Warning: mafia_model not initialized, caching not enabled")

    def _validate_config(self):
        """Validate that config has all required MAFIA hyperparameters."""
        required_params = [
            "mafia_T_w",
            "mafia_DC_multipliers",
            "mafia_D",
            "mafia_D_h",
            "mafia_encoder_layers",
            "mafia_encoder_heads",
            "mafia_M_tech",
            "mafia_M_dc",
            "mafia_learning_rate",
            "mafia_weight_decay",
        ]

        for param in required_params:
            if not hasattr(self.config, param):
                raise ValueError(f"Config missing required MAFIA parameter: {param}")

    def _compute_explicit_signals(
        self,
        market_close_price: Optional[float] = None,
        stock_closes: Optional[np.ndarray] = None,
        stock_volumes: Optional[np.ndarray] = None,
        market_volume: Optional[float] = None,
    ) -> th.Tensor:
        """
        Compute explicit signals for Direction Head Wide Path (Spec §3.5.1 v2.1).
        Returns (1, 6) tensor: [Vol_Std20, DC_Event_Flag, Breadth_Gap, Div_Signal, Signed_VPI_Zscore, Drawdown60]

        Args:
            market_close_price: Current market close price (for online inference)
            stock_closes: Current stock close prices (N,) for breadth calculation
            stock_volumes: Current stock volumes (N,) - optional
            market_volume: Current market volume - for VPI calculation

        Returns:
            explicit_signals: (1, 6) tensor with normalized values
        """
        # Initialize buffers on first call
        if self.price_history_buffer is None:
            self.price_history_buffer = []
        if not hasattr(self, "stock_price_history") or self.stock_price_history is None:
            self.stock_price_history = []
        if not hasattr(self, "rsi_history") or self.rsi_history is None:
            self.rsi_history = []
        if not hasattr(self, "volume_history") or self.volume_history is None:
            self.volume_history = []
        if not hasattr(self, "vpi_history") or self.vpi_history is None:
            self.vpi_history = []

        # Update price history
        if market_close_price is not None:
            self.price_history_buffer.append(market_close_price)
            if len(self.price_history_buffer) > self.price_history_window:
                self.price_history_buffer.pop(0)

        # Update stock history
        if stock_closes is not None:
            self.stock_price_history.append(stock_closes.copy())
            if len(self.stock_price_history) > self.price_history_window:
                self.stock_price_history.pop(0)

        # Update volume history
        if market_volume is not None:
            self.volume_history.append(market_volume)
            if len(self.volume_history) > 25:
                self.volume_history.pop(0)

        # Helper: Compute RSI
        def compute_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0  # Neutral
            deltas = np.diff(prices[-period - 1 :])
            gains = np.maximum(deltas, 0)
            losses = np.abs(np.minimum(deltas, 0))
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses) + 1e-8
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        # === 1. DC_Event_Flag: Structural break detection (Adaptive ATR-based) ===
        dc_event_flag = 0.0
        
        # Calculate ATR on buffer (Need at least 15 days)
        current_atr = 0.0
        dc_k = float(getattr(self.config, "mafia_DC_multipliers", [1.0])[1]) # Default k=1.0
        
        if len(self.price_history_buffer) >= 15:
            # Convert buffer to arrays
            p_hist = np.array(self.price_history_buffer[-15:])
            # Ideal ATR needs High/Low, here we approximate with Close differences
            # TR = |C_t - C_{t-1}| (simple approximation for close-only data)
            tr_vals = np.abs(np.diff(p_hist))
            current_atr = np.mean(tr_vals) # Simple MA of TR = ATR approximation
        
        if market_close_price is not None and len(self.price_history_buffer) >= 2:
            if self.dc_state["p_ext"] is None:
                self.dc_state["p_ext"] = self.price_history_buffer[0]
                self.dc_state["mode"] = "up"
            
            # Dynamic Threshold
            adaptive_threshold = dc_k * (current_atr / (market_close_price + 1e-8))
            adaptive_threshold = max(adaptive_threshold, 0.005) # Min 0.5% floor

            p_ext = self.dc_state["p_ext"]
            var = (market_close_price - p_ext) / (p_ext + 1e-8)

            if self.dc_state["mode"] == "up":
                if var < -adaptive_threshold:
                    dc_event_flag = 1.0  # Downward DC (Crash)
                    self.dc_state["mode"] = "down"
                    self.dc_state["p_ext"] = market_close_price
                elif market_close_price > p_ext:
                    self.dc_state["p_ext"] = market_close_price
            else:
                if var > adaptive_threshold:
                    dc_event_flag = 1.0  # Upward DC (Rally)
                    self.dc_state["mode"] = "up"
                    self.dc_state["p_ext"] = market_close_price
                elif market_close_price < p_ext:
                    self.dc_state["p_ext"] = market_close_price

        # === 2. Breadth_Gap: avg(RSI_stocks) - RSI_index ===
        # Detects "Xanh vỏ đỏ lòng" (Index up but stocks weak)
        breadth_gap = 0.0
        rsi_index = 50.0
        if len(self.price_history_buffer) >= 15:
            rsi_index = compute_rsi(self.price_history_buffer)
            self.rsi_history.append(rsi_index)
            if len(self.rsi_history) > 20:
                self.rsi_history.pop(0)

            if len(self.stock_price_history) >= 15 and stock_closes is not None:
                N = len(stock_closes)
                rsi_stocks_sum = 0.0
                for n in range(N):
                    stock_hist = [h[n] for h in self.stock_price_history[-15:]]
                    rsi_stocks_sum += compute_rsi(stock_hist)
                rsi_stocks_avg = rsi_stocks_sum / N
                # Normalize to [-1, 1]
                breadth_gap = (rsi_stocks_avg - rsi_index) / 100.0

        # === 3. Div_Signal: RSI slope vs Price slope divergence ===
        # Signal = -Sign(Price_slope) if divergence detected, else 0
        div_signal = 0.0
        lookback = 5
        if len(self.price_history_buffer) >= lookback + 1 and len(self.rsi_history) >= lookback + 1:
            # Price slope (percentage change over lookback)
            price_slope = (self.price_history_buffer[-1] - self.price_history_buffer[-lookback - 1]) / (
                self.price_history_buffer[-lookback - 1] + 1e-8
            )
            # RSI slope
            rsi_slope = self.rsi_history[-1] - self.rsi_history[-lookback - 1]

            # Divergence: RSI and Price moving in opposite directions
            if np.sign(price_slope) != np.sign(rsi_slope) and abs(rsi_slope) > 5:
                div_signal = -np.sign(price_slope)

        # === 4. Signed_VPI_Zscore: Volume-Price Efficiency Index ===
        # VPI = Sign(ΔP) × |ΔP| / (Vol / Vol_20_avg)
        vpi_zscore = 0.0
        if len(self.price_history_buffer) >= 2 and len(self.volume_history) >= 20 and market_volume is not None:
            # Price change
            delta_p = (self.price_history_buffer[-1] - self.price_history_buffer[-2]) / (
                self.price_history_buffer[-2] + 1e-8
            )

            # Volume ratio
            vol_20_avg = np.mean(self.volume_history[-20:]) + 1e-8
            vol_ratio = market_volume / vol_20_avg

            # VPI = Sign(ΔP) × |ΔP| / Vol_ratio
            vpi = np.sign(delta_p) * abs(delta_p) / (vol_ratio + 1e-8)
            self.vpi_history.append(vpi)
            if len(self.vpi_history) > 25:
                self.vpi_history.pop(0)

            # Z-score normalize
            if len(self.vpi_history) >= 10:
                vpi_mean = np.mean(self.vpi_history)
                vpi_std = np.std(self.vpi_history) + 1e-8
                vpi_zscore = (vpi - vpi_mean) / vpi_std
                vpi_zscore = float(np.clip(vpi_zscore, -3, 3))

        # === 0. Volatility: Relative Volatility (Robustness Fix) ===
        # Replaces raw Vol_Std20 with Vol10 / Vol30
        vol_relative = 1.0 # Default neutral
        if len(self.price_history_buffer) >= 31:
             recent_prices = np.array(self.price_history_buffer[-31:])
             returns = np.diff(recent_prices) / (recent_prices[:-1] + 1e-8)
             
             # Compute Vol10 and Vol30
             vol10 = np.std(returns[-10:])
             vol30 = np.std(returns[-30:])
             
             if vol30 > 1e-8:
                 vol_relative = vol10 / vol30
             
        # === 5. Drawdown60: Max Drawdown over last 60 steps ===
        drawdown60 = 0.0
        dd_lookback = 60
        if len(self.price_history_buffer) >= 2:
            # Use available history up to 60
            window = self.price_history_buffer[-dd_lookback:] if len(self.price_history_buffer) > dd_lookback else self.price_history_buffer
            window_np = np.array(window)
            
            # Calculate running max
            running_max = np.maximum.accumulate(window_np)
            # Calculate drawdown
            dd = (window_np - running_max) / (running_max + 1e-8)
            # Max drawdown is the minimum value (deepest trough)
            drawdown60 = np.min(dd)

        # Create tensor (1, 6) in proper order [Vol, DC, Breadth, Div, VPI, DD]
        explicit_signals = th.tensor(
            [[vol_relative, dc_event_flag, breadth_gap, div_signal, vpi_zscore, drawdown60]],
            dtype=th.float32,
            device=self.device,
        )

        return explicit_signals

    def _remap_legacy_layernorm_keys(self, state_dict: dict) -> dict:
        """
        Remap old checkpoint keys for backward compatibility.

        When StableLayerNorm replaced nn.LayerNorm, the parameter paths changed:
        - Old: input_norm.weight, input_norm.bias
        - New: input_norm.norm.weight, input_norm.norm.bias

        This method remaps old keys to new keys so old checkpoints can still be loaded.
        """
        remapped = {}
        remapped_count = 0

        for key, value in state_dict.items():
            new_key = key
            # Check if this is an old-style input_norm key (without .norm.)
            if ".input_norm.weight" in key and ".input_norm.norm." not in key:
                new_key = key.replace(".input_norm.weight", ".input_norm.norm.weight")
                remapped_count += 1
            elif ".input_norm.bias" in key and ".input_norm.norm." not in key:
                new_key = key.replace(".input_norm.bias", ".input_norm.norm.bias")
                remapped_count += 1

            remapped[new_key] = value

        if remapped_count > 0:
            smart_print(
                f"[MAFIA] Remapped {remapped_count} legacy LayerNorm keys (input_norm -> input_norm.norm)",
                flush=True,
            )

        return remapped

    def _ensure_temp_embeddings_from_state(self, state_dict: dict):
        """
        Materialize temporary layers (embeddings/projections) that may have been created
        on-the-fly during training so checkpoints with those weights can be loaded cleanly.

        - TAModule `_temp_embedding` is created when runtime token dim differs from config.
        - DenseMoEGatingRouter `_temp_proj` is created when market feature dim differs.
        """
        module_map = dict(self.mafia_model.named_modules())

        # Handle TAModule temp embeddings
        temp_prefixes = {
            key.split("._temp_embedding.")[0]
            for key in state_dict.keys()
            if "._temp_embedding." in key
        }
        for prefix in temp_prefixes:
            module = module_map.get(prefix)
            # Only TAModules carry D_h/D attributes; skip anything else
            if module is None or not hasattr(module, "D_h") or not hasattr(module, "D"):
                continue
            # If temp embedding already exists, leave it as-is
            existing = getattr(module, "_temp_embedding", None)
            if isinstance(existing, nn.Module):
                continue

            w1 = state_dict.get(f"{prefix}._temp_embedding.0.weight")
            w2 = state_dict.get(f"{prefix}._temp_embedding.2.weight")
            if w1 is None or w2 is None:
                continue

            in_features = w1.shape[1]
            hidden_dim = w1.shape[0]
            out_features = w2.shape[0]

            module._temp_embedding = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_features),
            ).to(self.device)
            smart_print(
                f"[MAFIA] Recreated temp embedding for {prefix} "
                f"(in={in_features}, hidden={hidden_dim}, out={out_features})",
                flush=True,
            )

        # Handle DenseMoEGatingRouter temp projections (market feature dim mismatch)
        temp_proj_prefixes = {
            key.split("._temp_proj.")[0]
            for key in state_dict.keys()
            if "._temp_proj." in key
        }
        for prefix in temp_proj_prefixes:
            module = module_map.get(prefix)
            if module is None:
                continue
            existing = getattr(module, "_temp_proj", None)
            if isinstance(existing, nn.Module):
                continue

            w = state_dict.get(f"{prefix}._temp_proj.weight")
            b = state_dict.get(f"{prefix}._temp_proj.bias")
            if w is None or b is None:
                continue

            in_features = w.shape[1]
            out_features = w.shape[0]

            module._temp_proj = nn.Linear(in_features, out_features).to(self.device)
            
            # CRITICAL FIX: Copy weights from checkpoint into the newly created layer
            # Without this, the layer has random weights and checkpoint weights are lost
            with th.no_grad():
                module._temp_proj.weight.copy_(w.to(self.device))
                module._temp_proj.bias.copy_(b.to(self.device))
            
            smart_print(
                f"[MAFIA] Recreated temp projection for {prefix} "
                f"(in={in_features}, out={out_features}) - weights loaded from checkpoint",
                flush=True,
            )

    # =========================================================================
    # Temporal Context Augmentation Buffer Management (Spec 3.6)
    # =========================================================================

    def _init_context_buffer(self, D: int, device: th.device):
        """
        Lazily initialize the context buffer with zeros.

        Args:
            D: Embedding dimension (mafia_D)
            device: Target device
        """
        if self._context_buffer_initialized:
            return

        W = self.router_context_window
        self.context_buffer = th.zeros(W, D, device=device, dtype=th.float32)
        self._context_buffer_initialized = True

    def _update_context_buffer(self, market_context: th.Tensor):
        """
        Push new market_context into FIFO buffer.
        Called on EVERY step (regardless of rebalance mask).

        Buffer layout: [oldest, ..., newest]
        FIFO: shift left, append new at end.

        Args:
            market_context: (batch, D) or (D,) tensor - current market context
        """
        # Ensure buffer is initialized
        D = market_context.size(-1)
        device = market_context.device
        self._init_context_buffer(D, device)

        # Handle batch dimension: take first element if batched
        if market_context.dim() == 2:
            context_to_add = market_context[0].detach()
        else:
            context_to_add = market_context.detach()

        # FIFO: drop oldest, append newest
        # buffer[i] = buffer[i+1] for i in [0, W-2], buffer[W-1] = new
        self.context_buffer = th.cat(
            [self.context_buffer[1:], context_to_add.unsqueeze(0)], dim=0
        )

    def _get_context_buffer(self) -> Optional[th.Tensor]:
        """
        Get the context buffer for router temporal augmentation.

        Returns:
            context_buffer: (W, D) tensor or None if not initialized
        """
        if not self._context_buffer_initialized:
            return None
        return self.context_buffer

    def _reset_context_buffer(self):
        """Reset context buffer (e.g., at episode boundaries)."""
        self._context_buffer_initialized = False
        self.context_buffer = None

    def _detach_context_buffer(self):
        """
        Detach all vectors in context_buffer to truncate gradient graph (Spec 3.6.3).

        Purpose: Cut gradient history while preserving values for momentum calculation.
        Called after each timestep during inference/validation.
        """
        if self.context_buffer is not None:
            self.context_buffer = self.context_buffer.detach()

    def predict(
        self,
        finemkt_feat=None,
        finestock_feat=None,
        raw_ochlv_data=None,
        pg_active: bool = True,
        collect_pg: bool = True,
        collect_eta: bool = True,
        force_topk_indices=None,
        holding_alpha_bias=None,
        **kwargs,
    ):
        """
        Predict market_vector and risk_eta using MAFIA model.

        Args:
            raw_ochlv_data: (N, M, T_w) - Direct MAFIA input (required)
            finemkt_feat: Deprecated - kept for interface compatibility
            finestock_feat: Deprecated - kept for interface compatibility
            holding_alpha_bias: (B, N) Smart Holding Bias: alpha × sensitivity × is_held
            **kwargs: Must include 'mode' ('train', 'valid', 'test')

        Returns:
            tuple: (market_vector, risk_eta, market_scores_full, market_context, direction_logits, topk_indices, topk_embeddings, topk_scores)
                - market_vector: (batch, N) numpy array - Top-K weights, zero elsewhere
                - risk_eta: (batch,) numpy array - Risk tolerance factor eta
                - market_scores_full: (batch, N) numpy array - Full market scores (softmax on all N assets, no Top-K mask)
                - market_context: (batch, D) numpy array - Market condition embedding from gating encoder
                - direction_logits: (batch, 3) numpy array - Logits for direction classes
                - topk_indices: (batch, K) numpy array - Indices selected by Gumbel/Top-K
                - topk_embeddings: (batch, K, D) numpy array - Embeddings of selected assets
                - topk_scores: (batch, K) numpy array - Market weights on selected assets
                - market_logits: (batch, N) numpy array - Raw logits (optional)
        """
        mode = kwargs.get("mode", "train")
        pg_active_flag = bool(pg_active)
        collect_pg_flag = bool(collect_pg)
        collect_eta_flag = bool(collect_eta)

        # Validate inputs
        self._validate_input_shapes(finemkt_feat, finestock_feat, **kwargs)

        # Determine input source and prepare OCHLV tensor
        if raw_ochlv_data is not None:
            # Direct MAFIA input
            ochlv_tensor = self._prepare_ochlv_tensor(raw_ochlv_data)
        elif "rawdata_window" in kwargs and kwargs["rawdata_window"] is not None:
            # Extract from rawdata_window if available
            ochlv_tensor = self._prepare_ochlv_tensor(kwargs["rawdata_window"])
        else:
            raise ValueError("raw_ochlv_data or rawdata_window must be provided")

        # Forward pass through MAFIA model
        # ochlv_tensor shape: (batch, N, M, T_w) where M=5 (OCHLV)
        # Need to ensure correct format: (batch, N, 5, T_w)
        if ochlv_tensor.shape[2] != 5:
            # If shape is (batch, N, T_w, 5), transpose to (batch, N, 5, T_w)
            if ochlv_tensor.shape[3] == 5:
                ochlv_tensor = ochlv_tensor.permute(0, 1, 3, 2)

        # Extract market-index OCHLV data if available
        market_index_ochlv_tensor = None
        if (
            "market_index_ochlv_data" in kwargs
            and kwargs["market_index_ochlv_data"] is not None
        ):
            # Direct market-index data provided
            market_index_ochlv_tensor = self._prepare_market_index_ochlv_tensor(
                kwargs["market_index_ochlv_data"]
            )
        elif "env" in kwargs and kwargs["env"] is not None:
            # Try to extract from environment
            env = kwargs["env"]
            if hasattr(env, "_extract_market_index_ochlv_window") and hasattr(
                env, "curData"
            ):
                try:
                    cur_date = (
                        env.curData["date"].iloc[0]
                        if hasattr(env.curData, "iloc")
                        else env.curData["date"][0]
                    )
                    market_index_ochlv_np = env._extract_market_index_ochlv_window(
                        cur_date, window_size=self.config.mafia_T_w
                    )
                    market_index_ochlv_tensor = self._prepare_market_index_ochlv_tensor(
                        market_index_ochlv_np
                    )
                except Exception as e:
                    # If extraction fails, skip market-index agent
                    smart_print(
                        f"Warning: Could not extract market-index OCHLV data: {e}"
                    )
                    market_index_ochlv_tensor = None

        # Get context buffer for temporal augmentation (Spec 3.6)
        router_context_buffer = self._get_context_buffer()

        # Get explicit signals for Direction Head (Spec 3.5 Extended)
        # If not provided, compute from market/stock price history
        explicit_signals = kwargs.get("explicit_signals", None)

        if explicit_signals is None:
            # Extract market close price from market_index_ochlv_tensor
            market_close_price = None
            if market_index_ochlv_tensor is not None:
                # market_index_ochlv_tensor: (batch=1, 1, 5, T_w)
                # Extract the most recent close price (index 1 in OCHLV)
                try:
                    market_close_price = float(
                        market_index_ochlv_tensor[0, 0, 1, -1].cpu().item()
                    )
                except Exception:
                    pass

            # Extract stock closes and volumes from ochlv_tensor
            # ochlv_tensor: (batch, N, 5, T_w) - batch, stocks, features, time
            stock_closes = None
            stock_volumes = None
            try:
                # Get most recent close prices (feature index 1) and volumes (feature index 4)
                stock_closes = ochlv_tensor[0, :, 1, -1].cpu().numpy()  # (N,)
                stock_volumes = ochlv_tensor[0, :, 4, -1].cpu().numpy()  # (N,)
            except Exception:
                pass

            # Compute explicit signals (5 signals)
            explicit_signals = self._compute_explicit_signals(
                market_close_price, stock_closes, stock_volumes
            )

            # If batch size > 1, expand to match
            batch_size = ochlv_tensor.shape[0]
            if batch_size > 1:
                explicit_signals = explicit_signals.expand(batch_size, -1)

        if mode == "train":
            self.mafia_model.train()
            (
                market_vector,
                risk_eta,
                market_scores_full,
                sigma_logits,
                market_context,
                topk_indices,
                topk_embeddings,
                topk_scores,
                market_logits,  # (batch, N) - Raw Logits
            ) = self.mafia_model(
                ochlv_tensor,
                market_index_ochlv_data=market_index_ochlv_tensor,
                force_topk_indices=force_topk_indices,
                router_context_buffer=router_context_buffer,
                explicit_signals=explicit_signals,  # Direction Head signals (Spec 3.5)
                holding_alpha_bias=holding_alpha_bias,  # Smart Holding Bias
                portfolio_state=kwargs.get("portfolio_state", None),  # Portfolio context
            )
        else:
            self.mafia_model.eval()
            with th.no_grad():
                (
                    market_vector,
                    risk_eta,
                    market_scores_full,
                    sigma_logits,
                    market_context,
                    topk_indices,
                    topk_embeddings,
                    topk_scores,
                    market_logits,  # (batch, N)
                ) = self.mafia_model(
                    ochlv_tensor,
                    market_index_ochlv_data=market_index_ochlv_tensor,
                    force_topk_indices=force_topk_indices,
                    router_context_buffer=router_context_buffer,
                    explicit_signals=explicit_signals,  # Direction Head signals (Spec 3.5)
                    holding_alpha_bias=kwargs.get("holding_alpha_bias", None),  # Smart Holding Bias
                    portfolio_state=kwargs.get("portfolio_state", None),  # Portfolio context
                )

        # Update context buffer with new market_context (Spec 3.6)
        # Called on EVERY step, not just rebalance days
        self._update_context_buffer(market_context)

        # Market direction prediction
        sigma_log_p = F.log_softmax(sigma_logits, dim=-1)  # (batch, 3)
        sigma_prob = sigma_log_p.exp()
        sigma_pred = th.argmax(sigma_prob, dim=-1)  # (batch,)
        self.last_sigma_log_p = sigma_log_p if mode == "train" else sigma_log_p.detach()
        self.last_sigma_pred = sigma_pred.detach()
        direction_logits_np = sigma_logits.detach().cpu().numpy()

        stage = kwargs.get("stage", "run")

        # ===== WARMUP-AWARE SAMPLE COLLECTION =====
        # During warmup phase, ALWAYS collect samples regardless of pg_active/collect_pg flags
        # This ensures we accumulate enough samples for the first training update
        warmup_samples = getattr(self.config, "observer_warmup_samples", 300)
        in_warmup = not getattr(self, "_warmup_complete", False)
        current_buffer = len(self.topk_scores_lst)
        force_collect_for_warmup = in_warmup and current_buffer < warmup_samples

        # Collect PG samples if: (normal conditions met) OR (in warmup and need more samples)
        should_collect_pg = (
            mode == "train" and stage == "run" and pg_active_flag and collect_pg_flag
        ) or (mode == "train" and stage == "run" and force_collect_for_warmup)

        if should_collect_pg:
            # Store for training (keep gradient for Policy Gradient)
            self.market_vector_lst.append(market_vector)  # Don't detach - need gradient
            self.topk_scores_lst.append(topk_scores)  # Top-K soft weights (with grad)
            self.topk_indices_lst.append(topk_indices.detach())
            self.last_action_dim = int(market_scores_full.shape[-1])
            self.risk_eta_lst.append(risk_eta.detach())

        # Collect eta samples if: (normal conditions met) OR (in warmup and need more samples)
        should_collect_eta = (
            mode == "train" and stage == "run" and collect_eta_flag
        ) or (mode == "train" and stage == "run" and force_collect_for_warmup)

        if should_collect_eta:
            eta_target_val = self._compute_eta_target_from_env(
                kwargs.get("env"), kwargs.get("current_date")
            )
            if eta_target_val is not None:
                target_tensor = th.full_like(risk_eta, float(eta_target_val))
                self.risk_eta_pred_tensor_lst.append(risk_eta)
                self.risk_eta_target_tensor_lst.append(target_tensor)

        # ===== WARMUP DISPLAY UPDATE (every step during data collection) =====
        # Update warmup progress in real-time as samples are collected
        if mode == "train" and stage == "run":
            warmup_samples = getattr(self.config, "observer_warmup_samples", 300)
            current_buffer = max(
                len(self.topk_scores_lst), len(self.risk_eta_pred_tensor_lst)
            )
            # Only update display during warmup phase (not after completion)
            if current_buffer < warmup_samples and not getattr(
                self, "_warmup_complete", False
            ):
                # Update every 5 samples or at key milestones
                if (
                    current_buffer % 5 == 0
                    or current_buffer == 1
                    or current_buffer >= warmup_samples - 1
                ):
                    # Note: Don't check display.enabled here - the function handles it internally
                    # to allow state updates for dashboard API even when terminal display is disabled
                    if LIVE_DISPLAY_AVAILABLE and get_display():
                        display_update_observer_warmup(
                            buffer_size=current_buffer,
                            target_samples=warmup_samples,
                            warmup_complete=False,
                        )

        # Convert to numpy
        market_vector_np = market_vector.detach().cpu().numpy()
        risk_eta_np = risk_eta.detach().cpu().numpy()
        topk_indices_np = topk_indices.detach().cpu().numpy()
        market_scores_full_np = market_scores_full.detach().cpu().numpy()
        market_context_np = market_context.detach().cpu().numpy()
        topk_scores_np = (
            topk_scores.detach().cpu().numpy() if topk_scores is not None else None
        )
        sigma_val_np = sigma_pred.detach().cpu().numpy()
        sigma_log_p_np = sigma_log_p.detach().cpu().numpy()
        topk_embeddings_np = None
        if topk_embeddings is not None:
            topk_embeddings_np = topk_embeddings.detach().cpu().numpy()
        self.last_topk_indices = topk_indices_np
        try:
            self.last_action_dim = int(market_scores_full.shape[-1])
        except Exception:
            pass

        # Clean NaN/Inf from outputs
        market_vector_np = np.nan_to_num(
            market_vector_np, nan=0.0, posinf=0.0, neginf=0.0
        )
        risk_eta_np = np.nan_to_num(
            risk_eta_np,
            nan=self.config.risk_market,
            posinf=self.config.risk_market,
            neginf=self.config.risk_market,
        )
        market_scores_full_np = np.nan_to_num(
            market_scores_full_np, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Ensure correct shapes
        if market_vector_np.ndim == 1:
            market_vector_np = market_vector_np.reshape(1, -1)
        if risk_eta_np.ndim == 0:
            risk_eta_np = risk_eta_np.reshape(1)
        if market_scores_full_np.ndim == 1:
            market_scores_full_np = market_scores_full_np.reshape(1, -1)

        # Handle size mismatch: pad or truncate market_vector to match action_dim
        actual_N = market_vector_np.shape[1]
        if actual_N != self.action_dim:
            if actual_N < self.action_dim:
                # Pad with zeros
                padding = np.zeros(
                    (market_vector_np.shape[0], self.action_dim - actual_N)
                )
                market_vector_np = np.concatenate([market_vector_np, padding], axis=1)
            else:
                # Truncate
                market_vector_np = market_vector_np[:, : self.action_dim]

        # Handle size mismatch for market_scores_full: pad or truncate to match action_dim
        actual_N_full = market_scores_full_np.shape[1]
        if actual_N_full != self.action_dim:
            if actual_N_full < self.action_dim:
                # Pad with uniform distribution (1/N) to maintain probability distribution
                padding = (
                    np.ones(
                        (
                            market_scores_full_np.shape[0],
                            self.action_dim - actual_N_full,
                        )
                    )
                    / self.action_dim
                )
                market_scores_full_np = np.concatenate(
                    [market_scores_full_np, padding], axis=1
                )
                # Renormalize to ensure sum = 1
                market_scores_full_np = market_scores_full_np / (
                    np.sum(market_scores_full_np, axis=1, keepdims=True) + 1e-8
                )
            else:
                # Truncate
                market_scores_full_np = market_scores_full_np[:, : self.action_dim]
                # Renormalize to ensure sum = 1
                market_scores_full_np = market_scores_full_np / (
                    np.sum(market_scores_full_np, axis=1, keepdims=True) + 1e-8
                )

        # Final validation: if market_vector is all zeros or contains NaN/Inf, use uniform distribution
        if (
            np.any(np.isnan(market_vector_np))
            or np.any(np.isinf(market_vector_np))
            or np.sum(np.abs(market_vector_np)) < 1e-8
        ):
            market_vector_np = (
                np.ones((market_vector_np.shape[0], self.action_dim)) / self.action_dim
            )

        # Final validation for market_scores_full: if all zeros or contains NaN/Inf, use uniform distribution
        if (
            np.any(np.isnan(market_scores_full_np))
            or np.any(np.isinf(market_scores_full_np))
            or np.sum(np.abs(market_scores_full_np)) < 1e-8
        ):
            market_scores_full_np = (
                np.ones((market_scores_full_np.shape[0], self.action_dim))
                / self.action_dim
            )

        # Market context cache
        if (
            hasattr(self, "last_market_context")
            and self.last_market_context is not None
        ):
            last_context_np = self.last_market_context.detach().cpu().numpy()
        else:
            last_context_np = np.zeros((market_vector_np.shape[0], self.config.mafia_D))

        if market_context_np.ndim == 1:
            market_context_np = market_context_np.reshape(1, -1)
        self.last_market_context = market_context.detach()

        # Fallback if NaN/Inf
        market_context_np = np.nan_to_num(
            market_context_np, nan=0.0, posinf=0.0, neginf=0.0
        )
        if market_context_np.shape[1] != self.config.mafia_D:
            market_context_np = last_context_np

        # Top-K embeddings (for state)
        if topk_embeddings_np is None:
            topk_embeddings_np = np.zeros(
                (
                    market_vector_np.shape[0],
                    self.config.mafia_top_k,
                    self.config.mafia_D,
                ),
                dtype=np.float32,
            )
        else:
            topk_embeddings_np = np.nan_to_num(
                topk_embeddings_np, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Pad/truncate to K
            if topk_embeddings_np.shape[1] < self.config.mafia_top_k:
                pad_k = self.config.mafia_top_k - topk_embeddings_np.shape[1]
                pad = np.zeros(
                    (topk_embeddings_np.shape[0], pad_k, self.config.mafia_D)
                )
                topk_embeddings_np = np.concatenate([topk_embeddings_np, pad], axis=1)
            elif topk_embeddings_np.shape[1] > self.config.mafia_top_k:
                topk_embeddings_np = topk_embeddings_np[:, : self.config.mafia_top_k, :]

        # Market direction outputs
        if sigma_val_np.ndim == 0:
            sigma_val_np = sigma_val_np.reshape(1)
        sigma_val_np = np.nan_to_num(
            sigma_val_np, nan=1.0, posinf=1.0, neginf=1.0
        ).astype(np.int64)
        sigma_log_p_np = np.nan_to_num(sigma_log_p_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Validate output shapes
        self._validate_output_shapes(market_vector_np, risk_eta_np)

        # Update config with latest predictions for LiveDisplay
        if hasattr(self, "config"):
            try:
                # Eta prediction
                self.config.last_mafia_eta_pred = float(risk_eta_np.flatten()[0])
                # Direction prediction
                dir_idx = int(sigma_val_np.flatten()[0])
                dir_names = ["BEAR", "FLAT", "BULL"]
                self.config.last_mafia_dir_pred = (
                    dir_names[dir_idx] if 0 <= dir_idx <= 2 else "FLAT"
                )
                # Direction confidence (probability of predicted class)
                sigma_prob_np = np.exp(sigma_log_p_np)
                if sigma_prob_np.ndim > 1:
                    sigma_prob_np = sigma_prob_np[0]
                self.config.last_mafia_dir_conf = (
                    float(sigma_prob_np[dir_idx])
                    if 0 <= dir_idx < len(sigma_prob_np)
                    else 0.0
                )
            except Exception:
                pass

        return (
            market_vector_np,
            risk_eta_np,
            market_scores_full_np,
            market_context_np,
            direction_logits_np,
            topk_indices_np,
            topk_embeddings_np,
            topk_scores_np,
            market_logits.detach().cpu().numpy(),
        )

    def _prepare_ochlv_tensor(self, raw_ochlv_data):
        """
        Prepare raw OCHLV data as tensor.

        Args:
            raw_ochlv_data: (N, M, T_w) or (B, N, M, T_w) numpy array

        Returns:
            torch.Tensor: (B, N, M, T_w) tensor on device
        """
        ochlv_tensor = th.from_numpy(raw_ochlv_data).to(th.float32)
        
        # Add batch dim only if missing (single sample input)
        if ochlv_tensor.dim() == 3:
            ochlv_tensor = ochlv_tensor.unsqueeze(0)  # (1, N, M, T_w)
            
        ochlv_tensor = ochlv_tensor.to(self.device)
        return ochlv_tensor

    def _prepare_market_index_ochlv_tensor(self, market_index_ochlv_data):
        """
        Prepare market-index OCHLV data as tensor.

        Args:
            market_index_ochlv_data: (1, 5, T_w) numpy array

        Returns:
            torch.Tensor: (1, 1, 5, T_w) tensor on device
        """
        mkt_ochlv_tensor = th.from_numpy(market_index_ochlv_data).to(th.float32)
        
        # Add batch dim only if missing (single sample input)
        if mkt_ochlv_tensor.dim() == 3:
            mkt_ochlv_tensor = mkt_ochlv_tensor.unsqueeze(0)  # (1, 1, 5, T_w)
            
        mkt_ochlv_tensor = mkt_ochlv_tensor.to(self.device)
        return mkt_ochlv_tensor

    def _validate_input_shapes(self, finemkt_feat, finestock_feat, **kwargs):
        """Validate input shapes match expected format.

        Note: finemkt_feat and finestock_feat are deprecated but kept for compatibility.
        Use raw_ochlv_data or rawdata_window instead.
        """
        # Deprecated parameter warnings
        if finestock_feat is not None:
            smart_print(
                "[MAFIA] Warning: finestock_feat is deprecated, use raw_ochlv_data"
            )
        if finemkt_feat is not None:
            smart_print("[MAFIA] Warning: finemkt_feat is deprecated")

        # Validate mode
        assert "mode" in kwargs, "kwargs must include 'mode'"
        assert kwargs["mode"] in ["train", "valid", "test"], (
            f"Invalid mode: {kwargs['mode']}"
        )

    def _validate_output_shapes(self, market_vector, risk_eta):
        """Validate output shapes match MarketObserver format."""
        assert market_vector.shape[1] == self.action_dim, (
            f"market_vector shape mismatch: {market_vector.shape[1]} != {self.action_dim}"
        )
        assert market_vector.ndim == 2, (
            f"market_vector must be 2D, got {market_vector.ndim}D"
        )
        assert risk_eta.ndim == 1, f"risk_eta must be 1D, got {risk_eta.ndim}D"
        assert market_vector.shape[0] == risk_eta.shape[0], "Batch sizes must match"

    def _compute_eta_target_from_env(self, env, current_date):
        """
        Compute supervised eta target using future price window (lookahead).

        Args:
            env: Trading environment (provides fine_market data)
            current_date: Current trading date

        Returns:
            float or None: eta_target in [1-λ, 1+λ] or None if insufficient data.

        Fallback Strategy:
            When lookahead data is insufficient, fallback to direction-based eta:
            - Bull (2): eta = 1 + λ (aggressive, higher risk tolerance)
            - Bear (0): eta = 1 - λ (defensive, lower risk tolerance)
            - Side (1): eta = 1.0 (neutral)
        """
        lam = float(getattr(self.config, "risk_eta_lambda", 0.3))

        if env is None or not hasattr(env, "extra_data"):
            return self._fallback_eta_from_direction(lam)
        extra_data = getattr(env, "extra_data", None)
        if extra_data is None or "fine_market" not in extra_data:
            return self._fallback_eta_from_direction(lam)
        fine_market = extra_data["fine_market"]
        if fine_market is None or len(fine_market) == 0:
            return self._fallback_eta_from_direction(lam)
        if "date" not in fine_market.columns:
            return self._fallback_eta_from_direction(lam)

        close_col = f"mkt_{getattr(self.config, 'finefreq', '1d')}_close"
        if close_col not in fine_market.columns:
            if "close" in fine_market.columns:
                close_col = "close"
            else:
                return self._fallback_eta_from_direction(lam)

        # Ensure chronological order and integer index
        fm_sorted = fine_market.sort_values("date").reset_index(drop=True)
        mask = fm_sorted["date"] == current_date
        idx_arr = np.flatnonzero(mask.to_numpy())
        if len(idx_arr) == 0:
            return self._fallback_eta_from_direction(lam)
        idx = int(idx_arr[-1])

        lookback = int(getattr(self.config, "risk_eta_label_lookback", 0))
        if idx < lookback:
            return self._fallback_eta_from_direction(lam)

        lookahead = int(getattr(self.config, "risk_eta_lookahead", 14))
        future_slice = fm_sorted.iloc[idx + 1 : idx + 1 + lookahead]
        if len(future_slice) < lookahead:
            # Insufficient lookahead data - fallback to direction-based eta
            return self._fallback_eta_from_direction(lam)

        try:
            current_price = float(fm_sorted.iloc[idx][close_col])
            future_prices = future_slice[close_col].astype(float).to_numpy()
        except Exception:
            return self._fallback_eta_from_direction(lam)

        if not np.isfinite(current_price) or not np.all(np.isfinite(future_prices)):
            return self._fallback_eta_from_direction(lam)

        mu_fut = float(np.mean(future_prices))
        std_fut = float(np.std(future_prices))
        eps = float(getattr(self.config, "risk_eta_label_epsilon", 1e-6))
        if std_fut < eps:
            std_fut = eps
        # Z_future = (mean_future - current_price) / (std_future + eps)
        z_score = (mu_fut - current_price) / (std_fut + eps)

        kappa = float(getattr(self.config, "risk_eta_sensitivity", 1.0))
        term_val = lam * np.tanh(kappa * z_score)

        # MaxDD Penalty (spec 5.1.2 - Hybrid Target)
        # MaxDD_fut = max_{tau} ( (max(P_{t..tau}) - P_{tau}) / max(P...) )
        lambda_dd = float(getattr(self.config, "risk_eta_lambda_dd", 0.5))
        dd_ref = float(getattr(self.config, "risk_eta_dd_ref", 0.10))

        rolling_max = np.maximum.accumulate(future_prices)
        drawdowns = (rolling_max - future_prices) / (rolling_max + 1e-8)
        max_dd_fut = float(drawdowns.max())
        term_dd = lambda_dd * np.clip(max_dd_fut / dd_ref, 0.0, 1.0)

        # Hybrid formula: eta = 1.0 + valuation_term - drawdown_penalty
        eta_target = 1.0 + term_val - term_dd
        eta_target = float(np.clip(eta_target, 1.0 - lam, 1.0 + lam))
        return eta_target

    def _fallback_eta_from_direction(self, lam: float) -> float:
        """
        Fallback eta target based on most recent predicted market direction.

        Uses last_sigma_pred from Observer's direction head:
        - Bull (2): eta = 1 + λ (aggressive)
        - Bear (0): eta = 1 - λ (defensive)
        - Side (1): eta = 1.0 (neutral)

        Args:
            lam: Lambda parameter for eta range [1-λ, 1+λ]

        Returns:
            float: eta_target based on predicted direction
        """
        if self.last_sigma_pred is None:
            # No prediction available yet, return neutral
            return 1.0

        try:
            # last_sigma_pred is a tensor, get the scalar value
            direction = (
                int(self.last_sigma_pred.item())
                if hasattr(self.last_sigma_pred, "item")
                else int(self.last_sigma_pred)
            )
        except Exception:
            return 1.0

        # Map direction to eta target
        # Bear (0): defensive -> lower eta
        # Side (1): neutral -> eta = 1.0
        # Bull (2): aggressive -> higher eta
        if direction == 0:  # Bear
            return float(1.0 - lam)
        elif direction == 2:  # Bull
            return float(1.0 + lam)
        else:  # Side (1) or unknown
            return 1.0

    def _compute_direction_label_from_env(self, env, current_date):
        """
        Compute future-based direction label using Dynamic Threshold + Path Dependency (Spec §5.1.3).

        Labeling Rules:
            Bear (0): R_fut < -δ_t  OR  IntraDD < StopLoss_limit
            Side (1): |R_fut| <= δ_t  AND  IntraDD >= StopLoss_limit
            Bull (2): R_fut > δ_t  AND  IntraDD >= StopLoss_limit

        Dynamic Threshold: δ_t = max(δ_min, k_atr * ATR / P_t)
        IntraDD = min(Low_{t+1..t+k}) / P_t - 1

        Fallback Strategy (when lookahead data is insufficient):
            - Use most recent predicted direction (last_sigma_pred) from previous step
            - This ensures consistency: Direction Head already predicted from macro context at t-1
        """
        if env is None or not hasattr(env, "extra_data"):
            return None
        extra_data = getattr(env, "extra_data", None)
        if extra_data is None or "fine_market" not in extra_data:
            return None
        fine_market = extra_data["fine_market"]
        if (
            fine_market is None
            or len(fine_market) == 0
            or "date" not in fine_market.columns
        ):
            return None

        # Column names based on finefreq
        finefreq = getattr(self.config, "finefreq", "1d")
        close_col = f"mkt_{finefreq}_close"
        high_col = f"mkt_{finefreq}_high"
        low_col = f"mkt_{finefreq}_low"

        # Fallback column names
        if close_col not in fine_market.columns:
            close_col = "close" if "close" in fine_market.columns else None
        if high_col not in fine_market.columns:
            high_col = "high" if "high" in fine_market.columns else None
        if low_col not in fine_market.columns:
            low_col = "low" if "low" in fine_market.columns else None

        if close_col is None:
            return None

        fm_sorted = fine_market.sort_values("date").reset_index(drop=True)
        mask = fm_sorted["date"] == current_date
        idx_arr = np.flatnonzero(mask.to_numpy())
        if len(idx_arr) == 0:
            return None
        idx = int(idx_arr[-1])

        # Config parameters (Spec §5.1.3)
        lookahead = int(getattr(self.config, "direction_label_lookahead", 21))
        atr_period = int(getattr(self.config, "direction_label_atr_period", 21))
        k_atr = float(getattr(self.config, "direction_label_atr_multiplier", 1.5))
        delta_min = float(getattr(self.config, "direction_label_delta_min", 0.015))
        stop_loss = float(getattr(self.config, "direction_label_stop_loss", -0.07))

        future_idx = idx + lookahead

        # Check if lookahead data is available
        if future_idx >= len(fm_sorted):
            # Fallback: Use most recent predicted direction from Direction Head
            if self.last_sigma_pred is not None:
                return int(self.last_sigma_pred.item())
            else:
                return None

        try:
            price_now = float(fm_sorted.iloc[idx][close_col])
            price_future = float(fm_sorted.iloc[future_idx][close_col])
        except Exception:
            return None

        if not np.isfinite(price_now) or not np.isfinite(price_future) or price_now <= 0:
            return None

        # Compute R_fut
        r_fut = (price_future - price_now) / price_now

        # Compute Dynamic Threshold δ_t = max(δ_min, k_atr * ATR / P_t)
        delta_t = delta_min  # Default fallback
        if high_col is not None and low_col is not None:
            try:
                # Get ATR window data
                atr_start = max(0, idx - atr_period)
                high_arr = fm_sorted.iloc[atr_start : idx + 1][high_col].astype(float).to_numpy()
                low_arr = fm_sorted.iloc[atr_start : idx + 1][low_col].astype(float).to_numpy()
                close_arr = fm_sorted.iloc[atr_start : idx + 1][close_col].astype(float).to_numpy()

                if len(high_arr) >= 2:
                    # True Range: TR = max(H-L, |H-Cp|, |L-Cp|)
                    tr1 = high_arr[1:] - low_arr[1:]
                    tr2 = np.abs(high_arr[1:] - close_arr[:-1])
                    tr3 = np.abs(low_arr[1:] - close_arr[:-1])
                    tr = np.maximum(np.maximum(tr1, tr2), tr3)
                    current_atr = float(np.mean(tr)) if len(tr) > 0 else 0.0

                    # Dynamic threshold
                    atr_ratio = current_atr / price_now if price_now > 0 else 0.0
                    delta_t = max(delta_min, k_atr * atr_ratio)
            except Exception:
                pass  # Keep delta_t = delta_min

        # Compute IntraDD = min(Low_{t+1..t+k}) / P_t - 1
        intra_dd = 0.0  # Default: no drawdown
        if low_col is not None:
            try:
                future_lows = fm_sorted.iloc[idx + 1 : future_idx + 1][low_col].astype(float).to_numpy()
                if len(future_lows) > 0:
                    min_low = float(np.min(future_lows))
                    if np.isfinite(min_low) and price_now > 0:
                        intra_dd = (min_low / price_now) - 1.0
            except Exception:
                pass  # Keep intra_dd = 0.0

        # Labeling Rules (Spec §5.1.3)
        # Bear (0): R_fut < -δ_t OR IntraDD < StopLoss
        if r_fut < -delta_t or intra_dd < stop_loss:
            return 0
        # Bull (2): R_fut > δ_t AND IntraDD >= StopLoss
        elif r_fut > delta_t and intra_dd >= stop_loss:
            return 2
        # Side (1): Otherwise
        else:
            return 1

    def train(self, **label_kwargs):
        """
        Train MAFIA using Policy Gradient method with Loss Masking & Training Cadence Alignment.

        NOTE: This method implements ONLINE training mode (streaming with pre-computed embeddings).
        For OFFLINE BATCH training per Spec §6/§7 (random t_s with hidden state reset),
        use ObserverOfflineBatchTrainer class instead.
        See: docs/OBSERVER_OFFLINE_BATCH_TRAINING.md

        Compatible with MarketObserver.train() interface.

        Called at end of episode.

        Args:
            **label_kwargs: May contain:
                - mode: 'train', 'valid', 'test'
                - ori_profit: Original profit rates
                - adj_profit: Adjusted profit rates
                - ori_risk: Original risk values
                - adj_risk: Adjusted risk values
                - topk_selection_mask: Binary mask (T,) or (T, B) indicating rebalance steps
                  0 = holding period (mask PG loss), 1 = rebalance (full loss)
        """
        # ===== TRAINING MODE GUARD: Skip if Observer is frozen =====
        allow_training = getattr(self.config, "mafia_allow_observer_training", True)
        if not allow_training:
            # Phase 2: Observer is frozen as Static Expert - skip all gradient updates
            if not getattr(self, "_observer_frozen_logged", False):
                smart_print(
                    "[OBSERVER] ❄️ Observer FROZEN (Phase 2: RL-Only mode) - Skipping gradient updates"
                )
                self._observer_frozen_logged = True
            return False

        has_pg_data = (
            len(self.topk_scores_lst) > 0
            and len(self.topk_indices_lst) > 0
            and len(self.rate_of_price_change_lst) > 0
            and len(self.baseline_return_lst) > 0
        )
        has_eta_labels = (
            len(self.risk_eta_pred_tensor_lst) > 0
            and len(self.risk_eta_target_tensor_lst) > 0
        )

        if not has_pg_data and not has_eta_labels:
            # No data collected, skip training
            return False

        # ===== WARMUP CHECK: Collect minimum samples before first training =====
        skip_warmup = getattr(self.config, "skip_observer_warmup", False)
        warmup_samples = (
            0 if skip_warmup else getattr(self.config, "observer_warmup_samples", 300)
        )
        buffer_size = max(
            len(self.topk_scores_lst) if has_pg_data else 0,
            len(self.risk_eta_pred_tensor_lst) if has_eta_labels else 0,
        )

        # Finetune mode: mark warmup complete immediately
        if skip_warmup and not getattr(self, "_warmup_complete", False):
            self._warmup_complete = True

        if buffer_size < warmup_samples:
            # Still in warmup phase - collect more samples before training
            # Update display every 10 steps or at specific milestones
            if buffer_size % 10 == 0 or buffer_size == 1:
                # Note: Don't check display.enabled - the function handles it internally
                # to allow state updates for dashboard API even when terminal display is disabled
                if LIVE_DISPLAY_AVAILABLE and get_display():
                    display_update_observer_warmup(
                        buffer_size=buffer_size,
                        target_samples=warmup_samples,
                        warmup_complete=False,
                    )
                elif buffer_size % 50 == 0 or buffer_size == 1:
                    # Fallback to smart_print when display update is not available
                    smart_print(
                        f"[OBSERVER-WARMUP] Collecting samples: {buffer_size}/{warmup_samples} "
                        f"(need {warmup_samples - buffer_size} more)"
                    )
            return False  # Skip training, continue collecting

        # Log warmup completion (only once)
        if not getattr(self, "_warmup_complete", False):
            # Note: Don't check display.enabled - the function handles it internally
            # to allow state updates for dashboard API even when terminal display is disabled
            if LIVE_DISPLAY_AVAILABLE and get_display():
                display_update_observer_warmup(
                    buffer_size=buffer_size,
                    target_samples=warmup_samples,
                    warmup_complete=True,
                )
            # Log phase transition: warmup → observer training
            if TRAINING_LOGGER_AVAILABLE and get_logger is not None:
                logger = get_logger()
                if logger is not None:
                    logger.print_phase_transition(
                        TrainingPhase.WARMUP,
                        TrainingPhase.PRETRAIN,
                        f"Observer buffer đạt {buffer_size} samples. Bắt đầu gradient updates.",
                    )
            else:
                smart_print(
                    f"[OBSERVER-WARMUP] ✅ Warmup complete! Buffer has {buffer_size} samples. Starting training.",
                    flush=True,
                )
            self._warmup_complete = True

        self.mafia_model.train()
        mode = label_kwargs.get("mode", "train")

        # ===== Sequence sampling config (contiguous mini-batch) =====
        # observer_seq_len / mafia_trajectory_length > 0 enables random contiguous slices
        #
        # SPEC COMPLIANCE NOTE (refactor_mafia.md §6 / §3.6.3):
        # ─────────────────────────────────────────────────────────────────────────
        # Spec requires "Reset hidden_state at the start of each Batch when t_s is
        # randomly sampled". In the current ONLINE streaming design, data is collected
        # sequentially during env.step() calls, with hidden_state maintained throughout
        # the episode. When _pick_seq_range() selects a random slice [start, end) from
        # the buffer, the tensors in that slice were computed with hidden_state context
        # from earlier steps (not freshly reset).
        #
        # IMPLICATIONS:
        # - Random slicing uses pre-computed embeddings with "warm" hidden state
        # - For strict spec compliance, set observer_seq_len=0 to use full trajectory
        # - Or use mafia_sampling_strategy="recent_trajectory" for tail sampling
        #
        # FUTURE ENHANCEMENT: Add offline batch mode that recomputes embeddings with
        # fresh hidden_state for each random t_s (requires significant refactoring).
        # ─────────────────────────────────────────────────────────────────────────
        seq_len_cfg = int(
            getattr(self.config, "observer_seq_len", None)
            or getattr(self.config, "mafia_trajectory_length", 0)
            or 0
        )
        rebalance_interval = int(
            getattr(
                self.config,
                "mafia_topk_rebalance_interval",
                getattr(self.config, "rebalance_interval", 15),
            )
            or 0
        )
        if seq_len_cfg > 0 and rebalance_interval > 0:
            min_seq = 2 * rebalance_interval
            if seq_len_cfg < min_seq:
                seq_len_cfg = min_seq
        sampling_cfg = str(
            getattr(
                self.config,
                "observer_seq_sampling",
                getattr(self.config, "mafia_sampling_strategy", "random"),
            )
        ).lower()
        if "recent" in sampling_cfg or "tail" in sampling_cfg:
            seq_sampling_mode = "recent"
        else:
            seq_sampling_mode = "random"

        def _pick_seq_range(total_steps: int):
            """Pick contiguous range [start, end) given total_steps and config."""
            if seq_len_cfg <= 0 or total_steps <= seq_len_cfg:
                return None
            max_start = total_steps - seq_len_cfg
            if max_start <= 0:
                return None
            if seq_sampling_mode in ("recent", "last", "tail"):
                start_idx = max_start
            else:
                start_idx = int(np.random.randint(0, max_start + 1))
            end_idx = start_idx + seq_len_cfg
            return start_idx, end_idx

        def _slice_tensor_range(tensor, seq_range):
            """Slice tensor on first dim using seq_range."""
            if tensor is None or seq_range is None:
                return tensor
            start_idx, end_idx = seq_range
            if tensor.ndim == 0:
                return tensor
            end_idx = min(end_idx, tensor.shape[0])
            start_idx = min(start_idx, end_idx)
            return tensor[start_idx:end_idx]

        # ===== NEW: Extract cadence mask for Loss Masking =====
        # Cadence signal from environment: 0 = holding, 1 = rebalance/regime
        selection_mask = label_kwargs.get("topk_selection_mask", None)
        if selection_mask is None:
            # Default: all days are rebalance days (full training)
            if has_pg_data:
                T = len(self.topk_scores_lst)
                selection_mask = th.ones(T, device=self.device, dtype=th.float32)
            else:
                T = len(self.risk_eta_pred_tensor_lst)
                selection_mask = th.ones(T, device=self.device, dtype=th.float32)
        else:
            # Convert to tensor if needed
            if not isinstance(selection_mask, th.Tensor):
                selection_mask = th.tensor(
                    selection_mask, device=self.device, dtype=th.float32
                )
            else:
                selection_mask = selection_mask.to(self.device).float()

        # ===== SPEC-COMPLIANT LOSS ACCUMULATION (Section 5.2) =====
        # L_total = (1/T_m) × Σ_t [m_t × λ_pg × L_PG(t) + λ_risk × L_Risk(t) + λ_dir × L_Dir(t)]
        # We accumulate per-timestep losses in lists, then aggregate at the end
        pg_losses_per_step = []  # Per-timestep L_PG (sparse, will be masked)
        pg_losses = []  # Will hold final stacked tensor for aggregation
        risk_losses_per_step = []  # Per-timestep L_Risk (dense)
        dir_losses_per_step = []  # Per-timestep L_Dir (dense)

        loss_market_vector = None
        mkt_direction_tensor = None
        loss_dict = {}  # Track masked losses for logging
        seq_range_global = None

        if has_pg_data:
            eps = 1e-8
            horizon = int(max(1, getattr(self.config, "mafia_pg_reward_horizon", 1)))
            # Stack sequences: (T, B, ...)
            topk_scores_stack = th.stack(self.topk_scores_lst, dim=0)  # (T, B, K)
            topk_indices_stack = th.stack(self.topk_indices_lst, dim=0)  # (T, B, K)
            rate_stack = th.stack(self.rate_of_price_change_lst, dim=0)  # (T, B, N)
            baseline_stack = th.stack(self.baseline_return_lst, dim=0)  # (T, B)
            T, B, K = topk_scores_stack.shape

            # ===== Apply contiguous mini-batch sampling (random start) =====
            seq_range = _pick_seq_range(T)
            if seq_range is None and seq_len_cfg > 0:
                seq_range = (0, T)  # Config enabled but buffer shorter than seq_len
            if seq_range is not None:
                # Log warning if random slice doesn't start from 0 (hidden state context issue)
                start_idx, end_idx = seq_range
                if start_idx > 0 and not getattr(
                    self, "_random_slice_warning_logged", False
                ):
                    smart_print(
                        f"[OBSERVER] ⚠️ Random trajectory slice [{start_idx}:{end_idx}] selected from buffer.\n"
                        f"   Note: Hidden state was NOT reset at t_s={start_idx} per Spec §6.\n"
                        f"   For strict compliance: set observer_seq_len=0 (use full trajectory)."
                    )
                    self._random_slice_warning_logged = True
                topk_scores_stack = _slice_tensor_range(topk_scores_stack, seq_range)
                topk_indices_stack = _slice_tensor_range(topk_indices_stack, seq_range)
                rate_stack = _slice_tensor_range(rate_stack, seq_range)
                baseline_stack = _slice_tensor_range(baseline_stack, seq_range)
                selection_mask = _slice_tensor_range(selection_mask, seq_range)
                T, B, K = topk_scores_stack.shape
            seq_range_global = seq_range

            # Precompute turnover and membership change per t (B,)
            def _compute_penalties(topk_scores_stack, topk_indices_stack):
                turnovers = []
                symdiffs = []
                T, B, K = topk_scores_stack.shape
                device = topk_scores_stack.device
                for t in range(T):
                    if t == 0:
                        turnovers.append(
                            th.zeros(B, device=device, dtype=topk_scores_stack.dtype)
                        )
                        symdiffs.append(
                            th.zeros(B, device=device, dtype=topk_scores_stack.dtype)
                        )
                        continue
                    prev_idx = topk_indices_stack[t - 1]
                    prev_w = topk_scores_stack[t - 1]
                    curr_idx = topk_indices_stack[t]
                    curr_w = topk_scores_stack[t]
                    turnover_b = th.zeros(
                        B, device=device, dtype=topk_scores_stack.dtype
                    )
                    symdiff_b = th.zeros(
                        B, device=device, dtype=topk_scores_stack.dtype
                    )
                    for b in range(B):
                        p_idx = prev_idx[b].tolist()
                        c_idx = curr_idx[b].tolist()
                        p_w = prev_w[b].tolist()
                        c_w = curr_w[b].tolist()
                        # Map index -> weight, default 0
                        w_prev = {int(i): float(w) for i, w in zip(p_idx, p_w)}
                        w_curr = {int(i): float(w) for i, w in zip(c_idx, c_w)}
                        all_keys = set(w_prev.keys()) | set(w_curr.keys())
                        turn = 0.0
                        for k_idx in all_keys:
                            turn += abs(w_curr.get(k_idx, 0.0) - w_prev.get(k_idx, 0.0))
                        turnover_b[b] = turn
                        # Symmetric difference normalized by K
                        sym = len(set(p_idx).symmetric_difference(set(c_idx))) / float(
                            max(1, K)
                        )
                        symdiff_b[b] = sym
                    turnovers.append(turnover_b)
                    symdiffs.append(symdiff_b)
                return th.stack(turnovers, dim=0), th.stack(symdiffs, dim=0)

            turnover_stack, symdiff_stack = _compute_penalties(
                topk_scores_stack, topk_indices_stack
            )  # (T, B)
            alpha_turnover = float(
                getattr(self.config, "mafia_pg_alpha_turnover", 0.08)
            )
            alpha_change = float(getattr(self.config, "mafia_pg_alpha_change", 0.08))

            pg_losses = []
            for t in range(T):
                # Future window [t, t+horizon) (cap to T)
                end_t = min(T, t + horizon)
                if end_t <= t:
                    continue
                rate_slice = rate_stack[t:end_t]  # (h, B, N)
                base_slice = baseline_stack[t:end_t]  # (h, B)

                # Pad rate_slice if indices exceed N
                topk_idx = topk_indices_stack[t]  # (B, K)
                max_idx = int(topk_idx.max().item()) if topk_idx.numel() > 0 else -1
                if max_idx >= rate_slice.shape[2]:
                    pad_size = max_idx - rate_slice.shape[2] + 1
                    pad = th.ones(
                        rate_slice.shape[0],
                        rate_slice.shape[1],
                        pad_size,
                        device=rate_slice.device,
                        dtype=rate_slice.dtype,
                    )
                    rate_slice = th.cat([rate_slice, pad], dim=2)

                # Gather selected asset price ratios across horizon: (h, B, K)
                gather_idx = topk_idx.unsqueeze(0).expand(rate_slice.shape[0], -1, -1)
                selected_prices = th.gather(rate_slice, 2, gather_idx)
                # Compound over horizon to approximate P_{t+T}/P_t
                compounded = th.prod(selected_prices, dim=0)  # (B, K)
                topk_returns = compounded - 1.0  # (B, K)
                mean_return = th.mean(topk_returns, dim=1)  # (B,)

                # Baseline: compound market returns over same horizon
                baseline_comp = th.prod(1.0 + base_slice, dim=0) - 1.0  # (B,)
                # Reward shaping with turnover/membership penalties at decision step t
                shaped_return = (
                    mean_return
                    - alpha_turnover * turnover_stack[t]
                    - alpha_change * symdiff_stack[t]
                )
                advantage = shaped_return - baseline_comp  # (B,)

                log_probs = th.log(topk_scores_stack[t] + eps).sum(dim=1)  # (B,)
                step_loss = -(
                    log_probs * advantage.detach()
                )  # stop grad through advantage
                # Store per-step loss for each batch element (B,)
                pg_losses_per_step.append(step_loss)

            if pg_losses_per_step:
                # Stack to get (T, B) tensor - we need per-timestep losses for masking
                pg_tensor = th.stack(pg_losses_per_step, dim=0)  # (T, B)

                # ===== Apply cadence masking to PG loss =====
                # selection_mask is (T,), expand to (T, B) to match pg_tensor
                if selection_mask.dim() == 1 and selection_mask.shape[0] == T:
                    mask_expanded = selection_mask.unsqueeze(-1).expand(T, B)  # (T, B)
                else:
                    # Fallback if mask shape doesn't match
                    mask_expanded = selection_mask.mean() * th.ones_like(pg_tensor)

                # Apply mask: 0 = holding (zero out PG), 1 = rebalance (keep PG)
                # Store raw and masked versions for logging
                loss_dict["loss_pg_raw"] = float(pg_tensor.mean().detach().cpu().item())
                loss_dict["selection_mask_mean"] = float(selection_mask.mean().item())

                # Don't average yet - we'll aggregate all losses together per spec
                pg_losses.append(pg_tensor)  # Store for final aggregation

                # Log latest penalties and reward components for monitoring/display
                try:
                    self.last_pg_turnover = float(
                        th.mean(turnover_stack).detach().cpu().item()
                    )
                    self.last_pg_change = float(
                        th.mean(symdiff_stack).detach().cpu().item()
                    )
                    # Store PG reward components for LiveDisplay
                    # R_t = mean_return - α_turnover × turnover - α_change × symdiff
                    self.last_pg_mean_return = float(
                        th.mean(mean_return).detach().cpu().item()
                    )
                    self.last_pg_turnover_penalty = (
                        alpha_turnover * self.last_pg_turnover
                    )
                    self.last_pg_change_penalty = alpha_change * self.last_pg_change
                    self.last_pg_shaped_return = (
                        self.last_pg_mean_return
                        - self.last_pg_turnover_penalty
                        - self.last_pg_change_penalty
                    )
                except Exception:
                    self.last_pg_turnover = 0.0
                    self.last_pg_change = 0.0
                    self.last_pg_mean_return = 0.0
                    self.last_pg_turnover_penalty = 0.0
                    self.last_pg_change_penalty = 0.0
                    self.last_pg_shaped_return = 0.0
            # Direction labels (if collected)
            if len(self.mkt_direction_lst) > 0:
                mkt_direction_tensor = th.cat(self.mkt_direction_lst, dim=0).long()

        # If PG data not present or seq_range not set, determine seq_range from other signals
        if seq_range_global is None and seq_len_cfg > 0:
            base_len = None
            if has_eta_labels:
                base_len = len(self.risk_eta_pred_tensor_lst)
            elif len(self.sigma_log_p_lst) > 0:
                base_len = len(self.sigma_log_p_lst)
            if base_len:
                seq_range_global = _pick_seq_range(base_len)
                if seq_range_global is None:
                    seq_range_global = (0, base_len)

        # Build direction labels tensor (apply sequence slice if needed)
        if mkt_direction_tensor is None and len(self.mkt_direction_lst) > 0:
            mkt_direction_tensor = th.cat(self.mkt_direction_lst, dim=0).long()
        if mkt_direction_tensor is not None and seq_range_global is not None:
            mkt_direction_tensor = _slice_tensor_range(
                mkt_direction_tensor, seq_range_global
            )

        # ===== NEW: Risk & Direction losses ALWAYS ACTIVE (no cadence masking) =====
        # These components provide daily adaptive signals regardless of selection trigger
        # Build per-timestep loss tensors for Risk and Direction to match spec aggregation

        # Supervised eta regression loss (MSE between predicted eta and engineered target)
        eta_loss = None
        eta_mae = None
        if (
            len(self.risk_eta_pred_tensor_lst) > 0
            and len(self.risk_eta_target_tensor_lst) > 0
        ):
            paired_len = min(
                len(self.risk_eta_pred_tensor_lst), len(self.risk_eta_target_tensor_lst)
            )
            pred_eta = th.cat(self.risk_eta_pred_tensor_lst[:paired_len], dim=0)
            target_eta = th.cat(self.risk_eta_target_tensor_lst[:paired_len], dim=0).to(
                pred_eta.device
            )
            if seq_range_global is not None:
                pred_eta = _slice_tensor_range(pred_eta, seq_range_global)
                target_eta = _slice_tensor_range(target_eta, seq_range_global)

            # Compute HybridRiskLoss (MSE + Correlation) on full batch
            # This provides better pattern learning and reduces overfitting
            eta_loss = self.risk_criterion(pred_eta, target_eta)  # scalar
            # Store as (1,) tensor for compatibility with aggregation logic
            risk_losses_per_step.append(eta_loss.unsqueeze(0))

            # For logging/monitoring
            eta_mae = F.l1_loss(pred_eta, target_eta)
            loss_dict["loss_risk"] = float(eta_loss.detach().cpu().item())

        # Optional market direction classification loss
        # Direction Head learns market regimes daily, independent of selection cadence
        direction_loss = None
        if (
            mkt_direction_tensor is not None
            and len(self.sigma_log_p_lst) > 0
            and len(self.mkt_direction_lst) > 0
        ):
            sigma_log_p_tensor = th.cat(self.sigma_log_p_lst, dim=0)  # (total_steps, 3)
            if seq_range_global is not None:
                sigma_log_p_tensor = _slice_tensor_range(
                    sigma_log_p_tensor, seq_range_global
                )
            min_len = min(sigma_log_p_tensor.shape[0], mkt_direction_tensor.shape[0])
            if min_len > 0:
                sigma_log_p_tensor = sigma_log_p_tensor[:min_len]
                direction_labels = mkt_direction_tensor[:min_len]

                # Compute per-sample NLL loss (don't reduce yet)
                direction_loss_per_sample = F.nll_loss(
                    sigma_log_p_tensor, direction_labels, reduction="none"
                )  # (T_dir,)
                dir_losses_per_step.append(direction_loss_per_sample)

                # For logging/monitoring, compute aggregated loss
                direction_loss = direction_loss_per_sample.mean()
                loss_dict["loss_direction"] = float(
                    direction_loss.detach().cpu().item()
                )

        # ===== SPEC-COMPLIANT AGGREGATION (Section 5.2) =====
        # L_total = λ_pg × (Σ m_t,b × L_PG / Σ m_t,b) + λ_risk × (Σ L_Risk / B×T_m) + λ_dir × (Σ L_Dir / B×T_m)
        # Each loss component has different normalization strategy!

        # Get hyperparameters
        weight_pg = float(getattr(self.config, "hidden_vec_loss_weight", 1.0))
        eta_weight = float(getattr(self.config, "scale_factor_risk", 10.0))
        dir_weight = float(getattr(self.config, "market_direction_loss_weight", 1.0))

        # Determine T_m (trajectory length) and B (batch size)
        T_m = None
        B = 1  # Default batch size
        if pg_losses and len(pg_losses) > 0:
            T_m, B = pg_losses[0].shape
        elif risk_losses_per_step and len(risk_losses_per_step) > 0:
            T_m = risk_losses_per_step[0].shape[0]
        elif dir_losses_per_step and len(dir_losses_per_step) > 0:
            T_m = dir_losses_per_step[0].shape[0]

        if T_m is None or T_m == 0:
            # No losses to compute - return early
            smart_print("[OBSERVER] ⚠️ No losses to compute (empty buffers)")
            return False

        # Initialize total loss
        total_loss = th.tensor(0.0, device=self.device)
        eps = 1e-8  # Prevent division by zero

        # === SEPARATED TRAINING: LOSS MASKING LOGIC ===
        from config import MafiaTrainMode # Import Enum for comparison

        # 1. PG Loss (Selection)
        # Only active if NOT in MACRO_ONLY mode
        if self.config.mafia_train_mode != MafiaTrainMode.MACRO_ONLY:
            if pg_losses and len(pg_losses) > 0 and len(selection_mask) > 0:
                pg_tensor = pg_losses[0]  # (T, B) - already stacked
                # Ensure mask matches (T, B)
                if selection_mask.dim() == 1:
                   cadence_mask = selection_mask.unsqueeze(-1).expand_as(pg_tensor)
                else:
                   cadence_mask = selection_mask 
                
                # Apply Mask: L_PG_masked = (L_PG * mask) / (sum(mask) + eps)
                masked_pg_loss = pg_tensor * cadence_mask
                
                # Normalize by effective number of rebalancing events
                mask_sum = cadence_mask.sum()
                pg_loss_term = masked_pg_loss.sum() / (mask_sum + eps)
                
                total_loss += weight_pg * pg_loss_term
        
        # 2. Risk Loss (Macro)
        # Only active if NOT in SELECTION_ONLY mode
        if self.config.mafia_train_mode != MafiaTrainMode.SELECTION_ONLY:
            if risk_losses_per_step and len(risk_losses_per_step) > 0:
                risk_tensor = risk_losses_per_step[0] # (T_m, ...)
                # Normalize by total samples (B*T_m)
                risk_loss_term = risk_tensor.mean() 
                total_loss += eta_weight * risk_loss_term

        # 3. Direction Loss (Macro)
        # Only active if NOT in SELECTION_ONLY mode
        if self.config.mafia_train_mode != MafiaTrainMode.SELECTION_ONLY:
            if dir_losses_per_step and len(dir_losses_per_step) > 0:
                dir_tensor = dir_losses_per_step[0]
                dir_loss_term = dir_tensor.mean()
                total_loss += dir_weight * dir_loss_term

        # 1. PG Loss (Sparse - normalized by EVENT COUNT)
        # L_PG_term = λ_pg × (Σ_{b,t} m_{t,b} × L_PG^{(t,b)}) / (Σ_{b,t} m_{t,b} + ε)
        if pg_losses and len(pg_losses) > 0:
            pg_tensor = pg_losses[0]  # (T, B)
            T_pg, B_pg = pg_tensor.shape

            # Expand mask to (T, B) if needed
            if selection_mask.dim() == 1 and selection_mask.shape[0] == T_pg:
                mask_expanded = selection_mask.unsqueeze(-1).expand(
                    T_pg, B_pg
                )  # (T, B)
            else:
                mask_expanded = selection_mask.mean() * th.ones_like(pg_tensor)

            # Apply mask: m_{t,b} × L_PG^{(t,b)}
            pg_masked = mask_expanded * pg_tensor  # (T, B)
            pg_sum = pg_masked.sum()  # Sum over all (b,t)
            mask_sum = mask_expanded.sum()  # Count rebalance events

            # Normalize by event count
            if mask_sum.item() > eps:
                pg_loss_normalized = pg_sum / (mask_sum + eps)
                total_loss = total_loss + weight_pg * pg_loss_normalized
                loss_dict["loss_pg_masked"] = float(
                    pg_loss_normalized.detach().cpu().item()
                )
            else:
                # No rebalance events - PG loss is zero
                loss_dict["loss_pg_masked"] = 0.0

        # 2. Risk Loss (Dense - normalized by B × T_m)
        # L_Risk_term = λ_risk × (Σ_{b,t} L_Risk^{(t,b)}) / (B × T_m)
        if risk_losses_per_step and len(risk_losses_per_step) > 0:
            risk_tensor = risk_losses_per_step[0]  # (T_eta,) - already per sample
            T_eta = risk_tensor.shape[0]

            # Note: risk_tensor is (T_eta,) which may differ from T_m if only eta samples collected
            # We normalize by actual sample count
            risk_sum = risk_tensor.sum()
            # If risk is per-timestep without batch, normalize by T_eta
            # If risk has implicit batch dim, we'd need B × T_eta
            # Based on code structure, risk is (T,) so normalize by T
            risk_loss_normalized = risk_sum / T_eta
            total_loss = total_loss + eta_weight * risk_loss_normalized

        # 3. Direction Loss (Dense - normalized by B × T_m)
        # L_Dir_term = λ_dir × (Σ_{b,t} L_Dir^{(t,b)}) / (B × T_m)
        if dir_losses_per_step and len(dir_losses_per_step) > 0:
            dir_tensor = dir_losses_per_step[0]  # (T_dir,)
            T_dir = dir_tensor.shape[0]

            # Similar to risk, direction is (T,) so normalize by T
            dir_sum = dir_tensor.sum()
            dir_loss_normalized = dir_sum / T_dir
            total_loss = total_loss + dir_weight * dir_loss_normalized

        # ===== Backward pass =====
        # NOTE: Post-backward gradient masking was REMOVED because:
        # 1. Loss masking (pg_tensor * mask) already prevents gradient flow for holding days
        # 2. The old logic was buggy: if ANY holding day existed, it zeroed ALL Selection gradients
        #    even for rebalance days, causing loss of valid training signal
        # See: pg_tensor_masked = pg_tensor * mask_expanded (line ~1062)
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        th.nn.utils.clip_grad_norm_(self.mafia_model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.lr_scheduler.step()

        # Persist latest losses on config for downstream logging/plots
        if hasattr(self, "config"):
            try:
                self.config.last_mafia_loss = float(total_loss.detach().cpu().item())
            except Exception:
                self.config.last_mafia_loss = None
            if eta_loss is not None:
                try:
                    self.config.last_mafia_eta_loss = float(
                        eta_loss.detach().cpu().item()
                    )
                except Exception:
                    self.config.last_mafia_eta_loss = None
                try:
                    self.config.last_mafia_eta_mae = (
                        float(eta_mae.detach().cpu().item())
                        if eta_mae is not None
                        else None
                    )
                except Exception:
                    self.config.last_mafia_eta_mae = None
            else:
                self.config.last_mafia_eta_loss = None
                self.config.last_mafia_eta_mae = None
            if direction_loss is not None:
                try:
                    self.config.last_mafia_direction_loss = float(
                        direction_loss.detach().cpu().item()
                    )
                except Exception:
                    self.config.last_mafia_direction_loss = None
            else:
                self.config.last_mafia_direction_loss = None

            # PG/Selection loss (from loss_dict)
            pg_loss_val = loss_dict.get("loss_pg_masked", None)
            if pg_loss_val is None:
                pg_loss_val = loss_dict.get("loss_pg_raw", None)
            if pg_loss_val is not None:
                try:
                    self.config.last_mafia_pg_loss = float(pg_loss_val)
                except Exception:
                    self.config.last_mafia_pg_loss = None
            else:
                self.config.last_mafia_pg_loss = None

        # Logging
        if th.cuda.is_available():
            th.cuda.synchronize()

        # ===== ENHANCED LOGGING: Descriptive loss breakdown by task =====
        total_loss_val = total_loss.detach().cpu().item()

        # Get loss component values
        loss_pg_raw = loss_dict.get("loss_pg_raw", None)
        loss_pg_masked = loss_dict.get("loss_pg_masked", None)
        selection_mask_mean = loss_dict.get("selection_mask_mean", None)
        eta_loss_val = eta_loss.detach().cpu().item() if eta_loss is not None else None
        eta_mae_val = eta_mae.detach().cpu().item() if eta_mae is not None else None
        direction_loss_val = (
            direction_loss.detach().cpu().item() if direction_loss is not None else None
        )
        masked_params = loss_dict.get("selection_params_masked", None)

        # Get TD3 info
        td3_actor = getattr(self.config, "last_td3_actor_loss", None)
        td3_critic = getattr(self.config, "last_td3_critic_loss", None)
        td3_reward = getattr(self.config, "last_td3_mean_reward", None)
        td3_updates = getattr(self.config, "last_td3_updates", None)

        # Format helper
        def fmt(val, decimals=6):
            return f"{val:.{decimals}f}" if val is not None else "N/A"

        # Compute effective loss contribution from Stock Selection
        # L_selection_effective = L_pg_raw × cadence_ratio (masked contribution to total loss)
        cadence_ratio = selection_mask_mean if selection_mask_mean is not None else 1.0
        rebalance_pct = cadence_ratio * 100 if cadence_ratio is not None else 0.0
        topk_interval = getattr(self.config, "mafia_topk_rebalance_interval", 10)

        # Count samples collected this epoch
        n_direction_samples = len(self.mkt_direction_lst)
        n_eta_samples = len(self.risk_eta_pred_tensor_lst)
        n_selection_samples = len(self.topk_scores_lst)

        # Clear realtime progress line and print epoch summary
        smart_print(
            f"\r{' ' * 100}\r"  # Clear realtime line
            f"\n{'─' * 100}\n"
            f"📊 [OBSERVER TRAINING] Epoch Update - Mode: {mode.upper()}\n"
            f"{'─' * 100}\n"
            f"  📈 L_total: {fmt(total_loss_val)}\n"
            f"  📦 Samples collected: Direction={n_direction_samples} | Risk η={n_eta_samples} | Selection={n_selection_samples}\n"
            f"\n  🎯 LOSS COMPONENTS (theo spec refactor_mafia.md):\n"
            f"    ┌─────────────────────────────────────────────────────────────────────────┐\n"
            f"    │ 1️⃣  L_selection (Stock Selection - Policy Gradient)                     │\n"
            f"    │     Mục tiêu: Chọn Top-K cổ phiếu tối ưu hóa risk-adjusted return       │\n"
            f"    ├─────────────────────────────────────────────────────────────────────────┤\n"
            f"    │  • L_pg:                  {fmt(loss_pg_masked):>12}  ← backprop qua rebalance days │\n"
            f"    │  • Rebalance ratio:       {rebalance_pct:>10.1f}%   ← % steps có refresh Top-K     │\n"
            f"    │  • Top-K refresh: mỗi {topk_interval} ngày (hoặc khi regime shift)              │\n"
            f"    └─────────────────────────────────────────────────────────────────────────┘\n"
            f"    ┌─────────────────────────────────────────────────────────────────────────┐\n"
            f"    │ 2️⃣  L_direction (Market Direction - Cross-Entropy)                      │\n"
            f"    │     Mục tiêu: Phân loại xu hướng thị trường (Bull/Side/Bear)           │\n"
            f"    ├─────────────────────────────────────────────────────────────────────────┤\n"
            f"    │  • L_direction:           {fmt(direction_loss_val):>12}  ← train DAILY (không mask)  │\n"
            f"    └─────────────────────────────────────────────────────────────────────────┘\n"
            f"    ┌─────────────────────────────────────────────────────────────────────────┐\n"
            f"    │ 3️⃣  L_risk (Risk Tolerance η - Hybrid MSE+Corr)                         │\n"
            f"    │     Mục tiêu: Dự đoán η ∈ [0.7, 1.3] để scale σ_target = σ_base × η    │\n"
            f"    ├─────────────────────────────────────────────────────────────────────────┤\n"
            f"    │  • L_risk (Hybrid):       {fmt(eta_loss_val):>12}  ← train DAILY (không mask)  │\n"
            f"    │  • MAE (|η_pred - η_true|): {fmt(eta_mae_val):>10}  ← prediction error         │\n"
            f"    └─────────────────────────────────────────────────────────────────────────┘\n",
            flush=True,
        )

        # TD3 info if available (spec: Porfolio_allocator_spec.md)
        if td3_actor is not None or td3_critic is not None:
            smart_print(
                f"    ┌─────────────────────────────────────────────────────────────────────────┐\n"
                f"    │ 🤖 TD3 PORTFOLIO ALLOCATOR (spec: Porfolio_allocator_spec.md)           │\n"
                f"    │     Mục tiêu: Phân bổ trọng số tối ưu trên Top-K từ Observer            │\n"
                f"    ├─────────────────────────────────────────────────────────────────────────┤\n"
                f"    │  • Gradient updates:      {td3_updates if td3_updates else 'N/A':>12}                              │\n"
                f"    │  • L_actor (Policy):      {fmt(td3_actor):>12}  ← maximize Q(s, π(s))       │\n"
                f"    │  • L_critic (Q-Network):  {fmt(td3_critic):>12}  ← minimize TD error        │\n"
                f"    │  • Mean reward (buffer):  {fmt(td3_reward):>12}                              │\n"
                f"    └─────────────────────────────────────────────────────────────────────────┘\n",
                flush=True,
            )

        # Cadence masking insight
        if masked_params and rebalance_pct < 100.0:
            holding_pct = 100.0 - rebalance_pct
            smart_print(
                f"    ┌─────────────────────────────────────────────────────────────────────────┐\n"
                f"    │ ⏰ TRAINING CADENCE ALIGNMENT                                           │\n"
                f"    ├─────────────────────────────────────────────────────────────────────────┤\n"
                f"    │  • Holding days (mask=0): {holding_pct:>10.1f}%   ← L_selection gradient = 0   │\n"
                f"    │  • Params frozen:         {masked_params:>12}   ← Selection head frozen       │\n"
                f"    │  • L_direction, L_risk:   train DAILY  ← không bị mask                  │\n"
                f"    └─────────────────────────────────────────────────────────────────────────┘\n",
                flush=True,
            )

        smart_print(f"{'─' * 100}", flush=True)

        # Reset buffers
        self.reset()

        if th.cuda.is_available():
            th.cuda.empty_cache()

        return True  # Training completed successfully

    def reset(self, preserve_warmup: bool = False):
        """
        Reset training buffers at start of new episode.

        Args:
            preserve_warmup: If True and warmup is not complete, preserve training
                           buffers to continue collecting samples across episodes.
                           This prevents warmup progress loss at episode boundaries.
        """
        # Check if we should preserve warmup buffers
        if preserve_warmup and not getattr(self, "_warmup_complete", False):
            # Warmup not complete - preserve training buffers
            warmup_samples = getattr(self.config, "observer_warmup_samples", 300)
            current_buffer = max(
                len(getattr(self, "topk_scores_lst", [])),
                len(getattr(self, "risk_eta_pred_tensor_lst", [])),
            )
            if current_buffer > 0 and current_buffer < warmup_samples:
                smart_print(
                    f"[OBSERVER] 🔄 Preserving warmup buffers across episode boundary "
                    f"({current_buffer}/{warmup_samples} samples)"
                )
                # Notify display about episode transition (cross-episode tracking)
                if (
                    LIVE_DISPLAY_AVAILABLE
                    and display_update_observer_warmup is not None
                ):
                    try:
                        display_update_observer_warmup(
                            buffer_size=current_buffer,
                            target_samples=warmup_samples,
                            warmup_complete=False,
                            new_episode=True,  # Signal episode boundary
                        )
                    except TypeError:
                        # Fallback for older display_integration without new_episode param
                        pass
                # Only reset non-training state, keep buffers intact
                self.last_sigma_log_p = None
                self.last_sigma_pred = None
                self.last_pg_turnover = 0.0
                self.last_pg_change = 0.0
                self.last_pg_mean_return = 0.0
                self.last_pg_turnover_penalty = 0.0
                self.last_pg_change_penalty = 0.0
                self.last_pg_shaped_return = 0.0
                # Don't reset context buffer during warmup to maintain temporal continuity
                return

        # Full reset (warmup complete or preserve_warmup=False)
        self.market_vector_lst = []
        self.risk_eta_lst = []
        self.risk_eta_pred_tensor_lst = []
        self.risk_eta_target_tensor_lst = []
        self.rate_of_price_change_lst = []
        self.mkt_direction_lst = []
        self.sigma_log_p_lst = []
        self.last_sigma_log_p = None
        self.last_sigma_pred = None
        self.topk_scores_lst = []
        self.topk_indices_lst = []
        self.baseline_return_lst = []
        self.last_pg_turnover = 0.0
        self.last_pg_change = 0.0
        # PG reward components
        self.last_pg_mean_return = 0.0
        self.last_pg_turnover_penalty = 0.0
        self.last_pg_change_penalty = 0.0
        self.last_pg_shaped_return = 0.0
        # Reset context buffer for temporal augmentation (Spec 3.6)
        self._reset_context_buffer()
        # Reset gating router temporal state (Spec 3.6.2)
        if hasattr(self, "mafia_model") and hasattr(
            self.mafia_model, "reset_temporal_state"
        ):
            self.mafia_model.reset_temporal_state()

    def detach_temporal_state(self):
        """
        Detach all temporal states to truncate gradient graph (Spec 3.6.3).

        This method performs:
        1. Detach LSTM hidden/cell states (h, c) via mafia_model
        2. Detach all vectors in context_buffer

        Purpose: Cut gradient history while preserving values for next step.
        Called after each timestep during inference/validation.
        """
        # Detach LSTM state
        if hasattr(self, "mafia_model") and hasattr(
            self.mafia_model, "detach_temporal_state"
        ):
            self.mafia_model.detach_temporal_state()
        # Detach context buffer (Spec 3.6.3.2)
        self._detach_context_buffer()

    def reset_training_buffers(self):
        """
        Force reset all training buffers regardless of warmup state.
        Use this when explicitly transitioning between training phases.
        """
        self._warmup_complete = False
        self._observer_frozen_logged = False
        self.reset(preserve_warmup=False)

    def update_hidden_vec_reward(
        self,
        mode,
        rate_of_price_change,
        mkt_direction=None,
        market_return=None,
        current_date=None,
        env=None,
        pg_active: bool = True,
    ):
        """
        Collect rewards for Policy Gradient training.
        Compatible with MarketObserver.update_hidden_vec_reward() interface.

        Args:
            mode: 'train', 'valid', 'test'
            rate_of_price_change: (batch, N+1) - includes cash as first element
            mkt_direction: (batch,) - market direction label {0, 1, 2} (optional if env/current_date provided)
            market_return: (batch,) baseline return (e.g., market index); optional
            current_date: current trading date (for future-based direction label)
            env: environment providing fine_market data for direction label computation
        """
        if mode != "train":
            return
        pg_enabled = bool(pg_active)

        # Get last market scores (from predict() call)
        if len(self.topk_scores_lst) == 0:
            return

        rate_of_price_change = (
            th.from_numpy(rate_of_price_change).to(th.float32).to(self.device)
        )
        if pg_enabled:
            # Remove cash (first element) to match market scores shape
            rate_of_price_change_stocks = rate_of_price_change[
                :, 1:
            ]  # (batch, N_actual)

            # Handle size mismatch: rate_of_price_change_stocks may have different size than market scores
            expected_size = getattr(
                self, "last_action_dim", rate_of_price_change_stocks.shape[1]
            )
            actual_size = rate_of_price_change_stocks.shape[
                1
            ]  # Actual number of stocks

            if actual_size != expected_size:
                # Pad or truncate to match expected size
                if actual_size < expected_size:
                    # Pad with 1.0 (no change) for missing stocks
                    padding_size = expected_size - actual_size
                    padding = th.ones(
                        rate_of_price_change_stocks.shape[0],
                        padding_size,
                        device=rate_of_price_change_stocks.device,
                        dtype=rate_of_price_change_stocks.dtype,
                    )
                    rate_of_price_change_stocks = th.cat(
                        [rate_of_price_change_stocks, padding], dim=1
                    )
                else:
                    # Truncate to expected size (take first N stocks)
                    rate_of_price_change_stocks = rate_of_price_change_stocks[
                        :, :expected_size
                    ]

            # Store rate_of_price_change for reward calculation in train()
            self.rate_of_price_change_lst.append(rate_of_price_change_stocks)

        # Direction label: prefer future-based computation if env/current_date provided
        direction_label = None
        if env is not None and current_date is not None:
            direction_label = self._compute_direction_label_from_env(env, current_date)
        if direction_label is None and mkt_direction is not None:
            try:
                direction_label = int(np.array(mkt_direction).flatten()[-1])
            except Exception:
                direction_label = None
        if direction_label is not None:
            mkt_direction_tensor = th.tensor(
                [direction_label], device=self.device, dtype=th.long
            )
            self.mkt_direction_lst.append(mkt_direction_tensor)

        # Baseline return (market index) for Advantage; default 0 if not provided
        if pg_enabled:
            if market_return is None:
                baseline_tensor = th.zeros(
                    rate_of_price_change.shape[0],
                    device=self.device,
                    dtype=rate_of_price_change.dtype,
                )
            else:
                baseline_tensor = (
                    th.from_numpy(market_return)
                    .to(rate_of_price_change.dtype)
                    .to(self.device)
                )
                if baseline_tensor.ndim == 0:
                    baseline_tensor = baseline_tensor.reshape(1)
                if baseline_tensor.shape[0] != rate_of_price_change.shape[0]:
                    baseline_tensor = baseline_tensor.view(1).repeat(
                        rate_of_price_change.shape[0]
                    )
            self.baseline_return_lst.append(baseline_tensor)

        # Store sigma log-probabilities for direction classification (keep gradient)
        if self.last_sigma_log_p is not None:
            self.sigma_log_p_lst.append(self.last_sigma_log_p)

        # Realtime progress logging (single line, overwrite)
        self._log_realtime_progress()

    def _log_realtime_progress(self):
        """Log realtime training progress on a single line (overwrite mode)."""
        # Only log periodically to avoid spam (every 5 steps or configurable)
        log_interval = getattr(self.config, "observer_progress_log_interval", 5)
        n_samples = len(self.sigma_log_p_lst)
        if n_samples == 0 or n_samples % log_interval != 0:
            return

        # Gather current buffer stats
        n_direction = len(self.mkt_direction_lst)
        n_eta = len(self.risk_eta_pred_tensor_lst)
        n_selection = len(self.topk_scores_lst)

        # Get last predicted values
        last_direction = "N/A"
        if self.last_sigma_pred is not None:
            try:
                d = (
                    int(self.last_sigma_pred.item())
                    if hasattr(self.last_sigma_pred, "item")
                    else int(self.last_sigma_pred)
                )
                last_direction = ["Bear", "Side", "Bull"][d] if 0 <= d <= 2 else str(d)
            except Exception:
                pass

        last_eta = "N/A"
        if len(self.risk_eta_pred_tensor_lst) > 0:
            try:
                last_eta = f"{self.risk_eta_pred_tensor_lst[-1].item():.3f}"
            except Exception:
                pass

        # Single line realtime update - route to log file when display is active
        smart_print(
            f"⏳ [OBSERVER] Collecting samples: "
            f"Direction={n_direction} (pred:{last_direction}) | "
            f"Risk η={n_eta} (pred:{last_eta}) | "
            f"Selection={n_selection}"
        )

    def save_checkpoint(self, checkpoint_path: str, epoch: int, **kwargs):
        """
        Save MAFIA observer checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
        """
        checkpoint = {
            "epoch": epoch,
            "mafia_model_state_dict": self.mafia_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config": {
                "mafia_T_w": self.config.mafia_T_w,
                "mafia_DC_multipliers": self.config.mafia_DC_multipliers,
                "mafia_D": self.config.mafia_D,
                "mafia_D_h": self.config.mafia_D_h,
                "mafia_encoder_layers": self.config.mafia_encoder_layers,
                "mafia_encoder_heads": self.config.mafia_encoder_heads,
                "mafia_M_tech": self.config.mafia_M_tech,
                "mafia_M_dc": self.config.mafia_M_dc,
                "mafia_learning_rate": self.config.mafia_learning_rate,
                "mafia_weight_decay": self.config.mafia_weight_decay,
            },
            "action_dim": self.action_dim,
        }
        th.save(checkpoint, checkpoint_path)
        smart_print(
            f"MAFIA Observer checkpoint saved to {checkpoint_path} (epoch {epoch})",
            flush=True,
        )

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load MAFIA observer checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer/scheduler state (default: True)

        Returns:
            epoch: Epoch number from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = th.load(checkpoint_path, map_location=self.device)

        # Load model state
        mafia_state = checkpoint["mafia_model_state_dict"]

        # Remap old checkpoint keys: input_norm.weight -> input_norm.norm.weight
        # This handles backward compatibility when StableLayerNorm replaced nn.LayerNorm
        mafia_state = self._remap_legacy_layernorm_keys(mafia_state)

        # Ensure dynamically created temp embeddings exist before strict load
        self._ensure_temp_embeddings_from_state(mafia_state)
        try:
            self.mafia_model.load_state_dict(mafia_state)
            smart_print("[MAFIA] Strict load SUCCESS.", flush=True)
        except RuntimeError as e:
            smart_print(
                f"[MAFIA] Warning: Strict load failed ({e}). Retrying with strict=False...",
                flush=True,
            )
            # DEBUG: Print exact mismatches
            model_keys = set(self.mafia_model.state_dict().keys())
            ckpt_keys = set(mafia_state.keys())
            missing_in_model = ckpt_keys - model_keys
            missing_in_ckpt = model_keys - ckpt_keys
            intersection = model_keys & ckpt_keys
            
            smart_print(f"[DEBUG] Checkpoint keys: {len(ckpt_keys)}", flush=True)
            smart_print(f"[DEBUG] Model keys: {len(model_keys)}", flush=True)
            smart_print(f"[DEBUG] Intersection (Matched): {len(intersection)}", flush=True)
            
            if len(missing_in_model) > 0:
                smart_print(f"[DEBUG] Keys in Checkpoint BUT NOT in Model: {list(missing_in_model)}", flush=True)
            if len(missing_in_ckpt) > 0:
                smart_print(f"[DEBUG] Keys in Model BUT NOT in Checkpoint: {list(missing_in_ckpt)}", flush=True)
                
            self.mafia_model.load_state_dict(mafia_state, strict=False)
        
        if load_optimizer:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as e:
                 smart_print(f"[MAFIA] Warning: Optimizer load failed ({e}). Skipping optimizer load.", flush=True)

            try:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            except KeyError as e:
                # Backward compatibility: scheduler type changed (e.g., StepLR -> LambdaLR)
                smart_print(
                    f"[MAFIA] Warning: LR scheduler state missing key {e}; using freshly-initialized scheduler state.",
                    flush=True,
                )

        epoch = checkpoint.get("epoch", 0)
        smart_print(
            f"MAFIA Observer checkpoint loaded from {checkpoint_path} (epoch {epoch})",
            flush=True,
        )

        return epoch

    def load_merged_checkpoints(
        self,
        frozen_checkpoint_path: str,
        trainable_checkpoint_path: str,
        load_optimizer: bool = False,
    ) -> int:
        """
        Merge two checkpoints for SELECTION_ONLY training:
        - Frozen params (Direction/Risk Heads, Temporal Encoder) from MACRO checkpoint
        - Trainable params (Gate Network, Stock Experts) from previous SELECTION checkpoint

        This enables:
        1. Macro heads aligned with current year's training window
        2. Selection knowledge transfer from previous iteration

        Args:
            frozen_checkpoint_path: Path to MACRO checkpoint (same year)
            trainable_checkpoint_path: Path to previous SELECTION checkpoint
            load_optimizer: Whether to load optimizer state from trainable checkpoint

        Returns:
            epoch: Epoch number from trainable checkpoint (for resume tracking)
        """
        if not os.path.exists(frozen_checkpoint_path):
            raise FileNotFoundError(f"Frozen (MACRO) checkpoint not found: {frozen_checkpoint_path}")
        if not os.path.exists(trainable_checkpoint_path):
            raise FileNotFoundError(f"Trainable (SELECTION) checkpoint not found: {trainable_checkpoint_path}")

        # Load both checkpoints
        frozen_ckpt = th.load(frozen_checkpoint_path, map_location=self.device)
        trainable_ckpt = th.load(trainable_checkpoint_path, map_location=self.device)

        frozen_state = frozen_ckpt["mafia_model_state_dict"]
        trainable_state = trainable_ckpt["mafia_model_state_dict"]

        # Remap legacy LayerNorm keys for backward compatibility
        frozen_state = self._remap_legacy_layernorm_keys(frozen_state)
        trainable_state = self._remap_legacy_layernorm_keys(trainable_state)

        # Define frozen param prefixes (SELECTION_ONLY frozen components)
        # These come from MACRO checkpoint (aligned with current year's window)
        frozen_prefixes = [
            # Macro Stream (Direction/Risk Heads)
            "signal_generator.direction_head.",
            "signal_generator.risk_head.",
            "signal_generator.macro_context_norm.",  # Separate LayerNorm for Macro path
            # Temporal Encoder (Router backbone)
            "signal_generator.gating_router.temporal_encoder.",
            "signal_generator.gating_router.mkt_proj.",
            "signal_generator.gating_router.context_norm.",  # Used by Selection stream
            # Market Agents
            "mkt_ta.",
            "mkt_dc_ta_list.",
        ]

        # Merge state dicts
        merged_state = {}
        frozen_count = 0
        trainable_count = 0

        for key in self.mafia_model.state_dict().keys():
            is_frozen = any(key.startswith(prefix) for prefix in frozen_prefixes)
            if is_frozen:
                if key in frozen_state:
                    merged_state[key] = frozen_state[key]
                    frozen_count += 1
                else:
                    smart_print(f"[WARN] Frozen key '{key}' not found in MACRO checkpoint, using current weights")
                    merged_state[key] = self.mafia_model.state_dict()[key]
            else:
                if key in trainable_state:
                    merged_state[key] = trainable_state[key]
                    trainable_count += 1
                elif key in frozen_state:
                    # Fallback to frozen state if not in trainable
                    merged_state[key] = frozen_state[key]
                    trainable_count += 1
                else:
                    smart_print(f"[WARN] Trainable key '{key}' not found in any checkpoint, using current weights")
                    merged_state[key] = self.mafia_model.state_dict()[key]

        # Ensure dynamically created embeddings exist
        self._ensure_temp_embeddings_from_state(merged_state)

        # Load merged state
        try:
            self.mafia_model.load_state_dict(merged_state)
        except RuntimeError as e:
            smart_print(f"[MAFIA] Warning: Strict merged load failed ({e}). Retrying with strict=False...")
            self.mafia_model.load_state_dict(merged_state, strict=False)

        # Optionally load optimizer from trainable checkpoint
        if load_optimizer:
            try:
                self.optimizer.load_state_dict(trainable_ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                smart_print(f"[MAFIA] Warning: Optimizer load failed ({e}). Using fresh optimizer.")

        epoch = trainable_ckpt.get("epoch", 0)
        smart_print(
            f"[MAFIA] Merged checkpoints loaded:",
            flush=True,
        )
        smart_print(f"  - Frozen params ({frozen_count}): {os.path.basename(frozen_checkpoint_path)}")
        smart_print(f"  - Trainable params ({trainable_count}): {os.path.basename(trainable_checkpoint_path)}")
        smart_print(f"  - Resume from epoch: {epoch}")

        return epoch

    def reset_lr_scheduler(self):
        """
        Reset LR scheduler to initial state (fresh start from initial learning rate).

        Used when resuming from checkpoint for a new walk-forward window.
        This ensures the new window starts with full learning rate instead of
        continuing from the decayed LR of the previous window.
        """
        config = self.config

        # Get differential learning rates
        lr_direction = getattr(config, "mafia_lr_direction_head", 5e-4)
        lr_risk = getattr(config, "mafia_lr_risk_head", 1e-4)
        lr_backbone = getattr(config, "mafia_lr_backbone", 3e-4)

        schedule_mode = getattr(config, "mafia_lr_schedule", "cosine")
        warmup_epochs = getattr(config, "mafia_lr_warmup_epochs", 2)
        min_lr = getattr(config, "mafia_lr_min", 1e-6)
        total_epochs = max(1, config.num_epochs)
        end_factor = getattr(config, "mafia_lr_end_factor", 0.2)
        frac = getattr(config, "mafia_lr_end_fraction", 0.5)

        if schedule_mode == "cosine":
            # Cosine Annealing with Warmup
            def _lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return max(min_lr / lr_backbone, (epoch + 1) / warmup_epochs)
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr / lr_backbone, cosine_decay)

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )

        elif schedule_mode == "linear":
            decay_epochs = max(1, int(total_epochs * frac))

            def _lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                adj_epoch = epoch - warmup_epochs
                adj_decay = max(1, decay_epochs - warmup_epochs)
                if adj_epoch >= adj_decay:
                    return end_factor
                return 1.0 - (1.0 - end_factor) * (adj_epoch / float(adj_decay))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )

        elif schedule_mode == "linear_per_epoch":
            cycle_steps = max(1, int(getattr(config, "observer_mini_epoch_steps", 1)))
            decay_steps = max(1, int(cycle_steps * frac))

            def _lr_lambda(step_idx):
                step_in_cycle = step_idx % cycle_steps
                if step_in_cycle >= decay_steps:
                    return end_factor
                return 1.0 + (end_factor - 1.0) * (step_in_cycle / float(decay_steps))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
        else:
            decay_steps = max(1, total_epochs // 3)
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=decay_steps, gamma=0.1
            )

        # Reset optimizer param groups to initial differential LRs
        lr_map = {
            "direction_head": lr_direction,
            "risk_head": lr_risk,
            "backbone": lr_backbone,
        }
        for param_group in self.optimizer.param_groups:
            group_name = param_group.get("name", "backbone")
            initial_lr = lr_map.get(group_name, lr_backbone)
            param_group["lr"] = initial_lr
            param_group["initial_lr"] = initial_lr

        smart_print(
            f"[MAFIA] LR scheduler reset: mode={schedule_mode}, "
            f"direction={lr_direction:.1e}, risk={lr_risk:.1e}, backbone={lr_backbone:.1e}",
            flush=True,
        )
