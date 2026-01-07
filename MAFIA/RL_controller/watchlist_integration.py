#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watchlist Integration Module for MAFIA Observer Trainer

This module provides helper functions to integrate the Watchlist System
into the Observer Offline Trainer. Following the spec:
- Daily: Run backbone on 122 stocks, screening, update embeddings buffer
- Rebalance: Filter logits to Watchlist, Selection, Post-selection cleanup

Usage:
    from RL_controller.watchlist_integration import WatchlistIntegration

    # In trainer __init__:
    self.watchlist_integration = WatchlistIntegration(config, observer, device)

    # In training loop:
    # Daily step:
    screening_result = self.watchlist_integration.daily_step(
        close_prices, volumes, market_close, ma20_prices, expert_logits, embeddings
    )

    # On rebalance:
    filtered_logits = self.watchlist_integration.filter_for_selection(market_logits)
    # ... run selection ...
    cleanup_result = self.watchlist_integration.post_rebalance_cleanup(prev_holdings, new_holdings)

Author: MAFIA Team
Spec: cozy-whistling-nova.md
"""

import numpy as np
import torch as th
from typing import Dict, Optional, Any, Tuple

from RL_controller.mafia_observer import WatchlistManager
from RL_controller.mafia_feature_processor import MAFIAFeatureProcessor


class WatchlistIntegration:
    """
    Helper class to integrate Watchlist System into Observer Trainer.

    Provides clean API for:
    1. Daily screening and embeddings update
    2. Logits filtering for Selection
    3. Post-rebalance cleanup
    4. State management for checkpointing
    """

    def __init__(self, config, observer, device: th.device):
        """
        Initialize Watchlist Integration.

        Args:
            config: Configuration object
            observer: MAFIAObserver instance
            device: Torch device
        """
        self.config = config
        self.observer = observer
        self.device = device
        self.enabled = getattr(config, 'watchlist_enabled', True)

        if not self.enabled:
            self.watchlist_manager = None
            self.feature_processor = None
            return

        # Get dimensions from model
        N = observer.action_dim  # Universe size (e.g., 122)
        D = getattr(config, 'mafia_D', 64)  # Embedding dimension

        # Initialize WatchlistManager
        self.watchlist_manager = WatchlistManager(
            config=config,
            universe_size=N,
            embedding_dim=D,
            device=device
        )

        # Feature processor for Technical Score
        self.feature_processor = MAFIAFeatureProcessor(config)

        # Track day counter for screening schedule
        self._day_counter = 0

        # Model trust: Only use model_scores after sufficient training
        # Default: not trusted until set_model_trusted(True) is called
        self._model_trusted = False
        self._use_model_after_iter = getattr(config, 'watchlist_use_model_after_iter', 1)

    def is_enabled(self) -> bool:
        """Check if Watchlist System is enabled."""
        return self.enabled and self.watchlist_manager is not None

    def set_model_trusted(self, trusted: bool):
        """
        Set whether model scores should be used for screening.

        Call this after model has been trained (e.g., after first iteration).
        Before this is called, screening uses Technical Score only.

        Args:
            trusted: True to use model scores, False for Technical-only screening
        """
        self._model_trusted = trusted
        if self.watchlist_manager is not None:
            self.watchlist_manager.set_use_model_scores(trusted)

    def is_model_trusted(self) -> bool:
        """Check if model scores are being used for screening."""
        return self._model_trusted

    def reset(self):
        """Reset Watchlist state (for new episode)."""
        if self.watchlist_manager is not None:
            self.watchlist_manager.reset()
        self._day_counter = 0
        # Note: Don't reset _model_trusted - that persists across episodes

    def daily_step(
        self,
        close_prices: np.ndarray,
        volumes: np.ndarray,
        market_close: np.ndarray,
        model_scores: np.ndarray,
        embeddings: Optional[th.Tensor] = None,
        ma20_prices: Optional[np.ndarray] = None,
        ma50_prices: Optional[np.ndarray] = None,
        return_20d: Optional[np.ndarray] = None,
        force_review: bool = False,
    ) -> Dict[str, Any]:
        """
        Run daily Watchlist operations: Screening + Embeddings update.

        Called EVERY day (not just rebalance days).

        Args:
            close_prices: (N, T) close prices for all stocks
            volumes: (N, T) volumes for all stocks
            market_close: (T,) market index close prices
            model_scores: (N,) market_scores_full from model - softmax scores for each stock
                          Represents model's allocation preference (higher = model prefers more)
            embeddings: (N, D) stock embeddings from model (optional)
            ma20_prices: (N,) SMA(20) prices for uptrend check
            ma50_prices: (N,) SMA(50) prices for emergency removal
            return_20d: (N,) 20-day returns for emergency removal
            force_review: Force full Watchlist review

        Returns:
            Dict with screening results
        """
        if not self.is_enabled():
            return {'action': 'disabled'}

        self._day_counter += 1

        # 1. Compute Technical Score for all stocks
        tech_scores, tech_components = self.feature_processor.compute_technical_score(
            close_prices=close_prices,
            volumes=volumes,
            market_close=market_close,
            momentum_window=getattr(self.config, 'watchlist_momentum_window', 60),
            volume_short_window=getattr(self.config, 'watchlist_volume_short_window', 20),
            volume_long_window=getattr(self.config, 'watchlist_volume_long_window', 60),
            ma_window=getattr(self.config, 'watchlist_ma_window', 20),
        )

        # 2. Compute Model Score percentiles (from market_scores_full)
        model_percentiles = self.feature_processor.compute_model_score_percentile(model_scores)

        # 3. Run screening with configurable component weights
        current_prices = close_prices[:, -1] if close_prices.ndim == 2 else close_prices
        screening_result = self.watchlist_manager.run_screening(
            technical_scores=tech_scores,
            model_percentiles=model_percentiles,
            current_prices=current_prices,
            ma20_prices=ma20_prices,
            ma50_prices=ma50_prices,
            return_20d=return_20d,
            force_review=force_review,
            tech_components=tech_components,  # Pass individual components for weighted scoring
        )

        # 4. Update embeddings buffer if provided
        if embeddings is not None and self.watchlist_manager._initialized:
            watchlist_indices = self.watchlist_manager.watchlist_indices
            if watchlist_indices is not None and len(watchlist_indices) > 0:
                # Extract Watchlist embeddings
                if embeddings.dim() == 2:  # (N, D)
                    watchlist_embeddings = embeddings[watchlist_indices]  # (watchlist_size, D)
                else:  # (batch, N, D)
                    watchlist_embeddings = embeddings[:, watchlist_indices, :]
                    watchlist_embeddings = watchlist_embeddings.mean(dim=0)  # Average over batch

                # Pad or truncate to match buffer size
                target_size = self.watchlist_manager.watchlist_size
                current_size = len(watchlist_indices)
                if current_size < target_size:
                    # Pad with zeros
                    pad_size = target_size - current_size
                    D = embeddings.size(-1)
                    padding = th.zeros(pad_size, D, dtype=embeddings.dtype, device=self.device)
                    watchlist_embeddings = th.cat([watchlist_embeddings, padding], dim=0)
                elif current_size > target_size:
                    # Truncate
                    watchlist_embeddings = watchlist_embeddings[:target_size]

                self.watchlist_manager.update_embeddings_buffer(watchlist_embeddings)

        screening_result['day_counter'] = self._day_counter
        screening_result['tech_scores_mean'] = float(tech_scores.mean())
        screening_result['tech_scores_std'] = float(tech_scores.std())

        return screening_result

    def filter_for_selection(
        self,
        market_logits: th.Tensor,
    ) -> Tuple[th.Tensor, np.ndarray]:
        """
        Filter market_logits to only Watchlist stocks for Selection.

        Called on REBALANCE days, before Gumbel-TopK.

        Args:
            market_logits: (batch, N) full universe logits

        Returns:
            Tuple of:
                - filtered_logits: (batch, watchlist_size) Watchlist-only logits
                - watchlist_indices: (watchlist_size,) universe indices of Watchlist stocks
        """
        if not self.is_enabled() or not self.watchlist_manager._initialized:
            # Fallback: return all logits
            N = market_logits.size(-1)
            return market_logits, np.arange(N)

        watchlist_indices = self.watchlist_manager.watchlist_indices
        if watchlist_indices is None or len(watchlist_indices) == 0:
            N = market_logits.size(-1)
            return market_logits, np.arange(N)

        # Filter logits to Watchlist stocks
        filtered_logits = self.watchlist_manager.filter_logits_to_watchlist(market_logits)

        return filtered_logits, watchlist_indices

    def map_selection_to_universe(
        self,
        watchlist_selection_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Map Selection indices (0 to watchlist_size-1) back to Universe indices.

        Called after Gumbel-TopK to get actual stock indices.

        Args:
            watchlist_selection_indices: (K,) indices within Watchlist

        Returns:
            universe_indices: (K,) indices within Universe
        """
        if not self.is_enabled() or not self.watchlist_manager._initialized:
            return watchlist_selection_indices

        return self.watchlist_manager.map_to_universe_indices(watchlist_selection_indices)

    def post_rebalance_cleanup(
        self,
        prev_holdings_indices: np.ndarray,
        new_holdings_indices: np.ndarray,
        current_prices: np.ndarray,
        portfolio_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Post-rebalance cleanup: ĐK B - Remove holdings not selected.

        Called AFTER Selection on rebalance days.

        Args:
            prev_holdings_indices: (K,) previous holdings (universe indices)
            new_holdings_indices: (K,) new selections (universe indices)
            current_prices: (N,) current prices for all stocks
            portfolio_weights: (N,) allocation weights (optional)

        Returns:
            Dict with cleanup results
        """
        if not self.is_enabled():
            return {'action': 'disabled', 'removed': 0}

        # Update holdings in WatchlistManager
        N = self.watchlist_manager.universe_size
        new_holdings_mask = np.zeros(N, dtype=bool)
        new_holdings_mask[new_holdings_indices] = True
        self.watchlist_manager.update_holdings(new_holdings_mask, current_prices)

        # Update portfolio weights if provided
        if portfolio_weights is not None:
            self.watchlist_manager.update_portfolio_weights(portfolio_weights)

        # ĐK B: Remove holdings not selected in new Top-K
        removed = self.watchlist_manager.post_selection_cleanup(
            prev_holdings_indices=prev_holdings_indices,
            new_holdings_indices=new_holdings_indices,
        )

        return {
            'action': 'post_rebalance_cleanup',
            'removed': removed,
            'new_watchlist_size': self.watchlist_manager.current_size,
        }

    def get_watchlist_context(self) -> Optional[th.Tensor]:
        """
        Get current Watchlist Context for monitoring/logging.

        Returns:
            watchlist_context: (D,) tensor or None
        """
        if not self.is_enabled():
            return None
        return self.watchlist_manager.get_watchlist_context()

    def get_watchlist_indices(self) -> Optional[np.ndarray]:
        """Get current Watchlist indices."""
        if not self.is_enabled():
            return None
        return self.watchlist_manager.watchlist_indices

    def get_watchlist_size(self) -> int:
        """Get current Watchlist size."""
        if not self.is_enabled():
            return 0
        return self.watchlist_manager.current_size

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        if not self.is_enabled():
            return {'enabled': False}

        return {
            'enabled': True,
            'watchlist_manager': self.watchlist_manager.state_dict(),
            'day_counter': self._day_counter,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        if not state.get('enabled', False) or not self.is_enabled():
            return

        if 'watchlist_manager' in state:
            self.watchlist_manager.load_state_dict(state['watchlist_manager'])
        self._day_counter = state.get('day_counter', 0)


# ============================================================
# Convenience functions for direct integration
# ============================================================

def create_watchlist_integration(config, observer, device: th.device) -> WatchlistIntegration:
    """
    Factory function to create WatchlistIntegration.

    Usage in trainer:
        self.watchlist = create_watchlist_integration(config, observer, device)
    """
    return WatchlistIntegration(config, observer, device)


def compute_screening_inputs(
    batch_data,
    feature_processor: MAFIAFeatureProcessor,
    config,
) -> Dict[str, np.ndarray]:
    """
    Compute screening inputs from batch data.

    Args:
        batch_data: TrajectoryBatch with stock_ochlv
        feature_processor: MAFIAFeatureProcessor
        config: Config object

    Returns:
        Dict with close_prices, volumes, market_close, ma20_prices, etc.
    """
    # Extract data from batch (last timestep for screening)
    # stock_ochlv: (B, T_m, N, 5) - [open, close, high, low, volume]
    stock_ochlv = batch_data.stock_ochlv
    B, T_m, N, _ = stock_ochlv.shape

    # Use last window for price history
    T_w = getattr(config, 'mafia_T_w', 30)
    window_start = max(0, T_m - T_w)

    # Extract close prices and volumes: (N, T_w)
    # Average over batch dimension for screening
    close_prices = stock_ochlv[:, window_start:, :, 1].mean(dim=0).cpu().numpy().T  # (N, T_w)
    volumes = stock_ochlv[:, window_start:, :, 4].mean(dim=0).cpu().numpy().T  # (N, T_w)

    # Market close prices
    if batch_data.market_ochlv is not None:
        market_close = batch_data.market_ochlv[:, window_start:, 0, 1].mean(dim=0).cpu().numpy()  # (T_w,)
    else:
        # Fallback: average of all stock closes
        market_close = close_prices.mean(axis=0)  # (T_w,)

    # Compute MA20
    if close_prices.shape[1] >= 20:
        ma20_prices = close_prices[:, -20:].mean(axis=1)  # (N,)
    else:
        ma20_prices = close_prices.mean(axis=1)

    # Compute MA50 (for emergency removal)
    if close_prices.shape[1] >= 50:
        ma50_prices = close_prices[:, -50:].mean(axis=1)  # (N,)
    else:
        ma50_prices = close_prices.mean(axis=1)

    # Compute 20-day return
    if close_prices.shape[1] >= 20:
        return_20d = (close_prices[:, -1] / close_prices[:, -20] - 1.0)  # (N,)
    else:
        return_20d = np.zeros(N)

    return {
        'close_prices': close_prices,
        'volumes': volumes,
        'market_close': market_close,
        'ma20_prices': ma20_prices,
        'ma50_prices': ma50_prices,
        'return_20d': return_20d,
    }
