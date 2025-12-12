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
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Callable, Any, Optional

th.autograd.set_detect_anomaly(True)

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

        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.mafia_model.parameters(),
            lr=config.mafia_learning_rate,
            weight_decay=config.mafia_weight_decay,
        )
        # Learning-rate schedule: align with TD3-style decay
        schedule_mode = getattr(config, "mafia_lr_schedule", "linear")
        start_lr = config.mafia_learning_rate
        end_lr = start_lr * getattr(config, "mafia_lr_end_factor", 0.2)
        frac = getattr(config, "mafia_lr_end_fraction", 0.5)

        if schedule_mode == "linear":
            decay_epochs = max(1, int(config.num_epochs * frac))

            def _lr_lambda(epoch):
                if epoch >= decay_epochs:
                    return end_lr / start_lr
                # Linear decay from start_lr to end_lr over decay_epochs (one-shot over full run)
                return 1.0 - (1.0 - end_lr / start_lr) * (epoch / float(decay_epochs))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
        elif schedule_mode == "linear_per_epoch":
            # Repeat linear decay every epoch: use a cycle on scheduler steps
            cycle_steps = max(
                1,
                int(getattr(config, "observer_mini_epoch_steps", 1)),
            )
            decay_steps = max(1, int(cycle_steps * frac))

            def _lr_lambda(step_idx):
                step_in_cycle = step_idx % cycle_steps
                if step_in_cycle >= decay_steps:
                    return end_lr / start_lr
                return 1.0 + (end_lr / start_lr - 1.0) * (
                    step_in_cycle / float(decay_steps)
                )

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
        else:
            decay_steps = max(1, config.num_epochs // 3)  # Ensure at least 1
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=decay_steps, gamma=0.1
            )

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

    def _validate_config(self):
        """Validate that config has all required MAFIA hyperparameters."""
        required_params = [
            "mafia_T_w",
            "mafia_DC_thresholds",
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
        Returns (1, 4) tensor: [DC_Event_Flag, Breadth_Gap, Div_Signal, Signed_VPI_Zscore]

        Args:
            market_close_price: Current market close price (for online inference)
            stock_closes: Current stock close prices (N,) for breadth calculation
            stock_volumes: Current stock volumes (N,) - optional
            market_volume: Current market volume - for VPI calculation

        Returns:
            explicit_signals: (1, 4) tensor with normalized values
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

        # === 1. DC_Event_Flag: Structural break detection (binary) ===
        dc_event_flag = 0.0
        if market_close_price is not None and len(self.price_history_buffer) >= 2:
            if self.dc_state["p_ext"] is None:
                self.dc_state["p_ext"] = self.price_history_buffer[0]
                self.dc_state["mode"] = "up"

            p_ext = self.dc_state["p_ext"]
            var = (market_close_price - p_ext) / (p_ext + 1e-8)

            if self.dc_state["mode"] == "up":
                if var < -self.dc_threshold:
                    dc_event_flag = 1.0  # Downward DC (Crash)
                    self.dc_state["mode"] = "down"
                    self.dc_state["p_ext"] = market_close_price
                elif market_close_price > p_ext:
                    self.dc_state["p_ext"] = market_close_price
            else:
                if var > self.dc_threshold:
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

        # Create tensor (1, 4)
        explicit_signals = th.tensor(
            [[dc_event_flag, breadth_gap, div_signal, vpi_zscore]],
            dtype=th.float32,
            device=self.device,
        )

        return explicit_signals

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
            smart_print(
                f"[MAFIA] Recreated temp projection for {prefix} "
                f"(in={in_features}, out={out_features})",
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
        **kwargs,
    ):
        """
        Predict market_vector and risk_eta using MAFIA model.

        Args:
            raw_ochlv_data: (N, M, T_w) - Direct MAFIA input (required)
            finemkt_feat: Deprecated - kept for interface compatibility
            finestock_feat: Deprecated - kept for interface compatibility
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
                market_context,
                sigma_logits,
                topk_indices,
                topk_embeddings,
                topk_scores,
            ) = self.mafia_model(
                ochlv_tensor,
                market_index_ochlv_data=market_index_ochlv_tensor,
                force_topk_indices=force_topk_indices,
                router_context_buffer=router_context_buffer,
                explicit_signals=explicit_signals,  # Direction Head signals (Spec 3.5)
            )
        else:
            self.mafia_model.eval()
            with th.no_grad():
                (
                    market_vector,
                    risk_eta,
                    market_scores_full,
                    market_context,
                    sigma_logits,
                    topk_indices,
                    topk_embeddings,
                    topk_scores,
                ) = self.mafia_model(
                    ochlv_tensor,
                    market_index_ochlv_data=market_index_ochlv_tensor,
                    force_topk_indices=force_topk_indices,
                    router_context_buffer=router_context_buffer,
                    explicit_signals=explicit_signals,  # Direction Head signals (Spec 3.5)
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
        )

    def _prepare_ochlv_tensor(self, raw_ochlv_data):
        """
        Prepare raw OCHLV data as tensor.

        Args:
            raw_ochlv_data: (N, M, T_w) numpy array

        Returns:
            torch.Tensor: (1, N, M, T_w) tensor on device
        """
        ochlv_tensor = th.from_numpy(raw_ochlv_data).to(th.float32)
        ochlv_tensor = ochlv_tensor.unsqueeze(0)  # Add batch dim: (1, N, M, T_w)
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
        mkt_ochlv_tensor = mkt_ochlv_tensor.unsqueeze(
            0
        )  # Add batch dim: (1, 1, 5, T_w)
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
        Compute future-based direction label using cumulative return over lookahead window.
        Labeling:
            Bear (0): R_fut < -delta
            Side (1): -delta <= R_fut <= delta
            Bull (2): R_fut > delta

        Fallback Strategy (when lookahead data is insufficient):
            - Use most recent predicted direction (last_sigma_pred) from previous step
            - This ensures consistency: Direction Head already predicted from macro context at t-1
            - Avoids noise from 1-step momentum or lookback-only approaches
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

        close_col = f"mkt_{getattr(self.config, 'finefreq', '1d')}_close"
        if close_col not in fine_market.columns:
            if "close" in fine_market.columns:
                close_col = "close"
            else:
                return None

        fm_sorted = fine_market.sort_values("date").reset_index(drop=True)
        mask = fm_sorted["date"] == current_date
        idx_arr = np.flatnonzero(mask.to_numpy())
        if len(idx_arr) == 0:
            return None
        idx = int(idx_arr[-1])

        lookahead = int(getattr(self.config, "direction_label_lookahead", 5))
        delta = float(getattr(self.config, "direction_label_delta", 0.025))
        future_idx = idx + lookahead

        # Check if lookahead data is available
        if future_idx >= len(fm_sorted):
            # Fallback: Use most recent predicted direction from Direction Head
            # This is consistent with model behavior since Direction Head predicted at t-1
            # from macro context, avoiding noise from simple lookback approaches
            if self.last_sigma_pred is not None:
                return int(self.last_sigma_pred.item())
            else:
                # If no previous prediction available, return None (conservative)
                return None

        try:
            price_now = float(fm_sorted.iloc[idx][close_col])
            price_future = float(fm_sorted.iloc[future_idx][close_col])
        except Exception:
            return None
        if (
            not np.isfinite(price_now)
            or not np.isfinite(price_future)
            or price_now <= 0
        ):
            return None
        r_fut = (price_future - price_now) / price_now
        if r_fut < -delta:
            return 0
        if r_fut > delta:
            return 2
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

            # Compute per-sample MSE (don't reduce yet)
            eta_loss_per_sample = F.mse_loss(
                pred_eta, target_eta, reduction="none"
            )  # (T_eta,)
            risk_losses_per_step.append(eta_loss_per_sample)

            # For logging/monitoring, compute aggregated loss
            eta_loss = eta_loss_per_sample.mean()
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
            f"    │ 3️⃣  L_risk (Risk Tolerance η - MSE Regression)                          │\n"
            f"    │     Mục tiêu: Dự đoán η ∈ [0.7, 1.3] để scale σ_target = σ_base × η    │\n"
            f"    ├─────────────────────────────────────────────────────────────────────────┤\n"
            f"    │  • L_risk (MSE):          {fmt(eta_loss_val):>12}  ← train DAILY (không mask)  │\n"
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
                "mafia_DC_thresholds": self.config.mafia_DC_thresholds,
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

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load MAFIA observer checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            epoch: Epoch number from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = th.load(checkpoint_path, map_location=self.device)

        # Load model state
        mafia_state = checkpoint["mafia_model_state_dict"]
        # Ensure dynamically created temp embeddings exist before strict load
        self._ensure_temp_embeddings_from_state(mafia_state)
        self.mafia_model.load_state_dict(mafia_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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

    def reset_lr_scheduler(self):
        """
        Reset LR scheduler to initial state (fresh start from initial learning rate).

        Used when resuming from checkpoint for a new walk-forward window.
        This ensures the new window starts with full learning rate instead of
        continuing from the decayed LR of the previous window.
        """
        config = self.config
        start_lr = config.mafia_learning_rate
        schedule_mode = getattr(config, "mafia_lr_schedule", "linear_per_epoch")
        end_lr = start_lr * getattr(config, "mafia_lr_end_factor", 0.2)
        frac = getattr(config, "mafia_lr_end_fraction", 0.5)

        if schedule_mode == "linear":
            decay_epochs = max(1, int(config.num_epochs * frac))

            def _lr_lambda(epoch):
                if epoch >= decay_epochs:
                    return end_lr / start_lr
                return 1.0 - (1.0 - end_lr / start_lr) * (epoch / float(decay_epochs))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
        elif schedule_mode == "linear_per_epoch":
            cycle_steps = max(
                1,
                int(getattr(config, "observer_mini_epoch_steps", 1)),
            )
            decay_steps = max(1, int(cycle_steps * frac))

            def _lr_lambda(step_idx):
                step_in_cycle = step_idx % cycle_steps
                if step_in_cycle >= decay_steps:
                    return end_lr / start_lr
                return 1.0 + (end_lr / start_lr - 1.0) * (
                    step_in_cycle / float(decay_steps)
                )

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
        else:
            decay_steps = max(1, config.num_epochs // 3)
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=decay_steps, gamma=0.1
            )

        # Reset optimizer param groups to initial LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr
            param_group["initial_lr"] = start_lr

        smart_print(
            f"[MAFIA] LR scheduler reset: mode={schedule_mode}, initial_lr={start_lr}",
            flush=True,
        )
