# ！/usr/bin/python
# -*- coding: utf-8 -*-#

"""
MAFIA Model Modules

This module contains the core neural network modules for MAFIA:
- CSAModule: Cross-Sectional Analysis module
- TAModule: Temporal Analysis module
- STFusionModule: Spatial-Temporal Fusion module
- SignalGenerator: Portfolio Generator (market_vector + eta)
- MAFIAModel: Complete model integrating all agents
- DenseMoEGatingRouter: Dense MoE Gating Router for expert weight computation
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class CSAModule(nn.Module):
    """
    Cross-Sectional Analysis (CSA) Module.

    Learns spatial correlations between N assets.

    Input: P_i ∈ ℝ^(N × T_w × M)
    Output: O_i^CSA ∈ ℝ^(N × D)
    """

    def __init__(self, config, agent_type: str = "tech"):
        """
        Initialize CSA Module.

        Args:
            config: Configuration object
            agent_type: 'tech' for Technical agent, 'dc' for DC agents
        """
        super().__init__()
        self.config = config
        self.agent_type = agent_type
        self.T_w = config.mafia_T_w
        self.D = config.mafia_D
        self.D_h = config.mafia_D_h

        if agent_type == "tech":
            self.M = config.mafia_M_tech  # 8
            self.token_dim = self.T_w * self.M  # 30 * 8 = 240
        else:  # dc
            self.M = config.mafia_M_dc  # 5
            self.token_dim = self.T_w * self.M  # 30 * 5 = 150

        # Token Generation: Reshape (N, T_w, M) -> (N, T_w*M)
        # This is done in forward pass

        # Embedding MLP
        self.embedding = nn.Sequential(
            nn.Linear(self.token_dim, self.D_h), nn.GELU(), nn.Linear(self.D_h, self.D)
        )

        # Transformer Encoder
        encoder_layers = []
        for _ in range(config.mafia_encoder_layers):
            encoder_layers.append(
                TransformerEncoderLayer(
                    d_model=self.D,
                    nhead=config.mafia_encoder_heads,
                    dim_feedforward=self.D_h * 2,
                    dropout=0.1,
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, P_i: th.Tensor) -> th.Tensor:
        """
        Forward pass through CSA module.

        Args:
            P_i: (batch, N, T_w, M) tensor

        Returns:
            O_i^CSA: (batch, N, D) tensor
        """
        batch_size, N, T_w, M = P_i.shape

        # Token Generation: Reshape (N, T_w, M) -> (N, T_w*M)
        # Reshape: (batch, N, T_w, M) -> (batch, N, T_w*M)
        tokens = P_i.reshape(batch_size, N, self.token_dim)

        # Embedding
        embedded = self.embedding(tokens)  # (batch, N, D)

        # Transformer Encoder
        # Encoder expects (batch, seq_len, d_model)
        encoded = self.encoder(embedded)  # (batch, N, D)

        return encoded


class TAModule(nn.Module):
    """
    Temporal Analysis (TA) Module.

    Learns temporal correlations between T_w time points.

    Input: P_i ∈ ℝ^(N × T_w × M)
    Output: O_i^TA ∈ ℝ^(T_w × D)
    """

    def __init__(
        self,
        config,
        agent_type: str = "tech",
        shared_mlp: Optional[nn.Module] = None,
        N: Optional[int] = None,
    ):
        """
        Initialize TA Module.

        Args:
            config: Configuration object
            agent_type: 'tech' for Technical agent, 'dc' for DC agents
            shared_mlp: Shared MLP for DC agents (if agent_type == 'dc')
            N: Number of stocks (if None, uses config.topK)
        """
        super().__init__()
        self.config = config
        self.agent_type = agent_type
        self.T_w = int(config.mafia_T_w)
        self.D = int(config.mafia_D)
        self.D_h = int(config.mafia_D_h)

        # Use actual N if provided, otherwise use config.topK
        actual_N = N if N is not None else config.topK

        if agent_type == "tech":
            self.M = int(config.mafia_M_tech)  # 8
            self.token_dim = int(actual_N * self.M)  # N * M
        elif agent_type == "mkt":
            self.M = int(config.mafia_M_mkt)  # 19
            self.token_dim = int(actual_N * self.M)  # N * M (N=1 for market-index)
        else:  # dc
            self.M = int(config.mafia_M_dc)  # 5
            self.token_dim = int(actual_N * self.M)  # N * M

        # Embedding MLP
        if shared_mlp is not None and agent_type == "dc":
            self.embedding = shared_mlp
        else:
            self.embedding = nn.Sequential(
                nn.Linear(self.token_dim, self.D_h),
                nn.GELU(),
                nn.Linear(self.D_h, self.D),
            )

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(self.D, self.T_w)

        # Event Detector (only for DC agents)
        if agent_type == "dc":
            self.event_detector = EventDetector(self.D)
        else:
            self.event_detector = None

        # Transformer Encoder
        encoder_layers = []
        for _ in range(config.mafia_encoder_layers):
            encoder_layers.append(
                TransformerEncoderLayer(
                    d_model=self.D,
                    nhead=config.mafia_encoder_heads,
                    dim_feedforward=self.D_h * 2,
                    dropout=0.1,
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(
        self, P_i: th.Tensor, dc_features: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """
        Forward pass through TA module.

        Args:
            P_i: (batch, N, T_w, M) tensor
            dc_features: (batch, N, T_w, 5) tensor - DC features for Event Detector (DC agents only)

        Returns:
            O_i^TA: (batch, T_w, D) tensor
        """
        batch_size, N, T_w, M = P_i.shape

        # Calculate actual token_dim from input shape (N may differ from initialization)
        actual_token_dim = N * M

        # Token Generation: Permute and reshape
        # (batch, N, T_w, M) -> (batch, T_w, N, M) -> (batch, T_w, N*M)
        P_permuted = P_i.permute(0, 2, 1, 3)  # (batch, T_w, N, M)
        tokens = P_permuted.reshape(
            batch_size, T_w, actual_token_dim
        )  # (batch, T_w, N*M)

        # If token_dim changed, we need to handle embedding differently
        # For now, use a linear projection if needed
        if actual_token_dim != self.token_dim:
            # Create a temporary embedding layer if dimensions don't match
            if (
                not hasattr(self, "_temp_embedding")
                or self._temp_embedding[0].in_features != actual_token_dim
            ):
                self._temp_embedding = nn.Sequential(
                    nn.Linear(actual_token_dim, self.D_h),
                    nn.GELU(),
                    nn.Linear(self.D_h, self.D),
                ).to(tokens.device)
            embedded = self._temp_embedding(tokens)  # (batch, T_w, D)
        else:
            embedded = self.embedding(tokens)  # (batch, T_w, D)

        # Signal Merging: Add positional encoding
        PE_time = self.pos_encoding(embedded)  # (batch, T_w, D)
        input_ta = embedded + PE_time

        # Add high-order DC signals (DC agents only)
        if (
            self.agent_type == "dc"
            and self.event_detector is not None
            and dc_features is not None
        ):
            H_DC = self.event_detector(dc_features)  # (batch, T_w, D)
            input_ta = input_ta + H_DC

        # Transformer Encoder
        encoded = self.encoder(input_ta)  # (batch, T_w, D)

        return encoded


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences.
    Uses sin(t') where t' is linearly mapped from [0, T_w-1] to [0, π/2].
    """

    def __init__(self, d_model: int, max_len: int = 30):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create learnable positional encoding
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float32).unsqueeze(1)

        # Map position to [0, π/2]
        t_prime = (position / (max_len - 1)) * (math.pi / 2)

        # Create sinusoidal encoding
        div_term = th.exp(
            th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = th.sin(t_prime * div_term)
        pe[:, 1::2] = th.cos(t_prime * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model) tensor

        Returns:
            PE: (batch, seq_len, d_model) positional encoding
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class EventDetector(nn.Module):
    """
    Event Detector for DC agents.
    Extracts high-order signals from DC features: Δ(State) and Δ(Duration).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Embedding for event signals
        # Input: (batch, T_w, 2) - [Δ(State), Δ(Duration)]
        # Output: (batch, T_w, d_model)
        self.event_embedding = nn.Sequential(
            nn.Linear(2, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model)
        )

    def forward(self, dc_features: th.Tensor) -> th.Tensor:
        """
        Extract event signals from DC features.

        Args:
            dc_features: (batch, N, T_w, 5) - DC features [State, Magnitude, Duration, Volume_Ratio, Event_Flag]

        Returns:
            H_DC: (batch, T_w, d_model) - High-order DC signals
        """
        batch_size, N, T_w, _ = dc_features.shape

        # Extract State and Duration
        state = dc_features[:, :, :, 0]  # (batch, N, T_w)
        duration = dc_features[:, :, :, 2]  # (batch, N, T_w)

        # Average across assets
        state_avg = th.mean(state, dim=1)  # (batch, T_w)
        duration_avg = th.mean(duration, dim=1)  # (batch, T_w)

        # Compute Δ(State) and Δ(Duration)
        # Pad first element with zeros
        state_padded = th.cat([state_avg[:, 0:1], state_avg], dim=1)  # (batch, T_w+1)
        duration_padded = th.cat(
            [duration_avg[:, 0:1], duration_avg], dim=1
        )  # (batch, T_w+1)

        delta_state = state_avg - state_padded[:, :-1]  # (batch, T_w)
        delta_duration = duration_avg - duration_padded[:, :-1]  # (batch, T_w)

        # Combine event signals
        event_signals = th.stack(
            [delta_state, delta_duration], dim=-1
        )  # (batch, T_w, 2)

        # Embed to d_model
        H_DC = self.event_embedding(event_signals)  # (batch, T_w, d_model)

        return H_DC


class STFusionModule(nn.Module):
    """
    Spatial-Temporal Fusion Module.
    Fuses CSA and TA outputs using attention mechanism.

    Input: O_i^CSA ∈ ℝ^(N × D), O_i^TA ∈ ℝ^(T_w × D)
    Output: O_i ∈ ℝ^(N × 1) (logits)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)

        # Final MLP: (N, D) -> (N, 1)
        self.output_proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, O_CSA: th.Tensor, O_TA: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass through ST-Fusion.

        Args:
            O_CSA: (batch, N, D) - CSA output
            O_TA: (batch, T_w, D) - TA output

        Returns:
            O_i: (batch, N, 1) - Agent logits
            O_i_ST: (batch, N, D) - ST-Fusion embedding (before projection)
        """
        batch_size, N, D = O_CSA.shape
        T_w = O_TA.size(1)

        # Attention: O_CSA as Query, O_TA as Key and Value
        # Compute attention scores: (batch, N, D) @ (batch, D, T_w) -> (batch, N, T_w)
        attention_scores = th.bmm(O_CSA, O_TA.transpose(1, 2))  # (batch, N, T_w)

        # Scale and softmax
        attention_scores = attention_scores * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, N, T_w)

        # Weighted sum: (batch, N, T_w) @ (batch, T_w, D) -> (batch, N, D)
        fused = th.bmm(attention_weights, O_TA)  # (batch, N, D)

        # Final projection: (batch, N, D) -> (batch, N, 1)
        output = self.output_proj(fused)  # (batch, N, 1)

        return output, fused


class AttentionBasedTemporalEncoder(nn.Module):
    """
    Attention-based Temporal Encoder for Dense MoE Gating.
    Uses learnable CLS token with cross-attention to aggregate temporal information.

    Input: O_mkt_TA (batch, T_w, D) - Temporal market index embedding
    Output: market_context (batch, D) - Aggregated market condition representation
    """

    def __init__(self, config):
        super().__init__()
        self.D = config.mafia_D
        self.num_heads = getattr(config, "mafia_gating_num_heads", 4)
        self.dropout = getattr(config, "mafia_gating_dropout", 0.1)

        # Learnable CLS token - learns what information to extract from temporal sequence
        self.cls_token = nn.Parameter(th.randn(1, 1, self.D))
        nn.init.normal_(self.cls_token, std=0.02)

        # Multi-head cross-attention: CLS token queries the temporal sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.D,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.D)

    def forward(self, O_mkt_TA: th.Tensor) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            O_mkt_TA: (batch, T_w, D) - Temporal market index embedding

        Returns:
            market_context: (batch, D) - Aggregated market condition
            attn_weights: (batch, num_heads, 1, T_w) - Attention weights for visualization
        """
        batch_size = O_mkt_TA.size(0)

        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, 1, self.D)  # (batch, 1, D)

        # Cross-attention: CLS token attends to temporal sequence
        attn_output, attn_weights = self.cross_attn(
            query=cls_tokens,  # (batch, 1, D) - "What market info to extract?"
            key=O_mkt_TA,  # (batch, T_w, D) - "Where to look?"
            value=O_mkt_TA,  # (batch, T_w, D) - "What to extract?"
            need_weights=True,
            average_attn_weights=False,  # Return per-head weights
        )

        # Extract CLS token output and normalize
        market_context = attn_output.squeeze(1)  # (batch, D)
        market_context = self.layer_norm(market_context)

        return market_context, attn_weights


class TemporalConvolutionEncoder(nn.Module):
    """
    Temporal Convolution Encoder for Dense MoE Gating.
    Uses multi-scale 1D convolutions to capture patterns at different time scales.

    Input: O_mkt_TA (batch, T_w, D) - Temporal market index embedding
    Output: market_context (batch, D) - Aggregated market condition representation
    """

    def __init__(self, config):
        super().__init__()
        self.D = config.mafia_D
        self.dropout = getattr(config, "mafia_gating_dropout", 0.1)
        self.kernel_sizes = getattr(config, "mafia_gating_conv_kernels", [3, 5, 7])

        # Multi-scale 1D convolutions
        self.conv_layers = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            padding = kernel_size // 2  # Same padding
            self.conv_layers.append(
                nn.Conv1d(self.D, self.D, kernel_size=kernel_size, padding=padding)
            )

        # Batch normalization and activation
        self.conv_norm = nn.BatchNorm1d(self.D * len(self.kernel_sizes))
        self.conv_act = nn.GELU()

        # Projection network to output dimension
        self.proj = nn.Sequential(
            nn.Linear(
                self.D * len(self.kernel_sizes) * 2, self.D * 2
            ),  # *2 for max+avg pooling
            nn.LayerNorm(self.D * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.D * 2, self.D),
            nn.LayerNorm(self.D),
        )

    def forward(self, O_mkt_TA: th.Tensor) -> Tuple[th.Tensor, None]:
        """
        Args:
            O_mkt_TA: (batch, T_w, D) - Temporal market index embedding

        Returns:
            market_context: (batch, D) - Aggregated market condition
            None - No attention weights for conv encoder
        """
        batch_size = O_mkt_TA.size(0)

        # Reshape for Conv1D: (batch, D, T_w)
        x = O_mkt_TA.transpose(1, 2)  # (batch, D, T_w)

        # Multi-scale convolutions
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)  # (batch, D, T_w)
            conv_outputs.append(conv_out)

        # Concatenate multi-scale features
        multi_scale = th.cat(conv_outputs, dim=1)  # (batch, D*num_kernels, T_w)

        # Normalize and activate
        multi_scale = self.conv_norm(multi_scale)
        multi_scale = self.conv_act(multi_scale)

        # Temporal pooling: max and average
        max_pool = th.max(multi_scale, dim=2)[0]  # (batch, D*num_kernels)
        avg_pool = th.mean(multi_scale, dim=2)  # (batch, D*num_kernels)

        # Concatenate pooled features
        pooled = th.cat([max_pool, avg_pool], dim=1)  # (batch, D*num_kernels*2)

        # Project to D dimension
        market_context = self.proj(pooled)  # (batch, D)

        return market_context, None


class UnidirectionalLSTMEncoder(nn.Module):
    """
    Unidirectional LSTM Encoder for Dense MoE Gating.
    Uses forward-only LSTM for causal sequential modeling (Spec 3.6.3).

    CAUSALITY FIX: Uses bidirectional=False to ensure state h_t is only
    computed from past information x_{0...t}, not future data.

    Input: O_mkt_TA (batch, T_w, D) - Temporal market index embedding
    Output: market_context (batch, D) - Aggregated market condition representation
    """

    def __init__(self, config):
        super().__init__()
        self.D = config.mafia_D
        self.num_layers = getattr(config, "mafia_gating_lstm_layers", 2)
        self.dropout = getattr(config, "mafia_gating_dropout", 0.1)

        # Unidirectional LSTM (Spec 3.6.3: MUST NOT use bidirectional=True)
        self.lstm = nn.LSTM(
            input_size=self.D,
            hidden_size=self.D,  # Unidirectional: output dim = hidden_size
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,  # CAUSALITY FIX: Forward-only
            dropout=self.dropout if self.num_layers > 1 else 0,
        )

        # Temporal attention for weighted aggregation
        self.use_attention = True
        if self.use_attention:
            self.attn_proj = nn.Linear(self.D, 1)

        # Post-processing MLP
        self.refine = nn.Sequential(
            nn.Linear(self.D, self.D * 2),
            nn.LayerNorm(self.D * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.D * 2, self.D),
            nn.LayerNorm(self.D),
        )

        # Continuous state (Spec 3.6.2): carry (h_t, c_t) across timesteps
        self._cached_state: Optional[Tuple[th.Tensor, th.Tensor]] = None
        self._reset_to_zero: bool = False

    def reset_state(
        self, batch_size: Optional[int] = None, device: Optional[th.device] = None
    ):
        """
        Reset cached LSTM hidden/cell states (e.g., at episode boundaries).

        Spec 3.6.3: set (h, c) ← 0.
        If batch_size/device provided, materialize zero tensors now; otherwise
        defer to next forward (will zero-init based on incoming batch).
        """
        self._reset_to_zero = True
        if batch_size is not None:
            if device is None:
                device = (
                    self._cached_state[0].device
                    if self._cached_state is not None
                    else th.device("cpu")
                )
            h0 = th.zeros(self.num_layers, batch_size, self.D, device=device)
            c0 = th.zeros(self.num_layers, batch_size, self.D, device=device)
            self._cached_state = (h0, c0)
        else:
            self._cached_state = None

    def detach_state(self):
        """
        Detach cached hidden/cell states to truncate backprop graph (Spec 3.6.3).
        Call before feeding state into next step when running TBPTT/collect loop.
        """
        if self._cached_state is None:
            return
        h, c = self._cached_state
        self._cached_state = (h.detach(), c.detach())

    def _prepare_state(self, batch_size: int, device: th.device):
        """
        Validate cached state against incoming batch size/device.
        Returns state tuple or None when shape mismatch (fallback: cold start).
        """
        if self._reset_to_zero:
            # Explicit zero-init request from reset_state()
            self._reset_to_zero = False
            h0 = th.zeros(self.num_layers, batch_size, self.D, device=device)
            c0 = th.zeros(self.num_layers, batch_size, self.D, device=device)
            self._cached_state = (h0, c0)
            return self._cached_state
        if self._cached_state is None:
            return None
        h, c = self._cached_state
        if h.size(1) != batch_size:
            # Batch size changed (e.g., new trajectory) → reset state
            return None
        if h.device != device:
            h = h.to(device)
            c = c.to(device)
            self._cached_state = (h, c)
        return self._cached_state

    def forward(self, O_mkt_TA: th.Tensor) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            O_mkt_TA: (batch, T_w, D) - Temporal market index embedding

        Returns:
            market_context: (batch, D) - Aggregated market condition
            attn_weights: (batch, T_w, 1) - Temporal attention weights (if use_attention=True)
        """
        batch_size = O_mkt_TA.size(0)

        # TBPTT: ensure cached state is detached before reuse
        self.detach_state()

        # Unidirectional LSTM processing with continuous state carry-over (Spec 3.6.2)
        init_state = self._prepare_state(batch_size, O_mkt_TA.device)
        lstm_out, (h_n, c_n) = self.lstm(O_mkt_TA, init_state)
        # lstm_out: (batch, T_w, D) - hidden states for all timesteps
        # h_n: (num_layers, batch, D) - final hidden states
        # Cache new state (detach to avoid backprop across long horizons)
        self._cached_state = (h_n.detach(), c_n.detach())

        attn_weights = None
        if self.use_attention:
            # Weighted aggregation with learned attention
            attn_scores = self.attn_proj(lstm_out)  # (batch, T_w, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, T_w, 1)

            # Weighted sum
            market_context = th.sum(attn_weights * lstm_out, dim=1)  # (batch, D)
        else:
            # Simple: take last timestep
            market_context = lstm_out[:, -1, :]  # (batch, D)

        # Refine with MLP
        market_context = self.refine(market_context)

        return market_context, attn_weights


class DenseMoEGatingRouter(nn.Module):
    """
    Dense MoE Gating Router for computing expert weights.
    Uses configurable temporal encoder to extract market condition, then MLP to generate gate weights.

    Input: x_mkt_seq (batch, T, D_m) raw/feature market sequence OR O_mkt_TA (batch, T, D) pre-embedded
    Output: gate_weights (batch, num_experts=4) - Weights for each expert

    Temporal Context Augmentation (Spec 3.6):
        When enabled, input to gate_network is augmented from D to 3D:
        C_aug = [C_mkt^(t), C_bar_mkt, Delta_C_mkt]
        - C_mkt^(t): Current market context (D)
        - C_bar_mkt: Rolling mean over context window (D)
        - Delta_C_mkt: Drift = C_mkt^(t) - C_mkt^(t-W+1) (D)
    """

    def __init__(self, config, num_experts=4):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.D = config.mafia_D
        self.dropout = getattr(config, "mafia_gating_dropout", 0.1)
        self.mkt_input_dim = getattr(config, "mafia_M_mkt", self.D)

        # Temporal Context Augmentation (Spec 3.6)
        self.use_temporal_augmentation = getattr(
            config, "router_use_temporal_augmentation", True
        )
        self.augmented_dim = self.D * 3 if self.use_temporal_augmentation else self.D

        # Select temporal encoder based on config
        encoder_type = getattr(
            config, "mafia_gating_encoder_type", "attention_based_aggregation"
        )

        if encoder_type == "attention_based_aggregation":
            self.temporal_encoder = AttentionBasedTemporalEncoder(config)
        elif encoder_type == "temporal_convolution":
            self.temporal_encoder = TemporalConvolutionEncoder(config)
        elif encoder_type in ("lstm", "unidirectional_lstm"):
            self.temporal_encoder = UnidirectionalLSTMEncoder(config)
        else:
            raise ValueError(
                f"Unknown gating encoder type: {encoder_type}. "
                f"Must be one of: 'attention_based_aggregation', 'temporal_convolution', 'lstm'"
            )
        self._stateful_encoder = isinstance(
            self.temporal_encoder, UnidirectionalLSTMEncoder
        )

        # Projection for market sequence when feature dim != D
        if self.mkt_input_dim != self.D:
            self.mkt_proj = nn.Linear(self.mkt_input_dim, self.D)
        else:
            self.mkt_proj = nn.Identity()
        self._temp_proj = None

        # Gate weight generator MLP
        # Input dimension: augmented_dim (3D if temporal augmentation, else D)
        self.gate_network = nn.Sequential(
            nn.Linear(self.augmented_dim, self.D * 2),
            nn.LayerNorm(self.D * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.D * 2, num_experts),
            nn.Softmax(dim=-1),  # Normalize weights to sum to 1
        )
        
        # Smart Init: Bias towards Technical Agent (Expert 0)
        # Experts: 0=Tech, 1=DC(Slow), 2=DC(Med), 3=DC(Fast)
        # Set bias of final linear layer to give slight preference to Tech
        with th.no_grad():
             # gate_network[-2] is the Linear layer before Softmax
             # Bias init: [1.0, 0.0, 0.0, 0.0] -> Softmax -> [~0.47, ~0.17, ~0.17, ~0.17]
             self.gate_network[-2].bias.zero_()
             self.gate_network[-2].bias[0] = 1.0

        # Signal Decoupling Adapters (Task-Specific Views)
        # Selection Adapter: Projects raw context for Gating (Selection Task)
        self.selection_adapter = nn.Linear(self.D, self.D)

    def reset_temporal_state(self):
        """Reset temporal encoder state (Spec 3.6.2) when encoder is stateful."""
        if self._stateful_encoder:
            self.temporal_encoder.reset_state()

    def detach_temporal_state(self):
        """Detach temporal encoder state to enforce TBPTT boundaries (Spec 3.6.3)."""
        if self._stateful_encoder and hasattr(self.temporal_encoder, "detach_state"):
            self.temporal_encoder.detach_state()

    def _maybe_project(self, x: th.Tensor) -> th.Tensor:
        """
        Project market sequence to model dimension if needed.
        """
        batch, seq_len, dim = x.shape
        if dim == self.D:
            return x
        if dim == self.mkt_input_dim:
            return self.mkt_proj.to(x.device)(x)
        # Fallback projection when runtime dim differs from configured dim
        if self._temp_proj is None or self._temp_proj.in_features != dim:
            self._temp_proj = nn.Linear(dim, self.D).to(x.device)
        return self._temp_proj(x)

    def forward(
        self,
        x_mkt_seq: Optional[th.Tensor] = None,  # (batch, T, D_m)
        O_mkt_TA: Optional[th.Tensor] = None,  # (batch, T, D)
        context_buffer: Optional[
            th.Tensor
        ] = None,  # (W, D) or (batch, W, D) - historical context
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args:
            x_mkt_seq: Optional (batch, T, D_m) raw/feature market sequence (preferred)
            O_mkt_TA: Optional (batch, T, D) pre-embedded market temporal features
            context_buffer: Optional (W, D) or (batch, W, D) historical market_context buffer
                           Used for temporal augmentation (Spec 3.6)

        Returns:
        Returns:
            gate_weights: (batch, num_experts) - Weights for each expert (sum to 1)
            raw_context: (batch, D) - Raw market latent state (shared source)
        """
        device = next(self.parameters()).device
        if x_mkt_seq is None and O_mkt_TA is None:
            # Fallback: uniform weights when market index is not available
            batch_size = 1  # Will be broadcasted if needed
            gate_weights = (
                th.ones(batch_size, self.num_experts, device=device) / self.num_experts
            )
            raw_context = th.zeros(batch_size, self.D, device=device)
            return gate_weights, raw_context

        market_tokens = None
        if x_mkt_seq is not None:
            market_tokens = self._maybe_project(x_mkt_seq)
        elif O_mkt_TA is not None:
            market_tokens = O_mkt_TA

        # Extract RAW market condition from temporal encoder
        # This is the shared latent state before task-specific projection
        raw_context, _ = self.temporal_encoder(market_tokens)  # (batch, D)

        # Apply Selection Adapter to create "Selection View" of the context
        context_selection = self.selection_adapter(raw_context)  # (batch, D)

        # ===== Temporal Context Augmentation (Spec 3.6) =====
        # Augment gate input: C_aug = [C_sel^(t), C_bar_sel, Delta_C_sel]
        if self.use_temporal_augmentation:
            if context_buffer is not None:
                # Ensure buffer has batch dimension
                if context_buffer.dim() == 2:  # (W, D)
                    context_buffer = context_buffer.unsqueeze(0)  # (1, W, D)

                # Expand buffer to match batch size if needed
                batch_size = raw_context.size(0)
                if context_buffer.size(0) == 1 and batch_size > 1:
                    context_buffer = context_buffer.expand(batch_size, -1, -1)

                # context_buffer stores RAW contexts (history)
                # We need to project history to Selection View for consistency
                # buffer shape: (batch, W, D)
                history_raw = context_buffer[:, 1:, :]  # (batch, W-1, D)
                current_raw = raw_context.unsqueeze(1)  # (batch, 1, D)
                rolling_raw = th.cat([history_raw, current_raw], dim=1) # (batch, W, D)

                # Compute statistics on RAW data first
                C_bar_raw = rolling_raw.mean(dim=1)  # (batch, D)
                C_oldest_raw = rolling_raw[:, 0, :]  # (batch, D)
                Delta_C_raw = raw_context - C_oldest_raw  # (batch, D)

                # Project statistics to Selection View
                C_bar_sel = self.selection_adapter(C_bar_raw)
                Delta_C_sel = self.selection_adapter(Delta_C_raw)

                # Concatenate: C_aug = [C_sel, C_bar_sel, Delta_C_sel]
                gate_input = th.cat(
                    [context_selection, C_bar_sel, Delta_C_sel], dim=-1
                )  # (batch, 3D)
            else:
                # No buffer available (cold start)
                zeros = th.zeros_like(context_selection)
                gate_input = th.cat(
                    [context_selection, zeros, zeros], dim=-1
                )  # (batch, 3D)
        else:
            # Augmentation disabled
            gate_input = context_selection

        # Sanitize gate_input to prevent NaN propagation (defensive)
        # NaN can occur from edge-case data or numerical instability
        if th.isnan(gate_input).any() or th.isinf(gate_input).any():
            gate_input = th.nan_to_num(gate_input, nan=0.0, posinf=1.0, neginf=-1.0)
            gate_input = th.clamp(gate_input, min=-10.0, max=10.0)

        # Generate gate weights from (augmented) selection context
        gate_weights = self.gate_network(gate_input)  # (batch, num_experts)

        # Sanitize raw_context for downstream use
        if th.isnan(raw_context).any() or th.isinf(raw_context).any():
            raw_context = th.nan_to_num(raw_context, nan=0.0, posinf=1.0, neginf=-1.0)

        return gate_weights, raw_context


class DirectionHead(nn.Module):
    """
    Wide & Deep Late Fusion Direction Head (Spec 3.5 v2.1)

    Uses Late Fusion architecture to solve Information Bottleneck problem.
    Explicit signals bypass ResBlock and fuse directly at classification layer.

    Architecture (Spec 3.5.2):
    - Deep Path: X_latent = [C_mkt, ΔC_mkt] → InputProj(2D→D) → ResBlock → h_deep
    - Wide Path: X_explicit (4 signals) → bypass (direct anchoring)
    - Late Fusion: H_final = Concat(h_deep, X_explicit) → Classifier(D+4 → 3)

    Output: logits for 3 classes (Bear/Side/Bull)
    """

    def __init__(self, config):
        super().__init__()
        self.D = config.mafia_D
        self.num_classes = 3
        self.dropout_rate = getattr(config, "direction_head_dropout", 0.2)

        # Explicit signals (Spec §3.5.1 v2.1): 6 dims (Wide Path)
        # 1. DC_Event_Flag: Structural break signal from Market-DC Agent
        # 2. Breadth_Gap: avg(RSI_stocks) - RSI_index ("Xanh vỏ đỏ lòng" detection)
        # 3. Div_Signal: RSI slope vs Price slope divergence (reversal signal)
        # 4. Signed_VPI_Zscore: Volume-price efficiency (money flow)
        # 5. Vol_Std20: Rolling volatility (Risk)
        # 6. Drawdown60: Rolling drawdown (Pain)
        self.explicit_dim = getattr(config, "mafia_explicit_dim", 6)

        # Deep Path: Input Projection for LATENT only (2D -> D)
        self.latent_dim = self.D * 2  # C_mkt(D) + Delta_C(D)
        self.input_proj = nn.Linear(self.latent_dim, self.D)

        # Residual Block (Spec 3.5.2)
        self.res_ln = nn.LayerNorm(self.D)
        self.res_fc1 = nn.Linear(self.D, self.D)
        self.res_fc2 = nn.Linear(self.D, self.D)
        self.res_dropout = nn.Dropout(self.dropout_rate)

        # Wide Path: No transformation (explicit signals bypass to fusion)

        # Classification Head: Late Fusion (D + 4 -> 3)
        self.classifier = nn.Linear(self.D + self.explicit_dim, self.num_classes)

        # Initialize bias to log-priors matching data distribution
        self._init_classifier_bias()
        # Smart-initialize explicit signal weights (hot-start for Wide Path)
        self._init_explicit_signal_weights()

    def _init_classifier_bias(self):
        """
        Initialize classifier bias to log-priors matching data distribution.

        VNINDEX distribution: Bear=22%, Side=44%, Bull=34%
        log(0.22)=-1.514, log(0.44)=-0.821, log(0.34)=-1.079

        This gives model prior knowledge about class frequencies,
        helping faster convergence without biasing toward majority class.
        """
        with th.no_grad():
            # Log-prior initialization (matches data distribution)
            self.classifier.bias.copy_(th.tensor([-1.514, -0.821, -1.079]))

    def _init_explicit_signal_weights(self):
        """
        Smart-initialize Wide Path weights to respect signal logic.
        SIGNAL ORDER (Input Tensor):
        0: Vol_Std20
        1: DC_Event_Flag
        2: Breadth_Gap
        3: Div_Signal
        4: Signed_VPI_Zscore (only if D+4 exists)
        5: Drawdown60 (only if D+5 exists)

        Convention: -1 (Bearish), +1 (Bullish)

        Weight logic:
        - Bear Class (0): Negative weight (so -1 input -> + logit)
        - Bull Class (2): Positive weight (so +1 input -> + logit)
        """
        with th.no_grad():
            # Indices in the concatenated input (Deep=0..D-1, Wide=D..D+5)

            # 0. Vol_Std20 (Index D+0) - Risk/Fear
            # High Vol -> Bearish Bias
            self.classifier.weight[0, self.D + 0] = 0.2   # High Vol increases Bear prob
            self.classifier.weight[2, self.D + 0] = -0.2  # High Vol decreases Bull prob

            # 1. DC Event Flag (Index D+1) - Strongest Signal (Trend Break)
            self.classifier.weight[:, self.D + 1] = 0.0

            # 2. Breadth_Gap (Index D+2) - "Xanh vỏ đỏ lòng"
            self.classifier.weight[0, self.D + 2] = -0.3 # Neg Gap -> Bear
            self.classifier.weight[2, self.D + 2] = 0.3  # Pos Gap -> Bull

            # 3. Div_Signal (Index D+3) - Reversal
            self.classifier.weight[0, self.D + 3] = -0.5
            self.classifier.weight[2, self.D + 3] = 0.5

            # 4. Signed_VPI_Zscore (Index D+4) - Money Flow
            if self.explicit_dim > 4:
                self.classifier.weight[0, self.D + 4] = -0.3
                self.classifier.weight[2, self.D + 4] = 0.3

            # 5. Drawdown60 (Index D+5) - Trend Following Logic (UPDATED)
            if self.explicit_dim > 5:
                # Drawdown is negative value (e.g. -0.2).
                # Bear Impact: Weight(-0.2) * Value(-0.2) = +0.04 -> Increases Bear score (Correct: Deep DD -> Bearish)
                self.classifier.weight[0, self.D + 5] = -0.2
                # Bull Impact: Weight(0.2) * Value(-0.2) = -0.04 -> Decreases Bull score (Correct: Deep DD -> Less Bullish)
                self.classifier.weight[2, self.D + 5] = 0.2

            # Ensure Side Class (1) remains neutral to these signals initially
            self.classifier.weight[1, self.D : self.D + self.explicit_dim] = 0.0

    def forward(
        self,
        c_mkt: th.Tensor,
        delta_c_mkt: th.Tensor,
        explicit_signals: th.Tensor,
    ) -> th.Tensor:
        """
        Wide & Deep Late Fusion forward pass (Spec 3.5.2 v2.1).

        Args:
            c_mkt: (B, D) - Current market context from Temporal Encoder
            delta_c_mkt: (B, D) - Momentum/Velocity from Shared Buffer (Spec 3.6)
            explicit_signals: (B, 6) - Wide path signals:
                [Vol_Std20, DC_Event_Flag, Breadth_Gap, Div_Signal, Signed_VPI_Zscore, Drawdown60]

        Returns:
            logits: (B, 3) - Bear/Side/Bull classification logits
        """
        # === 0. Signal Scaling (Robustness) ===
        # Volatility ~ 0.015, Gap ~ 10.0. huge discrepancy.
        # Scale to ~ [-1, 1] or [0, 1] range.
        c_vol = 100.0   # 0.01 -> 1.0
        c_gap = 0.1     # 10.0 -> 1.0
        c_vpi = 0.5     # 2.0 -> 1.0

        # Create scaled copy to avoid modifying original tensor inplace (safety)
        signals_scaled = explicit_signals.clone()
        
        # Vol_Std20 (idx 0)
        signals_scaled[:, 0] = signals_scaled[:, 0] * c_vol
        # Breadth_Gap (idx 2)
        signals_scaled[:, 2] = signals_scaled[:, 2] * c_gap
        # Signed_VPI (idx 4)
        if signals_scaled.shape[1] > 4:
            signals_scaled[:, 4] = signals_scaled[:, 4] * c_vpi

        # === 1. Deep Path: Process latent context ===
        x_latent = th.cat([c_mkt, delta_c_mkt], dim=-1)  # (B, 2D)
        
        # [ROBUSTNESS] Sanitize latent input
        if th.isnan(x_latent).any() or th.isinf(x_latent).any():
             x_latent = th.nan_to_num(x_latent, nan=0.0, posinf=1.0, neginf=-1.0)

        h = self.input_proj(x_latent)  # (B, D)

        # Residual Block with Skip Connection
        h_res = self.res_ln(h)
        h_res = F.gelu(self.res_fc1(h_res))
        h_res = self.res_dropout(h_res)
        h_res = self.res_fc2(h_res)
        h_deep = h + h_res  # (B, D)

        # === 2. Wide Path: Direct bypass (no transformation) ===
        x_wide = signals_scaled  # (B, 6)
        
        # [ROBUSTNESS] Sanitize wide input
        if th.isnan(x_wide).any() or th.isinf(x_wide).any():
             x_wide = th.nan_to_num(x_wide, nan=0.0, posinf=1.0, neginf=-1.0)
             
        # === 3. Late Fusion ===
        h_final = th.cat([h_deep, x_wide], dim=-1)  # (B, D+6)

        # === 4. Classification ===
        logits = self.classifier(h_final)  # (B, 3)
        
        # [ROBUSTNESS] Final Output Guard
        # If logits are still NaN (e.g. from weight corruption), zero them out
        if th.isnan(logits).any():
             logits = th.nan_to_num(logits, nan=0.0)

        return logits


class RiskHead(nn.Module):
    """
    Wide & Deep Risk Head (Spec v2.1)

    Architecture:
    - Deep Path: [C_mkt_macro, Delta_C_mkt] -> MLP -> h_deep
    - Wide Path: Explicit Signals (6 dims) -> Concat -> Output
    """

    def __init__(self, config):
        super().__init__()
        self.D = config.mafia_D
        self.explicit_dim = getattr(config, "mafia_explicit_dim", 6)

        # Deep Path
        self.input_dim = self.D * 2  # C_mkt + Delta_C
        self.deep_net = nn.Sequential(
            nn.Linear(self.input_dim, self.D),
            nn.LayerNorm(self.D),
            nn.GELU(),
            nn.Linear(self.D, self.D),
            nn.GELU(),
        )

        # Late Fusion
        self.fusion_net = nn.Linear(self.D + self.explicit_dim, 1)

        # Smart Initalization for Wide Path (Spec 3.5.1 v2.1)
        self._init_explicit_signal_weights()

    def _init_explicit_signal_weights(self):
        """
        Smart-initialize Wide Path weights for Risk Head.
        Reflects inverse relationship between volatility/risk signals and eta (Risk Tolerance).

        Target: High Risk Signal -> Low Eta (Defensive)
        Weights should be NEGATIVE for Risk Factors.
        """
        with th.no_grad():
            # Fusion net input: [Deep(D), Wide(6)] -> Output(1)
            # Wide signals start at index D.

            # 0. Vol_Std20 (Index D+0) - High Vol -> Low Eta
            self.fusion_net.weight[0, self.D + 0] = -0.5

            # 1. DC_Event_Flag (Index D+1) - Trend Break -> Low Eta
            self.fusion_net.weight[0, self.D + 1] = -0.3

            # 2. Breadth_Gap (Index D+2) - Negative Gap (Bear Trap) -> Low Eta
            # Gap is usually negative in bad times?
            # Metric: avg(RSI_stocks) - RSI_index.
            # If Indx high but stocks low -> Gap < 0. This is Bear Trap.
            # So Gap < 0 -> Low Eta. Gap > 0 -> High Eta.
            # Positive weight propagates sign correctly.
            self.fusion_net.weight[0, self.D + 2] = 0.3

            # 3. Div_Signal (Index D+3) - Reversal
            # -1 (Bearish Div) -> Low Eta. +1 (Bullish Div) -> High Eta.
            self.fusion_net.weight[0, self.D + 3] = 0.3

            # 4. Signed_VPI_Zscore (Index D+4)
            if self.explicit_dim > 4:
                 self.fusion_net.weight[0, self.D + 4] = 0.3

            # 5. Drawdown60 (Index D+5) - Pain
            # Drawdown is negative (e.g. -0.15). Deep drawback -> Low Eta.
            # Weight > 0 means (-0.15 * 0.5) -> negative impact. Correct.
            if self.explicit_dim > 5:
                 self.fusion_net.weight[0, self.D + 5] = 0.5

            # Initialize bias to 0.0 (Neutral start) or slight positive (1.0 base)
            # But output is passed to tanh... eta = base + amp * tanh(raw).
            # So 0.0 -> tanh(0)=0 -> eta=base. Correct.
            self.fusion_net.bias.fill_(0.0)

    def forward(
        self,
        c_mkt_macro: th.Tensor,
        delta_c_mkt: th.Tensor,
        explicit_signals: th.Tensor,
    ) -> th.Tensor:
        """
        Args:
            c_mkt_macro: (B, D) - Macro-adapted context
            delta_c_mkt: (B, D) - Momentum
            explicit_signals: (B, 6) - [Vol20, DC, Breadth, Div, VPI, DD60]
        """
        # === 0. Signal Scaling (Robustness) ===
        c_vol = 100.0
        c_gap = 0.1
        c_vpi = 0.5
        
        signals_scaled = explicit_signals.clone()
        signals_scaled[:, 0] = signals_scaled[:, 0] * c_vol
        signals_scaled[:, 2] = signals_scaled[:, 2] * c_gap
        if signals_scaled.shape[1] > 4:
            signals_scaled[:, 4] = signals_scaled[:, 4] * c_vpi

        # Deep Path
        x_deep = th.cat([c_mkt_macro, delta_c_mkt], dim=-1)
        
        # [ROBUSTNESS] Sanitize deep input
        if th.isnan(x_deep).any() or th.isinf(x_deep).any():
             x_deep = th.nan_to_num(x_deep, nan=0.0, posinf=1.0, neginf=-1.0)
             
        h_deep = self.deep_net(x_deep)


        # Wide Path & Fusion
        x_wide = signals_scaled
        # [ROBUSTNESS] Sanitize wide input
        if th.isnan(x_wide).any() or th.isinf(x_wide).any():
             x_wide = th.nan_to_num(x_wide, nan=0.0, posinf=1.0, neginf=-1.0)
             
        h_final = th.cat([h_deep, x_wide], dim=-1)
        eta_raw = self.fusion_net(h_final)

        return eta_raw.squeeze(-1)


class DenseMoESignalGenerator(nn.Module):
    """
    Dense Mixture of Experts Signal Generator.
    Uses market-index agent O_mkt_TA to compute gating weights for 4 stock experts.

    Architecture:
    - 4 Stock Experts (Tech + 3 DC agents) provide outputs
    - Market-index agent provides O_mkt_TA for gating
    - Gating router computes weights based on market condition
    - Weighted combination of expert outputs
    - Gumbel-TopK selection for portfolio
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.D = config.mafia_D
        self.T_w = config.mafia_T_w
        self.K = config.mafia_top_k
        self.tau_gumbel = getattr(config, "mafia_gumbel_temperature", 1.0)
        self.num_direction_classes = 3

        # Dense MoE Gating Router (for 4 stock experts: Tech + 3 DC)
        self.gating_router = DenseMoEGatingRouter(config, num_experts=4)

        # Gumbel-TopK temperature (trainable or fixed)
        self.topk_temperature = nn.Parameter(
            th.tensor(self.tau_gumbel), requires_grad=False
        )

        # Boundary risk network (Refactored to RiskHead v2.1)
        # Uses Market Context + Explicit Signals (Wide & Deep)
        self.risk_head = RiskHead(config)

        # Direction classification head (Spec 3.5: Context-Augmented Residual Architecture)
        # Uses augmented input: X_dir = [C_mkt, Delta_C_mkt, Explicit_Signals]
        self.direction_head = DirectionHead(config)

        # Macro Adapter: Projects raw context for Direction/Risk (Macro Task)
        self.macro_adapter = nn.Linear(self.D, self.D)

        # Holding Bias (Learnable Inertia) - Spec "Memory Injection"
        # Bias added to logits of currently held stocks to encourage retention
        # Initialize to 0.01 (Smart Init) to encourage holding from start (helps convergence vs turnover penalty)
        self.holding_bias = nn.Parameter(th.tensor([0.01]))

    def reset_router_state(self):
        """Reset stateful components inside the gating router (LSTM hidden/cache)."""
        self.gating_router.reset_temporal_state()

    def detach_router_state(self):
        """Detach gating router state (TBPTT boundary)."""
        if hasattr(self.gating_router, "detach_temporal_state"):
            self.gating_router.detach_temporal_state()

    def forward(
        self,
        expert_outputs: list,  # [O_tech, O_dc1, O_dc2, O_dc3]
        expert_ta_outputs: list,  # [O_tech_TA, O_dc1_TA, O_dc2_TA, O_dc3_TA]
        x_mkt_seq: Optional[
            th.Tensor
        ] = None,  # Raw/feature market sequence for gating (batch, T, D_m)
        O_mkt_TA: Optional[
            th.Tensor
        ] = None,  # Optional pre-embedded market TA output for gating
        expert_st_embeddings: Optional[
            list
        ] = None,  # Optional per-stock embeddings per expert [(batch, N, D)]
        force_topk_indices: Optional[
            th.Tensor
        ] = None,  # Optional forced Top-K indices to lock membership
        router_context_buffer: Optional[
            th.Tensor
        ] = None,  # Historical market_context buffer for temporal augmentation (W, D)
        explicit_signals: Optional[
            th.Tensor
        ] = None,  # (batch, 2) [vol_shock_flag, dc_event_flag] for Direction Head (Spec 3.5)
        prev_holdings: Optional[
            th.Tensor
        ] = None,  # (batch, N) Binary mask of stocks held at t-1 (Memory Injection)
    ) -> Tuple[
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        Optional[th.Tensor],
        th.Tensor,
        th.Tensor,
    ]:

        """
        Dense MoE forward pass.

        Args:
            expert_outputs: List of 4 tensors, each (batch, N, 1) - Stock expert logits
            expert_ta_outputs: List of 4 tensors, each (batch, T_w, D) - Expert TA outputs
            x_mkt_seq: Optional (batch, T_w, D_m) - Market-index sequence for gating
            O_mkt_TA: Optional (batch, T_w, D) - Market-index TA output for gating

        Returns:
            market_vector: (batch, N) - Portfolio weights (Top-K assets)
            eta: (batch,) - Risk tolerance factor
            market_scores_full: (batch, N) - Full softmax scores for all assets
            market_context: (batch, D) - Market condition embedding from gating encoder
            sigma_logits: (batch, 3) - Logits for market direction classification (up/hold/down)
            topk_indices: (batch, K) - Indices of selected assets
            topk_embeddings: Optional (batch, K, D) - Embeddings of selected assets (for RL state)
            topk_scores: (batch, K) - Market weights on selected assets
            market_logits: (batch, N) - Raw logits before softmax (for stable Loss calc)
            force_topk_indices: Optional tensor to override Top-K membership (keeps weights sorted by logits)
        """
        # Input validation
        num_experts = len(expert_outputs)
        if num_experts != 4:
            raise ValueError(
                f"DenseMoE expects exactly 4 stock experts (Tech + 3 DC), got {num_experts}"
            )

        batch_size = expert_outputs[0].size(0)
        N = expert_outputs[0].size(1)
        device = expert_outputs[0].device

        # Step 1: Compute gate weights from market condition
        # Pass context_buffer for temporal augmentation (Spec 3.6)
        # NOTE: returns raw_context (latent), not projected context
        gate_weights, raw_context = self.gating_router(
            x_mkt_seq=x_mkt_seq,
            O_mkt_TA=O_mkt_TA,
            context_buffer=router_context_buffer,
        )  # (batch, 4), (batch, D)

        # Expand gate_weights for batch if needed (when O_mkt_TA was None)
        if gate_weights.size(0) == 1 and batch_size > 1:
            gate_weights = gate_weights.expand(batch_size, -1)
            raw_context = raw_context.expand(batch_size, -1)

        # Step 2: Stack expert outputs and apply gating
        expert_logits_stacked = th.stack(
            [o.squeeze(-1) for o in expert_outputs], dim=1
        )  # (batch, 4, N)

        # Weighted combination using gate weights
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch, 4, 1)
        market_logits = th.sum(
            gate_weights_expanded * expert_logits_stacked, dim=1
        )  # (batch, N)

        # Apply Holding Bias (Memory Injection)
        # Logit_new = Logit_old + (Bias * Is_Held)
        if prev_holdings is not None:
            # Ensure shape match
            if prev_holdings.shape != market_logits.shape:
               raise ValueError(f"prev_holdings shape {prev_holdings.shape} mismatch with logits {market_logits.shape}")
            
            # Add bias (broadcast scalar * tensor)
            market_logits = market_logits + (self.holding_bias * prev_holdings)

        # Optional: fuse per-stock ST embeddings from experts using gate weights
        fused_stock_embedding = None
        if expert_st_embeddings is not None:
            if len(expert_st_embeddings) != num_experts:
                raise ValueError(
                    f"Expected {num_experts} expert ST embeddings, got {len(expert_st_embeddings)}"
                )
            stacked_emb = th.stack(expert_st_embeddings, dim=1)  # (batch, 4, N, D)
            gate_weights_exp = gate_weights.unsqueeze(-1).unsqueeze(
                -1
            )  # (batch, 4, 1, 1)
            fused_stock_embedding = th.sum(
                gate_weights_exp * stacked_emb, dim=1
            )  # (batch, N, D)

        # Step 3: Gumbel-TopK selection
        # Use dynamic temperature: tau_gumbel is updated by MAFIAObserver.update_temperature()
        # In inference mode, use lower temperature for sharper decisions
        training_mode = self.training
        hard_inference = getattr(self.config, "mafia_hard_topk_inference", True)

        if training_mode:
            # Training: use annealed temperature (updated externally)
            temperature = max(self.tau_gumbel, 1e-6)
        else:
            # Inference: use fixed low temperature for sharper decisions
            temperature = max(
                getattr(self.config, "mafia_gumbel_temp_inference", 0.1), 1e-6
            )

        # market_scores_full: Full softmax WITHOUT temperature (pure model distribution)
        # Per spec 3.3: "market_scores_full = softmax toàn bộ (không mask)"
        # Temperature τ is ONLY for Gumbel-TopK selection, NOT for full scores
        # Sanitize market_logits before softmax to prevent NaN
        if th.isnan(market_logits).any() or th.isinf(market_logits).any():
            market_logits = th.nan_to_num(market_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        market_scores_full = F.softmax(market_logits, dim=-1)  # (batch, N)

        if training_mode or not hard_inference:
            # Soft Gumbel-TopK with noise
            uniform = th.rand_like(market_logits).clamp_(min=1e-8, max=1 - 1e-8)
            gumbel_noise = -th.log(-th.log(uniform))
            logits_for_topk = (market_logits + gumbel_noise) / temperature
        else:
            # Hard TopK (deterministic)
            logits_for_topk = market_logits / temperature

        # Select top-K assets (optionally override with provided indices)
        top_k = min(self.K, N)
        topk_vals = None
        fallback_topk = None
        if force_topk_indices is not None:
            # Normalize shape: (batch, K)
            if not isinstance(force_topk_indices, th.Tensor):
                force_topk_indices = th.as_tensor(force_topk_indices, device=device)
            if force_topk_indices.dim() == 1:
                force_topk_indices = force_topk_indices.unsqueeze(0)
            if force_topk_indices.size(0) == 1 and batch_size > 1:
                force_topk_indices = force_topk_indices.expand(batch_size, -1)
            force_topk_indices = force_topk_indices.long().clamp(min=0, max=N - 1)
            if force_topk_indices.size(1) < top_k:
                if fallback_topk is None:
                    fallback_topk = th.topk(logits_for_topk, k=top_k, dim=-1).indices
                pad = top_k - force_topk_indices.size(1)
                force_topk_indices = th.cat(
                    [force_topk_indices, fallback_topk[:, :pad]], dim=1
                )
            elif force_topk_indices.size(1) > top_k:
                force_topk_indices = force_topk_indices[:, :top_k]
            topk_indices = force_topk_indices
            topk_vals = th.gather(logits_for_topk, 1, topk_indices)
        else:
            topk_vals, topk_indices = th.topk(logits_for_topk, k=top_k, dim=-1)

        # Create mask for selected indices
        mask = th.zeros_like(market_logits, dtype=th.bool)
        mask.scatter_(1, topk_indices, True)

        # Apply mask and softmax to get portfolio weights
        masked_logits = logits_for_topk.masked_fill(~mask, float("-inf"))
        market_vector = F.softmax(masked_logits, dim=-1)

        # Sanitize NaN from softmax (can occur with all -inf inputs)
        if th.isnan(market_vector).any():
            market_vector = th.nan_to_num(market_vector, nan=0.0)

        # Enforce hard support at inference
        if not training_mode and hard_inference:
            market_vector = market_vector.masked_fill(~mask, 0.0)

        # Step 4: Build portfolio context (weighted Top-K embeddings) and risk input
        eps = 1e-8
        # Renormalize weights strictly on selected assets for stability
        mv_weights = market_vector * mask.float()
        mv_weights = mv_weights / (mv_weights.sum(dim=1, keepdim=True) + eps)

        topk_embeddings = None
        portfolio_context = th.zeros(batch_size, self.D, device=device)
        if fused_stock_embedding is not None:
            portfolio_context = th.sum(
                mv_weights.unsqueeze(-1) * fused_stock_embedding, dim=1
            )  # (batch, D)
            # Gather explicit Top-K embeddings for downstream consumers (not returned separately here)
            topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, top_k, self.D)
            topk_embeddings = th.gather(fused_stock_embedding, 1, topk_indices_exp)
        topk_scores = th.gather(market_vector, 1, topk_indices)  # (batch, K)

        # === Risk Head (Wide & Deep v2.1) ===
        # 1. Apply Macro Adapter to raw_context -> Separation of Concerns
        context_macro = self.macro_adapter(raw_context)  # (batch, D)

        # 2. Compute Delta_C_mkt (Momentum/Velocity) using macro context
        delta_c_mkt = th.zeros_like(context_macro)  # Default: zeros (cold start)

        if router_context_buffer is not None:
            # router_context_buffer contains RAW contexts
            # (W, D) or (batch, W, D)
            if router_context_buffer.dim() == 2:  # (W, D)
                c_oldest_raw = router_context_buffer[0:1].expand(batch_size, -1)  # (B, D)
            else:  # (batch, W, D)
                c_oldest_raw = router_context_buffer[:, 0, :]  # (B, D)
            
            # Project oldest raw -> oldest macro
            # Spec 3.6 requires Delta = C_curr - C_(t-W+1).
            # Buffer contains [C_(t-W), C_(t-W+1), ... C_(t-1)]
            # Index 0 is C_(t-W). Index 1 is C_(t-W+1).
            # We use index 1 to align with Gating Router and Spec.
            if router_context_buffer.size(1) > 1:
                # Use C_(t-W+1) if window is large enough
                if router_context_buffer.dim() == 2:  # (W, D)
                    c_oldest_raw = router_context_buffer[1:2].expand(batch_size, -1)
                else:  # (batch, W, D)
                    c_oldest_raw = router_context_buffer[:, 1, :]
            else:
                # Fallback for short windows (W=1)
                if router_context_buffer.dim() == 2:
                    c_oldest_raw = router_context_buffer[0:1].expand(batch_size, -1)
                else:
                    c_oldest_raw = router_context_buffer[:, 0, :]
            
            c_oldest_macro = self.macro_adapter(c_oldest_raw)
            delta_c_mkt = context_macro - c_oldest_macro  # Macro momentum

        # Require explicit signals - Spec §3.5.1 Updated (6 signals)
        if explicit_signals is None:
            raise ValueError("explicit_signals is required for Risk/Direction Head.")

        # 3. Risk Prediction
        eta_raw = self.risk_head(
            context_macro, delta_c_mkt, explicit_signals
        )
        eta_base = getattr(self.config, "mafia_eta_base", 1.0)
        eta_amp = getattr(self.config, "mafia_eta_amplitude", 0.3)
        eta_min = getattr(self.config, "mafia_eta_min", eta_base - eta_amp)
        eta_max = getattr(self.config, "mafia_eta_max", eta_base + eta_amp)
        eta = eta_base + eta_amp * th.tanh(eta_raw)
        eta = th.clamp(eta, min=eta_min, max=eta_max)

        # Market direction logits (Spec 3.5: Context-Augmented Residual)
        sigma_logits = self.direction_head(
            context_macro, delta_c_mkt, explicit_signals
        )  # (batch, 3)

        return (
            market_vector,  # (batch, N)
            eta,  # (batch,)
            market_scores_full,  # (batch, N)
            sigma_logits,  # (batch, 3) - Direction classification
            raw_context,  # (batch, D) - Raw Latent State (for buffer continuity)
            topk_indices,  # (batch, K)
            topk_embeddings,  # (batch, K, D) or None
            topk_scores,  # (batch, K)
            market_logits,  # (batch, N) - RAW LOGITS
        )


class MAFIAModel(nn.Module):
    """
    Complete MAFIA Model.
    Consists of 1 Technical Agent + 3 DC Agents + 1 Market-index Agent (optional).
    """

    def __init__(self, config, action_dim: int):
        super().__init__()
        self.config = config
        self.N = action_dim  # Number of assets
        self.T_w = config.mafia_T_w
        self.D = config.mafia_D

        # Feature processor
        from RL_controller.mafia_feature_processor import MAFIAFeatureProcessor

        self.feature_processor = MAFIAFeatureProcessor(config)

        # Technical Agent (i=0) - Use actual N (action_dim) instead of config.topK
        self.tech_csa = CSAModule(config, agent_type="tech")
        self.tech_ta = TAModule(config, agent_type="tech", N=self.N)
        self.tech_st_fusion = STFusionModule(self.D)

        # DC Agents (i=1,2,3) - Each agent has its own CSA/TA weights
        self.dc_csa_list = nn.ModuleList(
            [
                CSAModule(config, agent_type="dc")
                for _ in range(len(self.config.mafia_DC_thresholds))
            ]
        )
        self.dc_ta_list = nn.ModuleList(
            [
                TAModule(config, agent_type="dc", shared_mlp=None, N=self.N)
                for _ in range(len(self.config.mafia_DC_thresholds))
            ]
        )
        self.dc_st_fusion_list = nn.ModuleList(
            [STFusionModule(self.D) for _ in range(3)]
        )

        # Market-index Agent (VNINDEX) - TA module for gating
        use_market_index = getattr(config, "mafia_use_market_index_agent", True)
        if use_market_index:
            self.mkt_ta = TAModule(
                config, agent_type="mkt", N=1
            )  # Only TA, N=1 for single asset, M=19
            # Market-DC Agent: TA modules per DC threshold (N=1)
            self.mkt_dc_ta_list = nn.ModuleList(
                [
                    TAModule(config, agent_type="dc", N=1)
                    for _ in range(len(self.config.mafia_DC_thresholds))
                ]
            )
        else:
            self.mkt_ta = None
            self.mkt_dc_ta_list = None

        # Dense MoE Signal Generator
        self.signal_generator = DenseMoESignalGenerator(config)

    def reset_temporal_state(self):
        """
        Reset temporal states for components that maintain running context
        (e.g., LSTM inside the gating router).
        """
        if hasattr(self.signal_generator, "reset_router_state"):
            self.signal_generator.reset_router_state()

    def detach_temporal_state(self):
        """
        Detach temporal states to enforce TBPTT boundaries without losing memory.
        """
        if hasattr(self.signal_generator, "detach_router_state"):
            self.signal_generator.detach_router_state()

    def forward(
        self,
        ochlv_data: th.Tensor,
        market_index_ochlv_data: Optional[th.Tensor] = None,
        force_topk_indices: Optional[th.Tensor] = None,
        router_context_buffer: Optional[
            th.Tensor
        ] = None,  # (W, D) buffer for temporal augmentation
        explicit_signals: Optional[
            th.Tensor
        ] = None,  # (batch, 2) [vol_shock, dc_flag] for Direction Head (Spec 3.5)
        prev_holdings: Optional[th.Tensor] = None,  # Memory Injection
    ) -> Tuple[
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        Optional[th.Tensor],
        th.Tensor,
        th.Tensor,
    ]:
        """
        Forward pass through MAFIA model with Dense MoE.

        Args:
            ochlv_data: (batch, N, 5, T_w) - Raw OCHLV data for stocks
            market_index_ochlv_data: Optional (batch, 1, 5, T_w) - Raw OCHLV data for market index (VNINDEX)

        Returns:
            market_vector: (batch, N) - Market trend vector (Top-K weights, zero elsewhere)
            eta: (batch,) - Risk tolerance factor
            market_scores_full: (batch, N) - Full market scores (softmax on all N assets, no Top-K mask)
            market_logits: (batch, N) - Raw logits before softmax (for stable Loss calc)
            market_context: (batch, D) - Raw market latent state (shared source)
            sigma_logits: (batch, 3) - Market direction logits (up/hold/down)
            topk_indices: (batch, K) - Indices of selected assets
            topk_embeddings: (batch, K, D) or None - Embeddings of selected assets
            topk_scores: (batch, K) - Market weights on selected assets
        """
        batch_size, N, M, T_w = ochlv_data.shape
        assert M == 5, f"Expected 5 features (OCHLV), got {M}"
        assert T_w == self.T_w, f"Expected window size {self.T_w}, got {T_w}"

        device = ochlv_data.device
        dtype = th.float32

        # Process each batch item
        tech_batches = []
        dc_batches = [[] for _ in range(len(self.config.mafia_DC_thresholds))]

        for b in range(batch_size):
            sample_np = ochlv_data[b].detach().cpu().numpy()  # (N, 5, T_w)
            if sample_np.shape[1] != 5 and sample_np.shape[2] == 5:
                sample_np = sample_np.transpose(0, 2, 1)

            P_tech_np = self.feature_processor.process_technical_features(
                sample_np
            )  # (N, T_w, 8)
            tech_batches.append(P_tech_np)

            for idx, threshold in enumerate(self.config.mafia_DC_thresholds):
                P_dc_np = self.feature_processor.process_dc_features(
                    sample_np, threshold
                )  # (N, T_w, 5)
                dc_batches[idx].append(P_dc_np)

        P_tech = th.from_numpy(np.stack(tech_batches, axis=0)).to(
            dtype=dtype, device=device
        )  # (batch, N, T_w, 8)
        P_dc_list = [
            th.from_numpy(np.stack(dc_batches[idx], axis=0)).to(
                dtype=dtype, device=device
            )
            for idx in range(len(self.config.mafia_DC_thresholds))
        ]  # Each: (batch, N, T_w, 5)
        dc_features_list = P_dc_list

        # Technical Agent forward pass
        O_tech_CSA = self.tech_csa(P_tech)  # (batch, N, D)
        O_tech_TA = self.tech_ta(P_tech)  # (batch, T_w, D)
        O_tech, O_tech_ST = self.tech_st_fusion(
            O_tech_CSA, O_tech_TA
        )  # (batch, N, 1), (batch, N, D)

        # DC Agents forward pass
        O_dc_list = []
        O_dc_TA_list = []
        O_dc_ST_list = []
        for i in range(len(self.config.mafia_DC_thresholds)):
            O_dc_CSA = self.dc_csa_list[i](P_dc_list[i])  # (batch, N, D)
            O_dc_TA = self.dc_ta_list[i](
                P_dc_list[i], dc_features=P_dc_list[i]
            )  # (batch, T_w, D)
            O_dc, O_dc_ST = self.dc_st_fusion_list[i](
                O_dc_CSA, O_dc_TA
            )  # (batch, N, 1), (batch, N, D)
            O_dc_list.append(O_dc)
            O_dc_TA_list.append(O_dc_TA)
            O_dc_ST_list.append(O_dc_ST)

        # Stock experts: Tech + 3 DC agents (no market-index)
        stock_expert_outputs = [O_tech] + O_dc_list
        stock_expert_ta_outputs = [O_tech_TA] + O_dc_TA_list

        # Market-index Agent: build market sequence for gating (and optional TA embedding)
        O_mkt_TA = None
        O_mkt_dc_TA_list = []
        macro_tokens = None
        use_market_index = (
            getattr(self.config, "mafia_use_market_index_agent", True)
            and self.mkt_ta is not None
        )
        if use_market_index and market_index_ochlv_data is not None:
            # Process market-index features
            mkt_batches = []
            mkt_dc_batches = [[] for _ in range(len(self.config.mafia_DC_thresholds))]
            for b in range(batch_size):
                mkt_sample_np = (
                    market_index_ochlv_data[b].detach().cpu().numpy()
                )  # (1, 5, T_w)
                if mkt_sample_np.shape[1] != 5 and mkt_sample_np.shape[2] == 5:
                    mkt_sample_np = mkt_sample_np.transpose(0, 2, 1)
                P_mkt_np = self.feature_processor.process_market_index_features(
                    mkt_sample_np,
                )  # (1, T_w, M_mkt)
                mkt_batches.append(P_mkt_np)
                for idx, threshold in enumerate(self.config.mafia_DC_thresholds):
                    P_dc_mkt_np = self.feature_processor.process_dc_features(
                        mkt_sample_np, threshold
                    )  # (1, T_w, 5)
                    mkt_dc_batches[idx].append(P_dc_mkt_np)

            P_mkt = th.from_numpy(np.stack(mkt_batches, axis=0)).to(
                dtype=dtype, device=device
            )  # (batch, 1, T_w, M_mkt)
            # DC features for market index per threshold
            P_dc_mkt_list = [
                th.from_numpy(np.stack(mkt_dc_batches[idx], axis=0)).to(
                    dtype=dtype, device=device
                )
                for idx in range(len(self.config.mafia_DC_thresholds))
            ]  # Each: (batch, 1, T_w, 5)

            # Optional: pre-embedded TA output (kept for backward compatibility)
            O_mkt_TA = (
                self.mkt_ta(P_mkt) if self.mkt_ta is not None else None
            )  # (batch, T_w, D)
            # Market-DC TA outputs (per threshold, kept separate)
            if self.mkt_dc_ta_list is not None and len(P_dc_mkt_list) > 0:
                for i in range(len(self.config.mafia_DC_thresholds)):
                    O_dc = self.mkt_dc_ta_list[i](
                        P_dc_mkt_list[i], dc_features=P_dc_mkt_list[i]
                    )  # (batch, T_w, D)
                    O_mkt_dc_TA_list.append(O_dc)

        # Build macro tokens for gating (concat Market-Index TA + Market-DC TA)
        tokens_list = []
        if O_mkt_TA is not None:
            tokens_list.append(O_mkt_TA)
        if O_mkt_dc_TA_list:
            tokens_list.extend(O_mkt_dc_TA_list)
        if tokens_list:
            macro_tokens = th.cat(tokens_list, dim=-1)

        # Dense MoE Signal Generator
        (
            market_vector,
            eta,
            market_scores_full,
            sigma_logits,
            market_context,
            topk_indices,
            topk_embeddings,
            topk_scores,
            market_logits,
        ) = self.signal_generator(
            stock_expert_outputs,
            stock_expert_ta_outputs,
            x_mkt_seq=macro_tokens,
            expert_st_embeddings=[O_tech_ST] + O_dc_ST_list,
            force_topk_indices=force_topk_indices,
            router_context_buffer=router_context_buffer,  # Temporal augmentation (Spec 3.6)
            explicit_signals=explicit_signals,  # Direction Head explicit signals (Spec 3.5)
            prev_holdings=prev_holdings,  # Memory Injection
        )

        return (
            market_vector,
            eta,
            market_scores_full,
            sigma_logits,
            market_context,
            topk_indices,
            topk_embeddings,
            topk_scores,
            market_logits,
        )
