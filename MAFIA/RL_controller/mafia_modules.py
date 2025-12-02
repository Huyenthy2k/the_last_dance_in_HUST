#！/usr/bin/python
# -*- coding: utf-8 -*-#

"""
MAFIA Model Modules

This module contains the core neural network modules for MAFIA:
- CSAModule: Cross-Sectional Analysis module
- TAModule: Temporal Analysis module
- STFusionModule: Spatial-Temporal Fusion module
- SignalGenerator: Portfolio Generator (market_vector + boundary_risk)
- MAFIAModel: Complete model integrating all agents

Author: MASA
See: Đặc Tả Kỹ Thuật (Technical Specification) - MAFIA.md
"""
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = None, dropout: float = 0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
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
    
    def __init__(self, config, agent_type: str = 'tech'):
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
        
        if agent_type == 'tech':
            self.M = config.mafia_M_tech  # 8
            self.token_dim = self.T_w * self.M  # 30 * 8 = 240
        else:  # dc
            self.M = config.mafia_M_dc  # 5
            self.token_dim = self.T_w * self.M  # 30 * 5 = 150
        
        # Token Generation: Reshape (N, T_w, M) -> (N, T_w*M)
        # This is done in forward pass
        
        # Embedding MLP
        self.embedding = nn.Sequential(
            nn.Linear(self.token_dim, self.D_h),
            nn.GELU(),
            nn.Linear(self.D_h, self.D)
        )
        
        # Transformer Encoder
        encoder_layers = []
        for _ in range(config.mafia_encoder_layers):
            encoder_layers.append(
                TransformerEncoderLayer(
                    d_model=self.D,
                    nhead=config.mafia_encoder_heads,
                    dim_feedforward=self.D_h * 2,
                    dropout=0.1
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
    
    def __init__(self, config, agent_type: str = 'tech', shared_mlp: Optional[nn.Module] = None, N: Optional[int] = None):
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
        self.T_w = config.mafia_T_w
        self.D = config.mafia_D
        self.D_h = config.mafia_D_h
        
        # Use actual N if provided, otherwise use config.topK
        actual_N = N if N is not None else config.topK
        
        if agent_type == 'tech':
            self.M = config.mafia_M_tech  # 8
            self.token_dim = actual_N * self.M  # N * M
        elif agent_type == 'mkt':
            self.M = config.mafia_M_mkt  # 19
            self.token_dim = actual_N * self.M  # N * M (N=1 for market-index)
        else:  # dc
            self.M = config.mafia_M_dc  # 5
            self.token_dim = actual_N * self.M  # N * M
        
        # Embedding MLP
        if shared_mlp is not None and agent_type == 'dc':
            self.embedding = shared_mlp
        else:
            self.embedding = nn.Sequential(
                nn.Linear(self.token_dim, self.D_h),
                nn.GELU(),
                nn.Linear(self.D_h, self.D)
            )
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(self.D, self.T_w)
        
        # Event Detector (only for DC agents)
        if agent_type == 'dc':
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
                    dropout=0.1
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)
    
    def forward(self, P_i: th.Tensor, dc_features: Optional[th.Tensor] = None) -> th.Tensor:
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
        tokens = P_permuted.reshape(batch_size, T_w, actual_token_dim)  # (batch, T_w, N*M)
        
        # If token_dim changed, we need to handle embedding differently
        # For now, use a linear projection if needed
        if actual_token_dim != self.token_dim:
            # Create a temporary embedding layer if dimensions don't match
            if not hasattr(self, '_temp_embedding') or self._temp_embedding[0].in_features != actual_token_dim:
                self._temp_embedding = nn.Sequential(
                    nn.Linear(actual_token_dim, self.D_h),
                    nn.GELU(),
                    nn.Linear(self.D_h, self.D)
                ).to(tokens.device)
            embedded = self._temp_embedding(tokens)  # (batch, T_w, D)
        else:
            embedded = self.embedding(tokens)  # (batch, T_w, D)
        
        # Signal Merging: Add positional encoding
        PE_time = self.pos_encoding(embedded)  # (batch, T_w, D)
        input_ta = embedded + PE_time
        
        # Add high-order DC signals (DC agents only)
        if self.agent_type == 'dc' and self.event_detector is not None and dc_features is not None:
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
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(t_prime * div_term)
        pe[:, 1::2] = th.cos(t_prime * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
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
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
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
        duration_padded = th.cat([duration_avg[:, 0:1], duration_avg], dim=1)  # (batch, T_w+1)
        
        delta_state = state_avg - state_padded[:, :-1]  # (batch, T_w)
        delta_duration = duration_avg - duration_padded[:, :-1]  # (batch, T_w)
        
        # Combine event signals
        event_signals = th.stack([delta_state, delta_duration], dim=-1)  # (batch, T_w, 2)
        
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
        self.num_heads = getattr(config, 'mafia_gating_num_heads', 4)
        self.dropout = getattr(config, 'mafia_gating_dropout', 0.1)
        
        # Learnable CLS token - learns what information to extract from temporal sequence
        self.cls_token = nn.Parameter(th.randn(1, 1, self.D))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Multi-head cross-attention: CLS token queries the temporal sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.D,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
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
            query=cls_tokens,      # (batch, 1, D) - "What market info to extract?"
            key=O_mkt_TA,          # (batch, T_w, D) - "Where to look?"
            value=O_mkt_TA,        # (batch, T_w, D) - "What to extract?"
            need_weights=True,
            average_attn_weights=False  # Return per-head weights
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
        self.dropout = getattr(config, 'mafia_gating_dropout', 0.1)
        self.kernel_sizes = getattr(config, 'mafia_gating_conv_kernels', [3, 5, 7])
        
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
            nn.Linear(self.D * len(self.kernel_sizes) * 2, self.D * 2),  # *2 for max+avg pooling
            nn.LayerNorm(self.D * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.D * 2, self.D),
            nn.LayerNorm(self.D)
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
        avg_pool = th.mean(multi_scale, dim=2)     # (batch, D*num_kernels)
        
        # Concatenate pooled features
        pooled = th.cat([max_pool, avg_pool], dim=1)  # (batch, D*num_kernels*2)
        
        # Project to D dimension
        market_context = self.proj(pooled)  # (batch, D)
        
        return market_context, None


class BidirectionalLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for Dense MoE Gating.
    Uses BiLSTM with optional temporal attention for sequential modeling.
    
    Input: O_mkt_TA (batch, T_w, D) - Temporal market index embedding
    Output: market_context (batch, D) - Aggregated market condition representation
    """
    
    def __init__(self, config):
        super().__init__()
        self.D = config.mafia_D
        self.num_layers = getattr(config, 'mafia_gating_lstm_layers', 2)
        self.dropout = getattr(config, 'mafia_gating_dropout', 0.1)
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=self.D,
            hidden_size=self.D // 2,  # Because bidirectional doubles the output
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
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
            nn.LayerNorm(self.D)
        )
        
    def forward(self, O_mkt_TA: th.Tensor) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            O_mkt_TA: (batch, T_w, D) - Temporal market index embedding
        
        Returns:
            market_context: (batch, D) - Aggregated market condition
            attn_weights: (batch, T_w, 1) - Temporal attention weights (if use_attention=True)
        """
        batch_size = O_mkt_TA.size(0)
        
        # BiLSTM processing
        lstm_out, (h_n, c_n) = self.bilstm(O_mkt_TA)
        # lstm_out: (batch, T_w, D) - hidden states for all timesteps
        # h_n: (num_layers*2, batch, D/2) - final hidden states
        
        attn_weights = None
        if self.use_attention:
            # Weighted aggregation with learned attention
            attn_scores = self.attn_proj(lstm_out)  # (batch, T_w, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, T_w, 1)
            
            # Weighted sum
            market_context = th.sum(
                attn_weights * lstm_out, 
                dim=1
            )  # (batch, D)
        else:
            # Simple: take last timestep
            market_context = lstm_out[:, -1, :]  # (batch, D)
        
        # Refine with MLP
        market_context = self.refine(market_context)
        
        return market_context, attn_weights


class DenseMoEGatingRouter(nn.Module):
    """
    Dense MoE Gating Router for computing expert weights.
    Uses temporal encoder to extract market condition, then MLP to generate gate weights.
    
    Input: O_mkt_TA (batch, T_w, D) - Temporal market index embedding
    Output: gate_weights (batch, num_experts=4) - Weights for each expert
    """
    
    def __init__(self, config, num_experts=4):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.D = config.mafia_D
        self.dropout = getattr(config, 'mafia_gating_dropout', 0.1)
        
        # Select temporal encoder based on config
        encoder_type = getattr(config, 'mafia_gating_encoder_type', 'attention_based_aggregation')
        
        if encoder_type == 'attention_based_aggregation':
            self.temporal_encoder = AttentionBasedTemporalEncoder(config)
        elif encoder_type == 'temporal_convolution':
            self.temporal_encoder = TemporalConvolutionEncoder(config)
        elif encoder_type == 'bidirectional_lstm':
            self.temporal_encoder = BidirectionalLSTMEncoder(config)
        else:
            raise ValueError(
                f"Unknown gating encoder type: {encoder_type}. "
                f"Must be one of: 'attention_based_aggregation', 'temporal_convolution', 'bidirectional_lstm'"
            )
        
        # Gate weight generator MLP
        self.gate_network = nn.Sequential(
            nn.Linear(self.D, self.D * 2),
            nn.LayerNorm(self.D * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.D * 2, num_experts),
            nn.Softmax(dim=-1)  # Normalize weights to sum to 1
        )
        
    def forward(self, O_mkt_TA: Optional[th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args:
            O_mkt_TA: Optional (batch, T_w, D) - Temporal market index embedding
                      If None, returns uniform weights
        
        Returns:
            gate_weights: (batch, num_experts) - Weights for each expert (sum to 1)
            market_context: (batch, D) - Market condition embedding
        """
        if O_mkt_TA is None:
            # Fallback: uniform weights when market index is not available
            batch_size = 1  # Will be broadcasted if needed
            device = next(self.parameters()).device
            gate_weights = th.ones(batch_size, self.num_experts, device=device) / self.num_experts
            market_context = th.zeros(batch_size, self.D, device=device)
            return gate_weights, market_context
        
        # Extract market condition from temporal sequence
        market_context, _ = self.temporal_encoder(O_mkt_TA)  # (batch, D)
        
        # Generate gate weights from market context
        gate_weights = self.gate_network(market_context)  # (batch, num_experts)
        
        return gate_weights, market_context


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
        self.tau_gumbel = getattr(config, 'mafia_gumbel_temperature', 1.0)
        self.num_direction_classes = 3
        
        # Dense MoE Gating Router (for 4 stock experts: Tech + 3 DC)
        self.gating_router = DenseMoEGatingRouter(config, num_experts=4)
        
        # Gumbel-TopK temperature (trainable or fixed)
        self.topk_temperature = nn.Parameter(
            th.tensor(self.tau_gumbel), 
            requires_grad=False
        )
        
        # Boundary risk network (from expert TA outputs)
        self.risk_network = nn.Sequential(
            nn.Linear(self.T_w * self.D * 4, self.D),  # 4 experts' TA outputs
            nn.LayerNorm(self.D),
            nn.GELU(),
            nn.Linear(self.D, 1),
            nn.Softplus()  # Ensure positive risk value
        )

        # Direction classification head (market direction: up/hold/down)
        self.direction_head = nn.Sequential(
            nn.Linear(self.T_w * self.D * 4, self.D),
            nn.LayerNorm(self.D),
            nn.GELU(),
            nn.Linear(self.D, self.num_direction_classes)
        )
    
    def forward(
        self, 
        expert_outputs: list,      # [O_tech, O_dc1, O_dc2, O_dc3]
        expert_ta_outputs: list,   # [O_tech_TA, O_dc1_TA, O_dc2_TA, O_dc3_TA]
        O_mkt_TA: Optional[th.Tensor],  # Market-index TA output for gating
        expert_st_embeddings: Optional[list] = None  # Optional per-stock embeddings per expert [(batch, N, D)]
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor], th.Tensor]:
        """
        Dense MoE forward pass.
        
        Args:
            expert_outputs: List of 4 tensors, each (batch, N, 1) - Stock expert logits
            expert_ta_outputs: List of 4 tensors, each (batch, T_w, D) - Expert TA outputs
            O_mkt_TA: Optional (batch, T_w, D) - Market-index TA output for gating
        
        Returns:
            market_vector: (batch, N) - Portfolio weights (Top-K assets)
            boundary_risk: (batch,) - Risk boundary value
            topk_indices: (batch, K) - Indices of selected assets
            market_scores_full: (batch, N) - Full softmax scores for all assets
            gate_weights: (batch, 4) - Expert weights from gating router
            market_context: (batch, D) - Market condition embedding from gating encoder
            fused_stock_embedding: (batch, N, D) or None - Weighted fusion of expert ST embeddings per stock
            sigma_logits: (batch, 3) - Logits for market direction classification (up/hold/down)
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
        gate_weights, market_context = self.gating_router(O_mkt_TA)  # (batch, 4), (batch, D)
        
        # Expand gate_weights for batch if needed (when O_mkt_TA was None)
        if gate_weights.size(0) == 1 and batch_size > 1:
            gate_weights = gate_weights.expand(batch_size, -1)
        
        # Step 2: Stack expert outputs and apply gating
        expert_logits_stacked = th.stack(
            [o.squeeze(-1) for o in expert_outputs], 
            dim=1
        )  # (batch, 4, N)
        
        # Weighted combination using gate weights
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch, 4, 1)
        market_logits = th.sum(
            gate_weights_expanded * expert_logits_stacked, 
            dim=1
        )  # (batch, N)
        
        # Optional: fuse per-stock ST embeddings from experts using gate weights
        fused_stock_embedding = None
        if expert_st_embeddings is not None:
            if len(expert_st_embeddings) != num_experts:
                raise ValueError(
                    f"Expected {num_experts} expert ST embeddings, got {len(expert_st_embeddings)}"
                )
            stacked_emb = th.stack(expert_st_embeddings, dim=1)  # (batch, 4, N, D)
            gate_weights_exp = gate_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 4, 1, 1)
            fused_stock_embedding = th.sum(gate_weights_exp * stacked_emb, dim=1)  # (batch, N, D)

        # Step 3: Gumbel-TopK selection
        temperature = max(self.tau_gumbel, 1e-6)
        market_scores_full = F.softmax(market_logits / temperature, dim=-1)  # (batch, N)
        
        # Determine if using hard or soft TopK
        training_mode = self.training
        hard_inference = getattr(self.config, 'mafia_hard_topk_inference', True)
        
        if training_mode or not hard_inference:
            # Soft Gumbel-TopK with noise
            uniform = th.rand_like(market_logits).clamp_(min=1e-8, max=1 - 1e-8)
            gumbel_noise = -th.log(-th.log(uniform))
            logits_for_topk = (market_logits + gumbel_noise) / temperature
        else:
            # Hard TopK (deterministic)
            logits_for_topk = market_logits / temperature
        
        # Select top-K assets
        top_k = min(self.K, N)
        topk_vals, topk_indices = th.topk(logits_for_topk, k=top_k, dim=-1)
        
        # Create mask for selected indices
        mask = th.zeros_like(market_logits, dtype=th.bool)
        mask.scatter_(1, topk_indices, True)
        
        # Apply mask and softmax to get portfolio weights
        masked_logits = logits_for_topk.masked_fill(~mask, float('-inf'))
        market_vector = F.softmax(masked_logits, dim=-1)
        
        # Enforce hard support at inference
        if not training_mode and hard_inference:
            market_vector = market_vector.masked_fill(~mask, 0.0)
        
        # Step 4: Compute boundary_risk from expert TA outputs
        # Stack and flatten TA outputs from all 4 experts
        expert_ta_stacked = th.stack(expert_ta_outputs, dim=1)  # (batch, 4, T_w, D)
        expert_ta_flat = expert_ta_stacked.reshape(batch_size, -1)  # (batch, 4*T_w*D)
        
        # Compute risk
        boundary_risk = self.risk_network(expert_ta_flat)  # (batch, 1)
        boundary_risk = boundary_risk.squeeze(-1)  # (batch,)

        # Market direction logits
        sigma_logits = self.direction_head(expert_ta_flat)  # (batch, 3)
        
        return (
            market_vector,
            boundary_risk,
            topk_indices,
            market_scores_full,
            gate_weights,
            market_context,
            fused_stock_embedding,
            sigma_logits,
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
        self.tech_csa = CSAModule(config, agent_type='tech')
        self.tech_ta = TAModule(config, agent_type='tech', N=self.N)
        self.tech_st_fusion = STFusionModule(self.D)
        
        # DC Agents (i=1,2,3) - Each agent has its own CSA/TA weights
        self.dc_csa_list = nn.ModuleList([
            CSAModule(config, agent_type='dc') for _ in range(len(self.config.mafia_DC_thresholds))
        ])
        self.dc_ta_list = nn.ModuleList([
            TAModule(config, agent_type='dc', shared_mlp=None, N=self.N) for _ in range(len(self.config.mafia_DC_thresholds))
        ])
        self.dc_st_fusion_list = nn.ModuleList([
            STFusionModule(self.D) for _ in range(3)
        ])
        
        # Market-index Agent (VNINDEX) - Only TA module for gating
        use_market_index = getattr(config, 'mafia_use_market_index_agent', True)
        if use_market_index:
            self.mkt_ta = TAModule(config, agent_type='mkt', N=1)  # Only TA, N=1 for single asset, M=19
        else:
            self.mkt_ta = None
        
        # Dense MoE Signal Generator
        self.signal_generator = DenseMoESignalGenerator(config)
    
    def forward(
        self,
        ochlv_data: th.Tensor,
        market_index_ochlv_data: Optional[th.Tensor] = None,
        market_regime_feats: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor], th.Tensor]:
        """
        Forward pass through MAFIA model with Dense MoE.
        
        Args:
            ochlv_data: (batch, N, 5, T_w) - Raw OCHLV data for stocks
            market_index_ochlv_data: Optional (batch, 1, 5, T_w) - Raw OCHLV data for market index (VNINDEX)
        
        Returns:
            market_vector: (batch, N) - Market trend vector (Top-K weights, zero elsewhere)
            boundary_risk: (batch,) - Risk boundary value
            topk_indices: (batch, K) - Indices of selected assets
            market_scores_full: (batch, N) - Full market scores (softmax on all N assets, no Top-K mask)
            gate_weights: (batch, 4) - Expert weights from Dense MoE gating
            market_context: (batch, D) - Market condition embedding from the gating encoder
            fused_stock_embedding: (batch, N, D) or None - Fused per-stock embedding from expert ST outputs
            sigma_logits: (batch, 3) - Market direction logits (up/hold/down)
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
            
            P_tech_np = self.feature_processor.process_technical_features(sample_np)  # (N, T_w, 8)
            tech_batches.append(P_tech_np)
            
            for idx, threshold in enumerate(self.config.mafia_DC_thresholds):
                P_dc_np = self.feature_processor.process_dc_features(sample_np, threshold)  # (N, T_w, 5)
                dc_batches[idx].append(P_dc_np)
        
        P_tech = th.from_numpy(np.stack(tech_batches, axis=0)).to(dtype=dtype, device=device)  # (batch, N, T_w, 8)
        P_dc_list = [th.from_numpy(np.stack(dc_batches[idx], axis=0)).to(dtype=dtype, device=device)
                     for idx in range(len(self.config.mafia_DC_thresholds))]  # Each: (batch, N, T_w, 5)
        dc_features_list = P_dc_list
        
        # Technical Agent forward pass
        O_tech_CSA = self.tech_csa(P_tech)  # (batch, N, D)
        O_tech_TA = self.tech_ta(P_tech)  # (batch, T_w, D)
        O_tech, O_tech_ST = self.tech_st_fusion(O_tech_CSA, O_tech_TA)  # (batch, N, 1), (batch, N, D)
        
        # DC Agents forward pass
        O_dc_list = []
        O_dc_TA_list = []
        O_dc_ST_list = []
        for i in range(len(self.config.mafia_DC_thresholds)):
            O_dc_CSA = self.dc_csa_list[i](P_dc_list[i])  # (batch, N, D)
            O_dc_TA = self.dc_ta_list[i](P_dc_list[i], dc_features=P_dc_list[i])  # (batch, T_w, D)
            O_dc, O_dc_ST = self.dc_st_fusion_list[i](O_dc_CSA, O_dc_TA)  # (batch, N, 1), (batch, N, D)
            O_dc_list.append(O_dc)
            O_dc_TA_list.append(O_dc_TA)
            O_dc_ST_list.append(O_dc_ST)
        
        # Stock experts: Tech + 3 DC agents (no market-index)
        stock_expert_outputs = [O_tech] + O_dc_list
        stock_expert_ta_outputs = [O_tech_TA] + O_dc_TA_list
        
        # Market-index Agent: Only compute O_mkt_TA for gating
        O_mkt_TA = None
        use_market_index = getattr(self.config, 'mafia_use_market_index_agent', True) and self.mkt_ta is not None
        if use_market_index and market_index_ochlv_data is not None:
            # Process market-index features
            mkt_batches = []
            regime_batches = []
            for b in range(batch_size):
                mkt_sample_np = market_index_ochlv_data[b].detach().cpu().numpy()  # (1, 5, T_w)
                if mkt_sample_np.shape[1] != 5 and mkt_sample_np.shape[2] == 5:
                    mkt_sample_np = mkt_sample_np.transpose(0, 2, 1)
                regime_np = None
                if market_regime_feats is not None:
                    regime_np = market_regime_feats[b].detach().cpu().numpy()
                P_mkt_np = self.feature_processor.process_market_index_features(
                    mkt_sample_np,
                    regime_feats=regime_np,
                )  # (1, T_w, M_mkt)
                mkt_batches.append(P_mkt_np)
            
            P_mkt = th.from_numpy(np.stack(mkt_batches, axis=0)).to(dtype=dtype, device=device)  # (batch, 1, T_w, M_mkt)
            
            # Market-index Agent: ONLY TA output for gating
            O_mkt_TA = self.mkt_ta(P_mkt)  # (batch, T_w, D)
        
        # Dense MoE Signal Generator
        (
            market_vector,
            boundary_risk,
            topk_indices,
            market_scores_full,
            gate_weights,
            market_context,
            fused_stock_embedding,
            sigma_logits,
        ) = self.signal_generator(
            stock_expert_outputs,
            stock_expert_ta_outputs,
            O_mkt_TA,
            expert_st_embeddings=[O_tech_ST] + O_dc_ST_list,
        )

        return (
            market_vector,
            boundary_risk,
            topk_indices,
            market_scores_full,
            gate_weights,
            market_context,
            fused_stock_embedding,
            sigma_logits,
        )
