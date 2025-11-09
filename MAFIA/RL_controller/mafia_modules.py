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
    
    def forward(self, O_CSA: th.Tensor, O_TA: th.Tensor) -> th.Tensor:
        """
        Forward pass through ST-Fusion.
        
        Args:
            O_CSA: (batch, N, D) - CSA output
            O_TA: (batch, T_w, D) - TA output
        
        Returns:
            O_i: (batch, N, 1) - Agent logits
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
        
        return output


class SignalGenerator(nn.Module):
    """
    Signal Generator: Portfolio Generator.
    Aggregates outputs from all agents to generate market_vector and boundary_risk.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.T_w = config.mafia_T_w
        self.D = config.mafia_D
        
        # Risk Head: Predicts boundary_risk from aggregated TA embeddings
        risk_input_dim = self.T_w * self.D
        self.risk_head = nn.Sequential(
            nn.Linear(risk_input_dim, self.D),
            nn.ReLU(),
            nn.Linear(self.D, 1),
            nn.Softplus()  # Ensure positive output
        )
    
    def forward(self, agent_outputs: list, agent_ta_outputs: list) -> Tuple[th.Tensor, th.Tensor]:
        """
        Generate market_vector and boundary_risk from all agents.
        
        Args:
            agent_outputs: List of (batch, N, 1) tensors - one per agent
            agent_ta_outputs: List of (batch, T_w, D) tensors - TA outputs from all agents
        
        Returns:
            market_vector: (batch, N) - Market trend vector
            boundary_risk: (batch, 1) - Risk boundary value
        """
        batch_size = agent_outputs[0].size(0)
        N = agent_outputs[0].size(1)
        
        # Generate market_vector: Sum all agent logits
        # agent_outputs: list of (batch, N, 1)
        market_vector = th.zeros(batch_size, N, 1, device=agent_outputs[0].device)
        for agent_out in agent_outputs:
            market_vector = market_vector + agent_out
        
        market_vector = market_vector.squeeze(-1)  # (batch, N)
        
        # Generate boundary_risk: Aggregate TA embeddings
        # Average all TA outputs
        O_agg_TA = th.stack(agent_ta_outputs, dim=0)  # (num_agents, batch, T_w, D)
        O_agg_TA = th.mean(O_agg_TA, dim=0)  # (batch, T_w, D)
        
        # Flatten
        O_flat_TA = O_agg_TA.reshape(batch_size, -1)  # (batch, T_w*D)
        
        # Risk Head
        boundary_risk = self.risk_head(O_flat_TA)  # (batch, 1)
        boundary_risk = boundary_risk.squeeze(-1)  # (batch,)
        
        return market_vector, boundary_risk


class MAFIAModel(nn.Module):
    """
    Complete MAFIA Model.
    Consists of 1 Technical Agent + 3 DC Agents.
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
        
        # DC Agents (i=1,2,3) - Share embedding MLP for TA, use actual N
        dc_ta_shared_mlp = nn.Sequential(
            nn.Linear(self.N * config.mafia_M_dc, config.mafia_D_h),
            nn.GELU(),
            nn.Linear(config.mafia_D_h, self.D)
        )
        
        self.dc_csa_list = nn.ModuleList([
            CSAModule(config, agent_type='dc') for _ in range(3)
        ])
        self.dc_ta_list = nn.ModuleList([
            TAModule(config, agent_type='dc', shared_mlp=dc_ta_shared_mlp, N=self.N) for _ in range(3)
        ])
        self.dc_st_fusion_list = nn.ModuleList([
            STFusionModule(self.D) for _ in range(3)
        ])
        
        # Signal Generator
        self.signal_generator = SignalGenerator(config)
    
    def forward(self, ochlv_data: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass through MAFIA model.
        
        Args:
            ochlv_data: (batch, N, 5, T_w) - Raw OCHLV data
        
        Returns:
            market_vector: (batch, N) - Market trend vector
            boundary_risk: (batch,) - Risk boundary value
        """
        batch_size, N, M, T_w = ochlv_data.shape
        assert M == 5, f"Expected 5 features (OCHLV), got {M}"
        assert T_w == self.T_w, f"Expected window size {self.T_w}, got {T_w}"
        
        # Process each batch item separately (for now - can be optimized later)
        # Convert to numpy for feature processing
        ochlv_np = ochlv_data[0].detach().cpu().numpy()  # (N, 5, T_w) - take first batch item
        # Ensure correct format: (N, 5, T_w)
        if ochlv_np.shape[1] != 5:
            # If shape is (N, T_w, 5), transpose
            ochlv_np = ochlv_np.transpose(0, 2, 1)  # (N, 5, T_w)
        
        # Process Technical Agent features
        P_tech_np = self.feature_processor.process_technical_features(ochlv_np)  # (N, T_w, 8)
        P_tech = th.from_numpy(P_tech_np).float().to(ochlv_data.device)
        P_tech = P_tech.unsqueeze(0)  # (1, N, T_w, 8)
        
        # Process DC Agent features
        P_dc_list = []
        dc_features_list = []
        for i, threshold in enumerate(self.config.mafia_DC_thresholds):
            P_dc_np = self.feature_processor.process_dc_features(ochlv_np, threshold)  # (N, T_w, 5)
            P_dc = th.from_numpy(P_dc_np).float().to(ochlv_data.device)
            P_dc = P_dc.unsqueeze(0)  # (1, N, T_w, 5)
            P_dc_list.append(P_dc)
            dc_features_list.append(P_dc)
        
        # Technical Agent forward pass
        O_tech_CSA = self.tech_csa(P_tech)  # (batch, N, D)
        O_tech_TA = self.tech_ta(P_tech)  # (batch, T_w, D)
        O_tech = self.tech_st_fusion(O_tech_CSA, O_tech_TA)  # (batch, N, 1)
        
        # DC Agents forward pass
        O_dc_list = []
        O_dc_TA_list = []
        for i in range(3):
            O_dc_CSA = self.dc_csa_list[i](P_dc_list[i])  # (batch, N, D)
            O_dc_TA = self.dc_ta_list[i](P_dc_list[i], dc_features=P_dc_list[i])  # (batch, T_w, D)
            O_dc = self.dc_st_fusion_list[i](O_dc_CSA, O_dc_TA)  # (batch, N, 1)
            O_dc_list.append(O_dc)
            O_dc_TA_list.append(O_dc_TA)
        
        # Collect all agent outputs
        all_agent_outputs = [O_tech] + O_dc_list
        all_agent_ta_outputs = [O_tech_TA] + O_dc_TA_list
        
        # Signal Generator
        market_vector, boundary_risk = self.signal_generator(all_agent_outputs, all_agent_ta_outputs)
        
        return market_vector, boundary_risk

