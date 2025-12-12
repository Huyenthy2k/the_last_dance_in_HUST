"""
Single-Stage MLP Feature Extractor for Portfolio Allocator (TD3).

Per spec Section 3.4: All raw features concatenated → single MLP → unified state vector.
No multi-branch processing, no GRU for history - only latest step used.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

# Support both gym and gymnasium Dict spaces
try:
    import gymnasium as gym
    DICT_TYPES = (spaces.Dict, gym.spaces.Dict)
except ImportError:  # gymnasium not installed
    DICT_TYPES = (spaces.Dict,)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MAFIAMultiBranchExtractor(BaseFeaturesExtractor):
    """Multi-Branch Feature Extractor per Portfolio Allocator Spec (Section 3).

    Architecture:
    1. Branch 1 (External): Market Context + Direction Logits -> Pass-through
    2. Branch 2 (Per-Stock): Weighted Pooling + Scores -> [h_pool, S_obs]
    3. Branch 3 (Internal): Scalars + History -> LayerNorm -> Pass-through

    Output: h_fusion (Concatenated vector)
    """

    def __init__(self, observation_space: spaces.Dict, config=None):
        assert isinstance(observation_space, DICT_TYPES), "Expect Dict observation space"
        super().__init__(observation_space, features_dim=1)
        self.config = config

        # Extract dimensions
        global_space: spaces.Box = observation_space["global_context"]
        per_stock_space: spaces.Box = observation_space["per_stock"]
        history_space: spaces.Box = observation_space["history"]

        # Branch 1: External Market
        self.global_dim = int(global_space.shape[0])  # D + 3

        # Branch 2: Per-Stock
        self.num_stocks = int(per_stock_space.shape[0])  # K
        self.per_stock_feat_dim = int(per_stock_space.shape[1])  # D + 1
        self.stock_emb_dim = self.per_stock_feat_dim - 1  # D

        # Branch 3: Internal State
        self.history_len = int(history_space.shape[0])  # T
        self.internal_dim = int(history_space.shape[1])  # 2K + 8

        # Calculate Output Dim (h_fusion)
        # 1. Global: D + 3
        # 2. Per-Stock: D (pooled) + K (scores)
        # 3. Internal: 2K + 8 (normalized)
        self.fusion_dim = (
            self.global_dim
            + self.stock_emb_dim
            + self.num_stocks
            + self.internal_dim
        )
        self._features_dim = self.fusion_dim

        # Branch 3 Normalization
        self.internal_norm = nn.LayerNorm(self.internal_dim)

    def _weighted_pool_per_stock(self, per_stock: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weighted pooling: h_pool = sum(softmax(S) * H)."""
        # per_stock: (batch, K, D+1). Score is at index 0.
        scores = per_stock[:, :, 0]  # (batch, K)
        embeddings = per_stock[:, :, 1:]  # (batch, K, D)

        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, K, 1)
        pooled = torch.sum(weights * embeddings, dim=1)  # (batch, D)
        return pooled, scores

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Branch 1: External Market (Pass-through)
        global_ctx = observations["global_context"]  # (batch, D+3)

        # 2. Branch 2: Per-Stock (Weighted Pooling)
        per_stock = observations["per_stock"]
        pooled_emb, scores = self._weighted_pool_per_stock(per_stock)

        # 3. Branch 3: Internal State (Select Latest + LayerNorm)
        history = observations["history"]
        latest_internal = history[:, -1, :]  # (batch, 2K+8)
        norm_internal = self.internal_norm(latest_internal)

        # Fusion
        h_fusion = torch.cat([
            global_ctx,
            pooled_emb,
            scores,
            norm_internal
        ], dim=-1)

        return h_fusion

# Alias for backward compatibility
MAFIAMultiModalExtractor = MAFIAMultiBranchExtractor

