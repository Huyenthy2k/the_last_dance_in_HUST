import math
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


class MAFIAMultiModalExtractor(BaseFeaturesExtractor):
    """Feature extractor cho obs dạng Dict (global, per_stock, history).

    Cấu hình (qua config trên env, lấy bằng getattr):
    - rl_features_dim: kích thước output cuối (default 256)
    - rl_per_stock_encoder: 'cnn' (default) hoặc 'transformer'
    - rl_per_stock_cnn_channels: kênh ẩn cho CNN (default 128)
    - rl_per_stock_d_model: d_model cho transformer (default 128)
    - rl_per_stock_nhead: số head transformer (default 4)
    - rl_per_stock_num_layers: số layer transformer (default 2)
    - rl_history_hidden_dim: hidden size GRU (default 64), bidirectional
    """

    def __init__(self, observation_space: spaces.Dict, config=None):
        assert isinstance(observation_space, DICT_TYPES), "Expect Dict observation space"
        # Initialize base module early to allow submodules registration
        super().__init__(observation_space, features_dim=1)
        self.config = config
        self.features_dim_cfg = getattr(config, "rl_features_dim", 256)

        global_space: spaces.Box = observation_space["global_context"]
        per_stock_space: spaces.Box = observation_space["per_stock"]
        history_space: spaces.Box = observation_space["history"]

        self.global_dim = int(global_space.shape[0])
        self.num_stocks, self.per_stock_feat_dim = int(per_stock_space.shape[0]), int(per_stock_space.shape[1])
        self.history_len, self.history_feat_dim = int(history_space.shape[0]), int(history_space.shape[1])

        # Global branch
        global_hidden = [128, 128]
        global_layers = []
        last_dim = self.global_dim
        global_layers.append(nn.LayerNorm(self.global_dim))
        for h in global_hidden:
            global_layers.append(nn.Linear(last_dim, h))
            global_layers.append(nn.ReLU())
            last_dim = h
        self.global_net = nn.Sequential(*global_layers)
        self.global_out_dim = last_dim

        # Per-stock branch
        # Default: weighted pooling theo score (cột đầu tiên), không học thêm
        self.per_stock_encoder_type = getattr(config, "rl_per_stock_encoder", "weighted") if config is not None else "weighted"
        if self.per_stock_encoder_type == "cnn":
            ch = int(getattr(config, "rl_per_stock_cnn_channels", 128)) if config is not None else 128
            self.per_stock_proj = nn.Conv1d(self.per_stock_feat_dim, ch, kernel_size=3, padding=1)
            self.per_stock_conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
            self.per_stock_ln = nn.LayerNorm(ch)
            self.per_stock_attn = nn.Linear(ch, 1)
            self.per_stock_out_dim = ch
        elif self.per_stock_encoder_type == "transformer":
            d_model = int(getattr(config, "rl_per_stock_d_model", 128)) if config is not None else 128
            nhead = int(getattr(config, "rl_per_stock_nhead", 4)) if config is not None else 4
            num_layers = int(getattr(config, "rl_per_stock_num_layers", 2)) if config is not None else 2
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.per_stock_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.per_stock_input_proj = nn.Linear(self.per_stock_feat_dim, d_model)
            self.per_stock_attn = nn.Linear(d_model, 1)
            self.per_stock_out_dim = d_model
        else:
            # 'weighted': không học thêm, pooling có trọng số theo score trong cột đầu tiên
            self.per_stock_out_dim = self.per_stock_feat_dim

        # History branch
        hist_hidden = int(getattr(config, "rl_history_hidden_dim", 64)) if config is not None else 64
        self.history_gru = nn.GRU(
            input_size=self.history_feat_dim,
            hidden_size=hist_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.history_out_dim = hist_hidden * 2

        # Projection after concat
        concat_dim = self.global_out_dim + self.per_stock_out_dim + self.history_out_dim
        self.final_ln = nn.LayerNorm(concat_dim)
        if concat_dim != self.features_dim_cfg:
            self.final_proj = nn.Linear(concat_dim, self.features_dim_cfg)
            final_dim = self.features_dim_cfg
        else:
            self.final_proj = nn.Identity()
            final_dim = concat_dim

        # Override features_dim set in base __init__
        self._features_dim = final_dim

    def _encode_global(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_net(x)

    def _encode_per_stock_cnn(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, N, F) -> (batch, F, N)
        x_perm = x.transpose(1, 2)
        h = F.gelu(self.per_stock_proj(x_perm))
        h = F.gelu(self.per_stock_conv2(h))  # (batch, ch, N)
        h = h.transpose(1, 2)  # (batch, N, ch)
        h = self.per_stock_ln(h)
        attn_scores = self.per_stock_attn(h)  # (batch, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * h, dim=1)  # (batch, ch)
        return pooled

    def _encode_per_stock_transformer(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, N, F)
        h = self.per_stock_input_proj(x)  # (batch, N, d_model)
        h = self.per_stock_transformer(h)  # (batch, N, d_model)
        attn_scores = self.per_stock_attn(h)  # (batch, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * h, dim=1)  # (batch, d_model)
        return pooled

    def _encode_per_stock(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_stock_encoder_type == "cnn":
            return self._encode_per_stock_cnn(x)
        if self.per_stock_encoder_type == "transformer":
            return self._encode_per_stock_transformer(x)
        # weighted pooling theo score (cột đầu tiên): (batch, N, F)
        scores = x[:, :, 0:1]  # (batch, N, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights * x, dim=1)  # (batch, F)
        return pooled

    def _encode_history(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, H, K)
        gru_out, h_n = self.history_gru(x)
        # h_n: (num_layers*2, batch, hidden)
        # take last layer (both directions)
        forward = h_n[-2, :, :]
        backward = h_n[-1, :, :]
        combined = torch.cat([forward, backward], dim=-1)  # (batch, 2*hidden)
        return combined

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        global_x = observations["global_context"]
        per_stock_x = observations["per_stock"]
        history_x = observations["history"]

        g = self._encode_global(global_x)
        p = self._encode_per_stock(per_stock_x)
        h = self._encode_history(history_x)

        fused = torch.cat([g, p, h], dim=-1)
        fused = self.final_ln(fused)
        fused = self.final_proj(fused)
        return fused
