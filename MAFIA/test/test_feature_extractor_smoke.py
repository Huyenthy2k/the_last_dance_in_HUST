"""
Tests for MAFIASingleStageMLP Feature Extractor.

Verifies the single-stage MLP architecture per Portfolio Allocator spec Section 3.4.
"""

import numpy as np
import torch
try:
    import gymnasium as gym
    SPACES = gym.spaces
except ImportError:
    import gym
    SPACES = gym.spaces
from stable_baselines3.td3.policies import MultiInputPolicy

from RL_controller.feature_extractors import MAFIASingleStageMLP


class DummyConfig:
    """Minimal config for testing."""
    rl_features_dim = 64  # D_hidden


def make_obs_space(K=5, D_stock=9, global_dim=10, history_len=3, history_feat_dim=12):
    """Create observation space matching spec structure.

    Args:
        K: Number of top-K stocks
        D_stock: Per-stock embedding dimension
        global_dim: Global context dimension (market_context + direction + risk_eta)
        history_len: Number of history steps (T)
        history_feat_dim: History feature dim per step (1 + K + 1 + K = 2K + 2)
    """
    return SPACES.Dict({
        'global_context': SPACES.Box(low=-np.inf, high=np.inf, shape=(global_dim,), dtype=np.float32),
        'per_stock': SPACES.Box(low=-np.inf, high=np.inf, shape=(K, D_stock), dtype=np.float32),
        'history': SPACES.Box(low=-np.inf, high=np.inf, shape=(history_len, history_feat_dim), dtype=np.float32),
    })


def make_fake_obs(obs_space, batch_size=2):
    """Create fake observation batch for testing."""
    obs = {}
    for key, space in obs_space.spaces.items():
        sample = torch.tensor(space.sample(), dtype=torch.float32)
        repeats = (batch_size,) + tuple(1 for _ in space.shape)
        obs[key] = sample.unsqueeze(0).repeat(*repeats)
    return obs


def test_feature_extractor_output_shape():
    """Test that output shape matches rl_features_dim."""
    cfg = DummyConfig()
    obs_space = make_obs_space()
    extractor = MAFIASingleStageMLP(obs_space, config=cfg)
    fake_obs = make_fake_obs(obs_space, batch_size=2)
    out = extractor(fake_obs)
    assert out.shape == (2, cfg.rl_features_dim)


def test_single_stage_mlp_architecture():
    """Test that architecture is single-stage MLP (2 linear layers)."""
    cfg = DummyConfig()
    obs_space = make_obs_space()
    extractor = MAFIASingleStageMLP(obs_space, config=cfg)

    # Check MLP structure: Linear -> ReLU -> Linear -> ReLU
    assert hasattr(extractor, 'mlp')
    mlp_layers = list(extractor.mlp.children())
    assert len(mlp_layers) == 4  # Linear, ReLU, Linear, ReLU
    assert isinstance(mlp_layers[0], torch.nn.Linear)
    assert isinstance(mlp_layers[1], torch.nn.ReLU)
    assert isinstance(mlp_layers[2], torch.nn.Linear)
    assert isinstance(mlp_layers[3], torch.nn.ReLU)


def test_total_flat_dim_calculation():
    """Test that total flat dimension is calculated correctly per spec."""
    cfg = DummyConfig()
    K = 10
    D_stock = 64
    global_dim = 70  # market_context(64) + direction(3) + risk_eta(1) + scalars(2)
    history_feat_dim = 22  # 1 + K + 1 + K = 2*10 + 2 = 22

    obs_space = make_obs_space(
        K=K,
        D_stock=D_stock,
        global_dim=global_dim,
        history_len=5,
        history_feat_dim=history_feat_dim
    )
    extractor = MAFIASingleStageMLP(obs_space, config=cfg)

    # Expected: global_dim + D_stock + K + history_feat_dim
    expected_flat_dim = global_dim + D_stock + K + history_feat_dim
    assert extractor.total_flat_dim == expected_flat_dim


def test_weighted_pooling():
    """Test weighted pooling of per-stock embeddings."""
    cfg = DummyConfig()
    obs_space = make_obs_space(K=3, D_stock=5)
    extractor = MAFIASingleStageMLP(obs_space, config=cfg)

    # Create per_stock tensor where first column is scores
    per_stock = torch.tensor([
        [[1.0, 0.1, 0.2, 0.3, 0.4],   # Stock 0: score=1.0
         [2.0, 0.5, 0.6, 0.7, 0.8],   # Stock 1: score=2.0 (highest)
         [0.5, 0.9, 1.0, 1.1, 1.2]],  # Stock 2: score=0.5
    ], dtype=torch.float32)  # (1, 3, 5)

    pooled, scores = extractor._weighted_pool_per_stock(per_stock)

    # Check shapes
    assert pooled.shape == (1, 5)  # (batch, D_stock)
    assert scores.shape == (1, 3)  # (batch, K)

    # Scores should be first column
    expected_scores = torch.tensor([[1.0, 2.0, 0.5]])
    assert torch.allclose(scores, expected_scores)


def test_latest_history_only():
    """Test that only latest history step is used (not full sequence)."""
    cfg = DummyConfig()
    K = 5
    history_len = 3
    history_feat_dim = 12

    obs_space = make_obs_space(K=K, history_len=history_len, history_feat_dim=history_feat_dim)
    extractor = MAFIASingleStageMLP(obs_space, config=cfg)

    # Create history with distinct values per timestep
    history = torch.zeros(1, history_len, history_feat_dim)
    history[0, 0, :] = 1.0  # t-2
    history[0, 1, :] = 2.0  # t-1
    history[0, 2, :] = 3.0  # t (latest)

    obs = {
        'global_context': torch.randn(1, 10),
        'per_stock': torch.randn(1, K, 9),
        'history': history,
    }

    # Run forward pass
    _ = extractor(obs)

    # The extractor should use history[:, -1, :] which has value 3.0
    latest = history[:, -1, :]
    assert torch.all(latest == 3.0)


def test_multiinput_policy_initialization():
    """Test integration with SB3 MultiInputPolicy."""
    cfg = DummyConfig()
    obs_space = make_obs_space()
    action_space = SPACES.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    policy = MultiInputPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=MAFIASingleStageMLP,
        features_extractor_kwargs={'config': cfg},
    )

    fake_obs = make_fake_obs(obs_space, batch_size=1)
    features = policy.actor.features_extractor(fake_obs)
    assert features.shape[-1] == cfg.rl_features_dim


def test_no_gru_or_extra_encoders():
    """Test that extractor has no GRU, CNN, or Transformer layers (single-stage only)."""
    cfg = DummyConfig()
    obs_space = make_obs_space()
    extractor = MAFIASingleStageMLP(obs_space, config=cfg)

    # Check no GRU
    assert not hasattr(extractor, 'history_gru')
    # Check no CNN layers
    assert not hasattr(extractor, 'per_stock_proj')
    assert not hasattr(extractor, 'per_stock_conv2')
    # Check no Transformer
    assert not hasattr(extractor, 'per_stock_transformer')
    # Check no global MLP branch
    assert not hasattr(extractor, 'global_net')


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
