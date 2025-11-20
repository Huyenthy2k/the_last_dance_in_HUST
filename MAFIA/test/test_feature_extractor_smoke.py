import numpy as np
import torch
try:
    import gymnasium as gym
    SPACES = gym.spaces
except ImportError:
    import gym
    SPACES = gym.spaces
from stable_baselines3.td3.policies import MultiInputPolicy

from RL_controller.feature_extractors import MAFIAMultiModalExtractor


class DummyConfig:
    rl_features_dim = 64
    rl_per_stock_encoder = 'weighted'  # default
    rl_per_stock_cnn_channels = 32
    rl_per_stock_d_model = 32
    rl_per_stock_nhead = 4
    rl_per_stock_num_layers = 1
    rl_history_hidden_dim = 16


def make_obs_space():
    global_dim = 10
    per_stock = (5, 9)  # N=5, F=9 (score + embedding)
    history = (3, 4)    # H=3, K=4
    return SPACES.Dict({
        'global_context': SPACES.Box(low=-np.inf, high=np.inf, shape=(global_dim,), dtype=np.float32),
        'per_stock': SPACES.Box(low=-np.inf, high=np.inf, shape=per_stock, dtype=np.float32),
        'history': SPACES.Box(low=-np.inf, high=np.inf, shape=history, dtype=np.float32),
    })


def make_fake_obs(batch_size=2):
    obs_space = make_obs_space()
    obs = {}
    for key, space in obs_space.spaces.items():
        sample = torch.tensor(space.sample(), dtype=torch.float32)
        # Repeat along batch dimension
        repeats = (batch_size,) + tuple(1 for _ in space.shape)
        obs[key] = sample.unsqueeze(0).repeat(*repeats)
    return obs


def test_feature_extractor_output_shape():
    cfg = DummyConfig()
    obs_space = make_obs_space()
    extractor = MAFIAMultiModalExtractor(obs_space, config=cfg)
    fake_obs = make_fake_obs(batch_size=2)
    out = extractor(fake_obs)
    assert out.shape == (2, cfg.rl_features_dim)


def test_multiinput_policy_initialization():
    cfg = DummyConfig()
    obs_space = make_obs_space()
    action_space = SPACES.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    policy = MultiInputPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=MAFIAMultiModalExtractor,
        features_extractor_kwargs={'config': cfg},
    )
    fake_obs = make_fake_obs(batch_size=1)
    features = policy.actor.features_extractor(fake_obs)
    assert features.shape[-1] == cfg.rl_features_dim
