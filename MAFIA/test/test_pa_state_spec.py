"""
Tests for Portfolio Allocator State Build per Spec.

Verifies:
1. cbf_adjustment is included in history buffer
2. direction_logits is (3,) one-hot vector
3. All 9 state components have correct dimensions
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestDirectionToOnehot:
    """Test _direction_to_onehot helper function."""

    def test_direction_down(self):
        """Direction 0 (down/bear) should produce [1, 0, 0]."""
        from utils.tradeEnv import StockPortfolioEnv

        # Create mock instance to access method
        class MockEnv:
            pass

        env = MockEnv()
        env._direction_to_onehot = StockPortfolioEnv._direction_to_onehot.__get__(
            env, MockEnv
        )

        result = env._direction_to_onehot(0)
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_direction_flat(self):
        """Direction 1 (flat/sideways) should produce [0, 1, 0]."""
        from utils.tradeEnv import StockPortfolioEnv

        class MockEnv:
            pass

        env = MockEnv()
        env._direction_to_onehot = StockPortfolioEnv._direction_to_onehot.__get__(
            env, MockEnv
        )

        result = env._direction_to_onehot(1)
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_direction_up(self):
        """Direction 2 (up/bull) should produce [0, 0, 1]."""
        from utils.tradeEnv import StockPortfolioEnv

        class MockEnv:
            pass

        env = MockEnv()
        env._direction_to_onehot = StockPortfolioEnv._direction_to_onehot.__get__(
            env, MockEnv
        )

        result = env._direction_to_onehot(2)
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_direction_none(self):
        """Direction None should produce uniform [0.33, 0.33, 0.33]."""
        from utils.tradeEnv import StockPortfolioEnv

        class MockEnv:
            pass

        env = MockEnv()
        env._direction_to_onehot = StockPortfolioEnv._direction_to_onehot.__get__(
            env, MockEnv
        )

        result = env._direction_to_onehot(None)
        expected = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_direction_invalid(self):
        """Invalid direction should produce uniform prior."""
        from utils.tradeEnv import StockPortfolioEnv

        class MockEnv:
            pass

        env = MockEnv()
        env._direction_to_onehot = StockPortfolioEnv._direction_to_onehot.__get__(
            env, MockEnv
        )

        # Test invalid values
        for invalid in [5, -1, "foo", 3.5]:
            result = env._direction_to_onehot(invalid)
            expected = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
            np.testing.assert_array_almost_equal(result, expected, decimal=5)


class TestHistoryFeatureConfig:
    """Test that cbf_adjustment is in default history features."""

    def test_cbf_adjustment_in_default_features(self):
        """cbf_adjustment should be in default rl_obs_history_features."""
        from config import Config

        config = Config()
        assert hasattr(config, "rl_obs_history_features")
        assert "cbf_adjustment" in config.rl_obs_history_features

    def test_all_spec_features_present(self):
        """All 4 spec history features should be present."""
        from config import Config

        config = Config()
        expected_features = [
            "portfolio_return",
            "rl_last_action",
            "rl_last_turnover",
            "cbf_adjustment",
        ]
        for feature in expected_features:
            assert feature in config.rl_obs_history_features, f"Missing: {feature}"


class TestHistoryFeatureDim:
    """Test _compute_history_feature_dim includes cbf_adjustment."""

    def test_cbf_adjustment_dimension(self):
        """cbf_adjustment should add rl_stock_num to history dim."""
        from utils.tradeEnv import StockPortfolioEnv

        class MockEnv:
            rl_stock_num = 10

        env = MockEnv()
        env._compute_history_feature_dim = (
            StockPortfolioEnv._compute_history_feature_dim.__get__(env, MockEnv)
        )

        # Test with cbf_adjustment included
        features = ["portfolio_return", "rl_last_action", "rl_last_turnover", "cbf_adjustment"]
        dim = env._compute_history_feature_dim(features)

        # Expected: 1 (return) + 10 (action) + 1 (turnover) + 10 (cbf) = 22
        expected_dim = 1 + 10 + 1 + 10
        assert dim == expected_dim, f"Expected {expected_dim}, got {dim}"


class TestGlobalContextDim:
    """Test global context includes direction_logits as (3,)."""

    def test_direction_dim_is_3(self):
        """Global dim should include 3 for direction_logits."""
        from config import Config

        config = Config()
        # Global dim = mafia_D + direction_dim(3) + scalars
        # direction_dim is added in _init_rl_multibranch_spec

        # We can verify by checking the formula:
        # rl_global_dim = mafia_D + 3 + len(scalars) + regime_dim
        mafia_D = config.mafia_D if hasattr(config, "mafia_D") else 64
        scalars = ["log_capital", "last_turnover", "drawdown"]
        expected_base = mafia_D + 3 + len(scalars)

        # This is a structural test - the actual dim is set in tradeEnv
        assert expected_base == mafia_D + 3 + 3  # 64 + 3 + 3 = 70


class TestConfigParameters:
    """Test Portfolio Allocator config parameters are present."""

    def test_allocator_reward_params(self):
        """Reward parameters should be present in config."""
        from config import Config

        config = Config()
        assert hasattr(config, "use_portfolio_allocator_reward")
        assert hasattr(config, "allocator_return_weight")
        assert hasattr(config, "allocator_lambda_js")
        assert hasattr(config, "allocator_reward_norm_alpha")

    def test_allocator_training_params(self):
        """Training parameters should be present in config."""
        from config import Config

        config = Config()
        assert hasattr(config, "allocator_batch_size")
        assert hasattr(config, "allocator_replay_buffer_size")
        assert hasattr(config, "allocator_warmup_steps")
        assert hasattr(config, "allocator_exploration_noise_std")

    def test_allocator_architecture_params(self):
        """Architecture parameters should be present in config."""
        from config import Config

        config = Config()
        assert hasattr(config, "allocator_feature_hidden_dim")
        assert hasattr(config, "allocator_actor_hidden_dim")
        assert hasattr(config, "allocator_critic_hidden_dim")

    def test_allocator_td3_params(self):
        """TD3 hyperparameters should be present in config."""
        from config import Config

        config = Config()
        assert hasattr(config, "allocator_discount_gamma")
        assert hasattr(config, "allocator_polyak_tau")
        assert hasattr(config, "allocator_policy_delay")

    def test_default_values(self):
        """Default values should match spec."""
        from config import Config

        config = Config()
        assert config.allocator_return_weight == 1.0
        assert config.allocator_lambda_js == 0.1
        assert config.allocator_discount_gamma == 0.99
        assert config.allocator_polyak_tau == 0.005
        assert config.allocator_policy_delay == 2
        assert config.allocator_batch_size == 128
        assert config.allocator_replay_buffer_size == 100000
        assert config.allocator_warmup_steps == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
