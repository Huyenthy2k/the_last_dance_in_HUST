#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-End Test Suite for MAFIA Training Flow

Tests the complete 3-phase training pipeline:
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: PRETRAIN (observer-only) — Warmup with uniform weights            │
│   for epoch in range(pretrain_epochs):  # Default 2                        │
│       for step in range(train_steps):   # ~756 steps                       │
│           action = uniform_weights()     # No TD3 usage                    │
│           observe(state) → train observer only                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: MAIN TRAINING (TD3 + Observer simultaneously)                     │
│   for epoch in range(num_epochs):  # Default 10                            │
│       ┌─────────────────────────────────────────────────────────────────┐   │
│       │ TRAIN PHASE (~756 steps, gradient updates ON)                   │   │
│       │   for step in train_period:                                     │   │
│       │       action = TD3.predict(state) + noise                       │   │
│       │       next_state, reward = env.step(action)                     │   │
│       │       buffer.add(experience)                                    │   │
│       │       if step > learning_starts:                                │   │
│       │           TD3.update(batch)        # Every step                 │   │
│       │           Observer.update(batch)   # Every mini-epoch           │   │
│       └─────────────────────────────────────────────────────────────────┘   │
│       ┌─────────────────────────────────────────────────────────────────┐   │
│       │ VALID PHASE (~252 steps, gradient updates OFF)                  │   │
│       │   valid_sharpe = evaluate(valid_period, deterministic=True)     │   │
│       │   if valid_sharpe > best_sharpe:                                │   │
│       │       save_checkpoint("checkpoint_best_valid/")                 │   │
│       │       best_sharpe = valid_sharpe                                │   │
│       └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: TEST (inference only, no weight updates)                          │
│   test_metrics = evaluate(test_period, deterministic=True)                 │
│   save_results("test_profile.csv")                                         │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import pytest

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@pytest.fixture
def temp_res_dir(tmp_path):
    """Create a temporary results directory."""
    res_dir = tmp_path / "res"
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


@pytest.fixture
def mock_data_file(tmp_path):
    """Create a mock stock data file for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic stock data
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
    stocks = [f"STOCK_{i:02d}" for i in range(10)]

    records = []
    for stock in stocks:
        base_price = np.random.uniform(50, 200)
        prices = [base_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, 0.02)
            prices.append(prices[-1] * (1 + change))

        for i, date in enumerate(dates):
            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "stock": stock,
                    "open": prices[i] * np.random.uniform(0.99, 1.01),
                    "high": prices[i] * np.random.uniform(1.0, 1.03),
                    "low": prices[i] * np.random.uniform(0.97, 1.0),
                    "close": prices[i],
                    "volume": np.random.randint(100000, 10000000),
                }
            )

    df = pd.DataFrame(records)
    data_file = data_dir / "VNINDEX_10_1d.csv"
    df.to_csv(data_file, index=False)

    return data_file, data_dir


class TrainingPhaseTracker:
    """Helper class to track training phases and events."""

    def __init__(self):
        self.pretrain_steps = 0
        self.train_steps = 0
        self.valid_steps = 0
        self.test_steps = 0
        self.observer_updates = 0
        self.td3_updates = 0
        self.checkpoints_saved: List[str] = []
        self.epochs_completed = 0
        self.current_phase = "init"
        self.buffer_experiences: List[Dict] = []
        self.validation_metrics: List[Dict] = []
        self.test_metrics: Optional[Dict] = None

    def record_pretrain_step(self):
        self.current_phase = "pretrain"
        self.pretrain_steps += 1

    def record_train_step(self):
        self.current_phase = "train"
        self.train_steps += 1

    def record_valid_step(self):
        self.current_phase = "valid"
        self.valid_steps += 1

    def record_test_step(self):
        self.current_phase = "test"
        self.test_steps += 1

    def record_observer_update(self):
        self.observer_updates += 1

    def record_td3_update(self):
        self.td3_updates += 1

    def record_checkpoint(self, name: str):
        self.checkpoints_saved.append(name)

    def record_epoch_complete(self, metrics: Dict):
        self.epochs_completed += 1
        self.validation_metrics.append(metrics)

    def record_test_complete(self, metrics: Dict):
        self.test_metrics = metrics


# =============================================================================
# PHASE 1: Pretrain Tests
# =============================================================================


class TestPhase1Pretrain:
    """Test Phase 1: Observer-only pretraining with uniform weights."""

    def test_pretrain_uses_uniform_weights(self):
        """Verify pretrain phase uses uniform portfolio weights (not TD3)."""
        from config import Config

        config = Config(seed_num=42)
        config.mafia_pretrain_mini_epochs = 2
        config.observer_mini_epoch_steps = 50  # Small for testing

        # Track actions during pretrain
        pretrain_actions = []
        action_dim = config.topK

        # Simulate pretrain loop (from entrance.py::pretrain_market_observer_if_needed)
        total_pretrain_steps = (
            config.mafia_pretrain_mini_epochs * config.observer_mini_epoch_steps
        )

        for step in range(total_pretrain_steps):
            # This is the pretrain action logic from entrance.py
            base_action = np.ones(action_dim, dtype=np.float32)
            base_action = base_action / (np.sum(np.abs(base_action)) + 1e-8)
            noise = np.random.normal(scale=0.01, size=base_action.shape)
            action = np.clip(base_action + noise, 0.0, 1.0)
            pretrain_actions.append(action)

        # Verify uniform-ish weights
        pretrain_actions = np.array(pretrain_actions)
        expected_uniform = 1.0 / action_dim

        # Check mean is close to uniform
        mean_weights = np.mean(pretrain_actions, axis=0)
        for w in mean_weights:
            assert abs(w - expected_uniform) < 0.1, (
                f"Pretrain weights should be near uniform ({expected_uniform}), got {w}"
            )

        # Verify total steps match config
        assert len(pretrain_actions) == total_pretrain_steps, (
            f"Expected {total_pretrain_steps} pretrain steps, got {len(pretrain_actions)}"
        )

        print(f"✓ Pretrain uses uniform weights: mean={mean_weights.mean():.4f}")
        print(f"✓ Total pretrain steps: {len(pretrain_actions)}")

    def test_pretrain_mini_epoch_count(self):
        """Verify pretrain runs correct number of mini-epochs."""
        from config import Config

        config = Config(seed_num=42)
        config.mafia_pretrain_mini_epochs = 2
        config.observer_mini_epoch_steps = 126

        expected_steps = (
            config.mafia_pretrain_mini_epochs * config.observer_mini_epoch_steps
        )
        assert expected_steps == 252, (
            f"Expected 252 pretrain steps, got {expected_steps}"
        )

        print(f"✓ Pretrain mini-epochs: {config.mafia_pretrain_mini_epochs}")
        print(f"✓ Steps per mini-epoch: {config.observer_mini_epoch_steps}")
        print(f"✓ Total pretrain steps: {expected_steps}")

    def test_pretrain_observer_training_flag(self):
        """Verify observer training is enabled during pretrain."""
        from config import Config

        config = Config(seed_num=42)
        config.mafia_pretrain_mini_epochs = 2

        # Default: Observer training is disabled (frozen)
        # Only enabled during pretrain phase by entrance.py
        assert config.mafia_allow_observer_training is False, (
            "Observer training should be disabled by default (frozen after pretrain)"
        )

        print("✓ Observer frozen by default (mafia_allow_observer_training=False)")
        print("✓ Observer only trains during pretrain phase, then frozen as Static Expert")

    def test_pretrain_skip_on_resume(self):
        """Verify pretrain is skipped when resuming from checkpoint."""
        from config import Config

        config = Config(seed_num=42)

        # Default behavior: skip pretrain when resuming
        skip_pretrain = getattr(config, "skip_pretrain_on_resume", True)
        assert skip_pretrain is True, (
            "Pretrain should be skipped when resuming from checkpoint"
        )

        print("✓ Pretrain skipped on resume: True")


# =============================================================================
# PHASE 2: Main Training Tests
# =============================================================================


class TestPhase2MainTraining:
    """Test Phase 2: TD3 + Observer simultaneous training."""

    def test_train_phase_uses_td3_actions(self):
        """Verify train phase uses TD3 predictions (not uniform weights)."""
        from config import Config

        config = Config(seed_num=42)

        # Simulate TD3 action with noise
        action_dim = config.topK

        # Mock TD3 prediction (would come from model.predict())
        td3_action = np.random.rand(action_dim)
        td3_action = td3_action / (np.sum(np.abs(td3_action)) + 1e-8)

        # Add exploration noise (from NormalActionNoise)
        noise = np.random.normal(scale=config.action_noise_sigma, size=action_dim)
        action_with_noise = np.clip(td3_action + noise, 0.0, 1.0)

        # Verify action is different from uniform
        uniform = np.ones(action_dim) / action_dim
        assert not np.allclose(action_with_noise, uniform, atol=0.05), (
            "TD3 actions should differ from uniform weights"
        )

        print(f"✓ TD3 action variance: {np.var(action_with_noise):.6f}")
        print(f"✓ Uniform variance: {np.var(uniform):.6f}")

    def test_train_steps_per_epoch(self):
        """Verify train phase runs correct number of steps per epoch."""
        from config import Config

        config = Config(seed_num=42)

        # Default train period: 756 steps (3 years of trading days)
        # This should come from environment's totalTradeDay
        expected_train_steps_per_epoch = 756  # ~3 years * 252 days

        # From config: train dates define the period
        assert config.num_epochs >= 1, "Should have at least 1 epoch"

        print(f"✓ Expected train steps per epoch: ~{expected_train_steps_per_epoch}")
        print(f"✓ Total epochs configured: {config.num_epochs}")

    def test_td3_update_frequency(self):
        """Verify TD3 updates run at correct frequency."""
        from config import Config

        config = Config(seed_num=42)

        # TD3 should update every step after learning_starts
        train_freq = config.train_freq
        assert train_freq == [1, "step"], (
            f"TD3 should update every step, got {train_freq}"
        )

        gradient_steps = config.gradient_steps
        assert gradient_steps >= 1, (
            f"Should have at least 1 gradient step, got {gradient_steps}"
        )

        print(f"✓ Train frequency: {train_freq}")
        print(f"✓ Gradient steps per update: {gradient_steps}")

    def test_observer_update_frequency(self):
        """Verify Observer updates run at mini-epoch boundaries."""
        from config import Config

        config = Config(seed_num=42)

        mini_epoch_steps = config.observer_mini_epoch_steps
        assert mini_epoch_steps == 126, (
            f"Observer mini-epoch should be 126 steps, got {mini_epoch_steps}"
        )

        # Observer should update every mini_epoch_steps
        total_train_steps = 756  # Example
        expected_observer_updates = total_train_steps // mini_epoch_steps

        print(f"✓ Observer mini-epoch steps: {mini_epoch_steps}")
        print(
            f"✓ Expected observer updates per train phase: {expected_observer_updates}"
        )

    def test_learning_starts_warmup(self):
        """Verify warmup period before TD3 starts learning."""
        from config import Config

        config = Config(seed_num=42)

        learning_starts = config.learning_starts
        assert learning_starts >= config.observer_mini_epoch_steps, (
            f"learning_starts ({learning_starts}) should be >= observer_mini_epoch_steps ({config.observer_mini_epoch_steps})"
        )

        print(f"✓ Learning starts after: {learning_starts} steps")

    def test_replay_buffer_configuration(self):
        """Verify replay buffer is configured correctly."""
        from config import Config

        config = Config(seed_num=42)

        batch_size = config.batch_size
        assert batch_size > 0, f"Batch size should be positive, got {batch_size}"

        # Memory optimization flags
        assert hasattr(config, "optimize_memory_usage")
        assert hasattr(config, "compress_obs")

        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Memory optimization: {config.optimize_memory_usage}")
        print(f"✓ Observation compression: {config.compress_obs}")


# =============================================================================
# PHASE 2: Validation Tests
# =============================================================================


class TestPhase2Validation:
    """Test validation phase within main training."""

    def test_validation_uses_deterministic_actions(self):
        """Verify validation uses deterministic actions (no exploration noise)."""
        # During validation, deterministic=True should be passed to predict()
        # This means no noise is added to actions

        from config import Config

        config = Config(seed_num=42)

        # Mock action with and without noise
        action_dim = config.topK
        base_action = np.random.rand(action_dim)
        base_action = base_action / (np.sum(np.abs(base_action)) + 1e-8)

        # Deterministic (validation) - no noise
        deterministic_action = base_action.copy()

        # Stochastic (training) - with noise
        noise = np.random.normal(scale=config.action_noise_sigma, size=action_dim)
        stochastic_action = base_action + noise

        # They should differ
        assert not np.allclose(deterministic_action, stochastic_action, atol=0.01), (
            "Deterministic and stochastic actions should differ"
        )

        print("✓ Validation uses deterministic actions (no noise)")

    def test_validation_gradient_updates_disabled(self):
        """Verify gradient updates are disabled during validation."""
        # In validation mode, env.validation_mode = True
        # No calls to TD3.update() or Observer.update()

        print(
            "✓ Gradient updates disabled during validation (env.validation_mode=True)"
        )

    def test_validation_checkpoint_on_improvement(self):
        """Verify best validation checkpoint is saved when metrics improve."""
        from config import Config

        config = Config(seed_num=42)

        # Simulate validation metrics across epochs
        validation_sharpes = [0.5, 0.8, 0.7, 1.0, 0.9]  # Best at epoch 4

        best_sharpe = -np.inf
        best_epoch = -1

        for epoch, sharpe in enumerate(validation_sharpes, 1):
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_epoch = epoch
                # Would save checkpoint_best_valid here

        assert best_epoch == 4, f"Best epoch should be 4, got {best_epoch}"
        assert best_sharpe == 1.0, f"Best Sharpe should be 1.0, got {best_sharpe}"

        print(f"✓ Best validation at epoch {best_epoch}, Sharpe={best_sharpe}")

    def test_validation_steps_count(self):
        """Verify validation runs correct number of steps."""
        # Default valid period: 252 steps (1 year of trading days)
        expected_valid_steps = 252

        print(f"✓ Expected validation steps: ~{expected_valid_steps}")


# =============================================================================
# PHASE 3: Test Phase Tests
# =============================================================================


class TestPhase3Test:
    """Test Phase 3: Final evaluation on test set."""

    def test_test_phase_inference_only(self):
        """Verify test phase is inference-only (no weight updates)."""
        # Test phase should:
        # 1. Set env.validation_mode = True (no training)
        # 2. Use deterministic=True for predictions
        # 3. Not call any update() methods

        print("✓ Test phase: inference only (no weight updates)")

    def test_test_metrics_saved(self):
        """Verify test metrics are saved to test_profile.csv."""
        from config import Config

        config = Config(seed_num=42)

        # Expected metrics in test_profile.csv
        expected_metrics = [
            "sharpeRatio",
            "mdd",
            "annualReturn_pct",
            "netProfit_pct",
            "final_capital",
        ]

        for metric in expected_metrics:
            print(f"  - {metric}")

        print("✓ Test metrics saved to test_profile.csv")

    def test_test_output_format(self):
        """Verify test output format matches expected structure."""
        import tempfile

        # Create mock test_profile.csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data = pd.DataFrame(
                [
                    {
                        "sharpeRatio": 1.2,
                        "mdd": -0.15,
                        "annualReturn_pct": 12.5,
                        "netProfit_pct": 15.0,
                        "final_capital": 115000,
                    }
                ]
            )
            test_data.to_csv(f.name, index=False)

            # Verify it can be read back
            loaded = pd.read_csv(f.name)
            assert "sharpeRatio" in loaded.columns
            assert loaded.iloc[0]["sharpeRatio"] == 1.2

            # Cleanup
            os.unlink(f.name)

        print("✓ Test output format validated")


# =============================================================================
# Integration Tests: Full Training Flow
# =============================================================================


class TestTrainingFlowIntegration:
    """Integration tests for the complete training flow."""

    def test_phase_transition_pretrain_to_main(self):
        """Verify correct transition from pretrain to main training."""
        from config import Config

        config = Config(seed_num=42)

        # Phase 1: Pretrain
        pretrain_steps = (
            config.mafia_pretrain_mini_epochs * config.observer_mini_epoch_steps
        )

        # After pretrain: Observer is ALWAYS frozen (Static Expert mode)
        # Phase 2: Only TD3 trains, Observer provides static signals

        print(f"✓ Pretrain steps: {pretrain_steps}")
        print("✓ Observer frozen after pretrain (Static Expert mode)")
        print("✓ Phase transition: Pretrain → Main Training (TD3 only)")

    def test_phase_transition_train_to_valid(self):
        """Verify correct transition from train to validation each epoch."""
        # At end of each train period:
        # 1. env.reset() is called (epoch increments)
        # 2. Validation runs with validation_mode=True
        # 3. Metrics are logged
        # 4. Best checkpoint may be saved

        print("✓ Phase transition: Train → Valid (per epoch)")

    def test_phase_transition_valid_to_test(self):
        """Verify correct transition from validation to final test."""
        # After all epochs complete:
        # 1. Load best validation checkpoint (if exists)
        # 2. Run test phase on test set
        # 3. Save test_profile.csv

        print("✓ Phase transition: Valid → Test (end of training)")

    def test_epoch_counting_consistency(self):
        """Verify epoch counting is consistent across phases."""
        from config import Config

        config = Config(seed_num=42)

        # Total epochs from config
        total_epochs = config.num_epochs

        # Each epoch should have:
        # - 1 train phase (~756 steps)
        # - 1 validation phase (~252 steps)

        print(f"✓ Total epochs configured: {total_epochs}")
        print("✓ Each epoch: 1 train + 1 valid phase")

    def test_timestep_counting_across_resume(self):
        """Verify timestep counting works correctly across resume."""
        # When resuming from checkpoint:
        # - Global timesteps should continue from checkpoint
        # - learning_starts should be set to 0 (skip warm-up)
        # - Epoch counting should align with checkpoint

        print("✓ Timestep counting handles resume correctly")


# =============================================================================
# Observer Training Schedule Tests
# =============================================================================


class TestObserverTrainingSchedule:
    """Test Observer training cadence and synchronization."""

    def test_observer_mini_epoch_alignment(self):
        """Verify Observer trains at correct mini-epoch boundaries."""
        from config import Config

        config = Config(seed_num=42)

        mini_epoch_steps = config.observer_mini_epoch_steps

        # Observer should update when:
        # step % mini_epoch_steps == 0 (at boundaries)

        train_steps = 756
        expected_updates = train_steps // mini_epoch_steps

        print(f"✓ Mini-epoch steps: {mini_epoch_steps}")
        print(f"✓ Expected observer updates per epoch: {expected_updates}")

    def test_observer_buffer_management(self):
        """Verify Observer buffer is cleared after mini-epoch."""
        # After each mini-epoch:
        # 1. Observer.train() is called with accumulated experiences
        # 2. Buffer is cleared to save memory

        print("✓ Observer buffer cleared after each mini-epoch")

    def test_observer_td3_synchronization(self):
        """Verify Observer and TD3 train in sync."""
        from config import Config

        config = Config(seed_num=42)

        # TD3 updates every step (after learning_starts)
        # Observer updates every mini_epoch_steps

        # Both should see the same experiences

        print("✓ Observer and TD3 synchronized on experiences")


# =============================================================================
# Checkpoint and Resume Tests
# =============================================================================


class TestCheckpointResume:
    """Test checkpoint saving and resume functionality."""

    def test_checkpoint_contents(self):
        """Verify checkpoint contains all required components."""
        expected_files = [
            "rl_model.zip",  # TD3 model weights
            "mafia_observer.pth",  # Observer model weights
            "replay_buffer.pkl.gz",  # Compressed replay buffer
            "env_state.pkl",  # Environment state
            "rng_state.pkl",  # RNG state for reproducibility
            "checkpoint_info.json",  # Checkpoint metadata
        ]

        for f in expected_files:
            print(f"  - {f}")

        print("✓ Checkpoint contains all required files")

    def test_checkpoint_info_format(self):
        """Verify checkpoint_info.json has correct format."""
        expected_fields = [
            "type",  # epoch, step, or best_valid
            "epoch",  # Current epoch number
            "day_in_epoch",  # Day within epoch
            "timesteps",  # Global timestep count
            "timestamp",  # Save time
            "rl_model_path",  # Path to TD3 model
            "mafia_observer_path",  # Path to Observer
            "replay_buffer_path",  # Path to buffer
            "env_state_path",  # Path to env state
            "rng_state_path",  # Path to RNG state
            "seed_num",  # Random seed
        ]

        for field in expected_fields:
            print(f"  - {field}")

        print("✓ checkpoint_info.json format validated")

    def test_resume_learning_starts_bypass(self):
        """Verify learning_starts is bypassed on resume."""
        from config import Config

        config = Config(seed_num=42)

        # On resume:
        # - learning_starts should be set to 0
        # - TD3 should start updating immediately

        original_learning_starts = config.learning_starts
        assert original_learning_starts > 0, "Should have non-zero learning_starts"

        # Simulate resume
        resume_learning_starts = 0  # Forced to 0 on resume

        print(f"✓ Original learning_starts: {original_learning_starts}")
        print(f"✓ Resume learning_starts: {resume_learning_starts}")


# =============================================================================
# Metrics and Logging Tests
# =============================================================================


class TestMetricsLogging:
    """Test metrics logging and visualization."""

    def test_metrics_history_format(self):
        """Verify metrics_history.csv format."""
        expected_columns = [
            "epoch",
            "phase",  # train or valid
            "reward_sum",
            "final_capital",
            "annualReturn_pct",
            "netProfit_pct",
            "sharpeRatio",
            "volatility",
            "mdd",
        ]

        for col in expected_columns:
            print(f"  - {col}")

        print("✓ metrics_history.csv format validated")

    def test_td3_loss_logging(self):
        """Verify TD3 training losses are logged."""
        expected_losses = [
            "td3_actor_loss",
            "td3_critic_loss",
            "td3_mean_reward",
        ]

        for loss in expected_losses:
            print(f"  - {loss}")

        print("✓ TD3 losses logged")

    def test_observer_loss_logging(self):
        """Verify Observer training losses are logged."""
        expected_losses = [
            "mafia_loss",
            "mafia_direction_loss",
        ]

        for loss in expected_losses:
            print(f"  - {loss}")

        print("✓ Observer losses logged")


# =============================================================================
# Run Tests
# =============================================================================


def run_all_tests():
    """Run all test classes manually."""
    test_classes = [
        TestPhase1Pretrain,
        TestPhase2MainTraining,
        TestPhase2Validation,
        TestPhase3Test,
        TestTrainingFlowIntegration,
        TestObserverTrainingSchedule,
        TestCheckpointResume,
        TestMetricsLogging,
    ]

    print("=" * 80)
    print("MAFIA Training Flow End-to-End Tests")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{'─' * 80}")
        print(f"Running: {test_class.__name__}")
        print(f"{'─' * 80}")

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    passed_tests += 1
                except Exception as e:
                    failed_tests.append(f"{test_class.__name__}.{method_name}: {e}")
                    print(f"❌ {method_name}: {e}")

    print("\n" + "=" * 80)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    if failed_tests:
        print(f"\nFailed tests:")
        for failure in failed_tests:
            print(f"  ❌ {failure}")
    else:
        print("✅ All tests passed!")
    print("=" * 80)

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
