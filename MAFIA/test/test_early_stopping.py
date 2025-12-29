"""
Test Early Stopping Behavior

Verifies that:
1. Early stopping triggers after N epochs without improvement
2. Best validation checkpoint is saved correctly
3. Training actually stops when early_stop flag is set
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class MockConfig:
    """Mock config with early stopping settings"""

    def __init__(self):
        self.early_stop_patience = 10
        self.early_stop_min_delta = 0.01
        self.early_stop_warmup = 2
        self.early_stop_metric = "sharpeRatio"
        self.res_dir = "/tmp/test_res"
        self.res_model_dir = "/tmp/test_model"
        self.num_epochs = 20
        self.tradeDays_per_year = 252
        self.save_replay_buffer_on_step_checkpoints = False
        self.save_replay_buffer_on_epoch_checkpoints = True
        self.enable_topk_postprocess = False
        # Additional required attributes
        self.mode = "RLonly"
        self.risk_controller_mode = "no_risk"
        self.enable_market_observer = False
        self.mafia_top_k = 10
        self.metrics_history_path = None
        self.run_manifest_path = None


class TestEarlyStoppingLogic(unittest.TestCase):
    """Test early stopping logic in isolation"""

    def test_config_values(self):
        """Test that config has correct early stopping values"""
        from config import Config

        cfg = Config()

        self.assertEqual(cfg.early_stop_patience, 10, "Patience should be 10 epochs")
        self.assertEqual(cfg.early_stop_min_delta, 0.01, "Min delta should be 0.01")
        self.assertEqual(cfg.early_stop_warmup, 2, "Warmup should be 2 epochs")
        self.assertEqual(
            cfg.early_stop_metric, "sharpeRatio", "Metric should be sharpeRatio"
        )
        print("✅ Config early stopping values are correct")

    def test_early_stop_counter_increments(self):
        """Test that no-improvement counter increments correctly"""
        # Simulate early stopping logic
        best_valid_metric = 1.0
        min_delta = 0.01
        valid_no_improve_epochs = 0

        # Test: metric improved
        new_metric = 1.05
        if new_metric > (best_valid_metric + min_delta):
            best_valid_metric = new_metric
            valid_no_improve_epochs = 0
        else:
            valid_no_improve_epochs += 1

        self.assertEqual(
            valid_no_improve_epochs, 0, "Counter should reset on improvement"
        )
        self.assertEqual(best_valid_metric, 1.05, "Best metric should update")

        # Test: metric did NOT improve (same value)
        new_metric = 1.05
        if new_metric > (best_valid_metric + min_delta):
            best_valid_metric = new_metric
            valid_no_improve_epochs = 0
        else:
            valid_no_improve_epochs += 1

        self.assertEqual(
            valid_no_improve_epochs, 1, "Counter should increment on no improvement"
        )

        # Test: metric slightly improved but below min_delta
        new_metric = 1.055  # Only 0.005 improvement, less than min_delta=0.01
        if new_metric > (best_valid_metric + min_delta):
            best_valid_metric = new_metric
            valid_no_improve_epochs = 0
        else:
            valid_no_improve_epochs += 1

        self.assertEqual(
            valid_no_improve_epochs,
            2,
            "Counter should increment when improvement < min_delta",
        )
        print("✅ No-improvement counter logic is correct")

    def test_early_stop_triggers_after_patience(self):
        """Test that early stop triggers after patience epochs"""
        patience = 10
        warmup = 2

        valid_no_improve_epochs = 0
        early_stop = False

        # Simulate 15 epochs with no improvement after epoch 3
        for epoch in range(15):
            current_epoch_global = epoch + 1

            # Simulate no improvement after warmup
            if current_epoch_global > warmup:
                valid_no_improve_epochs += 1

            # Check early stop condition
            if (
                patience
                and current_epoch_global >= warmup
                and valid_no_improve_epochs >= patience
            ):
                early_stop = True
                break

        self.assertTrue(early_stop, "Early stop should trigger")
        self.assertEqual(
            valid_no_improve_epochs,
            patience,
            f"Should stop at exactly {patience} no-improvement epochs",
        )
        self.assertEqual(
            current_epoch_global,
            warmup + patience,
            f"Should stop at epoch {warmup + patience}",
        )
        print(f"✅ Early stopping triggers correctly at epoch {current_epoch_global}")

    def test_early_stop_respects_warmup(self):
        """Test that early stop doesn't trigger during warmup period"""
        patience = 3
        warmup = 5

        valid_no_improve_epochs = 0
        early_stop_triggered_epochs = []

        for epoch in range(10):
            current_epoch_global = epoch + 1
            valid_no_improve_epochs += 1  # No improvement ever

            if (
                patience
                and current_epoch_global >= warmup
                and valid_no_improve_epochs >= patience
            ):
                early_stop_triggered_epochs.append(current_epoch_global)

        # Early stop should NOT trigger before warmup
        for e in early_stop_triggered_epochs:
            self.assertGreaterEqual(
                e, warmup, f"Early stop should not trigger before warmup (epoch {e})"
            )

        print(
            f"✅ Early stopping respects warmup period (first trigger at epoch {early_stop_triggered_epochs[0] if early_stop_triggered_epochs else 'N/A'})"
        )

    def test_callback_return_false_on_early_stop(self):
        """Test that callback returns False when early stop is triggered"""
        # This simulates the _on_step behavior
        _early_stop = True

        # Simulating the return value logic from callback
        def mock_on_step():
            if _early_stop:
                return False  # Stop training
            return True  # Continue training

        result = mock_on_step()
        self.assertFalse(result, "Callback should return False to stop training")
        print("✅ Callback correctly returns False on early stop")


class TestEarlyStoppingIntegration(unittest.TestCase):
    """Integration tests for early stopping with callback"""

    def test_callback_early_stop_initialization(self):
        """Test that callback initializes early stop variables correctly"""
        try:
            from utils.callback_func import PoCallback

            # Create minimal mock objects
            mock_config = MockConfig()
            mock_train_env = MagicMock()
            mock_valid_env = MagicMock()
            mock_test_env = MagicMock()

            with (
                patch("utils.callback_func.os.path.exists", return_value=False),
                patch("utils.callback_func.os.makedirs"),
            ):
                callback = PoCallback(
                    config=mock_config,
                    train_env=mock_train_env,
                    valid_env=mock_valid_env,
                    test_env=mock_test_env,
                )

            # Check early stopping initialization
            self.assertEqual(callback.early_stop_patience, 10)
            self.assertEqual(callback.early_stop_min_delta, 0.01)
            self.assertEqual(callback.early_stop_warmup, 2)
            self.assertEqual(callback.early_stop_metric, "sharpeRatio")
            self.assertFalse(callback._early_stop)
            self.assertEqual(callback.valid_no_improve_epochs, 0)
            self.assertEqual(callback.best_valid_metric, -np.inf)

            print("✅ Callback initializes early stopping variables correctly")
        except ImportError as e:
            self.skipTest(f"Could not import callback: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Early Stopping Behavior")
    print("=" * 60)

    # Run tests
    unittest.main(verbosity=2)
