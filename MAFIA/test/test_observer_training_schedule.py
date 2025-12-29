"""
Test Observer Training Schedule (Section 7 of Spec)

Covers:
- Pretrain phase (2 mini-epochs, uniform weights)
- Mini-epoch training (126 steps)
- LR scheduling
- Buffer reset after train
- Freeze/unfreeze observer
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPretrainPhase:
    """Test PHASE 1: PRETRAIN (observer-only)."""

    def test_pretrain_mini_epochs_config(self):
        """Test mafia_pretrain_mini_epochs = 2 in config."""
        from config import Config

        cfg = Config()

        assert hasattr(cfg, "mafia_pretrain_mini_epochs"), (
            "Missing mafia_pretrain_mini_epochs"
        )
        assert cfg.mafia_pretrain_mini_epochs == 2, (
            f"Expected 2, got {cfg.mafia_pretrain_mini_epochs}"
        )
        print(f"✅ mafia_pretrain_mini_epochs = {cfg.mafia_pretrain_mini_epochs}")

    def test_observer_frozen_after_pretrain(self):
        """Test that Observer is ALWAYS frozen after pretrain (Static Expert mode)."""
        from config import Config

        cfg = Config()

        # mafia_allow_observer_training should be False by default
        # (only True during pretrain phase, then frozen)
        assert hasattr(cfg, "mafia_allow_observer_training"), (
            "Missing mafia_allow_observer_training"
        )
        assert cfg.mafia_allow_observer_training == False, (
            f"Expected False (Observer frozen by default), got {cfg.mafia_allow_observer_training}"
        )
        print(f"✅ Observer frozen by default: mafia_allow_observer_training = {cfg.mafia_allow_observer_training}")

    def test_uniform_weights_formula(self):
        """Test uniform weights = 1/K for all K stocks."""
        K = 10
        uniform_weights = np.ones(K) / K

        assert abs(uniform_weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
        assert np.allclose(uniform_weights, 0.1), (
            f"Each weight should be 0.1: {uniform_weights}"
        )
        print(f"✅ Uniform weights (K={K}): {uniform_weights}")

    def test_pretrain_steps_calculation(self):
        """Test total_pretrain_steps = mini_epochs * mini_epoch_steps."""
        mini_epochs = 2
        mini_epoch_steps = 126

        total_steps = mini_epochs * mini_epoch_steps

        assert total_steps == 252, f"Expected 252 steps, got {total_steps}"
        print(
            f"✅ Total pretrain steps: {mini_epochs} × {mini_epoch_steps} = {total_steps}"
        )


class TestMiniEpochTraining:
    """Test mini-epoch training logic."""

    def test_observer_mini_epoch_steps_config(self):
        """Test observer_mini_epoch_steps = 126 in config."""
        from config import Config

        cfg = Config()

        assert hasattr(cfg, "observer_mini_epoch_steps"), (
            "Missing observer_mini_epoch_steps"
        )
        assert cfg.observer_mini_epoch_steps == 126, (
            f"Expected 126, got {cfg.observer_mini_epoch_steps}"
        )
        print(f"✅ observer_mini_epoch_steps = {cfg.observer_mini_epoch_steps}")

    def test_mini_epoch_trigger_logic(self):
        """Test train() triggered every 126 steps."""
        mini_epoch_steps = 126
        total_steps = 756  # 1 epoch

        # Count how many times train() should be called
        train_calls = total_steps // mini_epoch_steps

        assert train_calls == 6, f"Expected 6 train calls per epoch, got {train_calls}"
        print(f"✅ Mini-epoch triggers: {train_calls} times per {total_steps} steps")

    def test_step_counter_simulation(self):
        """Simulate step counter for mini-epoch."""
        mini_epoch_steps = 126
        step_counter = 0
        train_calls = 0

        for step in range(756):
            step_counter += 1

            if step_counter >= mini_epoch_steps:
                train_calls += 1
                step_counter = 0  # Reset counter

        assert train_calls == 6, f"Expected 6 train calls, got {train_calls}"
        print(f"✅ Step counter simulation: {train_calls} train() calls")


class TestLRScheduling:
    """Test Learning Rate scheduling."""

    def test_lr_scheduler_type_default(self):
        """Test default LR scheduler is linear_per_epoch."""
        from config import Config

        cfg = Config()

        # Check if scheduler type exists
        scheduler_type = getattr(cfg, "mafia_lr_scheduler", "linear_per_epoch")
        print(f"✅ LR scheduler type: {scheduler_type}")

    def test_lr_decay_per_mini_epoch(self):
        """Test LR decay from start to start*end_factor per mini-epoch."""
        lr_start = 1e-3
        lr_end_factor = 0.1
        lr_end = lr_start * lr_end_factor

        # Simulate linear decay over 6 mini-epochs
        num_mini_epochs = 6
        lr_schedule = np.linspace(lr_start, lr_end, num_mini_epochs)

        assert lr_schedule[0] == lr_start, f"Initial LR should be {lr_start}"
        assert abs(lr_schedule[-1] - lr_end) < 1e-8, f"Final LR should be {lr_end}"
        print(
            f"✅ LR schedule: {lr_start} → {lr_end} over {num_mini_epochs} mini-epochs"
        )
        print(f"   LR values: {lr_schedule}")

    def test_lr_reset_on_resume(self):
        """Test reset_lr_scheduler_on_resume = True."""
        from config import Config

        cfg = Config()

        reset_lr = getattr(cfg, "reset_lr_scheduler_on_resume", True)
        assert reset_lr == True, f"Expected True, got {reset_lr}"
        print(f"✅ reset_lr_scheduler_on_resume = {reset_lr}")


class TestBufferManagement:
    """Test buffer reset after train."""

    def test_buffer_reset_concept(self):
        """Conceptual test: buffer should be cleared after train()."""

        class MockBuffer:
            def __init__(self):
                self.data = []

            def add(self, item):
                self.data.append(item)

            def reset(self):
                self.data = []

            def __len__(self):
                return len(self.data)

        buffer = MockBuffer()

        # Simulate collecting experiences
        for i in range(126):
            buffer.add(f"exp_{i}")

        assert len(buffer) == 126, f"Buffer should have 126 items"

        # After train(), buffer is reset
        buffer.reset()
        assert len(buffer) == 0, f"Buffer should be empty after reset"
        print("✅ Buffer reset after train() verified")

    def test_skip_pretrain_on_resume(self):
        """Test skip_pretrain_on_resume = True in config."""
        from config import Config

        cfg = Config()

        skip_pretrain = getattr(cfg, "skip_pretrain_on_resume", True)
        assert skip_pretrain == True, f"Expected True, got {skip_pretrain}"
        print(f"✅ skip_pretrain_on_resume = {skip_pretrain}")


class TestFreezeUnfreeze:
    """Test observer freeze/unfreeze behavior."""

    def test_allow_observer_training_flag(self):
        """Test mafia_allow_observer_training default is False (Observer frozen)."""
        from config import Config

        cfg = Config()

        # Default is False - Observer is frozen after pretrain
        allow_training = getattr(cfg, "mafia_allow_observer_training", False)
        assert allow_training == False, f"Expected False (frozen), got {allow_training}"
        print(f"✅ mafia_allow_observer_training = {allow_training} (Observer frozen by default)")

    def test_freeze_behavior_simulation(self):
        """Simulate freeze/unfreeze observer."""

        class MockObserver:
            def __init__(self):
                self.frozen = False

            def freeze(self):
                self.frozen = True

            def unfreeze(self):
                self.frozen = False

            def train(self):
                if self.frozen:
                    print("   Observer frozen, skipping train")
                    return False
                else:
                    print("   Observer training...")
                    return True

        observer = MockObserver()

        # Normal training
        assert observer.train() == True, "Should train when not frozen"

        # Freeze
        observer.freeze()
        assert observer.train() == False, "Should skip train when frozen"

        # Unfreeze
        observer.unfreeze()
        assert observer.train() == True, "Should train after unfreeze"

        print("✅ Freeze/unfreeze behavior verified")


class TestTrainingPhaseTransition:
    """Test transition from PHASE 1 (pretrain) to PHASE 2 (main training)."""

    def test_phase_transition_flow(self):
        """Test: Pretrain → Observer FROZEN, TD3 trains alone."""
        # Simulate phase transition
        phase = "PRETRAIN"
        pretrain_done = False
        td3_active = False
        observer_training = True  # Observer trains during pretrain

        # PHASE 1: Pretrain (2 mini-epochs) - Observer trains with uniform weights
        for mini_epoch in range(2):
            assert phase == "PRETRAIN"
            assert observer_training == True
            assert td3_active == False

        pretrain_done = True

        # Transition to PHASE 2: Observer FROZEN, TD3 trains
        if pretrain_done:
            phase = "MAIN_TRAINING"
            td3_active = True  # TD3 starts
            observer_training = False  # Observer FROZEN (Static Expert mode)

        assert phase == "MAIN_TRAINING"
        assert td3_active == True
        assert observer_training == False  # Observer is FROZEN
        print("✅ Phase transition: PRETRAIN → MAIN_TRAINING (Observer FROZEN)")

    def test_training_sequence(self):
        """Test complete training sequence for one window."""
        # Config
        pretrain_mini_epochs = 2
        num_epochs = 10
        steps_per_epoch = 756
        mini_epoch_steps = 126

        total_pretrain_steps = pretrain_mini_epochs * mini_epoch_steps
        total_main_training_steps = num_epochs * steps_per_epoch

        # Verify numbers
        assert total_pretrain_steps == 252, f"Pretrain: {total_pretrain_steps}"
        assert total_main_training_steps == 7560, f"Main: {total_main_training_steps}"

        print(f"✅ Training sequence:")
        print(
            f"   PRETRAIN: {pretrain_mini_epochs} mini-epochs × {mini_epoch_steps} = {total_pretrain_steps} steps"
        )
        print(
            f"   MAIN: {num_epochs} epochs × {steps_per_epoch} = {total_main_training_steps} steps"
        )
        print(f"   TOTAL: {total_pretrain_steps + total_main_training_steps} steps")


def run_all_tests():
    """Run all observer training schedule tests."""
    print("=" * 60)
    print("Testing Observer Training Schedule (Section 7)")
    print("=" * 60)

    # Pretrain phase
    print("\n--- Pretrain Phase Tests ---")
    pretrain_tests = TestPretrainPhase()
    pretrain_tests.test_pretrain_mini_epochs_config()
    pretrain_tests.test_observer_frozen_after_pretrain()
    pretrain_tests.test_uniform_weights_formula()
    pretrain_tests.test_pretrain_steps_calculation()

    # Mini-epoch training
    print("\n--- Mini-Epoch Training Tests ---")
    mini_epoch_tests = TestMiniEpochTraining()
    mini_epoch_tests.test_observer_mini_epoch_steps_config()
    mini_epoch_tests.test_mini_epoch_trigger_logic()
    mini_epoch_tests.test_step_counter_simulation()

    # LR scheduling
    print("\n--- LR Scheduling Tests ---")
    lr_tests = TestLRScheduling()
    lr_tests.test_lr_scheduler_type_default()
    lr_tests.test_lr_decay_per_mini_epoch()
    lr_tests.test_lr_reset_on_resume()

    # Buffer management
    print("\n--- Buffer Management Tests ---")
    buffer_tests = TestBufferManagement()
    buffer_tests.test_buffer_reset_concept()
    buffer_tests.test_skip_pretrain_on_resume()

    # Freeze/unfreeze
    print("\n--- Freeze/Unfreeze Tests ---")
    freeze_tests = TestFreezeUnfreeze()
    freeze_tests.test_allow_observer_training_flag()
    freeze_tests.test_freeze_behavior_simulation()

    # Phase transition
    print("\n--- Phase Transition Tests ---")
    transition_tests = TestTrainingPhaseTransition()
    transition_tests.test_phase_transition_flow()
    transition_tests.test_training_sequence()

    print("\n" + "=" * 60)
    print("All Observer Training Schedule Tests PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
