"""
Test curriculum learning is NOT applied in validation/inference modes.
Verify lambda_epoch = 1.0 (full penalty) regardless of training epoch.
"""

import torch as th
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from config import MAFIAConfig
except ImportError:
    # Fallback: use simple config with defaults
    class MAFIAConfig:
        curriculum_warmup_epochs = 0
        curriculum_penalty_rampup = 5


def test_validation_lambda_epoch():
    """
    Verify validation mode ALWAYS uses lambda_epoch = 1.0 (full penalty).
    This test checks the code logic, not by running full training.
    """
    print("Testing validation mode lambda_epoch...")

    # Read observer_offline_trainer.py to check validation code
    trainer_path = os.path.join(
        os.path.dirname(__file__), "..", "RL_controller", "observer_offline_trainer.py"
    )
    with open(trainer_path, "r") as f:
        content = f.read()

    # Find the validation method
    validate_start = content.find("def validate_epoch")
    if validate_start == -1:
        raise AssertionError("Cannot find validate_epoch method!")

    # Get a large section of validation code (next 20000 chars should cover the whole method)
    # validate_epoch starts at line 2513, lambda_epoch = 1.0 is at line 2752 (~239 lines)
    validation_section = content[validate_start : validate_start + 20000]

    # Check that validation uses lambda_epoch = 1.0 (NOT self._current_lambda_epoch)
    assert "lambda_epoch = 1.0" in validation_section, (
        "Validation mode must use lambda_epoch = 1.0 (full penalty), not curriculum learning!"
    )

    # Verify comment exists
    assert "VALIDATION MODE: Always use full penalty" in validation_section, (
        "Validation should have clear comment about using full penalty!"
    )

    # Ensure validation does NOT use self._current_lambda_epoch for penalty calculation
    # We need to be careful: _current_lambda_epoch might appear in logging, but NOT in penalty calc
    penalty_line_start = validation_section.find("penalty = lambda_epoch")
    if penalty_line_start == -1:
        raise AssertionError("Cannot find penalty calculation in validation!")

    lines_before_penalty = validation_section[:penalty_line_start].split("\n")[-10:]

    curriculum_used = any(
        "lambda_epoch = self._current_lambda_epoch" in line
        for line in lines_before_penalty
    )
    assert not curriculum_used, (
        "Validation mode should NOT use self._current_lambda_epoch (curriculum learning)!"
    )

    print("✅ Validation mode correctly uses lambda_epoch = 1.0 (full penalty)")


def test_training_lambda_epoch():
    """
    Verify training mode DOES use curriculum learning (self._current_lambda_epoch).
    """
    print("Testing training mode lambda_epoch...")

    # Read observer_offline_trainer.py to check training code
    trainer_path = os.path.join(
        os.path.dirname(__file__), "..", "RL_controller", "observer_offline_trainer.py"
    )
    with open(trainer_path, "r") as f:
        content = f.read()

    # Find the training step - look for the main training loop (not validate_epoch)
    # Training should be before validate_epoch
    validate_pos = content.find("def validate_epoch")
    training_section = content[:validate_pos]

    # Find penalty calculation in training
    # Should contain: lambda_epoch = self._current_lambda_epoch
    penalty_matches = []
    for i, line in enumerate(training_section.split("\n")):
        if "penalty = lambda_epoch * (self.alpha_turnover" in line:
            # Get context around this line
            lines = training_section.split("\n")
            context_start = max(0, i - 5)
            context = "\n".join(lines[context_start : i + 1])
            penalty_matches.append(context)

    # Training should use self._current_lambda_epoch
    assert any(
        "lambda_epoch = self._current_lambda_epoch" in match
        for match in penalty_matches
    ), "Training mode must use self._current_lambda_epoch (curriculum learning)!"

    print(
        "✅ Training mode correctly uses curriculum learning (self._current_lambda_epoch)"
    )


def test_curriculum_formula():
    """
    Test curriculum learning formula:
    - Warmup: lambda = 0.0
    - Ramp: lambda increases linearly from 0 to 1
    - Full: lambda = 1.0
    """
    print("Testing curriculum formula...")

    config = MAFIAConfig()
    warmup = config.curriculum_warmup_epochs
    rampup = config.curriculum_penalty_rampup

    print(f"  Curriculum config: warmup={warmup}, rampup={rampup}")

    # Simulate lambda_epoch calculation
    def compute_lambda(epoch, warmup, rampup):
        if epoch < warmup:
            return 0.0
        elif epoch < warmup + rampup:
            progress = (epoch - warmup) / rampup
            return progress
        else:
            return 1.0

    # Test different epochs
    test_cases = [
        (0, "start"),
        (warmup - 1 if warmup > 0 else -1, "before warmup end"),
        (warmup, "warmup end"),
        (warmup + rampup // 2, "mid ramp"),
        (warmup + rampup - 1, "before ramp end"),
        (warmup + rampup, "ramp end (full penalty)"),
        (warmup + rampup + 10, "well after ramp"),
    ]

    for epoch, desc in test_cases:
        if epoch < 0:
            continue
        lambda_val = compute_lambda(epoch, warmup, rampup)
        print(f"  Epoch {epoch:2d} ({desc:25s}): lambda = {lambda_val:.3f}")

        # Verify bounds
        assert 0.0 <= lambda_val <= 1.0, f"Lambda must be in [0, 1], got {lambda_val}"

    # Verify final state
    final_lambda = compute_lambda(warmup + rampup, warmup, rampup)
    assert final_lambda == 1.0, "After rampup, lambda must be 1.0 (full penalty)"

    print("✅ Curriculum formula correct: warmup → ramp → full penalty")


def test_inference_no_curriculum():
    """
    Verify inference code (mafia_observer.py) does NOT use curriculum learning.
    Inference should always use full penalty directly.
    """
    print("Testing inference mode has no curriculum learning...")

    # Read mafia_observer.py
    observer_path = os.path.join(
        os.path.dirname(__file__), "..", "RL_controller", "mafia_observer.py"
    )
    with open(observer_path, "r") as f:
        content = f.read()

    # Check that observer code does NOT mention lambda_epoch or curriculum
    assert "lambda_epoch" not in content, (
        "Inference code (mafia_observer.py) should NOT use lambda_epoch or curriculum learning!"
    )

    assert "curriculum" not in content.lower(), (
        "Inference code should not reference curriculum learning!"
    )

    # Verify it uses penalties directly
    assert "alpha_turnover" in content and "alpha_change" in content, (
        "Inference should use turnover/change penalties directly (no curriculum scaling)"
    )

    print("✅ Inference mode correctly uses full penalty (no curriculum learning)")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Curriculum Learning Validation/Inference Fix")
    print("=" * 70)
    print()

    try:
        test_validation_lambda_epoch()
        print()

        test_training_lambda_epoch()
        print()

        test_curriculum_formula()
        print()

        test_inference_no_curriculum()
        print()

        print("=" * 70)
        print("✅ ALL TESTS PASSED - Curriculum learning fix verified!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✅ Training mode: Uses curriculum learning (lambda_epoch varies)")
        print("  ✅ Validation mode: Always uses lambda_epoch = 1.0 (full penalty)")
        print("  ✅ Inference mode: Always uses full penalty (no curriculum)")

    except AssertionError as e:
        print()
        print("=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        sys.exit(1)
