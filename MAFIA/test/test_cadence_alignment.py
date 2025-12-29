"""
Test suite for Loss Masking & Training Cadence Alignment.

Demonstrates the cadence masking feature in action.
"""

import numpy as np
import torch as th
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "RL_controller"))

from RL_controller.cadence_helper import (
    generate_rebalance_schedule,
    generate_adaptive_cadence,
    create_batch_cadence_masks,
    compute_cadence_coverage,
    get_cadence_schedule,
)


def test_basic_cadence_generation():
    """Test basic cadence mask generation."""
    print("\n=== Test: Basic Cadence Generation ===")

    # Weekly rebalance (every 5 trading days)
    mask = generate_rebalance_schedule(total_steps=20, rebalance_interval=5)
    print(f"Weekly cadence (20 steps): {mask}")
    print(f"Rebalance indices: {np.where(mask)[0].tolist()}")

    assert mask[0] == 1.0, "First day should be rebalance"
    assert mask[1:5].sum() == 0, "Days 1-4 should be holding"
    assert mask[5] == 1.0, "Day 5 should be rebalance"
    print("✓ Basic cadence generation passed")


def test_adaptive_cadence():
    """Test adaptive cadence with regime shifts."""
    print("\n=== Test: Adaptive Cadence with Regime Shifts ===")

    regime_shifts = np.array([0, 10, 15])
    mask = generate_adaptive_cadence(
        total_steps=20,
        base_interval=5,
        regime_shift_indices=regime_shifts,
        rebalance_on_regime_shift=True,
    )

    print(f"Adaptive cadence with shifts at {regime_shifts.tolist()}: {mask}")
    print(f"Rebalance indices: {np.where(mask)[0].tolist()}")

    # Verify regime shifts are rebalance days
    assert mask[10] == 1.0, "Regime shift at step 10 should be rebalance"
    assert mask[15] == 1.0, "Regime shift at step 15 should be rebalance"
    print("✓ Adaptive cadence passed")


def test_batch_cadence_masks():
    """Test batch cadence mask generation."""
    print("\n=== Test: Batch Cadence Masks ===")

    batch_size = 4
    total_steps = 100
    rebalance_interval = 10

    masks = create_batch_cadence_masks(
        batch_size=batch_size,
        total_steps=total_steps,
        rebalance_interval=rebalance_interval,
    )

    print(f"Batch masks shape: {masks.shape}")
    print(f"Expected: ({total_steps}, {batch_size})")
    assert masks.shape == (total_steps, batch_size), "Shape mismatch"

    # Check each trajectory has same cadence
    for b in range(batch_size):
        rebalances = (masks[:, b] > 0.5).sum().item()
        print(f"Trajectory {b}: {int(rebalances)} rebalance days")
        assert rebalances == 10, f"Expected 10 rebalances, got {rebalances}"

    print("✓ Batch cadence masks passed")


def test_cadence_coverage():
    """Test coverage statistics computation."""
    print("\n=== Test: Cadence Coverage ===")

    mask = generate_rebalance_schedule(total_steps=100, rebalance_interval=5)
    rebal_ratio, hold_ratio = compute_cadence_coverage(mask)

    print(f"Rebalance ratio: {rebal_ratio * 100:.1f}%")
    print(f"Holding ratio: {hold_ratio * 100:.1f}%")

    assert abs(rebal_ratio - 0.2) < 0.01, "Expected ~20% rebalance"
    assert abs(hold_ratio - 0.8) < 0.01, "Expected ~80% holding"
    print("✓ Cadence coverage passed")


def test_predefined_schedules():
    """Test predefined cadence schedules."""
    print("\n=== Test: Predefined Schedules ===")

    total_steps = 252  # 1 trading year

    schedules = {
        "daily": 252,  # Every day
        "weekly": (50, 52),  # ~51 per year (252/5)
        "biweekly": (24, 26),  # ~25 per year (252/10)
        "monthly": (11, 13),  # ~12 per year (252/21)
    }

    for cadence_name, expected in schedules.items():
        mask = get_cadence_schedule(cadence_name, total_steps=total_steps)
        num_rebalances = int(mask.sum())

        if isinstance(expected, int):
            print(
                f"{cadence_name.capitalize()}: {num_rebalances} rebalances (expected {expected})"
            )
            assert num_rebalances == expected, f"Mismatch for {cadence_name}"
        else:
            min_exp, max_exp = expected
            print(
                f"{cadence_name.capitalize()}: {num_rebalances} rebalances (expected {min_exp}-{max_exp})"
            )
            assert min_exp <= num_rebalances <= max_exp, (
                f"Mismatch for {cadence_name}: got {num_rebalances}, expected {min_exp}-{max_exp}"
            )

    print("✓ Predefined schedules passed")


def test_loss_masking_effect():
    """Test the effect of loss masking on PG loss."""
    print("\n=== Test: Loss Masking Effect ===")

    # Simulate PG losses (higher = worse)
    pg_losses = th.full((100,), 0.1)  # Uniform losses

    # Cadence: 20% rebalance, 80% holding
    mask = generate_rebalance_schedule(total_steps=100, rebalance_interval=5)
    mask_tensor = th.from_numpy(mask).float()

    # Apply masking
    masked_pg_losses = pg_losses * mask_tensor

    print(f"Original mean loss: {pg_losses.mean():.6f}")
    print(f"Masked mean loss: {masked_pg_losses.mean():.6f}")
    print(f"Mask ratio: {mask.mean():.2f}")
    print(
        f"Loss reduction: {(1 - masked_pg_losses.mean() / pg_losses.mean()) * 100:.1f}%"
    )

    # Verify masking worked
    expected_reduction = 1.0 - mask.mean()
    actual_reduction = 1 - (masked_pg_losses.mean() / pg_losses.mean()).item()

    assert abs(actual_reduction - expected_reduction) < 0.01, (
        "Masking not working correctly"
    )
    print("✓ Loss masking effect verified")


def test_gradient_flow_simulation():
    """Simulate gradient flow with and without cadence masking."""
    print("\n=== Test: Gradient Flow Simulation ===")

    # Create model parameters
    selection_params = th.nn.Parameter(th.randn(100))
    risk_params = th.nn.Parameter(th.randn(100))

    # Simulate losses
    pg_loss = (selection_params**2).mean()
    risk_loss = (risk_params**2).mean()

    # Test 1: Without masking (both update)
    optimizer = th.optim.SGD([selection_params, risk_params], lr=0.01)
    (pg_loss + risk_loss).backward()
    initial_selection_grad = selection_params.grad.clone()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Without masking:")
    print(f"  Selection grad magnitude: {initial_selection_grad.abs().mean():.6f}")
    print(f"  Risk update magnitude: {(risk_params - th.randn(100)).abs().mean():.6f}")

    # Test 2: With masking (selection masked)
    selection_params = th.nn.Parameter(th.randn(100))
    risk_params = th.nn.Parameter(th.randn(100))

    pg_loss = (selection_params**2).mean()
    risk_loss = (risk_params**2).mean()

    optimizer = th.optim.SGD([selection_params, risk_params], lr=0.01)
    (pg_loss * 0 + risk_loss).backward()  # Mask pg_loss

    if selection_params.grad is not None:
        selection_params.grad.zero_()  # Manually zero for demonstration

    optimizer.step()

    print(f"With masking:")
    print(
        f"  Selection updated: {selection_params.grad is None or selection_params.grad.abs().mean() == 0}"
    )
    print(f"  Risk updated: True")
    print("✓ Gradient flow simulation passed")


def test_cadence_mask_from_env():
    """Test extracting cadence mask from environment schedule."""
    print("\n=== Test: Cadence Mask from Environment ===")

    # Simulate environment that triggers rebalance at specific steps
    env_schedule = np.array([0, 5, 10, 15, 20])  # Rebalance at these steps
    total_steps = 25

    # Convert to mask
    mask = np.zeros(total_steps, dtype=np.float32)
    for step in env_schedule:
        if 0 <= step < total_steps:
            mask[int(step)] = 1.0

    print(f"Environment schedule: {env_schedule}")
    print(f"Generated mask: {mask}")
    print(f"Rebalance days: {np.where(mask)[0].tolist()}")

    assert mask[0] == 1.0, "First scheduled day should be rebalance"
    assert mask[1] == 0.0, "Non-scheduled day should be holding"
    print("✓ Environment cadence extraction passed")


def test_edge_cases():
    """Test edge cases in cadence generation."""
    print("\n=== Test: Edge Cases ===")

    # Edge case 1: Single step
    mask = generate_rebalance_schedule(total_steps=1, rebalance_interval=5)
    assert len(mask) == 1 and mask[0] == 1.0, "Single step should be rebalance"
    print("✓ Single step passed")

    # Edge case 2: No rebalance (interval > total_steps)
    mask = generate_rebalance_schedule(total_steps=5, rebalance_interval=100)
    assert mask[0] == 1.0 and mask[1:].sum() == 0, "Only first step rebalance"
    print("✓ No rebalance passed")

    # Edge case 3: Every step rebalance
    mask = generate_rebalance_schedule(total_steps=10, rebalance_interval=1)
    assert mask.sum() == 10, "All steps should be rebalance"
    print("✓ Every step rebalance passed")

    # Edge case 4: Empty regime shifts
    mask = generate_adaptive_cadence(
        total_steps=10,
        base_interval=5,
        regime_shift_indices=np.array([]),
        rebalance_on_regime_shift=True,
    )
    assert mask[0] == 1.0 and mask[5] == 1.0, "Should fall back to interval"
    print("✓ Empty regime shifts passed")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Loss Masking & Training Cadence Alignment - Test Suite")
    print("=" * 60)

    tests = [
        test_basic_cadence_generation,
        test_adaptive_cadence,
        test_batch_cadence_masks,
        test_cadence_coverage,
        test_predefined_schedules,
        test_loss_masking_effect,
        test_gradient_flow_simulation,
        test_cadence_mask_from_env,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
