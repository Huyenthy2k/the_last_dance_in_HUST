"""
Test Portfolio Allocator Reward Implementation
Validates the new 2-component reward function
"""

import numpy as np
import sys

sys.path.insert(0, "/Users/nguyensiry/Documents/the_last_dance/agents/MAFIA")

from RL_controller.portfolio_allocator_reward import (
    compute_log_return,
    compute_jensen_shannon_divergence,
    compute_reward,
    RewardNormalizer,
)


def test_log_return():
    """Test log return calculation."""
    print("\n" + "=" * 60)
    print("TEST: Log Return Calculation")
    print("=" * 60)

    # Normal case: +2% return
    lr = compute_log_return(0.02)
    print(f"Portfolio return +2%: log_return = {lr:.6f}")
    assert abs(lr - np.log(1.02)) < 1e-8, "Log return calculation failed"

    # Negative case: -1% return
    lr = compute_log_return(-0.01)
    print(f"Portfolio return -1%: log_return = {lr:.6f}")
    assert abs(lr - np.log(0.99)) < 1e-8, "Log return calculation failed"

    # Edge case: -100% return
    lr = compute_log_return(-1.0)
    print(f"Portfolio return -100%: log_return = {lr:.6f}")
    assert lr == -10.0, "Should return large negative value"

    # Edge case: < -100% return
    lr = compute_log_return(-1.5)
    print(f"Portfolio return -150%: log_return = {lr:.6f}")
    assert lr == -10.0, "Should return large negative value"

    print("✓ All log return tests passed!")


def test_jensen_shannon_divergence():
    """Test JS divergence calculation."""
    print("\n" + "=" * 60)
    print("TEST: Jensen-Shannon Divergence")
    print("=" * 60)

    # Identical distributions: JS = 0
    a1 = np.array([0.5, 0.3, 0.2])
    a2 = np.array([0.5, 0.3, 0.2])
    js = compute_jensen_shannon_divergence(a1, a2)
    print(f"Same distribution: JS = {js:.8f}")
    assert js < 1e-6, "JS divergence should be ~0 for identical distributions"

    # Different distributions
    a1 = np.array([0.5, 0.3, 0.2])
    a2 = np.array([0.3, 0.4, 0.3])
    js = compute_jensen_shannon_divergence(a1, a2)
    print(f"Different distributions: JS = {js:.8f}")
    assert 0 < js <= 1, "JS divergence should be in [0, 1]"

    # Opposite distributions
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.0, 0.0, 1.0])
    js = compute_jensen_shannon_divergence(a1, a2)
    print(f"Opposite distributions: JS = {js:.8f}")
    assert 0 < js <= 1, "JS divergence should be in [0, 1]"

    print("✓ All JS divergence tests passed!")


def test_reward_computation():
    """Test reward computation with 2-component formula."""
    print("\n" + "=" * 60)
    print("TEST: Reward Computation (2-component)")
    print("=" * 60)

    # Case 1: Good return, no divergence
    a_alloc = np.array([0.5, 0.3, 0.2])
    a_final = np.array([0.5, 0.3, 0.2])  # No change
    reward_dict = compute_reward(
        portfolio_return=0.02,
        a_alloc=a_alloc,
        a_final=a_final,
        w_return=1.0,
        lambda_js=0.1,
    )

    print(f"\nCase 1: +2% return, no divergence")
    print(f"  log_return = {reward_dict['log_return']:.6f}")
    print(f"  r_return = {reward_dict['r_return']:.6f}")
    print(f"  js_divergence = {reward_dict['js_divergence']:.6f}")
    print(f"  r_divergence = {reward_dict['r_divergence']:.6f}")
    print(f"  reward_total = {reward_dict['reward_total']:.6f}")

    # Should be ~log(1.02) = 0.0198
    assert reward_dict["r_return"] > 0, "Should have positive return component"
    # When divergence = 0, r_divergence = -0 (acceptable)
    assert reward_dict["r_divergence"] <= 0, "Divergence penalty should be ≤ 0"

    # Case 2: Good return, with divergence
    a_alloc = np.array([0.6, 0.3, 0.1])
    a_final = np.array([0.3, 0.4, 0.3])  # Controller adjusted significantly
    reward_dict = compute_reward(
        portfolio_return=0.02,
        a_alloc=a_alloc,
        a_final=a_final,
        w_return=1.0,
        lambda_js=0.1,
    )

    print(f"\nCase 2: +2% return, with divergence")
    print(f"  log_return = {reward_dict['log_return']:.6f}")
    print(f"  r_return = {reward_dict['r_return']:.6f}")
    print(f"  js_divergence = {reward_dict['js_divergence']:.6f}")
    print(f"  r_divergence = {reward_dict['r_divergence']:.6f}")
    print(f"  reward_total = {reward_dict['reward_total']:.6f}")

    # Should have larger divergence penalty
    assert reward_dict["js_divergence"] > 0, "Should have divergence"
    assert reward_dict["r_divergence"] < 0, "Divergence penalty should be negative"

    # Case 3: Negative return
    a_alloc = np.array([0.5, 0.3, 0.2])
    a_final = np.array([0.5, 0.3, 0.2])
    reward_dict = compute_reward(
        portfolio_return=-0.01,
        a_alloc=a_alloc,
        a_final=a_final,
        w_return=1.0,
        lambda_js=0.1,
    )

    print(f"\nCase 3: -1% return, no divergence")
    print(f"  log_return = {reward_dict['log_return']:.6f}")
    print(f"  r_return = {reward_dict['r_return']:.6f}")
    print(f"  js_divergence = {reward_dict['js_divergence']:.6f}")
    print(f"  r_divergence = {reward_dict['r_divergence']:.6f}")
    print(f"  reward_total = {reward_dict['reward_total']:.6f}")

    # Should have negative return component
    assert reward_dict["r_return"] < 0, "Should have negative return component"

    print("\n✓ All reward computation tests passed!")


def test_reward_normalizer():
    """Test running reward normalizer."""
    print("\n" + "=" * 60)
    print("TEST: Reward Normalizer (EMA)")
    print("=" * 60)

    norm = RewardNormalizer(alpha=0.1)

    # Update with some values
    values = [0.01, 0.02, -0.01, 0.015, -0.005, 0.01, 0.02]
    print(f"\nUpdating normalizer with values: {values}")

    for val in values:
        norm.update(val)

    stats = norm.get_stats()
    print(f"\nNormalizer stats after {len(values)} updates:")
    print(f"  mean = {stats['mean']:.6f}")
    print(f"  std = {stats['std']:.6f}")
    print(f"  count = {stats['count']}")

    # Normalize a new value
    new_value = 0.015
    normalized = norm.normalize(new_value)
    print(f"\nNormalizing value {new_value}: normalized = {normalized:.6f}")

    assert stats["count"] == len(values), "Count should match updates"
    assert (
        abs(normalized - (new_value - stats["mean"]) / (stats["std"] + 1e-8)) < 1e-6
    ), "Normalization failed"

    print("✓ Reward normalizer tests passed!")


def test_formula_verification():
    """Verify formula matches spec exactly."""
    print("\n" + "=" * 60)
    print("TEST: Formula Verification Against Spec")
    print("=" * 60)

    # According to spec: r_t = w_return · r_return - λ_js · D_JS(a_alloc || a_final)
    portfolio_return = 0.02
    a_alloc = np.array([0.6, 0.3, 0.1])
    a_final = np.array([0.4, 0.4, 0.2])
    w_return = 1.0
    lambda_js = 0.1

    result = compute_reward(
        portfolio_return=portfolio_return,
        a_alloc=a_alloc,
        a_final=a_final,
        w_return=w_return,
        lambda_js=lambda_js,
    )

    # Manually compute to verify
    expected_log_return = np.log(1.0 + portfolio_return)
    expected_r_return = w_return * expected_log_return

    # JS divergence via manual computation
    a_alloc_norm = a_alloc / np.sum(a_alloc)
    a_final_norm = a_final / np.sum(a_final)
    m = 0.5 * (a_alloc_norm + a_final_norm)
    from scipy.special import rel_entr

    kl_1 = np.sum(rel_entr(a_alloc_norm, m))
    kl_2 = np.sum(rel_entr(a_final_norm, m))
    expected_js_div = 0.5 * kl_1 + 0.5 * kl_2
    expected_r_divergence = -lambda_js * expected_js_div
    expected_reward = expected_r_return + expected_r_divergence

    print(f"\nManual verification:")
    print(f"  Expected r_return = {expected_r_return:.8f}")
    print(f"  Computed r_return = {result['r_return']:.8f}")
    print(f"  Match: {abs(result['r_return'] - expected_r_return) < 1e-8}")

    print(f"\n  Expected js_div = {expected_js_div:.8f}")
    print(f"  Computed js_div = {result['js_divergence']:.8f}")
    print(f"  Match: {abs(result['js_divergence'] - expected_js_div) < 1e-6}")

    print(f"\n  Expected reward = {expected_reward:.8f}")
    print(f"  Computed reward = {result['reward_total']:.8f}")
    print(f"  Match: {abs(result['reward_total'] - expected_reward) < 1e-6}")

    assert abs(result["r_return"] - expected_r_return) < 1e-8, "r_return mismatch"
    assert abs(result["js_divergence"] - expected_js_div) < 1e-6, (
        "js_divergence mismatch"
    )
    assert abs(result["reward_total"] - expected_reward) < 1e-6, "reward mismatch"

    print("\n✓ Formula verification passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PORTFOLIO ALLOCATOR REWARD TESTS")
    print("=" * 60)

    try:
        test_log_return()
        test_jensen_shannon_divergence()
        test_reward_computation()
        test_reward_normalizer()
        test_formula_verification()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
