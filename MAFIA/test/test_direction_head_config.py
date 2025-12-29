"""
Test case để verify DirectionHead đã được configure đúng theo Spec.

Chạy: python tests/test_direction_head_config.py
"""

import sys
import os
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th
import torch.nn.functional as F


def test_config_class_balanced_sampling():
    """Test 1: Verify class_balanced_sampling = False"""
    from config import Config

    config = Config()

    assert config.mafia_use_class_balanced_sampling == False, (
        f"FAIL: mafia_use_class_balanced_sampling should be False, "
        f"got {config.mafia_use_class_balanced_sampling}"
    )
    print("✓ Test 1 PASSED: class_balanced_sampling = False")


def test_config_focal_alpha():
    """Test 2: Verify focal loss alpha matches VNINDEX distribution"""
    from config import Config

    config = Config()
    # Based on VNINDEX (2015-2025): Bear=26.5%, Side=28.8%, Bull=44.8%
    expected_alpha = [1.69, 1.56, 1.00]

    assert config.mafia_focal_alpha == expected_alpha, (
        f"FAIL: mafia_focal_alpha should be {expected_alpha}, "
        f"got {config.mafia_focal_alpha}"
    )
    print("✓ Test 2 PASSED: focal_alpha = [1.69, 1.56, 1.00]")


def test_direction_head_bias_init():
    """Test 3: Verify DirectionHead bias initialization matches VNINDEX distribution"""
    from config import Config
    from RL_controller.mafia_modules import DirectionHead

    config = Config()

    # Create DirectionHead
    direction_head = DirectionHead(config)

    # Expected bias values (log-priors for Bear=26.5%, Side=28.8%, Bull=44.8%)
    expected_bias = th.tensor([-1.33, -1.25, -0.80])
    actual_bias = direction_head.classifier.bias.data

    # Check values match (with tolerance)
    assert th.allclose(actual_bias, expected_bias, atol=0.01), (
        f"FAIL: Classifier bias should be {expected_bias.tolist()}, "
        f"got {actual_bias.tolist()}"
    )
    print("✓ Test 3 PASSED: DirectionHead bias = [-1.33, -1.25, -0.80]")


def test_initial_softmax_distribution():
    """Test 4: Verify initial softmax output matches VNINDEX distribution"""
    from config import Config
    from RL_controller.mafia_modules import DirectionHead

    config = Config()
    direction_head = DirectionHead(config)

    # Get bias as logits and apply softmax
    bias = direction_head.classifier.bias.data
    initial_probs = F.softmax(bias, dim=0)

    # Expected distribution: Bear=26.5%, Side=28.8%, Bull=44.8%
    expected_probs = th.tensor([0.265, 0.288, 0.448])

    # Check with 2% tolerance
    assert th.allclose(initial_probs, expected_probs, atol=0.02), (
        f"FAIL: Initial softmax should be ~{expected_probs.tolist()}, "
        f"got {initial_probs.tolist()}"
    )

    print(f"✓ Test 4 PASSED: Initial softmax = {initial_probs.tolist()}")
    print(f"  → Bear={initial_probs[0]:.1%}, Side={initial_probs[1]:.1%}, Bull={initial_probs[2]:.1%}")


def test_no_double_correction():
    """Test 5: Verify không có double correction (sampling OFF + focal ON)"""
    from config import Config

    config = Config()

    # Should use focal loss alpha (not uniform)
    focal_not_uniform = config.mafia_focal_alpha != [1.0, 1.0, 1.0]

    # Should NOT use class balanced sampling
    sampling_off = config.mafia_use_class_balanced_sampling == False

    assert focal_not_uniform and sampling_off, (
        f"FAIL: Expected focal_alpha != [1,1,1] AND sampling=False. "
        f"Got focal={config.mafia_focal_alpha}, sampling={config.mafia_use_class_balanced_sampling}"
    )
    print("✓ Test 5 PASSED: No double correction (Focal Loss handles imbalance)")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Testing DirectionHead Configuration (Spec Compliance)")
    print("=" * 60)
    print()

    tests = [
        test_config_class_balanced_sampling,
        test_config_focal_alpha,
        test_direction_head_bias_init,
        test_initial_softmax_distribution,
        test_no_double_correction,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: ERROR - {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("🎉 All tests PASSED! Model configured correctly per Spec.")
    else:
        print(f"❌ {failed} test(s) FAILED. Please check configuration.")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
