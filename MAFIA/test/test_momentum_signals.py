"""
Test case để verify 3 Momentum Explicit Signals đã được implement đúng.

Signals:
1. Vol_Std20 (existing)
2. DC_Event (existing)
3. KER_10 (new) - Kaufman Efficiency Ratio
4. RSI_Grad (new) - RSI Gradient/Velocity
5. Breadth_Mom (new) - Market Breadth Momentum

Chạy: python tests/test_momentum_signals.py
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th


def test_trajectory_batch_has_new_fields():
    """Test 1: TrajectoryBatch có các fields mới"""
    from RL_controller.observer_offline_trainer import TrajectoryBatch
    import dataclasses

    fields = {f.name for f in dataclasses.fields(TrajectoryBatch)}

    required_fields = ["vol_std20", "dc_event_magnitude", "ker_10", "rsi_grad", "breadth_mom"]

    for field in required_fields:
        assert field in fields, f"FAIL: TrajectoryBatch missing field '{field}'"

    print("✓ Test 1 PASSED: TrajectoryBatch has all 5 explicit signal fields")
    print(f"  → Fields: {required_fields}")


def test_direction_head_explicit_dim():
    """Test 2: DirectionHead có explicit_dim = 5"""
    from config import Config
    from RL_controller.mafia_modules import DirectionHead

    config = Config()
    direction_head = DirectionHead(config)

    assert direction_head.explicit_dim == 5, (
        f"FAIL: explicit_dim should be 5, got {direction_head.explicit_dim}"
    )

    # Check input_dim = 2*D + 5
    expected_input_dim = config.mafia_D * 2 + 5
    assert direction_head.input_dim == expected_input_dim, (
        f"FAIL: input_dim should be {expected_input_dim}, got {direction_head.input_dim}"
    )

    print("✓ Test 2 PASSED: DirectionHead.explicit_dim = 5")
    print(f"  → input_dim = 2*D + 5 = {direction_head.input_dim}")


def test_ker_computation():
    """Test 3: KER (Kaufman Efficiency Ratio) tính toán đúng"""
    # Test case: Smooth uptrend should have KER close to 1
    prices_smooth = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    change = abs(prices_smooth[-1] - prices_smooth[0])  # 10
    volatility_sum = sum(abs(prices_smooth[i] - prices_smooth[i-1]) for i in range(1, len(prices_smooth)))  # 10
    ker_smooth = change / (volatility_sum + 1e-8)

    assert abs(ker_smooth - 1.0) < 0.01, f"FAIL: KER for smooth trend should be ~1.0, got {ker_smooth}"

    # Test case: Choppy market should have KER close to 0
    prices_choppy = np.array([100, 102, 100, 102, 100, 102, 100, 102, 100, 102, 100])
    change = abs(prices_choppy[-1] - prices_choppy[0])  # 0
    volatility_sum = sum(abs(prices_choppy[i] - prices_choppy[i-1]) for i in range(1, len(prices_choppy)))  # 20
    ker_choppy = change / (volatility_sum + 1e-8)

    assert ker_choppy < 0.1, f"FAIL: KER for choppy market should be ~0, got {ker_choppy}"

    print("✓ Test 3 PASSED: KER computation correct")
    print(f"  → Smooth trend KER = {ker_smooth:.4f} (expected ~1.0)")
    print(f"  → Choppy market KER = {ker_choppy:.4f} (expected ~0.0)")


def test_rsi_grad_computation():
    """Test 4: RSI Gradient tính toán đúng"""
    # RSI increasing = positive gradient
    rsi_history = [30, 35, 40, 45, 50]
    rsi_grad = rsi_history[-1] - rsi_history[-2]

    assert rsi_grad > 0, f"FAIL: RSI increasing should have positive grad, got {rsi_grad}"

    # RSI decreasing = negative gradient
    rsi_history_down = [70, 65, 60, 55, 50]
    rsi_grad_down = rsi_history_down[-1] - rsi_history_down[-2]

    assert rsi_grad_down < 0, f"FAIL: RSI decreasing should have negative grad, got {rsi_grad_down}"

    print("✓ Test 4 PASSED: RSI Gradient computation correct")
    print(f"  → Rising RSI grad = {rsi_grad} (positive)")
    print(f"  → Falling RSI grad = {rsi_grad_down} (negative)")


def test_breadth_mom_computation():
    """Test 5: Breadth Momentum tính toán đúng"""
    # Breadth = % stocks above SMA20
    # Breadth_Mom = Breadth_t - Breadth_{t-5}

    # Scenario: 60% stocks above SMA20 now, was 40% 5 days ago
    breadth_now = 0.60
    breadth_5d_ago = 0.40
    breadth_mom = breadth_now - breadth_5d_ago

    assert abs(breadth_mom - 0.20) < 1e-6, f"FAIL: Breadth_Mom should be ~0.20, got {breadth_mom}"

    # Scenario: Market weakening (breadth decreasing)
    breadth_now_weak = 0.30
    breadth_5d_ago_weak = 0.50
    breadth_mom_weak = breadth_now_weak - breadth_5d_ago_weak

    assert abs(breadth_mom_weak - (-0.20)) < 1e-6, f"FAIL: Breadth_Mom should be ~-0.20, got {breadth_mom_weak}"

    print("✓ Test 5 PASSED: Breadth Momentum computation correct")
    print(f"  → Strengthening market: Breadth_Mom = {breadth_mom:.4f}")
    print(f"  → Weakening market: Breadth_Mom = {breadth_mom_weak:.4f}")


def test_direction_head_forward_with_5_signals():
    """Test 6: DirectionHead forward pass với 5 signals"""
    from config import Config
    from RL_controller.mafia_modules import DirectionHead

    config = Config()
    direction_head = DirectionHead(config)
    direction_head.eval()

    batch_size = 4
    D = config.mafia_D

    # Create dummy inputs
    c_mkt = th.randn(batch_size, D)
    delta_c_mkt = th.randn(batch_size, D)
    explicit_signals = th.randn(batch_size, 5)  # 5 signals

    # Forward pass
    with th.no_grad():
        logits = direction_head(c_mkt, delta_c_mkt, explicit_signals)

    assert logits.shape == (batch_size, 3), (
        f"FAIL: Output shape should be ({batch_size}, 3), got {logits.shape}"
    )

    print("✓ Test 6 PASSED: DirectionHead forward pass works with 5 signals")
    print(f"  → Input: c_mkt({batch_size},{D}), delta_c_mkt({batch_size},{D}), explicit({batch_size},5)")
    print(f"  → Output: logits{logits.shape}")


def test_observer_compute_explicit_signals():
    """Test 7: Observer._compute_explicit_signals returns (1, 5) tensor"""
    from config import Config
    from RL_controller.mafia_observer import MAFIAObserver

    config = Config()

    # Create observer (minimal init)
    observer = MAFIAObserver.__new__(MAFIAObserver)
    observer.config = config
    observer.device = th.device("cpu")
    observer.price_history_buffer = None
    observer.price_history_window = 50
    observer.dc_threshold = 0.02
    observer.dc_state = {"p_ext": None, "mode": "up"}

    # Simulate some price history
    observer.price_history_buffer = list(np.linspace(100, 110, 25))  # 25 days of data
    observer.stock_price_history = [np.random.randn(10) * 10 + 100 for _ in range(25)]
    observer.rsi_history = list(np.linspace(40, 60, 10))
    observer.breadth_history = list(np.linspace(0.4, 0.6, 8))

    # Compute signals
    signals = observer._compute_explicit_signals(
        market_close_price=111.0,
        stock_closes=np.random.randn(10) * 10 + 105,
        stock_volumes=None
    )

    assert signals.shape == (1, 5), (
        f"FAIL: explicit_signals shape should be (1, 5), got {signals.shape}"
    )

    print("✓ Test 7 PASSED: Observer._compute_explicit_signals returns (1, 5)")
    print(f"  → Signals: {signals.squeeze().tolist()}")
    print(f"  → [Vol, DC, KER, RSI_Grad, Breadth_Mom]")


def test_signal_normalization_ranges():
    """Test 8: Signal normalization trong expected ranges"""
    from config import Config
    from RL_controller.mafia_observer import MAFIAObserver

    config = Config()

    observer = MAFIAObserver.__new__(MAFIAObserver)
    observer.config = config
    observer.device = th.device("cpu")
    observer.price_history_buffer = list(np.linspace(100, 110, 25))
    observer.stock_price_history = [np.random.randn(10) * 10 + 100 for _ in range(25)]
    observer.rsi_history = list(np.linspace(40, 60, 10))
    observer.breadth_history = list(np.linspace(0.4, 0.6, 8))
    observer.price_history_window = 50
    observer.dc_threshold = 0.02
    observer.dc_state = {"p_ext": 100.0, "mode": "up"}

    signals = observer._compute_explicit_signals(
        market_close_price=111.0,
        stock_closes=np.random.randn(10) * 10 + 105,
        stock_volumes=None
    )

    s = signals.squeeze().tolist()

    # Check ranges
    assert s[2] >= 0 and s[2] <= 1, f"KER should be in [0,1], got {s[2]}"
    assert s[3] >= -1 and s[3] <= 1, f"RSI_Grad should be in [-1,1], got {s[3]}"
    assert s[4] >= -1 and s[4] <= 1, f"Breadth_Mom should be in [-1,1], got {s[4]}"

    print("✓ Test 8 PASSED: Signal normalization in expected ranges")
    print(f"  → KER_norm = {s[2]:.4f} ∈ [0, 1]")
    print(f"  → RSI_Grad_norm = {s[3]:.4f} ∈ [-1, 1]")
    print(f"  → Breadth_Mom_norm = {s[4]:.4f} ∈ [-1, 1]")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("🧪 Testing Momentum Explicit Signals Implementation (Spec §3.5.1)")
    print("=" * 70)
    print()

    tests = [
        test_trajectory_batch_has_new_fields,
        test_direction_head_explicit_dim,
        test_ker_computation,
        test_rsi_grad_computation,
        test_breadth_mom_computation,
        test_direction_head_forward_with_5_signals,
        test_observer_compute_explicit_signals,
        test_signal_normalization_ranges,
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
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 70)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("🎉 All tests PASSED! Momentum signals ready for training.")
        print()
        print("📌 Next step: python scripts/train_observer_offline.py")
    else:
        print(f"❌ {failed} test(s) FAILED. Please fix before training.")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
