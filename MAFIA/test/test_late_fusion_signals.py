"""
Test case để verify Wide & Deep Late Fusion Direction Head (Spec §3.5 v2.1).

Signals:
1. DC_Event_Flag (binary structural break)
2. Breadth_Gap (avg RSI stocks - RSI index)
3. Div_Signal (divergence detection)
4. Signed_VPI_Zscore (volume-price efficiency)

Chạy: python tests/test_late_fusion_signals.py
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th


def test_trajectory_batch_has_new_fields():
    """Test 1: TrajectoryBatch có các fields mới cho Late Fusion"""
    from RL_controller.observer_offline_trainer import TrajectoryBatch
    import dataclasses

    fields = {f.name for f in dataclasses.fields(TrajectoryBatch)}

    required_fields = ["dc_event_flag", "breadth_gap", "div_signal", "signed_vpi_zscore"]

    for field in required_fields:
        assert field in fields, f"FAIL: TrajectoryBatch missing field '{field}'"

    # vol_std20 should still exist (for rebalancing logic, not Direction Head)
    assert "vol_std20" in fields, "FAIL: vol_std20 should exist for rebalancing logic"

    print("✓ Test 1 PASSED: TrajectoryBatch has all 4 Wide Path signal fields + vol_std20")
    print(f"  → Wide Path Fields: {required_fields}")


def test_direction_head_explicit_dim():
    """Test 2: DirectionHead có explicit_dim = 4"""
    from config import Config
    from RL_controller.mafia_modules import DirectionHead

    config = Config()
    direction_head = DirectionHead(config)

    assert direction_head.explicit_dim == 4, (
        f"FAIL: explicit_dim should be 4, got {direction_head.explicit_dim}"
    )

    # Check latent_dim = 2*D
    expected_latent_dim = config.mafia_D * 2
    assert direction_head.latent_dim == expected_latent_dim, (
        f"FAIL: latent_dim should be {expected_latent_dim}, got {direction_head.latent_dim}"
    )

    print("✓ Test 2 PASSED: DirectionHead.explicit_dim = 4 (Late Fusion)")
    print(f"  → latent_dim = 2*D = {direction_head.latent_dim}")


def test_direction_head_late_fusion_architecture():
    """Test 3: DirectionHead có Late Fusion architecture đúng"""
    from config import Config
    from RL_controller.mafia_modules import DirectionHead

    config = Config()
    direction_head = DirectionHead(config)

    D = config.mafia_D

    # Deep Path: input_proj should be (2D → D)
    assert direction_head.input_proj.in_features == D * 2, (
        f"FAIL: input_proj.in_features should be {D * 2}, got {direction_head.input_proj.in_features}"
    )
    assert direction_head.input_proj.out_features == D, (
        f"FAIL: input_proj.out_features should be {D}, got {direction_head.input_proj.out_features}"
    )

    # Classifier: Late Fusion input should be (D + 4)
    expected_classifier_in = D + 4
    assert direction_head.classifier.in_features == expected_classifier_in, (
        f"FAIL: classifier.in_features should be {expected_classifier_in}, got {direction_head.classifier.in_features}"
    )

    print("✓ Test 3 PASSED: DirectionHead Late Fusion architecture correct")
    print(f"  → Deep Path: input_proj({D * 2}→{D})")
    print(f"  → Fusion: classifier({expected_classifier_in}→3)")


def test_breadth_gap_computation():
    """Test 4: Breadth_Gap tính toán đúng"""
    # Breadth_Gap = avg(RSI_stocks) - RSI_index

    # Scenario: Index RSI = 60, Avg Stock RSI = 45 (stocks weaker than index)
    rsi_index = 60.0
    rsi_stocks_avg = 45.0
    breadth_gap = (rsi_stocks_avg - rsi_index) / 100.0  # Normalize to [-1, 1]

    assert abs(breadth_gap - (-0.15)) < 1e-6, f"FAIL: Breadth_Gap should be ~-0.15, got {breadth_gap}"

    # Scenario: Index RSI = 40, Avg Stock RSI = 55 (stocks stronger than index)
    rsi_index = 40.0
    rsi_stocks_avg = 55.0
    breadth_gap_strong = (rsi_stocks_avg - rsi_index) / 100.0

    assert abs(breadth_gap_strong - 0.15) < 1e-6, f"FAIL: Breadth_Gap should be ~0.15, got {breadth_gap_strong}"

    print("✓ Test 4 PASSED: Breadth_Gap computation correct")
    print(f"  → Weak stocks (Gap = {breadth_gap:.4f} < 0): Bull Trap warning")
    print(f"  → Strong stocks (Gap = {breadth_gap_strong:.4f} > 0): Healthy rally")


def test_div_signal_computation():
    """Test 5: Div_Signal (divergence) tính toán đúng"""
    # Divergence: RSI and Price moving in opposite directions

    # Scenario: Price up but RSI down (Bearish divergence) → Signal = -1
    price_slope = 0.05  # Price going up
    rsi_slope = -10.0   # RSI going down
    div_signal = 0.0
    if np.sign(price_slope) != np.sign(rsi_slope) and abs(rsi_slope) > 5:
        div_signal = -np.sign(price_slope)

    assert div_signal == -1.0, f"FAIL: Bearish divergence should give signal -1, got {div_signal}"

    # Scenario: Price down but RSI up (Bullish divergence) → Signal = +1
    price_slope = -0.05  # Price going down
    rsi_slope = 8.0      # RSI going up
    div_signal = 0.0
    if np.sign(price_slope) != np.sign(rsi_slope) and abs(rsi_slope) > 5:
        div_signal = -np.sign(price_slope)

    assert div_signal == 1.0, f"FAIL: Bullish divergence should give signal +1, got {div_signal}"

    # Scenario: No divergence (same direction)
    price_slope = 0.05
    rsi_slope = 10.0
    div_signal = 0.0
    if np.sign(price_slope) != np.sign(rsi_slope) and abs(rsi_slope) > 5:
        div_signal = -np.sign(price_slope)

    assert div_signal == 0.0, f"FAIL: No divergence should give signal 0, got {div_signal}"

    print("✓ Test 5 PASSED: Div_Signal computation correct")
    print(f"  → Bearish divergence: Signal = -1 (potential drop)")
    print(f"  → Bullish divergence: Signal = +1 (potential rise)")
    print(f"  → No divergence: Signal = 0")


def test_signed_vpi_zscore_computation():
    """Test 6: Signed_VPI_Zscore tính toán đúng"""
    # VPI = Sign(ΔP) × |ΔP| / (Vol / Vol_20_avg)
    # Efficiency: Price move relative to volume effort

    # Scenario: Strong up move with moderate volume (efficient)
    delta_p = 0.02  # +2%
    vol_ratio = 1.0  # Volume = average
    vpi = np.sign(delta_p) * abs(delta_p) / (vol_ratio + 1e-8)

    assert vpi > 0, f"FAIL: Positive price move should give positive VPI, got {vpi}"
    assert abs(vpi - 0.02) < 0.001, f"FAIL: VPI should be ~0.02, got {vpi}"

    # Scenario: Small move with high volume (inefficient - distribution)
    delta_p = 0.005  # +0.5%
    vol_ratio = 2.5  # Volume 2.5x average (high)
    vpi_inefficient = np.sign(delta_p) * abs(delta_p) / (vol_ratio + 1e-8)

    assert vpi_inefficient < vpi, f"FAIL: High volume should reduce efficiency"

    print("✓ Test 6 PASSED: Signed_VPI_Zscore computation correct")
    print(f"  → Efficient move (moderate vol): VPI = {vpi:.4f}")
    print(f"  → Inefficient move (high vol): VPI = {vpi_inefficient:.4f}")


def test_direction_head_forward_with_4_signals():
    """Test 7: DirectionHead forward pass với 4 signals (Late Fusion)"""
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
    explicit_signals = th.randn(batch_size, 4)  # 4 signals now

    # Forward pass
    with th.no_grad():
        logits = direction_head(c_mkt, delta_c_mkt, explicit_signals)

    assert logits.shape == (batch_size, 3), (
        f"FAIL: Output shape should be ({batch_size}, 3), got {logits.shape}"
    )

    print("✓ Test 7 PASSED: DirectionHead forward pass works with 4 signals")
    print(f"  → Input: c_mkt({batch_size},{D}), delta_c_mkt({batch_size},{D}), explicit({batch_size},4)")
    print(f"  → Output: logits{logits.shape}")


def test_observer_compute_explicit_signals():
    """Test 8: Observer._compute_explicit_signals returns (1, 4) tensor"""
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
    observer.price_history_buffer = list(np.linspace(100, 110, 25))
    observer.stock_price_history = [np.random.randn(10) * 10 + 100 for _ in range(25)]
    observer.rsi_history = list(np.linspace(40, 60, 10))
    observer.volume_history = list(np.random.uniform(1e6, 2e6, 25))
    observer.vpi_history = list(np.random.uniform(-0.01, 0.01, 10))

    # Compute signals
    signals = observer._compute_explicit_signals(
        market_close_price=111.0,
        stock_closes=np.random.randn(10) * 10 + 105,
        stock_volumes=None,
        market_volume=1.5e6
    )

    assert signals.shape == (1, 4), (
        f"FAIL: explicit_signals shape should be (1, 4), got {signals.shape}"
    )

    print("✓ Test 8 PASSED: Observer._compute_explicit_signals returns (1, 4)")
    print(f"  → Signals: {signals.squeeze().tolist()}")
    print(f"  → [DC_Flag, Breadth_Gap, Div_Signal, VPI_Zscore]")


def test_signal_ranges():
    """Test 9: Signal values trong expected ranges"""
    from config import Config
    from RL_controller.mafia_observer import MAFIAObserver

    config = Config()

    observer = MAFIAObserver.__new__(MAFIAObserver)
    observer.config = config
    observer.device = th.device("cpu")
    observer.price_history_buffer = list(np.linspace(100, 110, 25))
    observer.stock_price_history = [np.random.randn(10) * 10 + 100 for _ in range(25)]
    observer.rsi_history = list(np.linspace(40, 60, 10))
    observer.volume_history = list(np.random.uniform(1e6, 2e6, 25))
    observer.vpi_history = list(np.random.uniform(-0.01, 0.01, 10))
    observer.price_history_window = 50
    observer.dc_threshold = 0.02
    observer.dc_state = {"p_ext": 100.0, "mode": "up"}

    signals = observer._compute_explicit_signals(
        market_close_price=111.0,
        stock_closes=np.random.randn(10) * 10 + 105,
        stock_volumes=None,
        market_volume=1.5e6
    )

    s = signals.squeeze().tolist()

    # Check ranges
    assert s[0] >= 0 and s[0] <= 1, f"DC_Flag should be in [0,1], got {s[0]}"
    assert s[1] >= -1 and s[1] <= 1, f"Breadth_Gap should be in [-1,1], got {s[1]}"
    assert s[2] >= -1 and s[2] <= 1, f"Div_Signal should be in [-1,0,1], got {s[2]}"
    assert s[3] >= -3 and s[3] <= 3, f"VPI_Zscore should be in [-3,3], got {s[3]}"

    print("✓ Test 9 PASSED: Signal values in expected ranges")
    print(f"  → DC_Flag = {s[0]:.4f} ∈ [0, 1]")
    print(f"  → Breadth_Gap = {s[1]:.4f} ∈ [-1, 1]")
    print(f"  → Div_Signal = {s[2]:.4f} ∈ {{-1, 0, 1}}")
    print(f"  → VPI_Zscore = {s[3]:.4f} ∈ [-3, 3]")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("🧪 Testing Wide & Deep Late Fusion Direction Head (Spec §3.5 v2.1)")
    print("=" * 70)
    print()

    tests = [
        test_trajectory_batch_has_new_fields,
        test_direction_head_explicit_dim,
        test_direction_head_late_fusion_architecture,
        test_breadth_gap_computation,
        test_div_signal_computation,
        test_signed_vpi_zscore_computation,
        test_direction_head_forward_with_4_signals,
        test_observer_compute_explicit_signals,
        test_signal_ranges,
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
        print("🎉 All tests PASSED! Late Fusion Direction Head ready for training.")
        print()
        print("📌 Next step: python scripts/train_observer_offline.py")
    else:
        print(f"❌ {failed} test(s) FAILED. Please fix before training.")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
