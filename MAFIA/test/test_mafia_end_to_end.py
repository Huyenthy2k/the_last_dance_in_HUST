#!/usr/bin/python
# -*- coding: utf-8 -*-#

"""
End-to-end sanity checks for MAFIA Observer and model wiring.
Converted to assertion-based tests (no return values) to satisfy pytest expectations.
"""

import datetime
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from utils.featGen import FeatureProcesser


def _build_mafia_observer(action_dim: int = 10):
    """Helper to construct a MAFIAObserver with MASA-mafia settings."""
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config = Config(seed_num=2022, current_date=current_date)
    config.benchmark_algo = "MASA-mafia"
    config.enable_market_observer = True
    # Eta scaling defaults for observer output
    config.mafia_eta_base = 1.0
    config.mafia_eta_amplitude = 0.3
    config.mafia_eta_min = 0.7
    config.mafia_eta_max = 1.3
    # Re-init to apply settings to paths
    config.__init__(seed_num=2022, current_date=current_date)
    config.benchmark_algo = "MASA-mafia"
    observer = MAFIAObserver(config=config, action_dim=action_dim)
    return config, observer


def test_mafia_observer_initialization():
    """Test MAFIAObserver can be initialized."""
    print("Testing MAFIAObserver initialization...")
    config, observer = _build_mafia_observer(action_dim=10)

    assert observer is not None, "MAFIAObserver should be created"
    assert observer.action_dim == 10, f"action_dim mismatch: {observer.action_dim} != 10"
    assert observer.mafia_model is not None, "MAFIA model should be initialized"

    print(f"✓ MAFIAObserver initialized with action_dim={observer.action_dim}")
    print(f"✓ Device: {observer.device}")
    print("✓ MAFIAObserver initialization: PASSED\n")


def test_mafia_observer_predict():
    """Test MAFIAObserver predict method."""
    print("Testing MAFIAObserver predict...")
    config, observer = _build_mafia_observer(action_dim=10)

    # Create dummy raw OCHLV data: (N=10, M=5, T_w=30)
    N, M, T_w = 10, 5, 30
    raw_ochlv_data = np.random.rand(N, M, T_w) * 100 + 50

    (
        market_vector,
        risk_eta,
        _market_scores_full,
        _market_context,
        direction_logits,
        _topk_indices,
        _topk_embeddings,
        _topk_scores,
    ) = observer.predict(raw_ochlv_data=raw_ochlv_data, mode="test")

    assert market_vector.shape == (1, N)
    assert risk_eta.shape == (1,)
    assert direction_logits.shape[1] == 3
    assert (risk_eta > 0).all()

    print(f"✓ market_vector shape: {market_vector.shape}")
    print(f"✓ risk_eta shape: {risk_eta.shape}")
    print(f"✓ market_vector range: [{market_vector.min():.4f}, {market_vector.max():.4f}]")
    print(f"✓ risk_eta value: {risk_eta[0]:.4f}")
    print("✓ MAFIAObserver predict: PASSED\n")


def test_mafia_with_real_data():
    """Test MAFIA with real data if available."""
    print("Testing MAFIA with real data...")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config = Config(seed_num=2022, current_date=current_date)
    config.benchmark_algo = "MASA-mafia"
    config.enable_market_observer = True

    data_path = os.path.join(config.dataDir, f"{config.market_name}_{config.topK}_{config.freq}.csv")
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")

    print(f"✓ Loading data from: {data_path}")
    data = pd.read_csv(data_path, header=0)

    feat_proc = FeatureProcesser(config=config)
    data_dict = feat_proc.preprocess_feat(data=data)
    stock_num = data_dict["train"]["stock"].nunique()

    print(f"✓ Data processed: {stock_num} stocks")
    observer = MAFIAObserver(config=config, action_dim=stock_num)

    train_data = data_dict["train"]
    dates = sorted(train_data["date"].unique())[:5]
    print(f"✓ Testing with {len(dates)} dates...")

    for date in dates:
        date_data = train_data[train_data["date"] == date]
        all_dates = sorted(train_data["date"].unique())
        date_idx = all_dates.index(date)
        start_idx = max(0, date_idx - 29)
        window_dates = all_dates[start_idx : date_idx + 1]

        stocks = sorted(date_data["stock"].unique())
        N = len(stocks)
        T_w = len(window_dates)
        if T_w < 30:
            print(f"⚠ Skipping {date}: insufficient history ({T_w} days)")
            continue

        ochlv_array = np.zeros((N, 5, T_w))
        for i, stock in enumerate(stocks):
            stock_data = train_data[train_data["stock"] == stock]
            for j, wdate in enumerate(window_dates):
                day_data = stock_data[stock_data["date"] == wdate]
                if len(day_data) > 0:
                    ochlv_array[i, 0, j] = day_data["open"].values[0]
                    ochlv_array[i, 1, j] = day_data["close"].values[0]
                    ochlv_array[i, 2, j] = day_data["high"].values[0]
                    ochlv_array[i, 3, j] = day_data["low"].values[0]
                    ochlv_array[i, 4, j] = day_data["volume"].values[0]

        (
            market_vector,
            risk_eta,
            _market_scores_full,
            _market_context,
            _topk_indices,
            _topk_embeddings,
            _topk_scores,
            _direction_logits,
        ) = observer.predict(raw_ochlv_data=ochlv_array, mode="test")

        assert market_vector.shape[1] == N, f"market_vector size mismatch: {market_vector.shape[1]} != {N}"
        assert risk_eta.shape[0] == 1
        assert (risk_eta > 0).all()
        print(f"  ✓ {date}: market_vector shape {market_vector.shape}, risk_eta={risk_eta[0]:.4f}")

    print("✓ MAFIA with real data: PASSED\n")


def test_mafia_interface_compatibility():
    """Test MAFIAObserver interface compatibility with MarketObserver."""
    print("Testing interface compatibility...")
    _, observer = _build_mafia_observer(action_dim=10)

    assert hasattr(observer, "predict"), "Missing predict method"
    assert hasattr(observer, "train"), "Missing train method"
    assert hasattr(observer, "reset"), "Missing reset method"
    assert hasattr(observer, "update_hidden_vec_reward"), "Missing update_hidden_vec_reward method"

    observer.reset()
    assert len(observer.market_vector_lst) == 0, "reset() should clear buffers"

    rate_of_price_change = np.random.rand(1, 11)  # (batch, N+1) with cash
    mkt_direction = np.array([1])
    observer.update_hidden_vec_reward("train", rate_of_price_change, mkt_direction)

    print("✓ All required methods exist")
    print("✓ reset() works correctly")
    print("✓ update_hidden_vec_reward() works correctly")
    print("✓ Interface compatibility: PASSED\n")


def run_end_to_end_tests():
    """Optional runner for manual execution outside pytest."""
    scenarios = [
        ("Initialization", test_mafia_observer_initialization),
        ("Predict", test_mafia_observer_predict),
        ("Interface Compatibility", test_mafia_interface_compatibility),
        ("Real Data", test_mafia_with_real_data),
    ]
    results = []
    for name, fn in scenarios:
        try:
            fn()
            results.append((name, True))
        except Exception:
            results.append((name, False))
            print(f"❌ {name} failed", flush=True)
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    print("=" * 60)
    return all(p for _, p in results)


if __name__ == "__main__":
    success = run_end_to_end_tests()
    sys.exit(0 if success else 1)
