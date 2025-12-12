#!/usr/bin/env python3
"""
Debug script to analyze direction prediction bias and focal loss effectiveness.

This script checks:
1. True label distribution in training data
2. Model's prediction distribution (is it always predicting Side?)
3. Whether focal loss is actually balancing the gradients
4. Initial loss value analysis
"""

import os
import sys
import numpy as np
import torch as th

# Add repo root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer, FocalLoss
from utils.mafia_data_loader import load_mafia_data
import pandas as pd


def analyze_focal_loss():
    """Analyze focal loss behavior with different scenarios."""
    print("\n" + "=" * 70)
    print("FOCAL LOSS ANALYSIS")
    print("=" * 70)

    config = Config(create_dirs=False)
    alpha = config.mafia_focal_alpha  # [1.65, 0.70, 1.0] - [Bear, Side, Bull]
    gamma = config.mafia_focal_gamma  # 2.0

    print(f"\nConfig:")
    print(f"  alpha (class weights) = {alpha}")
    print(f"  gamma (focusing param) = {gamma}")
    print(f"  Order: [Bear(0), Side(1), Bull(2)]")

    # Create focal loss
    focal = FocalLoss(gamma=gamma, alpha=alpha)
    ce_loss = th.nn.CrossEntropyLoss()

    # Scenario 1: Model always predicts Side (index 1) with high confidence
    print("\n--- Scenario 1: Model biased toward Side ---")
    logits_side_bias = th.tensor([
        [-1.0, 2.0, -0.5],  # Predicts Side with high prob
        [-0.5, 1.5, 0.0],   # Predicts Side
        [-2.0, 3.0, -1.0],  # Strong Side prediction
    ], dtype=th.float32)

    probs = th.softmax(logits_side_bias, dim=1)
    print(f"  Logits: {logits_side_bias.tolist()}")
    print(f"  Probs:  {probs.tolist()}")

    for target_class, class_name in enumerate(["Bear", "Side", "Bull"]):
        targets = th.tensor([target_class, target_class, target_class], dtype=th.long)
        fl = focal(logits_side_bias, targets)
        ce = ce_loss(logits_side_bias, targets)
        print(f"  Target={class_name}: Focal Loss={fl.item():.4f}, CE Loss={ce.item():.4f}")

    # Scenario 2: Uniform predictions (no bias)
    print("\n--- Scenario 2: Uniform predictions ---")
    logits_uniform = th.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=th.float32)

    probs_uniform = th.softmax(logits_uniform, dim=1)
    print(f"  Probs: {probs_uniform[0].tolist()}")

    for target_class, class_name in enumerate(["Bear", "Side", "Bull"]):
        targets = th.tensor([target_class, target_class, target_class], dtype=th.long)
        fl = focal(logits_uniform, targets)
        ce = ce_loss(logits_uniform, targets)
        print(f"  Target={class_name}: Focal Loss={fl.item():.4f}, CE Loss={ce.item():.4f}")

    # Check what loss value corresponds to what probability
    print("\n--- Loss vs Probability Analysis ---")
    print("  For a single sample with label=Bear (class 0):")
    print("  Prob(Bear) -> Focal Loss | CE Loss")
    for prob_bear in [0.1, 0.2, 0.33, 0.5, 0.6, 0.8, 0.9]:
        # Create logits that give approximately this probability
        # softmax([x, 0, 0]) = [e^x / (e^x + 2), 1/(e^x+2), 1/(e^x+2)]
        # We want e^x / (e^x + 2) = prob_bear
        # e^x = prob_bear * (e^x + 2)
        # e^x * (1 - prob_bear) = 2 * prob_bear
        # e^x = 2 * prob_bear / (1 - prob_bear)
        if prob_bear < 1:
            x = np.log(2 * prob_bear / (1 - prob_bear + 1e-8))
        else:
            x = 10
        logits = th.tensor([[x, 0.0, 0.0]], dtype=th.float32)
        actual_prob = th.softmax(logits, dim=1)[0, 0].item()

        targets = th.tensor([0], dtype=th.long)  # Bear
        fl = focal(logits, targets)
        ce = ce_loss(logits, targets)
        print(f"    {actual_prob:.2f} -> Focal: {fl.item():.4f}, CE: {ce.item():.4f}")


def analyze_label_distribution():
    """Analyze the actual distribution of direction labels in training data."""
    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 70)

    config = Config(create_dirs=False)

    # Load data
    try:
        stock_data = load_mafia_data(config)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Load market data
    data_dir = getattr(config, "dataDir", "./data")
    market_file = os.path.join(data_dir, "vnindex_data.csv")
    if os.path.exists(market_file):
        market_data = pd.read_csv(market_file, parse_dates=["date"])
    else:
        market_file = os.path.join(data_dir, "VNINDEX_1d_index.csv")
        if os.path.exists(market_file):
            market_data = pd.read_csv(market_file, parse_dates=["date"])
        else:
            print("No market data found")
            return

    print(f"\nMarket data: {len(market_data)} rows")

    # Compute direction labels for VNINDEX
    dir_lookahead = int(getattr(config, "direction_label_lookahead", 14))
    delta_min = float(getattr(config, "direction_label_delta_min", 0.02))
    k_atr = float(getattr(config, "direction_label_atr_multiplier", 2.0))
    atr_period = int(getattr(config, "direction_label_atr_period", 14))
    stop_loss = float(getattr(config, "direction_label_stop_loss", -0.07))

    print(f"\nLabel config:")
    print(f"  dir_lookahead = {dir_lookahead} days")
    print(f"  delta_min = {delta_min} ({delta_min*100:.1f}%)")
    print(f"  k_atr = {k_atr}")
    print(f"  atr_period = {atr_period}")
    print(f"  stop_loss = {stop_loss} ({stop_loss*100:.1f}%)")

    # Sort by date
    market_data = market_data.sort_values("date").reset_index(drop=True)

    # Compute ATR
    high = market_data["high"].values
    low = market_data["low"].values
    close = market_data["close"].values

    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[tr[0]], tr])

    # SMA ATR
    atr = np.convolve(tr, np.ones(atr_period) / atr_period, mode="same")

    # Compute labels
    labels = []
    T = len(close)

    for t in range(T - dir_lookahead):
        p_t = close[t]
        future_idx = t + dir_lookahead
        p_fut = close[future_idx]

        # Dynamic threshold
        delta_t = max(delta_min, k_atr * (atr[t] / (p_t + 1e-8)))

        # Future return
        r_fut = (p_fut - p_t) / p_t

        # Intra-period drawdown
        min_low = low[t+1:future_idx+1].min()
        intra_dd = (min_low / p_t) - 1.0

        # Label logic
        if r_fut < -delta_t or intra_dd < stop_loss:
            label = 0  # Bear
        elif r_fut > delta_t and intra_dd >= stop_loss:
            label = 2  # Bull
        else:
            label = 1  # Side

        labels.append(label)

    labels = np.array(labels)

    # Count distribution
    bear_count = (labels == 0).sum()
    side_count = (labels == 1).sum()
    bull_count = (labels == 2).sum()
    total = len(labels)

    print(f"\n--- Label Distribution (VNINDEX) ---")
    print(f"  Bear (0): {bear_count:5d} ({bear_count/total*100:5.1f}%)")
    print(f"  Side (1): {side_count:5d} ({side_count/total*100:5.1f}%)")
    print(f"  Bull (2): {bull_count:5d} ({bull_count/total*100:5.1f}%)")
    print(f"  Total:    {total:5d}")

    # Analyze per year
    print(f"\n--- Per-Year Distribution ---")
    market_data = market_data.iloc[:len(labels)].copy()
    market_data["label"] = labels
    market_data["year"] = pd.to_datetime(market_data["date"]).dt.year

    for year in sorted(market_data["year"].unique()):
        year_labels = market_data[market_data["year"] == year]["label"]
        n = len(year_labels)
        if n > 0:
            b = (year_labels == 0).sum()
            s = (year_labels == 1).sum()
            u = (year_labels == 2).sum()
            print(f"  {year}: Bear={b/n*100:4.1f}%, Side={s/n*100:4.1f}%, Bull={u/n*100:4.1f}% (n={n})")

    # What if we change threshold?
    print(f"\n--- Sensitivity to delta_min ---")
    for test_delta in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
        test_labels = []
        for t in range(T - dir_lookahead):
            p_t = close[t]
            future_idx = t + dir_lookahead
            p_fut = close[future_idx]
            delta_t = max(test_delta, k_atr * (atr[t] / (p_t + 1e-8)))
            r_fut = (p_fut - p_t) / p_t
            min_low = low[t+1:future_idx+1].min()
            intra_dd = (min_low / p_t) - 1.0

            if r_fut < -delta_t or intra_dd < stop_loss:
                test_labels.append(0)
            elif r_fut > delta_t and intra_dd >= stop_loss:
                test_labels.append(2)
            else:
                test_labels.append(1)

        test_labels = np.array(test_labels)
        b = (test_labels == 0).sum() / len(test_labels) * 100
        s = (test_labels == 1).sum() / len(test_labels) * 100
        u = (test_labels == 2).sum() / len(test_labels) * 100
        print(f"  delta={test_delta:.3f}: Bear={b:4.1f}%, Side={s:4.1f}%, Bull={u:4.1f}%")


def analyze_model_predictions():
    """Check if model is biased in its predictions."""
    print("\n" + "=" * 70)
    print("MODEL PREDICTION BIAS CHECK")
    print("=" * 70)

    config = Config(create_dirs=False)

    # Create a fresh observer (random initialization)
    stock_list = [f"STOCK_{i}" for i in range(10)]  # Dummy
    observer = MAFIAObserver(config=config, action_dim=10)

    # Get the direction head
    model = observer.mafia_model

    # Check if direction head has bias
    print("\n--- Direction Head Bias Check ---")
    if hasattr(model, "direction_head"):
        dir_head = model.direction_head
        print(f"  Direction head: {dir_head}")

        # Check last layer bias
        for name, param in dir_head.named_parameters():
            if "bias" in name:
                print(f"  {name}: {param.data.tolist()}")

    # Do a forward pass with random input and check output distribution
    print("\n--- Random Input Test ---")

    # Create random hidden state
    hidden_dim = config.mafia_hidden_dim
    batch_size = 32

    # Simulate calling forward
    device = th.device("cpu")
    model.to(device)
    model.eval()

    with th.no_grad():
        # Create random market context (what the direction head sees)
        # Check model architecture to understand input
        print(f"  Model type: {type(model).__name__}")

        # Generate some random hidden states
        hidden = th.randn(batch_size, hidden_dim)

        # If direction_head takes hidden state directly
        if hasattr(model, "direction_head"):
            logits = model.direction_head(hidden)
            probs = th.softmax(logits, dim=-1)

            mean_probs = probs.mean(dim=0)
            print(f"  Mean output probs: Bear={mean_probs[0]:.3f}, Side={mean_probs[1]:.3f}, Bull={mean_probs[2]:.3f}")

            preds = logits.argmax(dim=-1)
            bear_pct = (preds == 0).float().mean().item() * 100
            side_pct = (preds == 1).float().mean().item() * 100
            bull_pct = (preds == 2).float().mean().item() * 100
            print(f"  Prediction distribution: Bear={bear_pct:.1f}%, Side={side_pct:.1f}%, Bull={bull_pct:.1f}%")


def main():
    print("\n" + "=" * 70)
    print("DIRECTION PREDICTION BIAS DEBUGGER")
    print("=" * 70)

    # 1. Analyze focal loss math
    analyze_focal_loss()

    # 2. Analyze label distribution
    analyze_label_distribution()

    # 3. Check model bias
    analyze_model_predictions()

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. If Side class dominates (>50%), consider:
   - LOWER delta_min (e.g., 0.015 instead of 0.02)
   - SHORTER dir_lookahead (e.g., 7 instead of 14)

2. If model always predicts Side:
   - Check initialization bias (direction head)
   - Increase alpha for Bear class (e.g., 2.0 instead of 1.65)
   - Decrease alpha for Side class (e.g., 0.3 instead of 0.70)

3. If loss starts too low (0.5):
   - Random uniform CE loss should be ~1.1
   - Low loss = model already biased toward majority class
   - May need to re-initialize or check data pipeline

4. Verify focal loss is working:
   - Add logging: print prediction distribution each batch
   - Compare focal vs regular CE training
""")


if __name__ == "__main__":
    main()
