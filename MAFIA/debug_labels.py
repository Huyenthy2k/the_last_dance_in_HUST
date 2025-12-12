#!/usr/bin/env python3
"""Debug script to check label assignment."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from unittest.mock import MagicMock
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer

# Mock Config
config = MagicMock()
config.mafia_trajectory_length = 10
config.mafia_batch_size = 30
config.mafia_pg_reward_horizon = 5
config.mafia_T_w = 5
config.mafia_risk_scaling_factor = 1000.0
config.direction_label_lookahead = 2
config.risk_eta_lookahead = 5
config.mafia_direction_threshold = 0.02
config.gradient_accumulation_steps = 1
config.mafia_use_class_balanced_sampling = True

# Mock Observer
observer = MagicMock()

# Initialize Trainer
trainer = ObserverOfflineBatchTrainer(config, observer, device=torch.device("cpu"))

# Create synthetic data
T_total = 300
N = 1
dates = np.arange(T_total)
stock_list = ["TEST"]

ochlv = torch.zeros((T_total, N, 5))
market_ochlv = torch.zeros((T_total, 1, 5))

base_price = 100.0

# Bear: 0-99
for i in range(100):
    market_ochlv[i, 0, 1] = base_price * np.exp(-i * 0.025)
    market_ochlv[i, 0, 2] = market_ochlv[i, 0, 1] * 1.005
    market_ochlv[i, 0, 3] = market_ochlv[i, 0, 1] * 0.98

# Side: 100-199
for i in range(100, 200):
    market_ochlv[i, 0, 1] = 50.0 + 0.5 * np.sin((i-100) * 0.3)
    market_ochlv[i, 0, 2] = market_ochlv[i, 0, 1] * 1.003
    market_ochlv[i, 0, 3] = market_ochlv[i, 0, 1] * 0.998

# Bull: 200-299
for i in range(200, 300):
    market_ochlv[i, 0, 1] = 50.0 * np.exp((i - 200) * 0.025)
    market_ochlv[i, 0, 2] = market_ochlv[i, 0, 1] * 1.005
    market_ochlv[i, 0, 3] = market_ochlv[i, 0, 1] * 0.999

ochlv[:, :, 1] = market_ochlv[:, 0, 1].unsqueeze(1)
ochlv[:, :, 2] = market_ochlv[:, 0, 2].unsqueeze(1)
ochlv[:, :, 3] = market_ochlv[:, 0, 3].unsqueeze(1)

data_tensors = {
    "ochlv": ochlv,
    "returns": torch.zeros((T_total, N)),
    "market_ochlv": market_ochlv,
    "market_returns": torch.zeros((T_total, 1)),
    "dates": dates,
    "T_total": T_total,
    "stock_list": stock_list
}

# Build class indices
min_start = 5
max_start = 280
dir_lookahead = 2

class_indices = trainer._build_class_indices(data_tensors, min_start, max_start, dir_lookahead)

print("Class indices count:")
for k, v in class_indices.items():
    print(f"  Class {k}: {len(v)} indices")
    if len(v) > 0:
        print(f"    First 10: {v[:10]}")
        print(f"    Last 10: {v[-10:]}")

# Sample some indices and check their labels
print("\nSample label checks:")
for test_idx in [10, 50, 90, 110, 150, 190, 210, 250, 270]:
    close = market_ochlv[:, 0, 1].cpu().numpy()
    low = market_ochlv[:, 0, 3].cpu().numpy()

    future_idx = min(test_idx + dir_lookahead, len(close) - 1)
    r_fut = (close[future_idx] / close[test_idx]) - 1.0 if close[test_idx] > 0 else 0

    # Intra drawdown
    future_lows = low[test_idx+1:future_idx+1]
    if len(future_lows) > 0:
        min_price = np.min(future_lows)
        intra_dd = (min_price / close[test_idx]) - 1.0 if close[test_idx] > 0 else 0
    else:
        intra_dd = 0

    # Determine label
    delta_t = 0.02  # Simplified
    stop_loss = -0.07

    if r_fut < -delta_t or intra_dd < stop_loss:
        label = 0  # Bear
    elif r_fut > delta_t and intra_dd >= stop_loss:
        label = 2  # Bull
    else:
        label = 1  # Side

    print(f"  t={test_idx}: close={close[test_idx]:.2f}, fut={close[future_idx]:.2f}, "
          f"r_fut={r_fut*100:.2f}%, intra_dd={intra_dd*100:.2f}%, label={label}")
