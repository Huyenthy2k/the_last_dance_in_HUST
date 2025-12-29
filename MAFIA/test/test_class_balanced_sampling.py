#!/usr/bin/env python3
"""
Test script to verify class-balanced sampling works correctly.
"""

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
config.mafia_batch_size = 30  # Divisible by 3 for easy testing
config.mafia_pg_reward_horizon = 5
config.mafia_T_w = 5
config.mafia_risk_scaling_factor = 1000.0
config.direction_label_lookahead = 2
config.risk_eta_lookahead = 5
config.mafia_direction_threshold = 0.02  # For _build_class_indices
config.direction_label_delta_min = 0.02  # For sample_trajectory_batch
config.direction_label_atr_multiplier = 2.0
config.direction_label_stop_loss = -0.07
config.gradient_accumulation_steps = 1
config.regime_vol_k = 3.0
config.regime_vol_window = 60
config.mafia_DC_threshold = 0.02
config.mafia_price_floor = 1e-6
config.mafia_extreme_return_threshold = 10.0
config.mafia_return_clip_min = -0.5
config.mafia_return_clip_max = 1.0
config.mafia_pg_alpha_turnover = 0.01
config.mafia_pg_alpha_change = 0.01
config.mafia_lambda_pg = 1.0
config.mafia_lambda_risk = 0.3
config.mafia_lambda_dir = 0.5
config.mafia_beta_entropy = 0.01
config.mafia_focal_gamma = 2.0
config.mafia_focal_alpha = [1.65, 0.70, 1.0]
config.mafia_top_k = 5
config.mafia_topk_rebalance_interval = 2
config.use_mixed_precision = False
config.mafia_use_class_balanced_sampling = True  # ENABLE CLASS-BALANCED SAMPLING

# Mock Observer
observer = MagicMock()
observer.optimizer = MagicMock()

# Initialize Trainer
trainer = ObserverOfflineBatchTrainer(config, observer, device=torch.device("cpu"))

# Create synthetic data with known class distribution
T_total = 300
N = 1
dates = np.arange(T_total)
stock_list = ["TEST"]

# Create OCHLV data
ochlv = torch.zeros((T_total, N, 5))
market_ochlv = torch.zeros((T_total, 1, 5))

# Set base prices
base_price = 100.0
ochlv[:, :, 1] = base_price  # Close
market_ochlv[:, 0, 1] = base_price

# Create 3 regions with SMOOTH trends (low ATR) but clear direction
# Goal: ATR/Price ratio < 1% so delta_t stays near 2% minimum

# Region 1 (0-99): Bear - smooth decline with >2% drop every 2 days
for i in range(100):
    market_ochlv[i, 0, 1] = base_price * (1.0 - i * 0.015)  # Linear decline -1.5%/day
    market_ochlv[i, 0, 2] = market_ochlv[i, 0, 1] + 0.10  # Tight range
    market_ochlv[i, 0, 3] = market_ochlv[i, 0, 1] - 0.10

# Region 2 (100-199): Side - flat/minimal movement
for i in range(100, 200):
    market_ochlv[i, 0, 1] = 50.0 + 0.25 * np.sin((i-100) * 0.1)  # Tiny oscillation
    market_ochlv[i, 0, 2] = market_ochlv[i, 0, 1] + 0.05
    market_ochlv[i, 0, 3] = market_ochlv[i, 0, 1] - 0.05

# Region 3 (200-299): Bull - smooth rise with >2% gain every 2 days
for i in range(200, 300):
    market_ochlv[i, 0, 1] = 50.0 + (i - 200) * 0.75  # Linear growth +1.5%/day
    market_ochlv[i, 0, 2] = market_ochlv[i, 0, 1] + 0.10
    market_ochlv[i, 0, 3] = market_ochlv[i, 0, 1] - 0.10

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

print("="*80)
print("TESTING CLASS-BALANCED SAMPLING")
print("="*80)

# Sample multiple batches and check class distribution
n_batches = 100
all_labels = []

for _ in range(n_batches):
    batch = trainer.sample_trajectory_batch(data_tensors, mode="TRAIN")
    # Get direction labels for first timestep of each trajectory
    labels = batch.direction_labels[:, 0].cpu().numpy()
    all_labels.extend(labels)

all_labels = np.array(all_labels)

# Count class distribution
from collections import Counter
label_counts = Counter(all_labels)
total = len(all_labels)

print(f"\nSampled {total} trajectories across {n_batches} batches")
print(f"Batch size: {config.mafia_batch_size}")
print("\nClass Distribution in Sampled Batches:")
print(f"  Bear  (0): {label_counts[0]:5d} ({100*label_counts[0]/total:.1f}%)")
print(f"  Side  (1): {label_counts[1]:5d} ({100*label_counts[1]/total:.1f}%)")
print(f"  Bull  (2): {label_counts[2]:5d} ({100*label_counts[2]/total:.1f}%)")

# Expected: roughly 33.3% each if class-balanced sampling works
bear_pct = label_counts[0] / total
side_pct = label_counts[1] / total
bull_pct = label_counts[2] / total

print("\nExpected with class-balanced sampling: ~33.3% each")
print(f"Actual: Bear={bear_pct*100:.1f}%, Side={side_pct*100:.1f}%, Bull={bull_pct*100:.1f}%")

# Check if within reasonable tolerance (±5%)
tolerance = 0.05
target = 1.0 / 3.0

if abs(bear_pct - target) < tolerance and abs(side_pct - target) < tolerance and abs(bull_pct - target) < tolerance:
    print("\n✅ SUCCESS: Class-balanced sampling is working correctly!")
else:
    print("\n❌ FAILED: Class distribution is not balanced!")

print("\n" + "="*80)
