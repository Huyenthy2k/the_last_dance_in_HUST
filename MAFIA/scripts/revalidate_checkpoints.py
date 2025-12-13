#!/usr/bin/env python3
"""
Re-validate all existing checkpoints to regenerate valid_metrics.csv
with the new metric columns (topk_advantage, reward, net_reward, etc.)

Usage:
    python scripts/revalidate_checkpoints.py --iter 0 --valid-year 2017
"""

import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd
import torch as th

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from RL_controller.validation_tracker import ValidationMetricsTracker
from utils.mafia_data_loader import load_mafia_data


def main():
    parser = argparse.ArgumentParser(description="Re-validate checkpoints")
    parser.add_argument("--iter", type=int, default=0, help="Iteration index")
    parser.add_argument("--valid-year", type=int, default=2017, help="Validation year")
    parser.add_argument("--start-year", type=int, default=2015, help="Training start year")
    parser.add_argument("--output-dir", type=str, default="./observer_offline", help="Output directory")
    args = parser.parse_args()

    iteration = args.iter
    valid_year = args.valid_year
    start_year = args.start_year
    output_dir = args.output_dir

    # Paths
    checkpoint_dir = os.path.join(output_dir, "checkpoints", f"temp_iter_{iteration}")
    iter_output_dir = os.path.join(output_dir, f"iter_{iteration}_valid_{valid_year}")

    print(f"[REVALIDATE] Iteration {iteration}, Valid Year {valid_year}")
    print(f"[REVALIDATE] Checkpoint dir: {checkpoint_dir}")
    print(f"[REVALIDATE] Output dir: {iter_output_dir}")

    # Find all checkpoints
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth")))
    if not checkpoint_files:
        print("[ERROR] No checkpoint files found!")
        return

    print(f"[REVALIDATE] Found {len(checkpoint_files)} checkpoints")

    # Load config
    config = Config()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"[REVALIDATE] Using device: {device}")

    # Load data
    print("[REVALIDATE] Loading data...")
    data = load_mafia_data(config)

    # Get action_dim from data (number of stocks)
    stock_list = data["stock"].unique().tolist()
    action_dim = len(stock_list)
    print(f"[REVALIDATE] Number of stocks (action_dim): {action_dim}")

    # Create observer and trainer
    observer = MAFIAObserver(config, action_dim=action_dim)
    observer.mafia_model.to(device)

    trainer = ObserverOfflineBatchTrainer(config, observer)
    trainer.device = device

    # Date ranges
    train_start = pd.Timestamp(f"{start_year}-01-01")
    train_end = pd.Timestamp(f"{valid_year}-06-30 23:59:59")
    valid_start = pd.Timestamp(f"{valid_year}-07-01")
    valid_end = pd.Timestamp(f"{valid_year}-12-31 23:59:59")

    # Prepare data tensors (need full range for validation)
    print("[REVALIDATE] Preparing data tensors...")
    data_tensors = trainer.prepare_data_tensors(
        data=data,
        stock_list=stock_list,
        start_date=train_start,  # Start from training start to include all data
        end_date=valid_end,      # End at validation end
    )

    # Create date masks for validation set
    dates = data_tensors["dates"]

    valid_mask = (dates >= valid_start) & (dates <= valid_end)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print(f"[ERROR] No validation data in range {valid_start} - {valid_end}")
        return

    print(f"[REVALIDATE] Validation data: {len(valid_indices)} days")

    valid_tensors = {
        "ochlv": data_tensors["ochlv"][valid_indices[0] : valid_indices[-1] + 1],
        "returns": data_tensors["returns"][valid_indices[0] : valid_indices[-1] + 1],
        "market_ochlv": data_tensors.get("market_ochlv", None)[valid_indices[0] : valid_indices[-1] + 1] if data_tensors.get("market_ochlv") is not None else None,
        "market_returns": data_tensors.get("market_returns", None)[valid_indices[0] : valid_indices[-1] + 1] if data_tensors.get("market_returns") is not None else None,
        "dates": dates[valid_indices],
        "stock_list": data_tensors["stock_list"],
        "T_total": len(valid_indices),
        "N": data_tensors["N"],
    }

    # Initialize tracker
    os.makedirs(iter_output_dir, exist_ok=True)
    tracker = ValidationMetricsTracker(iter_output_dir)

    # Compute batches for validation
    T_valid = valid_tensors["T_total"]
    T_m = trainer.T_m
    h = trainer.horizon
    T_w = trainer.T_w
    batch_size = trainer.batch_size
    available_starts = T_valid - T_m - h - T_w
    batches_per_epoch = max(1, int(np.ceil(available_starts / batch_size)))
    val_steps = max(1, batches_per_epoch // 4)

    print(f"[REVALIDATE] Validation steps per epoch: {val_steps}")

    # Re-validate each checkpoint
    for ckpt_path in checkpoint_files:
        ckpt_name = os.path.basename(ckpt_path)
        epoch = int(ckpt_name.replace("epoch_", "").replace(".pth", ""))

        print(f"\n[REVALIDATE] Processing {ckpt_name} (epoch {epoch})...")

        try:
            # Load checkpoint
            trainer.observer.load_checkpoint(ckpt_path)
            trainer.observer.mafia_model.to(device)
            trainer.observer.mafia_model.eval()

            # Run validation
            val_result = trainer.validate_epoch(
                data_tensors=valid_tensors,
                steps=val_steps,
            )

            # Add to tracker (will replace if epoch exists due to our fix)
            tracker.add_epoch(val_result)

            print(f"    Sharpe: {val_result.topk_sharpe_ratio:.4f}")
            print(f"    Dir F1: {val_result.direction_f1_macro:.4f}")
            print(f"    Risk MSE: {val_result.risk_mse:.4f}")
            print(f"    CES: {val_result.ces_score:.4f}")
            print(f"    topk_advantage: {val_result.topk_advantage:.4f}")
            print(f"    reward: {val_result.reward:.4f}")
            print(f"    net_reward: {val_result.net_reward:.4f}")

        except Exception as e:
            print(f"    [ERROR] Failed to validate: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    csv_path = tracker.save_validation_history()
    print(f"\n[REVALIDATE] Saved validation metrics to: {csv_path}")

    # Print best checkpoint
    best_info = tracker.get_best_checkpoint_info()
    print(f"\n[REVALIDATE] Best checkpoint:")
    print(f"    Epoch: {best_info['epoch']}")
    print(f"    CES: {best_info['ces_score']:.4f}")
    print(f"    Sharpe: {best_info['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    main()
