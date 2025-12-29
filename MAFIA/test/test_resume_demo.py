#!/usr/bin/env python3
"""
Demo script to test Observer training resume functionality.

This script:
1. Generates minimal demo data
2. Runs training for a few epochs
3. Simulates interrupt by stopping early
4. Resumes training and verifies continuity

Usage:
    cd agents/MAFIA
    python scripts/test_resume_demo.py
"""

import os
import sys
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

# Add MAFIA root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAFIA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if MAFIA_ROOT not in sys.path:
    sys.path.insert(0, MAFIA_ROOT)


def generate_demo_data(output_dir: str, num_stocks: int = 10, num_days: int = 900):
    """Generate minimal demo stock data for testing."""
    print(f"[DEMO] Generating demo data: {num_stocks} stocks, {num_days} days")
    np.random.seed(42)

    stocks = [f"STK{i:02d}" for i in range(num_stocks)]
    dates = pd.date_range("2015-01-01", periods=num_days, freq="D")

    records = []
    for stock in stocks:
        base_price = np.random.uniform(20, 80)
        returns = np.random.normal(0.0003, 0.02, num_days)
        prices = base_price * np.cumprod(1 + returns)

        for i, date in enumerate(dates):
            price = prices[i]
            high = price * (1 + np.random.uniform(0, 0.03))
            low = price * (1 - np.random.uniform(0, 0.03))
            open_p = np.random.uniform(low, high)
            volume = np.random.randint(100000, 5000000)

            records.append({
                "date": date,
                "stock": stock,
                "open": open_p,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["date", "stock"]).reset_index(drop=True)

    # Save stock data
    stock_file = os.path.join(output_dir, "stock_data.csv")
    df.to_csv(stock_file, index=False)
    print(f"[DEMO] Saved stock data: {stock_file} ({len(df)} rows)")

    # Generate market data
    base_price = 500
    returns = np.random.normal(0.0002, 0.012, num_days)
    prices = base_price * np.cumprod(1 + returns)

    market_records = []
    for i, date in enumerate(dates):
        price = prices[i]
        high = price * (1 + np.random.uniform(0, 0.015))
        low = price * (1 - np.random.uniform(0, 0.015))
        open_p = np.random.uniform(low, high)
        volume = np.random.randint(10000000, 100000000)

        market_records.append({
            "date": date,
            "open": open_p,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
        })

    market_df = pd.DataFrame(market_records)
    market_file = os.path.join(output_dir, "vnindex_data.csv")
    market_df.to_csv(market_file, index=False)
    print(f"[DEMO] Saved market data: {market_file} ({len(market_df)} rows)")

    return stock_file, market_file


def check_resume_state(output_dir: str) -> dict:
    """Check the current state for resume."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    state = {
        "has_checkpoints": False,
        "latest_checkpoint": None,
        "best_checkpoints": [],
        "completed_iterations": [],
        "in_progress_iteration": None,
    }

    if not os.path.exists(checkpoint_dir):
        return state

    state["has_checkpoints"] = True

    # Check for completed iterations (observer_best_YYYY.pth)
    for f in os.listdir(checkpoint_dir):
        if f.startswith("observer_best_") and f.endswith(".pth"):
            year = f.replace("observer_best_", "").replace(".pth", "")
            state["completed_iterations"].append(year)

    # Check for in-progress iterations (temp_iter_N)
    for d in os.listdir(checkpoint_dir):
        temp_dir = os.path.join(checkpoint_dir, d)
        if d.startswith("temp_iter_") and os.path.isdir(temp_dir):
            iter_num = d.replace("temp_iter_", "")

            latest_ckpt = os.path.join(temp_dir, "latest_checkpoint.pth")
            if os.path.exists(latest_ckpt):
                state["latest_checkpoint"] = latest_ckpt
                state["in_progress_iteration"] = iter_num

                # Get epoch from checkpoint
                import torch as th
                ckpt = th.load(latest_ckpt, map_location="cpu")
                state["latest_epoch"] = ckpt.get("epoch", -1)

            # Check for best checkpoints
            for f in os.listdir(temp_dir):
                if f.startswith("epoch_") and f.endswith(".pth"):
                    state["best_checkpoints"].append(f)

    return state


def run_training_phase(
    output_dir: str,
    data_dir: str,
    num_epochs: int,
    phase_name: str,
):
    """Run a phase of training."""
    from scripts.train_observer_offline import run_offline_observer_training

    print(f"\n{'=' * 60}")
    print(f"[{phase_name}] Starting training for {num_epochs} epochs")
    print(f"{'=' * 60}")

    # Check resume state before
    state_before = check_resume_state(output_dir)
    if state_before["latest_checkpoint"]:
        print(f"[{phase_name}] Resume state detected:")
        print(f"   Latest checkpoint: {state_before['latest_checkpoint']}")
        print(f"   Latest epoch: {state_before.get('latest_epoch', 'N/A')}")
        print(f"   In-progress iteration: {state_before['in_progress_iteration']}")

    # Patch config to use demo data
    from config import Config
    original_init = Config.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["create_dirs"] = False
        original_init(self, *args, **kwargs)
        self.dataDir = data_dir
        self.stock_data_file = "stock_data.csv"
        self.index_data_file = "vnindex_data.csv"

    Config.__init__ = patched_init

    try:
        results = run_offline_observer_training(
            start_year=2015,
            first_infer_year=2017,
            last_infer_year=2017,  # Single iteration for testing
            output_dir=output_dir,
            num_epochs=num_epochs,
            batches_per_epoch=2,  # Minimal for speed
            seed=2025,
            verbose=True,
        )
        return results
    finally:
        Config.__init__ = original_init


def verify_resume_results(output_dir: str, expected_epochs: int) -> bool:
    """Verify that resume worked correctly."""
    print(f"\n{'=' * 60}")
    print("[VERIFY] Checking resume results...")
    print(f"{'=' * 60}")

    # Check validation history
    iter_dir = os.path.join(output_dir, "iter_0_valid_2016")
    valid_csv = os.path.join(iter_dir, "valid_metrics.csv")

    if not os.path.exists(valid_csv):
        print(f"[FAIL] Validation CSV not found: {valid_csv}")
        return False

    df = pd.read_csv(valid_csv)
    actual_epochs = len(df)

    print(f"   Validation history: {actual_epochs} epochs")
    print(f"   Expected epochs: {expected_epochs}")

    if actual_epochs != expected_epochs:
        print(f"[FAIL] Epoch count mismatch: {actual_epochs} != {expected_epochs}")
        return False

    # Check epoch sequence
    epochs = df["epoch"].tolist()
    expected_sequence = list(range(1, expected_epochs + 1))

    if epochs != expected_sequence:
        print(f"[FAIL] Epoch sequence mismatch: {epochs} != {expected_sequence}")
        return False

    print(f"   Epoch sequence: {epochs}")
    print("[PASS] Resume verification successful!")
    return True


def main():
    print("\n" + "=" * 60)
    print("OBSERVER RESUME DEMO TEST")
    print("=" * 60)

    # Create temp directory for demo
    demo_dir = os.path.join(MAFIA_ROOT, "observer_resume_test")
    data_dir = os.path.join(demo_dir, "data")
    output_dir = os.path.join(demo_dir, "output")

    # Clean up previous run
    if os.path.exists(demo_dir):
        print(f"[DEMO] Cleaning up previous test: {demo_dir}")
        shutil.rmtree(demo_dir)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Generate demo data
    generate_demo_data(data_dir, num_stocks=10, num_days=900)

    # === Phase 1: Train for 3 epochs ===
    print("\n" + "=" * 60)
    print("PHASE 1: Initial training (3 epochs)")
    print("=" * 60)

    run_training_phase(
        output_dir=output_dir,
        data_dir=data_dir,
        num_epochs=3,
        phase_name="PHASE 1",
    )

    # Check state after phase 1
    state_1 = check_resume_state(output_dir)
    print(f"\n[PHASE 1 COMPLETE]")
    print(f"   Completed iterations: {state_1['completed_iterations']}")
    print(f"   In-progress: {state_1['in_progress_iteration']}")

    # If iteration completed (no temp dir), we're done
    if not state_1["in_progress_iteration"]:
        print("\n[INFO] Iteration completed in Phase 1. Testing resume with NEW iteration...")

        # === Phase 2: Train for 2 more epochs (should be no-op if iteration complete) ===
        print("\n" + "=" * 60)
        print("PHASE 2: Resume attempt (iteration already complete)")
        print("=" * 60)

        run_training_phase(
            output_dir=output_dir,
            data_dir=data_dir,
            num_epochs=3,  # Same as before
            phase_name="PHASE 2",
        )

        print("\n[DEMO] Test complete - iteration was fully completed in Phase 1")
        print(f"[DEMO] Output directory: {output_dir}")

        # Cleanup
        cleanup = input("\nCleanup demo directory? (y/n): ").strip().lower()
        if cleanup == "y":
            shutil.rmtree(demo_dir)
            print("[DEMO] Cleaned up.")
        return

    # === Manual interrupt simulation ===
    # In real scenario, user would Ctrl+C during training
    # For this demo, we'll just note that training stopped and resume

    print("\n" + "=" * 60)
    print("SIMULATING INTERRUPT...")
    print("(In real usage, you would Ctrl+C during training)")
    print("=" * 60)

    # === Phase 2: Resume training for more epochs ===
    print("\n" + "=" * 60)
    print("PHASE 2: Resume training (total 5 epochs)")
    print("=" * 60)

    run_training_phase(
        output_dir=output_dir,
        data_dir=data_dir,
        num_epochs=5,  # Total epochs
        phase_name="PHASE 2",
    )

    # Verify results
    verify_resume_results(output_dir, expected_epochs=5)

    print(f"\n[DEMO] Output directory: {output_dir}")
    print("[DEMO] You can inspect the files to verify resume worked correctly.")

    # Cleanup prompt
    cleanup = input("\nCleanup demo directory? (y/n): ").strip().lower()
    if cleanup == "y":
        shutil.rmtree(demo_dir)
        print("[DEMO] Cleaned up.")


if __name__ == "__main__":
    main()
