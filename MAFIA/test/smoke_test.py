#!/usr/bin/env python3
"""
Smoke Test: Minimal walk-forward run để phát hiện sớm các vấn đề làm crash luồng training.

Sử dụng cấu hình tối thiểu:
- 2 windows (để test checkpoint chaining)
- 2 tháng train, 1 tháng valid, 1 tháng test per window
- Step 2 tháng (non-overlapping)
- 2 epochs per window
- Giảm warmup steps
- Tắt verbose logging không cần thiết

Usage:
    python scripts/smoke_test.py
"""

import argparse
import datetime
import os
import sys

import pandas as pd

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config
from entrance import RLcontroller


def create_minimal_config(seed, tag, resume_checkpoint=None):
    """Create minimal config for smoke test."""
    cfg = Config(seed_num=seed, current_date=tag)

    # ========================================
    # MINIMAL EPOCHS
    # ========================================
    cfg.num_epochs = 2  # 2 epochs đủ để test checkpoint logic

    # ========================================
    # GIẢM WARMUP VÀ BUFFER SIZE
    # ========================================
    cfg.learning_starts = 50  # Giảm từ 300 xuống 50
    cfg.observer_mini_epoch_steps = 30  # Giảm từ 126 xuống 30
    cfg.mafia_pretrain_mini_epochs = 1  # Giảm từ 2 xuống 1

    # Buffer nhỏ hơn cho smoke test
    cfg.model_para["buffer_size"] = int(1e4)  # 10K thay vì 156K
    cfg.model_para["learning_starts"] = 50  # Sync với cfg.learning_starts

    # ========================================
    # TẮT VERBOSE LOGGING KHÔNG CẦN THIẾT
    # ========================================
    cfg.mafia_log_scheduler = False
    cfg.mafia_log_eta = False
    cfg.mafia_log_reward = False
    cfg.mafia_log_realtime = True  # Giữ realtime status
    cfg.reward_debug_steps = 5  # Giảm từ 30 xuống 5

    # ========================================
    # CHECKPOINT SETTINGS
    # ========================================
    cfg.checkpoint_freq = 1  # Save every epoch
    cfg.validation_freq = 1  # Validate every epoch
    cfg.early_stop_patience = 5  # Không cần early stop cho smoke test
    cfg.partial_checkpoint_steps = 0  # Tắt step-based checkpoint

    # Resume settings
    cfg.auto_resume_from_latest = False
    cfg.resume_from_checkpoint = resume_checkpoint

    return cfg


def find_checkpoint(res_dir):
    """Pick a checkpoint to resume from."""
    candidates = [
        os.path.join(
            res_dir, "checkpoints", "checkpoint_best_valid", "checkpoint_info.json"
        ),
        os.path.join(
            res_dir, "checkpoints", "checkpoint_final", "checkpoint_info.json"
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_one_window(
    window_idx,
    seed,
    train_start,
    train_end,
    valid_start,
    valid_end,
    test_start,
    test_end,
    resume_checkpoint=None,
):
    """Run one walk-forward window."""
    tag = f"smoke_win{window_idx}_seed{seed}"
    cfg = create_minimal_config(seed, tag, resume_checkpoint)

    # Set date splits
    cfg.train_date_start = train_start
    cfg.train_date_end = train_end
    cfg.valid_date_start = valid_start
    cfg.valid_date_end = valid_end
    cfg.test_date_start = test_start
    cfg.test_date_end = test_end

    # Recalibrate risk bounds for new dates
    cfg.risk_market = cfg.default_risk_market
    cfg._calibrate_risk_bounds()

    print(f"\n{'=' * 70}", flush=True)
    print(f"📊 WINDOW {window_idx} | seed={seed}", flush=True)
    print(f"   Train: {train_start.date()} → {train_end.date()}", flush=True)
    print(f"   Valid: {valid_start.date()} → {valid_end.date()}", flush=True)
    print(f"   Test:  {test_start.date()} → {test_end.date()}", flush=True)
    print(f"   Resume: {resume_checkpoint or 'None (fresh start)'}", flush=True)
    print(f"   Results: {cfg.res_dir}", flush=True)
    print(f"{'=' * 70}", flush=True)

    RLcontroller(cfg)

    next_checkpoint = find_checkpoint(cfg.res_dir)
    print(
        f"✅ Window {window_idx} complete. Next checkpoint: {next_checkpoint}",
        flush=True,
    )

    return cfg.res_dir, next_checkpoint


def run_smoke_test():
    """Run minimal 2-window walk-forward training to verify full pipeline works."""
    print("=" * 70, flush=True)
    print("🔥 SMOKE TEST: 2-Window Walk-Forward Training (2 Seeds)", flush=True)
    print(
        "   Testing: data loading, training, checkpointing, window chaining", flush=True
    )
    print("=" * 70, flush=True)

    seeds = [2025, 2026]  # 2 seeds để test multiple runs

    # ========================================
    # 2 WINDOWS: 2 months train, 1 month valid, 1 month test each
    # Step = 2 months (non-overlapping train periods)
    # ========================================
    # Window 0: Train Jan-Feb 2020, Valid Mar 2020, Test Apr 2020
    # Window 1: Train Mar-Apr 2020, Valid May 2020, Test Jun 2020

    windows = [
        {
            "train_start": pd.Timestamp("2020-01-02"),
            "train_end": pd.Timestamp("2020-02-28"),
            "valid_start": pd.Timestamp("2020-03-01"),
            "valid_end": pd.Timestamp("2020-03-31"),
            "test_start": pd.Timestamp("2020-04-01"),
            "test_end": pd.Timestamp("2020-04-30"),
        },
        {
            "train_start": pd.Timestamp("2020-03-02"),
            "train_end": pd.Timestamp("2020-04-30"),
            "valid_start": pd.Timestamp("2020-05-01"),
            "valid_end": pd.Timestamp("2020-05-31"),
            "test_start": pd.Timestamp("2020-06-01"),
            "test_end": pd.Timestamp("2020-06-30"),
        },
    ]

    print(f"\n📋 SMOKE TEST PLAN:", flush=True)
    print(f"   Windows: {len(windows)}", flush=True)
    print(f"   Seeds: {seeds}", flush=True)
    print(f"   Total runs: {len(windows) * len(seeds)}", flush=True)
    print(f"   Epochs per window: 2", flush=True)

    start_time = datetime.datetime.now()

    try:
        for seed in seeds:
            print(f"\n{'#' * 70}", flush=True)
            print(f"# SEED {seed}", flush=True)
            print(f"{'#' * 70}", flush=True)

            last_checkpoint = None  # Reset checkpoint chain for each seed

            for win_idx, win in enumerate(windows):
                _, last_checkpoint = run_one_window(
                    window_idx=win_idx,
                    seed=seed,
                    train_start=win["train_start"],
                    train_end=win["train_end"],
                    valid_start=win["valid_start"],
                    valid_end=win["valid_end"],
                    test_start=win["test_start"],
                    test_end=win["test_end"],
                    resume_checkpoint=last_checkpoint,
                )

        end_time = datetime.datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70, flush=True)
        print("✅ SMOKE TEST PASSED!", flush=True)
        print(f"   Duration: {duration}", flush=True)
        print(f"   Seeds: {seeds}", flush=True)
        print(f"   Windows per seed: {len(windows)}", flush=True)
        print(f"   Total runs: {len(windows) * len(seeds)}", flush=True)
        print("=" * 70, flush=True)
        return 0

    except Exception as e:
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70, flush=True)
        print("❌ SMOKE TEST FAILED!", flush=True)
        print(f"   Duration: {duration}", flush=True)
        print(f"   Error: {type(e).__name__}: {e}", flush=True)
        print("=" * 70, flush=True)

        import traceback

        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for MAFIA training pipeline"
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run extended smoke test with 3 epochs and more data",
    )
    args = parser.parse_args()

    exit_code = run_smoke_test()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
