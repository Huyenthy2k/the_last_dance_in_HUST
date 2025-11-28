#!/usr/bin/env python3
"""
Walk-forward / sliding-window training:
  - Train for N years
  - Validate for M years
  - Test for K years
  - Slide the window forward by S years and repeat.
  - Checkpoints (incl. replay buffer) are chained across non-overlapping windows by default.
    Use --resume-overlap to also chain when windows overlap (assumes buffer contains train-only data).

Example:
  python scripts/walk_forward.py \
    --start-date 2017-01-01 \
    --num-windows 3 \
    --train-years 3 --valid-years 1 --test-years 1 \
    --step-years 1 \
    --base-seed 2025 --num-seeds 1

Notes:
  - Each window runs independently with its own results directory (tagged wf_win{idx}_seed{seed}).
  - Only hparams from Config are used (no auto Optuna here). Adjust Config before calling if needed,
    or supply overrides programmatically when importing run_one_window.
"""

import argparse
import datetime
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from config import Config
from entrance import RLcontroller


HYPERPARAM_KEYS = [
    "lambda_1",
    "lambda_2",
    "lambda_tc",
    "lambda_change",
    "entropy_coef",
    "action_noise_sigma",
    "controller_reg_lambda",
    "controller_observer_bias_weight",
    "num_epochs",
]


def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward/sliding-window training")
    p.add_argument("--start-date", required=True, help="Start date for first window (YYYY-MM-DD)")
    p.add_argument("--num-windows", type=int, default=1, help="Number of walk-forward windows")
    p.add_argument("--train-years", type=int, default=3, help="Years for training window")
    p.add_argument("--valid-years", type=int, default=1, help="Years for validation window")
    p.add_argument("--test-years", type=int, default=1, help="Years for test window")
    p.add_argument("--step-years", type=int, default=1, help="How many years to slide the window each iteration")
    p.add_argument("--num-seeds", type=int, default=1, help="Number of seeds per window")
    p.add_argument("--base-seed", type=int, default=2025, help="Base seed; seeds are base + i")
    p.add_argument(
        "--resume-overlap",
        action="store_true",
        default=True,
        help=(
            "Reuse the previous window's checkpoint (and replay buffer) even when windows overlap. "
            "Safe as long as buffers only contain train data—not valid/test or future timesteps."
        ),
    )
    p.add_argument(
        "--no-resume-overlap",
        dest="resume_overlap",
        action="store_false",
        help="Disable chaining when windows overlap.",
    )
    return p.parse_args()


def make_seed_list(num_seeds: int, base_seed: int) -> List[int]:
    return [base_seed + i for i in range(num_seeds)]


def find_checkpoint(res_dir: str) -> Optional[str]:
    """Pick a checkpoint to resume from: prefer checkpoint_best_valid, else checkpoint_final."""
    candidates = [
        os.path.join(res_dir, "checkpoints", "checkpoint_best_valid", "checkpoint_info.json"),
        os.path.join(res_dir, "checkpoints", "checkpoint_final", "checkpoint_info.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def apply_hparam_overrides(cfg: Config, overrides: Optional[Dict[str, Any]] = None) -> None:
    """Apply a small, whitelisted set of hyperparameter overrides to the Config."""
    if not overrides:
        return
    applied = {}
    for key, value in overrides.items():
        if key in HYPERPARAM_KEYS and hasattr(cfg, key):
            setattr(cfg, key, value)
            applied[key] = value
    if applied:
        print(f"[WF] Applied overrides: {applied}", flush=True)
    ignored = set(overrides.keys()) - set(applied.keys())
    if ignored:
        print(f"[WF] Ignored overrides (not whitelisted or missing on Config): {sorted(ignored)}", flush=True)


def run_one_window(
    window_idx: int,
    seed: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    resume_checkpoint: Optional[str] = None,
    hparam_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    tag = f"wf_win{window_idx}_seed{seed}"
    cfg = Config(seed_num=seed, current_date=tag)
    apply_hparam_overrides(cfg, hparam_overrides)

    # Override date splits
    cfg.train_date_start = train_start
    cfg.train_date_end = train_end
    cfg.valid_date_start = valid_start
    cfg.valid_date_end = valid_end
    cfg.test_date_start = test_start
    cfg.test_date_end = test_end
    # Do not auto-resume from other checkpoints unless explicitly passed
    cfg.auto_resume_from_latest = False
    cfg.resume_from_checkpoint = resume_checkpoint
    if resume_checkpoint:
        print(f"[WF] Resuming from checkpoint: {resume_checkpoint}", flush=True)

    # Refresh risk metrics that depend on train_end
    cfg.risk_market = cfg._compute_market_risk(cutoff_date=cfg.train_date_end)
    cfg._calibrate_risk_bounds()

    print(
        f"[WF] Window {window_idx} seed={seed} | "
        f"train {train_start.date()}→{train_end.date()} | "
        f"valid {valid_start.date()}→{valid_end.date()} | "
        f"test {test_start.date()}→{test_end.date()} | "
        f"res_dir={cfg.res_dir}",
        flush=True,
    )
    RLcontroller(cfg)
    next_checkpoint = find_checkpoint(cfg.res_dir)
    if next_checkpoint:
        print(f"[WF] Next resume checkpoint: {next_checkpoint}", flush=True)

    # Collect test metrics
    test_profile_path = os.path.join(cfg.res_dir, "test_profile.csv")
    if not os.path.exists(test_profile_path):
        raise FileNotFoundError(f"Missing test_profile.csv at {test_profile_path}")
    df = pd.read_csv(test_profile_path)
    if df.empty:
        raise ValueError(f"Empty test_profile.csv at {test_profile_path}")
    row = df.iloc[-1]
    metrics = {
        "window": window_idx,
        "seed": seed,
        "res_dir": cfg.res_dir,
        "sharpeRatio": row.get("sharpeRatio", np.nan),
        "mdd": row.get("mdd", np.nan),
        "annualReturn_pct": row.get("annualReturn_pct", np.nan),
        "netProfit_pct": row.get("netProfit_pct", np.nan),
        "final_capital": row.get("final_capital", np.nan),
        "reward_sum": row.get("reward_sum", np.nan),
        "next_checkpoint": next_checkpoint,
    }
    return metrics


def main():
    args = parse_args()
    start_date = pd.Timestamp(args.start_date)
    train_years = datetime.timedelta(days=365 * args.train_years)
    valid_years = datetime.timedelta(days=365 * args.valid_years)
    test_years = datetime.timedelta(days=365 * args.test_years)
    step_years = datetime.timedelta(days=365 * args.step_years)

    seeds = make_seed_list(args.num_seeds, args.base_seed)
    all_metrics: List[Dict[str, float]] = []
    # Track per-seed checkpoint for chaining windows (only when non-overlapping)
    last_ckpt_by_seed: Dict[int, Optional[str]] = {s: None for s in seeds}
    last_window_end_by_seed: Dict[int, Optional[pd.Timestamp]] = {s: None for s in seeds}

    for win in range(args.num_windows):
        train_start = start_date + win * step_years
        train_end = train_start + train_years - datetime.timedelta(days=1)
        valid_start = train_end + datetime.timedelta(days=1)
        valid_end = valid_start + valid_years - datetime.timedelta(days=1)
        test_start = valid_end + datetime.timedelta(days=1)
        test_end = test_start + test_years - datetime.timedelta(days=1)

        for seed in seeds:
            resume_ckpt = None
            prev_end = last_window_end_by_seed.get(seed)
            prev_ckpt = last_ckpt_by_seed.get(seed)
            # Default: chain checkpoint forward even when windows overlap (buffer is reset on resume by default).
            if prev_ckpt:
                if args.resume_overlap:
                    resume_ckpt = prev_ckpt
                elif prev_end is not None and train_start > prev_end:
                    resume_ckpt = prev_ckpt
            m = run_one_window(
                window_idx=win,
                seed=seed,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
                resume_checkpoint=resume_ckpt,
                hparam_overrides=None,
            )
            last_ckpt_by_seed[seed] = m.get("next_checkpoint")
            last_window_end_by_seed[seed] = test_end
            all_metrics.append(m)

    # Aggregate and print summary
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        numeric_cols = [
            c
            for c in df.columns
            if c not in ("res_dir", "window", "seed", "next_checkpoint")
        ]
        summary = df.groupby("window")[numeric_cols].agg(["mean", "std"])
        last_win = df["window"].max()
        df_last = df[df["window"] == last_win]
        summary_last = (
            df_last[[c for c in numeric_cols if c not in ("window", "seed")]]
            .agg(["mean", "std"])
            .T
        )
        print("\n[WF] Per-window test metrics:")
        print(df.to_string(index=False))
        print("\n[WF] Summary (mean/std per window):")
        print(summary)
        print(f"\n[WF] Last window only (window={last_win}) mean/std across seeds:")
        print(summary_last)


if __name__ == "__main__":
    main()
