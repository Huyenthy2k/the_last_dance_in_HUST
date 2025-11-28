"""
Run repeated TD3+MAFIA training/validation/test across multiple random seeds,
log artifacts per run, and report averaged test metrics (printed + saved to CSV).

Usage examples:
- python agents/MAFIA/scripts/run_multi_seed_experiment.py
- python agents/MAFIA/scripts/run_multi_seed_experiment.py --num-seeds 5 --base-seed 2026
- python agents/MAFIA/scripts/run_multi_seed_experiment.py --seeds 2023,42,99
- python agents/MAFIA/scripts/run_multi_seed_experiment.py --output-dir ./res/multi_seed_logs
"""

import argparse
import datetime
import os
import random
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Ensure repo root on path so we can import entrance/Config when run from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from entrance import RLcontroller
from config import Config


def set_all_seeds(seed: int):
    os.environ["MAFIA_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(seed: int) -> Dict[str, float]:
    set_all_seeds(seed)
    cur_ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg = Config(seed_num=seed, current_date=f"{cur_ts}-seed{seed}")

    # Keep default checkpoint/resume behavior from Config (auto-resume on)
    cfg.reward_debug_steps = 0

    print(f"[MULTI-SEED] Starting run with seed={seed}, res_dir={cfg.res_dir}", flush=True)
    RLcontroller(cfg)

    # Collect test metrics (last row of test_profile.csv)
    test_profile_path = os.path.join(cfg.res_dir, "test_profile.csv")
    if not os.path.exists(test_profile_path):
        raise FileNotFoundError(f"Missing test_profile.csv at {test_profile_path}")
    df = pd.read_csv(test_profile_path)
    if df.empty:
        raise ValueError(f"Empty test_profile.csv at {test_profile_path}")
    row = df.iloc[-1]
    metrics = {
        "res_dir": cfg.res_dir,
        "seed": seed,
        "sharpeRatio": row.get("sharpeRatio", np.nan),
        "mdd": row.get("mdd", np.nan),
        "annualReturn_pct": row.get("annualReturn_pct", np.nan),
        "netProfit_pct": row.get("netProfit_pct", np.nan),
        "final_capital": row.get("final_capital", np.nan),
    }
    print(f"[MULTI-SEED] Finished seed={seed} | Sharpe={metrics['sharpeRatio']} | MDD={metrics['mdd']} | final_capital={metrics['final_capital']}", flush=True)
    return metrics


def aggregate(results: List[Dict[str, float]]):
    if not results:
        return None
    df = pd.DataFrame(results)
    numeric_cols = [c for c in df.columns if c not in ("res_dir",)]
    summary = df[numeric_cols].agg(["mean", "std"]).T
    return df, summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-seeds", type=int, default=10, help="Number of runs (ignored if --seeds is provided)")
    p.add_argument("--base-seed", type=int, default=2025, help="Base seed; seeds will be base_seed + i")
    p.add_argument("--seeds", type=str, default="", help="Comma-separated list of seeds to run (overrides num/base)")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save summary CSVs (default: <MAFIA_RES_ROOT or ./res>/multi_seed_logs/mseed_<ts>)",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Custom tag for output folder name (default: timestamp)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.base_seed + i for i in range(args.num_seeds)]

    print(f"[MULTI-SEED] Running seeds: {seeds}", flush=True)
    results = []
    for seed in seeds:
        metrics = run_one(seed)
        results.append(metrics)

    df, summary = aggregate(results)
    print("\n[MULTI-SEED] Per-run test metrics:")
    print(df.to_string(index=False))
    print("\n[MULTI-SEED] Summary (mean/std):")
    print(summary)

    # Persist summary to CSV
    ts = args.run_tag or datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    res_root = os.path.abspath(os.environ.get("MAFIA_RES_ROOT", os.path.join(REPO_ROOT, "res")))
    default_dir = os.path.join(res_root, "multi_seed_logs", f"mseed_{ts}")
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else default_dir
    os.makedirs(out_dir, exist_ok=True)
    runs_csv = os.path.join(out_dir, "runs.csv")
    summary_csv = os.path.join(out_dir, "summary.csv")
    info_json = os.path.join(out_dir, "run_info.json")
    df.to_csv(runs_csv, index=False)
    summary.to_csv(summary_csv)
    with open(info_json, "w") as f:
        json.dump(
            {
                "seeds": seeds,
                "num_seeds": len(seeds),
                "base_seed": args.base_seed,
                "runs_csv": runs_csv,
                "summary_csv": summary_csv,
                "timestamp": ts,
            },
            f,
            indent=2,
        )
    print(f"\n[MULTI-SEED] Saved per-run metrics -> {runs_csv}")
    print(f"[MULTI-SEED] Saved summary        -> {summary_csv}")
    print(f"[MULTI-SEED] Run info             -> {info_json}")


if __name__ == "__main__":
    main()
