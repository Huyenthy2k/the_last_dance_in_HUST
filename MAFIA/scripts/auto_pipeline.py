"""
Automated pipeline:
1) Run Optuna hparam search for a few trials/epochs to get best hyperparams.
2) Use best params to run full training (num_epochs) across multiple seeds.
3) Aggregate test metrics (mean/std) over seeds.

Usage:
  python agents/MAFIA/scripts/auto_pipeline.py \
    --hparam-trials 5 --hparam-epochs 5 \
    --train-epochs 100 --num-seeds 10 --base-seed 2025

Optional:
  --seeds 1,2,3              # explicit seeds
  --hparam-trials 10          # more search trials
"""

import argparse
import datetime
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from config import Config
from entrance import RLcontroller
from scripts.hparam_search_optuna import run_one_trial


def set_all_seeds(seed: int):
    os.environ["MAFIA_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_hparam_search(n_trials: int, mini_epochs: int, target_score: float | None = None) -> Tuple[Dict[str, float], str]:
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Use repo-level res as default to keep logs in a consistent location
    default_res_root = os.path.join(REPO_ROOT, "res")
    res_root = os.path.abspath(os.environ.get("MAFIA_RES_ROOT", default_res_root))
    log_root = os.path.join(res_root, "auto_pipeline_logs")
    log_run_dir = os.path.join(log_root, f"run_{run_ts}")
    os.makedirs(log_run_dir, exist_ok=True)
    trials_csv = os.path.join(log_run_dir, f"optuna_trials_{run_ts}.csv")
    best_json = os.path.join(log_run_dir, f"optuna_best_{run_ts}.json")

    print("\n" + "=" * 80)
    print(f"[AUTO] 🔍 Starting Optuna search | trials={n_trials}, mini_epochs={mini_epochs}")
    print("=" * 80)
    def objective(trial):
        return run_one_trial(trial, mini_epochs=mini_epochs)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    callbacks = []
    if target_score is not None:
        def _early_stop(study, trial):
            if study.best_value is not None and study.best_value >= target_score:
                study.stop()
        callbacks.append(_early_stop)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=callbacks)
    print("[AUTO] ✅ Optuna search done")
    print(f"[AUTO] Best score: {study.best_value}")
    print("[AUTO] Best params:")
    for k, v in study.best_params.items():
        print(f"   - {k}: {v}")

    # Persist search results for later inspection
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv, index=False)
    best_payload = {
        "timestamp": run_ts,
        "hparam_trials": n_trials,
        "mini_epochs": mini_epochs,
        "best_value": float(study.best_value),
        "best_params": {k: float(v) if isinstance(v, (float, int, np.floating)) else v for k, v in study.best_params.items()},
        "trials_csv": trials_csv,
        "log_dir": log_run_dir,
    }
    with open(best_json, "w") as f:
        json.dump(best_payload, f, indent=2)
    print(f"[AUTO] 📝 Saved hparam trials to {trials_csv}")
    print(f"[AUTO] 🏅 Saved best params to {best_json}")
    return study.best_params, log_run_dir


def run_one_seed(seed: int, best_params: Dict[str, float], train_epochs: int) -> Dict[str, float]:
    set_all_seeds(seed)
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg = Config(seed_num=seed, current_date=f"{ts}-seed{seed}")
    # Apply best hyperparameters
    for key in ["lambda_1", "lambda_2", "lambda_tc", "lambda_change", "entropy_coef", "action_noise_sigma"]:
        if key in best_params:
            setattr(cfg, key, best_params[key])

    # Training config
    cfg.num_epochs = train_epochs
    cfg.auto_resume_from_latest = True  # enable auto-resume from latest checkpoint
    cfg.resume_from_checkpoint = None
    cfg.early_stop_patience = 0  # avoid early stop during multi-seed run
    cfg.reward_debug_steps = 0

    print("\n" + "-" * 80)
    print(f"[AUTO] 🚀 Training seed={seed} | epochs={train_epochs} | res_dir={cfg.res_dir}", flush=True)
    RLcontroller(cfg)

    test_profile_path = os.path.join(cfg.res_dir, "test_profile.csv")
    if not os.path.exists(test_profile_path):
        raise FileNotFoundError(f"Missing test_profile.csv at {test_profile_path}")
    df = pd.read_csv(test_profile_path)
    if df.empty:
        raise ValueError(f"Empty test_profile.csv at {test_profile_path}")
    row = df.iloc[-1]
    metrics = {
        "seed": seed,
        "res_dir": cfg.res_dir,
        "sharpeRatio": row.get("sharpeRatio", np.nan),
        "mdd": row.get("mdd", np.nan),
        "annualReturn_pct": row.get("annualReturn_pct", np.nan),
        "netProfit_pct": row.get("netProfit_pct", np.nan),
        "final_capital": row.get("final_capital", np.nan),
        "reward_sum": row.get("reward_sum", np.nan),
    }
    print(f"[AUTO] ✅ Done seed={seed} | Sharpe={metrics['sharpeRatio']} | MDD={metrics['mdd']} | final_capital={metrics['final_capital']}", flush=True)
    return metrics


def aggregate_results(results: List[Dict[str, float]]):
    df = pd.DataFrame(results)
    numeric_cols = [c for c in df.columns if c not in ("res_dir",)]
    summary = df[numeric_cols].agg(["mean", "std"]).T
    return df, summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hparam-trials", type=int, default=100, help="Optuna trials")
    p.add_argument("--hparam-epochs", type=int, default=10, help="Epochs per hparam trial")
    p.add_argument("--hparam-target", type=float, default=None, help="Early stop hparam search when best score >= target (None to disable)")
    p.add_argument("--train-epochs", type=int, default=100, help="Epochs for full training per seed")
    p.add_argument("--num-seeds", type=int, default=10, help="Number of seeds to run (ignored if --seeds provided)")
    p.add_argument("--base-seed", type=int, default=2025, help="Base seed; seeds will be base_seed + i")
    p.add_argument("--seeds", type=str, default="", help="Comma-separated list of seeds (overrides num/base)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("[AUTO] PIPELINE START")
    print("=" * 80)
    # 1) Hyperparam search
    best_params, log_run_dir = run_hparam_search(
        n_trials=args.hparam_trials,
        mini_epochs=args.hparam_epochs,
        target_score=args.hparam_target,
    )

    # 2) Prepare seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.base_seed + i for i in range(args.num_seeds)]
    print("\n" + "=" * 80)
    print(f"[AUTO] 🌱 Running full training for seeds: {seeds}")
    print("=" * 80, flush=True)

    # 3) Run per-seed trainings
    results = []
    for seed in seeds:
        metrics = run_one_seed(seed, best_params, train_epochs=args.train_epochs)
        results.append(metrics)

    # 4) Aggregate and report
    df, summary = aggregate_results(results)
    print("\n" + "=" * 80)
    print("[AUTO] 📊 Per-run test metrics:")
    print(df.to_string(index=False))
    print("\n[AUTO] 📈 Summary (mean/std across seeds):")
    print(summary)
    print("=" * 80)
    # Persist aggregate outputs alongside hparam logs
    os.makedirs(log_run_dir, exist_ok=True)
    results_csv = os.path.join(log_run_dir, "seed_results.csv")
    summary_csv = os.path.join(log_run_dir, "seed_summary.csv")
    df.to_csv(results_csv, index=False)
    summary.to_csv(summary_csv)
    with open(os.path.join(log_run_dir, "run_info.json"), "w") as f:
        json.dump(
            {
                "seeds": seeds,
                "train_epochs": args.train_epochs,
                "hparam_trials": args.hparam_trials,
                "hparam_epochs": args.hparam_epochs,
                "best_params": best_params,
                "seed_results_csv": results_csv,
                "seed_summary_csv": summary_csv,
            },
            f,
            indent=2,
        )
    print(f"[AUTO] 📝 Saved seed results & summary to {log_run_dir}")
    print("[AUTO] PIPELINE DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
