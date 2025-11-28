"""
Automated pipeline:
1) Run Optuna hparam search for a few trials/epochs to get best hyperparams.
2) Use best params to run full training (num_epochs) across multiple seeds.
3) Aggregate test metrics (mean/std) over seeds.
4) (Optional) Run walk-forward sliding-window training with the best params; can reuse
   replay buffer across windows when resume_overlap is enabled.

Usage:
  python agents/MAFIA/scripts/auto_pipeline.py \
    --hparam-trials 10 --hparam-epochs 5 \
    --train-epochs 50 --num-seeds 10 --base-seed 2025

Optional:
  --seeds 1,2,3              # explicit seeds
  --hparam-trials 10          # more search trials (default)
  --walkforward-start-date 2017-01-01 --walkforward-num-windows 3 --walkforward-resume-overlap (default on)
"""

import argparse
import datetime
import json
import os
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from config import Config
from entrance import RLcontroller
from scripts.hparam_search_optuna import run_one_trial
from walk_forward import run_one_window as wf_run_one_window


def set_all_seeds(seed: int):
    os.environ["MAFIA_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_hparam_search(
    n_trials: int, mini_epochs: int, target_score: float | None = None
) -> Tuple[Dict[str, float], str, Dict[str, str]]:
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
    print(
        f"[AUTO] 🔍 Starting Optuna search | trials={n_trials}, mini_epochs={mini_epochs}"
    )
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
    best_trial = study.best_trial
    best_trial_info = {
        "res_dir": best_trial.user_attrs.get("res_dir"),
        "trial_ts": best_trial.user_attrs.get("trial_ts"),
        "trial_number": best_trial.number,
    }
    print(
        f"[AUTO] Best trial #{best_trial.number} res_dir: {best_trial_info['res_dir']}"
    )

    # Persist search results for later inspection
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv, index=False)
    best_payload = {
        "timestamp": run_ts,
        "hparam_trials": n_trials,
        "mini_epochs": mini_epochs,
        "best_value": float(study.best_value),
        "best_params": {
            k: float(v) if isinstance(v, (float, int, np.floating)) else v
            for k, v in study.best_params.items()
        },
        "trials_csv": trials_csv,
        "log_dir": log_run_dir,
        "best_trial": best_trial_info,
    }
    with open(best_json, "w") as f:
        json.dump(best_payload, f, indent=2)
    print(f"[AUTO] 📝 Saved hparam trials to {trials_csv}")
    print(f"[AUTO] 🏅 Saved best params to {best_json}")
    return study.best_params, log_run_dir, best_trial_info


def run_one_seed(
    seed: int, best_params: Dict[str, float], train_epochs: int
) -> Dict[str, float]:
    set_all_seeds(seed)
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg = Config(seed_num=seed, current_date=f"{ts}-seed{seed}")
    # Apply best hyperparameters
    for key in [
        "lambda_1",
        "lambda_2",
        "lambda_tc",
        "lambda_change",
        "entropy_coef",
        "action_noise_sigma",
        "controller_reg_lambda",
        "controller_observer_bias_weight",
    ]:
        if key in best_params:
            setattr(cfg, key, best_params[key])

    # Training config
    cfg.num_epochs = train_epochs
    cfg.auto_resume_from_latest = True  # enable auto-resume from latest checkpoint
    cfg.resume_from_checkpoint = None
    cfg.early_stop_patience = 0  # avoid early stop during multi-seed run
    cfg.reward_debug_steps = 0

    print("\n" + "-" * 80)
    print(
        f"[AUTO] 🚀 Training seed={seed} | epochs={train_epochs} | res_dir={cfg.res_dir}",
        flush=True,
    )
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
    print(
        f"[AUTO] ✅ Done seed={seed} | Sharpe={metrics['sharpeRatio']} | MDD={metrics['mdd']} | final_capital={metrics['final_capital']}",
        flush=True,
    )
    return metrics


def aggregate_results(results: List[Dict[str, float]]):
    df = pd.DataFrame(results)
    numeric_cols = [c for c in df.columns if c not in ("res_dir",)]
    summary = df[numeric_cols].agg(["mean", "std"]).T
    return df, summary


def run_walk_forward_stage(
    start_date: str,
    num_windows: int,
    train_years: int,
    valid_years: int,
    test_years: int,
    step_years: int,
    end_date: Optional[str],
    seeds: List[int],
    hparam_overrides: Optional[Dict[str, float]],
    resume_overlap: bool,
    log_dir: str,
):
    print("\n" + "=" * 80)
    print(
        f"[AUTO] 🧊 Walk-forward stage | start={start_date} | windows={num_windows} | "
        f"train/valid/test={train_years}/{valid_years}/{test_years} yrs | step={step_years} yrs | "
        f"resume_overlap={resume_overlap}"
    )
    if resume_overlap:
        print(
            "[AUTO]    resume_overlap=1 -> replay buffer/ckpt will be reused even when windows overlap "
            "(assumes buffers only contain train data, not valid/test or future timesteps)."
        )
    print("=" * 80, flush=True)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else None
    train_delta = datetime.timedelta(days=365 * train_years)
    valid_delta = datetime.timedelta(days=365 * valid_years)
    test_delta = datetime.timedelta(days=365 * test_years)
    step_delta = datetime.timedelta(days=365 * step_years)
    if num_windows <= 0 and end_ts is not None:
        num_windows = walk_forward.compute_max_windows(
            start_date=start_ts,
            end_date=end_ts,
            train_delta=train_delta,
            valid_delta=valid_delta,
            test_delta=test_delta,
            step_delta=step_delta,
        )
        print(
            f"[AUTO] 🧊 Auto-computed num_windows={num_windows} until {end_ts.date()}",
            flush=True,
        )

    last_ckpt_by_seed: Dict[int, Optional[str]] = {s: None for s in seeds}
    last_window_end_by_seed: Dict[int, Optional[pd.Timestamp]] = {
        s: None for s in seeds
    }
    all_metrics: List[Dict[str, float]] = []

    for win in range(num_windows):
        train_start = start_ts + win * step_delta
        train_end = train_start + train_delta - datetime.timedelta(days=1)
        valid_start = train_end + datetime.timedelta(days=1)
        valid_end = valid_start + valid_delta - datetime.timedelta(days=1)
        test_start = valid_end + datetime.timedelta(days=1)
        test_end = test_start + test_delta - datetime.timedelta(days=1)

        print(
            f"[AUTO][WF] Window {win} dates | "
            f"train {train_start.date()}→{train_end.date()} | "
            f"valid {valid_start.date()}→{valid_end.date()} | "
            f"test {test_start.date()}→{test_end.date()}",
            flush=True,
        )

        for seed in seeds:
            resume_ckpt = None
            prev_end = last_window_end_by_seed.get(seed)
            prev_ckpt = last_ckpt_by_seed.get(seed)
            if prev_ckpt:
                if resume_overlap:
                    resume_ckpt = prev_ckpt
                elif prev_end is not None and train_start > prev_end:
                    resume_ckpt = prev_ckpt

            m = wf_run_one_window(
                window_idx=win,
                seed=seed,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
                resume_checkpoint=resume_ckpt,
                hparam_overrides=hparam_overrides,
            )
            last_ckpt_by_seed[seed] = m.get("next_checkpoint")
            last_window_end_by_seed[seed] = test_end
            all_metrics.append(m)

    df = pd.DataFrame(all_metrics)
    if df.empty:
        summary = pd.DataFrame()
        metrics_csv = os.path.join(log_dir, "walk_forward_metrics.csv")
        summary_csv = os.path.join(log_dir, "walk_forward_summary.csv")
        os.makedirs(log_dir, exist_ok=True)
        df.to_csv(metrics_csv, index=False)
        summary.to_csv(summary_csv)
        print("[AUTO] Walk-forward stage produced no metrics (empty DataFrame).", flush=True)
        return df, summary, metrics_csv, summary_csv

    numeric_cols = [
        c for c in df.columns if c not in ("res_dir", "window", "seed", "next_checkpoint")
    ]
    summary = df.groupby("window")[numeric_cols].agg(["mean", "std"])
    last_win = df["window"].max()
    df_last = df[df["window"] == last_win]
    summary_last = (
        df_last[[c for c in numeric_cols if c not in ("window", "seed")]]
        .agg(["mean", "std"])
        .T
    )

    os.makedirs(log_dir, exist_ok=True)
    metrics_csv = os.path.join(log_dir, "walk_forward_metrics.csv")
    summary_csv = os.path.join(log_dir, "walk_forward_summary.csv")
    last_summary_csv = os.path.join(log_dir, "walk_forward_last_window_summary.csv")
    df.to_csv(metrics_csv, index=False)
    summary.to_csv(summary_csv)
    summary_last.to_csv(last_summary_csv)

    print("\n[AUTO] 🧊 Walk-forward per-window metrics:")
    print(df.to_string(index=False))
    print("\n[AUTO] 🧊 Walk-forward summary (mean/std per window):")
    print(summary)
    print(f"\n[AUTO] 🧊 Last window only (window={last_win}) mean/std across seeds:")
    print(summary_last)
    return df, summary, metrics_csv, summary_csv, last_summary_csv


def copy_checkpoint(
    trial_dir: str,
    checkpoint_name: str,
    tag: str,
    reset_counters: bool = True,
):
    trial_dir = os.path.abspath(trial_dir)
    base_dir = os.path.dirname(trial_dir)  # e.g., .../VNINDEX-10
    dest_run_dir = os.path.join(base_dir, tag)

    src_ckpt_dir = os.path.join(trial_dir, "checkpoints", checkpoint_name)
    if not os.path.exists(src_ckpt_dir):
        raise FileNotFoundError(f"Checkpoint folder not found: {src_ckpt_dir}")

    dest_ckpt_dir = os.path.join(dest_run_dir, "checkpoints", checkpoint_name)
    os.makedirs(os.path.dirname(dest_ckpt_dir), exist_ok=True)
    if os.path.exists(dest_ckpt_dir):
        shutil.rmtree(dest_ckpt_dir)
    shutil.copytree(src_ckpt_dir, dest_ckpt_dir)

    info_path = os.path.join(dest_ckpt_dir, "checkpoint_info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    source_info_path = os.path.join(src_ckpt_dir, "checkpoint_info.json")
    for key in [
        "rl_model_path",
        "mafia_observer_path",
        "replay_buffer_path",
        "env_state_path",
        "rng_state_path",
    ]:
        if key in info and info[key]:
            info[key] = os.path.join(dest_ckpt_dir, os.path.basename(info[key]))
    info["type"] = f"{info.get('type', 'epoch_best_valid')}_fork"
    info["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info["source_checkpoint"] = source_info_path
    if reset_counters:
        info["epoch"] = 0
        info["day_in_epoch"] = 0
        info["timesteps"] = 0

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    for sub in ["graph", "model"]:
        os.makedirs(os.path.join(dest_run_dir, sub), exist_ok=True)

    print(f"[AUTO] Forked checkpoint -> {dest_ckpt_dir}")
    return info_path, info, dest_run_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hparam-trials", type=int, default=10, help="Optuna trials")
    p.add_argument(
        "--hparam-epochs", type=int, default=5, help="Epochs per hparam trial"
    )
    p.add_argument(
        "--hparam-target",
        type=float,
        default=None,
        help="Early stop hparam search when best score >= target (None to disable)",
    )
    p.add_argument(
        "--train-epochs",
        type=int,
        default=50,
        help="Epochs for full training per seed",
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seeds to run (ignored if --seeds provided)",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=2025,
        help="Base seed; seeds will be base_seed + i",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds (overrides num/base)",
    )
    p.add_argument(
        "--fork-best-trial",
        action="store_true",
        default=True,
        help="Fork best hparam trial checkpoint into an official run (default: on)",
    )
    p.add_argument(
        "--no-fork-best-trial",
        dest="fork_best_trial",
        action="store_false",
        help="Disable forking best trial checkpoint",
    )
    p.add_argument(
        "--official-tag",
        type=str,
        default="official-epoch1",
        help="Tag/name for the official run directory",
    )
    p.add_argument(
        "--official-epochs",
        type=int,
        default=None,
        help="Epochs to train for the official fork (defaults to --train-epochs)",
    )
    p.add_argument(
        "--official-checkpoint-name",
        type=str,
        default="checkpoint_best_valid",
        help="Checkpoint folder name to fork",
    )
    p.add_argument(
        "--official-seed",
        type=int,
        default=2022,
        help="Seed for the official forked run",
    )
    p.add_argument(
        "--official-keep-counters",
        action="store_true",
        help="Keep epoch/timestep counters from checkpoint instead of resetting to 0",
    )
    p.add_argument(
        "--walkforward-start-date",
        type=str,
        default="2017-01-01",
        help="Enable walk-forward stage starting at this date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--walkforward-end-date",
        type=str,
        default="2023-12-31",
        help="End date (inclusive) for auto window computation (num-windows<=0).",
    )
    p.add_argument(
        "--walkforward-num-windows",
        type=int,
        default=0,
        help="Number of walk-forward windows (<=0 => auto-compute until end-date; 0 will still enable walk-forward).",
    )
    p.add_argument(
        "--walkforward-train-years",
        type=int,
        default=3,
        help="Years for training window in walk-forward stage",
    )
    p.add_argument(
        "--walkforward-valid-years",
        type=int,
        default=1,
        help="Years for validation window in walk-forward stage",
    )
    p.add_argument(
        "--walkforward-test-years",
        type=int,
        default=1,
        help="Years for test window in walk-forward stage",
    )
    p.add_argument(
        "--walkforward-step-years",
        type=int,
        default=1,
        help="Slide step (years) between walk-forward windows",
    )
    p.add_argument(
        "--walkforward-epochs",
        type=int,
        default=50,
        help="Epochs per walk-forward window (default: use --train-epochs)",
    )
    p.add_argument(
        "--walkforward-resume-overlap",
        action="store_true",
        default=True,
        help="Reuse checkpoint even when windows overlap (replay buffer is reset by default on resume).",
    )
    p.add_argument(
        "--no-walkforward-resume-overlap",
        dest="walkforward_resume_overlap",
        action="store_false",
        help="Disable chaining when walk-forward windows overlap.",
    )
    p.add_argument(
        "--skip-seed-stage",
        action="store_true",
        help="Skip the per-seed training stage (useful if only running walk-forward).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("[AUTO] PIPELINE START")
    print("=" * 80)
    # 1) Hyperparam search
    best_params, log_run_dir, best_trial_info = run_hparam_search(
        n_trials=args.hparam_trials,
        mini_epochs=args.hparam_epochs,
        target_score=args.hparam_target,
    )

    # 1b) Optional: fork best trial checkpoint and run official training
    official_run_dir = None
    if args.fork_best_trial:
        official_epochs = args.official_epochs or args.train_epochs
        cfg = Config(seed_num=args.official_seed, current_date=args.official_tag)
        for key in [
            "lambda_1",
            "lambda_2",
            "lambda_tc",
            "lambda_change",
            "entropy_coef",
            "action_noise_sigma",
            "controller_reg_lambda",
            "controller_observer_bias_weight",
        ]:
            if key in best_params:
                setattr(cfg, key, best_params[key])
        cfg.resume_from_checkpoint = None  # Fresh run; no checkpoint copy
        cfg.auto_resume_from_latest = True  # allow resume within official run directory
        cfg.reward_debug_steps = 0
        cfg.num_epochs = official_epochs
        official_run_dir = cfg.res_dir
        print("\n" + "=" * 80)
        print(f"[AUTO] 🏁 Official run (fresh) with best hparams")
        print(f"[AUTO]    tag: {args.official_tag}")
        print(f"[AUTO]    epochs: {official_epochs}")
        print("=" * 80, flush=True)
        RLcontroller(cfg)

    # 2) Prepare seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.base_seed + i for i in range(args.num_seeds)]
    print("\n" + "=" * 80)
    print(f"[AUTO] 🌱 Running full training for seeds: {seeds}")
    print("=" * 80, flush=True)

    # 3) Run per-seed trainings (optional)
    results: List[Dict[str, float]] = []
    results_csv = None
    summary_csv = None
    if args.skip_seed_stage:
        print("[AUTO] ⏭️ Skipping per-seed stage (--skip-seed-stage).")
        df = pd.DataFrame()
        summary = pd.DataFrame()
    else:
        for seed in seeds:
            metrics = run_one_seed(seed, best_params, train_epochs=args.train_epochs)
            results.append(metrics)

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
        print(f"[AUTO] 📝 Saved seed results & summary to {log_run_dir}")

    # 4) Optional walk-forward stage
    wf_metrics_csv = None
    wf_summary_csv = None
    wf_last_summary_csv = None
    wf_df = pd.DataFrame()
    wf_summary = pd.DataFrame()
    run_walkforward = (
        args.walkforward_start_date is not None and args.walkforward_num_windows > 0
    )
    if run_walkforward:
        wf_epochs = args.walkforward_epochs or args.train_epochs
        wf_overrides = dict(best_params)
        wf_overrides["num_epochs"] = wf_epochs
        (
            wf_df,
            wf_summary,
            wf_metrics_csv,
            wf_summary_csv,
            wf_last_summary_csv,
        ) = run_walk_forward_stage(
            start_date=args.walkforward_start_date,
            num_windows=args.walkforward_num_windows,
            train_years=args.walkforward_train_years,
            valid_years=args.walkforward_valid_years,
            test_years=args.walkforward_test_years,
            step_years=args.walkforward_step_years,
            end_date=args.walkforward_end_date,
            seeds=seeds,
            hparam_overrides=wf_overrides,
            resume_overlap=args.walkforward_resume_overlap,
            log_dir=log_run_dir,
        )
    else:
        print(
            "[AUTO] Walk-forward stage skipped "
            "(set --walkforward-start-date and --walkforward-num-windows>0 to enable)."
        )

    # Persist run info
    os.makedirs(log_run_dir, exist_ok=True)
    with open(os.path.join(log_run_dir, "run_info.json"), "w") as f:
        json.dump(
            {
                "seeds": seeds,
                "train_epochs": args.train_epochs,
                "seed_stage_skipped": args.skip_seed_stage,
                "hparam_trials": args.hparam_trials,
                "hparam_epochs": args.hparam_epochs,
                "best_params": best_params,
                "best_trial": best_trial_info,
                "seed_results_csv": results_csv,
                "seed_summary_csv": summary_csv,
                "walkforward": {
                    "enabled": run_walkforward,
                    "start_date": args.walkforward_start_date,
                    "num_windows": args.walkforward_num_windows,
                    "train_years": args.walkforward_train_years,
                    "valid_years": args.walkforward_valid_years,
                    "test_years": args.walkforward_test_years,
                    "step_years": args.walkforward_step_years,
                    "epochs": args.walkforward_epochs or args.train_epochs,
                    "resume_overlap": args.walkforward_resume_overlap,
                    "metrics_csv": wf_metrics_csv,
                    "summary_csv": wf_summary_csv,
                    "last_window_summary_csv": wf_last_summary_csv,
                },
                "official_run_dir": official_run_dir,
            },
            f,
            indent=2,
        )
    print("[AUTO] PIPELINE DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
