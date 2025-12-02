"""
Lightweight Optuna search for RL hyperparameters.

Runs short mini-runs (default 10 epochs) and scores by validation Sharpe.
Adjust the search space or objective as needed.
"""

import datetime
import os
import sys
import json

import optuna
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path so we can import entrance/config when run from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from entrance import RLcontroller
from config import Config


def run_one_trial(trial, mini_epochs=10):
    # Build config for this trial (use unique timestamp to separate artifacts)
    cur_ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg = Config(current_date=f"{cur_ts}-trial{trial.number}")
    # Hparam trials must start fresh: disable any checkpoint resume
    cfg.auto_resume_from_latest = False
    cfg.resume_from_checkpoint = None
    # Track artifacts in Optuna user attrs so auto_pipeline can log them
    trial.set_user_attr("res_dir", cfg.res_dir)
    trial.set_user_attr("trial_ts", cfg.cur_datetime)

    # Hyperparameters to tune (scales aligned with reward_sum/Sharpe)
    cfg.lambda_1 = trial.suggest_float("lambda_1", 100.0, 1000.0, log=True)
    cfg.lambda_2 = trial.suggest_float("lambda_2", 10.0, 200.0, log=True)
    cfg.lambda_tc = trial.suggest_float(
        "lambda_tc", 0.0001, 0.1, log=True
    )  # turnover penalty
    cfg.lambda_change = trial.suggest_float(
        "lambda_change", 0.0001, 0.1, log=True
    )  # membership-change penalty
    cfg.entropy_coef = trial.suggest_float("entropy_coef", 5e-4, 3e-3, log=True)
    cfg.action_noise_sigma = trial.suggest_float("action_noise_sigma", 0.01, 0.22)
    cfg.controller_reg_lambda = trial.suggest_float(
        "controller_reg_lambda", 0.05, 5.0, log=True
    )
    cfg.controller_observer_bias_weight = trial.suggest_float(
        "controller_observer_bias_weight", 0.05, 0.8
    )

    # Make the run lightweight
    cfg.num_epochs = mini_epochs
    # Allow checkpointing/resume for search runs (defaults are resume-friendly but controllable via env)
    ckpt_freq = int(os.environ.get("HSEARCH_CHECKPOINT_FREQ", "1"))
    partial_steps = int(os.environ.get("HSEARCH_PARTIAL_STEPS", "0"))
    save_rb_epoch = os.environ.get("HSEARCH_SAVE_RB", "1") not in (
        "0",
        "false",
        "False",
    )
    save_rb_step = os.environ.get("HSEARCH_SAVE_STEP_RB", "0") not in (
        "0",
        "false",
        "False",
    )
    max_ckpt_keep = int(os.environ.get("HSEARCH_MAX_CKPT_KEEP", "1"))

    cfg.checkpoint_freq = ckpt_freq  # save every N epochs (default 1 for resume)
    cfg.partial_checkpoint_steps = partial_steps  # 0 disables step-based checkpoints
    cfg.validation_freq = 1
    cfg.enable_topk_postprocess = False
    cfg.save_replay_buffer_on_epoch_checkpoints = save_rb_epoch
    cfg.save_replay_buffer_on_step_checkpoints = save_rb_step
    cfg.enable_checkpoint_cleanup = True
    cfg.max_checkpoints_to_keep = max_ckpt_keep

    print(
        f"[HSEARCH] Trial {trial.number} window: "
        f"train {cfg.train_date_start.date()}→{cfg.train_date_end.date()}, "
        f"valid {cfg.valid_date_start.date()}→{cfg.valid_date_end.date()}, "
        f"test {cfg.test_date_start.date()}→{cfg.test_date_end.date()}, "
        f"epochs={cfg.num_epochs}",
        flush=True,
    )

    # Run training (includes validation each epoch)
    RLcontroller(cfg)

    # Read validation metrics from metrics_history
    metrics_path = cfg.metrics_history_path
    if not os.path.exists(metrics_path):
        raise RuntimeError(f"metrics_history not found at {metrics_path}")

    df = pd.read_csv(metrics_path)
    valid_rows = df[df["phase"] == "valid"]
    if valid_rows.empty:
        raise RuntimeError("No validation rows found in metrics_history")

    sharpe_val = valid_rows["sharpeRatio"].iloc[-1]
    mdd_val = valid_rows["mdd"].iloc[-1]
    annret_val = (
        valid_rows["annualReturn_pct"].iloc[-1]
        if "annualReturn_pct" in valid_rows
        else 0.0
    )
    reward_sum_val = (
        valid_rows["reward_sum"].iloc[-1] if "reward_sum" in valid_rows else 0.0
    )

    # Composite objective: emphasize Sharpe and reward_sum, downweight drawdown
    score = reward_sum_val + 1.0 * sharpe_val - 0.3 * mdd_val + 0.2 * annret_val
    return score


def objective(trial):
    print(f"[HSEARCH] Trial {trial.number} params sampling...", flush=True)
    val = run_one_trial(trial, mini_epochs=10)
    print(
        f"[HSEARCH] Trial {trial.number} done | score={val} | params={trial.params}",
        flush=True,
    )
    return val


def save_search_outputs(study: optuna.Study, run_ts: str):
    """
    Persist Optuna search results to res/auto_pipeline_logs for visibility,
    matching auto_pipeline's logging convention.
    """
    res_root = os.path.abspath(os.environ.get("MAFIA_RES_ROOT", "./res"))
    log_dir = os.path.join(res_root, "auto_pipeline_logs")
    os.makedirs(log_dir, exist_ok=True)

    trials_csv = os.path.join(log_dir, f"optuna_trials_{run_ts}.csv")
    best_json = os.path.join(log_dir, f"optuna_best_{run_ts}.json")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv, index=False)

    best_payload = {
        "timestamp": run_ts,
        "best_trial": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": {
            k: float(v) if isinstance(v, (float, int, np.floating)) else v
            for k, v in study.best_params.items()
        },
        "n_trials": len(study.trials),
        "trials_csv": trials_csv,
    }
    with open(best_json, "w") as f:
        json.dump(best_payload, f, indent=2)

    print(f"[HSEARCH] 📝 Saved trials to {trials_csv}")
    print(f"[HSEARCH] 🏅 Saved best summary to {best_json}")


def main():
    TARGET_SCORE = 1.5
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    def _early_stop(study, trial):
        if study.best_value is not None and study.best_value >= TARGET_SCORE:
            study.stop()

    # Increase max trials; early-stop when đạt TARGET_SCORE
    study.optimize(objective, n_trials=20, n_jobs=2, callbacks=[_early_stop])
    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # Persist search artifacts to auto_pipeline_logs (same place auto_pipeline uses)
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_search_outputs(study, run_ts)


if __name__ == "__main__":
    main()
