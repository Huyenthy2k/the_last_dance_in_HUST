"""
Lightweight Optuna search for RL hyperparameters.

Runs short mini-runs (default 2 epochs) and scores by validation Sharpe.
Adjust the search space or objective as needed.
"""

import datetime
import os
import sys

import optuna
import pandas as pd

# Ensure repo root is on sys.path so we can import entrance/config when run from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from entrance import RLcontroller
from config import Config


def run_one_trial(trial, mini_epochs=5):
    # Build config for this trial (use unique timestamp to separate artifacts)
    cur_ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg = Config(current_date=f"{cur_ts}-trial{trial.number}")

    # Do not resume checkpoints during hyperparam trials
    cfg.auto_resume_from_latest = False
    cfg.resume_from_checkpoint = None

    # Hyperparameters to tune
    # Reward weights: broaden search to rebalance penalties vs profit
    cfg.lambda_1 = trial.suggest_loguniform("lambda_1", 100.0, 1000.0)
    cfg.lambda_2 = trial.suggest_loguniform("lambda_2", 1.0, 50.0)
    cfg.lambda_tc = trial.suggest_uniform("lambda_tc", 1.0, 15.0)  # turnover is raw sum |w_t - w_{t-1}|
    cfg.lambda_change = trial.suggest_uniform("lambda_change", 5.0, 30.0)
    cfg.entropy_coef = trial.suggest_loguniform("entropy_coef", 5e-4, 3e-3)
    cfg.action_noise_sigma = trial.suggest_uniform("action_noise_sigma", 0.1, 0.2)

    # Make the run lightweight
    cfg.num_epochs = mini_epochs
    cfg.checkpoint_freq = 0
    cfg.validation_freq = 1
    cfg.enable_topk_postprocess = False
    cfg.save_replay_buffer_on_epoch_checkpoints = False
    cfg.save_replay_buffer_on_step_checkpoints = False
    # Skip best-valid checkpointing / early-stop to avoid extra I/O during search
    cfg.early_stop_patience = None

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

    # Read validation Sharpe from metrics_history
    metrics_path = cfg.metrics_history_path
    if not os.path.exists(metrics_path):
        raise RuntimeError(f"metrics_history not found at {metrics_path}")

    df = pd.read_csv(metrics_path)
    valid_rows = df[df["phase"] == "valid"]
    if valid_rows.empty:
        raise RuntimeError("No validation rows found in metrics_history")

    sharpe_val = valid_rows["sharpeRatio"].iloc[-1]
    mdd_val = valid_rows["mdd"].iloc[-1]

    # Example score: Sharpe minus soft penalty for MDD>0.
    score = sharpe_val - 0.5 * max(0.0, mdd_val - 0.2)
    return score


def objective(trial):
    return run_one_trial(trial, mini_epochs=5)


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
    study.optimize(objective, n_trials=200, n_jobs=2, callbacks=[_early_stop])
    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
