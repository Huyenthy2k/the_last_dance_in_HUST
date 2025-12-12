#!/usr/bin/env python3
"""
Walk-Forward Observer Training Pipeline (Expanding Window per Spec §7)

Implements Stage 1 schedule:
    Iter 1: Train 01/2015→06/2017 → Valid 07/2017→12/2017 → Infer 2018 (State_2018)
    Iter 2: Train 01/2015→06/2018 → Valid 07/2018→12/2018 → Infer 2019 (State_2019)
    Iter 3: Train 01/2015→06/2019 → Valid 07/2019→12/2019 → Infer 2020 (State_2020)
    Iter 4: Train 01/2015→06/2020 → Valid 07/2020→12/2020 → Infer 2021 (State_2021)
    Iter 5: Train 01/2015→06/2021 → Valid 07/2021→12/2021 → Infer 2022 (State_2022)

Each iteration: Train/Validate Observer → pick checkpoint with max Sharpe on validation
→ freeze → generate cached RL states for the next year.

Usage:
    python scripts/train_observer_walkforward.py \
        --start-year 2015 \
        --first-infer-year 2018 \
        --last-infer-year 2022 \
        --output-dir ./observer_walkforward

    # Resume from specific iteration
    python scripts/train_observer_walkforward.py \
        --start-year 2015 \
        --first-infer-year 2018 \
        --last-infer-year 2022 \
        --resume-from-iter 2 \
        --checkpoint-path ./observer_walkforward/checkpoints/observer_best_2019.pth
"""

import argparse
import datetime
import copy
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config
from entrance import RLcontroller
from scripts.generate_rl_states import generate_single_year
from scripts.train_observer_offline import run_offline_observer_training

# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import smart_print
except ImportError:
    smart_print = print  # Fallback

# Simple logger for clean terminal output
try:
    from utils.simple_logger import get_simple_logger, init_simple_logger
except ImportError:
    get_simple_logger = None
    init_simple_logger = None


def build_expanding_schedule(
    start_year: int, first_infer_year: int, last_infer_year: int
) -> List[Dict]:
    """
    Build the expanding-window schedule per spec Section 7.

    Train: 01/2015 → 06/(valid_year)
    Valid: 07/(valid_year) → 12/(valid_year)
    Infer: (valid_year + 1)
    """
    infer_years = list(range(first_infer_year, last_infer_year + 1))
    schedule = []
    train_start = pd.Timestamp(f"{start_year}-01-01 00:00:00")

    for idx, infer_year in enumerate(infer_years):
        valid_year = infer_year - 1
        train_end = pd.Timestamp(f"{valid_year}-06-30 23:59:59")
        valid_start = pd.Timestamp(f"{valid_year}-07-01 00:00:00")
        valid_end = pd.Timestamp(f"{valid_year}-12-31 23:59:59")
        infer_start = pd.Timestamp(f"{infer_year}-01-01 00:00:00")
        infer_end = pd.Timestamp(f"{infer_year}-12-31 23:59:59")

        schedule.append(
            {
                "iter_index": idx,
                "iter_display": idx + 1,
                "train_start": train_start,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "valid_year": valid_year,
                "infer_year": infer_year,
                "infer_start": infer_start,
                "infer_end": infer_end,
                "ckpt_name": f"Ckpt_Best_{valid_year}",
                "train_label": f"{train_start.date()} → {train_end.date()}",
                "valid_label": f"{valid_start.date()} → {valid_end.date()}",
            }
        )

    return schedule


def setup_walkforward_config(
    config: Config,
    schedule_entry: Dict,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "./observer_walkforward",
    total_iterations: Optional[int] = None,
) -> Config:
    """
    Configure config for a specific walk-forward iteration.

    Args:
        config: Base config object
        schedule_entry: Dict describing the expanding-window slice
        checkpoint_path: Path to checkpoint for finetuning (iteration > 0)
        output_dir: Base output directory
        total_iterations: Total iterations in the schedule (for display)

    Returns:
        Configured config object
    """
    iteration = schedule_entry["iter_index"]
    valid_year = schedule_entry["valid_year"]
    infer_year = schedule_entry["infer_year"]

    # Set observer-only training mode
    config.observer_only_training = True
    config.freeze_observer_during_rl = False  # Not applicable for Phase 1
    config.mafia_allow_observer_training = (
        True  # CRITICAL: Allow Observer to train and collect samples
    )

    # Expanding window flags (per spec Section 7)
    config.expanding_window_mode = True
    config.expanding_train_start = schedule_entry["train_start"].strftime("%Y-%m-%d")
    config.expanding_valid_months = 6
    config.walkforward_total_iterations = total_iterations

    # Walk-forward iteration settings
    config.walkforward_iteration = iteration
    config.walkforward_is_finetune = iteration > 0  # For dashboard display
    config.walkforward_train_start_year = schedule_entry["train_start"].year
    config.walkforward_train_end_year = schedule_entry["train_end"].year
    config.walkforward_valid_year = valid_year
    config.walkforward_infer_year = infer_year
    config.walkforward_train_years = schedule_entry["train_label"]
    config.walkforward_valid_range = schedule_entry["valid_label"]
    config.expanding_checkpoint_name = schedule_entry["ckpt_name"]

    # Set date ranges (must be pd.Timestamp, not strings)
    config.train_date_start = schedule_entry["train_start"]
    config.train_date_end = schedule_entry["train_end"]
    config.valid_date_start = schedule_entry["valid_start"]
    config.valid_date_end = schedule_entry["valid_end"]
    config.test_date_start = schedule_entry["infer_start"]
    config.test_date_end = schedule_entry["infer_end"]

    # Epoch schedule based on iteration
    if iteration == 0:
        # Base training: from scratch
        config.num_epochs = config.walkforward_base_epochs
        config.mafia_learning_rate = config.walkforward_base_lr
        config.early_stopping_patience = config.walkforward_base_patience
        config.observer_pretrained_path = None
    else:
        # Finetuning: load previous checkpoint
        config.num_epochs = config.walkforward_finetune_epochs
        config.mafia_learning_rate = config.walkforward_finetune_lr
        config.early_stopping_patience = config.walkforward_finetune_patience
        config.observer_pretrained_path = checkpoint_path
        # Finetune: skip observer warm-up (already trained)
        config.observer_warmup_samples = 0
        config.skip_observer_warmup = True

    # Output directories for walk-forward artifacts (shared across iterations)
    config.walkforward_checkpoint_dir = os.path.join(output_dir, "checkpoints")
    config.walkforward_states_dir = os.path.join(output_dir, "rl_states")
    os.makedirs(config.walkforward_checkpoint_dir, exist_ok=True)
    os.makedirs(config.walkforward_states_dir, exist_ok=True)

    # ============================================================
    # FIX: Properly set ALL result paths for this iteration
    # Previously only res_root was set, leaving res_dir pointing to TD3 path
    # Now using Config.rebuild_result_paths() helper for consistency
    # ============================================================
    iter_output_dir = os.path.join(output_dir, f"iter_{iteration}_valid_{valid_year}")

    # Use helper method to rebuild all result paths consistently
    config.rebuild_result_paths(iter_output_dir, create_dirs=True)

    # Update cur_datetime to reflect iteration (for logging clarity)
    config.cur_datetime = f"observer_iter{iteration}_valid{valid_year}"

    # Disable TD3 for observer-only training
    # We'll use uniform weights instead
    config.mafia_pretrain_mini_epochs = 0  # Skip normal pretrain
    config.skip_pretrain_on_resume = True

    return config


def train_observer_iteration(
    config: Config,
    iteration: int,
    verbose: bool = True,
) -> Tuple[str, Dict]:
    """
    Run one iteration of observer training.

    Args:
        config: Configured config object
        iteration: Iteration number
        verbose: Print progress

    Returns:
        Tuple of (best_checkpoint_path, validation_metrics)
    """
    # Skip verbose output if LiveDisplay will be used (it shows iteration info in UI)
    use_live_display = os.environ.get("MAFIA_NO_LIVE_DISPLAY", "") not in (
        "1",
        "true",
        "yes",
    )

    if verbose and not use_live_display:
        # Use simple logger for clean output
        train_range = getattr(
            config,
            "walkforward_train_years",
            f"{config.walkforward_train_start_year}→{config.walkforward_train_end_year}",
        )
        valid_range = getattr(
            config, "walkforward_valid_range", str(config.walkforward_valid_year)
        )
        if get_simple_logger:
            slogger = get_simple_logger()
            slogger.walkforward_iteration_start(
                iteration=iteration,
                total=getattr(config, "walkforward_total_iterations", 5),
                train_range=train_range,
                valid_range=valid_range,
                infer_year=config.walkforward_infer_year,
            )
        else:
            smart_print(f"\n{'=' * 70}")
            smart_print(f"WALK-FORWARD ITERATION {iteration}")
            smart_print(f"{'=' * 70}")
            smart_print(f"  Train: {train_range}")
            smart_print(f"  Valid: {valid_range}")
            smart_print(f"  Infer: {config.walkforward_infer_year}")
            smart_print(f"  Epochs: {config.num_epochs}")
            smart_print(f"  LR: {config.mafia_learning_rate}")
        if config.observer_pretrained_path:
            smart_print(f"  Resume from: {config.observer_pretrained_path}")
        smart_print(f"{'=' * 70}\n")

    # Run training using RLcontroller with observer_only_training mode
    # TD3 is frozen, only Observer trains with gradient updates
    # TD3.learn() still runs to collect rollouts, but train() skips gradient updates
    try:
        RLcontroller(config)
    except Exception as e:
        smart_print(f"[Error] Training failed: {e}")
        raise

    # Get best checkpoint path
    valid_year = config.walkforward_valid_year
    checkpoint_path = os.path.join(
        config.walkforward_checkpoint_dir, f"observer_best_{valid_year}.pth"
    )

    # Load validation metrics if available
    metrics = {}
    metrics_file = os.path.join(config.res_root, "validation_metrics.json")
    if os.path.exists(metrics_file):
        import json

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

    return checkpoint_path, metrics


def run_walkforward_observer_training(
    start_year: int = 2015,
    first_infer_year: int = 2018,
    last_infer_year: int = 2022,
    output_dir: str = "./observer_walkforward",
    resume_from_iter: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    seed: int = 2025,
    verbose: bool = True,
    use_offline_trainer: bool = False,
) -> List[Dict]:
    """
    Run full walk-forward observer training pipeline.

    Generates states for TD3 training from first_infer_year to last_infer_year.

    Example: To train TD3 on 2018-2022 (5 years):
        first_infer_year=2018, last_infer_year=2022
        -> Generates: State_2018, State_2019, State_2020, State_2021, State_2022

    Expanding-window iterations (per spec, no overlap between train & valid):
        Iter 0: Train 01/2015→06/2017 | Valid 07/2017→12/2017 | Infer 2018 → State_2018
        Iter 1: Train 01/2015→06/2018 | Valid 07/2018→12/2018 | Infer 2019 → State_2019
        Iter 2: Train 01/2015→06/2019 | Valid 07/2019→12/2019 | Infer 2020 → State_2020
        Iter 3: Train 01/2015→06/2020 | Valid 07/2020→12/2020 | Infer 2021 → State_2021
        Iter 4: Train 01/2015→06/2021 | Valid 07/2021→12/2021 | Infer 2022 → State_2022

    Args:
        start_year: Start year for expanding training window (default: 2015)
        first_infer_year: First inference year for TD3 states (default: 2018)
        last_infer_year: Last inference year for TD3 states (default: 2022)
        output_dir: Output directory for checkpoints and states
        resume_from_iter: Resume from specific iteration (0-indexed)
        resume_checkpoint: Checkpoint path for resuming
        seed: Random seed
        verbose: Print progress

    Returns:
        List of results for each iteration
    """
    # Build expanding-window schedule
    schedule = build_expanding_schedule(start_year, first_infer_year, last_infer_year)
    num_iterations = len(schedule)

    # Skip verbose output if LiveDisplay will be used
    use_live_display = os.environ.get("MAFIA_NO_LIVE_DISPLAY", "") not in (
        "1",
        "true",
        "yes",
    )

    if verbose and not use_live_display:
        schedule_str = [f"{s['valid_year']}→{s['infer_year']}" for s in schedule]
        smart_print(f"\n{'#' * 70}")
        smart_print(f"# WALK-FORWARD OBSERVER TRAINING PIPELINE")
        smart_print(f"{'#' * 70}")
        smart_print(f"  Start year: {start_year}")
        smart_print(f"  First infer year: {first_infer_year}")
        smart_print(f"  Last infer year: {last_infer_year}")
        smart_print(f"  Total iterations: {num_iterations}")
        smart_print(f"  Schedule: {schedule_str}")
        smart_print(f"  Output dir: {output_dir}")
        smart_print(f"{'#' * 70}\n")

    os.makedirs(output_dir, exist_ok=True)

    if use_offline_trainer:
        # Spec §7 compliant offline Collect→Train→Discard loop
        return run_offline_observer_training(
            start_year=start_year,
            first_infer_year=first_infer_year,
            last_infer_year=last_infer_year,
            output_dir=output_dir,
            seed=seed,
            verbose=verbose,
            generate_states=True,
            states_dir=os.path.join(output_dir, "rl_states"),
            rebalance_interval=None,
        )

    results = []
    prev_checkpoint = resume_checkpoint

    # Determine starting iteration
    start_iter = resume_from_iter if resume_from_iter is not None else 0

    for sched in schedule[start_iter:]:
        i = sched["iter_index"]

        # Create config for this iteration
        # Pass create_dirs=False to prevent auto-creation of TD3 directories
        config = Config(create_dirs=False)
        config.seed = seed
        config.market_name = "VNINDEX"
        config.walkforward_total_iterations = num_iterations

        # Setup walk-forward config
        config = setup_walkforward_config(
            config=config,
            schedule_entry=sched,
            checkpoint_path=prev_checkpoint,
            output_dir=output_dir,
            total_iterations=num_iterations,
        )

        # Train this iteration
        try:
            checkpoint_path, metrics = train_observer_iteration(
                config=config,
                iteration=i,
                verbose=verbose,
            )

            # State generation skipped per "Live Inference Stream" requirement (Spec 9.1)
            state_path = None
            infer_status = "live_stream"

            results.append(
                {
                    "iteration": i,
                    "train_range": sched["train_label"],
                    "valid_range": sched["valid_label"],
                    "valid_year": sched["valid_year"],
                    "infer_year": sched["infer_year"],
                    "checkpoint_path": checkpoint_path,
                    "state_path": state_path if infer_status == "success" else None,
                    "metrics": metrics,
                    "status": "success",
                    "infer_status": infer_status,
                }
            )

            # Update prev_checkpoint for next iteration
            prev_checkpoint = checkpoint_path

            if verbose:
                smart_print(f"\n✅ Iteration {i} complete!")
                smart_print(f"   Checkpoint: {checkpoint_path}")
                if metrics:
                    smart_print(f"   Metrics: {metrics}")
                smart_print(f"   State cache: {state_path} ({infer_status})")

        except Exception as e:
            results.append(
                {
                    "iteration": i,
                    "train_range": sched["train_label"],
                    "valid_range": sched["valid_label"],
                    "valid_year": sched["valid_year"],
                    "infer_year": sched["infer_year"],
                    "checkpoint_path": None,
                    "state_path": None,
                    "metrics": {},
                    "status": f"failed: {e}",
                    "infer_status": "skipped",
                }
            )
            smart_print(f"\n❌ Iteration {i} failed: {e}")
            # Continue with next iteration using last successful checkpoint

    # Save summary
    summary_file = os.path.join(output_dir, "walkforward_summary.json")
    import json

    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        smart_print(f"\n{'=' * 70}")
        smart_print("📊 WALK-FORWARD OBSERVER TRAINING SUMMARY (Stage 1)")
        smart_print(f"{'=' * 70}")
        successful = sum(1 for r in results if r['status'] == 'success')
        smart_print(f"  Total Iterations:  {num_iterations}")
        smart_print(f"  Successful:        {successful}/{num_iterations}")

        # List all checkpoints
        smart_print(f"\n  📦 Checkpoints Generated:")
        for r in results:
            if r['status'] == 'success':
                year = r.get('valid_year', 'N/A')
                infer = r.get('infer_year', 'N/A')
                smart_print(f"     • Ckpt_Best_{year} → Ready for {infer}")

        smart_print(f"\n  📁 Summary saved to: {summary_file}")
        smart_print(f"  📌 Next Step: Run Stage 2 (TD3 Training) with frozen Observer")
        smart_print(f"{'=' * 70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Observer Training Pipeline"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year for expanding training window (default: 2015)",
    )
    parser.add_argument(
        "--first-infer-year",
        type=int,
        default=2018,
        help="First inference year for TD3 states (default: 2018)",
    )
    parser.add_argument(
        "--last-infer-year",
        type=int,
        default=2024,
        help="Last inference year (default: 2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./observer_walkforward",
        help="Output directory (default: ./observer_walkforward)",
    )
    parser.add_argument(
        "--resume-from-iter",
        type=int,
        default=None,
        help="Resume from specific iteration (0-indexed)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint path for resuming",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed (default: 2025)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--offline-trainer",
        action="store_true",
        help="Use spec §7 offline Collect→Train→Discard trainer (recommended)",
    )

    args = parser.parse_args()

    results = run_walkforward_observer_training(
        start_year=args.start_year,
        first_infer_year=args.first_infer_year,
        last_infer_year=args.last_infer_year,
        output_dir=args.output_dir,
        resume_from_iter=args.resume_from_iter,
        resume_checkpoint=args.checkpoint_path,
        seed=args.seed,
        verbose=not args.quiet,
        use_offline_trainer=args.offline_trainer,
    )

    # Exit with error if any iteration failed
    if any(r["status"] != "success" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
