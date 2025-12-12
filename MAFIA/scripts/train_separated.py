#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CRITICAL: Set warnings filter BEFORE any imports to suppress gym deprecation warnings.
import warnings
import os

# Suppress all warnings early
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress gym deprecation warning via env var (must be set before gym import)
os.environ.setdefault("GYM_IGNORE_DEPRECATION_WARNINGS", "1")

# EARLY DEBUG: Print immediately to confirm script started
import sys
print("[DEBUG] Script started, importing modules...", file=sys.stderr, flush=True)

"""
Separated Training Pipeline - Phase 1 (Observer) and Phase 2 (TD3)

This script provides ready-to-use configurations for separated training:
- Phase 1: Train Observer only (frozen TD3)
- Phase 2: Train TD3 only (frozen Observer)

IMPORTANT: Set warnings filter BEFORE any imports to suppress gym deprecation warnings.

Usage:
    # Phase 1: Train Observer (Walk-Forward)
    python scripts/train_separated.py --phase 1 \
        --start-year 2015 \
        --first-infer-year 2018 \
        --last-infer-year 2022 \
        --output-dir ./walkforward_results

    # Phase 2: Train TD3 with frozen Observer
    python scripts/train_separated.py --phase 2 \
        --observer-checkpoint ./walkforward_results/checkpoints/observer_best_2021.pth \
        --train-years 2018 2019 2020 2021 2022

    # Show config only (dry run)
    python scripts/train_separated.py --phase 1 --dry-run
    
    # Disable LiveDisplay for clean terminal logging
    python scripts/train_separated.py --phase 1 --no-display
"""

import argparse
import os
import sys
import gc
from typing import Optional, List

# Parse --no-display EARLY, before any imports that use display
if "--no-display" in sys.argv:
    os.environ["MAFIA_NO_LIVE_DISPLAY"] = "1"
else:
    # When LiveDisplay is active, suppress startup messages (they would be overwritten anyway)
    os.environ["MAFIA_QUIET_STARTUP"] = "1"

print("[DEBUG] Importing pandas...", file=sys.stderr, flush=True)
import pandas as pd
print("[DEBUG] pandas imported OK", file=sys.stderr, flush=True)

# Ensure repo root on sys.path and change working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Change to MAFIA directory so relative paths (./data) work correctly
os.chdir(REPO_ROOT)

print("[DEBUG] Importing Config...", file=sys.stderr, flush=True)
from config import Config
print("[DEBUG] Config imported OK", file=sys.stderr, flush=True)

# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import smart_print
except ImportError:
    smart_print = print  # Fallback


def create_phase1_config(
    start_year: int = 2015,
    first_infer_year: int = 2018,
    last_infer_year: int = 2022,
    output_dir: str = "./observer_walkforward",
    seed: int = 2025,
) -> Config:
    """
    Create config for Phase 1: Observer-only training.

    Observer trains with Walk-Forward methodology:
    - Iteration 0: Train from scratch (50 epochs, LR 1e-4)
    - Iteration 1+: Finetune from previous best (20 epochs, LR 1e-5)

    Args:
        start_year: Start of expanding training window
        first_infer_year: First year to generate states for
        last_infer_year: Last year to generate states for
        output_dir: Directory for checkpoints and states
        seed: Random seed

    Returns:
        Configured Config object for Phase 1
    """
    # Pass create_dirs=False to prevent auto-creation of TD3 directories
    config = Config(create_dirs=False)

    # Phase 1 settings
    config.observer_only_training = True
    config.freeze_observer_during_rl = False
    config.mafia_allow_observer_training = True

    # Walk-forward settings
    config.walkforward_train_start_year = start_year
    config.seed = seed

    # Output directories
    config.walkforward_checkpoint_dir = os.path.join(output_dir, "checkpoints")
    config.walkforward_states_dir = os.path.join(output_dir, "rl_states")

    # Training hyperparameters (already set in config defaults)
    # - walkforward_base_epochs = 50
    # - walkforward_finetune_epochs = 20
    # - walkforward_base_lr = 1e-4
    # - walkforward_finetune_lr = 1e-5

    return config


def create_phase2_config(
    observer_checkpoint: str,
    train_years: Optional[List[int]] = None,
    valid_years: int = 2,
    test_years: int = 1,
    seed: int = 2025,
) -> Config:
    """
    Create config for Phase 2: TD3 training with frozen Observer.

    Observer is loaded from checkpoint and frozen.
    TD3 learns portfolio allocation on pre-computed states.

    Args:
        observer_checkpoint: Path to observer_best.pth
        train_years: List of years to train on (e.g., [2018, 2019, 2020, 2021, 2022])
        valid_years: Number of years for validation after train_years (default: 2)
        test_years: Number of years for testing after valid (default: 1)
        seed: Random seed

    Returns:
        Configured Config object for Phase 2
    """
    # Pass create_dirs=False to prevent auto-creation of TD3 directories
    config = Config(create_dirs=False)

    # Phase 2 settings - Observer FROZEN
    config.observer_only_training = False
    config.freeze_observer_during_rl = True
    config.mafia_allow_observer_training = False
    config.observer_pretrained_path = observer_checkpoint

    # Set date ranges based on train_years
    if train_years is not None and len(train_years) > 0:
        train_years_sorted = sorted(train_years)
        train_start = train_years_sorted[0]
        train_end = train_years_sorted[-1]

        # Valid: next `valid_years` years after training
        valid_start = train_end + 1
        valid_end = valid_start + valid_years - 1

        # Test: next `test_years` years after validation
        test_start = valid_end + 1
        test_end = test_start + test_years - 1

        # Apply date ranges to config
        config.train_date_start = pd.Timestamp(f"{train_start}-01-02 00:00:00")
        config.train_date_end = pd.Timestamp(f"{train_end}-12-30 23:59:59")
        config.valid_date_start = pd.Timestamp(f"{valid_start}-01-02 00:00:00")
        config.valid_date_end = pd.Timestamp(f"{valid_end}-12-31 23:59:59")
        config.test_date_start = pd.Timestamp(f"{test_start}-01-02 00:00:00")
        config.test_date_end = pd.Timestamp(f"{test_end}-12-31 23:59:59")

    # TD3 settings (use defaults from config)
    # - allocator_learning_rate_actor = 1e-4
    # - allocator_learning_rate_critic = 1e-3
    # - allocator_discount_gamma = 0.99
    # - allocator_polyak_tau = 0.005
    # - allocator_policy_delay = 2
    # - allocator_batch_size = 128
    # - allocator_replay_buffer_size = 100000
    # - allocator_reward_scale = 100.0

    config.seed = seed

    return config


def create_phase2_iterative_config(
    observer_checkpoint: str,
    train_years: List[int],
    valid_years: int = 1,
    test_years: int = 1,
    seed: int = 2025,
    iteration_label: str = "iter_0",
    output_basedir: str = "./td3_walkforward",
    resume_checkpoint: Optional[str] = None,
    # Walk-Forward Metadata
    iteration_index: int = 0,
    total_iterations: int = 1,
) -> Config:
    """
    Create config for a specific iteration of Phase 2 Walk-Forward.

    Args:
        observer_checkpoint: Path to observer checkpoint for this window (e.g. best_2017)
        train_years: List of training years for this window
        valid_years: Number of validation years
        test_years: Number of test years
        seed: Random seed
        iteration_label: Label for this iteration (folder name)
        output_basedir: Base directory for TD3 outputs
        resume_checkpoint: Path to TD3 checkpoint from previous iteration (if any)
        iteration_index: Current iteration index (0-based)
        total_iterations: Total expected iterations

    Returns:
        Config object
    """
    # Pass create_dirs=False to prevent auto-creation of TD3 directories
    config = Config(create_dirs=False)

    # Phase 2 settings - Observer FROZEN
    config.training_phase = 2  # Explicitly set Phase 2
    config.training_mode = "RL_ONLY"
    config.observer_only_training = False
    config.freeze_observer_during_rl = True
    config.mafia_allow_observer_training = False
    config.observer_pretrained_path = observer_checkpoint

    # Define date ranges
    train_start = min(train_years)
    train_end = max(train_years)
    
    # Valid starts after train
    valid_start = train_end + 1
    valid_end = valid_start + valid_years - 1
    
    # Test starts after valid
    test_start = valid_end + 1
    test_end = test_start + test_years - 1

    config.train_date_start = pd.Timestamp(f"{train_start}-01-02 00:00:00")
    config.train_date_end = pd.Timestamp(f"{train_end}-12-30 23:59:59")
    config.valid_date_start = pd.Timestamp(f"{valid_start}-01-02 00:00:00")
    config.valid_date_end = pd.Timestamp(f"{valid_end}-12-31 23:59:59")
    config.test_date_start = pd.Timestamp(f"{test_start}-01-02 00:00:00")
    config.test_date_end = pd.Timestamp(f"{test_end}-12-31 23:59:59")

    # Output directory override
    # Structure: output_basedir / iteration_label / ...
    config.res_dir = os.path.join(output_basedir, iteration_label)
    
    # Ensure directories exist
    config.res_model_dir = os.path.join(config.res_dir, "model")
    config.res_img_dir = os.path.join(config.res_dir, "graph")
    config.checkpoint_dir = os.path.join(config.res_dir, "checkpoints")
    config.metrics_history_path = os.path.join(config.res_dir, "metrics_history.csv")
    config.run_manifest_path = os.path.join(config.res_dir, "run_manifest.json")
    
    os.makedirs(config.res_dir, exist_ok=True)
    os.makedirs(config.res_model_dir, exist_ok=True)
    os.makedirs(config.res_img_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Resume settings
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        config.resume_from_checkpoint = resume_checkpoint
        config.auto_resume_from_latest = False
        config.filter_replay_buffer_on_resume = True # Enable buffer transfer
        config.reset_lr_scheduler_on_resume = True   # Reset LR for new window
    else:
        config.resume_from_checkpoint = None
        config.auto_resume_from_latest = False

    config.seed = seed
    
    # Expanding Window Mode settings
    config.expanding_window_mode = True
    config.expanding_train_start = f"{train_start}-01-01"
    
    # Store Walk-Forward Metadata for Dashboard/Callback
    config.wf_window_index = iteration_index 
    config.walkforward_info = {
        "iteration": iteration_index,
        "total_iterations": total_iterations,
        "train_years": f"{train_start}→{train_end}",
        "valid_year": valid_start,
        "infer_year": test_start,
        "train_range": f"{config.train_date_start} → {config.train_date_end}",
        "valid_range": f"{config.valid_date_start} → {config.valid_date_end}",
        "infer_range": f"{config.test_date_start} → {config.test_date_end}",
        "is_finetune": resume_checkpoint is not None,
    }

    return config


def print_config_summary(config: Config, phase: int):
    """Log a compact config summary for the requested phase."""
    smart_print(f"\n{'=' * 70}")
    smart_print(f"PHASE {phase} CONFIGURATION SUMMARY")
    smart_print(f"{'=' * 70}")

    if phase == 1:
        smart_print("\n[Observer Training Settings]")
        smart_print(f"  observer_only_training: {config.observer_only_training}")
        smart_print(
            f"  mafia_allow_observer_training: {config.mafia_allow_observer_training}"
        )
        smart_print(f"  freeze_observer_during_rl: {config.freeze_observer_during_rl}")
        smart_print(f"\n[Walk-Forward Settings]")
        smart_print(
            f"  walkforward_train_start_year: {config.walkforward_train_start_year}"
        )
        smart_print(f"  walkforward_base_epochs: {config.walkforward_base_epochs}")
        smart_print(
            f"  walkforward_finetune_epochs: {config.walkforward_finetune_epochs}"
        )
        smart_print(f"  walkforward_base_lr: {config.walkforward_base_lr}")
        smart_print(f"  walkforward_finetune_lr: {config.walkforward_finetune_lr}")
        smart_print(f"\n[Validation Score Weights]")
        smart_print(f"  w_sharpe: {config.walkforward_score_w_sharpe}")
        smart_print(f"  w_ic: {config.walkforward_score_w_ic}")
        smart_print(f"  w_f1: {config.walkforward_score_w_f1}")
        smart_print(f"\n[Output Directories]")
        smart_print(f"  checkpoints: {config.walkforward_checkpoint_dir}")
        smart_print(f"  rl_states: {config.walkforward_states_dir}")

    elif phase == 2:
        smart_print("\n[Date Ranges]")
        smart_print(f"  Train: {config.train_date_start} -> {config.train_date_end}")
        smart_print(f"  Valid: {config.valid_date_start} -> {config.valid_date_end}")
        smart_print(f"  Test:  {config.test_date_start} -> {config.test_date_end}")
        smart_print("\n[Observer Settings - FROZEN]")
        smart_print(f"  observer_pretrained_path: {config.observer_pretrained_path}")
        smart_print(f"  freeze_observer_during_rl: {config.freeze_observer_during_rl}")
        smart_print(
            f"  mafia_allow_observer_training: {config.mafia_allow_observer_training}"
        )
        smart_print(f"\n[TD3/Allocator Settings]")
        smart_print(
            f"  allocator_learning_rate_actor: {config.allocator_learning_rate_actor}"
        )
        smart_print(
            f"  allocator_learning_rate_critic: {config.allocator_learning_rate_critic}"
        )
        smart_print(f"  allocator_discount_gamma: {config.allocator_discount_gamma}")
        smart_print(f"  allocator_polyak_tau: {config.allocator_polyak_tau}")
        smart_print(f"  allocator_policy_delay: {config.allocator_policy_delay}")
        smart_print(f"\n[Reward Settings]")
        smart_print(f"  allocator_return_weight: {config.allocator_return_weight}")
        smart_print(f"  allocator_lambda_js: {config.allocator_lambda_js}")
        smart_print(f"  allocator_reward_scale: {config.allocator_reward_scale}")
        smart_print(f"\n[Replay Buffer]")
        smart_print(f"  allocator_batch_size: {config.allocator_batch_size}")
        smart_print(
            f"  allocator_replay_buffer_size: {config.allocator_replay_buffer_size}"
        )
        smart_print(f"  allocator_warmup_steps: {config.allocator_warmup_steps}")

    smart_print(f"\n[Common Settings]")
    smart_print(f"  seed: {config.seed}")
    smart_print(f"  topK: {config.topK}")
    smart_print(f"  topk_rebalance_interval: {config.topk_rebalance_interval}")
    smart_print(f"{'=' * 70}\n")


def run_phase1(
    start_year: int,
    first_infer_year: int,
    last_infer_year: int,
    output_dir: str,
    seed: int,
    use_offline_trainer: bool = True,
    dry_run: bool = False,
):
    """Run Phase 1: Observer Walk-Forward Training."""
    print("[DEBUG] run_phase1: entered function", file=sys.stderr, flush=True)
    # Skip startup messages if LiveDisplay will be used
    use_live_display = os.environ.get("MAFIA_NO_LIVE_DISPLAY", "") not in (
        "1",
        "true",
        "yes",
    )

    if not use_live_display:
        smart_print("\n" + "#" * 70)
        smart_print("# PHASE 1: OBSERVER WALK-FORWARD TRAINING")
        smart_print("#" * 70)

    config = create_phase1_config(
        start_year=start_year,
        first_infer_year=first_infer_year,
        last_infer_year=last_infer_year,
        output_dir=output_dir,
        seed=seed,
    )

    if not use_live_display:
        print_config_summary(config, phase=1)

    if dry_run:
        smart_print("[DRY RUN] Would run walk-forward training with above config")
        smart_print(f"  Iterations: {last_infer_year - first_infer_year + 1}")
        smart_print(f"  Years: {list(range(first_infer_year, last_infer_year + 1))}")
        return

    # Import and run walk-forward training
    print("[DEBUG] run_phase1: importing run_walkforward_observer_training...", file=sys.stderr, flush=True)
    from scripts.train_observer_walkforward import run_walkforward_observer_training
    print("[DEBUG] run_phase1: import OK, calling function...", file=sys.stderr, flush=True)

    results = run_walkforward_observer_training(
        start_year=start_year,
        first_infer_year=first_infer_year,
        last_infer_year=last_infer_year,
        output_dir=output_dir,
        seed=seed,
        verbose=True,
        use_offline_trainer=use_offline_trainer,
    )

    # Summary
    smart_print("\n" + "=" * 70)
    smart_print("PHASE 1 COMPLETE")
    smart_print("=" * 70)
    successful = sum(1 for r in results if r.get("status") == "success")
    smart_print(f"  Successful iterations: {successful}/{len(results)}")
    smart_print(f"  Checkpoints saved to: {config.walkforward_checkpoint_dir}")
    smart_print(f"  RL states saved to: {config.walkforward_states_dir}")

    return results


def run_phase2(
    observer_checkpoint: str,
    train_years: Optional[List[int]] = None,
    valid_years: int = 2,
    test_years: int = 1,
    seed: int = 2025,
    dry_run: bool = False,
):
    """Run Phase 2: TD3 Training with Frozen Observer."""
    smart_print("\n" + "#" * 70)
    smart_print("# PHASE 2: TD3 TRAINING WITH FROZEN OBSERVER")
    smart_print("#" * 70)

    if not dry_run and not os.path.exists(observer_checkpoint):
        smart_print(f"\n[ERROR] Observer checkpoint not found: {observer_checkpoint}")
        smart_print("  Please run Phase 1 first or provide valid checkpoint path")
        sys.exit(1)

    config = create_phase2_config(
        observer_checkpoint=observer_checkpoint,
        train_years=train_years,
        valid_years=valid_years,
        test_years=test_years,
        seed=seed,
    )

    print_config_summary(config, phase=2)

    if dry_run:
        smart_print("[DRY RUN] Would run TD3 training with above config")
        smart_print(f"  Observer checkpoint: {observer_checkpoint}")
        if train_years:
            smart_print(f"  Training years: {train_years}")
        return

    # Import and run RL training
    from entrance import RLcontroller

    RLcontroller(config)

    smart_print("\n" + "=" * 70)
    smart_print("PHASE 2 COMPLETE")
    smart_print("=" * 70)


def run_phase2_walkforward(
    observer_checkpoint_dir: str,
    start_year: int,
    first_infer_year: int,
    last_infer_year: int,
    output_dir: str,
    valid_years: int = 1,
    test_years: int = 1,
    seed: int = 2025,
    dry_run: bool = False,
):
    """
    Run Phase 2 (TD3) in Iterative Walk-Forward mode.
    
    Loop:
    1. Identify target year (infer year).
    2. Train window = [start_year, ..., target_year - 1].
    3. Load Observer Checkpoint for (target_year - 1).
    4. Resume TD3 from previous iteration's best checkpoint (if > 0).
    5. Train.
    """
    smart_print("\n" + "#" * 70)
    smart_print("# PHASE 2: TD3 ITERATIVE WALK-FORWARD")
    smart_print("#" * 70)
    
    if not os.path.exists(observer_checkpoint_dir):
        smart_print(f"[ERROR] Observer checkpoint directory not found: {observer_checkpoint_dir}")
        sys.exit(1)

    previous_td3_checkpoint = None
    results = []

    # Loop through expanding windows
    for target_year in range(first_infer_year, last_infer_year + 1):
        # Define expanding training window [start, target-1]
        train_end_year = target_year - 1
        train_years = list(range(start_year, train_end_year + 1))
        
        iteration_label = f"iter_{target_year}" # e.g., iter_2018
        smart_print(f"\n>>> Starting Iteration: Target Year {target_year} (Train: {start_year}-{train_end_year})")

        # 1. Find corresponding Observer checkpoint
        # Expecting naming pattern from Phase 1: observer_best_{year}.pth
        obs_ckpt_name = f"observer_best_{train_end_year}.pth"
        obs_ckpt_path = os.path.join(observer_checkpoint_dir, obs_ckpt_name)
        
        if not os.path.exists(obs_ckpt_path) and not dry_run:
            smart_print(f"[ERROR] Missing Observer checkpoint for {train_end_year}: {obs_ckpt_path}")
            smart_print("Ensure Phase 1 covers this year.")
            sys.exit(1)
            
        smart_print(f"  • Observer: {obs_ckpt_name}")
        # 2. Config setup
        current_iter_idx = target_year - first_infer_year
        total_iters = last_infer_year - first_infer_year + 1
        
        config = create_phase2_iterative_config(
            observer_checkpoint=obs_ckpt_path,
            train_years=train_years,
            valid_years=valid_years,
            test_years=test_years,
            seed=seed,
            iteration_label=iteration_label,
            output_basedir=output_dir,
            resume_checkpoint=previous_td3_checkpoint,
            iteration_index=current_iter_idx,
            total_iterations=total_iters
        )
        
        if previous_td3_checkpoint:
            smart_print(f"  • Resuming TD3: {os.path.basename(previous_td3_checkpoint)}")
        else:
            smart_print(f"  • Resuming TD3: None (Starting Fresh)")

        if dry_run:
            smart_print("[DRY RUN] Config generated. Skipping training.")
            continue

        # 3. specific imports locally to avoid global side effects if possible
        from entrance import RLcontroller
        
        # 4. Run Training
        try:
            RLcontroller(config)
            
            # 5. Find best checkpoint from this iteration to pass to next
            # Look in config.checkpoint_dir for best_valid or final
            ckpt_dir = config.checkpoint_dir
            # Prefer EvalCallback best_model.zip; otherwise fall back to latest rl_model*.zip checkpoint.
            best_model_path = os.path.join(config.res_model_dir, "best_model.zip")
            
            next_ckpt = None
            if os.path.exists(best_model_path):
                next_ckpt = best_model_path
                smart_print(f"  • Found Best Model: {best_model_path}")
            else:
                # Fallback to latest checkpoint in checkpoints/
                candidates = [
                    os.path.join(ckpt_dir, f)
                    for f in os.listdir(ckpt_dir)
                    if "rl_model" in f and f.endswith(".zip")
                ]
                if candidates:
                    # Sort by modification time
                    latest_ckpt = max(candidates, key=os.path.getmtime)
                    next_ckpt = latest_ckpt
                    smart_print(f"  • Found Latest Checkpoint: {latest_ckpt}")
            
            # Explicit garbage collection to prevent OOM
            gc.collect()

            # Pass to next iter
            if next_ckpt:
                 # Verify it has corresponding info json
                 # If absent, might issue warning on valid resume, but we try anyway
                 previous_td3_checkpoint = next_ckpt
                 results.append({"year": target_year, "status": "success", "ckpt": next_ckpt})
            else:
                smart_print("  [WARNING] No checkpoint found to pass to next iteration.")
                results.append({"year": target_year, "status": "no_ckpt"})

        except Exception as e:
            smart_print(f"  [ERROR] Iteration {target_year} failed: {e}")
            results.append({"year": target_year, "status": "failed", "error": str(e)})
            # Optional: break or continue?
            # If failed, next iter starts fresh or breaks? 
            # Break usually safer to avoid waste.
            break

    smart_print("\n" + "=" * 70)
    smart_print("PHASE 2 WALK-FORWARD COMPLETE")
    smart_print("=" * 70)
    return results



def run_all_phases(
    start_year: int,
    first_infer_year: int,
    last_infer_year: int,
    output_dir: str,
    train_years: Optional[List[int]],
    valid_years: int,
    test_years: int,
    seed: int,
    use_offline_trainer: bool = True,
    dry_run: bool = False,
):
    """
    Run both phases sequentially:
    Phase 1: Observer Walk-Forward Training
    Phase 2: TD3 Training with Frozen Observer (using best checkpoint from Phase 1)
    """
    print("[DEBUG] run_all_phases: entered function", file=sys.stderr, flush=True)
    smart_print("\n" + "=" * 70)
    smart_print("  FULL PIPELINE: PHASE 1 (Observer) -> PHASE 2 (TD3)")
    smart_print("=" * 70)

    print("[DEBUG] run_all_phases: calling run_phase1...", file=sys.stderr, flush=True)
    # Phase 1: Train Observer
    results = run_phase1(
        start_year=start_year,
        first_infer_year=first_infer_year,
        last_infer_year=last_infer_year,
        output_dir=output_dir,
        seed=seed,
        use_offline_trainer=use_offline_trainer,
        dry_run=dry_run,
    )

    # Force garbage collection after heavy Phase 1
    gc.collect()

    if dry_run:
        smart_print("\n[DRY RUN] Phase 1 complete. Would proceed to Phase 2...")
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        best_checkpoint = os.path.join(
            checkpoint_dir, f"observer_best_{last_infer_year - 1}.pth"
        )
        smart_print(f"  Expected checkpoint: {best_checkpoint}")
        run_phase2(
            observer_checkpoint=best_checkpoint,
            train_years=train_years
            or list(range(first_infer_year, last_infer_year + 1)),
            valid_years=valid_years,
            test_years=test_years,
            seed=seed,
            dry_run=True,
        )
        return

    # Find the best checkpoint from Phase 1 (last successful iteration)
    successful_results = [r for r in results if r.get("status") == "success"]
    if not successful_results:
        smart_print(
            "\n[ERROR] Phase 1 failed - no successful iterations. Cannot proceed to Phase 2."
        )
        sys.exit(1)

    # Use the last successful checkpoint (most recent valid year)
    last_result = successful_results[-1]
    best_checkpoint = last_result["checkpoint_path"]

    smart_print("\n" + "-" * 70)
    smart_print(f"  Phase 1 complete. Best checkpoint: {best_checkpoint}")
    smart_print(f"  Proceeding to Phase 2 (TD3 Training)...")
    smart_print("-" * 70)

    # Phase 2: Train TD3 with frozen Observer
    # Check if we should use Walk-Forward mode (default) validation
    # If train_years is None, assume full walk-forward matching Phase 1 params
    
    phase1_ckpt_dir = os.path.join(output_dir, "checkpoints") # Output from Phase 1
    phase2_output_dir = os.path.join(output_dir, "phase2_td3")

    if train_years is None:
        smart_print("\n[INFO] Running Phase 2 in Iterative Walk-Forward Mode (matching Phase 1 years)")
        run_phase2_walkforward(
            observer_checkpoint_dir=phase1_ckpt_dir,
            start_year=start_year,
            first_infer_year=first_infer_year,
            last_infer_year=last_infer_year,
            output_dir=phase2_output_dir,
            valid_years=valid_years,
            test_years=test_years,
            seed=seed,
            dry_run=dry_run
        )
    else:
        # Legacy/Single-Window mode
        run_phase2(
            observer_checkpoint=best_checkpoint,
            train_years=train_years,
            valid_years=valid_years,
            test_years=test_years,
            seed=seed,
            dry_run=dry_run,
        )

    smart_print("\n" + "=" * 70)
    smart_print("  FULL PIPELINE COMPLETE")
    smart_print("=" * 70)
    smart_print(f"  Phase 1 Checkpoints: {phase1_ckpt_dir}")
    smart_print(f"  Phase 2 Output: {phase2_output_dir}")
    smart_print("=" * 70 + "\n")


def main():
    print("[DEBUG] Entering main()...", file=sys.stderr, flush=True)
    parser = argparse.ArgumentParser(
        description="Separated Training Pipeline for Observer and TD3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run BOTH phases automatically (Observer -> TD3)
  python scripts/train_separated.py --phase all --output-dir ./results

  # Phase 1 only: Train Observer
  python scripts/train_separated.py --phase 1 --output-dir ./results

  # Phase 2 only: Train TD3 with frozen Observer
  python scripts/train_separated.py --phase 2 \\
      --observer-checkpoint ./results/checkpoints/observer_best_2021.pth

  # Dry run to see config
  python scripts/train_separated.py --phase all --dry-run
        """,
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "all"],
        required=True,
        help="Training phase: 1=Observer only, 2=TD3 only, all=Observer then TD3",
    )

    # Phase 1 arguments
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year for training window (Phase 1)",
    )
    parser.add_argument(
        "--first-infer-year",
        type=int,
        default=2018,
        help="First inference year (Phase 1)",
    )
    parser.add_argument(
        "--last-infer-year",
        type=int,
        default=2022,
        help="Last inference year (Phase 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./observer_walkforward",
        help="Output directory (Phase 1)",
    )

    # Phase 2 arguments
    parser.add_argument(
        "--observer-checkpoint",
        type=str,
        default=None,
        help="Path to observer checkpoint (Phase 2, required)",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=None,
        help="Years to train TD3 on (Phase 2). Default: same as infer years from Phase 1",
    )
    parser.add_argument(
        "--valid-years",
        type=int,
        default=2,
        help="Number of years for validation after training (Phase 2, default: 2)",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        default=1,
        help="Number of years for testing after validation (Phase 2, default: 1)",
    )
    
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Enable Iterative Walk-Forward mode for Phase 2 (requires --observer-checkpoint to be a directory or use Phase 1 outputs)",
    )
    # Re-using start/infer years from Phase 1 args if not provided for Phase 2 WF


    # Common arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show config without running",
    )
    parser.add_argument(
        "--online-mode",
        action="store_true",
        help="Use legacy online RLcontroller for Phase 1 (default: offline trainer per spec §7)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable LiveDisplay, use simple terminal logging only",
    )

    args = parser.parse_args()
    print(f"[DEBUG] Args parsed: phase={args.phase}", file=sys.stderr, flush=True)

    # Note: --no-display is handled at top of script before imports
    if args.no_display:
        print("[INFO] LiveDisplay disabled, using simple terminal logging")

    if args.phase == "1":
        run_phase1(
            start_year=args.start_year,
            first_infer_year=args.first_infer_year,
            last_infer_year=args.last_infer_year,
            output_dir=args.output_dir,
            seed=args.seed,
            use_offline_trainer=not args.online_mode,
            dry_run=args.dry_run,
        )
    elif args.phase == "2":
        # Check if Walk-Forward Mode requested
        if args.walk_forward:
            if not args.observer_checkpoint: # Must be dir
                 # Default to standard path if not provided? No, force explicit
                 if not args.dry_run:
                     parser.error("--observer-checkpoint (directory) is required for Phase 2 Walk-Forward")
                 
            run_phase2_walkforward(
                observer_checkpoint_dir=args.observer_checkpoint,
                start_year=args.start_year,
                first_infer_year=args.first_infer_year,
                last_infer_year=args.last_infer_year,
                output_dir=args.output_dir,
                valid_years=args.valid_years,
                test_years=args.test_years,
                seed=args.seed,
                dry_run=args.dry_run
            )
        else:
            # Single Window Mode
            if not args.observer_checkpoint and not args.dry_run:
                parser.error("--observer-checkpoint is required for Phase 2")
                # For dry run, use placeholder
            checkpoint = args.observer_checkpoint or "/path/to/observer_best.pth"
            run_phase2(
                observer_checkpoint=checkpoint,
                train_years=args.train_years,
                valid_years=args.valid_years,
                test_years=args.test_years,
                seed=args.seed,
                dry_run=args.dry_run,
            )
    elif args.phase == "all":
        print("[DEBUG] Calling run_all_phases...", file=sys.stderr, flush=True)
        run_all_phases(
            start_year=args.start_year,
            first_infer_year=args.first_infer_year,
            last_infer_year=args.last_infer_year,
            output_dir=args.output_dir,
            train_years=args.train_years,
            valid_years=args.valid_years,
            test_years=args.test_years,
            seed=args.seed,
            use_offline_trainer=not args.online_mode,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
