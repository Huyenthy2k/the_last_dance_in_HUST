#!/usr/bin/env python3
"""
Offline Batch Training for MAFIA Observer - YEARLY Expanding Window

This script trains the MAFIA Observer using offline batch mode with
YEARLY expanding window schedule per training_strategy_proposal.md.

Schedule (5 iterations):
| Iter | Train Period | Valid Year |
|------|--------------|------------|
| 0    | 2015-2019    | 2020       |
| 1    | 2015-2020    | 2021       |
| 2    | 2015-2021    | 2022       |
| 3    | 2015-2022    | 2023       |
| 4    | 2015-2023    | 2024       |

Usage:
    # MACRO_ONLY training (Phase 1A)
    python scripts/train_observer_offline.py \\
        --mode MACRO_ONLY \\
        --output-dir ./observer_offline

    # SELECTION_ONLY training (Phase 1B) - after MACRO_ONLY
    python scripts/train_observer_offline.py \\
        --mode SELECTION_ONLY \\
        --init-checkpoint ./observer_offline/checkpoints/macro/Observer_Best_2024.pth \\
        --output-dir ./observer_offline
"""

import argparse
import os
import sys
import gc
import copy
import traceback
from typing import Dict, List, Optional
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch as th
from torch.utils.tensorboard import SummaryWriter

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config, get_phase_subdir, MafiaTrainMode
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from visualize_training import generate_epoch_report
from utils.mafia_data_loader import load_mafia_data
# NOTE: generate_single_year removed - state generation now in train_walkforward.py

# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import (
        smart_print, 
        update_observer, 
        update_walkforward_iteration, 
        update_walkforward_score
    )
except ImportError:
    smart_print = print  # Fallback
    def update_observer(*args, **kwargs): pass
    def update_walkforward_iteration(*args, **kwargs): pass
    def update_walkforward_score(*args, **kwargs): pass

# Simple logger for clean terminal output
try:
    from utils.simple_logger import get_simple_logger, init_simple_logger
except ImportError:
    get_simple_logger = None
    init_simple_logger = None

# TensorBoard logger utility
try:
    from utils.tensorboard_logger import create_tensorboard_logger
except ImportError:
    create_tensorboard_logger = None


def build_expanding_schedule(
    start_year: int, first_infer_year: int, last_infer_year: int,
    valid_duration_years: int = 1
) -> List[Dict]:
    """
    Build YEARLY expanding-window schedule per training_strategy_proposal.md.

    Strategy (5 iterations for 2020-2024, valid_duration_years=1):
    | Iter | Train Period | Valid/Infer Year |
    |------|--------------|------------------|
    | 0    | 2015-2019    | 2020             |
    | 1    | 2015-2020    | 2021             |
    | 2    | 2015-2021    | 2022             |
    | 3    | 2015-2022    | 2023             |
    | 4    | 2015-2023    | 2024             |

    With valid_duration_years=2, first_infer_year=2021:
    | Iter | Train Period | Valid Period     |
    |------|--------------|------------------|
    | 0    | 2015-2020    | 2021-2022        |

    Args:
        start_year: Training start year (fixed, e.g., 2015)
        first_infer_year: First year to generate states for (e.g., 2020)
        last_infer_year: Last year to generate states for (e.g., 2024)
        valid_duration_years: Number of years for validation period (default: 1)

    Returns:
        List of schedule dicts with train/valid/infer periods
    """
    schedule = []

    for iter_idx, valid_year in enumerate(range(first_infer_year, last_infer_year + 1)):
        # Training: start_year -> (valid_year - 1)
        train_end_year = valid_year - 1

        train_start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
        train_end = pd.Timestamp(f"{train_end_year}-12-31 23:59:59")
        valid_start = pd.Timestamp(f"{valid_year}-01-01 00:00:00")
        # Support multi-year validation
        valid_end_year = valid_year + valid_duration_years - 1
        valid_end = pd.Timestamp(f"{valid_end_year}-12-31 23:59:59")

        # Build validation label
        if valid_duration_years == 1:
            valid_label = f"{valid_start.date()} -> {valid_end.date()} (Full Year)"
            valid_year_label = str(valid_year)
        else:
            valid_label = f"{valid_start.date()} -> {valid_end.date()} ({valid_duration_years} Years)"
            valid_year_label = f"{valid_year}-{valid_end_year}"

        schedule.append({
            "iter_index": iter_idx,
            "iter_display": iter_idx + 1,
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "valid_year": valid_year,  # Start year of validation
            "valid_year_label": valid_year_label,  # e.g., "2021" or "2021-2022"
            "valid_quarter": None,  # YEARLY schedule, no quarters
            "infer_year": valid_year,
            "infer_quarter": None,  # YEARLY schedule, no quarters
            "infer_start": valid_start,
            "infer_end": valid_end,
            "ckpt_name": f"Observer_Best_{valid_year_label}",
            "train_label": f"{train_start.date()} -> {train_end.date()}",
            "valid_label": valid_label,
        })

    return schedule


def load_stock_data(config: Config) -> pd.DataFrame:
    """
    Load stock data for training.

    Returns:
        DataFrame with columns: date, stock, open, high, low, close, volume
    """
    # Try to use mafia_data_loader
    try:
        data = load_mafia_data(config)
        return data
    except Exception as e:
        smart_print(f"[WARN] Failed to load data via mafia_data_loader: {e}")

    # Fallback: Load from CSV or pickle
    data_dir = getattr(config, "data_dir", "./data")
    data_file = os.path.join(data_dir, "stock_data.csv")

    if os.path.exists(data_file):
        return pd.read_csv(data_file, parse_dates=["date"])

    raise FileNotFoundError(f"No data file found at {data_file}")


def train_observer_offline_iteration(
    config: Config,
    schedule_entry: Dict,
    trainer: ObserverOfflineBatchTrainer,
    data_tensors: Dict[str, th.Tensor],
    checkpoint_dir: str,
    results_dir: str,  # Phase-aware results directory
    num_epochs: Optional[int] = None,
    batches_per_epoch: Optional[int] = None,
    prev_checkpoint: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
    global_step_start: int = 0,
    verbose: bool = True,
    training_mode: str = "FULL",
    init_checkpoint: Optional[str] = None,  # MACRO checkpoint (frozen params for SELECTION_ONLY)
    prev_selection_checkpoint: Optional[str] = None,  # Prev SELECTION checkpoint (trainable params)
    sampling_strategy: str = "auto",  # ADDED: 'auto', 'uniform', 'hybrid_weighted'
    skip_recovery: bool = False,  # ADDED: Skip validation history recovery on resume
) -> str:
    """
    Train one iteration using offline batch mode with CES Validation.

    Args:
        config: Configuration object
        schedule_entry: Dict with train/valid date ranges
        trainer: ObserverOfflineBatchTrainer instance
        data_tensors: Prepared data tensors (full dataset)
        checkpoint_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batches_per_epoch: Steps per epoch
        prev_checkpoint: Path to previous checkpoint for finetuning (same phase)
        init_checkpoint: MACRO checkpoint for SELECTION_ONLY (frozen params)
        prev_selection_checkpoint: Previous SELECTION checkpoint (trainable params for knowledge transfer)
        verbose: Print progress

    Returns:
        Path to best checkpoint (selected via CES)
    """
    from RL_controller.validation_tracker import ValidationMetricsTracker
    
    iteration = schedule_entry["iter_index"]
    valid_year = schedule_entry["valid_year"]

    # Determine num_epochs logic (Base vs Finetune)
    if num_epochs is None:
        if iteration == 0:
            num_epochs = int(getattr(config, "mafia_observer_base_epochs", 50))
        else:
            num_epochs = int(getattr(config, "mafia_observer_finetune_epochs", 5))

    if verbose:
        train_mode = "Base Training" if iteration == 0 else "Finetuning"
        smart_print(f"[OFFLINE] {train_mode}: {num_epochs} epochs")

    # Scale curriculum warmup/rampup for finetune iterations
    # This ensures lambda reaches 1.0 within the shorter training period
    is_finetune = (iteration > 0)
    trainer.scale_curriculum_for_finetune(num_epochs=num_epochs, is_finetune=is_finetune)

    # === ITERATION-AWARE SAMPLING STRATEGY ===
    # 'auto': Iteration 0 = uniform, Iteration 1+ = hybrid_weighted
    # 'uniform': Force uniform sampling for all iterations
    # 'hybrid_weighted': Force hybrid sampling for all iterations
    if sampling_strategy == "auto":
        if iteration == 0:
            config.mafia_sampling_strategy = "uniform"
            if verbose:
                smart_print(f"[SAMPLING] Iteration 0: Using UNIFORM sampling (full data coverage)")
        else:
            config.mafia_sampling_strategy = "hybrid_weighted"
            if verbose:
                smart_print(f"[SAMPLING] Iteration {iteration}: Using HYBRID_WEIGHTED sampling (recency bias)")
    else:
        config.mafia_sampling_strategy = sampling_strategy
        if verbose:
            smart_print(f"[SAMPLING] Override: Using {sampling_strategy.upper()} sampling for all iterations")

    # Update Dashboard Header
    update_walkforward_iteration(
        iteration=iteration,
        total_iterations=10, # Estimate or pass from outside. 
        train_years=str(schedule_entry["train_label"]),
        valid_year=valid_year,
        infer_year=schedule_entry["infer_year"],
        train_range=schedule_entry["train_label"],
        valid_range=schedule_entry["valid_label"],
        is_finetune=(iteration > 0)
    )

    # === CHECKPOINT LOADING LOGIC ===
    # Priority for SELECTION_ONLY mode:
    # 1. If both init_checkpoint (MACRO) AND prev_selection_checkpoint exist:
    #    -> Use load_merged_checkpoints() to merge frozen from MACRO, trainable from SELECTION
    # 2. If only init_checkpoint (MACRO) exists:
    #    -> Load full MACRO checkpoint (Iter 0 or no prev SELECTION)
    # 3. For other modes (MACRO_ONLY, FULL):
    #    -> Use standard prev_checkpoint loading for same-phase resume

    if training_mode == MafiaTrainMode.SELECTION_ONLY:
        if init_checkpoint and prev_selection_checkpoint:
            # MERGED LOADING: Frozen from MACRO, Trainable from prev SELECTION
            if os.path.exists(init_checkpoint) and os.path.exists(prev_selection_checkpoint):
                smart_print(f"[OFFLINE] Loading MERGED checkpoints:")
                smart_print(f"  - Frozen (MACRO): {os.path.basename(init_checkpoint)}")
                smart_print(f"  - Trainable (SELECTION): {os.path.basename(prev_selection_checkpoint)}")
                trainer.observer.load_merged_checkpoints(
                    frozen_checkpoint_path=init_checkpoint,
                    trainable_checkpoint_path=prev_selection_checkpoint,
                    load_optimizer=False,  # Fresh optimizer for new iteration
                )
            elif os.path.exists(init_checkpoint):
                smart_print(f"[OFFLINE] Loading init_checkpoint (MACRO only): {init_checkpoint}")
                trainer.observer.load_checkpoint(init_checkpoint)
        elif init_checkpoint and os.path.exists(init_checkpoint):
            # Iter 0: No prev SELECTION, load full MACRO
            smart_print(f"[OFFLINE] Loading init_checkpoint (MACRO, Iter 0): {init_checkpoint}")
            trainer.observer.load_checkpoint(init_checkpoint)
    else:
        # MACRO_ONLY or FULL mode: Use standard prev_checkpoint loading
        if prev_checkpoint and os.path.exists(prev_checkpoint):
            smart_print(f"[OFFLINE] Loading prev_checkpoint (Resume/WarmStart): {prev_checkpoint}")
            trainer.observer.load_checkpoint(prev_checkpoint)

        # Load INIT checkpoint if provided (overrides prev_checkpoint)
        if init_checkpoint and os.path.exists(init_checkpoint):
            smart_print(f"[OFFLINE] Loading init_checkpoint (Transfer): {init_checkpoint}")
            trainer.observer.load_checkpoint(init_checkpoint)

    # Create observer and trainer
    # ... (observer creation logic) ...
    # Initialize mode
    if hasattr(trainer.observer, "set_train_mode"):
        trainer.observer.set_train_mode(training_mode)
        if verbose:
             smart_print(f"[OFFLINE] Set Training Mode: {training_mode}")

    # Prepare train/valid splits
    dates = data_tensors["dates"]
    train_mask = (dates >= schedule_entry["train_start"]) & (dates <= schedule_entry["train_end"])
    valid_mask = (dates >= schedule_entry["valid_start"]) & (dates <= schedule_entry["valid_end"])

    train_indices = np.where(train_mask)[0]
    valid_indices = np.where(valid_mask)[0]

    if len(train_indices) == 0:
        raise ValueError(f"No training data in range {schedule_entry['train_label']}")
    if len(valid_indices) == 0:
        raise ValueError(f"No validation data in range {schedule_entry['valid_label']}")

    # [OPTIMIZATION] Include h (rebalance_interval) days from VALIDATION as suffix for training
    # This allows training on LAST days of training period (need horizon for ground truth/returns)
    h = trainer.rebalance_interval  # Rebalance interval (21 days default)
    suffix_length = min(h, len(valid_indices))  # Don't exceed validation length
    train_data_end = train_indices[-1] + 1 + suffix_length  # Extend into validation


    
    # Create views with suffix
    train_tensors = {
        "ochlv": data_tensors["ochlv"][train_indices[0] : train_data_end],
        "returns": data_tensors["returns"][train_indices[0] : train_data_end],
        "market_ochlv": data_tensors.get("market_ochlv", None)[train_indices[0] : train_data_end] if data_tensors.get("market_ochlv") is not None else None,
        "market_returns": data_tensors.get("market_returns", None)[train_indices[0] : train_data_end] if data_tensors.get("market_returns") is not None else None,
        "dates": dates[train_indices[0] : train_indices[-1] + 1 + suffix_length],
        "stock_list": data_tensors["stock_list"],
        "T_total": len(train_indices) + suffix_length,  # Include suffix
        "N": data_tensors["N"],
        # NEW: Suffix metadata
        "suffix_length": suffix_length,  # How many days are lookahead suffix (from validation)
        "train_end_idx": len(train_indices),  # Index where actual training ends (suffix starts)
    }
    
    if verbose:
        smart_print(f"[TRAIN] Training data: {train_tensors['T_total']} days (train={len(train_indices)}, suffix={suffix_length} from valid)")
    

    # Auto-compute batches per epoch
    # Auto-compute batches per epoch
    if batches_per_epoch is None:
        # [FIX] For hybrid_weighted, use fixed budget (18) unless overridden
        # User explicitly requested 18 batches/epoch to match previous iteration budget
        if config.mafia_sampling_strategy == "hybrid_weighted":
            batches_per_epoch = 18
            if verbose:
                smart_print(f"[SAMPLING] Hybrid Weighted active: Defaulting to fixed {batches_per_epoch} batches/epoch (Budget-based)")
        else:
            # Standard auto-compute for uniform/full coverage
            T_train = train_tensors["T_total"]
            T_m = trainer.T_m
            h = trainer.horizon
            T_w = trainer.T_w
            batch_size = trainer.batch_size
            available_starts = T_train - T_m - h - T_w
            batches_per_epoch = max(1, int(np.ceil(available_starts / batch_size)))

    # [OPTIMIZATION] Include T_w days from TRAINING data as prefix for validation warm-up
    # This allows 100% validation utilization (no discarded days at start of validation year)
    T_w = trainer.T_w  # Observation window = 30 days (lookback for OCHLV features)
    prefix_length = T_w  # Only need T_w days for feature lookback
    prefix_start = max(0, valid_indices[0] - prefix_length)

    
    # Slice data including prefix + validation period
    valid_data_start = prefix_start
    valid_data_end = valid_indices[-1] + 1
    
    # Calculate actual prefix length (may be less if valid_indices[0] < prefix_length)
    actual_prefix_length = valid_indices[0] - prefix_start
    
    valid_tensors = {
        "ochlv": data_tensors["ochlv"][valid_data_start : valid_data_end],
        "returns": data_tensors["returns"][valid_data_start : valid_data_end],
        "market_ochlv": data_tensors.get("market_ochlv", None)[valid_data_start : valid_data_end] if data_tensors.get("market_ochlv") is not None else None,
        "market_returns": data_tensors.get("market_returns", None)[valid_data_start : valid_data_end] if data_tensors.get("market_returns") is not None else None,
        "dates": dates[prefix_start : valid_indices[-1] + 1],  # Include prefix dates
        "stock_list": data_tensors["stock_list"],
        "T_total": valid_data_end - valid_data_start,
        "N": data_tensors["N"],
        # NEW: Prefix metadata for validation logic
        "prefix_length": actual_prefix_length,  # How many days are warm-up prefix (from training)
        "valid_start_idx": actual_prefix_length,  # Index where actual validation starts
    }
    
    if verbose:
        smart_print(f"[VALID] Validation data: {valid_tensors['T_total']} days (prefix={actual_prefix_length} from train, valid={len(valid_indices)})")

    # Initialize ValidationMetricsTracker (inside phase results directory)
    iter_output_dir = os.path.join(results_dir, f"iter_{iteration}_valid_{valid_year}")
    os.makedirs(iter_output_dir, exist_ok=True)
    tracker = ValidationMetricsTracker(iter_output_dir, training_mode=training_mode)
    
    # Checkpoint storage
    temp_ckpt_dir = os.path.join(checkpoint_dir, f"temp_iter_{iteration}")
    os.makedirs(temp_ckpt_dir, exist_ok=True)

    current_global_step = global_step_start

            # Resume logic: Check for existing checkpoints in temp dir
    # Priority: latest_checkpoint.pth > epoch_{N}.pth (best checkpoints)
    # [FIX] Skip resume logic if init_checkpoint is explicitly provided (warm start from specific checkpoint)
    start_epoch = 0
    skip_resume_for_init = init_checkpoint is not None and os.path.exists(init_checkpoint)
    if skip_resume_for_init:
        smart_print(f"[INIT] Skipping resume logic - using init_checkpoint: {init_checkpoint}")
        smart_print(f"[INIT] Training will start from epoch 0 with warm-started weights")

    smart_print(f"[DEBUG] Checking for resume in: {temp_ckpt_dir}")
    if os.path.exists(temp_ckpt_dir) and not skip_resume_for_init:
        # fast check contents
        dir_contents = os.listdir(temp_ckpt_dir)
        smart_print(f"[DEBUG] temp_ckpt_dir exists, contents: {dir_contents}")
        
        latest_ckpt_path = os.path.join(temp_ckpt_dir, "latest_checkpoint.pth")
        
        # Stale Checkpoint Detection
        # If 'latest_checkpoint.pth' is missing but we have 'epoch_X.pth', it implies 
        # the previous run might have finished/crashed but 'latest' wasn't written 
        # or we are picking up a "best" checkpoint from a long-ago run.
        has_epoch_ckpts = any(f.startswith("epoch_") and f.endswith(".pth") for f in dir_contents)
        if has_epoch_ckpts and not os.path.exists(latest_ckpt_path):
            smart_print("\n" + "!" * 80)
            smart_print("⚠️  WARNING: Stale Checkpoint Risk!")
            smart_print("   Found 'epoch_X.pth' (best ckpts) but NO 'latest_checkpoint.pth'.")
            smart_print("   This usually means a previous run finished or was stopped, and we are")
            smart_print("   dangerously resuming from a BEST checkpoint rather than the LATEST state.")
            smart_print("   This can cause 'time travel' where we resume from epoch 8 (best) ")
            smart_print("   even though the run actually went to epoch 50.")
            smart_print("!" * 80 + "\n")

        # Try latest_checkpoint.pth first (most recent training state)
        resume_path = None
        if os.path.exists(latest_ckpt_path):
            resume_path = latest_ckpt_path
            smart_print(f"[RESUME] Found latest checkpoint: {latest_ckpt_path}")
        elif has_epoch_ckpts:
            # Fallback: Find best epoch checkpoints (epoch_{N}.pth)
            existing_ckpts = [f for f in dir_contents if f.startswith("epoch_") and f.endswith(".pth")]
            epoch_nums = [int(f.split("_")[1].split(".")[0]) for f in existing_ckpts]
            max_epoch = max(epoch_nums)
            resume_path = os.path.join(temp_ckpt_dir, f"epoch_{max_epoch}.pth")
            smart_print(f"[RESUME] WARN: Fallback to best checkpoint: {resume_path}")

        if resume_path:
            try:
                loaded_epoch = trainer.observer.load_checkpoint(resume_path)
                start_epoch = loaded_epoch + 1

                # [RESUME-FIX] Restore Curriculum State (only for SELECTION_ONLY mode)
                if training_mode == MafiaTrainMode.SELECTION_ONLY:
                    trainer._current_lambda_epoch = trainer._compute_lambda_epoch(loaded_epoch)
                    smart_print(f"[RESUME] Restored lambda_epoch = {trainer._current_lambda_epoch:.4f} for loaded epoch {loaded_epoch}")
                else:
                    trainer._current_lambda_epoch = 0.0  # Not used in MACRO_ONLY

                # Restore Validation Tracker History
                valid_csv_path = os.path.join(iter_output_dir, "valid_metrics.csv")
                if os.path.exists(valid_csv_path):
                    smart_print(f"[RESUME] Restoring validation history from: {valid_csv_path}")
                    try:
                        tracker.load_history_from_csv(valid_csv_path)
                    except Exception as load_err:
                        smart_print(f"\n[CRITICAL] Failed to load validation history: {load_err}")
                        smart_print("[CRITICAL] Aborting training to prevent data loss/corruption.")
                        raise load_err
                    
                    # Truncate history to remove any stale "future" epochs
                    # (e.g. if we resumed from epoch 10 but history has data up to epoch 15 from a failed run)
                    # We strictly trust the CHECKPOINT's epoch count.
                    original_len = len(tracker.history)
                    tracker.history = [h for h in tracker.history if h.epoch < start_epoch]
                    new_len = len(tracker.history)
                    
                    if new_len < original_len:
                        smart_print(f"[RESUME] Truncated validation history from {original_len} to {new_len} records (removed future epochs).")
                        # [RESUME-FIX] Save truncated history back to disk immediately
                        tracker.save_validation_history()
                        smart_print(f"[RESUME] Saved truncated valid_metrics.csv to disk.")
                    
                    tracker._recompute_ces_scores()
                
                # [RESUME-FIX] Full History Recovery
                # Iterate from 0 to loaded_epoch. If ANY checkpoint exists but is missing from metrics, recover it.
                # This handles:
                # 1. Crash before validation (last epoch missing)
                # 2. Corrupted/Deleted CSV (all epochs missing)
                # 3. Resume from middle of training with partial CSV
                
                smart_print("[RESUME] Checking for missing validation history...")
                
                # Identify missing epochs that HAVE checkpoints
                epochs_to_recover = []
                for ep in range(loaded_epoch + 1):
                    # Check if in tracker
                    in_tracker = any(h.epoch == ep for h in tracker.history)
                    if in_tracker:
                        continue
                        
                    # Check if checkpoint exists
                    ckpt_name = f"epoch_{ep}.pth"
                    # We have `temp_ckpt_dir` variable in scope from earlier in script
                    # Correct variable name for checkpoint dir is `temp_ckpt_dir` (checked in earlier view)
                    ckpt_path = os.path.join(temp_ckpt_dir, ckpt_name)
                    
                    if os.path.exists(ckpt_path):
                        epochs_to_recover.append(ep)
                
                if epochs_to_recover:
                    # DEBUG
                    smart_print(f"[DEBUG] skip_recovery = {skip_recovery}")
                    if skip_recovery:
                        smart_print(f"[RESUME-RECOVERY] Found {len(epochs_to_recover)} missing epochs but --skip-recovery is set. Skipping...")
                    else:
                        smart_print(f"[RESUME-RECOVERY] Found {len(epochs_to_recover)} missing epochs with checkpoints: {epochs_to_recover}")
                        smart_print(f"                  Starting Batch Recovery...")

                    for recover_ep in ([] if skip_recovery else epochs_to_recover):
                        smart_print(f"\n[RECOVERY] ♻️ Recovering Epoch {recover_ep}...")
                        
                        try:
                            # 1. Load Checkpoint
                            ckpt_path = os.path.join(temp_ckpt_dir, f"epoch_{recover_ep}.pth")
                            trainer.observer.load_checkpoint(ckpt_path)

                            # [DEBUG] Check model weight sum to verify it actually changes
                            p_sum = sum(p.sum().item() for p in trainer.observer.mafia_model.parameters())
                            smart_print(f"           [DEBUG] Epoch {recover_ep} Model Param Sum: {p_sum:.6f}")
                            
                            # 2. Set Trainer Epoch
                            trainer._epoch = recover_ep
                            
                            # 3. Run Validation
                            val_result = trainer.validate_epoch(
                                data_tensors=valid_tensors,
                                compute_loss=True,  # Ensure losses are computed for best selection
                                steps=max(1, batches_per_epoch // 4),
                            )
                            
                            # 4. Add to Tracker
                            min_full = trainer.curriculum_warmup_epochs + trainer.curriculum_penalty_rampup
                            is_full_penalty = (recover_ep >= min_full)
                            is_best = tracker.add_epoch(val_result, allow_best_update=is_full_penalty)
                            smart_print(f"           ✅ Recovered Epoch {recover_ep} (Score: {val_result.phase_score:.5f})")

                            # 5. Save incrementally
                            tracker.save_validation_history()

                            # 6. Plots
                            min_best_for_plot = trainer.curriculum_warmup_epochs + trainer.curriculum_penalty_rampup
                            generate_epoch_report(tracker.output_dir, recover_ep, min_best_epoch=min_best_for_plot,
                                                  training_mode=training_mode)

                            # 7. Update Best Checkpoint if needed
                            if is_best:
                                best_ckpt_path = os.path.join(temp_ckpt_dir, "best_checkpoint.pth")
                                trainer.observer.save_checkpoint(best_ckpt_path, epoch=recover_ep)
                                smart_print(f"           🏆 Updated best_checkpoint.pth (Score={val_result.phase_score:.4f})")
                            
                        except Exception as e:
                            smart_print(f"           ❌ Failed to recover Epoch {recover_ep}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Restore state to `loaded_epoch` before continuing (only if recovery happened)
                    if not skip_recovery and epochs_to_recover[-1] != loaded_epoch:
                        smart_print(f"[RECOVERY] 🔄 Restoring observer state to Loaded Epoch {loaded_epoch}...")
                        ckpt_path_final = os.path.join(temp_ckpt_dir, f"epoch_{loaded_epoch}.pth")
                        trainer.observer.load_checkpoint(ckpt_path_final)
                        trainer._epoch = loaded_epoch

                    if not skip_recovery:
                        smart_print("[RESUME-RECOVERY] ✅ All missing history recovered.")
                else:
                    smart_print("[RESUME] History is consistent with checkpoints.")


                # [RESUME-FIX] Train Metrics Recovery
                # Check if train metrics exist for this epoch.
                # If missing, we evaluate on TRAINING set to populate it.
                train_csv_path = os.path.join(iter_output_dir, "train_metrics.csv")
                train_metrics_exist = False
                if os.path.exists(train_csv_path):
                    try:
                        df_t = pd.read_csv(train_csv_path)
                        if "epoch" in df_t.columns and loaded_epoch in df_t["epoch"].values:
                            train_metrics_exist = True
                    except:
                        pass
                
                if not train_metrics_exist:
                    smart_print(f"\n[RESUME-RECOVERY] 🚨 Checkpoint at Epoch {loaded_epoch} exists, but TRAIN metrics missing.")
                    smart_print(f"                  Executing IMMEDIATE metrics recovery on TRAINING SET for Epoch {loaded_epoch}...")
                    
                    try:
                        # Run evaluation on TRAIN set
                        # Note: This gives 'inference' loss, not 'training' loss (with dropout/grad), 
                        # but it's the best proxy we have for a lost log.
                        # We use a subset of training data if it's too huge? Or full? 
                        # train_epoch iterates all. validate_epoch can take full train_tensors.
                        # Let's use full train_tensors.
                        train_eval_res = trainer.validate_epoch(
                             data_tensors=train_tensors,
                             steps=batches_per_epoch, # Use same steps as training? Or just full? validate_epoch default iterates all if steps large?
                             # Actually validate_epoch implementation:
                             # It samples `steps` batches.
                             # So yes, passing batches_per_epoch is correct to cover rough size of data or more.
                             compute_loss=True  # Log training losses
                        )
                        
                        # Convert to dict and save
                        t_dict = asdict(train_eval_res)
                        # Remove CES stuff
                        keys_to_remove = [k for k in t_dict.keys() if "ces_score" in k or "ces_rank" in k]
                        for k in keys_to_remove:
                            del t_dict[k]
                        
                        df_t_new = pd.DataFrame([t_dict])
                        if "epoch" in df_t_new.columns:
                            df_t_new["epoch"] = df_t_new["epoch"].astype(int)
                            
                        header = not os.path.exists(train_csv_path)
                        df_t_new.to_csv(train_csv_path, mode='a', header=header, index=False, float_format='%.5f')
                        smart_print(f"[RESUME-RECOVERY] ✅ Recovered train_metrics.csv for Epoch {loaded_epoch}.")
                    except Exception as rec_err:
                        smart_print(f"[RESUME-RECOVERY] ⚠️ Failed to recover train metrics: {rec_err}")
                        smart_print(f"                  Skipping recovery and proceeding with training.")


                # [RESUME-FIX] Truncate train_metrics.csv if exists
                train_csv_path = os.path.join(iter_output_dir, "train_metrics.csv")
                if os.path.exists(train_csv_path):
                    smart_print(f"[RESUME] Checking train history from: {train_csv_path}")
                    try:
                        df_train = pd.read_csv(train_csv_path)
                        if "epoch" in df_train.columns:
                            # Convert epoch column to numeric if mixed
                            df_train["epoch"] = pd.to_numeric(df_train["epoch"], errors='coerce')
                            df_train = df_train.dropna(subset=["epoch"])
                            
                            original_len_t = len(df_train)
                            df_train = df_train[df_train["epoch"] < start_epoch]
                            new_len_t = len(df_train)
                            
                            if new_len_t < original_len_t:
                                smart_print(f"[RESUME] Truncating train history from {original_len_t} to {new_len_t} records (removed future epochs).")
                                
                                # [SAFETY-BACKUP] Backup before overwrite
                                import shutil
                                backup_path = train_csv_path + ".bak"
                                shutil.copy2(train_csv_path, backup_path)
                                smart_print(f"[RESUME] Created backup at: {backup_path}")

                                # [SAFETY-CHECK] Prevent wiping history if we are resuming late
                                if new_len_t == 0 and start_epoch > 0:
                                    smart_print(f"[CRITICAL] 🚨 Resume would wipe ALL training history despite start_epoch={start_epoch}!")
                                    smart_print(f"           Aborting truncation. Restoring from backup.")
                                    shutil.copy2(backup_path, train_csv_path)
                                else:
                                    df_train.to_csv(train_csv_path, index=False)
                    except Exception as e:
                        smart_print(f"[WARN] Failed to truncate train_metrics.csv: {e}")

                # [RESUME-FIX] Truncate trajectory_details.csv if exists
                traj_csv_path = os.path.join(iter_output_dir, "trajectory_details.csv")
                if os.path.exists(traj_csv_path):
                    smart_print(f"[RESUME] Checking trajectory details from: {traj_csv_path}")
                    try:
                        df_traj = pd.read_csv(traj_csv_path)
                        if "epoch" in df_traj.columns:
                            df_traj["epoch"] = pd.to_numeric(df_traj["epoch"], errors='coerce')
                            df_traj = df_traj.dropna(subset=["epoch"])

                            original_len_traj = len(df_traj)
                            df_traj = df_traj[df_traj["epoch"] < start_epoch]
                            new_len_traj = len(df_traj)

                            if new_len_traj < original_len_traj:
                                smart_print(f"[RESUME] Truncating trajectory_details from {original_len_traj} to {new_len_traj} records (removed future epochs).")
                                
                                # [SAFETY-BACKUP]
                                traj_backup = traj_csv_path + ".bak"
                                shutil.copy2(traj_csv_path, traj_backup)
                                
                                if new_len_traj == 0 and start_epoch > 0:
                                     smart_print(f"[CRITICAL] 🚨 Resume would wipe trajectory history! Restoring from backup.")
                                     shutil.copy2(traj_backup, traj_csv_path)
                                else:
                                     df_traj.to_csv(traj_csv_path, index=False)
                        else:
                            # Old format without epoch column - delete entire file
                            smart_print(f"[RESUME] trajectory_details.csv missing 'epoch' column (old format). Deleting file.")
                            os.remove(traj_csv_path)
                    except Exception as e:
                        smart_print(f"[WARN] Failed to truncate trajectory_details.csv: {e}")

                smart_print(f"[RESUME] Will continue from epoch {start_epoch}")
            except Exception as e:
                smart_print(f"[WARN] Failed to load checkpoint: {e}")
                # If critical error loading history, we already raised. 
                # If just checkpoint load fail, maybe safer to crash than start from 0 if files exist?
                # For now, let's allow fail->start 0 coupled with warning, 
                # BUT if we start at 0, we should probably backup existing metric file?
                # If we fail to resume but have data, we might overwrite it.
                valid_csv_path_check = os.path.join(iter_output_dir, "valid_metrics.csv")
                if os.path.exists(valid_csv_path_check):
                     smart_print("[CRITICAL] Checkpoint load failed but valid_metrics.csv exists.")
                     smart_print("[CRITICAL] Starting from Epoch 0 would overwrite it. Aborting for safety.")
                     raise e
                start_epoch = 0

    # Training Loop
    if start_epoch >= num_epochs:
        smart_print(f"[OFFLINE] Iteration already completed ({start_epoch} >= {num_epochs}). Skipping training loop.")

    # IMPORTANT: Sync trainer._epoch with start_epoch - 1 for correct epoch numbering
    # train_epoch() increments _epoch at the start (self._epoch += 1), so we init to start-1.
    # This ensures the first trained epoch has index `start_epoch`.
    # e.g., start_epoch=0 -> init -1 -> first run becomes 0 (0-based indexing).
    # e.g., resume start=10 -> init 9 -> next run becomes 10.
    trainer._epoch = start_epoch - 1
    if start_epoch > 0:
        # Also update current_global_step based on previously completed epochs
        current_global_step = global_step_start + (start_epoch * (batches_per_epoch or 10))
        if verbose:
            smart_print(f"[RESUME] Synced trainer._epoch to {trainer._epoch} (will become {start_epoch} after train_epoch())")
            smart_print(f"[RESUME] Synced global_step to {current_global_step}")

    # Show chart location at training start (inside iteration folder)
    plots_dir = os.path.join(iter_output_dir, "plots")
    if verbose:
        smart_print(f"\n📊 Charts will be saved to: {os.path.abspath(plots_dir)}")
        smart_print(f"   (Updated after each batch for real-time progress)\n")

    # Wrap loop in try-finally to ensure validation metrics are saved
    try:
        for epoch in range(start_epoch, num_epochs):
            # Create per-batch visualization callback
            # Calculate min_best_epoch for plotting (Curriculum Logic)
            min_best_epoch_val = trainer.curriculum_warmup_epochs + trainer.curriculum_penalty_rampup
            
            def on_batch_viz(batch_idx, total_batches):
                # Generate charts after each batch for real-time progress
                try:
                    generate_epoch_report(tracker.output_dir, epoch, min_best_epoch=min_best_epoch_val,
                                          training_mode=training_mode)
                except Exception:
                    pass  # Silent fail - don't interrupt training
            
            # Update res_root so that detailed logging goes to the correct iteration folder
            config.res_root = iter_output_dir
            
            # 1. Train
            # Use distinct name for batch-level metrics log
            batch_log_path = os.path.join(iter_output_dir, "batch_training_log.csv")
            
            if verbose:
                smart_print(f"\n[OFFLINE] Used {batches_per_epoch} batches per epoch.")
                smart_print(f"[OFFLINE] Starting Epoch {epoch} Training...")

            train_res = trainer.train_epoch(
                data_tensors=train_tensors,
                steps_per_epoch=batches_per_epoch,
                writer=writer,
                global_step_offset=current_global_step,
                on_batch_done=on_batch_viz,  # Real-time chart updates
                log_file=batch_log_path,      # Save batch-level dynamics
                verbose=verbose,            # Pass verbose flag
                training_mode=training_mode,  # Pass training mode ("MACRO_ONLY", "SELECTION_ONLY")
                epoch=epoch,  # Sync epoch display with outer loop
            )
            current_global_step += batches_per_epoch

            # Save training metrics to CSV (filtered by training phase)
            train_metrics_path = os.path.join(iter_output_dir, "train_metrics.csv")
            try:
                # Convert ObserverValidationResult to dict (filtered by phase)
                train_dict = train_res.to_dict(filter_by_phase=True)

                # For training metrics, also remove CES columns (always 0.0 during training)
                train_dict = {k: v for k, v in train_dict.items()
                              if not (k.startswith("ces_score") or k.startswith("ces_rank"))}

                # Append to CSV
                df_train = pd.DataFrame([train_dict])

                # [FORMAT-FIX] Ensure epoch is integer and consistent numbering
                if "epoch" in df_train.columns:
                    df_train["epoch"] = df_train["epoch"].astype(int)

                header = not os.path.exists(train_metrics_path)
                df_train.to_csv(train_metrics_path, mode='a', header=header, index=False, float_format='%.5f')
            except Exception as e:
                smart_print(f"[WARN] Failed to save train metrics: {e}")

            # 2. Save checkpoint IMMEDIATELY after training (before validation to prevent loss)
            # Always save latest checkpoint for resume (overwrites previous)
            latest_ckpt_path = os.path.join(temp_ckpt_dir, "latest_checkpoint.pth")
            trainer.observer.save_checkpoint(latest_ckpt_path, epoch=epoch)

            # Always save epoch checkpoint (Full History)
            epoch_ckpt_path = os.path.join(temp_ckpt_dir, f"epoch_{epoch}.pth")
            trainer.observer.save_checkpoint(epoch_ckpt_path, epoch=epoch)

            if verbose:
                smart_print(f"      [CHECKPOINT] Saved epoch {epoch} checkpoint before validation")

            # 3. Validate (returns ObserverValidationResult)
            val_result = trainer.validate_epoch(
                data_tensors=valid_tensors,
                compute_loss=True,
                steps=2,  # Fixed 2 steps - đủ cover validation data, nhanh hơn ~2-3x
            )

            # 3. Track metrics and compute CES
            # Restrict "Best Model" updates to epochs with full penalty coefficients (lambda >= 1.0)
            # MACRO_ONLY: No curriculum learning, always allow best checkpoint updates
            # SELECTION_ONLY: Use curriculum logic (block updates when lambda < 1.0)
            if training_mode == MafiaTrainMode.MACRO_ONLY:
                is_full_penalty = True  # Always allow updates in MACRO_ONLY
            else:
                current_lambda = getattr(trainer, "_current_lambda_epoch", 1.0)
                is_full_penalty = current_lambda >= 0.999

            is_best = tracker.add_epoch(val_result, allow_best_update=is_full_penalty)

            if verbose and not is_full_penalty and training_mode == MafiaTrainMode.SELECTION_ONLY:
                smart_print(f"      [INFO] Best Checkpoint update BLOCKED (Lambda {current_lambda:.3f} < 1.0)")
            
            # 4. Log to TensorBoard
            if writer:
                writer.add_scalar("Valid/Loss/Total", val_result.loss_total, epoch)
                writer.add_scalar("Valid/Loss/PG", val_result.loss_pg, epoch)
                writer.add_scalar("Valid/Loss/Risk", val_result.loss_risk, epoch)
                writer.add_scalar("Valid/Loss/Dir", val_result.loss_dir, epoch)
                writer.add_scalar("Valid/Metrics/Sharpe", val_result.topk_sharpe_ratio, epoch)
                writer.add_scalar("Valid/Metrics/Dir_F1", val_result.direction_f1_macro, epoch)
                writer.add_scalar("Valid/Metrics/Risk_MSE", val_result.risk_mse, epoch)
                writer.add_scalar("Valid/Metrics/Risk_Corr", val_result.risk_correlation, epoch)
                writer.add_scalar("Valid/Metrics/Risk_Corr_Loss", 1.0 - val_result.risk_correlation, epoch)
                writer.add_scalar("Valid/Config/Risk_Alpha", trainer.risk_corr_alpha, epoch)
                writer.add_scalar("Valid/Metrics/PhaseScore", val_result.phase_score, epoch)

            # 5. Update Live Dashboard
            update_observer(
                loss=train_res.loss_total,
                loss_eta=val_result.risk_mse,
                loss_dir=val_result.loss_dir,
            )

            update_walkforward_score(
                current_score=val_result.phase_score,
                best_score=tracker.best_score,
                best_epoch=tracker.best_epoch if tracker.best_epoch is not None else epoch,
            )

            # 6. Save best checkpoint (after validation determines is_best)
            if is_best:
                best_ckpt_path = os.path.join(temp_ckpt_dir, "best_checkpoint.pth")
                trainer.observer.save_checkpoint(best_ckpt_path, epoch=epoch)

            # 7. Progress logging
            if verbose:
                smart_print(val_result.summary_str())

            # Save validation history after each epoch to support resume
            tracker.save_validation_history()
            
            # 8. Generate Real-time Visualization (Insights)
            if verbose:
                smart_print(f"      Running Visualization for Epoch {epoch}...")
            try:
                generate_epoch_report(tracker.output_dir, epoch, min_best_epoch=min_best_epoch_val,
                                      training_mode=training_mode)
                
                # Log generated charts to TensorBoard
                if trainer.tb_logger is not None and trainer.tb_logger.log_images:
                    plots_dir = os.path.join(tracker.output_dir, "plots")
                    
                    # Log dashboard chart
                    dashboard_path = os.path.join(plots_dir, "dashboard_latest.png")
                    if os.path.exists(dashboard_path):
                        trainer.tb_logger.log_image("charts/dashboard", dashboard_path, epoch, phase="valid")
                    
                    # Log loss history chart
                    loss_history_path = os.path.join(plots_dir, "loss_history.png")
                    if os.path.exists(loss_history_path):
                        trainer.tb_logger.log_image("charts/loss_history", loss_history_path, epoch, phase="valid")

                    # Flush to ensure images are written
                    trainer.tb_logger.flush()
                    
            except Exception as e:
                smart_print(f"      [WARN] Visualization failed: {e}")

    finally:
        # Save validation history to CSV (ensure save on interrupt/error)
        csv_path = tracker.save_validation_history()
        if verbose:
            smart_print(f"\n[OFFLINE] Validation history saved to: {csv_path}")
    
    # Get best checkpoint info (Correctly filters for Full Penalty epochs via ValidationTracker)
    best_info = tracker.get_best_checkpoint_info()
    best_epoch = best_info["epoch"]
    best_score = best_info["phase_score"]

    if verbose:
        score_name = "MACRO_SCORE" if training_mode == "MACRO_ONLY" else "Sharpe"
        smart_print(f"\n[OFFLINE] Best Checkpoint Selection for Iter {iteration}:")
        smart_print(f"  Best Epoch: {best_epoch}")
        smart_print(f"  {score_name}: {best_score:.4f}")
        if training_mode == "MACRO_ONLY":
            smart_print(f"  Dir_F1: {best_info['direction_f1']:.3f}")
            smart_print(f"  Risk_MSE: {best_info['risk_mse']:.4f}")
        else:
            smart_print(f"  Sharpe: {best_info['sharpe_ratio']:.3f}")

    # [LOG-FIX] Save Best Checkpoint Metrics to FILE
    # Find the full result object for the best epoch
    best_result = next((h for h in tracker.history if h.epoch == best_epoch), None)
    if best_result:
        best_stats_path = os.path.join(iter_output_dir, "best_checkpoint_stats.csv")
        try:
            best_dict = best_result.to_dict()
            df_best = pd.DataFrame([best_dict])
            
            # Format
            if "epoch" in df_best.columns:
                df_best["epoch"] = df_best["epoch"].astype(int)
            
            # Reorder columns (CES first) if possible - match valid_metrics order
            # We can just read valid_metrics.csv to get columns order? 
            # Or simplified: use the same dict order which is now updated in class definition.
            
            df_best.to_csv(best_stats_path, index=False, float_format='%.5f')
            smart_print(f"[OFFLINE] 📝 Logged best checkpoint stats to: {best_stats_path}")
        except Exception as e:
            smart_print(f"[WARN] Failed to log best stats: {e}")
    
    # Copy best checkpoint to final location
    final_ckpt_path = os.path.join(checkpoint_dir, f"observer_best_{valid_year}.pth")
    best_temp_path = os.path.join(temp_ckpt_dir, f"epoch_{best_epoch}.pth")
    
    if os.path.exists(best_temp_path):
        trainer.observer.load_checkpoint(best_temp_path)
        trainer.observer.save_checkpoint(final_ckpt_path, epoch=best_epoch, extra_data={"phase_score": best_score})
    else:
        # Fallback: save current state if no best checkpoint found
        trainer.observer.save_checkpoint(final_ckpt_path, epoch=num_epochs-1, extra_data={"phase_score": best_score})
    
    # [PERSISTENCE-FIX] Do NOT delete temp checkpoints. User wants full history.
    # import shutil
    # try:
    #     shutil.rmtree(temp_ckpt_dir)
    # except:
    #     pass
    smart_print(f"[OFFLINE] All epoch checkpoints preserved in: {temp_ckpt_dir}")

    return final_ckpt_path




def run_offline_observer_training(
    start_year: int = 2015,
    first_infer_year: int = 2018,
    last_infer_year: int = 2024,
    valid_duration_years: int = 1,  # ADDED: Multi-year validation support
    output_dir: str = "./observer_offline",
    num_epochs: Optional[int] = None,
    batches_per_epoch: Optional[int] = None,
    traj_len: Optional[int] = None,
    seed: int = 2025,
    log_details: bool = False,
    rebalance_interval: Optional[int] = None,
    verbose: bool = True,
    max_iterations: Optional[int] = None,
    batch_size: Optional[int] = None,
    mode: str = "FULL",  # ADDED
    init_checkpoint: Optional[str] = None, # ADDED: Allow manual init checkpoint (e.g. for testing)
    mps_batch_size: int = 64,  # ADDED: MPS-specific batch size limit
    low_memory: bool = False,  # ADDED: Low memory mode for smaller Macs
    sampling_strategy: str = "auto",  # ADDED: 'auto', 'uniform', 'hybrid_weighted'
    debug_logging: bool = False,  # ADDED: Per-timestep terminal logging
    skip_recovery: bool = False,  # ADDED: Skip validation history recovery on resume
) -> List[Dict]:
    """
    Run full walk-forward offline observer training.

    Per Spec §6 Training Duration:
    - Base training (iter 0): 50 epochs (default)
    - Finetune (iter > 0): 5 epochs (default)
    - Steps per epoch: auto-computed as ceil((Len(Data) - T_m - h) / Batch_Size)

    Args:
        start_year: Start year for expanding training window
        first_infer_year: First inference year
        last_infer_year: Last inference year
        output_dir: Output directory
        num_epochs: Override epochs (default: None, use config per iteration)
        batches_per_epoch: Override steps per epoch (default: None, auto-compute)
        traj_len: Override trajectory length T_m
        seed: Random seed
        verbose: Print progress

    Returns:
        List of results for each iteration
    """
    # Set random seeds
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)

    # Mode-dependent validation duration:
    # - MACRO_ONLY: always 1 year validation
    # - SELECTION_ONLY: 2 years validation (for stock selection robustness)
    if mode == "SELECTION_ONLY":
        effective_valid_duration = 2
        if verbose and valid_duration_years != 2:
            smart_print(f"[CONFIG] SELECTION_ONLY mode: Using 2-year validation (overriding {valid_duration_years})")
    else:
        effective_valid_duration = 1
        if verbose and valid_duration_years != 1:
            smart_print(f"[CONFIG] MACRO_ONLY mode: Using 1-year validation (overriding {valid_duration_years})")

    # Build schedule
    schedule = build_expanding_schedule(
        start_year, first_infer_year, last_infer_year, effective_valid_duration
    )
    num_iterations = len(schedule)
    if max_iterations is not None and max_iterations > 0:
        schedule = schedule[:max_iterations]
        num_iterations = len(schedule)
        if verbose:
            smart_print(f"[CONFIG] Limiting training to first {num_iterations} iterations")

    if verbose:
        smart_print("\n" + "#" * 70)
        smart_print("# OFFLINE BATCH OBSERVER TRAINING (Spec §6/§7 Compliant)")
        smart_print("#" * 70)
        smart_print(f"  Mode: {mode}")
        smart_print(f"  Start year: {start_year}")
        smart_print(f"  First infer year: {first_infer_year}")
        smart_print(f"  Last infer year: {last_infer_year}")
        smart_print(f"  Valid duration: {effective_valid_duration} year(s) {'(auto: SELECTION_ONLY)' if mode == 'SELECTION_ONLY' else ''}")
        smart_print(f"  Total iterations: {num_iterations}")
        smart_print(f"  Epochs/iteration: {num_epochs}")
        smart_print(f"  Batches/epoch: {batches_per_epoch}")
        smart_print(f"  Output dir: {output_dir}")
        smart_print(f"{'#' * 70}\n")

    # Create output directories with phase-aware structure
    os.makedirs(output_dir, exist_ok=True)

    # Build phase-aware directories based on training mode
    phase_subdir = get_phase_subdir(mode)
    results_dir = os.path.join(output_dir, phase_subdir)
    os.makedirs(results_dir, exist_ok=True)

    # Checkpoints separated by phase (macro/selection)
    ckpt_phase = "macro" if mode == "MACRO_ONLY" else "selection"
    checkpoint_dir = os.path.join(output_dir, "checkpoints", ckpt_phase)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load config and data
    # Pass create_dirs=False to prevent auto-creation of TD3 directories
    config = Config(create_dirs=False)
    config.seed = seed
    config.res_root = results_dir  # Phase-aware results directory
    if traj_len:
        config.mafia_trajectory_length = traj_len
    if rebalance_interval is None:
        rebalance_interval = getattr(config, "mafia_topk_rebalance_interval", 14)
    if batch_size is not None:
        config.mafia_batch_size = batch_size
        if verbose:
            smart_print(f"[CONFIG] Overriding batch size: {batch_size}")

    # [MPS-FIX] Set MPS batch size limit in config
    config.mps_max_batch_size = mps_batch_size

    # [CRITICAL-FIX] Set training mode BEFORE creating observer
    # This ensures loss aggregation respects the mode (PG loss disabled in MACRO_ONLY, etc.)
    # Bug: Previously config.mafia_train_mode stayed at default FULL, causing wrong loss calculation
    config.mafia_train_mode = mode
    if verbose:
        smart_print(f"[CONFIG] Training mode set: {mode}")

    # [MPS-FIX] Low memory mode for smaller Macs (8GB RAM)
    if low_memory:
        config.mafia_batch_size = 8
        config.mafia_trajectory_length = 64
        config.mps_max_batch_size = 8
        if verbose:
            smart_print("[CONFIG] Low memory mode ENABLED (batch=8, traj_len=64)")

    # Enable trajectory logging if requested
    if log_details:
        config.log_trajectory_details = True
        if verbose:
            smart_print("[CONFIG] Detailed trajectory logging ENABLED")
            smart_print(f"  Output: {results_dir}/trajectory_details.csv\n")

    # Enable per-timestep debug logging on terminal
    if debug_logging:
        config.mafia_debug_logging = True
        if verbose:
            smart_print("[CONFIG] Per-timestep debug logging ENABLED (terminal output)")

    # Initialize TensorBoard Writer (inside phase results directory)
    log_dir = os.path.abspath(os.path.join(results_dir, "tb_logs"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    global_step_counter = 0

    # Load full dataset
    if verbose:
        smart_print("[OFFLINE] Loading stock data...")

    try:
        stock_data = load_mafia_data(config)
        stock_list = sorted(stock_data["stock"].unique().tolist())

        # Load Market Data (VNINDEX)
        index_file_name = getattr(config, "index_data_file", "vnindex_data.csv")
        market_file = os.path.join(getattr(config, "dataDir", "./data"), index_file_name)
        if os.path.exists(market_file):
            market_data = pd.read_csv(market_file, parse_dates=["date"])
            if verbose:
                smart_print(f"[OFFLINE] Loaded Market Data (VNINDEX): {len(market_data)} rows")
        else:
            # SAFETY CRITICAL: Do NOT fallback to synthetic data
            # Missing market data causes silent labeling failures (all Side)
            raise FileNotFoundError(
                f"[CRITICAL] Market Data (VNINDEX) not found at {market_file}. "
                f"Aborting to prevent training on invalid synthetic labels."
            )

        if verbose:
            smart_print(
                f"[OFFLINE] Loaded {len(stock_data)} rows, {len(stock_list)} stocks"
            )
    except Exception as e:
        smart_print(f"[ERROR] Failed to load stock data: {e}")
        raise

    # Create observer and trainer
    # Device selection: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if th.cuda.is_available():
        device = th.device("cuda")
    elif th.backends.mps.is_available():
        device = th.device("mps")
    else:
        device = th.device("cpu")
    smart_print(f"[DEVICE] Training on device: {device}")
    
    # Determine action_dim (N) from loaded data
    action_dim = len(stock_list)
    
    observer = MAFIAObserver(config=config, action_dim=action_dim)
    
    # Create TensorBoard logger if enabled
    tb_logger = None
    if getattr(config, "use_tensorboard", True) and create_tensorboard_logger is not None:
        tb_base_dir = os.path.join(output_dir, getattr(config, "tensorboard_log_dir", "tensorboard"))
        tb_logger = create_tensorboard_logger(
            base_dir=tb_base_dir,
            run_name=f"observer_offline_{start_year}_{last_infer_year}",
            config=config,
            enabled=True,
        )
        smart_print(f"[TensorBoard] Logger created at: {tb_base_dir}")
    
    trainer = ObserverOfflineBatchTrainer(
        config=config,
        observer=observer,
        device=device,
        tensorboard_logger=tb_logger,
    )

    # Prepare full data tensors
    full_start = pd.Timestamp(f"{start_year}-01-01")
    full_end = pd.Timestamp(f"{last_infer_year}-12-31")

    data_tensors = trainer.prepare_data_tensors(
        data=stock_data,
        stock_list=stock_list,
        start_date=full_start,
        end_date=full_end,
        market_data=market_data,
    )

    # CRITICAL: Keep stock_data for Inference phase!
    # del stock_data 
    if 'market_data' in locals() and market_data is not None:
        del market_data
    # gc.collect() # Optional

    if verbose:
        smart_print(
            f"[OFFLINE] Prepared tensors: T_total={data_tensors['T_total']}, N={data_tensors['N']}"
        )

    # Run iterations
    results = []
    prev_checkpoint = None
    
    # Initialize quarterly state tracker
    # Format: {year: {q: path}}
    yearly_state_tracker = {y: {} for y in range(first_infer_year, last_infer_year + 1)}
    
    # Import for Inference
    from utils.tradeEnv import StockPortfolioEnv
    # NOTE: RegimeShiftDetector logic moved to observer_offline_trainer.generate_inference_state

    for sched in schedule:
        iteration = sched["iter_index"]
        
        # Determine training mode
        is_cold_start = (iteration == 0)
        training_mode = "COLD START - Base Training" if is_cold_start else "FINETUNE (Warm Start)"
        num_train_epochs = num_epochs if num_epochs else (50 if is_cold_start else 20)

        if verbose:
            smart_print(f"\n{'=' * 70}")
            if is_cold_start:
                smart_print(f"ITERATION {iteration + 1}/{num_iterations} ({training_mode})")
            else:
                smart_print(f"ITERATION {iteration + 1}/{num_iterations} (FINETUNE - Warm Start)")
            smart_print(f"{'=' * 70}")
            smart_print(f"  📚 Training Window (Expanding): {sched['train_label']}")
            smart_print(f"  ✅ Validation Period:           {sched['valid_label']}")
            # YEARLY schedule - show year only
            smart_print(f"  🎯 Inference Target (Live):     {sched['infer_year']} (Full Year)")
            smart_print(f"")

        # === RESUME / TRAIN LOGIC ===
        # NAMING CONVENTION FOR LIVE INFERENCE:
        # FULL: Observer_Infer_{year}_Q{q}.pth
        # MACRO: Observer_Macro_{year}_Q{q}.pth
        # SELECTION: Observer_Selection_{year}_Q{q}.pth
        
        ckpt_prefix = "Observer_Infer"
        if mode == "MACRO_ONLY":
            ckpt_prefix = "Observer_Macro"
        elif mode == "SELECTION_ONLY":
            ckpt_prefix = "Observer_Selection"
            
        infer_year = sched["infer_year"]
        # YEARLY schedule - checkpoint named by year only
        final_iter_ckpt = os.path.join(checkpoint_dir, f"{ckpt_prefix}_{infer_year}.pth")
        
        # Logic for Phase 1B (Selection): Must load corresponding Macro Checkpoint as init
        # AND chain trainable components from previous SELECTION iteration
        iter_init_ckpt = init_checkpoint  # Use passed argument if provided
        iter_prev_selection_ckpt = None  # Previous SELECTION checkpoint for trainable params

        if mode == "SELECTION_ONLY":
            # YEARLY schedule - checkpoint named by year only
            macro_ckpt_name = f"Observer_Macro_{infer_year}.pth"
            # Look in macro checkpoint directory (checkpoints/macro/), not selection
            macro_checkpoint_dir = os.path.join(output_dir, "checkpoints", "macro")
            macro_ckpt_path = os.path.join(macro_checkpoint_dir, macro_ckpt_name)

            # [FIX] Respect --init-checkpoint if passed and exists
            if iter_init_ckpt and os.path.exists(iter_init_ckpt):
                smart_print(f"[CONFIG] Phase 1B: Using manual init_checkpoint: {iter_init_ckpt}")
            elif os.path.exists(macro_ckpt_path):
                iter_init_ckpt = macro_ckpt_path
                smart_print(f"[CONFIG] Phase 1B: Found MACRO checkpoint: {macro_ckpt_name}")
            else:
                # [CRITICAL] MACRO checkpoint is REQUIRED for SELECTION_ONLY
                raise FileNotFoundError(
                    f"[CRITICAL] SELECTION_ONLY requires MACRO checkpoint.\n"
                    f"  Expected at: {macro_ckpt_path}\n"
                    f"  Or pass --init-checkpoint <path>\n"
                    f"Run MACRO_ONLY training first for year {infer_year}!"
                )

            # Find previous SELECTION checkpoint for trainable params (knowledge transfer)
            if iteration > 0:
                prev_infer_year = infer_year - 1
                prev_selection_name = f"Observer_Selection_{prev_infer_year}.pth"
                prev_selection_path = os.path.join(checkpoint_dir, prev_selection_name)

                if os.path.exists(prev_selection_path):
                    iter_prev_selection_ckpt = prev_selection_path
                    smart_print(f"[CONFIG] Phase 1B: Found prev SELECTION checkpoint: {prev_selection_name}")
                    smart_print(f"         Will merge: Frozen→MACRO_{infer_year}, Trainable→SELECTION_{prev_infer_year}")
                else:
                    smart_print(f"[WARN] Phase 1B: No prev SELECTION checkpoint found at {prev_selection_path}")
                    smart_print(f"       Trainable params will be initialized from MACRO checkpoint.")
        
        # 1. Train if not already done
        # [FIX] Check both checkpoint exists AND training actually completed (all epochs trained)
        valid_year = sched["valid_year"]
        iter_output_dir = os.path.join(results_dir, f"iter_{iteration}_valid_{valid_year}")
        train_csv_path = os.path.join(iter_output_dir, "train_metrics.csv")

        training_complete = False
        epochs_trained = 0
        if os.path.exists(final_iter_ckpt) and os.path.exists(train_csv_path):
            try:
                df = pd.read_csv(train_csv_path)
                epochs_trained = len(df)
                if epochs_trained >= num_train_epochs:
                    training_complete = True
            except Exception as e:
                smart_print(f"[WARN] Failed to read train_metrics.csv: {e}")

        if training_complete:
             smart_print(f"[RESUME] Found completed training ({epochs_trained}/{num_train_epochs} epochs)")
             smart_print(f"         Checkpoint: {final_iter_ckpt}")
             smart_print(f"         Skipping training step.")
             prev_checkpoint = final_iter_ckpt
        else:
             # Need to train: either no checkpoint or incomplete training
             if os.path.exists(final_iter_ckpt):
                 # Checkpoint exists but training incomplete - remove stale checkpoint
                 smart_print(f"[RESUME] ⚠️ Incomplete training detected ({epochs_trained}/{num_train_epochs} epochs)")
                 smart_print(f"         Removing stale checkpoint and retraining...")
                 os.remove(final_iter_ckpt)

             # CALL TRAINER
             # Note: train_observer_offline_iteration saves its own "best" checkpoint internally to a temp name?
             # No, it returns a path to "observer_best_{valid_year}.pth" (or similar).
             # We need to Rename/Copy it to our Target Name.

             best_ckpt_path = train_observer_offline_iteration(
                config=config,
                schedule_entry=sched,
                trainer=trainer,
                data_tensors=data_tensors,
                checkpoint_dir=checkpoint_dir,
                results_dir=results_dir,  # Phase-aware results directory
                num_epochs=num_train_epochs,
                batches_per_epoch=batches_per_epoch,
                prev_checkpoint=prev_checkpoint,
                writer=writer,
                global_step_start=global_step_counter,
                verbose=verbose,
                training_mode=mode,
                init_checkpoint=iter_init_ckpt,  # MACRO checkpoint (frozen params)
                prev_selection_checkpoint=iter_prev_selection_ckpt,  # Prev SELECTION (trainable params)
                sampling_strategy=sampling_strategy,  # ADDED: Pass sampling strategy
                skip_recovery=skip_recovery,  # ADDED: Skip validation history recovery
             )
             
             # Rename/Copy to Final Inference Name
             if best_ckpt_path and os.path.exists(best_ckpt_path):
                 import shutil
                 shutil.copy2(best_ckpt_path, final_iter_ckpt)
                 smart_print(f"[SAVE] ✅ Saved Inference Checkpoint: {final_iter_ckpt}")
                 prev_checkpoint = final_iter_ckpt
             else:
                 smart_print(f"[ERROR] Training failed to produce checkpoint.")
                 continue

             # Record Result
             results.append({
                 "iteration": iteration,
                 "valid_year": sched["valid_year"], # For record keeping
                 "train_range": sched["train_label"],
                 "final_checkpoint": final_iter_ckpt,
                 "status": "success",
                 # We can try to get metrics from trainer or reload them.
                 # Ideally train_observer_offline_iteration returns metrics too?
                 # For now, just mark success.
                 "best_metrics": {} # detailed metrics loading requires parsing csv, omitted for now
             })

             global_step_counter += (num_train_epochs * (batches_per_epoch or 10))

    return results   # End of loop


    # Save summary
    import json

    summary_file = os.path.join(output_dir, "offline_training_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        smart_print(f"  Summary saved to: {summary_file}")
        
    # Close TensorBoard logger
    if trainer.tb_logger is not None:
        trainer.tb_logger.close()
        
    # Close writer
    writer.close()

    if verbose:
        smart_print(f"\n{'=' * 70}")
        smart_print("📊 OBSERVER TRAINING SUMMARY (Stage 1)")
        smart_print(f"{'=' * 70}")
        smart_print(f"  Total Iterations:  {num_iterations}")
        successful = sum(1 for r in results if r['status'] == 'success')
        smart_print(f"  Successful:        {successful}/{num_iterations}")

        # Best checkpoint info
        best_phase_score = -1.0
        best_year = None
        best_metrics = {}
        for r in results:
            if r['status'] == 'success':
                metrics = r.get('best_metrics', {})
                score = metrics.get('phase_score', 0.0)
                if score > best_phase_score:
                    best_phase_score = score
                    best_year = r.get('valid_year')
                    best_metrics = metrics

        if best_year:
            score_name = "MACRO_SCORE" if mode == "MACRO_ONLY" else "Sharpe"
            smart_print(f"\n  🏆 Best Checkpoint: Ckpt_Best_{best_year}")
            smart_print(f"     {score_name}:    {best_phase_score:.4f}")
            if mode == "MACRO_ONLY":
                smart_print(f"     Direction F1:   {best_metrics.get('direction_f1_macro', 0.0):.4f}")
                smart_print(f"     Risk MSE:       {best_metrics.get('risk_mse', 0.0):.6f}")
            else:
                smart_print(f"     Sharpe Ratio:   {best_metrics.get('topk_sharpe_ratio', 0.0):.4f}")

        smart_print(f"\n  📁 Summary saved to: {summary_file}")
        smart_print(f"  📌 Next Step: Run Stage 2 (TD3 Training) with frozen Observer")
        smart_print(f"{'=' * 70}\n")

    # =========================================================================
    # AUTO-GENERATE CHARTS (Spec Requirement)
    # =========================================================================
    try:
        from RL_controller.plotting_utils import create_all_charts
        
        if verbose:
            smart_print("\n[CHARTS] Auto-generating training graphs...")

        # 1. Prepare Walk-Forward Summary
        iterations_summary = []
        last_valid_csv = None
        
        for r in results:
            if r["status"] == "success":
                metrics = r.get("best_metrics", {})
                
                # Format train_range (e.g. "2015-01-01 -> 2017-06-30" to "2015-2017")
                t_range = r.get("train_range", "N/A")
                try:
                   start_y = t_range.split(" -> ")[0].split("-")[0]
                   end_y = t_range.split(" -> ")[1].split("-")[0]
                   short_range = f"{start_y}-{end_y}"
                except:
                   short_range = t_range

                iterations_summary.append({
                    "year": r.get("valid_year"),
                    "phase_score": metrics.get("phase_score", 0.0),
                    "train_range": short_range,
                    "sharpe": metrics.get("topk_sharpe_ratio", 0.0),
                    "dir_f1": metrics.get("direction_f1_macro", 0.0),
                    "risk_mse": metrics.get("risk_mse", 0.0),
                })
                
                # Track last valid metrics file
                # Path: ./observer_offline/iter_X_.../valid_metrics.csv
                # We can reconstruct it or find it. r does not save the csv path directly, 
                # but we know the dir structure.
                # Actually, wait, `train_observer_offline_iteration` returns checkpoint_path,
                # but we need valid_metrics.csv path.
                # It is likely in os.path.dirname(checkpoint_path) if checkpoint_dir was passed.
                # Let's verify existing file structure logic or just search.
                
                # Construct expected path (inside phase results directory)
                iter_dir_name = f"iter_{r['iteration']}_valid_{r['valid_year']}"
                possible_path = os.path.join(results_dir, iter_dir_name, "valid_metrics.csv")
                if os.path.exists(possible_path):
                    last_valid_csv = possible_path

        # 2. Trajectory CSV (inside phase results directory)
        traj_csv = os.path.join(results_dir, "trajectory_details.csv")
        if not os.path.exists(traj_csv):
            traj_csv = None  # Passed as None if not found

        # 3. Generate Charts (save in phase results directory)
        if last_valid_csv:
            create_all_charts(
                valid_csv=last_valid_csv,
                iterations_summary=iterations_summary,
                trajectory_csv=traj_csv,
                output_dir=results_dir,  # Save in phase results dir
                training_mode=mode  # Pass training mode for phase label
            )
            if verbose: smart_print(f"[CHARTS] Graphs saved to: {results_dir}")
        else:
            if verbose: smart_print("[CHARTS] Warning: No valid_metrics.csv found, skipping chart generation.")
            
    except Exception as e:
        smart_print(f"[CHARTS] Error generating charts: {e}")

    return results


def main():
    # Force line buffering for realtime output compliance
    sys.stdout.reconfigure(line_buffering=True)
    
    parser = argparse.ArgumentParser(
        description="Offline Batch Observer Training (Spec §6/§7 Compliant)"
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
        default=2021,
        help="First inference/validation year (default: 2021)",
    )

    parser.add_argument(
        "--last-infer-year",
        type=int,
        default=2021,
        help="Last inference year (default: 2021)",
    )
    parser.add_argument(
        "--valid-duration-years",
        type=int,
        default=1,
        help="Number of years for validation period (default: 1). Set to 2 for 2-year validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./observer_offline",
        help="Output directory (default: ./observer_offline)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs per iteration (default: None, use config: 50 for base, 5 for finetune)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        help="Steps per epoch (default: None, auto-compute per Spec §6: ceil((Len(Data) - T_m - h) / B))",
    )
    parser.add_argument(
        "--traj-len",
        type=int,
        default=None,
        help="Trajectory length T_m (default: None, use config)",
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
        "--log-details",
        action="store_true",
        help="Enable detailed trajectory logging (stock symbols, triggers, rewards)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Limit number of iterations (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: use config)",
    )
    parser.add_argument(
        "--mps-batch-size",
        type=int,
        default=64,
        help="Max batch size for MPS (Apple Silicon) to prevent OOM (default: 64)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low memory mode: batch_size=8, traj_len=64 (for 8GB RAM Macs)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="FULL",
        choices=["FULL", "MACRO_ONLY", "SELECTION_ONLY"],
        help="Training mode: FULL, MACRO_ONLY (Phase 1A), SELECTION_ONLY (Phase 1B)",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Path to initial checkpoint (required for SELECTION_ONLY to load Macro weights)",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="auto",
        choices=["auto", "uniform", "hybrid_weighted"],
        help="Sampling strategy: 'auto' (iter 0=uniform, iter 1+=hybrid), 'uniform', or 'hybrid_weighted'",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable per-timestep detailed logging on terminal (verbose training output)",
    )
    parser.add_argument(
        "--skip-recovery",
        action="store_true",
        help="Skip validation history recovery on resume (faster but incomplete metrics history)",
    )

    args = parser.parse_args()

    # DEBUG: Print skip_recovery value
    print(f"[DEBUG] --skip-recovery flag = {args.skip_recovery}")

    results = run_offline_observer_training(
        start_year=args.start_year,
        first_infer_year=args.first_infer_year,
        last_infer_year=args.last_infer_year,
        valid_duration_years=args.valid_duration_years,  # Multi-year validation
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batches_per_epoch=args.batches,
        traj_len=args.traj_len,
        seed=args.seed,
        log_details=args.log_details,  # NEW
        verbose=not args.quiet,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        mode=args.mode,  # Pass parsed mode
        init_checkpoint=args.init_checkpoint, # Pass init checkpoint
        mps_batch_size=args.mps_batch_size,  # [MPS-FIX] MPS batch size limit
        low_memory=args.low_memory,  # [MPS-FIX] Low memory mode
        sampling_strategy=args.sampling_strategy,  # NEW: Sampling strategy
        debug_logging=args.debug,  # NEW: Per-timestep terminal logging
        skip_recovery=args.skip_recovery,  # NEW: Skip validation history recovery
    )

    # Exit with error if any iteration failed
    if any(r["status"] != "success" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
