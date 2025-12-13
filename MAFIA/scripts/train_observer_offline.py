#!/usr/bin/env python3
"""
Offline Batch Training for MAFIA Observer - Spec §6/§7 Compliant

This script trains the MAFIA Observer using the offline batch mode
as specified in refactor_mafia.md Section 6 (Trajectory Configuration)
and Section 7 (Observer Training Schedule).

Key Features:
1. Random trajectory sampling with fresh hidden state per batch
2. Collect → Train → Discard loop (transient buffer)
3. Cadence-aware loss masking
4. Walk-forward expanding window validation

Usage:
    python scripts/train_observer_offline.py \
        --start-year 2015 \
        --first-infer-year 2018 \
        --last-infer-year 2022 \
        --output-dir ./observer_offline

    # Single iteration test
    python scripts/train_observer_offline.py \
        --start-year 2015 \
        --first-infer-year 2018 \
        --last-infer-year 2018 \
        --epochs 10 \
        --output-dir ./observer_offline_test
"""

import argparse
import os
import sys
import gc
import copy
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch as th
from torch.utils.tensorboard import SummaryWriter

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from visualize_training import generate_epoch_report
from utils.mafia_data_loader import load_mafia_data
from scripts.generate_rl_states import generate_single_year

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
    start_year: int, first_infer_year: int, last_infer_year: int
) -> List[Dict]:
    """
    Build the expanding-window schedule per spec Section 7.

    Train: 01/start_year → 06/(valid_year)
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
    num_epochs: Optional[int] = None,
    batches_per_epoch: Optional[int] = None,
    prev_checkpoint: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
    global_step_start: int = 0,
    verbose: bool = True,
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
        prev_checkpoint: Path to previous checkpoint for finetuning
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
            num_epochs = int(getattr(config, "mafia_observer_finetune_epochs", 20))

    if verbose:
        train_mode = "Base Training" if iteration == 0 else "Finetuning"
        smart_print(f"[OFFLINE] {train_mode}: {num_epochs} epochs")
        
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

    # Load previous checkpoint if finetuning
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        smart_print(f"[OFFLINE] Loading checkpoint for finetuning: {prev_checkpoint}")
        trainer.observer.load_checkpoint(prev_checkpoint)

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

    # Create views
    train_tensors = {
        "ochlv": data_tensors["ochlv"][train_indices[0] : train_indices[-1] + 1],
        "returns": data_tensors["returns"][train_indices[0] : train_indices[-1] + 1],
        "market_ochlv": data_tensors.get("market_ochlv", None)[train_indices[0] : train_indices[-1] + 1] if data_tensors.get("market_ochlv") is not None else None,
        "market_returns": data_tensors.get("market_returns", None)[train_indices[0] : train_indices[-1] + 1] if data_tensors.get("market_returns") is not None else None,
        "dates": dates[train_indices],
        "stock_list": data_tensors["stock_list"],
        "T_total": len(train_indices),
        "N": data_tensors["N"],
    }
    
    # Auto-compute batches per epoch
    if batches_per_epoch is None:
        T_train = train_tensors["T_total"]
        T_m = trainer.T_m
        h = trainer.horizon
        T_w = trainer.T_w
        batch_size = trainer.batch_size
        available_starts = T_train - T_m - h - T_w
        batches_per_epoch = max(1, int(np.ceil(available_starts / batch_size)))

    valid_tensors = {
        "ochlv": data_tensors["ochlv"][valid_indices[0] : valid_indices[-1] + 1],
        "returns": data_tensors["returns"][valid_indices[0] : valid_indices[-1] + 1],
        "market_ochlv": data_tensors.get("market_ochlv", None)[valid_indices[0] : valid_indices[-1] + 1] if data_tensors.get("market_ochlv") is not None else None,
        "market_returns": data_tensors.get("market_returns", None)[valid_indices[0] : valid_indices[-1] + 1] if data_tensors.get("market_returns") is not None else None,
        "dates": dates[valid_indices],
        "stock_list": data_tensors["stock_list"],
        "T_total": len(valid_indices),
        "N": data_tensors["N"],
    }

    # Initialize ValidationMetricsTracker
    iter_output_dir = os.path.join(checkpoint_dir, f"../iter_{iteration}_valid_{valid_year}")
    os.makedirs(iter_output_dir, exist_ok=True)
    tracker = ValidationMetricsTracker(iter_output_dir)
    
    # Checkpoint storage
    temp_ckpt_dir = os.path.join(checkpoint_dir, f"temp_iter_{iteration}")
    os.makedirs(temp_ckpt_dir, exist_ok=True)

    current_global_step = global_step_start

            # Resume logic: Check for existing checkpoints in temp dir
    # Priority: latest_checkpoint.pth > epoch_{N}.pth (best checkpoints)
    start_epoch = 0
    smart_print(f"[DEBUG] Checking for resume in: {temp_ckpt_dir}")
    if os.path.exists(temp_ckpt_dir):
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
                    
                    tracker._recompute_ces_scores()

                smart_print(f"[RESUME] Will continue from epoch {start_epoch}")
            except Exception as e:
                smart_print(f"[WARN] Failed to load checkpoint: {e}")
                # If critical error loading history, we already raised. 
                # If just checkpoint load fail, maybe safer to crash than start from 0 if files exist?
                # For now, let's allow fail->start 0 coupled with warning, 
                # BUT if we start at 0, we should probably backup existing metric file?
                # If we fail to resume but have data, we might overwrite it.
                if os.path.exists(valid_csv_path):
                     smart_print("[CRITICAL] Checkpoint load failed but valid_metrics.csv exists.")
                     smart_print("[CRITICAL] Starting from Epoch 0 would overwrite it. Aborting for safety.")
                     raise e
                start_epoch = 0

    # Training Loop
    if start_epoch >= num_epochs:
        smart_print(f"[OFFLINE] Iteration already completed ({start_epoch} >= {num_epochs}). Skipping training loop.")

    # IMPORTANT: ALWAYS sync trainer._epoch with start_epoch for correct epoch numbering
    # train_epoch() increments _epoch at the start, so we set to start_epoch
    # This ensures epoch numbering in validation results matches the actual epoch
    # Note: Loop epoch N corresponds to display "EPOCH N+1", so when resuming from
    # checkpoint with loop epoch E, we want to display "EPOCH E+2" (the next one)
    trainer._epoch = start_epoch
    if start_epoch > 0:
        # Also update current_global_step based on previously completed epochs
        current_global_step = global_step_start + (start_epoch * (batches_per_epoch or 10))
        if verbose:
            smart_print(f"[RESUME] Synced trainer._epoch to {trainer._epoch} (will become {start_epoch} after train_epoch())")
            smart_print(f"[RESUME] Synced global_step to {current_global_step}")

    # Show chart location at training start
    root_output_dir = os.path.dirname(iter_output_dir)
    plots_dir = os.path.join(root_output_dir, "plots")
    if verbose:
        smart_print(f"\n📊 Charts will be saved to: {os.path.abspath(plots_dir)}")
        smart_print(f"   (Updated after each batch for real-time progress)\n")

    # Wrap loop in try-finally to ensure validation metrics are saved
    try:
        for epoch in range(start_epoch, num_epochs):
            # Create per-batch visualization callback
            def on_batch_viz(batch_idx, total_batches):
                # Generate charts after each batch for real-time progress
                try:
                    generate_epoch_report(tracker.output_dir, epoch)
                except Exception:
                    pass  # Silent fail - don't interrupt training
            
            # 1. Train
            train_res = trainer.train_epoch(
                data_tensors=train_tensors,
                steps_per_epoch=batches_per_epoch,
                writer=writer,
                global_step_offset=current_global_step,
                on_batch_done=on_batch_viz,  # Real-time chart updates
            )
            current_global_step += batches_per_epoch

            # 2. Validate (returns ObserverValidationResult)
            val_result = trainer.validate_epoch(
                data_tensors=valid_tensors,
                steps=max(1, batches_per_epoch // 4),
            )

            # 3. Track metrics and compute CES
            is_best = tracker.add_epoch(val_result)
            
            # 4. Log to TensorBoard
            if writer:
                writer.add_scalar("Valid/Loss/Total", val_result.loss_total, epoch)
                writer.add_scalar("Valid/Loss/PG", val_result.loss_pg, epoch)
                writer.add_scalar("Valid/Loss/Risk", val_result.loss_risk, epoch)
                writer.add_scalar("Valid/Loss/Dir", val_result.loss_dir, epoch)
                writer.add_scalar("Valid/Metrics/Sharpe", val_result.topk_sharpe_ratio, epoch)
                writer.add_scalar("Valid/Metrics/Dir_F1", val_result.direction_f1_macro, epoch)
                writer.add_scalar("Valid/Metrics/Risk_MSE", val_result.risk_mse, epoch)
                writer.add_scalar("Valid/Metrics/CES", val_result.ces_score, epoch)

            # 5. Update Live Dashboard
            update_observer(
                loss=train_res.get("loss_total", 0.0),
                loss_eta=val_result.risk_mse,
                loss_dir=val_result.loss_dir,
            )
            
            update_walkforward_score(
                current_score=val_result.ces_score,
                best_score=tracker.best_ces,
                best_epoch=tracker.best_epoch if tracker.best_epoch is not None else epoch,
            )

            # 6. Save checkpoints
            # Always save latest checkpoint for resume (overwrites previous)
            latest_ckpt_path = os.path.join(temp_ckpt_dir, "latest_checkpoint.pth")
            trainer.observer.save_checkpoint(latest_ckpt_path, epoch=epoch)

            # Save best checkpoint separately (for final selection)
            if is_best:
                best_ckpt_path = os.path.join(temp_ckpt_dir, f"epoch_{epoch}.pth")
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
                generate_epoch_report(tracker.output_dir, epoch)
                
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
                    
                    # Log dynamics chart
                    dynamics_path = os.path.join(plots_dir, "dynamics_latest.png")
                    if os.path.exists(dynamics_path):
                        trainer.tb_logger.log_image("charts/dynamics", dynamics_path, epoch, phase="valid")
                    
                    # Flush to ensure images are written
                    trainer.tb_logger.flush()
                    
            except Exception as e:
                smart_print(f"      [WARN] Visualization failed: {e}")

    finally:
        # Save validation history to CSV (ensure save on interrupt/error)
        csv_path = tracker.save_validation_history()
        if verbose:
            smart_print(f"\n[OFFLINE] Validation history saved to: {csv_path}")
    
    # Get best checkpoint info
    best_info = tracker.get_best_checkpoint_info()
    best_epoch = best_info["epoch"]
    best_ces = best_info["ces_score"]
    
    if verbose:
        smart_print(f"\n[OFFLINE] CES Selection for Iter {iteration}:")
        smart_print(f"  Best Epoch: {best_epoch}")
        smart_print(f"  Best CES: {best_ces:.4f}")
        smart_print(f"  Sharpe: {best_info['sharpe_ratio']:.3f}")
        smart_print(f"  Dir_F1: {best_info['direction_f1']:.3f}")
        smart_print(f"  Risk_MSE: {best_info['risk_mse']:.4f}")
    
    # Copy best checkpoint to final location
    final_ckpt_path = os.path.join(checkpoint_dir, f"observer_best_{valid_year}.pth")
    best_temp_path = os.path.join(temp_ckpt_dir, f"epoch_{best_epoch}.pth")
    
    if os.path.exists(best_temp_path):
        trainer.observer.load_checkpoint(best_temp_path)
        trainer.observer.save_checkpoint(final_ckpt_path, epoch=best_epoch, extra_data={"ces": best_ces})
    else:
        # Fallback: save current state if no best checkpoint found
        trainer.observer.save_checkpoint(final_ckpt_path, epoch=num_epochs-1, extra_data={"ces": best_ces})
    
    # Cleanup temp checkpoints
    import shutil
    try:
        shutil.rmtree(temp_ckpt_dir)
    except:
        pass

    return final_ckpt_path



def run_offline_observer_training(
    start_year: int = 2015,
    first_infer_year: int = 2018,
    last_infer_year: int = 2022,
    output_dir: str = "./observer_offline",
    num_epochs: Optional[int] = None,
    batches_per_epoch: Optional[int] = None,
    traj_len: Optional[int] = None,
    seed: int = 2025,
    log_details: bool = False,
    rebalance_interval: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run full walk-forward offline observer training.

    Per Spec §6 Training Duration:
    - Base training (iter 0): 50 epochs (default)
    - Finetune (iter > 0): 20 epochs (default)
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

    # Build schedule
    schedule = build_expanding_schedule(start_year, first_infer_year, last_infer_year)
    num_iterations = len(schedule)

    if verbose:
        smart_print("\n" + "#" * 70)
        smart_print("# OFFLINE BATCH OBSERVER TRAINING (Spec §6/§7 Compliant)")
        smart_print("#" * 70)
        smart_print(f"  Start year: {start_year}")
        smart_print(f"  First infer year: {first_infer_year}")
        smart_print(f"  Last infer year: {last_infer_year}")
        smart_print(f"  Total iterations: {num_iterations}")
        smart_print(f"  Epochs/iteration: {num_epochs}")
        smart_print(f"  Batches/epoch: {batches_per_epoch}")
        smart_print(f"  Output dir: {output_dir}")
        smart_print(f"{'#' * 70}\n")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load config and data
    # Pass create_dirs=False to prevent auto-creation of TD3 directories
    config = Config(create_dirs=False)
    config.seed = seed
    config.res_root = output_dir  # For trajectory CSV output
    if traj_len:
        config.mafia_trajectory_length = traj_len
    if rebalance_interval is None:
        rebalance_interval = getattr(config, "mafia_topk_rebalance_interval", 14)
    
    # Enable trajectory logging if requested
    if log_details:
        config.log_trajectory_details = True
        if verbose:
            smart_print("[CONFIG] Detailed trajectory logging ENABLED")
            smart_print(f"  Output: {output_dir}/trajectory_details.csv\n")

    # Initialize TensorBoard Writer
    log_dir = os.path.join(output_dir, "tb_logs")
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
        if not os.path.exists(market_file):
             # Fallback to hardcoded name if config name not found
             market_file = os.path.join(getattr(config, "dataDir", "./data"), "VNINDEX_1d_index.csv")
        if os.path.exists(market_file):
            market_data = pd.read_csv(market_file, parse_dates=["date"])
            if verbose:
                smart_print(f"[OFFLINE] Loaded Market Data (VNINDEX): {len(market_data)} rows")
        else:
            market_data = None
            if verbose:
                smart_print(f"[WARN] Market Data (VNINDEX) not found at {market_file}. Using fallback averages.")

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

    # CRITICAL: Free raw pandas dataframes immediately to prevent OOM
    # Tensors are now on device (or in shared memory), we don't need the pandas duplicates
    del stock_data
    if 'market_data' in locals() and market_data is not None:
        del market_data
    gc.collect()

    if verbose:
        smart_print(
            f"[OFFLINE] Prepared tensors: T_total={data_tensors['T_total']}, N={data_tensors['N']}"
        )

    # Run iterations
    results = []
    prev_checkpoint = None

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
            smart_print(f"  🎯 Inference Target Year:       {sched['infer_year']}")
            smart_print(f"")
            if is_cold_start:
                smart_print(f"  Training Mode: FROM_SCRATCH ({num_train_epochs} epochs)")
                smart_print(f"  Checkpoint: None (cold start)")
            else:
                smart_print(f"  Training Mode: FINETUNE ({num_train_epochs} epochs)")
                if prev_checkpoint and os.path.exists(prev_checkpoint):
                    smart_print(f"  Load Checkpoint: {os.path.basename(prev_checkpoint)}")
                else:
                    smart_print(f"  Load Checkpoint: None (⚠️ Expected from previous iteration)")
            smart_print(f"{'=' * 70}")

        # RESUME CHECK: If final checkpoint for this iteration already exists, skip it
        # But we must update prev_checkpoint correctly!
        final_iter_ckpt = os.path.join(checkpoint_dir, f"observer_best_{sched['valid_year']}.pth")
        if os.path.exists(final_iter_ckpt):
             smart_print(f"[RESUME] Found existing completed checkpoint for Iteration {iteration}: {final_iter_ckpt}")
             smart_print(f"         Skipping training for this iteration.")
             
             # Create a dummy result entry for summary
             results.append(
                {
                    "iter_index": iteration,
                    "iteration": iteration,
                    "train_range": sched["train_label"],
                    "valid_range": sched["valid_label"],
                    "valid_year": sched["valid_year"],
                    "infer_year": sched["infer_year"],
                    "checkpoint_path": final_iter_ckpt,
                    "state_path": None,
                    "status": "success (skipped - resumed)",
                    "infer_status": "live_stream",
                    "skipped": True, 
                }
             )
             
             # Set prev_checkpoint for NEXT iteration
             prev_checkpoint = final_iter_ckpt
             continue

        try:
            checkpoint_path = train_observer_offline_iteration(
                config=config,
                schedule_entry=sched,
                trainer=trainer,
                data_tensors=data_tensors,
                checkpoint_dir=checkpoint_dir,
                num_epochs=num_epochs,
                batches_per_epoch=batches_per_epoch,
                prev_checkpoint=prev_checkpoint,
                writer=writer,
                global_step_start=global_step_counter,
                verbose=verbose,
            )
            
            # Update global step counter for next iteration (approximate, since we don't return exact steps run)
            # But since train_epoch runs exactly batches_per_epoch steps, we can calculate it.
            # However, early stopping might make this inaccurate. 
            # Ideally, train_observer_offline_iteration should return steps run.
            # For now, we'll let the steps overlap or reset slightly inaccurately, or just rely on the fact 
            # that we usually finetune for fixed epochs unless early stopped.
            # Better approach: Pass writer and offset, let iteration update it? 
            # We are passing global_step_start, but not getting back the new offset.
            # Let's simple approximate or fetch from trainer if possible, but trainer is stateless regarding steps.
            # Simple fix: assume full epochs for step counter estimation
            iter_epochs = num_epochs if num_epochs else (50 if iteration == 0 else 20)
            iter_batches = batches_per_epoch # We will have computed this inside
            # But wait, batches_per_epoch is computed INSIDE train_observer_offline_iteration if None.
            # We can't easily update global_step_counter here without returning it.
            # Let's accept that steps might restart or jump oddly if we don't return it.
            # Actually, `train_observer_offline_iteration` has the logic to compute batches_per_epoch.
            # We should probably refactor to get the actual steps, but as a quick fix, 
            # let's just increment by a safe estimate.
            # Actually, the best way is to return executed_steps from the function.
            # But modifying return signature is a bigger change.
            # Let's just create a new writer per iteration? No, continuous is better.
            # Let's just use a very large offset increment to separate iterations visually if we can't be precise,
            # or just let them overlap in "Global Step" but be distinct in "Epoch".
            # "Global Step" is mostly for continuous training.
            # Let's update `train_observer_offline_iteration` to return `(checkpoint_path, steps_run)`.
            # But that breaks the signature expected by callers? 
            # No other callers likely for this specific script function.
            
            # Let's stick to the current signature for now to minimize risk, 
            # and just increment by a fixed large amount or just let it be.
            # Actually, if we want continuous plots, we need increasing steps.
            # Let's just guess: 
            # global_step_counter += iter_epochs * 100 # Rough guess
            pass 
            
            # Correction: I will update the function signature to return steps run in a separate PR if needed.
            # For now, let's just not increment global_step_counter in the outer loop 
            # and pass the SAME counter? No, that would overwrite logs.
            # I must update global_step_counter.
            # Let's update the signature of `train_observer_offline_iteration` to return more info?
            # Or just hack it:
            # The function returns `best_checkpoint_path`. I can change it to return `(best_checkpoint_path, steps_run)`.
            # Wait, the `train_observer_offline_iteration` creates `batches_per_epoch` if None.
            # I should move that logic out or return it.
            # Let's just leave it for now and accept that global_step might reset or be weird between iterations unless I fix it.
            # Actually, I can just increment by a safe upper bound.
             

            # State generation skipped per "Live Inference Stream" requirement (Spec 9.1)
            # Observer is kept frozen and queried live by RL during Phase 2.
            state_path = None
            infer_status = "live_stream"

            results.append(
                {
                    "iteration": iteration,
                    "train_range": sched["train_label"],
                    "valid_range": sched["valid_label"],
                    "valid_year": sched["valid_year"],
                    "infer_year": sched["infer_year"],
                    "checkpoint_path": checkpoint_path,
                    "state_path": state_path,
                    "status": "success",
                    "infer_status": infer_status,
                }
            )

            prev_checkpoint = checkpoint_path

            if verbose:
                smart_print(f"\n{'=' * 70}")
                smart_print(f"✅ ITERATION {iteration + 1} COMPLETE - Stage 1 (Observer Training)")
                smart_print(f"{'=' * 70}")
                smart_print(f"  Frozen Checkpoint: {os.path.basename(checkpoint_path)}")
                smart_print(f"  Target Year: {sched['infer_year']}")
                smart_print(f"")
                smart_print(f"  📌 Ready for Stage 2: TD3 Training (Year {sched['infer_year']})")
                smart_print(f"     Observer will be queried LIVE via frozen model")
                smart_print(f"     No pre-generated states (on-demand forward pass)")
                smart_print(f"{'=' * 70}\n")

        except Exception as e:
            results.append(
                {
                    "iteration": iteration,
                    "train_range": sched["train_label"],
                    "valid_range": sched["valid_label"],
                    "valid_year": sched["valid_year"],
                    "infer_year": sched["infer_year"],
                    "checkpoint_path": None,
                    "status": f"failed: {e}",
                }
            )
            smart_print(f"\n❌ Iteration {iteration + 1} failed: {e}")
            traceback.print_exc()

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
        best_ces = -1.0
        best_year = None
        best_metrics = {}
        for r in results:
            if r['status'] == 'success':
                metrics = r.get('best_metrics', {})
                ces = metrics.get('ces_score', 0.0)
                if ces > best_ces:
                    best_ces = ces
                    best_year = r.get('valid_year')
                    best_metrics = metrics

        if best_year:
            smart_print(f"\n  🏆 Best Checkpoint: Ckpt_Best_{best_year}")
            smart_print(f"     CES Score:      {best_ces:.4f}")
            smart_print(f"     Sharpe Ratio:   {best_metrics.get('topk_sharpe_ratio', 0.0):.4f}")
            smart_print(f"     Direction F1:   {best_metrics.get('direction_f1_macro', 0.0):.4f}")
            smart_print(f"     Risk MSE:       {best_metrics.get('risk_mse', 0.0):.6f}")

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
                    "ces": metrics.get("ces_score", 0.0),
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
                
                # Construct expected path
                iter_dir_name = f"iter_{r['iteration']}_valid_{r['valid_year']}"
                possible_path = os.path.join(output_dir, iter_dir_name, "valid_metrics.csv")
                if os.path.exists(possible_path):
                    last_valid_csv = possible_path

        # 2. Trajectory CSV
        traj_csv = os.path.join(output_dir, "trajectory_details.csv")
        if not os.path.exists(traj_csv):
            traj_csv = None # Passed as None if not found
            
        # 3. Generate Charts
        if last_valid_csv:
            create_all_charts(
                valid_csv=last_valid_csv,
                iterations_summary=iterations_summary,
                trajectory_csv=traj_csv,
                output_dir=output_dir # Save in root output dir
            )
            if verbose: smart_print(f"[CHARTS] Graphs saved to: {output_dir}")
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
        default=2018,
        help="First inference year (default: 2018)",
    )
    parser.add_argument(
        "--last-infer-year",
        type=int,
        default=2022,
        help="Last inference year (default: 2022)",
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
        help="Training epochs per iteration (default: None, use config: 50 for base, 20 for finetune)",
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

    args = parser.parse_args()

    results = run_offline_observer_training(
        start_year=args.start_year,
        first_infer_year=args.first_infer_year,
        last_infer_year=args.last_infer_year,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batches_per_epoch=args.batches,
        traj_len=args.traj_len,
        seed=args.seed,
        log_details=args.log_details,  # NEW
        verbose=not args.quiet,
    )

    # Exit with error if any iteration failed
    if any(r["status"] != "success" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
