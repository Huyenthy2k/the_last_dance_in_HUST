#!/usr/bin/env python3
"""
MAFIA Walk-Forward Training Script (Yearly Expanding Window)

Implements the strategy:
    Iter 0 (Base): Train [2015-2019], Valid [2020]
    Iter 1 (Finetune): Train [2015-2020], Valid [2021]
    Iter 2 (Finetune): Train [2015-2021], Valid [2022]
    ...

Usage:
    python scripts/train_walkforward.py
"""

import os
import sys
import pandas as pd
import torch as th
import numpy as np
import gc
from typing import List, Dict

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer, create_offline_trainer
from utils.mafia_data_loader import load_mafia_data

# Function to build schedule
def build_yearly_schedule(start_year=2015, end_train_year_base=2019, max_year=2024) -> List[Dict]:
    schedule = []
    iter_idx = 0
    
    current_train_end_year = end_train_year_base
    
    while current_train_end_year < max_year:
        valid_year = current_train_end_year + 1
        
        train_start = f"{start_year}-01-01 00:00:00"
        train_end = f"{current_train_end_year}-12-31 23:59:59"
        valid_start = f"{valid_year}-01-01 00:00:00"
        valid_end = f"{valid_year}-12-31 23:59:59"
        
        schedule.append({
            "iter_index": iter_idx,
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "valid_year": valid_year,
            "is_base": (iter_idx == 0)
        })
        
        current_train_end_year += 1
        iter_idx += 1
        
    return schedule

def run_training_cycle(args):
    # 1. Initialize Config
    config = Config()
    config.expanding_window_mode = True
    
    if args.dry_run:
        config.mafia_trajectory_length = 8
        config.mafia_seq_len = 8
        config.mafia_T_w = 4
        print("[WALKFORWARD] DRY RUN MODE: Reduced mafia_trajectory_length=8, T_w=4.")

    
    # Setup Output Directory
    output_dir = os.path.join(REPO_ROOT, "res", "walkforward_yearly")
    os.makedirs(output_dir, exist_ok=True)
    config.res_dir = output_dir # Override
    
    # 2. Load Data (Once)
    print(f"[WALKFORWARD] Loading Data...")
    raw_data = load_mafia_data(config)
    
    # 3. Build Schedule
    schedule = build_yearly_schedule(start_year=2015, end_train_year_base=2019, max_year=2024)
    
    if args.limit_iters:
        schedule = schedule[:args.limit_iters]
        
    print(f"[WALKFORWARD] Planned {len(schedule)} Iterations:")
    for s in schedule:
        print(f"  Iter {s['iter_index']}: Train [{s['train_start']} -> {s['train_end']}] | Valid [{s['valid_year']}]")
        
    # 4. Loop
    prev_ckpt_path = None
    
    for entry in schedule:
        print(f"\n{'='*60}")
        print(f"🚀 STARTING ITERATION {entry['iter_index']}")
        print(f"   Train: {entry['train_start']} -> {entry['train_end']}")
        print(f"   Valid: {entry['valid_year']}")
        print(f"{'='*60}")
        
        # Update Config Dates
        config.update_dates_for_walkforward(
            train_start_str=entry["train_start"],
            train_end_str=entry["train_end"],
            valid_start_str=entry["valid_start"],
            valid_end_str=entry["valid_end"]
        )
        
        # Prepare Stock List and N
        stock_list = raw_data.stock.unique().tolist()
        stock_list.sort() # Ensure consistent order
        config.stock_list = stock_list
        action_dim = len(stock_list)
        print(f"[WALKFORWARD] N={action_dim} stocks: {stock_list[:5]}...")

        # Create Trainer and Observer
        # We assume observer structure is constant, only weights change
        # DEBUG: Print config before trainer creation
        if args.dry_run:
             config.mafia_trajectory_length = 8
             config.mafia_seq_len = 8
             config.mafia_T_w = 4
        print(f"[DEBUG] config.mafia_trajectory_length before trainer init: {getattr(config, 'mafia_trajectory_length', 'MISSING')}")

        # 4. Train Observer (Phase 1)
        # Instantiate Observer first
        observer = MAFIAObserver(config, action_dim=action_dim)
        # Pass to factory
        trainer = create_offline_trainer(config, observer)
        
        # Prepare Data Tensors for this Iteration
        # (We do this inside loop to cut correct slices)
        # Note: trainer.prepare_data_tensors filters by date internally using config.train_date_start/end
        # But we need VALIDATION tensors too.
        
        # Let's manually prepare full range tensors for this iter (Train + Valid) to support both?
        # Or separately.
        
        print("[WALKFORWARD] Preparing Training Data Tensors...")
        train_tensors = trainer.prepare_data_tensors(
            raw_data, 
            config.stock_list if hasattr(config, 'stock_list') else raw_data.stock.unique().tolist(),
            config.train_date_start,
            config.train_date_end
        )
        
        print("[WALKFORWARD] Preparing Validation Data Tensors...")
        valid_tensors = trainer.prepare_data_tensors(
            raw_data,
            config.stock_list if hasattr(config, 'stock_list') else raw_data.stock.unique().tolist(),
            config.valid_date_start,
            config.valid_date_end
        )
        
        # Add metadata
        train_tensors["T_total"] = len(train_tensors["dates"])
        valid_tensors["T_total"] = len(valid_tensors["dates"])
        
        # Load Checkpoint logic
        if entry["is_base"]:
            print("[WALKFORWARD] Iter 0: Training from Scratch.")
            num_epochs = args.base_epochs
        else:
            print(f"[WALKFORWARD] Iter {entry['iter_index']}: Finetuning from {prev_ckpt_path}")
            if prev_ckpt_path and os.path.exists(prev_ckpt_path):
                observer.load_checkpoint(prev_ckpt_path)
            else:
                raise ValueError(f"Missing previous checkpoint: {prev_ckpt_path}")
            num_epochs = args.finetune_epochs
            
        # Training Loop
        best_valid_score = -999.0
        best_epoch = -1
        
        iter_dir = os.path.join(output_dir, f"iter_{entry['iter_index']}_{entry['valid_year']}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Override res_root for logging
        config.res_root = iter_dir
        
        for epoch in range(num_epochs):
            # Dry Run Limit
            limit_batches = 5 if args.dry_run else None

            # Train
            train_res = trainer.train_epoch(
                data_tensors=train_tensors,
                verbose=True,
                limit_batches=limit_batches
            )
            
            # Valid
            valid_res = trainer.validate_epoch(
                data_tensors=valid_tensors,
                compute_loss=True,
                limit_batches=limit_batches
            )
            
            print(f"   [Ep {epoch}] Train Loss: {train_res.loss_total:.4f} | Valid Score: {valid_res.phase_score:.4f} (Sharpe: {valid_res.topk_sharpe_ratio:.4f})")
            
            # Save Checkpoint
            if valid_res.phase_score > best_valid_score:
                best_valid_score = valid_res.phase_score
                best_epoch = epoch
                save_path = os.path.join(iter_dir, "best_checkpoint.pth")
                observer.save_checkpoint(save_path, epoch=epoch)
                print(f"   >>> New Best Checkpoint Saved! <<<")
                
        # Set prev_ckpt_path for next iteration
        prev_ckpt_path = os.path.join(iter_dir, "best_checkpoint.pth")
        print(f"[WALKFORWARD] Iteration {entry['iter_index']} Completed. Best Score: {best_valid_score:.4f}")
        
        # ============================================================
        # PHASE 1: GENERATE & SAVE VINTAGE STATE (Data Factory)
        # ============================================================
        print(f"[WALKFORWARD] Generating Vintage State for Year {entry['valid_year']} (Using Best Ckpt)...")
        
        # 1. Reload Best Checkpoint (Crucial!)
        observer.load_checkpoint(prev_ckpt_path)
        
        # 2. Generate State on Validation Set (The "Future" Year)
        # Ensure valid_tensors are on correct device inside generator
        vintage_state = trainer.generate_inference_state(
            data_tensors=valid_tensors,
            desc=f"Gen State {entry['valid_year']}"
        )
        
        # 3. Save State to Disk
        state_save_path = os.path.join(output_dir, f"rl_state_{entry['valid_year']}.pt")
        th.save(vintage_state, state_save_path)
        print(f"   >>> SAVED: {state_save_path} (Keys: {list(vintage_state.keys())}) <<<")
        
        # Cleanup to save memory
        del trainer
        del observer
        del train_tensors
        del valid_tensors
        gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-iters", type=int, default=None, help="Limit number of iterations to run")
    parser.add_argument("--base-epochs", type=int, default=50, help="Epochs for Iter 0")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="Epochs for Finetuning")
    parser.add_argument("--dry-run", action="store_true", help="Run with 1 epoch for testing")
    args = parser.parse_args()
    
    # Override globals/constants via args for testing
    if args.dry_run:
        args.base_epochs = 1
        args.finetune_epochs = 1
        args.limit_iters = 1
        args.limit_iters = 1
        print("[WALKFORWARD] DRY RUN MODE: Reduced epochs and iters.")
        
    config = Config()
    if args.dry_run:
        config.mafia_seq_len = 8
        print("[WALKFORWARD] DRY RUN MODE: Reduced mafia_seq_len to 8.")

    run_training_cycle(args)
