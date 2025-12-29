#!/usr/bin/env python3
"""
Recover Missing Train Metrics for Iteration 0
Targets: agents/MAFIA/observer_offline_1/phase1_macro/iter_0_valid_2017
"""
import os
import sys
import pandas as pd
import torch as th
import numpy as np
from dataclasses import asdict

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config, MafiaTrainMode
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from utils.mafia_data_loader import load_mafia_data

def recover_metrics():
    # Configuration matches Iter 0: Valid 2017
    start_year = 2015
    train_end_year = 2016
    valid_year = 2017
    
    # Target Directory
    # observer_offline_1 is in /Users/nguyensiry/Documents/the_last_dance/
    base_dir = "/Users/nguyensiry/Documents/the_last_dance/observer_offline_1" 
    phase_dir = os.path.join(base_dir, "phase1_macro")
    iter_dir = os.path.join(phase_dir, "iter_0_valid_2017")
    ckpt_dir = os.path.join(base_dir, "checkpoints", "macro", "temp_iter_0")
    
    print(f"[RECOVER] Target Iteration Dir: {iter_dir}")
    print(f"[RECOVER] Checkpoint Dir: {ckpt_dir}")
    
    if not os.path.exists(ckpt_dir):
        print(f"[ERROR] Checkpoint directory not found: {ckpt_dir}")
        return

    # Load Data
    config = Config()
    config.mafia_train_mode = MafiaTrainMode.MACRO_ONLY
    # Load default MAFIA hyperparameters (DC thresholds, etc)
    config.use_features = ["close", "open", "high", "low", "volume"]
    config.load_market_observer_config()
    
    # Speed up on CPU
    config.batch_size = 32
    config.mafia_batch_size = 32
    
    # Manually set overrides or missing params
    config.mafia_T_w = 30 
    config.mafia_trajectory_length = 128
    
    # Define Data Range
    train_start = pd.Timestamp(f"{start_year}-01-01")
    train_end = pd.Timestamp(f"{train_end_year}-12-31 23:59:59")
    
    print(f"[RECOVER] Loading data for range: {train_start} -> {train_end}")
    
    raw_data = load_mafia_data(config)
    stock_list = sorted(raw_data.stock.unique().tolist())
    action_dim = len(stock_list)
    
    # Init Model
    # FORCE CPU to avoid MPS hangs/OOM during recovery
    config.device = th.device("cpu")
    th.set_default_device("cpu") # PyTorch 2.x
    
    observer = MAFIAObserver(config, action_dim=action_dim)
    # Ensure observer uses CPU (sometimes it re-cals based on availability)
    observer.device = th.device("cpu")
    if hasattr(observer, "mafia_model"):
        observer.mafia_model.to("cpu")
        
    trainer = ObserverOfflineBatchTrainer(config, observer, device=th.device("cpu"))
    
    # Prepare Train Tensors
    print("[RECOVER] Preparing Training Tensors...")
    train_tensors = trainer.prepare_data_tensors(
        data=raw_data,
        stock_list=stock_list,
        start_date=train_start,
        end_date=train_end
    )
    
    # Identify missing epochs
    # Read existing metrics
    metrics_path = os.path.join(iter_dir, "train_metrics.csv")
    existing_epochs = []
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            if "epoch" in df.columns:
                existing_epochs = df["epoch"].astype(int).tolist()
        except:
            pass
            
    print(f"[RECOVER] Existing epochs in CSV: {existing_epochs}")
    
    # Find checkpoints
    files = os.listdir(ckpt_dir)
    ckpt_files = [f for f in files if f.startswith("epoch_") and f.endswith(".pth")]
    
    recovered_rows = []
    
    for f in sorted(ckpt_files, key=lambda x: int(x.split("_")[1].split(".")[0])):
        ep = int(f.split("_")[1].split(".")[0])
        
        if ep in existing_epochs:
            print(f"[RECOVER] Skipping Epoch {ep} (Already exists)")
            continue
            
        print(f"[RECOVER] Recovering Epoch {ep}...")
        path = os.path.join(ckpt_dir, f)
        
        # Load
        trainer.observer.load_checkpoint(path)
        
        # Evaluate on Train Set
        # Use subset for speed on CPU
        val_res = trainer.validate_epoch(
            data_tensors=train_tensors,
            compute_loss=True,
            steps=5 # fast approximation
        )
        
        row = val_res.to_dict(filter_by_phase=True)
        # Force epoch
        row["epoch"] = ep
        # Remove CES/Val specific
        keys_to_remove = [k for k in row.keys() if "ces" in k or "rank" in k]
        for k in keys_to_remove:
            del row[k]
            
        recovered_rows.append(row)
        print(f"         > Loss Total: {row.get('loss_total', 0):.4f}")
        
        # Incremental Save
        df_row = pd.DataFrame([row])
        # Add to CSV
        if not os.path.exists(metrics_path):
             df_row.to_csv(metrics_path, index=False, float_format='%.5f')
        else:
             df_row.to_csv(metrics_path, mode='a', header=False, index=False, float_format='%.5f')
        print(f"[RECOVER] Saved Epoch {ep} to CSV")

    # Sort at the end to be clean
    if os.path.exists(metrics_path):
        df_final = pd.read_csv(metrics_path)
        if "epoch" in df_final.columns:
            df_final["epoch"] = pd.to_numeric(df_final["epoch"], errors='coerce')
            df_final = df_final.dropna(subset=["epoch"])
            df_final["epoch"] = df_final["epoch"].astype(int)
            df_final = df_final.sort_values("epoch")
            df_final.to_csv(metrics_path, index=False, float_format='%.5f')
            print(f"[RECOVER] Final Sort Complete. Total epochs: {len(df_final)}")

if __name__ == "__main__":
    recover_metrics()
