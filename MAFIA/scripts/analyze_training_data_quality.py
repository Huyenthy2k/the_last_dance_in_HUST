import sys
import os
import torch as th
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from agents.MAFIA.config import Config, MafiaTrainMode
from agents.MAFIA.RL_controller.mafia_observer import MAFIAObserver
from agents.MAFIA.RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from agents.MAFIA.utils.mafia_data_loader import load_mafia_data

def analyze_data_quality():
    print("=== Analyzing Training Data Quality (2015-2019) ===")
    
    # 1. Setup Config
    config = Config()
    # Emulate the training phase for Iter 0 (Valid 2020)
    # This usually means training data up to end of 2019
    train_start = "2015-01-01 00:00:00"
    train_end = "2019-12-31 23:59:59"
    config.update_dates_for_walkforward(
        train_start_str=train_start,
        train_end_str=train_end
    )
    # Ensure config has correct hyperparams
    config.mafia_train_mode = MafiaTrainMode.MACRO_ONLY
    
    # 2. Load Data
    print("\n--- 1. Loading Data ---")
    try:
        data = load_mafia_data(config)
        print(f"Loaded {len(data)} rows.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Initialize Components
    device = th.device("cpu") # CPU for analysis
    observer = MAFIAObserver(config, action_dim=10)
    observer.mafia_model.to(device)
    trainer = ObserverOfflineBatchTrainer(config, observer, device=device)
    
    # 4. Prepare Tensors
    print("\n--- 2. Preparing Tensors ---")
    # Sort stock list for consistency
    stock_list = sorted(data["stock"].unique().tolist())
    start_date = pd.Timestamp(train_start)
    end_date = pd.Timestamp(train_end)
    
    data_tensors = trainer.prepare_data_tensors(
        data, 
        stock_list=stock_list, 
        start_date=start_date, 
        end_date=end_date
    )
    print(f"Stock Data Shape: {data_tensors['ochlv'].shape}") 
    
    # 5. Sample Massive Batch
    print("\n--- 3. Sampling Batch for Analysis ---")
    # Set a large batch size to get a representative distribution
    # Set a small batch size for CPU analysis
    trainer.batch_size = 32
    trainer.T_m = 129 # Reduced window size slightly just in case
    trainer.T_m = 128
    
    # Force 'TRAIN' mode sampling
    params_bkp = trainer.config.mafia_use_class_balanced_sampling
    trainer.config.mafia_use_class_balanced_sampling = False # Uniform sampling to see natural distribution
    
    try:
        batch = trainer.sample_trajectory_batch(data_tensors, mode="TRAIN")
    except Exception as e:
        print(f"Error sampling batch: {e}")
        return

    # 6. Analyze Explicit Signals
    print("\n=== Explicit Signal Analysis ===")
    signals = {
        "Vol_Rel": batch.vol_std20,
        "DC_Flag": batch.dc_event_flag,
        "Breadth": batch.breadth_gap,
        "Div": batch.div_signal,
        "VPI": batch.signed_vpi_zscore,
        "DD60": batch.drawdown60
    }
    
    for name, tensor in signals.items():
        t = tensor.float()
        mean = t.mean().item()
        std = t.std().item()
        min_v = t.min().item()
        max_v = t.max().item()
        zeros = (t == 0).float().mean().item() * 100
        
        print(f"{name:10} | Mean: {mean:7.4f} | Std: {std:7.4f} | Range: [{min_v:7.4f}, {max_v:7.4f}] | Zeros: {zeros:5.1f}%")
        
    # Specific Checks
    dc_counts = batch.dc_event_flag.sum().item()
    if dc_counts == 0:
        print("\n❌ WARNING: NO DC Events detected in this batch! (Crash detection might be broken)")
    else:
        print(f"\n✅ DC Events Detected: {int(dc_counts)} instances")
        
    if batch.vol_std20.std() < 0.05:
        print("❌ WARNING: Volatility signal has very low variance!")

    # 7. Analyze Targets
    print("\n=== Target Analysis ===")
    
    # Direction Labels
    labels = batch.direction_labels
    total = labels.numel()
    bear = (labels == 0).sum().item()
    side = (labels == 1).sum().item()
    bull = (labels == 2).sum().item()
    
    print(f"Direction Distribution:")
    print(f"  Bear: {bear/total*100:5.1f}%")
    print(f"  Side: {side/total*100:5.1f}%")
    print(f"  Bull: {bull/total*100:5.1f}%")
    
    if side/total > 0.8:
        print("❌ WARNING: Significant class imbalance (Dominant Side class)!")
        
    # Risk Targets (Eta)
    eta = batch.risk_targets
    print(f"\nRisk Targets (Eta):")
    print(f"  Mean: {eta.mean().item():.4f}")
    print(f"  Std:  {eta.std().item():.4f}")
    print(f"  Min:  {eta.min().item():.4f}")
    print(f"  Max:  {eta.max().item():.4f}")
    
    if eta.std() < 0.01:
        print("❌ WARNING: Risk Targets are effectively constant!")

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    analyze_data_quality()
