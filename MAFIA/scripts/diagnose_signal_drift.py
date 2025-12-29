#!/usr/bin/env python3
import sys
import os
import torch as th
import numpy as np
import pandas as pd

# Add path
sys.path.insert(0, os.path.join(os.getcwd(), "agents", "MAFIA"))

from config import Config
from scripts.train_observer_offline import load_mafia_data
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from RL_controller.mafia_observer import MAFIAObserver

def diagnose_signals():
    print("Initializing Config and Trainer...")
    config = Config(create_dirs=False)
    # Force CPU for analysis to avoid MPS OOM/Complications with small scripts
    config.device = th.device("cpu") 
    
    # Set dates for Train (2015-2019) and Valid (2020)
    train_start = "2015-01-01 00:00:00"
    train_end = "2019-12-31 00:00:00"
    valid_start = "2020-01-01 00:00:00"
    valid_end = "2020-12-31 00:00:00"
    
    config.update_dates_for_walkforward(train_start, train_end, valid_start, valid_end)
    
    # Load Data
    print("Loading Data...")
    stock_df = load_mafia_data(config)
    
    # Create Trainer 
    stock_list = sorted(stock_df["stock"].unique().tolist())
    observer = MAFIAObserver(config, action_dim=len(stock_list))
    trainer = ObserverOfflineBatchTrainer(config, observer, device=th.device("cpu"))
    
    # Manually prepare tensors
    print("Preparing Tensors...")
    train_tensors = trainer.prepare_data_tensors(stock_df, stock_list, config.train_date_start, config.train_date_end)
    valid_tensors = trainer.prepare_data_tensors(stock_df, stock_list, config.valid_date_start, config.valid_date_end)
    
    # Extract Signals for Train
    print("Extracting Training Signals...")
    # Use sample_trajectory_batch instead of prepare_trajectories
    batch_train = trainer.sample_trajectory_batch(train_tensors, mode="TRAIN")
    
    train_vol = batch_train.vol_std20.flatten().numpy()
    train_dc = batch_train.dc_event_flag.flatten().numpy()
    train_breadth = batch_train.breadth_gap.flatten().numpy()
    train_vpi = batch_train.signed_vpi_zscore.flatten().numpy()
    
    # Extract Signals for Valid
    print("Extracting Validation Signals...")
    trainer.batch_size = 256 
    batch_valid = trainer.sample_trajectory_batch(valid_tensors, mode="EVAL")
    
    valid_vol = batch_valid.vol_std20.flatten().numpy()
    valid_dc = batch_valid.dc_event_flag.flatten().numpy()
    valid_breadth = batch_valid.breadth_gap.flatten().numpy()
    valid_vpi = batch_valid.signed_vpi_zscore.flatten().numpy()
    
    # Statistics
    print("\n" + "="*60)
    print("SIGNAL DISTRIBUTION DIAGNOSIS")
    print("="*60)
    
    def print_stats(name, train_data, valid_data):
        t_mean, t_std = train_data.mean(), train_data.std()
        v_mean, v_std = valid_data.mean(), valid_data.std()
        print(f"\n[{name}]")
        print(f"  TRAIN: Mean={t_mean:.6f}, Std={t_std:.6f}, Min={train_data.min():.4f}, Max={train_data.max():.4f}")
        print(f"  VALID: Mean={v_mean:.6f}, Std={v_std:.6f}, Min={valid_data.min():.4f}, Max={valid_data.max():.4f}")
        print(f"  Shift (Valid/Train): Mean Ratio={v_mean/(t_mean+1e-8):.2f}, Std Ratio={v_std/(t_std+1e-8):.2f}")
    
    print_stats("Vol_Std20", train_vol, valid_vol)
    print_stats("DC_Event_Flag", train_dc, valid_dc)
    print_stats("Breadth_Gap", train_breadth, valid_breadth)
    print_stats("Signed_VPI_Zscore", train_vpi, valid_vpi)
    
    # Correlation check for Risk
    # Risk target uses Volatility + Drawdown.
    
    return

if __name__ == "__main__":
    diagnose_signals()
