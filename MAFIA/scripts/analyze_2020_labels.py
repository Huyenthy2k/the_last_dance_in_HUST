#!/usr/bin/env python3
import sys
import os
import torch as th
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.getcwd(), "agents", "MAFIA"))
from config import Config
from scripts.train_observer_offline import load_mafia_data
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from RL_controller.mafia_observer import MAFIAObserver

def analyze_2020_labels():
    config = Config(create_dirs=False)
    config.device = th.device("cpu")
    
    # 2020 Validation
    train_start = "2015-01-01 00:00:00"
    train_end = "2019-12-31 00:00:00"
    valid_start = "2020-01-01 00:00:00"
    valid_end = "2020-12-31 00:00:00"
    config.update_dates_for_walkforward(train_start, train_end, valid_start, valid_end)
    
    print("Loading Data...")
    stock_df = load_mafia_data(config)
    stock_list = sorted(stock_df["stock"].unique().tolist())
    
    observer = MAFIAObserver(config, action_dim=len(stock_list))
    trainer = ObserverOfflineBatchTrainer(config, observer, device=th.device("cpu"))
    
    print("Generating Labels for 2020...")
    valid_tensors = trainer.prepare_data_tensors(stock_df, stock_list, config.valid_date_start, config.valid_date_end)
    
    # Sample ALL trajectories to get full label statistics
    # Use a large batch size or iterate covers
    trainer.batch_size = 1000 # Should cover most of 252 days * overlap
    batch = trainer.sample_trajectory_batch(valid_tensors, mode="EVAL")
    
    labels = batch.direction_labels.flatten().numpy()
    
    # Statistics
    total = len(labels)
    bear = (labels == 0).sum()
    side = (labels == 1).sum()
    bull = (labels == 2).sum()
    
    print("\n" + "="*50)
    print("2020 LABEL REGIME ANALYSIS")
    print("="*50)
    print(f"Total Labels: {total}")
    print(f"Bear: {bear} ({bear/total*100:.2f}%)")
    print(f"Side: {side} ({side/total*100:.2f}%)")
    print(f"Bull: {bull} ({bull/total*100:.2f}%)")
    
    # Also check Training distribution for reference
    print("\nCheck Training (2015-2019)...")
    train_tensors = trainer.prepare_data_tensors(stock_df, stock_list, config.train_date_start, config.train_date_end)
    batch_train = trainer.sample_trajectory_batch(train_tensors, mode="TRAIN")
    t_labels = batch_train.direction_labels.flatten().numpy()
    t_total = len(t_labels)
    t_bear = (t_labels == 0).sum()
    t_side = (t_labels == 1).sum()
    t_bull = (t_labels == 2).sum()
    
    print(f"Training Bear: {t_bear/t_total*100:.2f}%")
    print(f"Training Side: {t_side/t_total*100:.2f}%")
    print(f"Training Bull: {t_bull/t_total*100:.2f}%")
    
    return

if __name__ == "__main__":
    analyze_2020_labels()
