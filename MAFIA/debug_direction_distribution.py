import os
import sys
import glob
import pandas as pd
import torch as th
import numpy as np

# Ensure Python path includes current directory
sys.path.append(os.getcwd())

from config import Config
from RL_controller.mafia_modules import MAFIAModel
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import create_offline_trainer
from scripts.train_observer_offline import load_mafia_data, build_expanding_schedule

def debug_distribution():
    print("[DEBUG] Starting distribution analysis...")
    
    # 1. Setup Config & Data
    config = Config(create_dirs=False)
    config.seed = 2025
    
    # Use same params as training script
    start_year = 2015
    first_infer_year = 2018
    
    print("[DEBUG] Loading data...")
    stock_data = load_mafia_data(config)
    stock_list = sorted(stock_data["stock"].unique().tolist())
    
    # Load Market Data
    market_file = os.path.join(getattr(config, "dataDir", "./data"), "VNINDEX_1d_index.csv")
    if os.path.exists(market_file):
        market_data = pd.read_csv(market_file, parse_dates=["date"])
    else:
        market_data = None
        print("[WARN] No market data found!")

    # Setup Trainer (for data prep utils)
    action_dim = len(stock_list)
    observer = MAFIAObserver(config=config, action_dim=action_dim)
    trainer = create_offline_trainer(config, observer)
    
    # Prepare Tensors
    full_start = pd.Timestamp(f"{start_year}-01-01")
    full_end = pd.Timestamp(f"2022-12-31")
    
    data_tensors = trainer.prepare_data_tensors(
        data=stock_data,
        stock_list=stock_list,
        start_date=full_start,
        end_date=full_end,
        market_data=market_data,
    )
    
    # 2. Target Iteration 0 (Validation 2017)
    schedule = build_expanding_schedule(start_year, first_infer_year, 2022)
    sched = schedule[0] # Iter 0
    print(f"[DEBUG] Targeting Iteration 0: Valid Year {sched['valid_year']}")
    
    # Filter for validation period
    valid_mask = (data_tensors["dates"] >= sched["valid_start"]) & (data_tensors["dates"] <= sched["valid_end"])
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("[ERROR] No validation data found!")
        return

    # Check Labels in Validation Set
    print("\n[DEBUG] Analyzing Ground Truth Labels...")

    # We reuse the logic from regenerate_metrics
    trainer_valid_tensors = {
        "ochlv": data_tensors["ochlv"][valid_indices[0] : valid_indices[-1] + 1],
        "returns": data_tensors["returns"][valid_indices[0] : valid_indices[-1] + 1],
        "market_ochlv": data_tensors.get("market_ochlv")[valid_indices[0] : valid_indices[-1] + 1],
        "market_returns": data_tensors.get("market_returns")[valid_indices[0] : valid_indices[-1] + 1],
        "dates": data_tensors["dates"][valid_indices],
        "stock_list": data_tensors["stock_list"],
        "T_total": len(valid_indices),
        "N": data_tensors["N"],
    }
    
    steps = 20 # Sample enough to cover most/all of the validation period (sliding window)
    print(f"[DEBUG] Running {steps} validation steps (sampling)...")
    
    all_labels = []
    
    with th.no_grad():
        for step in range(steps):
             # Sample Eval batch (sequential/random depending on implementation, usually random but EVAL mode might differ)
             # Actually sample_trajectory_batch 'EVAL' usually just samples random? 
             # Let's check: in trainer, EVAL mode just sets no_grad usually? 
             # trainer code: sample_trajectory_batch(mode="EVAL") -> checks mode?
             # Actually I recall seeing "random_trajectory" enforcing in init.
             # So it might be random sampling. To be safe we sample enough.
             
             batch = trainer.sample_trajectory_batch(trainer_valid_tensors, mode="EVAL")
             
             # Collect Labels
             # batch.direction_labels is (B, T_m)
             labels = batch.direction_labels.cpu().numpy().flatten()
             all_labels.extend(labels)
             
    all_labels = np.array(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    print("\n[DEBUG] Label Distribution (Validation 2017):")
    total = len(all_labels)
    found_bear = False
    for u, c in zip(unique, counts):
        label_name = ["Bear", "Side", "Bull"][int(u)]
        print(f"  {label_name} ({int(u)}): {c} ({c/total*100:.2f}%)")
        if int(u) == 0:
            found_bear = True
        
    if not found_bear:
        print("\n[CONCLUSION] ROOT CAUSE FOUND: No 'Bear' labels in validation set!")
    else:
        print("\n[CONCLUSION] 'Bear' labels exist. Issue is likely model predictions (0 Bear preds).")

if __name__ == "__main__":
    debug_distribution()
