
import sys
import os
import torch as th
import numpy as np
import pandas as pd
from collections import Counter

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.MAFIA.config import Config
from agents.MAFIA.RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from agents.MAFIA.utils.mafia_data_loader import load_mafia_data
from agents.MAFIA.scripts.train_observer_offline import build_expanding_schedule

def analyze_validation_labels():
    print("Initializing Config and Trainer...")
    config = Config()
    
    # Fix data path
    config.dataDir = os.path.abspath(os.path.join(REPO_ROOT, "agents/MAFIA/data"))
    
    # Use metrics from iter_0_valid_2017 as referenced in current task
    start_year = 2015
    first_infer_year = 2018
    last_infer_year = 2018 # Just checking the first iteration context
    
    schedule = build_expanding_schedule(start_year, first_infer_year, last_infer_year)
    iter_0_schedule = schedule[0] # Iteration 0
    
    print(f"Target Period: {iter_0_schedule['valid_label']}")
    
    # Load Data
    print("Loading Data...")
    df = load_mafia_data(config)
    
    
    # Calculate action_dim (N) from data
    N = df["stock"].nunique()
    print(f"Number of stocks (N): {N}")
    
    # Prepare Trainer
    from agents.MAFIA.RL_controller.mafia_observer import MAFIAObserver
    print("Initializing MAFIAObserver...")
    observer = MAFIAObserver(config, action_dim=N)
    
    print("Initializing Trainer...")
    trainer = ObserverOfflineBatchTrainer(config, observer)
    
    # Prepare arguments for prepare_data_tensors
    stock_list = sorted(df["stock"].unique().tolist())
    start_date = df["date"].min()
    end_date = df["date"].max()
    
    
    # Load Market Data (VNINDEX)
    index_path = os.path.join(config.dataDir, "VNINDEX_1d_index.csv")
    print(f"Loading Market Data: {index_path}")
    market_df = pd.read_csv(index_path, parse_dates=["date"])
    
    # Pre-compute data tensors
    print("Preparing data tensors...")
    trainer.data_tensors = trainer.prepare_data_tensors(
        data=df, 
        stock_list=stock_list, 
        start_date=start_date, 
        end_date=end_date,
        market_data=market_df
    )
    
    
    # Analyze Validation Period (2017-10-01 to 2017-12-31)
    valid_start = iter_0_schedule["valid_start"]
    valid_end = iter_0_schedule["valid_end"]
    
    print(f"\n[ANALYSIS] Checking VALIDATION Period: {valid_start.date()} -> {valid_end.date()}")
    
    dates = trainer.data_tensors["dates"]
    valid_mask_indices = np.where((dates >= valid_start) & (dates <= valid_end))[0]
    total_valid = len(valid_mask_indices)
    print(f"Total Valid Samples: {total_valid}")
    
    # Compute labels
    min_start = 0
    max_start = len(dates) - trainer.horizon - trainer.T_w - 1
    dir_lookahead = config.direction_label_lookahead
    
    print("Computing class indices (labels)...")
    class_indices = trainer._build_class_indices(
        data_tensors=trainer.data_tensors,
        min_start=min_start,
        max_start=max_start,
        dir_lookahead=dir_lookahead
    )
    
    print("\n--- Validation Data Label Distribution ---")
    
    classes = {0: "Bear", 1: "Side", 2: "Bull"}
    valid_set = set(valid_mask_indices)
    
    total_counted = 0
    for cls_idx in [0, 1, 2]:
        cls_set = set(class_indices[cls_idx])
        valid_in_class = valid_set.intersection(cls_set)
        count = len(valid_in_class)
        pct = (count / total_valid) * 100 if total_valid > 0 else 0
        print(f"Class {cls_idx} ({classes[cls_idx]}): {count:5d} ({pct:.2f}%)")
        total_counted += count
        
    print(f"Total Labelled: {total_counted}")  
    print("------------------------------------------")
    print(f"Total Labelled: {total_counted}")
    print("------------------------------------------")
    
    # --- DEBUGGING SECTOR ---
    print("\n[DEBUG] Labeling Parameters:")
    print(f"  ATR Period: {getattr(config, 'direction_label_atr_period', 14)}")
    print(f"  ATR Multiplier (k): {getattr(config, 'direction_label_atr_multiplier', 2.0)}")
    print(f"  Delta Min: {getattr(config, 'direction_label_delta_min', 0.02)}")
    print(f"  Stop Loss: {getattr(config, 'direction_label_stop_loss', -0.07)}")
    print(f"  Lookahead: {getattr(config, 'direction_label_lookahead', 14)}")
    
    # Dump validation period data with labels for inspection
    debug_data = []
    market_close = trainer.data_tensors["market_ochlv"][:, 0, 1].cpu().numpy()
    
    for i, idx in enumerate(valid_mask_indices):
        date_val = dates[idx]
        if isinstance(date_val, (int, float, np.integer)): # Handle if dates are not timestamps
             # Try to map back if possible, or just print index
             pass
        
        # Get label from class_indices
        label = -1
        for k in class_indices:
            if idx in class_indices[k]:
                label = k
                break
        
        # Get Price
        price = market_close[idx]
        
        # Get Future Price (approx logic)
        fut_idx = min(idx + 14, len(market_close)-1)
        fut_price = market_close[fut_idx]
        return_14d = (fut_price / price) - 1.0
        
        debug_data.append({
            "idx": idx,
            "date": str(date_val), # Convert to string to be safe
            "price": price,
            "future_price": fut_price,
            "return_14d": return_14d,
            "label": label,
            "class": classes.get(label, "Unknown")
        })
        
    debug_df = pd.DataFrame(debug_data)
    print("\n[DEBUG] Validation Data Head:")
    print(debug_df.head(10))
    
    debug_csv = "debug_validation_labels.csv"
    debug_df.to_csv(debug_csv, index=False)
    print(f"\n[DEBUG] Saved label debug data to: {debug_csv}")
    print("------------------------------------------")

if __name__ == "__main__":
    analyze_validation_labels()
