import os
import sys
import glob
import pandas as pd
import torch as th
import numpy as np
from sklearn.metrics import confusion_matrix

# Ensure Python path includes current directory
sys.path.append(os.getcwd())

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import create_offline_trainer
from scripts.train_observer_offline import load_mafia_data, build_expanding_schedule

def debug_confusion_matrix():
    print("[DEBUG] Starting Confusion Matrix analysis...")
    
    # 1. Setup Config & Data
    config = Config(create_dirs=False)
    config.seed = 2025
    
    # Ensure we pick up the FIX:
    # index_data_file should be "VNINDEX_1d_index.csv" in config or fallback
    print(f"[DEBUG] Config Index File: {getattr(config, 'index_data_file', 'Not Set')}")
    
    print("[DEBUG] Loading data...")
    stock_data = load_mafia_data(config)
    stock_list = sorted(stock_data["stock"].unique().tolist())
    
    # Load Market Data
    index_file = getattr(config, "index_data_file", "VNINDEX_1d_index.csv")
    market_file = os.path.join(getattr(config, "dataDir", "./data"), index_file)
    if os.path.exists(market_file):
        market_data = pd.read_csv(market_file, parse_dates=["date"])
        print(f"[DEBUG] Loaded Market Data: {len(market_data)} rows")
    else:
        market_data = None
        print("[ERROR] Market Data NOT Found! This will cause the issue.")

    # Setup Trainer
    action_dim = len(stock_list)
    observer = MAFIAObserver(config=config, action_dim=action_dim)
    trainer = create_offline_trainer(config, observer)
    
    # Prepare Tensors
    full_start = pd.Timestamp("2015-01-01")
    full_end = pd.Timestamp("2022-12-31")
    
    data_tensors = trainer.prepare_data_tensors(
        data=stock_data,
        stock_list=stock_list,
        start_date=full_start,
        end_date=full_end,
        market_data=market_data,
    )
    
    # 2. Target Iteration 0 (Validation 2017)
    schedule = build_expanding_schedule(2015, 2018, 2022)
    sched = schedule[0] # Iter 0
    print(f"[DEBUG] Targeting Iteration 0: Valid Year {sched['valid_year']}")
    
    valid_mask = (data_tensors["dates"] >= sched["valid_start"]) & (data_tensors["dates"] <= sched["valid_end"])
    valid_indices = np.where(valid_mask)[0]
    
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
    
    # 3. Load Latest Checkpoint (from buggy training)
    # Trying to find the best checkpoint from before
    ckpt_pattern = "observer_offline/checkpoints/temp_iter_0/epoch_*.pth"
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        ckpts = glob.glob("observer_offline/checkpoints/epoch_*.pth")
        
    if ckpts:
        # Sort to get latest
        latest_ckpt = sorted(ckpts, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))[-1]
        print(f"[DEBUG] Loading checkpoint: {latest_ckpt}")
        trainer.observer.load_checkpoint(latest_ckpt)
        trainer.observer.mafia_model.eval()
    else:
        print("[ERROR] No checkpoints found to evaluate!")
        return

    # 4. Run Evaluation
    # Just run 20 steps to get a stable estimate
    steps = 20 
    
    print(f"[DEBUG] Running {steps} evaluation batches...")
    
    # We can use validate_epoch's internal logic, but let's just use the API 
    # and trust that if F1 is still 0, it proves the point.
    
    try:
        result = trainer.validate_epoch(trainer_valid_tensors, steps=steps)
        print(f"\n[DEBUG] Validation Metrics (Avg over {steps} batches):")
        print(f"  Accuracy: {result.direction_accuracy:.4f}")
        print(f"  F1 Bear:  {result.direction_f1_bear:.4f}")
        print(f"  F1 Side:  {result.direction_f1_side:.4f}")
        print(f"  F1 Bull:  {result.direction_f1_bull:.4f}")
        
        if result.direction_f1_bear < 0.0001:
            print("\n[CONCLUSION] F1 Bear is ZERO.")
            print("             This confirms the MODEL predicts 0 Bears (or near 0).")
            print("             The fix is VALID (code ran without error), but the Model needs RETRAINING.")
        else:
            print("\n[CONCLUSION] F1 Bear is NON-ZERO! The model has some capability.")
            
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")

if __name__ == "__main__":
    debug_confusion_matrix()
