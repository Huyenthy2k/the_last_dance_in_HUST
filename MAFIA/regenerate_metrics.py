import argparse
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
from RL_controller.validation_tracker import ValidationMetricsTracker
from scripts.train_observer_offline import load_mafia_data, build_expanding_schedule

def regenerate_metrics():
    parser = argparse.ArgumentParser(description="Regenerate valid_metrics.csv from checkpoints")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year of training data")
    parser.add_argument("--first-infer-year", type=int, default=2018, help="First inference year")
    parser.add_argument("--last-infer-year", type=int, default=2022, help="Last inference year")
    parser.add_argument("--iter", type=int, default=0, help="Iteration index to regenerate (0-indexed)")
    parser.add_argument("--output-dir", type=str, default="observer_offline", help="Base output directory")
    args = parser.parse_args()

    print(f"[REGEN] Starting metric regeneration for Iteration {args.iter}...")
    
    # 1. Setup Config & Data
    config = Config(create_dirs=False)
    config.seed = 2025
    
    # 2. Build Schedule to target specific iteration
    schedule = build_expanding_schedule(args.start_year, args.first_infer_year, args.last_infer_year)
    if args.iter < 0 or args.iter >= len(schedule):
        print(f"[REGEN] Error: Iteration {args.iter} out of range (0-{len(schedule)-1})")
        return

    sched = schedule[args.iter]
    print(f"[REGEN] Targeting Iteration {args.iter}: Valid Year {sched['valid_year']}")
    print(f"        Valid Range: {sched['valid_label']}")
    
    # 3. Load Data
    print("[REGEN] Loading data...")
    stock_data = load_mafia_data(config)
    stock_list = sorted(stock_data["stock"].unique().tolist())
    
    # Load Market Data
    index_file = getattr(config, "index_data_file", "VNINDEX_1d_index.csv")
    market_file = os.path.join(getattr(config, "dataDir", "./data"), index_file)
    if os.path.exists(market_file):
        market_data = pd.read_csv(market_file, parse_dates=["date"])
    else:
        market_data = None

    # Setup Device
    if th.cuda.is_available():
        device = th.device("cuda")
    elif th.backends.mps.is_available():
        device = th.device("mps")
    else:
        device = th.device("cpu")
        
    action_dim = len(stock_list)
    observer = MAFIAObserver(config=config, action_dim=action_dim)
    trainer = create_offline_trainer(config, observer)
    
    # Prepare Tensors (Full Range to be safe)
    # We load full range because prepare_data_tensors handles filtering internally if needed,
    # but here we need to ensure we cover the validation period.
    full_start = pd.Timestamp(f"{args.start_year}-01-01")
    full_end = pd.Timestamp(f"{args.last_infer_year}-12-31")
    
    data_tensors = trainer.prepare_data_tensors(
        data=stock_data,
        stock_list=stock_list,
        start_date=full_start,
        end_date=full_end,
        market_data=market_data,
    )
    
    # 4. Prepare Validation Tensors
    valid_mask = (data_tensors["dates"] >= sched["valid_start"]) & (data_tensors["dates"] <= sched["valid_end"])
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print(f"[REGEN] Error: No validation data found for range {sched['valid_label']}")
        return

    valid_tensors = {
        "ochlv": data_tensors["ochlv"][valid_indices[0] : valid_indices[-1] + 1],
        "returns": data_tensors["returns"][valid_indices[0] : valid_indices[-1] + 1],
        "market_ochlv": data_tensors.get("market_ochlv", None)[valid_indices[0] : valid_indices[-1] + 1] if data_tensors.get("market_ochlv") is not None else None,
        "market_returns": data_tensors.get("market_returns", None)[valid_indices[0] : valid_indices[-1] + 1] if data_tensors.get("market_returns") is not None else None,
        "dates": data_tensors["dates"][valid_indices],
        "stock_list": data_tensors["stock_list"],
        "T_total": len(valid_indices),
        "N": data_tensors["N"],
    }

    # 5. Locate Checkpoints
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"temp_iter_{args.iter}")
    if not os.path.exists(ckpt_dir):
        print(f"[REGEN] Warning: Checkpoint directory not found: {ckpt_dir}")
        print(f"        Trying fallback to root checkpoints dir...")
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    print(f"[REGEN] Searching for checkpoints in: {ckpt_dir}")
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    
    # Also include latest?
    latest = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    if os.path.exists(latest):
        ckpts.append(latest)
    
    if not ckpts:
        print(f"[REGEN] No checkpoints found at {ckpt_dir}. Exiting.")
        return

    # Sort by epoch number to process in order
    def extract_epoch(p):
        base = os.path.basename(p)
        if "latest" in base: return 999999
        try:
            return int(base.split("_")[1].split(".")[0])
        except:
            return -1
    
    ckpts.sort(key=extract_epoch)
    
    # Tracker
    out_dir = os.path.join(args.output_dir, f"iter_{args.iter}_valid_{sched['valid_year']}")
    os.makedirs(out_dir, exist_ok=True)
    tracker = ValidationMetricsTracker(out_dir)
    
    # 6. Process
    for ckpt in ckpts:
        try:
            print(f"[REGEN] processing {ckpt}...")
            loaded_epoch = trainer.observer.load_checkpoint(ckpt)
            
            # Sync trainer epoch for logging
            trainer._epoch = loaded_epoch 
            
            # Use 'validate_epoch' which calculates all metrics
            # Note: steps=None lets it auto-calc based on data size (usually ~10 steps)
            result = trainer.validate_epoch(valid_tensors)
            
            tracker.add_epoch(result)
            print(f"[REGEN] Epoch {loaded_epoch} processed. CES: {result.ces_score:.4f}")
            print(f"        F1-Bear: {result.direction_f1_bear:.4f} | F1-Side: {result.direction_f1_side:.4f} | F1-Bull: {result.direction_f1_bull:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed {ckpt}: {e}")

    # 7. Save (Final)
    csv_path = tracker.save_validation_history()
    print(f"[REGEN] Saved recovered metrics to: {csv_path}")

if __name__ == "__main__":
    regenerate_metrics()
