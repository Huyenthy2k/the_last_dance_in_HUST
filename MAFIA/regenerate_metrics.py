import argparse
import os
import sys
import glob
import pandas as pd
import torch as th
import numpy as np
import copy
from dataclasses import asdict

# Ensure Python path includes current directory
sys.path.append(os.getcwd())

from config import Config
from RL_controller.mafia_modules import MAFIAModel
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import create_offline_trainer
from RL_controller.validation_tracker import ValidationMetricsTracker
from scripts.train_observer_offline import load_mafia_data, build_expanding_schedule

def regenerate_metrics():
    parser = argparse.ArgumentParser(description="Regenerate metrics (train & valid) from checkpoints")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year of training data")
    parser.add_argument("--first-infer-year", type=int, default=2018, help="First inference year")
    parser.add_argument("--last-infer-year", type=int, default=2022, help="Last inference year")
    parser.add_argument("--iter", type=int, default=0, help="Iteration index to regenerate (0-indexed)")
    parser.add_argument("--output-dir", type=str, default="observer_offline", help="Base output directory")
    parser.add_argument("--epochs", type=str, default=None, help="Specific epochs to regenerate (e.g. '0-5' or '1,3,5')")
    parser.add_argument("--valid-only", action="store_true", help="Skip training metrics regeneration")
    args = parser.parse_args()

    print(f"[REGEN] Starting metric regeneration for Iteration {args.iter}...")
    
    # Parse target epochs
    target_epochs = set()
    if args.epochs:
        if "-" in args.epochs:
            start, end = map(int, args.epochs.split("-"))
            target_epochs = set(range(start, end + 1))
        else:
            target_epochs = set(map(int, args.epochs.split(",")))
        print(f"[REGEN] Targeting specific epochs: {sorted(list(target_epochs))}")

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
    print(f"        Train Range: {sched['train_label']}")
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
    
    # Prepare Tensors (Full Range)
    full_start = pd.Timestamp(f"{args.start_year}-01-01")
    full_end = pd.Timestamp(f"{args.last_infer_year}-12-31")
    
    data_tensors = trainer.prepare_data_tensors(
        data=stock_data,
        stock_list=stock_list,
        start_date=full_start,
        end_date=full_end,
        market_data=market_data,
    )
    
    # 4. Prepare TRAIN and VALID Tensors
    # Slice for VALID
    valid_mask = (data_tensors["dates"] >= sched["valid_start"]) & (data_tensors["dates"] <= sched["valid_end"])
    valid_indices = np.where(valid_mask)[0]
    
    # Slice for TRAIN
    train_mask = (data_tensors["dates"] >= sched["train_start"]) & (data_tensors["dates"] <= sched["train_end"])
    train_indices = np.where(train_mask)[0]

    if len(valid_indices) == 0:
        print(f"[REGEN] Error: No validation data found for range {sched['valid_label']}")
        return
    if len(train_indices) == 0:
        print(f"[REGEN] Error: No training data found for range {sched['train_label']}")
        return

    # Helper to slice
    def get_slice(indices):
        return {
            "ochlv": data_tensors["ochlv"][indices[0] : indices[-1] + 1],
            "returns": data_tensors["returns"][indices[0] : indices[-1] + 1],
            "market_ochlv": data_tensors.get("market_ochlv", None)[indices[0] : indices[-1] + 1] if data_tensors.get("market_ochlv") is not None else None,
            "market_returns": data_tensors.get("market_returns", None)[indices[0] : indices[-1] + 1] if data_tensors.get("market_returns") is not None else None,
            "dates": data_tensors["dates"][indices],
            "stock_list": data_tensors["stock_list"],
            "T_total": len(indices),
            "N": data_tensors["N"],
        }
    
    valid_tensors = get_slice(valid_indices)
    train_tensors = get_slice(train_indices)

    # 5. Locate Checkpoints
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"temp_iter_{args.iter}")
    if not os.path.exists(ckpt_dir):
        print(f"[REGEN] Warning: Checkpoint directory not found: {ckpt_dir}")
        print(f"        Trying fallback to root checkpoints dir...")
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    print(f"[REGEN] Searching for checkpoints in: {ckpt_dir}")
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    
    if not ckpts:
        print(f"[REGEN] No epoch checkpoints found at {ckpt_dir}. Exiting.")
        return

    # Sort by epoch
    def extract_epoch(p):
        try:
            return int(os.path.basename(p).split("_")[1].split(".")[0])
        except:
            return -1
    
    ckpts.sort(key=extract_epoch)
    
    # Valid Tracker
    iter_out_dir = os.path.join(args.output_dir, f"iter_{args.iter}_valid_{sched['valid_year']}")
    os.makedirs(iter_out_dir, exist_ok=True)
    tracker = ValidationMetricsTracker(iter_out_dir)
    
    # Train Metrics CSV
    train_csv_path = os.path.join(iter_out_dir, "train_metrics.csv")
    
    # Load existing train metrics
    existing_train_epochs = set()
    if os.path.exists(train_csv_path):
        df_t = pd.read_csv(train_csv_path)
        if "epoch" in df_t.columns:
            existing_train_epochs = set(df_t["epoch"].values)
    
    print(f"[REGEN] Existing valid epochs: (using tracker history)")
    valid_csv_path = os.path.join(iter_out_dir, "valid_metrics.csv")
    if os.path.exists(valid_csv_path):
        tracker.load_history_from_csv(valid_csv_path)
    
    existing_valid_epochs = set([h.epoch for h in tracker.history])

    # 6. Process
    for ckpt in ckpts:
        try:
            epoch_num = extract_epoch(ckpt)
            if epoch_num == -1: continue

            # Filter by specific epochs if requested
            if target_epochs and epoch_num not in target_epochs:
                continue

            # Skip Logic:
            # - If target_epochs is set, NEVER skip (Force Regen)
            # - Else, use standard skip if exists
            
            force_regen = (epoch_num in target_epochs)
            
            if not force_regen:
                # Optimization: Skip if BOTH exist
                if epoch_num in existing_train_epochs and epoch_num in existing_valid_epochs:
                    # print(f"[REGEN] Skipping Epoch {epoch_num} (already exists in both).")
                    continue
                
            print(f"[REGEN] Processing Epoch {epoch_num}...")
            loaded_epoch = trainer.observer.load_checkpoint(ckpt)
            trainer._epoch = loaded_epoch
            
            # Recalculate Lambda for correct "Best Model" logic
            trainer._current_lambda_epoch = trainer._compute_lambda_epoch(loaded_epoch)
            is_full_penalty = trainer._current_lambda_epoch >= 0.999
            
            # --- VALIDATION ---
            # Run if missing OR forced
            if epoch_num not in existing_valid_epochs or force_regen:
                val_result = trainer.validate_epoch(valid_tensors)
                # Pass logic flag
                tracker.add_epoch(val_result, allow_best_update=is_full_penalty)
                print(f"        [VALID] Added. CES: {val_result.ces_score:.4f} (Lambda={trainer._current_lambda_epoch:.2f})")
            else:
                pass # Already have it

            # --- TRAINING ---
            if not args.valid_only and (epoch_num not in existing_train_epochs or force_regen):
                print("        [TRAIN] Calculating metrics on Train Set...")
                train_res = trainer.validate_epoch(train_tensors) # Returns ObserverValidationResult
                
                # Convert to dict
                train_dict = asdict(train_res)
                train_dict["epoch"] = epoch_num # Ensure correct epoch
                
                # Remove CES columns
                keys_to_remove = [k for k in train_dict.keys() if "ces_" in k]
                for k in keys_to_remove:
                    del train_dict[k]
                
                # Append to CSV immediately
                df_new = pd.DataFrame([train_dict])
                header = not os.path.exists(train_csv_path)
                df_new.to_csv(train_csv_path, mode='a', header=header, index=False)
                print(f"        [TRAIN] Saved to {train_csv_path}")
                existing_train_epochs.add(epoch_num)

        except Exception as e:
            print(f"[ERROR] Failed {ckpt}: {e}")

    # Final save for validation tracker
    out_valid = tracker.save_validation_history()
    print(f"[REGEN] Valid metrics synced to: {out_valid}")

if __name__ == "__main__":
    regenerate_metrics()
