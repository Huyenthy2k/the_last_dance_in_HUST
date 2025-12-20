import argparse
import os
import sys
import glob
import pandas as pd
import torch as th
import numpy as np
from dataclasses import asdict
import re

# Ensure Python path includes current directory
sys.path.append(os.getcwd())

from config import Config
from RL_controller.mafia_modules import MAFIAModel
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import create_offline_trainer, ObserverOfflineBatchTrainer
from scripts.train_observer_offline import load_mafia_data, build_expanding_schedule

def regenerate_all_metrics():
    parser = argparse.ArgumentParser(description="Regenerate all metrics (train/valid) from checkpoints")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing observer_offline outputs")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year of training data")
    parser.add_argument("--first-infer-year", type=int, default=2018, help="First inference year")
    parser.add_argument("--last-infer-year", type=int, default=2022, help="Last inference year")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    print(f"[REGEN-ALL] Scanning root directory: {root_dir}")

    if not os.path.exists(root_dir):
        print(f"[ERROR] Root directory not found: {root_dir}")
        return

    # 1. Setup Global Config & Data
    config = Config(create_dirs=False)
    # Point to correct data directory relative to root (assuming script run from repo root)
    # config.dataDir = os.path.join(os.getcwd(), "agents", "MAFIA", "data")
    config.seed = 2025
    # Force CPU
    config.device = th.device("cpu")
    config.mafia_batch_size = 32 # Reduce batch size for CPU safety


    # 2. Load Data Once
    print("[REGEN-ALL] Loading data...")
    stock_data = load_mafia_data(config)
    stock_list = sorted(stock_data["stock"].unique().tolist())
    
    # Load Market Data
    index_file = getattr(config, "index_data_file", "VNINDEX_1d_index.csv")
    market_file = os.path.join(getattr(config, "dataDir", "./data"), index_file)
    if os.path.exists(market_file):
        market_data = pd.read_csv(market_file, parse_dates=["date"])
    else:
        market_data = None
        
    action_dim = len(stock_list)
    observer = MAFIAObserver(config=config, action_dim=action_dim)
    trainer = ObserverOfflineBatchTrainer(config=config, observer=observer, device=config.device)

    # Prepare Full Tensors
    full_start = pd.Timestamp(f"{args.start_year}-01-01")
    full_end = pd.Timestamp(f"{args.last_infer_year}-12-31")
    
    data_tensors = trainer.prepare_data_tensors(
        data=stock_data,
        stock_list=stock_list,
        start_date=full_start,
        end_date=full_end,
        market_data=market_data,
    )

    # 3. Build Schedule to map iterations
    schedule = build_expanding_schedule(args.start_year, args.first_infer_year, args.last_infer_year)

    # 4. Find Iterations in Root Dir
    # Look for folders like 'iter_0_valid_2017'
    iter_folders = glob.glob(os.path.join(root_dir, "iter_*"))
    
    for folder_path in sorted(iter_folders):
        folder_name = os.path.basename(folder_path)
        print(f"\n[REGEN-ALL] Found iteration folder: {folder_name}")
        
        # Parse iter index from name
        try:
            # format: iter_X_valid_YYYY
            match = re.match(r"iter_(\d+)_valid_(\d+)", folder_name)
            if not match:
                print(f"  [SKIP] Skipping unknown folder format: {folder_name}")
                continue
            
            iter_idx = int(match.group(1))
            valid_year = int(match.group(2))
            
            if iter_idx >= len(schedule):
                print(f"  [WARN] Iteration {iter_idx} in folder name exceeds schedule length. Skipping.")
                continue

            sched = schedule[iter_idx]
            
            # Verify valid year matches
            if sched['valid_year'] != valid_year:
                print(f"  [WARN] Folder valid year {valid_year} does not match schedule valid year {sched['valid_year']}. Using schedule.")
                
            print(f"  [INFO] Mapped to Iteration {iter_idx}: Train {sched['train_label']} | Valid {sched['valid_label']}")
            
            # 5. Prepare Tensors for this Iteration
            # TRAIN Tensors
            train_mask = (data_tensors["dates"] >= sched["train_start"]) & (data_tensors["dates"] <= sched["train_end"])
            train_indices = np.where(train_mask)[0]
            
            if len(train_indices) == 0:
                print("  [ERROR] No training data found.")
                continue

            # Pad start index with T_w to include lookback context
            pad_start = max(0, train_indices[0] - trainer.T_w)
            
            iter_train_tensors = {
                "ochlv": data_tensors["ochlv"][pad_start : train_indices[-1] + 1],
                "returns": data_tensors["returns"][pad_start : train_indices[-1] + 1],
                "market_ochlv": data_tensors.get("market_ochlv")[pad_start : train_indices[-1] + 1] if data_tensors.get("market_ochlv") is not None else None,
                "market_returns": data_tensors.get("market_returns")[pad_start : train_indices[-1] + 1] if data_tensors.get("market_returns") is not None else None,
                "dates": data_tensors["dates"][pad_start : train_indices[-1] + 1],
                "stock_list": data_tensors["stock_list"],
                "T_total": len(data_tensors["dates"][pad_start : train_indices[-1] + 1]),
                "N": data_tensors["N"],
            }

            # VALID Tensors
            valid_mask = (data_tensors["dates"] >= sched["valid_start"]) & (data_tensors["dates"] <= sched["valid_end"])
            valid_indices = np.where(valid_mask)[0]
             
            if len(valid_indices) == 0:
                 print("  [ERROR] No validation data found.")
                 continue
                 
            # Pad start index with T_w for validation too
            pad_valid_start = max(0, valid_indices[0] - trainer.T_w)
            
            iter_valid_tensors = {
                "ochlv": data_tensors["ochlv"][pad_valid_start : valid_indices[-1] + 1],
                "returns": data_tensors["returns"][pad_valid_start : valid_indices[-1] + 1],
                "market_ochlv": data_tensors.get("market_ochlv")[pad_valid_start : valid_indices[-1] + 1] if data_tensors.get("market_ochlv") is not None else None,
                "market_returns": data_tensors.get("market_returns")[pad_valid_start : valid_indices[-1] + 1] if data_tensors.get("market_returns") is not None else None,
                "dates": data_tensors["dates"][pad_valid_start : valid_indices[-1] + 1],
                "stock_list": data_tensors["stock_list"],
                "T_total": len(data_tensors["dates"][pad_valid_start : valid_indices[-1] + 1]),
                "N": data_tensors["N"],
            }
            
            # 6. Find Checkpoints
            ckpt_dir = os.path.join(root_dir, "checkpoints", f"temp_iter_{iter_idx}")
            if not os.path.exists(ckpt_dir):
                 print(f"  [WARN] Checkpoint dir {ckpt_dir} not found. Trying root checkpoints.")
                 ckpt_dir = os.path.join(root_dir, "checkpoints")

            ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
            
            def extract_epoch(p):
                base = os.path.basename(p)
                try: return int(base.split("_")[1].split(".")[0])
                except: return -1
            ckpts.sort(key=extract_epoch)
            
            if not ckpts:
                print(f"  [WARN] No checkpoints found in {ckpt_dir}")
                continue
                
            print(f"  [INFO] Found {len(ckpts)} checkpoints. Regenerating...")

            # 7. Process Checkpoints
            train_records = []
            valid_records = []
            
            for ckpt in ckpts:
                try:
                    loaded_epoch = trainer.observer.load_checkpoint(ckpt)
                    trainer.observer.mafia_model.to(config.device)
                    trainer._epoch = loaded_epoch
                    
                    print(f"    [Running] Epoch {loaded_epoch} (Train Calc)...", flush=True)

                    # --- TRAIN METRICS ---
                    # Use subset steps for speed on training set, but compute_loss=True
                    # Calculate available steps
                    train_available = iter_train_tensors["T_total"] - trainer.T_m - trainer.horizon - trainer.T_w
                    # Use enough steps to get a meaningful loss (e.g., 20)
                    steps_train = 5 # Reduced from 20 for speed on CPU
                    
                    res_train = trainer.validate_epoch(iter_train_tensors, steps=steps_train, compute_loss=True)
                    rec_train = asdict(res_train)
                    
                    # Standardize format
                    rec_train["direction_accuracy"] = round(rec_train["direction_accuracy"] * 100.0, 5)
                    rec_train["rl_advantage"] = rec_train.get("topk_advantage", 0.0)
                    rec_train["rl_reward"] = rec_train.get("net_reward", 0.0)
                    # Clean CES
                    for k in [k for k in rec_train if "ces_" in k]: del rec_train[k]
                    # Ensure epoch is correct (sometimes checkpoint epoch is diff from filename if renamed)
                    # Use filename epoch to be safe/consistent with sorting
                    epoch_from_file = extract_epoch(ckpt)
                    if epoch_from_file != -1: rec_train["epoch"] = epoch_from_file

                    train_records.append(rec_train)
                    
                    # --- VALID METRICS ---
                    # Use subset steps or full (validation is usually small enough for full)
                    # Train script uses batches_per_epoch // 4. Let's use 10 steps or full.
                    # A quarter is typically 60 days. T_m=30. available ~ 30. batch=64.
                    # Usually just 1 batch covers the whole validation set if batch size large.
                    # Let's use steps=5 to be safe and cover enough ground.
                    
                    print(f"    [Running] Epoch {loaded_epoch} (Valid Calc)...", flush=True)
                    res_valid = trainer.validate_epoch(iter_valid_tensors, steps=5, compute_loss=True)
                    rec_valid = asdict(res_valid)
                    
                    # Assume valid metrics format is clean (has CES etc if available, but here compute_loss=True)
                    if epoch_from_file != -1: rec_valid["epoch"] = epoch_from_file
                    
                    valid_records.append(rec_valid)
                    
                    print(f"    [Done] Epoch {rec_train['epoch']} | L_total: {rec_train.get('loss_total',0):.4f} | L_bal: {rec_train.get('loss_bal',0):.4f}", flush=True)

                except Exception as e:
                    print(f"    [ERROR] Failed {ckpt}: {e}")

            # 8. Save CSVs
            # Setup paths
            train_csv = os.path.join(folder_path, "train_metrics.csv")
            valid_csv = os.path.join(folder_path, "valid_metrics.csv")
            
            # Save Train
            if train_records:
                update_csv(train_csv, train_records)
            
            # Save Valid
            if valid_records:
                update_csv(valid_csv, valid_records)

        except Exception as e:
             print(f"  [ERROR] Processing folder {folder_name}: {e}")

def update_csv(path, records):
    new_df = pd.DataFrame(records)
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            # Ensure epoch is int
            if "epoch" in existing.columns:
                 existing["epoch"] = pd.to_numeric(existing["epoch"], errors='coerce').fillna(-1).astype(int)
            existing.set_index("epoch", inplace=True)
            new_df.set_index("epoch", inplace=True)
            
            existing.update(new_df)
             # Add new rows
            new_epochs = new_df.index.difference(existing.index)
            if not new_epochs.empty:
                existing = pd.concat([existing, new_df.loc[new_epochs]])
            
            final = existing.reset_index().sort_values("epoch")
            final.to_csv(path, index=False)
            print(f"    [SAVE] Updated {os.path.basename(path)}")
        except:
             new_df.sort_values("epoch").to_csv(path, index=False)
             print(f"    [SAVE] Overwrote {os.path.basename(path)}")
    else:
        new_df.sort_values("epoch").to_csv(path, index=False)
        print(f"    [SAVE] Created {os.path.basename(path)}")

if __name__ == "__main__":
    regenerate_all_metrics()
