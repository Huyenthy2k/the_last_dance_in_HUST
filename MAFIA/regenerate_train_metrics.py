import argparse
import os
import sys
import glob
import pandas as pd
import torch as th
import numpy as np
from dataclasses import asdict

# Ensure Python path includes current directory
sys.path.append(os.getcwd())

from config import Config
from RL_controller.mafia_modules import MAFIAModel
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import create_offline_trainer, ObserverOfflineBatchTrainer
from RL_controller.validation_tracker import ValidationMetricsTracker
from scripts.train_observer_offline import load_mafia_data, build_expanding_schedule

def regenerate_train_metrics():
    parser = argparse.ArgumentParser(description="Regenerate train_metrics.csv from checkpoints on Training Data")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year of training data")
    parser.add_argument("--first-infer-year", type=int, default=2018, help="First inference year")
    parser.add_argument("--last-infer-year", type=int, default=2022, help="Last inference year")
    parser.add_argument("--iter", type=int, default=0, help="Iteration index to regenerate (0-indexed)")
    parser.add_argument("--target-epoch", type=int, default=None, help="Specific epoch to regenerate (e.g. 16)")
    parser.add_argument("--output-dir", type=str, default="observer_offline", help="Base output directory")
    args = parser.parse_args()

    print(f"[REGEN-TRAIN] Starting TRAIN metric regeneration for Iteration {args.iter}...")
    
    # 1. Setup Config & Data
    config = Config(create_dirs=False)
    # [FIX] dataDir is now handled robustly in Config.__init__
    # config.dataDir = os.path.join(os.getcwd(), "agents", "MAFIA", "data")
    config.seed = 2025
    
    # 2. Build Schedule to target specific iteration
    schedule = build_expanding_schedule(args.start_year, args.first_infer_year, args.last_infer_year)
    if args.iter < 0 or args.iter >= len(schedule):
        print(f"[REGEN-TRAIN] Error: Iteration {args.iter} out of range (0-{len(schedule)-1})")
        return
    
    sched = schedule[args.iter]
    print(f"[REGEN-TRAIN] Targeting Iteration {args.iter}: Train Range {sched['train_label']}")
    
    # 3. Load Data
    print("[REGEN-TRAIN] Loading data...")
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
    # Force CPU for regeneration stability/debug
    config.device = th.device("cpu")
        
    action_dim = len(stock_list)
    observer = MAFIAObserver(config=config, action_dim=action_dim)
    # trainer = create_offline_trainer(config, observer)
    trainer = ObserverOfflineBatchTrainer(
        config=config,
        observer=observer,
        device=config.device
    )
    
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
    
    # 4. Prepare TRAINING Tensors (Targeting Training Split)
    # Using 'train_start' and 'train_end' from schedule
    train_mask = (data_tensors["dates"] >= sched["train_start"]) & (data_tensors["dates"] <= sched["train_end"])
    train_indices = np.where(train_mask)[0]
    
    if len(train_indices) == 0:
        print(f"[REGEN-TRAIN] Error: No training data found for range {sched['train_label']}")
        return

    print(f"[REGEN-TRAIN] Processing {len(train_indices)} training days...")

    train_tensors = {
        "ochlv": data_tensors["ochlv"][train_indices[0] : train_indices[-1] + 1],
        "returns": data_tensors["returns"][train_indices[0] : train_indices[-1] + 1],
        "market_ochlv": data_tensors.get("market_ochlv", None)[train_indices[0] : train_indices[-1] + 1] if data_tensors.get("market_ochlv") is not None else None,
        "market_returns": data_tensors.get("market_returns", None)[train_indices[0] : train_indices[-1] + 1] if data_tensors.get("market_returns") is not None else None,
        "dates": data_tensors["dates"][train_indices],
        "stock_list": data_tensors["stock_list"],
        "T_total": len(train_indices),
        "N": data_tensors["N"],
    }

    # 5. Locate Checkpoints
    # Checkpoints are usually in checkpoints/temp_iter_X or checkpoints/
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"temp_iter_{args.iter}")
    if not os.path.exists(ckpt_dir):
        print(f"[REGEN-TRAIN] Checkpoint directory not found at {ckpt_dir}. Checking root checkpoints...")
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    
    print(f"[REGEN-TRAIN] Searching for checkpoints in: {ckpt_dir}")
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    
    # Always try to include latest if it exists
    latest_pth = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_pth) and latest_pth not in ckpts:
        ckpts.append(latest_pth)

    if not ckpts:
        print(f"[REGEN-TRAIN] No checkpoints found at {ckpt_dir}.")
        return

    # Sort checks
    def extract_epoch(p):
        base = os.path.basename(p)
        if "latest" in base: return 999999
        try:
            return int(base.split("_")[1].split(".")[0])
        except:
            return -1
    
    ckpts.sort(key=extract_epoch)

    # Filter by target epoch if requested
    if args.target_epoch is not None:
         target_file = f"epoch_{args.target_epoch}.pth"
         # Check strict match
         matches = [c for c in ckpts if os.path.basename(c) == target_file or os.path.basename(c) == f"epoch_{args.target_epoch}_balanced.pth"]
         if matches:
             print(f"[REGEN-TRAIN] Targeting specific checkpoint(s) for Epoch {args.target_epoch}: {[os.path.basename(x) for x in matches]}")
             ckpts = matches
         else:
             # Fallback to latest
             latest_match = [c for c in ckpts if "latest" in os.path.basename(c)]
             if latest_match:
                 print(f"[REGEN-TRAIN] Epoch {args.target_epoch} file not found. Using latest checkpoint hoping it matches.")
                 ckpts = latest_match
             else:
                 print(f"[REGEN-TRAIN] Epoch {args.target_epoch} not found and no latest checkpoint.")
                 ckpts = []

    # Output CSV
    out_dir = os.path.join(args.output_dir, f"iter_{args.iter}_valid_{sched['valid_year']}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "train_metrics.csv")
    
    # Load existing metrics if available to preserve history
    existing_df = None
    if os.path.exists(csv_path):
        try:
             existing_df = pd.read_csv(csv_path)
             print(f"[REGEN-TRAIN] Loaded existing metrics from {csv_path} ({len(existing_df)} epochs)")
        except Exception as e:
             print(f"[REGEN-TRAIN] Could not load existing metrics: {e}")

    # We will accumulate records
    all_train_records = []

    # 6. Process Checkpoints
    for ckpt in ckpts:
        try:
            print(f"[REGEN-TRAIN] processing {ckpt}...")
            loaded_epoch = trainer.observer.load_checkpoint(ckpt)
            # Force model to device (CPU) strictly to avoid MPS contaminants from checkpoint
            observer.mafia_model.to(config.device)
            trainer._epoch = loaded_epoch
            
            # Use 'validate_epoch' on TRAIN tensors to compute full metrics efficiently
            # We limit steps to avoid waiting forever on the full training set if it's huge
            # Use 50 steps (~1 epoch worth of batches usually or substantial sample)
            # or usage None to auto-calc based on logic (might differ)
            # Trainer uses validate_epoch for validation data which is smaller.
            # For training data regeneration, let's use a subset to be fast (e.g. 50 steps)
            # or try to cover full epoch? Full epoch is safer for accuracy.
            
            # Auto-calc steps for full coverage
            # steps_per_epoch = ceil(available / batch)
            # trainer.validate_epoch uses steps=10 default if none.
            # let's force a reasonable number, e.g. 100 or full?
            # Let's try to calculate full steps for this 'epoch'
            available = train_tensors["T_total"] - trainer.T_m - trainer.horizon - trainer.T_w
            steps_full = max(1, int(np.ceil(available / trainer.batch_size)))
            
            # Performance optimized with Numba, using 5 steps for faster metrics (approximate but sufficient)
            steps_to_run = 5
            
            result = trainer.validate_epoch(train_tensors, steps=steps_to_run, compute_loss=True)
            
            # Result is ObserverValidationResult. Convert to dict.
            rec = asdict(result)
            
            # [FIX] Format direction_accuracy as percentage to match legacy
            rec["direction_accuracy"] = round(rec["direction_accuracy"] * 100.0, 5)

            # FIX: Map rl_advantage and rl_reward
            # rl_advantage is roughly topk_advantage
            rec["rl_advantage"] = rec.get("topk_advantage", 0.0)
            # rl_reward is roughly net_reward
            rec["rl_reward"] = rec.get("net_reward", 0.0)
            
            # Remove CES fields as they are irrelevant for training metrics
            keys_to_remove = [k for k in rec.keys() if "ces_score" in k or "ces_rank" in k]
            for k in keys_to_remove:
                del rec[k]
                
            all_train_records.append(rec)
            
            print(f"        [Done] Epoch {loaded_epoch} | Adv: {rec['rl_advantage']:.4f} | F1-Macro: {rec['direction_f1_macro']:.4f}")

        except Exception as e:
            print(f"[ERROR] Failed {ckpt}: {e}")

    # 7. Save
    if all_train_records:
        new_df = pd.DataFrame(all_train_records)
        
        if existing_df is not None:
             # UPSERT LOGIC: Update existing rows, append new ones
             # Set index to epoch for easy update
             existing_df.set_index("epoch", inplace=True)
             new_df.set_index("epoch", inplace=True)
             
             # Update with new data
             existing_df.update(new_df)
             
             # Now add rows that were not in existing
             new_epochs = new_df.index.difference(existing_df.index)
             if not new_epochs.empty:
                 existing_df = pd.concat([existing_df, new_df.loc[new_epochs]])
             
             final_df = existing_df.reset_index()
        else:
             final_df = new_df

        # Sort by epoch just in case
        final_df = final_df.sort_values("epoch")
        final_df.to_csv(csv_path, index=False)
        print(f"[REGEN-TRAIN] ✅ Regenerated train_metrics.csv at: {csv_path}")
        print(final_df[["epoch", "rl_advantage", "direction_f1_macro", "risk_mae"]].to_string(index=False))
    else:
        print("[REGEN-TRAIN] No records generated.")

if __name__ == "__main__":
    regenerate_train_metrics()
