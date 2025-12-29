#!/usr/bin/env python3
"""
Recover train_metrics.csv from TensorBoard logs
Target: observer_offline_1/phase1_macro/iter_0_valid_2017
Source: observer_offline_1/phase1_macro/tb_logs
"""
import os
import sys
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def recover_from_tb():
    # Paths
    base_dir = "/Users/nguyensiry/Documents/the_last_dance/observer_offline_1/phase1_macro"
    log_dir = os.path.join(base_dir, "tb_logs")
    target_csv_path = os.path.join(base_dir, "iter_0_valid_2017/train_metrics.csv")
    
    print(f"[RECOVER] Scanning TensorBoard logs in: {log_dir}")
    
    if not os.path.exists(log_dir):
        print(f"[ERROR] Log directory not found: {log_dir}")
        return

    # Find event file
    event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        print("[ERROR] No event files found.")
        return
        
    event_path = os.path.join(log_dir, event_files[0]) # Assuming one main run or grabbing first
    print(f"[RECOVER] Reading events from: {event_path}")
    
    # Load events
    ea = EventAccumulator(event_path)
    ea.Reload()
    
    # Tags to extract
    # Map TB tag to CSV column
    tag_map = {
        "Train/Epoch/Loss_Total": "loss_total",
        "Train/Epoch/Loss_Risk": "loss_risk",
        "Train/Epoch/Loss_Dir": "loss_dir",
        "Train/Epoch/Dir_F1": "direction_f1_macro",
        "Train/Epoch/Risk_MSE": "risk_mse",
        "Train/Epoch/Sharpe": "topk_sharpe_ratio", 
        
        # New Detailed Tags
        "Train/Epoch/Dir_Acc": "direction_accuracy",
        "Train/Epoch/Dir_F1_Bear": "direction_f1_bear",
        "Train/Epoch/Dir_F1_Side": "direction_f1_side",
        "Train/Epoch/Dir_F1_Bull": "direction_f1_bull",
        "Train/Epoch/Risk_MAE": "risk_mae",
        "Train/Epoch/Risk_Corr": "risk_correlation",
        # Valid metrics might be useful if train allows it, but user wants train_metrics
        # Note: Direction/Accuracy etc. seem dependent on whether detailed logging was on.
        # Based on logs, we have summary metrics.
    }
    
    # Check available tags
    avail_tags = ea.Tags()['scalars']
    print(f"[RECOVER] Available tags: {avail_tags}")
    
    data = {}
    
    # Extract data
    for tb_tag, csv_col in tag_map.items():
        if tb_tag in avail_tags:
            events = ea.Scalars(tb_tag)
            # events is list of ScalarEvent(wall_time, step, value)
            for e in events:
                step = e.step
                val = e.value
                
                # In TB, 'step' corresponds to epoch if logged per epoch
                # BUT train_observer_offline logs per EPOCH using global_step=epoch
                # Let's verify assumption
                epoch = step 
                
                if epoch not in data:
                    data[epoch] = {"epoch": epoch, "training_mode": "MACRO_ONLY"}
                
                data[epoch][csv_col] = val
                
    # Flatten to list
    rows = list(data.values())
    if not rows:
        print("[RECOVER] No matching data found in logs.")
        return
        
    df = pd.DataFrame(rows)
    df = df.sort_values("epoch")
    
    # Filter only relevant epochs (0 to 22)
    df = df[df["epoch"] <= 22]
    
    # Check if we need to merge with existing manual recoveries
    if os.path.exists(target_csv_path):
        print(f"[RECOVER] Merging with existing CSV...")
        df_old = pd.read_csv(target_csv_path)
        # Prefer TensorBoard data as it's the original run
        # Exclude epochs present in TB from old DF
        tb_epochs = df["epoch"].unique()
        df_old_filtered = df_old[~df_old["epoch"].isin(tb_epochs)]
        df_final = pd.concat([df_old_filtered, df], ignore_index=True)
    else:
        df_final = df

    df_final = df_final.sort_values("epoch")
    
    # Reorder columns to match standard format if possible
    preferred_cols = [
        "epoch","training_mode","phase_score","direction_score","risk_score",
        "loss_total","loss_risk","loss_dir","loss_pg", 
        "direction_accuracy",
        "direction_f1_bear","direction_f1_side","direction_f1_bull","direction_f1_macro",
        "risk_mse","risk_mae","risk_correlation",
        # "topk_sharpe_ratio", "topk_mean_return", "topk_volatility" # Removed for MACRO phase as per user request
    ]
    
    # Add derivation logic for missing fields
    for idx, row in df_final.iterrows():
        # Derive risk_score if missing
        if pd.isna(row.get("risk_score")) and not pd.isna(row.get("risk_mse")):
             df_final.at[idx, "risk_score"] = 1.0 - min(row["risk_mse"], 1.0)
        
        # Derive direction_score if missing
        if pd.isna(row.get("direction_score")) and not pd.isna(row.get("direction_f1_macro")):
             df_final.at[idx, "direction_score"] = row["direction_f1_macro"]
             
        # Derive phase_score for MACRO_ONLY if missing (0.5 dir + 0.5 risk)
        if pd.isna(row.get("phase_score")) and row.get("training_mode") == "MACRO_ONLY":
             d_score = df_final.at[idx, "direction_score"]
             r_score = df_final.at[idx, "risk_score"]
             if not pd.isna(d_score) and not pd.isna(r_score):
                  df_final.at[idx, "phase_score"] = 0.5 * d_score + 0.5 * r_score

    # Ensure all columns exist
    for c in preferred_cols:
        if c not in df_final.columns:
            df_final[c] = float('nan') # Fill missing with NaN to signal "unknown" but keep structure
            
            # Special defaults for essential but missing scalar fields to prevent crashes
            if c in ["direction_f1_bear", "direction_f1_side", "direction_f1_bull"]:
                  # If macro F1 exists, maybe fill them with macro average as placeholder? 
                  # Better leave NaN so user knows it's not real data.
                  pass 

    # Reorder
    df_final = df_final[preferred_cols]

    # Save
    df_final.to_csv(target_csv_path, index=False, float_format='%.5f')
    print(f"[RECOVER] Successfully recovered {len(df_final)} epochs to: {target_csv_path}")
    print("Sample Data:")
    print(df_final[["epoch", "loss_total", "phase_score", "direction_f1_macro"]].head())

if __name__ == "__main__":
    recover_from_tb()
