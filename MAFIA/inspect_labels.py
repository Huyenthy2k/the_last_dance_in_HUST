import torch as th
import pandas as pd
import numpy as np
import os
import sys

# Add path
sys.path.append(os.getcwd())

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer

def inspect_labels():
    print("Inspecting Label Distribution for Validation Set (2017)...")
    
    # 1. Config
    config = Config(create_dirs=False)
    # Ensure correct params
    config.direction_label_lookahead = 14
    config.direction_label_atr_period = 14
    config.direction_label_atr_multiplier = 2.0
    config.direction_label_delta_min = 0.02
    
    # 2. Load Data
    data_path = "data/stock_data_dynamic143.csv" 
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Validation Range 2017
    start_date = pd.Timestamp("2017-07-01")
    end_date = pd.Timestamp("2017-12-31")
    
    stock_list = df["stock"].unique().tolist()
    action_dim = len(stock_list)
    print(f"Action Dim (N): {action_dim}")

    # 3. Setup Trainer (Minimal)
    observer = MAFIAObserver(config, action_dim=action_dim)
    trainer = ObserverOfflineBatchTrainer(config, observer, device=th.device("cpu"))
    
    # Mock Market Data (using simple mean if index not available)
    # But wait, trainer.prepare_data_tensors handles market data logic.
    # We will simulate "No Market Index" for simplicity, or we can try to find VNINDEX.
    # The default data usually includes VNINDEX (stock="VNINDEX")?
    if "VNINDEX" in stock_list:
        print("Found VNINDEX in data.")
        market_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & (df["stock"] == "VNINDEX")
        market_df = df[market_mask]
    else:
        print("VNINDEX not found, Trainer will use Mean of Stocks.")
        market_df = None
        
    # 4. Prepare Tensors
    print("Preparing tensors...")
    # Passing None for market_data relies on the optional logic in prepare_data_tensors
    # But wait, prepare_data_tensors expects market_data DF if we have it?
    # No, signature is: prepare_data_tensors(self, data, stock_list, start_date, end_date, market_data=None)
    
    # If we pass all data in `data`, and market_data is None, it uses stock mean?
    # Actually prepare_data_tensors logic:
    # It creates ochlv_array.
    # Then it checks `self.observer.feature_processor.process_market_index_features`.
    # And `_build_class_indices` needs `data_tensors["market_ochlv"]`.
    # If `market_ochlv` is None, it returns None.
    
    # So we MUST provide `market_data` if available, or hope `prepare_data_tensors` creates it from mean.
    # Let's check `prepare_data_tensors` again? 
    # It does NOT create `market_ochlv` from mean automatically for `_build_class_indices`.
    # `_build_class_indices` returns None if `market_ochlv` is None.
    
    # BUT `sample_trajectory_batch` HAS logic to fallback to mean if `market_ochlv` is None.
    # The labels printed in `sample_trajectory_batch` (batch_direction_labels) use fallback logic.
    # "if use_market_index: ... else: Fallback: Mean of stocks" (Line 1039).
    
    # So we can just call `sample_trajectory_batch`?
    # But we need `train_tensors`.
    
    train_tensors = trainer.prepare_data_tensors(
        df, stock_list, start_date, end_date, market_data=None
    )
    
    # Add dummy market tensor if needed required by sample?
    # sample_trajectory_batch checks `market_ochlv` inside dict.
    
    # Let's hijack `_build_class_indices` if possible, but it requires market tensor.
    # Instead, we will replicate the counting logic using the Fallback Mean logic.
    
    # Extract Mean OCHLV
    ochlv = train_tensors["ochlv"] # (T, N, 5)
    # Mean across stocks
    market_proxy = ochlv.mean(dim=1) # (T, 5)
    
    T_total = market_proxy.shape[0]
    dates = train_tensors["dates"]
    
    print(f"Validation T_total: {T_total}")
    
    # Simulating the loop logic for labeling
    atr_period = 14
    k_atr = 2.0
    delta_min = 0.02
    stop_loss = -0.07
    lookahead = 14
    
    labels = []
    
    # Quick ATR calc
    high = market_proxy[:, 2].numpy()
    low = market_proxy[:, 3].numpy()
    close = market_proxy[:, 1].numpy()
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[tr[0]], tr])
    atr = np.convolve(tr, np.ones(atr_period) / atr_period, mode="same")
    
    counts = {0: 0, 1: 0, 2: 0}
    
    print("\n--- Generating Labels ---")
    for t in range(atr_period, T_total - lookahead):
        p_t = close[t]
        current_atr = atr[t]
        
        # Dynamic Threshold
        delta_t = max(delta_min, k_atr * (current_atr / (p_t + 1e-8)))
        
        # Future
        future_idx = t + lookahead
        p_fut = close[future_idx]
        
        r_fut = (p_fut - p_t) / p_t if p_t > 0 else 0
        
        # Intra DD
        future_lows = low[t+1 : future_idx+1]
        min_low = np.min(future_lows)
        intra_dd = (min_low / p_t) - 1.0 if p_t > 0 else 0
        
        label = 1 # Side
        if r_fut < -delta_t or intra_dd < stop_loss:
            label = 0 # Bear
        elif r_fut > delta_t and intra_dd >= stop_loss:
            label = 2 # Bull
            
        counts[label] += 1
        labels.append(label)
        
    print("\n--- Label Distribution ---")
    total = sum(counts.values())
    print(f"Total Samples: {total}")
    print(f"Bear (0): {counts[0]} ({counts[0]/total*100:.2f}%)")
    print(f"Side (1): {counts[1]} ({counts[1]/total*100:.2f}%)")
    print(f"Bull (2): {counts[2]} ({counts[2]/total*100:.2f}%)")
    
    if counts[0] == 0:
        print("⚠️ WARNING: No Bear labels found! F1 Bear will be 0.0.")
    if counts[2] == 0:
        print("⚠️ WARNING: No Bull labels found! F1 Bull will be 0.0.")

if __name__ == "__main__":
    inspect_labels()
