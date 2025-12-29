
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_volatility():
    # Load Data
    df = pd.read_csv("agents/MAFIA/data/VNINDEX_1d_index.csv", parse_dates=["date"])
    df = df.sort_values("date")
    
    # Calculate Returns
    df["return"] = df["close"].pct_change()
    
    # Calculate Rolling Volatility (14-day std dev of returns)
    df["vol_14"] = df["return"].rolling(14).std()
    
    # Calculate 21-day Forward Return (MUST match config: direction_label_lookahead = 21)
    df["ret_21"] = df["close"].shift(-21) / df["close"] - 1
    
    # Define Periods
    train_mask = (df["date"] >= "2015-01-01") & (df["date"] <= "2016-12-31")
    valid_mask = (df["date"] >= "2017-01-01") & (df["date"] <= "2017-12-31")
    
    train_df = df[train_mask]
    valid_df = df[valid_mask]
    
    print("--- Volatility Regime Analysis ---")
    print(f"Train (2015-2016) Mean Vol (14d): {train_df['vol_14'].mean():.4f}")
    print(f"Valid (2017)      Mean Vol (14d): {valid_df['vol_14'].mean():.4f}")
    print(f"Ratio (Valid/Train): {valid_df['vol_14'].mean() / train_df['vol_14'].mean():.2f}")
    
    print("\n--- Return Distribution (21d Forward) ---")
    print(f"Train Mean Return: {train_df['ret_21'].mean():.4f}")
    print(f"Valid Mean Return: {valid_df['ret_21'].mean():.4f}")
    
    # Quantiles
    print("\n--- Quantiles of 21d Returns (Absolute) ---")
    # Implement Full Hybrid Labeling Logic to count classes
    valid_df = df[valid_mask].copy()
    train_df = df[train_mask].copy()
    
    def get_class_counts(sub_df, name):
        # Hybrid Threshold
        # delta_t = max(0.01, 1.5 * ATR_norm)
        # Assuming ATR_norm is already calculated in 'atr_norm' column
        # If not, we need to ensure it is.
        
        # Recalculate ATR just to be safe if not present (handled globally below)
        pass 

    # Ensure ATR is calculated globally first
    if "high" in df.columns and "low" in df.columns:
        df["tr"] = np.maximum(df["high"] - df["low"], 
                              np.maximum(abs(df["high"] - df["close"].shift(1)), 
                                         abs(df["low"] - df["close"].shift(1))))
        df["atr_21"] = df["tr"].rolling(21).mean()  # ATR period = 21 (match config)
        df["atr_norm"] = df["atr_21"] / df["close"]
    else:
        # Fallback if no high/low (unlikely for VNI)
        df["atr_norm"] = 0.01 # dummy
    
    # Apply logic
    # Need stop loss? In training code, intra_dd < -0.07 is Bear.
    # We need Future Return (ret_21) and IntraDD.
    # IntraDD is hard to calc vectorially quickly without rolling min of future window.
    # Let's approximate IntraDD with Low.
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)  # 21-day lookahead
    df["future_low_min"] = df["low"].rolling(window=indexer).min()
    df["intra_dd"] = (df["future_low_min"] / df["close"]) - 1.0
    
    # Vectorized Labeling (Synced with config.py §5.1.3)
    k_atr = 1.5       # config: direction_label_atr_multiplier
    delta_min = 0.015 # config: direction_label_delta_min (1.5%)
    stop_loss = -0.07
    
    df["delta_t"] = np.maximum(delta_min, k_atr * df["atr_norm"])
    
    # Label: 0=Bear, 1=Side, 2=Bull
    conditions = [
        (df["ret_21"] < -df["delta_t"]) | (df["intra_dd"] < stop_loss), # Bear
        (df["ret_21"] > df["delta_t"]) & (df["intra_dd"] >= stop_loss)  # Bull
    ]
    choices = [0, 2]
    df["label"] = np.select(conditions, choices, default=1)
    
    # Re-slice
    train_df = df[train_mask]
    valid_df = df[valid_mask]
    
    def print_dist(dframe, subset_name):
        counts = dframe["label"].value_counts(normalize=True).sort_index()
        print(f"\n--- {subset_name} Class Distribution ---")
        if 0 in counts: print(f"Bear (0): {counts[0]*100:.1f}%")
        if 1 in counts: print(f"Side (1): {counts[1]*100:.1f}%")
        if 2 in counts: print(f"Bull (2): {counts[2]*100:.1f}%")
        
        # Calculate Inverse Frequency Weights (Traditional)
        # W_c = N_total / (N_classes * N_c)
        if len(dframe) > 0:
            n_total = len(dframe)
            n_bear = len(dframe[dframe["label"]==0])
            n_side = len(dframe[dframe["label"]==1])
            n_bull = len(dframe[dframe["label"]==2])
            
            # Avoid div by zero
            w_bear = n_total / (3 * (n_bear if n_bear>0 else 1))
            w_side = n_total / (3 * (n_side if n_side>0 else 1))
            w_bull = n_total / (3 * (n_bull if n_bull>0 else 1))
            
            print(f"Inverse Freq Weights (Raw): [{w_bear:.2f}, {w_side:.2f}, {w_bull:.2f}]")
            
            # Focal Loss often works best with dampened weights, or even [1,1,1] if gamma is high.
            # But standard is Inverse Class Freq.

    print_dist(train_df, "TRAIN (Iter 0: 2015-2016)")
    print_dist(valid_df, "VALID (Iter 0: 2017)")
    
    # Global Analysis
    global_mask = (df["date"] >= "2015-01-01") & (df["date"] <= "2023-12-31")
    global_df = df[global_mask]
    print_dist(global_df, "GLOBAL DATASET (2015-2023)")

    print("\n--- Annualized Math ---")
    print("1% return / 21 days -> Annualized (x12): {:.2f}%".format(((1.01)**(250/21) - 1)*100))


if __name__ == "__main__":
    analyze_volatility()
