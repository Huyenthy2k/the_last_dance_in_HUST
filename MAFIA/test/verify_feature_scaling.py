
import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from agents.MAFIA.RL_controller.mafia_feature_processor import MAFIAFeatureProcessor

class MockConfig:
    mafia_T_w = 30
    mafia_DC_thresholds = [0.005, 0.01, 0.02]
    mafia_M_tech = 8
    mafia_M_dc = 5
    mafia_M_mkt = 19
    topK = 5

def generate_random_ochlv(N, T_w, start_price=20000.0):
    # Generate random walk prices
    returns = np.random.normal(0.000, 0.02, (N, T_w)) # 2% daily vol
    price_paths = np.zeros((N, T_w))
    price_paths[:, 0] = start_price
    for t in range(1, T_w):
        price_paths[:, t] = price_paths[:, t-1] * (1 + returns[:, t])
    
    # OCHLV
    close = price_paths
    open_p = close * np.random.uniform(0.99, 1.01, (N, T_w))
    high = close * np.random.uniform(1.00, 1.03, (N, T_w))
    low = close * np.random.uniform(0.97, 1.00, (N, T_w))
    volume = np.random.uniform(100000, 5000000, (N, T_w))
    
    # Stack [open, close, high, low, volume]
    ochlv = np.stack([open_p, close, high, low, volume], axis=1)
    return ochlv

def check_range(name, data, min_val, max_val, strict=False):
    d_min = np.min(data)
    d_max = np.max(data)
    d_mean = np.mean(data)
    status = "✅"
    if strict:
        if d_min < min_val or d_max > max_val:
            status = "❌"
    else:
        # Soft check for unbounded but scaled features (like Z-score)
        if d_mean > max_val or d_mean < min_val: # Check if mean is exploded
            status = "❌ (Mean Exploded)"
            
    print(f"{status} {name:<20} | Range: [{d_min:8.4f}, {d_max:8.4f}] | Mean: {d_mean:8.4f} | Target: [{min_val}, {max_val}]")
    return status == "✅"

def main():
    config = MockConfig()
    processor = MAFIAFeatureProcessor(config)
    
    print("="*60)
    print("VERIFYING FEATURE NORMALIZATION")
    print("="*60)
    
    # 1. Verify Technical Features
    print("\n--- Technical Agent Features (N=2, T=30) ---")
    ochlv_stock = generate_random_ochlv(2, 60)[:, :, -30:] # Generate longer then cut to 30 to have history? No processor takes T_w only usually but SMA needs history
    # The processor takes (N, 5, T_w). It computes SMA on that window. 
    # If window=20, first 19 will be NaN/Filled.
    
    # Actually processor takes T_w data. 
    # SMA won't be fully valid until index 19.
    ochlv_stock = generate_random_ochlv(2, 30)
    
    p_tech = processor.process_technical_features(ochlv_stock)
    # p_tech shape: (2, 30, 8)
    
    # Check SMA Dist (Index 5)
    check_range("SMA_Dist (idx 5)", p_tech[:, :, 5], -0.2, 0.2)
    # Check RSI (Index 6)
    check_range("RSI_Norm (idx 6)", p_tech[:, :, 6], 0.0, 1.0, strict=True)
    # Check ATR Ratio (Index 7)
    check_range("ATR_Ratio (idx 7)", p_tech[:, :, 7], 0.0, 0.1) # ATR/Price usually < 10%
    
    # Check Volume Change (Index 4)
    # Volume can spike 5x or 10x, so change can be 4.0 or 9.0.
    check_range("Vol_Change (idx 4)", p_tech[:, :, 4], -1.0, 10.0)
    
    # 2. Verify DC Features
    print("\n--- DC Agent Features (Thresh=0.01) ---")
    p_dc = processor.process_dc_features(ochlv_stock, dc_threshold=0.01)
    # Duration is Index 2
    check_range("DC_Duration_Norm", p_dc[:, :, 2], 0.0, 1.5) # Can exceed 1.0 if > 30 days
    
    # 3. Verify Market Features
    print("\n--- Market Agent Features (N=1, T=30) ---")
    ochlv_mkt = generate_random_ochlv(1, 30, start_price=1200.0)
    p_mkt = processor.process_market_index_features(ochlv_mkt)
    
    # [5] SMA
    check_range("Mkt_SMA_Dist", p_mkt[:, :, 5], -0.2, 0.2)
    # [6] RSI
    check_range("Mkt_RSI_Norm", p_mkt[:, :, 6], 0.0, 1.0, strict=True)
    # [8] MACD Dist
    check_range("Mkt_MACD_Dist", p_mkt[:, :, 8], -0.1, 0.1)
    # [10] Stoch K
    check_range("Mkt_Stoch_K", p_mkt[:, :, 10], 0.0, 1.0, strict=True)
    # [12] ADX
    check_range("Mkt_ADX", p_mkt[:, :, 12], 0.0, 1.0, strict=True)
    # [13] OBV (Z-score)
    check_range("Mkt_OBV_Z", p_mkt[:, :, 13], -5.0, 5.0)
    # [14] MFI
    check_range("Mkt_MFI", p_mkt[:, :, 14], 0.0, 1.0, strict=True)
    # [15] CCI (Scaled / 100)
    check_range("Mkt_CCI_Scaled", p_mkt[:, :, 15], -3.0, 3.0)
    # [18] Regime
    check_range("Mkt_Regime_Dist", p_mkt[:, :, 18], -0.2, 0.2)

if __name__ == "__main__":
    main()
