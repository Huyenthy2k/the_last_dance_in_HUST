
import numpy as np
import pandas as pd
from agents.MAFIA.RL_controller.mafia_feature_processor import MAFIAFeatureProcessor
from agents.MAFIA.config import Config

def verify_vol_edge_case():
    print("=== Verifying Relative Volatility Edge Case: 'First Days of Training' ===")
    
    # 1. Setup minimal processor
    config = Config()
    config.mafia_T_w = 30
    config.mafia_DC_multipliers = [0.5, 1.0, 2.0]
    processor = MAFIAFeatureProcessor(config)
    
    # 2. Simulate a "Cold Start" Window (e.g. at the start of training)
    # T_w = 30. 
    # Let's say we have constant prices, then a small wiggle.
    
    # Edge Case A: Insufficient history (First few ticks of the window)
    # The processor computes rolling std over the INPUT window.
    # At index t < 10, does it crash? Does it produce 1.0?
    
    print("\n[Test A] Normal initialization behavior (t=0 to t=15)")
    prices = np.array([100.0 + i*0.1 for i in range(30)]) # Slow drift
    # Add noise
    np.random.seed(42)
    prices += np.random.normal(0, 0.5, 30)
    
    ochlv_data = np.zeros((1, 5, 30))
    ochlv_data[0, 1, :] = prices
    
    P_Mkt = processor.process_market_index_features(ochlv_data)
    vol_feat = P_Mkt[0, :, 16]
    
    print(f"{'Time':<5} | {'Vol_Rel':<10} | {'Status'}")
    print("-" * 30)
    for t in range(15):
        val = vol_feat[t]
        status = "Neutral (Warmup)" if abs(val - 1.0) < 1e-6 else "Active"
        print(f"{t:<5} | {val:.6f}   | {status}")
        
    # Edge Case B: Pure Constant Price (Zero Volatility)
    print("\n[Test B] Zero Volatility (Flatline)")
    prices_flat = np.full(30, 100.0)
    ochlv_data[0, 1, :] = prices_flat
    
    P_Mkt_flat = processor.process_market_index_features(ochlv_data)
    vol_feat_flat = P_Mkt_flat[0, :, 16]
    
    print(f"Mean Feature Value: {np.mean(vol_feat_flat)}")
    print(f"Max Feature Value: {np.max(vol_feat_flat)}")
    print(f"Any NaNs? {np.isnan(vol_feat_flat).any()}")
    
    # Edge Case C: Zero Pads (Start of dataset padding)
    print("\n[Test C] Zero Padding (Zeros at start)")
    prices_pad = np.array([0.0]*10 + [100.0 + i for i in range(20)])
    ochlv_data[0, 1, :] = prices_pad
    
    # Note: pct_change on 0 -> 0 is NaN. 0 -> 100 is inf?
    # fillna(0) handles NaNs. clipped to finite?
    # Let's see how robustness handles standard 0-padding.
    
    P_Mkt_pad = processor.process_market_index_features(ochlv_data)
    vol_feat_pad = P_Mkt_pad[0, :, 16]
    
    print("Zero-padded segment (first 10):", vol_feat_pad[:10])
    print("Transition segment (10-15):", vol_feat_pad[10:15])

if __name__ == "__main__":
    verify_vol_edge_case()
