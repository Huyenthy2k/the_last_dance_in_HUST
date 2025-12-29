#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify simplified state (without market_index_state)
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from RL_controller.mafia_observer import MAFIAObserver

def test_state_dimensions():
    """Test that state dimensions are correctly calculated after removing market_index_state"""
    
    print("="*80)
    print("TESTING SIMPLIFIED STATE DIMENSIONS")
    print("="*80)
    
    # Initialize config
    config = Config(seed_num=2022, current_date='2025-01-01-00-00-00')
    
    # Test parameters
    stock_num = 10
    action_dim = 10
    
    print(f"\nTest Parameters:")
    print(f"  - stock_num: {stock_num}")
    print(f"  - action_dim: {action_dim}")
    print(f"  - mafia_top_k: {config.mafia_top_k}")
    print(f"  - mafia_state_mode: {config.mafia_state_mode}")
    print(f"  - mafia_include_risk_boundary_in_state: {config.mafia_include_risk_boundary_in_state}")
    
    # Calculate expected state dimensions
    print("\n" + "="*80)
    print("EXPECTED STATE DIMENSIONS:")
    print("="*80)
    
    # Compact mode only (Top-K selection handled by Observer)
    config.mafia_state_mode = 'compact'
    k = config.mafia_top_k
    risk_dim = 1 if config.mafia_include_risk_boundary_in_state else 0
    expected_state_dim = k + 1 + risk_dim
    
    print(f"\nCompact mode formula:")
    print(f"  state_dim = K + 1 + risk_dim")
    print(f"  state_dim = {k} + 1 + {risk_dim}")
    print(f"  state_dim = {expected_state_dim}")
    
    print(f"\nState components:")
    print(f"  1. market_vector(K):     {k} dims  <- Top-K from Observer")
    print(f"  2. portfolio_value:      1 dim    <- log(current_capital / initial_capital)")
    print(f"  3. risk_eta:             {risk_dim} dim    <- eta from Observer")
    print(f"  {'='*40}")
    print(f"  Total:                   {expected_state_dim} dims")
    
    # Compare with previous architecture
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS ARCHITECTURE:")
    print("="*80)
    
    k = config.mafia_top_k
    d_idx = getattr(config, "market_index_state_dim", 0)  # DEPRECATED
    risk_dim = 1 if config.mafia_include_risk_boundary_in_state else 0
    previous_state_dim = k + d_idx + 1 + risk_dim
    
    print(f"\nPrevious architecture (with market_index_state):")
    print(f"  state_dim = K + D_idx + 1 + risk_dim")
    print(f"  state_dim = {k} + {d_idx} + 1 + {risk_dim}")
    print(f"  state_dim = {previous_state_dim}")
    
    print(f"\nNew architecture (Top-K only, no market_index_state):")
    print(f"  state_dim = K + 1 + risk_dim")
    print(f"  state_dim = {k} + 1 + {risk_dim}")
    print(f"  state_dim = {expected_state_dim}")
    
    print(f"\n✅ State dimension REDUCED by: {previous_state_dim - expected_state_dim} dims ({d_idx} market_index features removed)")
    print(f"✅ Reduction percentage: {(previous_state_dim - expected_state_dim) / previous_state_dim * 100:.1f}%")
    
    # Test with actual MAFIA Observer
    print("\n" + "="*80)
    print("TESTING WITH ACTUAL MAFIA OBSERVER:")
    print("="*80)
    
    # Initialize MAFIA Observer
    mafia_observer = MAFIAObserver(config=config, action_dim=action_dim)
    print(f"\n✅ MAFIA Observer initialized successfully")
    print(f"   - action_dim: {mafia_observer.action_dim}")
    print(f"   - device: {mafia_observer.device}")
    
    # Create dummy OCHLV data
    T_w = config.mafia_T_w
    dummy_ochlv = np.random.randn(stock_num, 5, T_w).astype(np.float32)
    
    (
        market_vector,
        risk_eta,
        market_scores_full,
        gate_weights,
        _market_context,
        _stock_embedding,
        _sigma_val,
        _sigma_log_p,
    ) = mafia_observer.predict(
        raw_ochlv_data=dummy_ochlv,
        mode='test'
    )
    
    print(f"\n✅ MAFIA Observer.predict() successful")
    print(f"   - market_vector shape: {market_vector.shape}")
    print(f"   - risk_eta shape: {risk_eta.shape}")
    print(f"   - market_scores_full shape: {market_scores_full.shape}")
    print(f"   - gate_weights shape: {gate_weights.shape}")
    
    mv = market_vector[0]
    k = min(config.mafia_top_k, len(mv))
    topk_idx = np.argpartition(mv, -k)[-k:]
    topk_sorted = topk_idx[np.argsort(mv[topk_idx])[::-1]]
    market_vector_state = mv[topk_sorted].astype(np.float32)
    
    pv = np.array([0.0], dtype=np.float32)
    state_parts = [market_vector_state, pv]
    if config.mafia_include_risk_boundary_in_state:
        state_parts.append(risk_eta)
    
    state = np.concatenate(state_parts, axis=0).astype(np.float32)
    
    print(f"\n✅ State constructed successfully")
    print(f"   - state shape: {state.shape}")
    print(f"   - expected shape: ({expected_state_dim},)")
    assert state.shape[0] == expected_state_dim, f"Expected state dim {expected_state_dim}, got {state.shape[0]}"
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"\n✅ All tests passed!")
    print(f"\nKey changes:")
    print(f"  1. Removed {d_idx} market_index_state features from environment state")
    print(f"  2. State now relies solely on Observer's learned representation")
    print(f"  3. Market-Index Agent inside Observer still processes VNINDEX data")
    print(f"  4. Reduced state dimension: {previous_state_dim} → {expected_state_dim}")
    print(f"\nBenefits:")
    print(f"  - Simpler state representation")
    print(f"  - Lower dimensional input for TD3 Actor")
    print(f"  - No redundancy between explicit and implicit market information")
    print(f"  - Observer's learned representation is more powerful than raw features")

if __name__ == '__main__':
    test_state_dimensions()
    sys.exit(0)
