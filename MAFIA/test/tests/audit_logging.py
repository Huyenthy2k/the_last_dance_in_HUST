#!/usr/bin/env python3
"""
Audit Logging Script
Purpose: Deterministically verify that Observer logging captures changing data per timestep.
Focus:
- Hierarchy: Iteration > Epoch > Batch > Trajectory > Timestep
- Dynamic Data: Verify T=1 indexes != T=2 indexes.
"""
import sys
import os
import torch as th
import pandas as pd
from unittest.mock import MagicMock

# Add agents/MAFIA to path
current_dir = os.path.dirname(os.path.abspath(__file__))
mafia_dir = os.path.dirname(current_dir)
if mafia_dir not in sys.path:
    sys.path.insert(0, mafia_dir)

from config import Config
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer

# Force unbuffered output for test
sys.stdout.reconfigure(line_buffering=True)

def run_audit():
    print(">>> AUDIT START: Setting up determinisitc environment...")
    
    # 1. Config
    config = Config()
    config.mafia_trajectory_length = 3  # Short: 3 steps
    config.mafia_batch_size = 1         # 1 trajectory for clarity
    config.mafia_top_k = 3
    config.rebalance_interval = 1       # Rebalance EVERY step to force portfolio updates
    config.always_show_details = True   # Force detailed logs
    
    # 2. Mock Model with TIME-VARYING outputs
    mock_observer = MagicMock()
    mock_model = MagicMock()
    mock_observer.mafia_model = mock_model
    mock_observer.optimizer = MagicMock()
    
    # We want indices to change: [0,1,2] -> [3,4,5] -> [6,7,8]
    # We use a mutable counter in the side_effect
    step_counter = {"t": 0}
    
    def forward_side_effect(*args, **kwargs):
        t = step_counter["t"]
        B = 1
        N = 10
        K = 3
        D = 16
        
        # Changing Portfolio Indices
        start_idx = (t * K) % N
        indices = th.tensor([[ (start_idx + k) % N for k in range(K) ]], dtype=th.long)
        
        # Changing Direction (0, 1, 2)
        dir_logits = th.zeros(B, 3, requires_grad=True)
        # We can't easily modify tensor in place to requires_grad, create new
        vals = [0.0, 0.0, 0.0]
        vals[t % 3] = 10.0
        dir_logits = th.tensor([vals], requires_grad=True)
        
        step_counter["t"] += 1
        
        return (
            th.randn(B, N, requires_grad=True), # mkt vec
            th.rand(B, requires_grad=True),     # risk
            th.randn(B, N, requires_grad=True), # scores
            th.randn(B, D, requires_grad=True), # context
            dir_logits,                         # direction
            indices,                            # topk indices
            th.randn(B, K, D, requires_grad=True), # embeddings
            th.rand(B, K, requires_grad=True)   # probs
        )
        
    mock_model.side_effect = forward_side_effect
    mock_model.D = 16
    
    # 3. Trainer
    trainer = ObserverOfflineBatchTrainer(config, mock_observer)
    
    # 4. Data
    data_tensors = {
        "ochlv": th.randn(100, 10, 5),
        "returns": th.randn(100, 10),
        "dates": pd.date_range("2023-01-01", periods=100),
        "T_total": 100,
        "N": 10,
        "stock_list": [f"Stock_{i}" for i in range(10)], # Stock_0...Stock_9
        "market_ochlv": th.randn(100, 1, 5),
        "market_returns": th.randn(100, 1)
    }
    
    print("\n>>> RUNNING STEP (Expect: Stock_0..2, then Stock_3..5, then Stock_6..8)")
    print("-" * 60)
    
    batch = trainer.sample_trajectory_batch(data_tensors)
    # Force rebalance mask to ALL ONES to see every step
    batch.rebalance_mask = th.ones_like(batch.rebalance_mask) 
    
    try:
        trainer.collect_and_train_step(batch, data_tensors)
    except Exception as e:
        print(f"Trapped error (expected if mock output shapes wrong): {e}")
        import traceback
        traceback.print_exc()

    print("-" * 60)
    print(">>> AUDIT COMPLETE. Check logs above for portfolio variation.")

if __name__ == "__main__":
    run_audit()
