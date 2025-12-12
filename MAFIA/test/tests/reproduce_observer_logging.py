#!/usr/bin/env python3
"""
Reproduction script for Observer Training Logging.
Runs a few steps of Observer Offline Trainer with mock data to visualize logging output.
"""
import sys
import os
import torch as th
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Add agents/MAFIA to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# agents/MAFIA/tests -> agents/MAFIA
mafia_dir = os.path.dirname(current_dir)
if mafia_dir not in sys.path:
    sys.path.insert(0, mafia_dir)

print(f"DEBUG: Added {mafia_dir} to sys.path")

from config import Config
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer

def run_repro():
    print(">>> SETTING UP MOCK ENVIRONMENT...")
    
    # 1. Config
    config = Config()
    config.mafia_trajectory_length = 30  # Short trajectory for demo
    config.mafia_batch_size = 2          # Small batch
    config.mafia_T_w = 10
    config.mafia_horizon = 2
    config.mafia_sampling_strategy = "random_trajectory"
    config.mafia_top_k = 3
    config.rebalance_interval = 5
    config.log_trajectory_details = True # ENABLE LOGGING
    config.direction_label_lookahead = 5
    
    # 2. Mock Observer & Model
    mock_observer = MagicMock()
    mock_model = MagicMock()
    mock_observer.mafia_model = mock_model
    mock_observer.optimizer = MagicMock()
    mock_observer.lr_scheduler = MagicMock()
    
    # Mock Forward Pass Output
    B = config.mafia_batch_size
    N = 10 # 10 stocks
    K = config.mafia_top_k
    D = 16
    
    def forward_side_effect(*args, **kwargs):
        # Return dummy tensors with gradients enabled to pass backward check
        market_vector = th.randn(B, N, requires_grad=True)
        risk_eta = th.rand(B, requires_grad=True)
        market_scores_full = th.randn(B, N, requires_grad=True) 
        market_context = th.randn(B, D, requires_grad=True)
        direction_logits = th.tensor([[0.0, 10.0, 0.0]] * B, requires_grad=True) # Always Side for stability
        topk_indices = th.randint(0, N, (B, K))
        topk_embeddings = th.randn(B, K, D, requires_grad=True)
        topk_scores = th.rand(B, K, requires_grad=True)
        return (
            market_vector, risk_eta, market_scores_full, market_context, 
            direction_logits, topk_indices, topk_embeddings, topk_scores
        )
    
    mock_model.side_effect = forward_side_effect
    mock_model.D = D
    
    # 3. Trainer
    trainer = ObserverOfflineBatchTrainer(config, mock_observer)
    
    # 4. Mock Data Tensors
    T_total = 200
    dates = pd.date_range(start="2023-01-01", periods=T_total)
    data_tensors = {
        "ochlv": th.randn(T_total, N, 5),
        "returns": th.randn(T_total, N) * 0.01,
        "dates": dates,
        "T_total": T_total,
        "N": N,
        "stock_list": [f"Stock_{i}" for i in range(N)],
        "market_ochlv": th.randn(T_total, 1, 5),
        "market_returns": th.randn(T_total, 1) * 0.01
    }
    
    print(">>> STARTING TRAINING LOOP (1 STEP)...")
    print("-" * 50)
    
    # Sample Batch
    batch = trainer.sample_trajectory_batch(data_tensors)
    
    # Run Step
    trainer.collect_and_train_step(batch, data_tensors)
    
    print("-" * 50)
    print(">>> DONE.")

if __name__ == "__main__":
    run_repro()
