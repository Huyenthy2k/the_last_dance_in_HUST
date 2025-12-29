import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch as th
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer, TrajectoryBatch
from config import Config

class MockMAFIAModel(th.nn.Module):
    def __init__(self, B, N, K, D):
        super().__init__()
        self.B = B
        self.N = N
        self.K = K
        self.D = D
        # Mock predictable direction logits
        self.step_counter = 0
        # Dummy layers to ensure grad_fn
        self.dummy_layer = th.nn.Linear(1, 1)

    def forward(self, ochlv_data, market_index_ochlv_data=None, force_topk_indices=None, **kwargs):
        # Fake input for dummy layer to get gradient history
        dummy_in = th.ones(self.B, 1, requires_grad=True)
        dummy_out = self.dummy_layer(dummy_in) # (B, 1)
        
        # Outputs connected to graph
        market_vector = th.randn(self.B, self.N) + dummy_out.mean()
        risk_eta = th.ones(self.B) + dummy_out.mean()
        market_scores_full = th.randn(self.B, self.N) + dummy_out.mean()
        market_context = th.randn(self.B, self.D) + dummy_out.mean()
        
        # Controlled Direction Logits
        # Step 0-5: Side (Index 1) -> logits = [0, 10, 0]
        # Step 6: Bear (Index 0) -> logits = [10, 0, 0] (Reversal!)
        # Step 7+: Side (Index 1)
        direction_logits = th.zeros(self.B, 3)
        if self.step_counter == 6:
            direction_logits[:, 0] = 10.0 # Bear
        else:
            direction_logits[:, 1] = 10.0 # Side
            
        self.step_counter += 1

        topk_indices = th.randint(0, self.N, (self.B, self.K))
        topk_embeddings = th.randn(self.B, self.K, self.D)
        topk_scores = th.randn(self.B, self.K)

        return (
            market_vector,
            risk_eta,
            market_scores_full,
            market_context,
            direction_logits,
            topk_indices,
            topk_embeddings,
            topk_scores,
        )
    
    def reset_temporal_state(self):
        pass

class TestObserverRegimeShift(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.mafia_trajectory_length = 10
        self.config.mafia_batch_size = 2
        self.config.mafia_T_w = 5
        self.config.mafia_top_k = 3
        # Disable regular rebalance (infinite interval) to isolate triggers
        self.config.mafia_topk_rebalance_interval = 999 
        
        # Init components
        self.observer = MagicMock()
        self.observer.mafia_model = MockMAFIAModel(B=2, N=10, K=3, D=4)
        self.observer.optimizer = MagicMock()
        self.observer.lr_scheduler = MagicMock()
        
        self.trainer = ObserverOfflineBatchTrainer(self.config, self.observer)
        self.trainer.full_ochlv = th.randn(100, 10, 5) # Mock data

    def test_regime_shift_triggers(self):
        B = 2
        T_m = 10
        N = 10
        
        # 1. Prepare Batch
        # Rebalance Mask: Only Step 0 is 1 (per regular schedule constraint)
        rebalance_mask = th.zeros(B, T_m)
        rebalance_mask[:, 0] = 1.0
        
        # Vol Shock Mask: Trigger at Step 8
        vol_shock_mask = th.zeros(B, T_m)
        vol_shock_mask[:, 8] = 1.0
        
        batch = TrajectoryBatch(
            stock_ochlv=th.randn(B, T_m, N, 5),
            market_ochlv=None,
            price_returns=th.randn(B, T_m, N),
            market_returns=th.randn(B, T_m),
            rebalance_mask=rebalance_mask,
            direction_labels=th.zeros(B, T_m, dtype=th.long),
            risk_targets=th.zeros(B, T_m),
            start_indices=th.zeros(B, dtype=th.long), # Point to 0
            dates=np.zeros((B, T_m)),
            vol_shock_mask=vol_shock_mask
        )
        
        # Mock data tensors needed for full_ochlv lookup
        data_tensors = {
            "ochlv": th.randn(50, N, 5),
            "T_total": 50
        }
        
        # 2. Run Step
        metrics = self.trainer.collect_and_train_step(batch, data_tensors)
        
        # 3. Assertions
        # Expected Rebalances:
        # Step 0: Schedule (Mask=1) -> 1
        # Step 6: Side -> Bear (Reversal) -> 1
        # Step 7: Bear -> Side (Reversal Back) -> 1  <-- Forgot this one!
        # Step 8: Vol Shock (Mask=1) -> 1
        # Total = 4. Ratio = 0.4
        
        print(f"Rebalance Ratio: {metrics['rebalance_ratio']}")
        
        # Allow small float error
        self.assertAlmostEqual(metrics['rebalance_ratio'], 0.4, places=5, 
                               msg="Rebalance ratio should reflect Schedule(0) + Dir(6) + Return(7) + Vol(8)")

if __name__ == '__main__':
    unittest.main()
