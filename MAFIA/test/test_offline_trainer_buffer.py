
import unittest
import torch as th
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer, TrajectoryBatch
from RL_controller.mafia_observer import MAFIAObserver

class MockConfig:
    def __init__(self):
        self.mafia_trajectory_length = 5
        self.mafia_batch_size = 2
        self.mafia_T_w = 4
        self.mafia_D = 8
        self.router_context_window = 3
        self.mafia_top_k = 2
        
class TestTrainerBuffer(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.device = th.device("cpu")
        
        # Mock Observer and Model
        self.observer = MagicMock(spec=MAFIAObserver)
        self.model = MagicMock()
        self.model.D = self.config.mafia_D
        self.observer.mafia_model = self.model
        
        # Create trainer
        self.trainer = ObserverOfflineBatchTrainer(self.config, self.observer, device=self.device)

    def test_buffer_flow(self):
        """Verify buffer init, passing, and update in training loop."""
        
        B = self.config.mafia_batch_size
        T_m = self.config.mafia_trajectory_length
        N = 4 # stocks
        
        # Mock Data Tensors (minimal needed for collect_and_train_step)
        data_tensors = {
            "ochlv": th.randn(100, N, 5), # Full data
            # "market_ochlv": None
        }
        
        # Mock Batch
        start_indices = th.tensor([10, 20])
        batch = TrajectoryBatch(
            stock_ochlv=th.randn(B, T_m, N, 5), # Not really used since we index full data
            market_ochlv=None,
            price_returns=th.zeros(B, T_m + 5, N),
            market_returns=th.zeros(B, T_m),
            rebalance_mask=th.ones(B, T_m),
            direction_labels=th.zeros(B, T_m, dtype=th.long),
            risk_targets=th.zeros(B, T_m),
            start_indices=start_indices,
            dates=None,
            vol_shock_mask=th.zeros(B, T_m)
        )
        
        # Mock Model Forward Pass
        # We need to capture the `router_context_buffer` passed to it
        captured_buffers = []
        
        def mock_forward(*args, **kwargs):
            buf = kwargs.get('router_context_buffer')
            if buf is not None:
                # Clone to save state at that step
                captured_buffers.append(buf.clone())
            
            # Return dummy outputs matching signature
            # (market_vector, eta, scores_full, market_context, logs, idx, emb, scores)
            market_vector = th.zeros(B, N)
            eta = th.zeros(B)
            scores_full = th.zeros(B, N)
            
            # Return specific market_context to verify update: Step number as value
            step_idx = len(captured_buffers) 
            market_context = th.full((B, self.config.mafia_D), float(step_idx)) 
            
            logs = th.zeros(B, 3)
            idx = th.zeros(B, 2, dtype=th.long)
            emb = th.zeros(B, 2, self.config.mafia_D)
            scores = th.zeros(B, 2)
            
            return (market_vector, eta, scores_full, market_context, logs, idx, emb, scores)
            
        self.model.side_effect = mock_forward
        self.model.reset_temporal_state = MagicMock()
        
        # Run step
        try:
            self.trainer.collect_and_train_step(batch, data_tensors)
        except Exception as e:
            # Ignore backward error since we didn't return tensors with grad
            # pass
            print(f"Caught expected error (no grad): {e}")

        # Verification
        
        # 1. Check number of calls matches T_m
        self.assertEqual(len(captured_buffers), T_m)
        
        # 2. Check Initial Buffer (Step 0) -> Should be all zeros
        init_buffer = captured_buffers[0]
        self.assertEqual(init_buffer.shape, (B, self.config.router_context_window, self.config.mafia_D))
        self.assertTrue(th.allclose(init_buffer, th.zeros_like(init_buffer)))
        
        # 3. Check Update Logic (Step 1)
        # Step 0 returned market_context = 1.0
        # Step 1 buffer should have [0, 0, 1] (if W=3)
        # Let's inspect Step 1 buffer
        step1_buffer = captured_buffers[1]
        
        # Expectation:
        # Buffer was [0, 0, 0]
        # Push 1.0 -> [0, 0, 1]
        expected_last = th.ones(B, self.config.mafia_D) * 1.0
        self.assertTrue(th.allclose(step1_buffer[:, -1, :], expected_last), f"Step 1 buffer last element should be 1.0, got {step1_buffer[0, -1, 0]}")
        
        # 4. Check Step 2
        # Step 1 returned 2.0
        # Buffer should be [0, 1, 2]
        step2_buffer = captured_buffers[2]
        expected_mid = th.ones(B, self.config.mafia_D) * 1.0
        expected_last = th.ones(B, self.config.mafia_D) * 2.0
        self.assertTrue(th.allclose(step2_buffer[:, -2, :], expected_mid))
        self.assertTrue(th.allclose(step2_buffer[:, -1, :], expected_last))

        print("Buffer flow test passed!")

if __name__ == "__main__":
    unittest.main()
