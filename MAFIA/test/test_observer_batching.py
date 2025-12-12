
import unittest
import torch as th
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

# Add agents/MAFIA to sys.path
# Assuming tests/ is at root level and agents/ is at root level
# We need to find the absolute path to agents/MAFIA
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # Documents/the_last_dance
mafia_dir = os.path.join(root_dir, "agents", "MAFIA")
sys.path.append(mafia_dir)

from config import Config
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer, TrajectoryBatch

class TestObserverBatching(unittest.TestCase):
    def setUp(self):
        # Setup Config
        self.config = Config()
        self.config.mafia_trajectory_length = 10
        self.config.mafia_batch_size = 4
        self.config.mafia_T_w = 5
        self.config.mafia_horizon = 2
        self.config.mafia_sampling_strategy = "random_trajectory"
        self.config.mafia_top_k = 2
        
        # Setup Mock Observer and Model
        self.mock_observer = MagicMock()
        self.mock_model = MagicMock()
        self.mock_observer.mafia_model = self.mock_model
        
        # Setup Trainer
        self.trainer = ObserverOfflineBatchTrainer(self.config, self.mock_observer)
        
        # Create Dummy Data Tensors
        self.T_total = 100
        self.N = 5
        self.data_tensors = {
            "ochlv": th.randn(self.T_total, self.N, 5),
            "returns": th.randn(self.T_total, self.N),
            "dates": pd.date_range(start="2023-01-01", periods=self.T_total),
            "T_total": self.T_total,
            "N": self.N,
            "stock_list": [f"Stock_{i}" for i in range(self.N)],
            "market_ochlv": th.randn(self.T_total, 1, 5),
            "market_returns": th.randn(self.T_total, 1)
        }

    def test_trajectory_sampling_shapes(self):
        """Test 1: Verify output shapes of sample_trajectory_batch."""
        batch = self.trainer.sample_trajectory_batch(self.data_tensors)
        
        B = self.config.mafia_batch_size
        T_m = self.config.mafia_trajectory_length
        N = self.N
        
        self.assertIsInstance(batch, TrajectoryBatch)
        self.assertEqual(batch.stock_ochlv.shape, (B, T_m, N, 5))
        self.assertEqual(batch.price_returns.shape, (B, T_m, N))
        self.assertEqual(batch.start_indices.shape, (B,))
        self.assertEqual(len(batch.dates), B)
        self.assertEqual(len(batch.dates[0]), T_m)

    def test_start_index_randomness(self):
        """Test 1b: Verify start indices are random (or at least different)."""
        # Run multiple times to ensure variance
        starts = []
        for _ in range(5):
            batch = self.trainer.sample_trajectory_batch(self.data_tensors)
            starts.append(batch.start_indices.cpu().numpy())
            
        # Check that we don't always get the same start indices
        # (It's theoretically possible but highly unlikely with T=100)
        # We check variance across different calls
        all_starts = np.concatenate(starts)
        self.assertGreater(np.std(all_starts), 0, "Start indices should vary")

    def test_time_consistency(self):
        """Test 2: Verify time-consistency within trajectories."""
        batch = self.trainer.sample_trajectory_batch(self.data_tensors)
        
        dates = batch.dates # List of arrays (B, T_m)
        
        for b in range(self.config.mafia_batch_size):
            traj_dates = dates[b]
            # Verify sequential days
            for t in range(1, len(traj_dates)):
                # traj_dates are numpy datetime64
                d1 = traj_dates[t]
                d0 = traj_dates[t-1]
                # Check 1 day difference
                diff = (d1 - d0).astype('timedelta64[D]').astype(int)
                self.assertEqual(diff, 1, f"Trajectory {b} is not sequential at step {t}")

    def test_hidden_state_reset(self):
        """Test 3: Verify hidden state is reset at start of batch."""
        batch = self.trainer.sample_trajectory_batch(self.data_tensors)
        
        # Mock model forward to return dummy outputs
        B = self.config.mafia_batch_size
        K = self.config.mafia_top_k
        D = 16
        
        # Create dummy tensors with requires_grad=True to support loss calc test downstream
        # We need to return NEW tensors each call if we want to mimic real forward, 
        # but for reset test, simple return is fine.
        dummy_outputs = (
            th.randn(B, self.N, requires_grad=True), 
            th.randn(B, requires_grad=True), 
            th.randn(B, self.N, requires_grad=True), 
            th.randn(B, D, requires_grad=True),
            th.randn(B, 3, requires_grad=True), 
            th.randint(0, self.N, (B, K)), 
            th.randn(B, K, D, requires_grad=True), 
            th.softmax(th.randn(B, K, requires_grad=True), dim=-1)
        )
        self.mock_model.return_value = dummy_outputs
        
        # Setup mock for reset_temporal_state
        self.mock_model.reset_temporal_state = MagicMock()
        
        # Run step
        # We need to patch loss.backward because dummy loss might not be valid for backward 
        # but here we just check reset call.
        with patch('torch.Tensor.backward') as mock_backward:
             self.trainer.collect_and_train_step(batch, self.data_tensors)
        
        # Verify reset called ONCE
        self.mock_model.reset_temporal_state.assert_called_once()


    def test_loss_masking_logic(self):
        """Test 4: Verify logic flow for masked losses (L_PG)."""
        # Ensure lambdas are non-zero
        self.trainer.lambda_pg = 1.0
        self.trainer.lambda_risk = 1.0
        self.trainer.lambda_dir = 1.0
        
        batch = self.trainer.sample_trajectory_batch(self.data_tensors)
        
        # Mock model outputs
        B = self.config.mafia_batch_size
        K = self.config.mafia_top_k
        D = 16
        
        # We iterate T_m times. We need the list to contain T_m outputs.
        # But mock return_value returns the SAME tuple every time.
        # That's fine, but the tensors need requires_grad=True.
        
        dummy_outputs = (
            th.randn(B, self.N, requires_grad=True), 
            th.randn(B, requires_grad=True), 
            th.randn(B, self.N, requires_grad=True), 
            th.randn(B, D, requires_grad=True),
            th.randn(B, 3, requires_grad=True), 
            th.randint(0, self.N, (B, K)), 
            th.randn(B, K, D, requires_grad=True), 
            th.softmax(th.randn(B, K, requires_grad=True), dim=-1)
        )
        self.mock_model.return_value = dummy_outputs
        
        # Mock optimizer
        self.mock_observer.optimizer = MagicMock()
        self.mock_observer.mafia_model.parameters = MagicMock(return_value=[th.randn(1)]) # Mock params
        
        # Run step, but prevent actual backward execution to avoid "element 0..." error
        # if the graph is constructed weirdly with mocks.
        # Instead, we verify L_total is calculated and 'backward' is called on it.
        with patch('torch.Tensor.backward') as mock_backward:
            metrics = self.trainer.collect_and_train_step(batch, self.data_tensors)
            
            # Verify backward was called
            self.assertTrue(mock_backward.called, "L_total.backward() should be called")
        
        # Check metrics exist
        self.assertIn("loss_pg", metrics)
        self.assertIn("loss_total", metrics)
        
        # Basic sanity check
        self.assertIsInstance(metrics["loss_pg"], float)

if __name__ == '__main__':
    unittest.main()
