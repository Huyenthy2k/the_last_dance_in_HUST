
import unittest
import numpy as np
import torch as th
import sys
import os

# Adjust path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add agents/MAFIA logic

from agents.MAFIA.RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from config import Config

class MockTrainer(ObserverOfflineBatchTrainer):
    def __init__(self):
        self.config = Config(create_dirs=False)
        self.config.mafia_tradeDays_per_year = 252
        # Mock other needed attributes
        self.alpha_turnover = 0.0
        self.alpha_change = 0.0
        self.lambda_dir = 0.0
        self.prev_topk_indices = None
    
    # We only need compute_selection_metrics, which is inherited.

class TestAdaptiveMetrics(unittest.TestCase):
    def setUp(self):
        self.trainer = MockTrainer()

    def test_short_window_arithmetic(self):
        """Test that short window (<= 180 days) uses Arithmetic Annualization for Return."""
        # Setup: 60 days (Quarter)
        T_m = 60
        days = 60
        B = 1
        K = 2
        N = 5
        
        # Indices: (B, T_m, K)
        topk_indices = th.zeros((B, T_m, K), dtype=th.long) 
        topk_scores = th.ones((B, T_m, K), dtype=th.float)
        
        # Price returns: (B, T_m+1, N) -> Dim 1 is Time
        # Need T_m + 1 to allow shift slicing
        price_returns = th.zeros((B, T_m + 1, N), dtype=th.float)
        price_returns[:, :, 0] = 0.001 # Stock 0 has 0.1% daily return
        
        # Market returns: (B, T_m+1)
        market_returns = th.zeros((B, T_m + 1), dtype=th.float)
        
        metrics = self.trainer.compute_selection_metrics(topk_indices, topk_scores, price_returns, market_returns)
        
        # Expected Return (Short Window <= 180): Arithmetic
        # Avg Daily of 0.001 = 0.001
        # Arith Annual = 0.001 * 252 = 0.252
        
        expected_ret = 0.252
        actual_ret = metrics["mean_return"]
        
        print(f"Short Window (60d): Expected {expected_ret:.4f}, Got {actual_ret:.4f}")
        self.assertTrue(np.isclose(actual_ret, expected_ret, atol=1e-4), f"Short window should be Arithmetic. Got {actual_ret} vs {expected_ret}")
        
    def test_long_window_cagr(self):
        """Test that long window (> 180 days) uses CAGR."""
        T_m = 200 # Long window
        B = 1
        K = 1
        N = 5
        
        topk_indices = th.zeros((B, T_m, K), dtype=th.long)
        topk_scores = th.ones((B, T_m, K), dtype=th.float)
        
        price_returns = th.zeros((B, T_m + 1, N), dtype=th.float)
        price_returns[:, :, 0] = 0.001 
        
        market_returns = th.zeros((B, T_m + 1), dtype=th.float)
        
        metrics = self.trainer.compute_selection_metrics(topk_indices, topk_scores, price_returns, market_returns)
        
        # Expected Return (Long Window > 180): CAGR
        # (1 + 0.001)^252 - 1
        expected_ret = (1.001)**252 - 1
        actual_ret = metrics["mean_return"]
        
        print(f"Long Window (200d): Expected {expected_ret:.4f} (CAGR), Got {actual_ret:.4f}")
        self.assertTrue(np.isclose(actual_ret, expected_ret, atol=1e-4), f"Long window should be CAGR. Got {actual_ret} vs {expected_ret}")

if __name__ == '__main__':
    unittest.main()
