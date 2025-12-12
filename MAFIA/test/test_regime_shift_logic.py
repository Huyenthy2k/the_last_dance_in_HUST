
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.MAFIA.utils.tradeEnv import StockPortfolioEnv

class TestRegimeShiftLogic(unittest.TestCase):
    def setUp(self):
        # Mock config
        self.config = MagicMock()
        self.config.regime_confirmation_window = 3
        self.config.regime_trend_z_threshold = 1.0
        self.config.regime_vol_k = 3.0
        self.config.regime_dc_threshold_pct = 0.02
        self.config.enable_cov_features = False
        self.config.mafia_allow_observer_training = False
        self.config.risk_eta_window = 50
        self.config.trade_pattern = 1
        self.config.benchmark_algo = 'MAFIA'  # Random string not in list
        self.config.only_long_algo_lst = []
        self.config.mafia_top_k = 5
        self.config.mafia_D = 16
        self.config.mafia_regime_dim = 0
        self.config.rl_obs_global_scalars = ["log_capital", "last_turnover", "drawdown"]
        self.config.regime_confirmation_window = 3
        self.config.regime_vol_window = 20
        self.config.market_name = 'TEST'
        self.config.mkt_rf = {'TEST': 0.0}
        
        # Mock rawdata
        self.rawdata = MagicMock()
        
        # Create minimal environment
        # Only mock what's needed for _detect_regime_shift
        with patch('agents.MAFIA.utils.tradeEnv.StockPortfolioEnv._build_ctl_state'), \
             patch('agents.MAFIA.utils.tradeEnv.StockPortfolioEnv._ensure_minimal_features'):
            self.env = StockPortfolioEnv(
                config=self.config,
                rawdata=self.rawdata,
                mode='train',
                stock_num=5,
                action_dim=5
            )
        
        # Reset internal buffers
        self.env._direction_history = []
        self.env._last_dc_direction = 0

    def test_trend_reversal_bear_to_bull(self):
        """Test Trend Reversal Trigger: Bear -> Bull"""
        # Scenario: 3 days of Bear, then 3 days of Bull
        # History window = 3
        # Prev regime (index -4) should be Bear
        # Current seq (last 3) should be Bull
        
        # Fill history: Bear, Bear, Bear, Bull, Bull, Bull
        # Mapping: 0=Bear, 1=Flat, 2=Bull in test logic? 
        # Wait, let's check tradeEnv mapping: 
        # In run_mkt_observer: 0=Bear, 1=Flat, 2=Bull (via get_display DEFAULT)
        # BUT in _detect_regime_shift logic:
        # bull, bear = 0, 2 (Lines 5162)
        # Wait, usually 0 is Bear and 2 is Bull?
        # Let's re-read line 5148: "0 bull, 1 side, 2 bear"
        # Wait, line 5148 comment says: "0 bull, 1 side, 2 bear"
        # BUT line 4279 says: "DIRECTION_NAMES = {0: "BEAR", 1: "FLAT", 2: "BULL"}"
        # CONTRADICTION FOUND IN COMMENTS vs LOGIC?
        # Let's check _detect_regime_shift again.
        # line 5162: bull, bear = 0, 2
        # If 0 is Bull, then logic: all(v == bull) -> all are 0.
        # If 0 is Bear, then logic is confused.
        
        # Let's assume the Code Logic in _detect_regime_shift uses 0 for Bull and 2 for Bear based on line 5162 variable names.
        # But wait, typically index 0 is Bear in MAFIA.
        # Let's check MAFIAObserver output.
        # Standard: 0=Down, 1=Flat, 2=Up.
        # If 5162 says `bull, bear = 0, 2`, then it expects 0=Bull, 2=Bear.
        # DATA MISMATCH RISK?
        # Function signature comment 5148: "0 bull, 1 side, 2 bear" <-- This is likely the source of truth for THIS function, even if weird.
        # So for this test, I will use 0=Bull, 2=Bear to match the function's internal logic.
        
        # Standard MAFIA direction mapping: 0=Bear, 1=Side, 2=Bull
        bull = 2
        bear = 0
        
        # Bear -> Bull Reversal (Bear, Bull, Bull, Bull)
        # window = 3. Hist[-4] = Bear. Hist[-3:] = [Bull, Bull, Bull]
        hist = [bear] + [bull] * 3
        self.env._direction_history = hist
        
        # Trigger call
        is_shift, reason = self.env._detect_regime_shift(
            dir_label=bull, # Current label
            trend_z=1.5,    # Strong trend
            vol_current=0.1, vol_mu=0.1, vol_sigma=0.01, dc_triggered=False
        )
        
        self.assertTrue(is_shift)
        self.assertIn("direction_reversal_bear_to_bull", reason)

    def test_volatility_shock(self):
        """Test Volatility Shock Trigger"""
        # vol_current > mu + k * sigma
        # k=3.0
        mu = 0.01
        sigma = 0.005
        threshold = mu + 3.0 * sigma # 0.01 + 0.015 = 0.025
        
        # Case 1: No Shock
        is_shift, _ = self.env._detect_regime_shift(
            dir_label=1, trend_z=None,
            vol_current=0.02, vol_mu=mu, vol_sigma=sigma, dc_triggered=False
        )
        self.assertFalse(is_shift)
        
        # Case 2: Shock
        is_shift, reason = self.env._detect_regime_shift(
            dir_label=1, trend_z=None,
            vol_current=0.03, vol_mu=mu, vol_sigma=sigma, dc_triggered=False
        )
        self.assertTrue(is_shift)
        self.assertIn("vol_shock", reason)

    def test_structural_break_downward(self):
        """Test Structural Break (Downward DC)"""
        # Spec: triggers only on Downward DC
        
        # Case 1: Downward DC (Should trigger)
        self.env._last_dc_direction = -1 # Bearish DC
        is_shift, reason = self.env._detect_regime_shift(
            dir_label=1, trend_z=None, vol_current=0.01, vol_mu=0.01, vol_sigma=0.01,
            dc_triggered=True
        )
        self.assertTrue(is_shift)
        self.assertEqual(reason, "dc_trigger_index_bear")
        
        # Case 2: Upward DC (Should NOT trigger)
        self.env._last_dc_direction = 1 # Bullish DC
        is_shift, reason = self.env._detect_regime_shift(
            dir_label=1, trend_z=None, vol_current=0.01, vol_mu=0.01, vol_sigma=0.01,
            dc_triggered=True
        )
        self.assertFalse(is_shift)
        self.assertIsNone(reason)

        self.assertFalse(is_shift)
        self.assertIsNone(reason)

    def test_execution_workflow_state_teleportation(self):
        """Test Execution Workflow: State Teleportation (Spec 8.3 & 8.5)"""
        # Mock dependencies for run_mkt_observer
        self.env.mkt_observer = MagicMock()
        self.env.config.enable_market_observer = True
        self.env.config.mafia_T_w = 30
        self.env.config.finefreq = '1d'
        self.env.config.mafia_allow_observer_training = False
        
        # Mock Data extraction
        self.env.extra_data = {
            "fine_market": pd.DataFrame({'date': ['2023-01-01'], 'mkt_1d_close': [100], 'mkt_1d_ma': [100]}),
            "fine_stock": pd.DataFrame({'date': ['2023-01-01'], 'stock': ['A'], 'stock_1d_ma': [10]})
        }
        self.env.curData = pd.DataFrame({'date': ['2023-01-01'], 'stock': ['A'], 'close': [10]})
        self.env.stock_lst = np.array(['A'])
        self.env.stock_index_map = {'A': 0}
        self.env.curTradeDay = 10
        
        # Setup pre-condition
        self.env.days_since_rebalance = 5
        self.env.rl_last_action = np.array([0.9, 0.1, 0.0, 0.0, 0.0])
        self.env.rl_stock_num = 5
        self.env.stock_num = 5
        
        # Mock Methods to skip complex logic
        self.env._extract_raw_ochlv_window = MagicMock(return_value=np.zeros((5, 5, 30)))
        self.env._get_current_topk_indices = MagicMock(return_value=np.array([0, 1, 2, 3, 4]))
        self.env._detect_regime_shift = MagicMock(return_value=(True, "test_shift")) # FORCE REGIME SHIFT
        self.env._build_mafia_state = MagicMock()
        
        # Mock mkt_observer.predict return
        # Returns: market_vec, risk_eta, scores, context, logits, topk_idx, topk_emb, topk_scores
        self.env.mkt_observer.predict.return_value = (
            np.zeros(10), np.zeros(1), np.zeros(5), np.zeros(16), np.zeros(3), 
            np.array([0,1,2,3,4]), np.zeros((5, 16)), np.zeros(5)
        )

        # Run
        with patch('agents.MAFIA.utils.tradeEnv.deque', side_effect=lambda maxlen: MagicMock()):
             self.env.run_mkt_observer(stage="run", selection_trigger=False)
        
        # Assertions
        # 1. days_since_rebalance should be reset to 0
        self.assertEqual(self.env.days_since_rebalance, 0, "days_since_rebalance must be reset to 0 on regime shift")
        
        # 2. rl_last_action should be uniform
        expected_uniform = np.ones(5, dtype=np.float32) / 5.0
        np.testing.assert_array_almost_equal(self.env.rl_last_action, expected_uniform, 
                                             err_msg="rl_last_action should be reset to uniform")

if __name__ == '__main__':
    unittest.main()
