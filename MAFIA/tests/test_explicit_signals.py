
import unittest
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.MAFIA.RL_controller.mafia_observer import MAFIAObserver
from agents.MAFIA.config import Config as MAFIAConfig

class TestExplicitSignals(unittest.TestCase):
    def setUp(self):
        self.config = MAFIAConfig()
        # Set multipliers for adaptive DC
        self.config.mafia_DC_multipliers = [0.5, 1.0, 2.0]
        # Dummy action_dim=10 for test
        self.observer = MAFIAObserver(self.config, action_dim=10)
        
        # Mock history buffers
        self.observer.price_history_buffer = []
        self.observer.volume_history = []
        self.observer.stock_price_history = []
        self.observer.rsi_history = []
        self.observer.vpi_history = []
        
        # device
        self.observer.device = torch.device("cpu")

    def test_adaptive_dc_signal(self):
        """Test that DC signal adapts to volatility (ATR)."""
        # 1. Simulate LOW volatility regime (ATR ~ 1.0)
        # Price stable around 100
        low_vol_prices = [100.0] * 20
        for p in low_vol_prices:
            self.observer.price_history_buffer.append(p)
            
        # Trigger explicit signal calc (should update internal states)
        # Drop price by 1.5% (100 -> 98.5). 
        # With ATR~0 (flat), min threshold 0.5% applies.
        # Drop 1.5% > 0.5% => Should trigger DC.
        
        # Manually set buffer and state to simulate detection
        self.observer.price_history_buffer = [100.0] * 15 # History
        self.observer.dc_state["p_ext"] = 100.0
        self.observer.dc_state["mode"] = "up"
        
        # Call with drop
        signals = self.observer._compute_explicit_signals(
            market_close_price=98.5, 
            market_volume=1000.0
        )
        
        dc_flag = signals[0, 1].item()
        self.assertEqual(dc_flag, 1.0, "Should trigger DC in low vol regime for 1.5% drop (threshold clamped to 0.5%)")

        # 2. Simulate HIGH volatility regime (ATR ~ 5.0)
        # Reset
        self.observer.price_history_buffer = []
        self.observer.dc_state["p_ext"] = 100.0
        self.observer.dc_state["mode"] = "up"
        
        # Create zigzag history to pump ATR
        # 100, 105, 100, 105... TR=5.0. ATR~5.0.
        # Threshold (k=1.0) ~ 5.0/100 = 5%.
        high_vol_prices = []
        for i in range(20):
            high_vol_prices.append(100.0 if i % 2 == 0 else 105.0)
            
        for p in high_vol_prices:
            self.observer.price_history_buffer.append(p)
            
        # Drop price by 3% (100 -> 97).
        # Threshold ~ 5%. Drop 3% < 5% => Should NOT trigger DC.
        signals_high_vol = self.observer._compute_explicit_signals(
            market_close_price=97.0,
            market_volume=1000.0
        )
        
        dc_flag_high = signals_high_vol[0, 1].item()
        self.assertEqual(dc_flag_high, 0.0, "Should NOT trigger DC in high vol regime for 3% drop (threshold ~5%)")

    def test_relative_volatility_signal(self):
        """Test that Volatility signal is relative (Vol10 / Vol30)."""
        # Fill buffer with enough data > 30
        self.observer.price_history_buffer = [100.0] * 40
        
        # 1. Steady state (Vol10 == Vol30 == 0) -> Returns 1.0 default
        signals = self.observer._compute_explicit_signals(market_close_price=100.0, market_volume=1000.0)
        self.assertEqual(signals[0, 0].item(), 1.0)
        
        # 2. Volatility Spike (Recent 10 days volatile, older stable)
        # Old history: stable 100
        # Recent 10: 100, 102, 100, 102...
        buffer = [100.0] * 20 # 20 days stable
        for i in range(11): # 11 days volatile (last one is current)
            buffer.append(100.0 if i % 2 == 0 else 102.0)
            
        self.observer.price_history_buffer = buffer
        
        signals_shock = self.observer._compute_explicit_signals(market_close_price=102.0, market_volume=1000.0)
        rel_vol = signals_shock[0, 0].item()
        
        # Vol10 should be high, Vol30 (diluted by 20 zeros) should be lower
        # Thus RelVol > 1.0
        self.assertGreater(rel_vol, 1.2, f"Relative Vol should detect spike (got {rel_vol})")

if __name__ == '__main__':
    unittest.main()
