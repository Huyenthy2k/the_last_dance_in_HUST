
import unittest
import numpy as np
import pandas as pd
import torch as th
from agents.MAFIA.RL_controller.mafia_feature_processor import MAFIAFeatureProcessor
from agents.MAFIA.config import Config

class TestFeatureRobustness(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.mafia_T_w = 30
        self.config.mafia_DC_multipliers = [0.5, 1.0, 2.0]
        self.processor = MAFIAFeatureProcessor(self.config)

    def test_soft_clip(self):
        # Test basic tanh behavior
        limit = 1.0
        x = np.array([0.0, 0.5, 1.0, 2.0, 5.0, -5.0])
        clipped = self.processor._soft_clip(x, limit)
        
        # Check constraints
        self.assertTrue(np.all(np.abs(clipped) <= limit))
        
        # Check near-zero linearity: tanh(x) approx x for small x
        self.assertAlmostEqual(clipped[1], 1.0 * np.tanh(0.5/1.0), places=4)
        
        # Check compression
        self.assertLess(np.abs(clipped[3]), 2.0)  # Should be compressed
        self.assertLess(np.abs(clipped[4]), 1.0)  # Should be bounded by limit
        
    def test_relative_volatility(self):
        # Create mock data with 30 days
        # First 20 days calm, last 10 days volatile
        
        # High volatility in returns
        r_calm = np.random.normal(0, 0.01, 20)
        r_shock = np.random.normal(0, 0.05, 10)
        returns = np.concatenate([r_calm, r_shock])
        
        # Reconstruct prices from returns
        price = 100 * (1 + returns).cumprod()
        
        # Create input (1, 5, 30) for Close
        ochlv_data = np.zeros((1, 5, 30))
        ochlv_data[0, 1, :] = price # Close prices
        # Fill others with dummy data to avoid div/0 in other features
        ochlv_data[0, 0, :] = price 
        ochlv_data[0, 2, :] = price * 1.01
        ochlv_data[0, 3, :] = price * 0.99
        ochlv_data[0, 4, :] = 1000.0
        
        # Run processor
        P_Mkt = self.processor.process_market_index_features(ochlv_data)
        
        # Extract relative vol feature (index 16)
        # It's an array of length 30
        vol_feat = P_Mkt[0, :, 16]
        
        # Check that the end of period (volatile) has higher relative vol than beginning
        # Note: First few values might be NaN or 0 depending on rolling window min_periods
        # The code uses min_periods=1, so we should get values.
        
        # Vol_10 at end (shock) should be higher than Vol_30 at end (blend of calm+shock)
        # Expected ratio > 1.0
        
        # In calm period (t=10), vol_10 approx vol_30 approx 0.01 -> Ratio ~ 1.0
        
        # Let's check mean of last 5 days vs mean of days 10-15
        vol_shock_period = np.mean(vol_feat[-5:])
        vol_calm_period = np.mean(vol_feat[10:15])
        
        # The relative volatility should be higher during the shock
        # (Though vol_30 also rises, vol_10 rises faster)
        self.assertGreater(vol_shock_period, 1.0)
        
    def test_adaptive_dc_thresholds(self):
        # Test that higher volatility leads to larger effective thresholds
        # But wait, we can't easily check internal threshold values since they aren't returned.
        # We can check the triggering of events.
        
        # Create a price series that has constant % changes
        # If volatility (ATR) is high, the dynamic threshold should be high.
        # If we have fixed % moves that are smaller than the dynamic threshold, NO events should trigger.
        
        multiplier = 1.0 # k=1.0
        
        # Case A: Low Volatility (ATR low), Price Move 2% -> Should Trigger
        # ATR ~ 1.0, Price ~ 100 => Threshold ~ 1%
        # Move 2% > 1% => Trigger!
        
        # Case B: High Volatility (ATR high), Price Move 2% -> Should NOT Trigger
        # ATR ~ 5.0, Price ~ 100 => Threshold ~ 5%
        # Move 2% < 5% => No Trigger!
        
        T_w = 30
        prices = np.full(T_w, 100.0)
        high = np.full(T_w, 100.0)
        low = np.full(T_w, 100.0)
        volume = np.full(T_w, 1000.0)
        
        # Setup High Volatility context (High-Low range large)
        high_vol = 5.0 # ATR approx 5
        high[:] = prices + high_vol/2
        low[:] = prices - high_vol/2
        
        # Inject a 2% price DROP sequence at the end (to trigger Downward DC from initial Upward trend)
        # t=20: 100, t=21: 98 (2% drop)
        prices[20:] = 98.0
        high[20:] = 98.0 + high_vol/2
        low[20:] = 98.0 - high_vol/2
        
        ochlv_data = np.zeros((1, 5, 30))
        ochlv_data[0, 1, :] = prices
        ochlv_data[0, 2, :] = high
        ochlv_data[0, 3, :] = low
        ochlv_data[0, 4, :] = volume
        
        # Run with large ATR => Threshold ~ 5% => 2% move should NOT trigger event
        P_DC = self.processor.process_dc_features(ochlv_data, k_multiplier=1.0)
        events = P_DC[0, :, 4] # Event_Flag
        
        # Check events at the jump (t=20, 21...)
        # Event flag 1.0 means event. 0.5 means OS.
        # With threshold 5%, a 2% move is NOT an event.
        # However, it might be an OS event?
        # If threshold is not crossed, it stays in same trend.
        # Depending on initialization trend.
        
        # Let's count total events (1.0). Should be near 0 or just the initial one.
        # The first point is always an event (init).
        num_events_high_vol = np.sum(events == 1.0)
        
        # Now Low Volatility
        low_vol = 0.5 # ATR approx 0.5 => Threshold ~ 0.5%
        high[:] = prices + low_vol/2
        low[:] = prices - low_vol/2
        # Move is still 2% (100 -> 102).
        # 2% > 0.5% => Should trigger event!
        
        high[20:] = 98.0 + low_vol/2
        low[20:] = 98.0 - low_vol/2
        
        ochlv_data[0, 2, :] = high
        ochlv_data[0, 3, :] = low
        
        P_DC_low = self.processor.process_dc_features(ochlv_data, k_multiplier=1.0)
        events_low = P_DC_low[0, :, 4]
        num_events_low_vol = np.sum(events_low == 1.0)
        
        # Expect more events in low vol case (because threshold is tighter) given same price move
        self.assertGreater(num_events_low_vol, num_events_high_vol)

if __name__ == '__main__':
    unittest.main()
