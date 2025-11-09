#！/usr/bin/python
# -*- coding: utf-8 -*-#

"""
MAFIA Feature Processor

This module handles feature preprocessing for MAFIA agents:
- Technical Agent: Computes SMA(20), RSI(14), ATR(14) from raw OCHLV
- DC Agents: Generates Directional Change features (State, Magnitude, Duration, Volume_Ratio, Event_Flag)

Author: MASA
See: Đặc Tả Kỹ Thuật (Technical Specification) - MAFIA.md
"""
import numpy as np
import pandas as pd
import torch as th
from typing import Tuple, List, Optional

# Try to import talib for technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using fallback implementations for technical indicators.")


class MAFIAFeatureProcessor:
    """
    Feature processor for MAFIA agents.
    
    Generates features for:
    - Technical Agent (i=0): Raw OCHLV + Technical Indicators (SMA, RSI, ATR)
    - DC Agents (i=1,2,3): Directional Change features (State, Magnitude, Duration, Volume_Ratio, Event_Flag)
    """
    
    def __init__(self, config):
        """
        Initialize feature processor.
        
        Args:
            config: Configuration object with MAFIA hyperparameters
        """
        self.config = config
        self.T_w = config.mafia_T_w  # Observation window size (30)
        self.DC_thresholds = config.mafia_DC_thresholds  # [0.005, 0.01, 0.02]
        self.M_tech = config.mafia_M_tech  # 8 features for Technical agent
        self.M_dc = config.mafia_M_dc  # 5 features for DC agents
        
    def process_technical_features(self, ochlv_data: np.ndarray) -> np.ndarray:
        """
        Process features for Technical Agent (i=0).
        
        Input: Raw OCHLV data
        - ochlv_data: (N, 5, T_w) where 5 = [open, close, high, low, volume]
        
        Output: P_Tech ∈ ℝ^(N × T_w × 8)
        - 5 features: Open, Close, High, Low, Volume
        - 3 features: SMA(20), RSI(14), ATR(14)
        
        Args:
            ochlv_data: (N, 5, T_w) numpy array
        
        Returns:
            P_Tech: (N, T_w, 8) numpy array
        """
        N, M, T_w = ochlv_data.shape
        assert M == 5, f"Expected 5 features (OCHLV), got {M}"
        assert T_w == self.T_w, f"Expected window size {self.T_w}, got {T_w}"
        
        # Extract individual features
        open_prices = ochlv_data[:, 0, :]  # (N, T_w)
        close_prices = ochlv_data[:, 1, :]  # (N, T_w)
        high_prices = ochlv_data[:, 2, :]  # (N, T_w)
        low_prices = ochlv_data[:, 3, :]  # (N, T_w)
        volumes = ochlv_data[:, 4, :]  # (N, T_w)
        
        # Initialize output array
        P_Tech = np.zeros((N, T_w, self.M_tech))
        
        # Copy raw OCHLV features (first 5)
        P_Tech[:, :, 0] = open_prices
        P_Tech[:, :, 1] = close_prices
        P_Tech[:, :, 2] = high_prices
        P_Tech[:, :, 3] = low_prices
        P_Tech[:, :, 4] = volumes
        
        # Compute technical indicators for each asset
        for n in range(N):
            # SMA(20) on Close
            sma_20 = self._compute_sma(close_prices[n, :], window=20)
            P_Tech[n, :, 5] = sma_20
            
            # RSI(14) on Close
            rsi_14 = self._compute_rsi(close_prices[n, :], period=14)
            P_Tech[n, :, 6] = rsi_14
            
            # ATR(14) on High, Low, Close
            atr_14 = self._compute_atr(
                high_prices[n, :],
                low_prices[n, :],
                close_prices[n, :],
                period=14
            )
            P_Tech[n, :, 7] = atr_14
        
        return P_Tech
    
    def process_dc_features(self, ochlv_data: np.ndarray, dc_threshold: float) -> np.ndarray:
        """
        Process features for a DC Agent with given threshold.
        
        Input: Raw price data
        - ochlv_data: (N, 5, T_w) where 5 = [open, close, high, low, volume]
        
        Output: P_DC ∈ ℝ^(N × T_w × 5)
        - State: +1 (Up-trend), -1 (Down-trend)
        - Magnitude: % change from last DC event
        - Duration: Days since last DC event
        - Volume_Ratio: Volume[t] / Mean(Volume[0...T_w-1])
        - Event_Flag: 1.0 (DC event) or 0.5 (OS event)
        
        Args:
            ochlv_data: (N, 5, T_w) numpy array
            dc_threshold: DC threshold (e.g., 0.005, 0.01, 0.02)
        
        Returns:
            P_DC: (N, T_w, 5) numpy array
        """
        N, M, T_w = ochlv_data.shape
        assert M == 5, f"Expected 5 features (OCHLV), got {M}"
        assert T_w == self.T_w, f"Expected window size {self.T_w}, got {T_w}"
        
        # Extract prices and volumes
        close_prices = ochlv_data[:, 1, :]  # (N, T_w)
        high_prices = ochlv_data[:, 2, :]  # (N, T_w)
        low_prices = ochlv_data[:, 3, :]  # (N, T_w)
        volumes = ochlv_data[:, 4, :]  # (N, T_w)
        
        # Initialize output array
        P_DC = np.zeros((N, T_w, self.M_dc))
        
        # Compute DC features for each asset
        for n in range(N):
            state, magnitude, duration, event_flag = self._compute_dc_sequence(
                close_prices[n, :],
                high_prices[n, :],
                low_prices[n, :],
                dc_threshold
            )
            
            P_DC[n, :, 0] = state  # State
            P_DC[n, :, 1] = magnitude  # Magnitude
            P_DC[n, :, 2] = duration  # Duration
            
            # Volume_Ratio: Volume[t] / Mean(Volume[0...T_w-1])
            volume_mean = np.mean(volumes[n, :])
            if volume_mean > 0:
                P_DC[n, :, 3] = volumes[n, :] / volume_mean
            else:
                P_DC[n, :, 3] = 1.0
            
            P_DC[n, :, 4] = event_flag  # Event_Flag
        
        return P_DC
    
    def _compute_dc_sequence(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, 
                            threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute DC sequence for a single asset.
        
        DC Logic:
        - Upward DC: p_t >= p_{t-1}^l * (1 + threshold), where p_{t-1}^l is low in downward trend
        - Downward DC: p_t <= p_{t-1}^h * (1 - threshold), where p_{t-1}^h is high in upward trend
        
        Returns:
            state: (T_w,) - Current trend state (+1 or -1)
            magnitude: (T_w,) - % change from last DC event
            duration: (T_w,) - Days since last DC event
            event_flag: (T_w,) - 1.0 (DC event) or 0.5 (OS event)
        """
        T_w = len(close)
        state = np.zeros(T_w)
        magnitude = np.zeros(T_w)
        duration = np.zeros(T_w)
        event_flag = np.zeros(T_w)
        
        # Initialize: start with upward trend
        current_trend = 1  # +1 for up, -1 for down
        current_extreme = close[0]  # Current extreme price (high for up, low for down)
        event_price = close[0]  # Price at last DC event
        event_time = 0  # Time of last DC event
        
        state[0] = current_trend
        magnitude[0] = 0.0
        duration[0] = 0
        event_flag[0] = 1.0  # First point is considered an event
        
        for t in range(1, T_w):
            if current_trend == 1:  # Upward trend
                # Check for Downward DC
                if close[t] <= current_extreme * (1 - threshold):
                    # Downward DC event
                    current_trend = -1
                    current_extreme = close[t]
                    event_price = close[t]
                    event_time = t
                    event_flag[t] = 1.0
                else:
                    # Continue upward (OS - Overshoot)
                    if high[t] > current_extreme:
                        current_extreme = high[t]
                    event_flag[t] = 0.5
            else:  # Downward trend
                # Check for Upward DC
                if close[t] >= current_extreme * (1 + threshold):
                    # Upward DC event
                    current_trend = 1
                    current_extreme = close[t]
                    event_price = close[t]
                    event_time = t
                    event_flag[t] = 1.0
                else:
                    # Continue downward (OS - Overshoot)
                    if low[t] < current_extreme:
                        current_extreme = low[t]
                    event_flag[t] = 0.5
            
            state[t] = current_trend
            magnitude[t] = (close[t] / event_price) - 1.0 if event_price > 0 else 0.0
            duration[t] = t - event_time
        
        return state, magnitude, duration, event_flag
    
    def _compute_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Compute Simple Moving Average."""
        if TALIB_AVAILABLE:
            # Use TA-Lib if available
            result = talib.SMA(prices, timeperiod=window)
            # Fill NaN values with first valid value
            if np.isnan(result[0]):
                first_valid = np.where(~np.isnan(result))[0]
                if len(first_valid) > 0:
                    result[:first_valid[0]] = result[first_valid[0]]
        else:
            # Fallback: pandas rolling mean
            result = pd.Series(prices).rolling(window=window, min_periods=1).mean().values
        
        return result
    
    def _compute_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute Relative Strength Index."""
        if TALIB_AVAILABLE:
            result = talib.RSI(prices, timeperiod=period)
            # Fill NaN values with 50 (neutral RSI)
            result = np.nan_to_num(result, nan=50.0)
        else:
            # Fallback implementation
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            # Use exponential moving average
            avg_gain = pd.Series(gain).ewm(alpha=1.0/period, adjust=False).mean().values
            avg_loss = pd.Series(loss).ewm(alpha=1.0/period, adjust=False).mean().values
            
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
            rsi = 100 - (100 / (1 + rs))
            
            # Pad first value
            result = np.concatenate([[50.0], rsi])
            if len(result) < len(prices):
                result = np.pad(result, (0, len(prices) - len(result)), mode='edge')
        
        return result
    
    def _compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Compute Average True Range."""
        if TALIB_AVAILABLE:
            result = talib.ATR(high, low, close, timeperiod=period)
            # Fill NaN values with first valid value
            if np.isnan(result[0]):
                first_valid = np.where(~np.isnan(result))[0]
                if len(first_valid) > 0:
                    result[:first_valid[0]] = result[first_valid[0]]
        else:
            # Fallback implementation
            tr = np.zeros(len(close))
            for i in range(1, len(close)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
            # Use exponential moving average
            result = pd.Series(tr).ewm(alpha=1.0/period, adjust=False).mean().values
        
        return result

