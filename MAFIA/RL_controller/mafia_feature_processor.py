# ！/usr/bin/python
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
from typing import Tuple, List, Optional, Dict

# Try to import talib for technical indicators
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print(
        "Warning: TA-Lib not available. Using fallback implementations for technical indicators."
    )

# Numba enabled for performance
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not found. DC feature computation will be slow.")

@jit(nopython=True)
def _compute_dc_sequence_jit(close, high, low, threshold):
    T_w = len(close)
    state = np.zeros(T_w)
    magnitude = np.zeros(T_w)
    duration = np.zeros(T_w)
    event_flag = np.zeros(T_w)

    # Initialize: start with upward trend
    current_trend = 1
    current_extreme = close[0]
    event_price = close[0]
    event_time = 0

    state[0] = current_trend
    magnitude[0] = 0.0
    duration[0] = 0
    event_flag[0] = 1.0

    for t in range(1, T_w):
        # Handle dynamic threshold (array) or static (scalar)
        thresh_t = threshold[t] if threshold.ndim > 0 else threshold
        
        if current_trend == 1:
            if close[t] <= current_extreme * (1 - thresh_t):
                current_trend = -1
                current_extreme = close[t]
                event_price = close[t]
                event_time = t
                event_flag[t] = 1.0
            else:
                if high[t] > current_extreme:
                    current_extreme = high[t]
                event_flag[t] = 0.5
        else:
            if close[t] >= current_extreme * (1 + thresh_t):
                current_trend = 1
                current_extreme = close[t]
                event_price = close[t]
                event_time = t
                event_flag[t] = 1.0
            else:
                if low[t] < current_extreme:
                    current_extreme = low[t]
                event_flag[t] = 0.5

        state[t] = current_trend
        magnitude[t] = (close[t] / event_price) - 1.0 if event_price > 0 else 0.0
        duration[t] = t - event_time

    return state, magnitude, duration, event_flag

class MAFIAFeatureProcessor:
    """
    Feature processor for MAFIA agents.

    Generates features for:
    - Technical Agent (i=0): Raw OCHLV + Technical Indicators (SMA, RSI, ATR)
    - DC Agents (i=1,2,3): Directional Change features (State, Magnitude, Duration, Volume_Ratio, Event_Flag)

    Normalization:
    - CHANGE features: Clamped to [-clip, +clip] to bound outliers
    - Volume: Log-transform + z-score normalization
    - DC Magnitude: Clamped to prevent extreme values
    - DC Volume_Ratio: Log-transform + clamp
    """

    # Normalization constants (can be overridden via config)
    DEFAULT_PRICE_CHANGE_CLIP = 0.3  # ±30% max daily change for OCHLV
    DEFAULT_VOLUME_CHANGE_CLIP = 3.0  # ±3 std for log-volume changes
    DEFAULT_DC_MAGNITUDE_CLIP = 0.5  # ±50% max magnitude from DC event
    DEFAULT_DC_VOLUME_RATIO_CLIP = 5.0  # Max volume ratio (relative to mean)

    @staticmethod
    def _soft_clip(x: np.ndarray, limit: float) -> np.ndarray:
        """
        Apply soft clipping using tanh to compress outliers while preserving order.
        output = limit * tanh(x / limit)
        """
        if limit <= 0:
            return x
        return limit * np.tanh(x / limit)

    @staticmethod
    def _compute_rolling_vol(returns: pd.Series, window: int) -> np.ndarray:
        """
        Compute safe rolling volatility (std dev of returns).
        """
        return returns.rolling(window=window, min_periods=1).std(ddof=0).values
    


    def __init__(self, config):
        """
        Initialize feature processor.

        Args:
            config: Configuration object with MAFIA hyperparameters
        """
        self.config = config
        self.T_w = config.mafia_T_w  # Observation window size (30)
        # Use multipliers for adaptive thresholds (k * ATR) or fallback to static list if not found
        self.DC_multipliers = getattr(config, 'mafia_DC_multipliers', [0.5, 1.0, 2.0])
        self.M_tech = config.mafia_M_tech  # 8 features for Technical agent
        self.M_dc = config.mafia_M_dc  # 5 features for DC agents

        # Normalization parameters (from config or defaults)
        self.price_change_clip = getattr(
            config, 'mafia_price_change_clip', self.DEFAULT_PRICE_CHANGE_CLIP
        )
        self.volume_change_clip = getattr(
            config, 'mafia_volume_change_clip', self.DEFAULT_VOLUME_CHANGE_CLIP
        )
        self.dc_magnitude_clip = getattr(
            config, 'mafia_dc_magnitude_clip', self.DEFAULT_DC_MAGNITUDE_CLIP
        )
        self.dc_volume_ratio_clip = getattr(
            config, 'mafia_dc_volume_ratio_clip', self.DEFAULT_DC_VOLUME_RATIO_CLIP
        )

        # Cache for DC features to avoid redundant computation
        # Key: (data_hash, threshold) -> Value: P_DC array
        self._dc_cache = {}

        # Cache for pre-computed technical indicators
        # Key: stock_id -> {'sma_20': array, 'rsi_14': array, 'atr_14': array, 'close': array}
        self._tech_cache = {}
        self._tech_cache_enabled = False

        # Cache for pre-computed DC features (per stock, per multiplier)
        # Key: (stock_id, multiplier) -> {'state': array, 'magnitude': array, ...}
        self._dc_precomputed_cache = {}
        self._dc_cache_enabled = False

        # Per-forward-pass cache for ATR to avoid redundant computation
        # Reset at the start of each forward pass
        self._atr_cache = None  # Will be (N, T_w) array
        self._atr_cache_hash = None  # Hash of input data to detect changes

    def precompute_technical_indicators(self, rawdata: pd.DataFrame, stock_list: list):
        """
        Pre-compute technical indicators for entire time series.
        Call this once during data loading to enable caching.

        Args:
            rawdata: DataFrame with columns [stock, date, open, high, low, close, volume]
            stock_list: List of stock symbols to pre-compute for
        """
        print("[MAFIAFeatureProcessor] Pre-computing technical indicators...")
        self._tech_cache = {}

        for stock_id in stock_list:
            stock_data = rawdata[rawdata['stock'] == stock_id].sort_values('date')
            if len(stock_data) < 30:  # Skip if not enough data
                continue

            close_prices = stock_data['close'].values.astype(np.float64)
            high_prices = stock_data['high'].values.astype(np.float64)
            low_prices = stock_data['low'].values.astype(np.float64)
            open_prices = stock_data['open'].values.astype(np.float64)
            volumes = stock_data['volume'].values.astype(np.float64)
            dates = stock_data['date'].values

            # Compute indicators on full time series
            sma_20 = self._compute_sma(close_prices, window=20)
            rsi_14 = self._compute_rsi(close_prices, period=14)
            atr_14 = self._compute_atr(high_prices, low_prices, close_prices, period=14)

            self._tech_cache[stock_id] = {
                'sma_20': sma_20,
                'rsi_14': rsi_14,
                'atr_14': atr_14,
                'close': close_prices,
                'high': high_prices,
                'low': low_prices,
                'open': open_prices,
                'volume': volumes,
                'dates': dates,
            }

        self._tech_cache_enabled = True
        print(f"[MAFIAFeatureProcessor] Cached indicators for {len(self._tech_cache)} stocks")

    def get_cached_indicators(self, stock_id: str, day_indices: np.ndarray) -> Optional[dict]:
        """
        Get pre-computed indicators for a stock at specific day indices.

        Args:
            stock_id: Stock symbol
            day_indices: Array of indices into the stock's time series

        Returns:
            Dict with 'sma_20', 'rsi_14', 'atr_14', 'close' arrays, or None if not cached
        """
        if not self._tech_cache_enabled or stock_id not in self._tech_cache:
            return None

        cache = self._tech_cache[stock_id]
        T = len(day_indices)

        # Validate indices
        max_idx = len(cache['close']) - 1
        valid_indices = np.clip(day_indices, 0, max_idx).astype(int)

        return {
            'sma_20': cache['sma_20'][valid_indices],
            'rsi_14': cache['rsi_14'][valid_indices],
            'atr_14': cache['atr_14'][valid_indices],
            'close': cache['close'][valid_indices],
        }

    def clear_cache(self):
        """Clear all caches to free memory."""
        self._tech_cache = {}
        self._tech_cache_enabled = False
        self._dc_precomputed_cache = {}
        self._dc_cache_enabled = False
        self._dc_cache = {}
        self._atr_cache = None
        self._atr_cache_hash = None

    def _get_cached_atr(self, high_prices: np.ndarray, low_prices: np.ndarray,
                        close_prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Get ATR for all stocks, using cache if available.
        This avoids recomputing ATR 4x per timestep (1x tech + 3x DC).

        Args:
            high_prices: (N, T_w) array
            low_prices: (N, T_w) array
            close_prices: (N, T_w) array
            period: ATR period (default 14)

        Returns:
            atr_all: (N, T_w) array of ATR values for all stocks
        """
        # Create hash of input to detect if data changed
        data_hash = hash((close_prices.tobytes(), high_prices.shape))

        if self._atr_cache is not None and self._atr_cache_hash == data_hash:
            return self._atr_cache

        # Compute ATR for all stocks
        N, T_w = close_prices.shape
        atr_all = np.zeros((N, T_w), dtype=np.float64)

        for n in range(N):
            atr_all[n, :] = self._compute_atr(
                high_prices[n, :], low_prices[n, :], close_prices[n, :], period
            )

        # Cache for subsequent calls in same forward pass
        self._atr_cache = atr_all
        self._atr_cache_hash = data_hash

        return atr_all

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
        P_Tech = np.zeros((N, T_w, self.M_tech), dtype=np.float32)

        # Compute CHANGE features (ΔO, ΔC, ΔH, ΔL, ΔV) with normalization
        raw_feat_stack = [open_prices, close_prices, high_prices, low_prices, volumes]
        for feat_idx, feat_vals in enumerate(raw_feat_stack):
            if feat_vals.shape[1] <= 1:
                change = np.zeros_like(feat_vals)
            elif feat_idx == 4:  # Volume: use log-transform + z-score
                # Log-transform to handle skewed distribution
                log_vol = np.log1p(feat_vals)  # log(1 + volume)
                # Compute change in log-space (log-return)
                log_change = np.diff(log_vol, axis=1, prepend=log_vol[:, :1])
                # Z-score normalization per-stock across the window
                log_mean = np.mean(log_change, axis=1, keepdims=True)
                log_std = np.std(log_change, axis=1, keepdims=True)
                log_std = np.where(log_std == 0, 1.0, log_std)  # Avoid div by zero
                change = (log_change - log_mean) / log_std
                # Clip to bound outliers
                change = self._soft_clip(change, self.volume_change_clip)
            else:  # Price features: percentage change with clipping
                prev_vals = feat_vals[:, :-1]
                cur_vals = feat_vals[:, 1:]
                change = np.divide(
                    cur_vals,
                    prev_vals,
                    out=np.ones_like(cur_vals),
                    where=prev_vals != 0,
                )
                change = change - 1.0
                change = np.concatenate(
                    [np.zeros((N, 1), dtype=change.dtype), change], axis=1
                )
                # Clip price changes to bound extreme values (e.g., stock halts)
                change = self._soft_clip(change, self.price_change_clip)
            P_Tech[:, :, feat_idx] = change.astype(np.float32)

        # Get cached ATR for all stocks (computed once, used by both tech and DC features)
        atr_all = self._get_cached_atr(high_prices, low_prices, close_prices, period=14)

        # [PERF-FIX] Vectorized computation for all N stocks at once (no loop)
        # SMA(20) -> (Close - SMA) / Close (Scale invariant distance)
        sma_20_all = self._compute_sma_vectorized(close_prices, window=20)  # (N, T_w)
        safe_close = np.where(close_prices == 0, 1.0, close_prices)  # (N, T_w)
        P_Tech[:, :, 5] = (close_prices - sma_20_all) / safe_close

        # RSI(14) -> Scale to [0, 1]
        rsi_14_all = self._compute_rsi_vectorized(close_prices, period=14)  # (N, T_w)
        P_Tech[:, :, 6] = rsi_14_all / 100.0

        # ATR(14) -> Scale by Close (Volatility %)
        P_Tech[:, :, 7] = atr_all / safe_close

        return P_Tech.astype(np.float32)

    def process_dc_features(
        self, ochlv_data: np.ndarray, k_multiplier: float
    ) -> np.ndarray:
        """
        Process features for a DC Agent with adaptive threshold.

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
            k_multiplier: Multiplier for ATR-based threshold (Threshold = k * ATR / Close)

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
        P_DC = np.zeros((N, T_w, self.M_dc), dtype=np.float32)

        # Get cached ATR for all stocks (computed once, reused across DC multipliers)
        atr_all = self._get_cached_atr(high_prices, low_prices, close_prices, period=14)

        # VECTORIZED: Pre-compute safe_close and dynamic_thresholds for all stocks
        safe_close_all = np.where(close_prices == 0, 1.0, close_prices)  # (N, T_w)
        dynamic_thresholds_all = np.maximum(
            k_multiplier * atr_all / safe_close_all, 1e-4
        )  # (N, T_w)

        # VECTORIZED: Pre-compute volume ratios for all stocks
        volume_means = np.mean(volumes, axis=1, keepdims=True)  # (N, 1)
        volume_means = np.where(volume_means == 0, 1.0, volume_means)  # Avoid div by zero
        vol_ratios = volumes / volume_means  # (N, T_w)
        vol_ratios_log = np.log1p(vol_ratios) - np.log(2)  # (N, T_w)
        vol_ratios_clipped = self._soft_clip(vol_ratios_log, np.log(self.dc_volume_ratio_clip))

        # Compute DC features for each asset (DC sequence must be per-stock due to state machine)
        for n in range(N):
            state, magnitude, duration, event_flag = self._compute_dc_sequence(
                close_prices[n, :], high_prices[n, :], low_prices[n, :],
                dynamic_thresholds_all[n, :]
            )

            P_DC[n, :, 0] = state  # State (already bounded: +1/-1)
            P_DC[n, :, 1] = self._soft_clip(magnitude, self.dc_magnitude_clip)  # Magnitude
            P_DC[n, :, 2] = duration / float(self.T_w)  # Duration
            P_DC[n, :, 3] = vol_ratios_clipped[n, :]  # Volume ratio (pre-computed)
            P_DC[n, :, 4] = event_flag  # Event_Flag

        return P_DC.astype(np.float32)

    def _compute_dc_sequence(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        if NUMBA_AVAILABLE:
            return _compute_dc_sequence_jit(close, high, low, threshold)
        
        # Pure Python implementation (Fallback)
        T_w = len(close)
        state = np.zeros(T_w)
        magnitude = np.zeros(T_w)
        duration = np.zeros(T_w)
        event_flag = np.zeros(T_w)

        # Initialize: start with upward trend
        current_trend = 1
        current_extreme = close[0]
        event_price = close[0]
        event_time = 0

        state[0] = current_trend
        magnitude[0] = 0.0
        duration[0] = 0
        event_flag[0] = 1.0

        for t in range(1, T_w):
            if current_trend == 1:
                if close[t] <= current_extreme * (1 - threshold):
                    current_trend = -1
                    current_extreme = close[t]
                    event_price = close[t]
                    event_time = t
                    event_flag[t] = 1.0
                else:
                    if high[t] > current_extreme:
                        current_extreme = high[t]
                    event_flag[t] = 0.5
            else:
                if close[t] >= current_extreme * (1 + threshold):
                    current_trend = 1
                    current_extreme = close[t]
                    event_price = close[t]
                    event_time = t
                    event_flag[t] = 1.0
                else:
                    if low[t] < current_extreme:
                        current_extreme = low[t]
                    event_flag[t] = 0.5

            state[t] = current_trend
            magnitude[t] = (close[t] / event_price) - 1.0 if event_price > 0 else 0.0
            duration[t] = t - event_time

        return state, magnitude, duration, event_flag

    def _compute_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Compute Simple Moving Average."""
        # TA-Lib expects float64 input; ensure dtype to avoid type errors
        prices = np.asarray(prices, dtype=np.float64)
        if TALIB_AVAILABLE:
            # Use TA-Lib if available
            result = talib.SMA(prices, timeperiod=window)
            # Fill NaN values with first valid value
            if np.isnan(result[0]):
                first_valid = np.where(~np.isnan(result))[0]
                if len(first_valid) > 0:
                    result[: first_valid[0]] = result[first_valid[0]]
        else:
            # Fallback: pandas rolling mean
            result = (
                pd.Series(prices).rolling(window=window, min_periods=1).mean().values
            )

        return result

    def _compute_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute Relative Strength Index."""
        prices = np.asarray(prices, dtype=np.float64)
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
            avg_gain = (
                pd.Series(gain).ewm(alpha=1.0 / period, adjust=False).mean().values
            )
            avg_loss = (
                pd.Series(loss).ewm(alpha=1.0 / period, adjust=False).mean().values
            )

            rs = np.divide(
                avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0
            )
            rsi = 100 - (100 / (1 + rs))

            # Pad first value
            result = np.concatenate([[50.0], rsi])
            if len(result) < len(prices):
                result = np.pad(result, (0, len(prices) - len(result)), mode="edge")

        return result

    def _compute_sma_vectorized(self, prices: np.ndarray, window: int) -> np.ndarray:
        """
        Compute SMA for all stocks at once using cumsum trick.

        Args:
            prices: (N, T) array of close prices
            window: SMA window size

        Returns:
            sma: (N, T) array of SMA values
        """
        N, T = prices.shape
        prices = prices.astype(np.float64)

        # Cumsum trick for efficient rolling mean
        cumsum = np.cumsum(prices, axis=1)
        cumsum = np.concatenate([np.zeros((N, 1)), cumsum], axis=1)

        sma = np.zeros_like(prices)
        for t in range(T):
            start = max(0, t - window + 1)
            count = t - start + 1
            sma[:, t] = (cumsum[:, t + 1] - cumsum[:, start]) / count

        return sma

    def _compute_rsi_vectorized(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Compute RSI for all stocks at once.

        Args:
            prices: (N, T) array of close prices
            period: RSI period

        Returns:
            rsi: (N, T) array of RSI values [0-100]
        """
        N, T = prices.shape
        prices = prices.astype(np.float64)

        # Compute price changes
        delta = np.diff(prices, axis=1)  # (N, T-1)

        # Separate gains and losses
        gains = np.maximum(delta, 0)
        losses = np.maximum(-delta, 0)

        # EMA smoothing factor
        alpha = 1.0 / period

        # Initialize
        rsi = np.full((N, T), 50.0, dtype=np.float64)
        avg_gain = np.zeros(N)
        avg_loss = np.zeros(N)

        # First period: simple average
        if T > period:
            avg_gain = gains[:, :period].mean(axis=1)
            avg_loss = losses[:, :period].mean(axis=1)

            # RSI for first period
            rs = np.divide(avg_gain, avg_loss, out=np.ones(N), where=avg_loss > 0)
            rsi[:, period] = 100 - (100 / (1 + rs))

            # Subsequent periods: EMA
            for t in range(period, T - 1):
                avg_gain = alpha * gains[:, t] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[:, t] + (1 - alpha) * avg_loss
                rs = np.divide(avg_gain, avg_loss, out=np.ones(N), where=avg_loss > 0)
                rsi[:, t + 1] = 100 - (100 / (1 + rs))

        return rsi

    def _compute_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Compute Average True Range."""
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        if TALIB_AVAILABLE:
            result = talib.ATR(high, low, close, timeperiod=period)
            # Fill NaN values with first valid value
            if np.isnan(result[0]):
                first_valid = np.where(~np.isnan(result))[0]
                if len(first_valid) > 0:
                    result[: first_valid[0]] = result[first_valid[0]]
        else:
            # Fallback implementation
            tr = np.zeros(len(close))
            for i in range(1, len(close)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )
            # Use exponential moving average
            result = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values

        return result

    def process_market_index_features(self, ochlv_data: np.ndarray) -> np.ndarray:
        """
        Process features for Market-index Agent (VNINDEX).

        Input: Raw OCHLV data for single asset (VNINDEX)
        - ochlv_data: (1, 5, T_w) where 5 = [open, close, high, low, volume]

        Output: P_Mkt ∈ ℝ^(1 × T_w × M_mkt)
        - 5 features: Change of OHLCV (ΔO, ΔC, ΔH, ΔL, ΔV)
        - 3 features: SMA(20), RSI(14), ATR(14)
        - 11 extended features: MACD_hist, BB_width, Stoch_K, Stoch_D, ADX14, OBV, MFI14, CCI20, Vol_std20, Drawdown60, Regime_sma20_60

        Args:
            ochlv_data: (1, 5, T_w) numpy array

        Returns:
            P_Mkt: (1, T_w, M_mkt) numpy array where M_mkt = 19
        """
        N, M, T_w = ochlv_data.shape
        assert N == 1, f"Market-index agent expects single asset (N=1), got {N}"
        assert M == 5, f"Expected 5 features (OCHLV), got {M}"
        assert T_w == self.T_w, f"Expected window size {self.T_w}, got {T_w}"

        # Extract individual features (squeeze N dimension)
        open_prices = ochlv_data[0, 0, :]  # (T_w,)
        close_prices = ochlv_data[0, 1, :]  # (T_w,)
        high_prices = ochlv_data[0, 2, :]  # (T_w,)
        low_prices = ochlv_data[0, 3, :]  # (T_w,)
        volumes = ochlv_data[0, 4, :]  # (T_w,)

        M_mkt = getattr(self.config, "mafia_M_mkt", 19)  # change(5) + basic(3) + extended(11)
        P_Mkt = np.zeros((1, T_w, M_mkt), dtype=np.float32)

        # Compute CHANGE features (ΔO, ΔC, ΔH, ΔL, ΔV) with normalization
        raw_feat_stack = [open_prices, close_prices, high_prices, low_prices, volumes]
        for feat_idx, feat_vals in enumerate(raw_feat_stack):
            if len(feat_vals) <= 1:
                change = np.zeros_like(feat_vals)
            elif feat_idx == 4:  # Volume: use log-transform + z-score
                # Log-transform to handle skewed distribution
                log_vol = np.log1p(feat_vals)  # log(1 + volume)
                # Compute change in log-space (log-return)
                log_change = np.diff(log_vol, prepend=log_vol[0])
                # Z-score normalization across the window
                log_mean = np.mean(log_change)
                log_std = np.std(log_change)
                log_std = log_std if log_std > 0 else 1.0  # Avoid div by zero
                change = (log_change - log_mean) / log_std
                # Clip to bound outliers
                change = self._soft_clip(change, self.volume_change_clip)
            else:  # Price features: percentage change with clipping
                prev_vals = feat_vals[:-1]
                cur_vals = feat_vals[1:]
                change = np.divide(
                    cur_vals,
                    prev_vals,
                    out=np.ones_like(cur_vals),
                    where=prev_vals != 0,
                )
                change = change - 1.0
                change = np.concatenate([np.zeros(1, dtype=change.dtype), change])
                # Clip price changes to bound extreme values
                change = self._soft_clip(change, self.price_change_clip)
            P_Mkt[0, :, feat_idx] = change.astype(np.float32)

        # Basic technical indicators (same as Technical Agent)
        # SMA(20) -> (Close - SMA) / Close
        sma_20 = self._compute_sma(close_prices, window=20)
        safe_close = np.where(close_prices == 0, 1.0, close_prices)
        P_Mkt[0, :, 5] = (close_prices - sma_20) / safe_close

        # RSI(14) -> Scale to [0, 1]
        rsi_14 = self._compute_rsi(close_prices, period=14)
        P_Mkt[0, :, 6] = rsi_14 / 100.0

        # ATR(14) -> Scale by Close
        atr_14 = self._compute_atr(high_prices, low_prices, close_prices, period=14)
        P_Mkt[0, :, 7] = atr_14 / safe_close

        # Extended indicators
        # MACD(12,26,9) histogram -> Scale by Close
        ema12 = pd.Series(close_prices).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close_prices).ewm(span=26, adjust=False).mean().values
        macd_line = ema12 - ema26
        macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        macd_hist = macd_line - macd_signal
        P_Mkt[0, :, 8] = macd_hist / safe_close

        # Bollinger Band Width(20,2) (Already Ratio, kept as is)
        bb_mid = pd.Series(close_prices).rolling(window=20, min_periods=1).mean()
        bb_std = pd.Series(close_prices).rolling(window=20, min_periods=1).std(ddof=0)
        bb_up = bb_mid + 2 * bb_std
        bb_low = bb_mid - 2 * bb_std
        bb_width = (bb_up - bb_low) / (bb_mid.replace(0, np.nan)).replace(np.nan, 1.0)
        P_Mkt[0, :, 9] = np.nan_to_num(bb_width.values, nan=0.0, posinf=0.0, neginf=0.0)

        # Stochastic %K/%D(14,3) -> Scale to [0, 1]
        rolling_high14 = (
            pd.Series(high_prices).rolling(window=14, min_periods=1).max().values
        )
        rolling_low14 = (
            pd.Series(low_prices).rolling(window=14, min_periods=1).min().values
        )
        stoch_k = (
            np.divide(
                close_prices - rolling_low14,
                rolling_high14 - rolling_low14,
                out=np.zeros_like(close_prices),
                where=(rolling_high14 - rolling_low14) != 0,
            )
            * 100.0
        )
        stoch_d = pd.Series(stoch_k).rolling(window=3, min_periods=1).mean().values
        P_Mkt[0, :, 10] = stoch_k / 100.0
        P_Mkt[0, :, 11] = stoch_d / 100.0

        # ADX(14) -> Scale to [0, 1]
        plus_dm = np.zeros(len(close_prices))
        minus_dm = np.zeros(len(close_prices))
        for i in range(1, len(close_prices)):
            up_move = high_prices[i] - high_prices[i - 1]
            down_move = low_prices[i - 1] - low_prices[i]
            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr = np.zeros(len(close_prices))
        for i in range(1, len(close_prices)):
            tr[i] = max(
                high_prices[i] - low_prices[i],
                abs(high_prices[i] - close_prices[i - 1]),
                abs(low_prices[i] - close_prices[i - 1]),
            )
        tr_series = pd.Series(tr)
        atr14_series = tr_series.ewm(alpha=1.0 / 14, adjust=False).mean()
        plus_di = 100 * (
            pd.Series(plus_dm).ewm(alpha=1.0 / 14, adjust=False).mean()
            / atr14_series.replace(0, np.nan)
        ).replace(np.nan, 0.0)
        minus_di = 100 * (
            pd.Series(minus_dm).ewm(alpha=1.0 / 14, adjust=False).mean()
            / atr14_series.replace(0, np.nan)
        ).replace(np.nan, 0.0)
        dx = 100 * (
            abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        ).replace(np.nan, 0.0)
        adx14 = dx.ewm(alpha=1.0 / 14, adjust=False).mean().values
        P_Mkt[0, :, 12] = adx14 / 100.0

        # OBV -> Rolling Z-Score (Window 20)
        # Prevents unbounded drift, captures relative volume pressure
        price_diff = np.diff(close_prices, prepend=close_prices[0])
        vol_sign = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
        raw_obv = np.cumsum(vol_sign * volumes)
        obv_series = pd.Series(raw_obv)
        obv_mean = obv_series.rolling(window=20, min_periods=1).mean()
        obv_std = obv_series.rolling(window=20, min_periods=1).std(ddof=0)
        obv_z = (obv_series - obv_mean) / (obv_std.replace(0, np.nan)).replace(
            np.nan, 1.0
        )
        P_Mkt[0, :, 13] = np.nan_to_num(obv_z.values, nan=0.0)

        # MFI(14) -> Scale to [0, 1]
        typical_price = (high_prices + low_prices + close_prices) / 3.0
        tp_diff = np.diff(typical_price, prepend=typical_price[0])
        raw_mf = typical_price * volumes
        pos_mf = np.where(tp_diff > 0, raw_mf, 0.0)
        neg_mf = np.where(tp_diff < 0, raw_mf, 0.0)
        pos_mf14 = pd.Series(pos_mf).rolling(window=14, min_periods=1).sum()
        neg_mf14 = pd.Series(neg_mf).rolling(window=14, min_periods=1).sum()
        mfr = np.divide(pos_mf14, neg_mf14.replace(0, np.nan)).replace(np.nan, 1.0)
        mfi14 = 100 - (100 / (1 + mfr))
        P_Mkt[0, :, 14] = mfi14.values / 100.0

        # CCI(20) -> Scale by 100 (approx range -2 to 2)
        sma_tp20 = pd.Series(typical_price).rolling(window=20, min_periods=1).mean()
        md20 = (
            pd.Series(typical_price)
            .rolling(window=20, min_periods=1)
            .apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        )
        cci20 = (typical_price - sma_tp20) / (0.015 * md20.replace(0, np.nan)).replace(
            np.nan, 1.0
        )
        P_Mkt[0, :, 15] = (
            np.nan_to_num(cci20.values, nan=0.0, posinf=0.0, neginf=0.0) / 100.0
        )

        # Volatility: Relative Volatility (Vol_10 / Vol_30)
        # Using 10-day (fast) vs 30-day (window baseline) to detect regime shifts
        returns = pd.Series(close_prices).pct_change(fill_method=None).fillna(0.0)
        vol_std10 = self._compute_rolling_vol(returns, window=10)
        vol_std30 = self._compute_rolling_vol(returns, window=30)
        # Relative vol: >1 means current short-term is more volatile than the full window baseline
        vol_relative = np.divide(
            vol_std10, 
            vol_std30, 
            out=np.ones_like(vol_std10), 
            where=vol_std30 > 1e-6
        )
        P_Mkt[0, :, 16] = np.clip(vol_relative - 1.0, -1.0, 4.0)  # Center at 0.0 (Ratio 1.0 -> 0.0)

        # Drawdown (relative to rolling 60-day peak) (Already %, kept as is)
        rolling_peak60 = (
            pd.Series(close_prices).rolling(window=60, min_periods=1).max().values
        )
        drawdown60 = np.divide(
            rolling_peak60 - close_prices,
            rolling_peak60,
            out=np.zeros_like(close_prices),
            where=rolling_peak60 != 0,
        )
        P_Mkt[0, :, 17] = drawdown60

        # Regime proxy: (SMA20 - SMA60) / Close
        sma60 = pd.Series(close_prices).rolling(window=60, min_periods=1).mean().values
        regime = (sma_20 - sma60) / safe_close
        P_Mkt[0, :, 18] = regime

        return P_Mkt.astype(np.float32)

    # ============================================================
    # Watchlist System: Technical Score Computation
    # ============================================================

    def compute_technical_score(
        self,
        close_prices: np.ndarray,
        volumes: np.ndarray,
        market_close: np.ndarray,
        momentum_window: int = 60,
        volume_short_window: int = 20,
        volume_long_window: int = 60,
        ma_window: int = 20,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute Technical Score for Watchlist screening.

        Technical Score = (momentum_score + volume_score + ma20_score) / 3
        Where:
        - momentum_score: Percentile rank of excess return (stock - market) over 60 days
        - volume_score: Clip((vol_20d / vol_60d - 1.0) / 0.4, 0, 1)
        - ma20_score: 1.0 if close > SMA(20), else 0.0

        Args:
            close_prices: (N, T) array of stock close prices
            volumes: (N, T) array of stock volumes
            market_close: (T,) array of market index close prices
            momentum_window: Window for momentum calculation (default 60)
            volume_short_window: Short window for volume ratio (default 20)
            volume_long_window: Long window for volume ratio (default 60)
            ma_window: SMA window for trend filter (default 20)

        Returns:
            technical_scores: (N,) array of Technical Scores in [0, 1]
            components: Dict with individual score components for debugging
        """
        N, T = close_prices.shape

        # Ensure we have enough data
        min_lookback = max(momentum_window, volume_long_window, ma_window) + 1
        if T < min_lookback:
            # Not enough data - return neutral scores
            return np.full(N, 0.5), {
                'momentum_score': np.full(N, 0.5),
                'volume_score': np.full(N, 0.5),
                'ma20_score': np.full(N, 0.5),
            }

        # 1. Momentum Score (percentile rank of excess return)
        # Stock return over momentum_window
        stock_returns = (close_prices[:, -1] / close_prices[:, -momentum_window] - 1.0)
        # Market return over momentum_window
        market_return = (market_close[-1] / market_close[-momentum_window] - 1.0)
        # Excess return
        excess_returns = stock_returns - market_return

        # Percentile rank: convert to [0, 1]
        # Using argsort twice trick for percentile rank
        sorted_indices = np.argsort(np.argsort(excess_returns))
        momentum_score = sorted_indices / (N - 1) if N > 1 else np.full(N, 0.5)

        # 2. Volume Score
        # Volume ratio = mean_vol_20d / mean_vol_60d
        mean_vol_short = np.mean(volumes[:, -volume_short_window:], axis=1)
        mean_vol_long = np.mean(volumes[:, -volume_long_window:], axis=1)
        # Avoid division by zero
        vol_ratio = np.divide(
            mean_vol_short,
            mean_vol_long,
            out=np.ones(N),
            where=mean_vol_long > 0
        )
        # Scale: (ratio - 1.0) / 0.4, clipped to [0, 1]
        # If ratio=1.0, score=0; if ratio=1.4, score=1
        volume_score = np.clip((vol_ratio - 1.0) / 0.4, 0.0, 1.0)

        # 3. MA20 Score (binary)
        # Compute SMA(20) for current day
        sma_20 = np.mean(close_prices[:, -ma_window:], axis=1)
        current_close = close_prices[:, -1]
        ma20_score = (current_close > sma_20).astype(np.float32)

        # Composite Technical Score = average of 3 components
        technical_scores = (momentum_score + volume_score + ma20_score) / 3.0

        return technical_scores.astype(np.float32), {
            'momentum_score': momentum_score.astype(np.float32),
            'volume_score': volume_score.astype(np.float32),
            'ma20_score': ma20_score.astype(np.float32),
            'excess_returns': excess_returns.astype(np.float32),
            'vol_ratio': vol_ratio.astype(np.float32),
            'sma_20': sma_20.astype(np.float32),
        }

    def compute_model_score_percentile(
        self,
        model_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Convert model scores to percentile ranks for Watchlist screening.

        Args:
            model_scores: (N,) array of market_scores_full from model
                          This is softmax(expert_logits), representing model's
                          allocation preference for each stock (higher = model prefers more)

        Returns:
            model_percentiles: (N,) array of percentile ranks in [0, 100]
                - 100 = top 1 stock (best)
                - 0 = bottom 1 stock (worst)
        """
        N = len(model_scores)
        if N <= 1:
            return np.full(N, 50.0)

        # Percentile rank using argsort twice trick
        # Higher score = higher rank = higher percentile
        sorted_indices = np.argsort(np.argsort(model_scores))
        percentiles = (sorted_indices / (N - 1)) * 100.0

        return percentiles.astype(np.float32)

    def compute_screening_decision(
        self,
        technical_scores: np.ndarray,
        model_percentiles: np.ndarray,
        current_watchlist: Optional[np.ndarray] = None,
        holdings_mask: Optional[np.ndarray] = None,
        holdings_returns: Optional[np.ndarray] = None,
        tech_add_threshold: float = 0.7,
        tech_remove_threshold: float = 0.3,
        model_add_percentile: float = 50.0,
        model_remove_percentile: float = 30.0,
        holdings_tech_floor: float = 0.3,
        holdings_loss_threshold: float = -0.10,
    ) -> Dict[str, np.ndarray]:
        """
        Compute ADD/REMOVE decisions for Watchlist screening.

        ADD conditions (must satisfy BOTH):
        - Technical Score > tech_add_threshold (e.g., 0.7)
        - Model percentile > model_add_percentile (e.g., 50%)

        REMOVE conditions (satisfy ANY):
        - Technical Score < tech_remove_threshold (e.g., 0.3)
        - Model percentile < model_remove_percentile (e.g., 30%)

        Holdings Protection:
        - Holdings are NOT removed unless:
          - Technical < holdings_tech_floor AND
          - r_hold < holdings_loss_threshold

        Args:
            technical_scores: (N,) Technical Scores in [0, 1]
            model_percentiles: (N,) Model percentiles in [0, 100]
            current_watchlist: (N,) boolean mask of stocks currently in Watchlist
            holdings_mask: (N,) boolean mask of stocks currently held
            holdings_returns: (N,) returns since entry for held stocks
            tech_add_threshold: Technical Score threshold for ADD
            tech_remove_threshold: Technical Score threshold for REMOVE
            model_add_percentile: Model percentile threshold for ADD
            model_remove_percentile: Model percentile threshold for REMOVE
            holdings_tech_floor: Technical floor for holdings protection
            holdings_loss_threshold: Loss threshold for holdings removal

        Returns:
            Dict with:
                - 'add_mask': (N,) boolean - stocks to ADD
                - 'remove_mask': (N,) boolean - stocks to REMOVE
                - 'keep_mask': (N,) boolean - stocks to KEEP unchanged
        """
        N = len(technical_scores)

        # Initialize masks
        add_mask = np.zeros(N, dtype=bool)
        remove_mask = np.zeros(N, dtype=bool)

        # If no current watchlist, use empty mask
        if current_watchlist is None:
            current_watchlist = np.zeros(N, dtype=bool)
        if holdings_mask is None:
            holdings_mask = np.zeros(N, dtype=bool)
        if holdings_returns is None:
            holdings_returns = np.zeros(N, dtype=np.float32)

        # ADD: Technical > threshold AND Model in top percentile
        tech_pass_add = technical_scores > tech_add_threshold
        model_pass_add = model_percentiles > model_add_percentile
        # Only ADD if not already in Watchlist
        add_mask = tech_pass_add & model_pass_add & (~current_watchlist)

        # REMOVE: Technical < threshold OR Model in bottom percentile
        tech_fail = technical_scores < tech_remove_threshold
        model_fail = model_percentiles < model_remove_percentile
        # Only REMOVE if currently in Watchlist
        base_remove = (tech_fail | model_fail) & current_watchlist

        # Holdings Protection: Override REMOVE for holdings with good P&L
        # Remove holding only if: tech < floor AND r_hold < loss_threshold
        holdings_protected = holdings_mask & (
            (technical_scores >= holdings_tech_floor) |  # Tech not terrible
            (holdings_returns >= holdings_loss_threshold)  # Still profitable or small loss
        )
        # Protected holdings cannot be removed
        remove_mask = base_remove & (~holdings_protected)

        # Keep: In Watchlist but neither ADD nor REMOVE
        keep_mask = current_watchlist & (~remove_mask)

        return {
            'add_mask': add_mask,
            'remove_mask': remove_mask,
            'keep_mask': keep_mask,
            'tech_pass_add': tech_pass_add,
            'model_pass_add': model_pass_add,
            'holdings_protected': holdings_protected,
        }
