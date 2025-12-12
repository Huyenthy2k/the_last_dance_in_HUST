#!/usr/bin/env python3
"""
Ground Truth Statistics Analysis for VNINDEX

Analyzes direction labels and risk targets to propose optimal thresholds
for Regime Shift Selection Logic (Spec Section 8).

Usage:
    python scripts/analyze_ground_truth.py [--data_file PATH] [--output_dir DIR]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_vnindex_data(data_file: str) -> pd.DataFrame:
    """Load VNINDEX data from CSV."""
    df = pd.read_csv(data_file)

    # Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute returns
    df['returns'] = df['close'].pct_change()

    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    return atr


def compute_direction_labels(df: pd.DataFrame,
                              lookahead: int = 14,
                              atr_period: int = 14,
                              k_atr: float = 2.0,
                              delta_min: float = 0.02,
                              stop_loss: float = -0.07) -> pd.DataFrame:
    """
    Compute direction labels using dynamic threshold (Spec 5.1.3).

    Returns DataFrame with direction_label and intermediate calculations.
    """
    n = len(df)
    close = df['close'].values

    # Compute ATR
    atr = compute_atr(df, period=atr_period).values

    results = {
        'date': df['date'].values if 'date' in df.columns else np.arange(n),
        'close': close,
        'atr': atr,
        'atr_price_ratio': np.zeros(n),
        'dynamic_threshold': np.zeros(n),
        'future_return': np.zeros(n),
        'intra_drawdown': np.zeros(n),
        'direction_label': np.ones(n, dtype=int),  # Default: Side (1)
    }

    for t in range(n):
        # ATR/Price ratio
        atr_ratio = atr[t] / close[t] if close[t] > 0 else 0
        results['atr_price_ratio'][t] = atr_ratio

        # Dynamic threshold
        delta_t = max(delta_min, k_atr * atr_ratio)
        results['dynamic_threshold'][t] = delta_t

        # Future window
        future_end = min(t + lookahead, n - 1)
        if future_end <= t:
            continue

        # Future return
        r_fut = (close[future_end] / close[t]) - 1.0 if close[t] > 0 else 0
        results['future_return'][t] = r_fut

        # Intra-period drawdown (worst point in the window)
        future_prices = close[t+1:future_end+1]
        if len(future_prices) > 0:
            min_price = np.min(future_prices)
            intra_dd = (min_price / close[t]) - 1.0 if close[t] > 0 else 0
            results['intra_drawdown'][t] = intra_dd

        # Label assignment (Spec 5.1.3)
        if r_fut < -delta_t or results['intra_drawdown'][t] < stop_loss:
            results['direction_label'][t] = 0  # Bear
        elif r_fut > delta_t and results['intra_drawdown'][t] >= stop_loss:
            results['direction_label'][t] = 2  # Bull
        else:
            results['direction_label'][t] = 1  # Side

    return pd.DataFrame(results)


def compute_risk_targets(df: pd.DataFrame,
                          lookahead: int = 14,
                          lambda_val: float = 0.3,
                          lambda_dd: float = 0.5,
                          dd_ref: float = 0.10,
                          zscore_window: int = 60) -> pd.DataFrame:
    """
    Compute risk targets using hybrid formula (Spec 5.1.2).

    eta = 1 + lambda_val * tanh(Z_fut) - lambda_dd * clip(MaxDD_fut / dd_ref, 0, 1)
    """
    n = len(df)
    close = df['close'].values
    returns = df['returns'].values if 'returns' in df.columns else np.diff(close, prepend=close[0]) / close

    results = {
        'date': df['date'].values if 'date' in df.columns else np.arange(n),
        'close': close,
        'z_score': np.zeros(n),
        'max_dd_future': np.zeros(n),
        'risk_eta': np.ones(n),
    }

    # Compute rolling mean and std for Z-score
    rolling_mean = pd.Series(returns).rolling(window=zscore_window, min_periods=10).mean().values
    rolling_std = pd.Series(returns).rolling(window=zscore_window, min_periods=10).std().values

    for t in range(n):
        future_end = min(t + lookahead, n - 1)
        if future_end <= t:
            continue

        # Z-score of future returns
        future_returns = returns[t+1:future_end+1]
        if len(future_returns) > 0:
            r_mean = rolling_mean[t] if not np.isnan(rolling_mean[t]) else 0
            r_std = rolling_std[t] if not np.isnan(rolling_std[t]) and rolling_std[t] > 0 else 0.01

            r_fut_mean = np.mean(future_returns)
            z_fut = (r_fut_mean - r_mean) / r_std
            results['z_score'][t] = np.clip(z_fut, -5, 5)

        # Maximum Drawdown in future window
        future_prices = close[t+1:future_end+1]
        if len(future_prices) > 0:
            running_max = np.maximum.accumulate(np.concatenate([[close[t]], future_prices]))
            drawdowns = (future_prices - running_max[1:]) / running_max[1:]
            max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0
            results['max_dd_future'][t] = max_dd

        # Compute eta (Spec 5.1.2)
        z_fut = results['z_score'][t]
        max_dd = abs(results['max_dd_future'][t])

        eta = 1.0 + lambda_val * np.tanh(z_fut) - lambda_dd * np.clip(max_dd / dd_ref, 0, 1)
        eta = np.clip(eta, 0.1, 2.0)
        results['risk_eta'][t] = eta

    return pd.DataFrame(results)


def compute_volatility_stats(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute volatility statistics for regime shift detection."""
    returns = df['returns'].values if 'returns' in df.columns else np.diff(df['close'].values, prepend=df['close'].values[0]) / df['close'].values

    vol = pd.Series(np.abs(returns)).rolling(window=window, min_periods=5).std().values
    vol_mean = pd.Series(vol).rolling(window=window, min_periods=5).mean().values
    vol_std = pd.Series(vol).rolling(window=window, min_periods=5).std().values

    results = {
        'date': df['date'].values if 'date' in df.columns else np.arange(len(df)),
        'returns': returns,
        'vol_current': vol,
        'vol_mean': vol_mean,
        'vol_std': vol_std,
    }

    # Compute shock threshold for different k values
    for k in [2.0, 2.5, 3.0, 3.5]:
        threshold = vol_mean + k * vol_std
        shock_flag = (vol > threshold).astype(int)
        results[f'vol_shock_k{k}'] = shock_flag

    return pd.DataFrame(results)


def compute_transition_stats(labels: np.ndarray) -> dict:
    """Compute state transition statistics."""
    n = len(labels)
    transitions = Counter()
    durations = {0: [], 1: [], 2: []}  # Duration in each state

    current_state = labels[0]
    current_duration = 1

    for i in range(1, n):
        if labels[i] != current_state:
            # Record transition
            transitions[(current_state, labels[i])] += 1
            durations[current_state].append(current_duration)
            current_state = labels[i]
            current_duration = 1
        else:
            current_duration += 1

    # Final duration
    durations[current_state].append(current_duration)

    # Critical transitions: Bull <-> Bear
    bull_to_bear = transitions.get((2, 0), 0)
    bear_to_bull = transitions.get((0, 2), 0)

    return {
        'transitions': dict(transitions),
        'durations': {k: np.array(v) for k, v in durations.items()},
        'bull_to_bear': bull_to_bear,
        'bear_to_bull': bear_to_bull,
        'total_reversals': bull_to_bear + bear_to_bull,
    }


def print_statistics(direction_df: pd.DataFrame, risk_df: pd.DataFrame, vol_df: pd.DataFrame):
    """Print comprehensive statistics."""

    print("\n" + "="*80)
    print("GROUND TRUTH STATISTICS FOR VNINDEX")
    print("="*80)

    # -------------------------------------------------------------------------
    # Direction Labels Statistics
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("1. DIRECTION LABELS (Bear=0, Side=1, Bull=2)")
    print("-"*40)

    labels = direction_df['direction_label'].values
    total = len(labels)

    label_counts = Counter(labels)
    print(f"\nDistribution:")
    print(f"  Bear  (0): {label_counts[0]:5d} ({100*label_counts[0]/total:.1f}%)")
    print(f"  Side  (1): {label_counts[1]:5d} ({100*label_counts[1]/total:.1f}%)")
    print(f"  Bull  (2): {label_counts[2]:5d} ({100*label_counts[2]/total:.1f}%)")

    # Transition stats
    trans_stats = compute_transition_stats(labels)
    print(f"\nState Transitions:")
    print(f"  Bull -> Bear reversals: {trans_stats['bull_to_bear']}")
    print(f"  Bear -> Bull reversals: {trans_stats['bear_to_bull']}")
    print(f"  Total major reversals:  {trans_stats['total_reversals']}")

    print(f"\nState Durations (days):")
    for state, name in [(0, 'Bear'), (1, 'Side'), (2, 'Bull')]:
        durs = trans_stats['durations'][state]
        if len(durs) > 0:
            print(f"  {name}: mean={np.mean(durs):.1f}, median={np.median(durs):.1f}, "
                  f"max={np.max(durs)}, std={np.std(durs):.1f}")

    # ATR/Price ratio (for dynamic threshold)
    atr_ratio = direction_df['atr_price_ratio'].dropna()
    print(f"\nATR/Price Ratio (for k_atr tuning):")
    print(f"  Mean:   {atr_ratio.mean()*100:.3f}%")
    print(f"  Median: {atr_ratio.median()*100:.3f}%")
    print(f"  P10:    {np.percentile(atr_ratio, 10)*100:.3f}%")
    print(f"  P90:    {np.percentile(atr_ratio, 90)*100:.3f}%")

    # Intra-period drawdown
    intra_dd = direction_df['intra_drawdown'].dropna()
    print(f"\nIntra-Period Drawdown (for stop_loss tuning):")
    print(f"  Mean:   {intra_dd.mean()*100:.2f}%")
    print(f"  Median: {intra_dd.median()*100:.2f}%")
    print(f"  P10:    {np.percentile(intra_dd, 10)*100:.2f}%")
    print(f"  P25:    {np.percentile(intra_dd, 25)*100:.2f}%")
    print(f"  P5:     {np.percentile(intra_dd, 5)*100:.2f}%")

    # -------------------------------------------------------------------------
    # Risk Target Statistics
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("2. RISK TARGETS (eta)")
    print("-"*40)

    eta = risk_df['risk_eta'].dropna()
    print(f"\neta Distribution:")
    print(f"  Mean:   {eta.mean():.3f}")
    print(f"  Std:    {eta.std():.3f}")
    print(f"  Min:    {eta.min():.3f}")
    print(f"  Max:    {eta.max():.3f}")
    print(f"  P10:    {np.percentile(eta, 10):.3f}")
    print(f"  P25:    {np.percentile(eta, 25):.3f}")
    print(f"  P50:    {np.percentile(eta, 50):.3f}")
    print(f"  P75:    {np.percentile(eta, 75):.3f}")
    print(f"  P90:    {np.percentile(eta, 90):.3f}")

    # eta by direction
    merged = pd.merge(direction_df[['date', 'direction_label']],
                      risk_df[['date', 'risk_eta']], on='date')
    print(f"\neta by Direction:")
    for state, name in [(0, 'Bear'), (1, 'Side'), (2, 'Bull')]:
        eta_state = merged[merged['direction_label'] == state]['risk_eta']
        if len(eta_state) > 0:
            print(f"  {name}: mean={eta_state.mean():.3f}, std={eta_state.std():.3f}")

    # Z-score distribution
    z_score = risk_df['z_score'].dropna()
    print(f"\nZ-Score Distribution:")
    print(f"  Mean:   {z_score.mean():.3f}")
    print(f"  Std:    {z_score.std():.3f}")
    print(f"  P10:    {np.percentile(z_score, 10):.3f}")
    print(f"  P90:    {np.percentile(z_score, 90):.3f}")

    # -------------------------------------------------------------------------
    # Volatility Statistics (for regime shift)
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("3. VOLATILITY SHOCK STATISTICS (for regime_vol_k)")
    print("-"*40)

    vol = vol_df['vol_current'].dropna()
    print(f"\nRealized Volatility (20-day rolling):")
    print(f"  Mean:   {vol.mean()*100:.4f}%")
    print(f"  Std:    {vol.std()*100:.4f}%")
    print(f"  P90:    {np.percentile(vol, 90)*100:.4f}%")
    print(f"  P95:    {np.percentile(vol, 95)*100:.4f}%")
    print(f"  P99:    {np.percentile(vol, 99)*100:.4f}%")

    print(f"\nVolatility Shock Detection Rate by k:")
    for k in [2.0, 2.5, 3.0, 3.5]:
        shock_rate = vol_df[f'vol_shock_k{k}'].mean() * 100
        shock_days = vol_df[f'vol_shock_k{k}'].sum()
        print(f"  k={k}: {shock_days:.0f} days ({shock_rate:.2f}%)")

    # -------------------------------------------------------------------------
    # Recommended Thresholds
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("RECOMMENDED THRESHOLDS FOR REGIME SHIFT SELECTION LOGIC")
    print("="*80)

    # Direction class weights based on distribution
    bear_pct = label_counts[0] / total
    side_pct = label_counts[1] / total
    bull_pct = label_counts[2] / total

    # Inverse frequency weighting, normalized
    weights_raw = [1/bear_pct if bear_pct > 0 else 1,
                   1/side_pct if side_pct > 0 else 1,
                   1/bull_pct if bull_pct > 0 else 1]
    weights_norm = [w / weights_raw[2] for w in weights_raw]  # Normalize to Bull=1.0

    print(f"\n1. FOCAL LOSS CLASS WEIGHTS (mafia_focal_alpha):")
    print(f"   Distribution: Bear={bear_pct*100:.1f}%, Side={side_pct*100:.1f}%, Bull={bull_pct*100:.1f}%")
    print(f"   Recommended:  [{weights_norm[0]:.2f}, {weights_norm[1]:.2f}, {weights_norm[2]:.2f}]")
    print(f"                 (Bear={weights_norm[0]:.2f} - minority class boost)")

    # ATR multiplier recommendation
    atr_mean = atr_ratio.mean()
    print(f"\n2. DIRECTION LABELING THRESHOLDS:")
    print(f"   ATR/Price mean: {atr_mean*100:.3f}%")
    print(f"   Recommended k_atr: 2.0 (gives delta ~ {2.0*atr_mean*100:.2f}%)")
    print(f"   Current delta_min: 2.0% (matches spec)")

    # Stop-loss threshold based on P10 intra-DD
    dd_p10 = np.percentile(intra_dd, 10)
    print(f"\n3. STOP-LOSS THRESHOLD (direction_label_stop_loss):")
    print(f"   Intra-DD P10: {dd_p10*100:.2f}%")
    print(f"   Recommended:  -7.0% (current, aligns with P10)")

    # Volatility shock k
    # Target: ~1-3% of days should be shocks (rare but impactful)
    print(f"\n4. VOLATILITY SHOCK (regime_vol_k):")
    print(f"   k=3.0 triggers {vol_df['vol_shock_k3.0'].mean()*100:.2f}% of days")
    print(f"   Recommended: 3.0 (captures ~1% extreme events)")

    # DC threshold
    print(f"\n5. DC THRESHOLD (regime_dc_threshold_pct):")
    print(f"   Current: 0.05 (5%)")
    print(f"   Spec requires: 0.02 (2%) for Major Reversal trigger")
    print(f"   Recommended: 0.02 (align with spec Section 8.2)")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze ground truth statistics for VNINDEX")
    parser.add_argument('--data_file', type=str,
                        default='data/VNINDEX_1d_index.csv',
                        help='Path to VNINDEX CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='analysis_output',
                        help='Directory to save analysis outputs')
    parser.add_argument('--lookahead', type=int, default=14,
                        help='Lookahead days for direction/risk labels')
    args = parser.parse_args()

    # Resolve paths relative to MAFIA directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mafia_dir = os.path.dirname(script_dir)

    data_file = args.data_file
    if not os.path.isabs(data_file):
        data_file = os.path.join(mafia_dir, data_file)

    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    print(f"Loading data from: {data_file}")
    df = load_vnindex_data(data_file)
    print(f"Loaded {len(df)} trading days ({df['date'].min()} to {df['date'].max()})")

    # Compute direction labels
    print("\nComputing direction labels...")
    direction_df = compute_direction_labels(df, lookahead=args.lookahead)

    # Compute risk targets
    print("Computing risk targets...")
    risk_df = compute_risk_targets(df, lookahead=args.lookahead)

    # Compute volatility stats
    print("Computing volatility statistics...")
    vol_df = compute_volatility_stats(df)

    # Print comprehensive statistics
    print_statistics(direction_df, risk_df, vol_df)

    # Save to output dir if specified
    output_dir = os.path.join(mafia_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    direction_df.to_csv(os.path.join(output_dir, 'direction_labels.csv'), index=False)
    risk_df.to_csv(os.path.join(output_dir, 'risk_targets.csv'), index=False)
    vol_df.to_csv(os.path.join(output_dir, 'volatility_stats.csv'), index=False)

    print(f"\nAnalysis outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
