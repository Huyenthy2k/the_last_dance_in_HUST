#!/usr/bin/env python3
"""
Statistical Analysis for Turnover, Symdiff Penalty & Advantage

This script analyzes the reasonableness of penalty parameters and advantage computation
from trajectory_details.csv generated during Observer training.

Usage:
    python analyze_penalties.py --input ./observer_offline/trajectory_details.csv
    python analyze_penalties.py --input ./observer_offline/trajectory_details.csv --plot
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def load_trajectory_data(filepath: str) -> pd.DataFrame:
    """Load trajectory details CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    # Check required columns
    required_cols = ['raw_return', 'turnover_penalty', 'symdiff_penalty', 'advantage', 'net_reward', 'baseline']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

    return df


def compute_basic_stats(df: pd.DataFrame) -> Dict:
    """Compute basic statistics for penalties and advantage."""
    stats = {}

    # Turnover statistics
    # Note: turnover_penalty = alpha_turnover * turnover_rate
    # With alpha_turnover = 0.01, turnover_rate = turnover_penalty / 0.01
    alpha_turnover = 0.01
    alpha_change = 0.01

    df['turnover_rate'] = df['turnover_penalty'] / alpha_turnover
    df['symdiff_rate'] = df['symdiff_penalty'] / alpha_change

    # Filter only rebalance days (where penalties > 0)
    rebal_df = df[df['turnover_penalty'] > 0]

    stats['total_samples'] = len(df)
    stats['rebalance_days'] = len(rebal_df)
    stats['hold_days'] = len(df) - len(rebal_df)
    stats['rebalance_ratio'] = len(rebal_df) / len(df) if len(df) > 0 else 0

    # Turnover Rate Stats (on rebalance days only)
    if len(rebal_df) > 0:
        stats['turnover_rate'] = {
            'mean': rebal_df['turnover_rate'].mean(),
            'std': rebal_df['turnover_rate'].std(),
            'min': rebal_df['turnover_rate'].min(),
            'max': rebal_df['turnover_rate'].max(),
            'median': rebal_df['turnover_rate'].median(),
            'p25': rebal_df['turnover_rate'].quantile(0.25),
            'p75': rebal_df['turnover_rate'].quantile(0.75),
            'p90': rebal_df['turnover_rate'].quantile(0.90),
        }
    else:
        stats['turnover_rate'] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

    # Penalty Impact Analysis
    # Total penalties as % of absolute raw returns
    total_penalty = df['turnover_penalty'] + df['symdiff_penalty']
    abs_raw_return = df['raw_return'].abs()

    # Avoid division by zero
    valid_mask = abs_raw_return > 1e-8
    penalty_ratio = np.where(valid_mask, total_penalty / abs_raw_return, 0)

    stats['penalty_impact'] = {
        'total_turnover_penalty': df['turnover_penalty'].sum(),
        'total_symdiff_penalty': df['symdiff_penalty'].sum(),
        'total_raw_return': df['raw_return'].sum(),
        'total_net_reward': df['net_reward'].sum(),
        'penalty_to_return_ratio_mean': penalty_ratio[valid_mask].mean() if valid_mask.sum() > 0 else 0,
        'penalty_to_return_ratio_median': np.median(penalty_ratio[valid_mask]) if valid_mask.sum() > 0 else 0,
        'penalty_to_return_ratio_p90': np.percentile(penalty_ratio[valid_mask], 90) if valid_mask.sum() > 0 else 0,
    }

    # Advantage Statistics
    stats['advantage'] = {
        'mean': df['advantage'].mean(),
        'std': df['advantage'].std(),
        'min': df['advantage'].min(),
        'max': df['advantage'].max(),
        'median': df['advantage'].median(),
        'positive_ratio': (df['advantage'] > 0).mean(),
        'p10': df['advantage'].quantile(0.10),
        'p90': df['advantage'].quantile(0.90),
    }

    # Normalized Advantage (if available)
    if 'advantage_norm' in df.columns:
        stats['advantage_norm'] = {
            'mean': df['advantage_norm'].mean(),
            'std': df['advantage_norm'].std(),
            'min': df['advantage_norm'].min(),
            'max': df['advantage_norm'].max(),
        }

    # Correlation Analysis
    stats['correlations'] = {
        'turnover_vs_advantage': df['turnover_penalty'].corr(df['advantage']),
        'penalty_vs_raw_return': total_penalty.corr(df['raw_return']),
        'advantage_vs_raw_return': df['advantage'].corr(df['raw_return']),
    }

    return stats


def reasonableness_checks(stats: Dict) -> Dict[str, Tuple[bool, str]]:
    """
    Perform reasonableness checks on the computed statistics.

    Returns dict of {check_name: (passed, message)}
    """
    checks = {}

    # Check 1: Penalty ratio should be < 20% on average
    penalty_ratio = stats['penalty_impact']['penalty_to_return_ratio_mean']
    passed = penalty_ratio < 0.20
    checks['penalty_ratio'] = (
        passed,
        f"Penalty/Return ratio = {penalty_ratio:.2%} {'< 20%' if passed else '>= 20% (Too High!)'}"
    )

    # Check 2: Positive advantage ratio should be > 50% (beating baseline)
    adv_pos_ratio = stats['advantage']['positive_ratio']
    passed = adv_pos_ratio > 0.50
    checks['advantage_positive'] = (
        passed,
        f"Positive advantage ratio = {adv_pos_ratio:.2%} {'> 50%' if passed else '<= 50% (Not beating baseline!)'}"
    )

    # Check 3: Average turnover rate should be reasonable (< 80% per rebalance)
    avg_turnover = stats['turnover_rate']['mean']
    passed = avg_turnover < 0.80
    checks['turnover_rate'] = (
        passed,
        f"Avg turnover rate = {avg_turnover:.2%} {'< 80%' if passed else '>= 80% (Excessive trading!)'}"
    )

    # Check 4: Rebalance frequency should be reasonable (5-30% of days)
    rebal_ratio = stats['rebalance_ratio']
    passed = 0.05 <= rebal_ratio <= 0.30
    checks['rebalance_frequency'] = (
        passed,
        f"Rebalance frequency = {rebal_ratio:.2%} {'(5-30% range)' if passed else '(Outside 5-30% range!)'}"
    )

    # Check 5: Advantage std should not be too extreme (indicates instability)
    adv_std = stats['advantage']['std']
    adv_mean = abs(stats['advantage']['mean'])
    # Coefficient of variation check
    cv = adv_std / (adv_mean + 1e-8)
    passed = cv < 10  # Std should not be more than 10x the mean
    checks['advantage_stability'] = (
        passed,
        f"Advantage CoV = {cv:.2f} {'< 10' if passed else '>= 10 (High variance!)'}"
    )

    # Check 6: Net reward should be positive overall
    net_total = stats['penalty_impact']['total_net_reward']
    passed = net_total > 0
    checks['net_positive'] = (
        passed,
        f"Total net reward = {net_total:.4f} {'>0' if passed else '<=0 (Net loss!)'}"
    )

    return checks


def print_report(stats: Dict, checks: Dict):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("PENALTY & ADVANTAGE STATISTICAL ANALYSIS")
    print("=" * 70)

    # Basic Info
    print(f"\n{'SAMPLE INFO':=^70}")
    print(f"  Total samples:        {stats['total_samples']:,}")
    print(f"  Rebalance days:       {stats['rebalance_days']:,} ({stats['rebalance_ratio']:.1%})")
    print(f"  Hold days:            {stats['hold_days']:,}")

    # Turnover Stats
    print(f"\n{'TURNOVER RATE (on rebalance days)':=^70}")
    tr = stats['turnover_rate']
    print(f"  Mean:                 {tr['mean']:.2%}")
    print(f"  Std:                  {tr['std']:.2%}")
    print(f"  Median:               {tr['median']:.2%}")
    print(f"  Min / Max:            {tr['min']:.2%} / {tr['max']:.2%}")
    print(f"  P25 / P75 / P90:      {tr['p25']:.2%} / {tr['p75']:.2%} / {tr['p90']:.2%}")

    # Penalty Impact
    print(f"\n{'PENALTY IMPACT':=^70}")
    pi = stats['penalty_impact']
    print(f"  Total Turnover Pen:   {pi['total_turnover_penalty']:.6f}")
    print(f"  Total Symdiff Pen:    {pi['total_symdiff_penalty']:.6f}")
    print(f"  Total Raw Return:     {pi['total_raw_return']:.6f}")
    print(f"  Total Net Reward:     {pi['total_net_reward']:.6f}")
    print(f"  Penalty/Return Ratio:")
    print(f"    Mean:               {pi['penalty_to_return_ratio_mean']:.2%}")
    print(f"    Median:             {pi['penalty_to_return_ratio_median']:.2%}")
    print(f"    P90:                {pi['penalty_to_return_ratio_p90']:.2%}")

    # Advantage Stats
    print(f"\n{'ADVANTAGE':=^70}")
    adv = stats['advantage']
    print(f"  Mean:                 {adv['mean']:.6f}")
    print(f"  Std:                  {adv['std']:.6f}")
    print(f"  Median:               {adv['median']:.6f}")
    print(f"  Min / Max:            {adv['min']:.6f} / {adv['max']:.6f}")
    print(f"  Positive Ratio:       {adv['positive_ratio']:.2%}")
    print(f"  P10 / P90:            {adv['p10']:.6f} / {adv['p90']:.6f}")

    if 'advantage_norm' in stats:
        print(f"\n  Normalized Advantage:")
        an = stats['advantage_norm']
        print(f"    Mean:               {an['mean']:.4f}")
        print(f"    Std:                {an['std']:.4f}")

    # Correlations
    print(f"\n{'CORRELATIONS':=^70}")
    corr = stats['correlations']
    print(f"  Turnover vs Advantage:    {corr['turnover_vs_advantage']:+.4f}")
    print(f"  Penalty vs Raw Return:    {corr['penalty_vs_raw_return']:+.4f}")
    print(f"  Advantage vs Raw Return:  {corr['advantage_vs_raw_return']:+.4f}")

    # Reasonableness Checks
    print(f"\n{'REASONABLENESS CHECKS':=^70}")
    all_passed = True
    for name, (passed, message) in checks.items():
        status = "PASS" if passed else "FAIL"
        icon = "[OK]" if passed else "[!!]"
        print(f"  {icon} {name}: {message}")
        if not passed:
            all_passed = False

    # Summary
    print(f"\n{'SUMMARY':=^70}")
    if all_passed:
        print("  All checks PASSED. Penalty parameters appear reasonable.")
    else:
        failed = [name for name, (passed, _) in checks.items() if not passed]
        print(f"  FAILED checks: {', '.join(failed)}")
        print("  Consider adjusting alpha_turnover and/or alpha_change parameters.")

    print("=" * 70)


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualization charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Compute turnover_rate
    alpha_turnover = 0.01
    df['turnover_rate'] = df['turnover_penalty'] / alpha_turnover

    # Filter rebalance days
    rebal_df = df[df['turnover_penalty'] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Turnover Rate Distribution
    ax = axes[0, 0]
    if len(rebal_df) > 0:
        ax.hist(rebal_df['turnover_rate'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(rebal_df['turnover_rate'].mean(), color='red', linestyle='--', label=f"Mean: {rebal_df['turnover_rate'].mean():.2%}")
    ax.set_title('Turnover Rate Distribution (Rebalance Days)')
    ax.set_xlabel('Turnover Rate')
    ax.set_ylabel('Frequency')
    ax.legend()

    # 2. Penalty Breakdown
    ax = axes[0, 1]
    total_turn = df['turnover_penalty'].sum()
    total_sym = df['symdiff_penalty'].sum()
    labels = ['Turnover Penalty', 'Symdiff Penalty']
    sizes = [total_turn, total_sym]
    colors = ['#ff9999', '#66b3ff']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Total Penalty Breakdown')

    # 3. Advantage Distribution
    ax = axes[1, 0]
    ax.hist(df['advantage'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=2)
    ax.axvline(df['advantage'].mean(), color='red', linestyle='--', label=f"Mean: {df['advantage'].mean():.4f}")
    ax.set_title('Advantage Distribution')
    ax.set_xlabel('Advantage (R_net - Baseline)')
    ax.set_ylabel('Frequency')
    ax.legend()

    # 4. Turnover vs Advantage Scatter
    ax = axes[1, 1]
    if len(rebal_df) > 0:
        ax.scatter(rebal_df['turnover_rate'], rebal_df['advantage'], alpha=0.5, s=10)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        # Add trend line
        z = np.polyfit(rebal_df['turnover_rate'], rebal_df['advantage'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(rebal_df['turnover_rate'].min(), rebal_df['turnover_rate'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend')
    ax.set_title('Turnover Rate vs Advantage')
    ax.set_xlabel('Turnover Rate')
    ax.set_ylabel('Advantage')
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'penalty_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")

    # Additional: Time series plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Raw return vs Net reward over time
    ax = axes[0]
    ax.plot(df.index, df['raw_return'], label='Raw Return', alpha=0.7)
    ax.plot(df.index, df['net_reward'], label='Net Reward', alpha=0.7)
    ax.set_ylabel('Return')
    ax.legend()
    ax.set_title('Raw Return vs Net Reward Over Time')
    ax.grid(True, alpha=0.3)

    # Cumulative penalties
    ax = axes[1]
    ax.plot(df.index, df['turnover_penalty'].cumsum(), label='Cum. Turnover Penalty')
    ax.plot(df.index, df['symdiff_penalty'].cumsum(), label='Cum. Symdiff Penalty')
    ax.set_ylabel('Cumulative Penalty')
    ax.legend()
    ax.set_title('Cumulative Penalties Over Time')
    ax.grid(True, alpha=0.3)

    # Advantage over time
    ax = axes[2]
    ax.plot(df.index, df['advantage'], alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.fill_between(df.index, 0, df['advantage'], where=df['advantage'] > 0, alpha=0.3, color='green', label='Positive')
    ax.fill_between(df.index, 0, df['advantage'], where=df['advantage'] <= 0, alpha=0.3, color='red', label='Negative')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Advantage')
    ax.legend()
    ax.set_title('Advantage Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 'penalty_timeseries.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time series saved to: {output_path2}")


def main():
    parser = argparse.ArgumentParser(description="Analyze turnover, symdiff penalty & advantage statistics")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./observer_offline/trajectory_details.csv",
        help="Path to trajectory_details.csv"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization charts"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./observer_offline/analysis",
        help="Output directory for plots"
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.input}")
    try:
        df = load_trajectory_data(args.input)
        print(f"Loaded {len(df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Compute statistics
    stats = compute_basic_stats(df)

    # Run checks
    checks = reasonableness_checks(stats)

    # Print report
    print_report(stats, checks)

    # Generate plots if requested
    if args.plot:
        print("\nGenerating visualizations...")
        create_visualizations(df, args.output_dir)

    # Exit with error if any check failed
    if not all(passed for passed, _ in checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
