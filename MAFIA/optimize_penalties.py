#!/usr/bin/env python3
"""
Quantitative Optimization for Penalty Parameters (alpha_turnover, alpha_change)

This script performs a grid search to find optimal penalty coefficients that:
1. Minimize turnover rate (reduce excessive trading)
2. Maximize positive advantage ratio (beat baseline consistently)
3. Keep penalty/return ratio reasonable (<20%)

Methodology:
- Grid search over (alpha_turnover, alpha_change) parameter space
- Short training runs (1 epoch) per configuration
- Collect metrics and compute composite score
- Generate Pareto frontier analysis

Usage:
    python optimize_penalties.py --epochs 1 --output-dir ./penalty_optimization
"""

import argparse
import os
import sys
import json
import itertools
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def run_single_config(
    alpha_turnover: float,
    alpha_change: float,
    epochs: int = 1,
    output_base: str = "./penalty_optimization",
    seed: int = 2025,
) -> Dict:
    """
    Run training with specific penalty parameters and collect metrics.

    Returns dict with configuration and resulting metrics.
    """
    import torch as th
    from config import Config
    from RL_controller.mafia_observer import MAFIAObserver
    from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
    from utils.mafia_data_loader import load_mafia_data

    # Set seeds
    np.random.seed(seed)
    th.manual_seed(seed)

    # Create config with specified alphas
    config = Config()
    config.mafia_pg_alpha_turnover = alpha_turnover
    config.mafia_pg_alpha_change = alpha_change
    config.seed = seed

    # Output directory for this config
    config_name = f"alpha_t{alpha_turnover:.3f}_c{alpha_change:.3f}"
    output_dir = os.path.join(output_base, config_name)
    os.makedirs(output_dir, exist_ok=True)
    config.res_root = output_dir
    config.log_trajectory_details = True

    # Load data
    try:
        stock_data = load_mafia_data(config)
        stock_list = sorted(stock_data["stock"].unique().tolist())

        # Load market data
        index_file = os.path.join(getattr(config, "dataDir", "./data"), "vnindex_data.csv")
        market_data = pd.read_csv(index_file, parse_dates=["date"]) if os.path.exists(index_file) else None
    except Exception as e:
        return {"error": str(e), "alpha_turnover": alpha_turnover, "alpha_change": alpha_change}

    # Initialize model and trainer
    if th.cuda.is_available():
        device = th.device("cuda")
    elif th.backends.mps.is_available():
        device = th.device("mps")
    else:
        device = th.device("cpu")
    action_dim = len(stock_list)
    observer = MAFIAObserver(config=config, action_dim=action_dim)
    trainer = ObserverOfflineBatchTrainer(config=config, observer=observer, device=device)

    # Prepare data tensors
    full_start = pd.Timestamp("2015-01-01")
    full_end = pd.Timestamp("2017-12-31")  # Short period for optimization

    data_tensors = trainer.prepare_data_tensors(
        data=stock_data,
        stock_list=stock_list,
        start_date=full_start,
        end_date=full_end,
        market_data=market_data,
    )

    # Train for specified epochs
    for epoch in range(epochs):
        trainer.train_epoch(
            data_tensors=data_tensors,
            steps_per_epoch=3,  # Very short run for optimization (was 10)
        )

    # Collect trajectory details
    traj_file = os.path.join(output_dir, "trajectory_details.csv")
    if os.path.exists(traj_file):
        df = pd.read_csv(traj_file)
        metrics = compute_metrics(df, alpha_turnover, alpha_change)
    else:
        # Fallback: compute from trainer's internal log
        if hasattr(trainer, '_trajectory_log') and trainer._trajectory_log:
            df = pd.DataFrame(trainer._trajectory_log)
            df.to_csv(traj_file, index=False)
            metrics = compute_metrics(df, alpha_turnover, alpha_change)
        else:
            metrics = {"error": "No trajectory data"}

    # Add config info
    metrics["alpha_turnover"] = alpha_turnover
    metrics["alpha_change"] = alpha_change
    metrics["config_name"] = config_name

    return metrics


def compute_metrics(df: pd.DataFrame, alpha_turnover: float, alpha_change: float) -> Dict:
    """Compute optimization metrics from trajectory data."""

    if df.empty:
        return {"error": "Empty dataframe"}

    # Check required columns
    required = ['raw_return', 'turnover_penalty', 'symdiff_penalty', 'advantage', 'net_reward']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}

    # Compute turnover rate from penalty
    df['turnover_rate'] = df['turnover_penalty'] / (alpha_turnover + 1e-10)

    # Filter rebalance days
    rebal_df = df[df['turnover_penalty'] > 0]

    # Metrics
    total_samples = len(df)
    rebalance_days = len(rebal_df)

    # Turnover statistics
    avg_turnover = rebal_df['turnover_rate'].mean() if len(rebal_df) > 0 else 0

    # Penalty impact
    total_penalty = df['turnover_penalty'].sum() + df['symdiff_penalty'].sum()
    total_raw_return = df['raw_return'].sum()
    total_net_reward = df['net_reward'].sum()

    # Penalty/Return ratio (handle division by zero)
    abs_raw = df['raw_return'].abs()
    valid_mask = abs_raw > 1e-8
    penalty_ratios = np.where(
        valid_mask,
        (df['turnover_penalty'] + df['symdiff_penalty']) / abs_raw,
        0
    )
    avg_penalty_ratio = penalty_ratios[valid_mask].mean() if valid_mask.sum() > 0 else 0

    # Advantage statistics
    positive_adv_ratio = (df['advantage'] > 0).mean()
    avg_advantage = df['advantage'].mean()

    return {
        "total_samples": total_samples,
        "rebalance_days": rebalance_days,
        "rebalance_ratio": rebalance_days / total_samples if total_samples > 0 else 0,
        "avg_turnover_rate": avg_turnover,
        "avg_penalty_ratio": avg_penalty_ratio,
        "positive_advantage_ratio": positive_adv_ratio,
        "avg_advantage": avg_advantage,
        "total_raw_return": total_raw_return,
        "total_net_reward": total_net_reward,
        "total_penalty": total_penalty,
    }


def compute_composite_score(metrics: Dict, weights: Dict = None) -> float:
    """
    Compute composite optimization score.

    Objectives:
    1. Minimize turnover rate (target < 0.8)
    2. Maximize positive advantage ratio (target > 0.5)
    3. Minimize penalty ratio (target < 0.2)
    4. Maximize net reward

    Higher score = better configuration.
    """
    if "error" in metrics:
        return -999.0

    if weights is None:
        weights = {
            "turnover": 0.25,      # Penalize high turnover
            "penalty_ratio": 0.25, # Penalize high penalty/return ratio
            "advantage": 0.30,     # Reward high positive advantage ratio
            "net_reward": 0.20,    # Reward positive net returns
        }

    # Turnover score: 1.0 if turnover < 0.5, 0.0 if turnover > 1.0
    turnover = metrics.get("avg_turnover_rate", 1.0)
    turnover_score = max(0, 1 - (turnover - 0.5) / 0.5) if turnover > 0.5 else 1.0

    # Penalty ratio score: 1.0 if ratio < 0.1, 0.0 if ratio > 0.5
    penalty_ratio = metrics.get("avg_penalty_ratio", 1.0)
    penalty_score = max(0, 1 - (penalty_ratio - 0.1) / 0.4) if penalty_ratio > 0.1 else 1.0

    # Advantage score: positive_advantage_ratio directly (0-1)
    adv_ratio = metrics.get("positive_advantage_ratio", 0.0)
    advantage_score = adv_ratio

    # Net reward score: normalize by total raw return
    net_reward = metrics.get("total_net_reward", 0.0)
    raw_return = metrics.get("total_raw_return", 1.0)
    net_reward_score = max(0, min(1, net_reward / (abs(raw_return) + 1e-8)))

    # Weighted composite
    score = (
        weights["turnover"] * turnover_score +
        weights["penalty_ratio"] * penalty_score +
        weights["advantage"] * advantage_score +
        weights["net_reward"] * net_reward_score
    )

    return score


def run_grid_search(
    alpha_turnover_range: List[float],
    alpha_change_range: List[float],
    epochs: int = 1,
    output_dir: str = "./penalty_optimization",
) -> pd.DataFrame:
    """
    Run grid search over parameter space.
    """
    results = []
    total_configs = len(alpha_turnover_range) * len(alpha_change_range)

    print(f"\n{'='*70}")
    print("PENALTY PARAMETER OPTIMIZATION - GRID SEARCH")
    print(f"{'='*70}")
    print(f"  alpha_turnover range: {alpha_turnover_range}")
    print(f"  alpha_change range:   {alpha_change_range}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Epochs per config:    {epochs}")
    print(f"{'='*70}\n")

    for i, (alpha_t, alpha_c) in enumerate(itertools.product(alpha_turnover_range, alpha_change_range)):
        print(f"[{i+1}/{total_configs}] Running alpha_turnover={alpha_t:.3f}, alpha_change={alpha_c:.3f}...")

        try:
            metrics = run_single_config(
                alpha_turnover=alpha_t,
                alpha_change=alpha_c,
                epochs=epochs,
                output_base=output_dir,
            )

            # Compute composite score
            score = compute_composite_score(metrics)
            metrics["composite_score"] = score

            results.append(metrics)

            if "error" not in metrics:
                print(f"    Turnover: {metrics['avg_turnover_rate']:.2%}, "
                      f"Penalty Ratio: {metrics['avg_penalty_ratio']:.2%}, "
                      f"Adv+: {metrics['positive_advantage_ratio']:.2%}, "
                      f"Score: {score:.4f}")
            else:
                print(f"    ERROR: {metrics['error']}")

        except Exception as e:
            print(f"    EXCEPTION: {e}")
            results.append({
                "alpha_turnover": alpha_t,
                "alpha_change": alpha_c,
                "error": str(e),
                "composite_score": -999.0,
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


def find_optimal_config(results_df: pd.DataFrame) -> Dict:
    """Find the optimal configuration from grid search results."""

    # Filter out errors
    valid_df = results_df[~results_df.get('error', pd.Series([None]*len(results_df))).notna()].copy()

    if len(valid_df) == 0:
        # Fallback: use composite_score even with errors
        valid_df = results_df[results_df['composite_score'] > -900].copy()

    if len(valid_df) == 0:
        return {"error": "No valid configurations found"}

    # Find best by composite score
    best_idx = valid_df['composite_score'].idxmax()
    best_config = valid_df.loc[best_idx].to_dict()

    return best_config


def generate_optimization_report(results_df: pd.DataFrame, output_dir: str):
    """Generate optimization report with visualizations."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Filter valid results
    valid_df = results_df[results_df['composite_score'] > -900].copy()

    if len(valid_df) == 0:
        print("No valid results to visualize")
        return

    # 1. Heatmap of composite scores
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Pivot for heatmap
    pivot_score = valid_df.pivot_table(
        values='composite_score',
        index='alpha_turnover',
        columns='alpha_change',
        aggfunc='mean'
    )

    ax = axes[0, 0]
    im = ax.imshow(pivot_score.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(pivot_score.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in pivot_score.columns])
    ax.set_yticks(range(len(pivot_score.index)))
    ax.set_yticklabels([f"{y:.3f}" for y in pivot_score.index])
    ax.set_xlabel('alpha_change')
    ax.set_ylabel('alpha_turnover')
    ax.set_title('Composite Score Heatmap')
    plt.colorbar(im, ax=ax)

    # Mark best config
    best_idx = valid_df['composite_score'].idxmax()
    best = valid_df.loc[best_idx]
    ax.scatter(
        list(pivot_score.columns).index(best['alpha_change']),
        list(pivot_score.index).index(best['alpha_turnover']),
        marker='*', s=300, c='blue', edgecolors='white', linewidths=2
    )

    # 2. Turnover rate vs alpha_turnover
    ax = axes[0, 1]
    for alpha_c in valid_df['alpha_change'].unique():
        subset = valid_df[valid_df['alpha_change'] == alpha_c]
        ax.plot(subset['alpha_turnover'], subset['avg_turnover_rate'],
                marker='o', label=f'alpha_c={alpha_c:.3f}')
    ax.axhline(0.8, color='red', linestyle='--', label='Target < 80%')
    ax.set_xlabel('alpha_turnover')
    ax.set_ylabel('Avg Turnover Rate')
    ax.set_title('Turnover Rate vs alpha_turnover')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Positive Advantage Ratio
    ax = axes[1, 0]
    for alpha_c in valid_df['alpha_change'].unique():
        subset = valid_df[valid_df['alpha_change'] == alpha_c]
        ax.plot(subset['alpha_turnover'], subset['positive_advantage_ratio'],
                marker='s', label=f'alpha_c={alpha_c:.3f}')
    ax.axhline(0.5, color='green', linestyle='--', label='Target > 50%')
    ax.set_xlabel('alpha_turnover')
    ax.set_ylabel('Positive Advantage Ratio')
    ax.set_title('Advantage Ratio vs alpha_turnover')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Pareto frontier (Turnover vs Advantage)
    ax = axes[1, 1]
    scatter = ax.scatter(
        valid_df['avg_turnover_rate'],
        valid_df['positive_advantage_ratio'],
        c=valid_df['composite_score'],
        cmap='RdYlGn',
        s=100,
        edgecolors='black'
    )
    ax.scatter(
        best['avg_turnover_rate'],
        best['positive_advantage_ratio'],
        marker='*', s=300, c='blue', edgecolors='white', linewidths=2,
        label='Best Config'
    )
    ax.axvline(0.8, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Avg Turnover Rate')
    ax.set_ylabel('Positive Advantage Ratio')
    ax.set_title('Pareto Analysis: Turnover vs Advantage')
    plt.colorbar(scatter, ax=ax, label='Composite Score')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {os.path.join(output_dir, 'optimization_results.png')}")


def main():
    parser = argparse.ArgumentParser(description="Optimize penalty parameters via grid search")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per config")
    parser.add_argument("--output-dir", type=str, default="./penalty_optimization", help="Output directory")
    parser.add_argument("--alpha-turnover-min", type=float, default=0.01, help="Min alpha_turnover")
    parser.add_argument("--alpha-turnover-max", type=float, default=0.15, help="Max alpha_turnover")
    parser.add_argument("--alpha-change-min", type=float, default=0.01, help="Min alpha_change")
    parser.add_argument("--alpha-change-max", type=float, default=0.15, help="Max alpha_change")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid points per dimension")
    args = parser.parse_args()

    # Generate parameter ranges
    alpha_turnover_range = np.linspace(args.alpha_turnover_min, args.alpha_turnover_max, args.grid_size).tolist()
    alpha_change_range = np.linspace(args.alpha_change_min, args.alpha_change_max, args.grid_size).tolist()

    # Run grid search
    results_df = run_grid_search(
        alpha_turnover_range=alpha_turnover_range,
        alpha_change_range=alpha_change_range,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )

    # Save results
    results_file = os.path.join(args.output_dir, "grid_search_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Find optimal config
    optimal = find_optimal_config(results_df)

    print(f"\n{'='*70}")
    print("OPTIMAL CONFIGURATION")
    print(f"{'='*70}")
    if "error" not in optimal:
        print(f"  alpha_turnover:           {optimal.get('alpha_turnover', 'N/A'):.4f}")
        print(f"  alpha_change:             {optimal.get('alpha_change', 'N/A'):.4f}")
        print(f"  Composite Score:          {optimal.get('composite_score', 'N/A'):.4f}")
        print(f"")
        print(f"  Avg Turnover Rate:        {optimal.get('avg_turnover_rate', 0):.2%}")
        print(f"  Avg Penalty Ratio:        {optimal.get('avg_penalty_ratio', 0):.2%}")
        print(f"  Positive Advantage Ratio: {optimal.get('positive_advantage_ratio', 0):.2%}")
        print(f"  Total Net Reward:         {optimal.get('total_net_reward', 0):.6f}")
    else:
        print(f"  ERROR: {optimal['error']}")
    print(f"{'='*70}")

    # Save optimal config
    optimal_file = os.path.join(args.output_dir, "optimal_config.json")
    with open(optimal_file, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in optimal.items()}, f, indent=2)
    print(f"\nOptimal config saved to: {optimal_file}")

    # Generate visualizations
    generate_optimization_report(results_df, args.output_dir)

    # Print recommended config.py update
    if "error" not in optimal:
        print(f"\n{'='*70}")
        print("RECOMMENDED CONFIG UPDATE")
        print(f"{'='*70}")
        print(f"# In config.py, update these values:")
        print(f"self.mafia_pg_alpha_turnover = {optimal.get('alpha_turnover', 0.01):.4f}")
        print(f"self.mafia_pg_alpha_change = {optimal.get('alpha_change', 0.01):.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
