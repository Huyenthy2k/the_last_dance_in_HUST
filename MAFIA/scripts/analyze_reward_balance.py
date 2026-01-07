#!/usr/bin/env python3
"""
Reward Balance Analysis for SELECTION_ONLY Mode - Simplified Version

This script analyzes reward component distributions on simulated data to help
calibrate reward parameters for medium-term trading behavior.

Usage:
    python scripts/analyze_reward_balance.py \
        --output-dir ./reward_analysis/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config


def analyze_reward_balance(output_dir: str):
    """
    Analyze reward component balance using historical VNINDEX data.

    This simulation calculates the expected magnitudes of each reward component
    based on typical market conditions to help calibrate reward parameters.
    """

    print("=" * 60)
    print("REWARD BALANCE ANALYSIS FOR SELECTION_ONLY MODE")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    config = Config(seed_num=2022, create_dirs=False)

    # Get reward parameters
    scale_factor = float(getattr(config, "scale_factor_reward", 100.0))
    alpha_turnover = float(getattr(config, "mafia_pg_alpha_turnover", 3.0))
    alpha_change = float(getattr(config, "mafia_pg_alpha_change", 3.0))
    alpha_exit = float(getattr(config, "mafia_reward_alpha_exit", 0.5))
    horizon = int(getattr(config, "mafia_pg_reward_horizon", 21))
    rebalance_interval = int(getattr(config, "mafia_rebalance_interval", 21))
    K = int(getattr(config, "topK", 10))
    entropy_coef = float(getattr(config, "entropy_coef", 0.2))

    print(f"\n[CONFIG] Current Reward Parameters:")
    print(f"  - scale_factor_reward: {scale_factor}")
    print(f"  - alpha_turnover: {alpha_turnover}")
    print(f"  - alpha_change: {alpha_change}")
    print(f"  - alpha_exit (r_hold): {alpha_exit}")
    print(f"  - entropy_coef: {entropy_coef}")
    print(f"  - horizon: {horizon} days")
    print(f"  - rebalance_interval: {rebalance_interval} days")
    print(f"  - Top-K: {K}")

    # Load VNINDEX data for realistic simulations
    print("\n[DATA] Loading VNINDEX historical data...")
    from utils.mafia_data_loader import load_mafia_data

    raw_data = load_mafia_data(config)
    stock_list = raw_data["stock"].unique().tolist()
    N_stocks = len(stock_list)
    print(f"[DATA] Found {N_stocks} stocks")

    # Filter to training period
    train_start = pd.Timestamp("2015-01-01")
    train_end = pd.Timestamp("2020-12-31")
    train_data = raw_data[
        (raw_data["date"] >= train_start) & (raw_data["date"] <= train_end)
    ].copy()

    # Pivot to get price matrix
    price_df = train_data.pivot(index="date", columns="stock", values="close")
    price_df = price_df.fillna(method="ffill").fillna(method="bfill")

    # Calculate daily returns
    returns_df = price_df.pct_change().fillna(0)

    # Get market returns (average of all stocks as proxy)
    market_returns = returns_df.mean(axis=1)

    print(f"[DATA] Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")
    print(f"[DATA] Trading days: {len(price_df)}")

    # Simulation parameters
    n_batches = 2  # Number of independent batches
    n_simulations_per_batch = 500  # Rebalance cycles per batch
    target_churn_rates = [0.35, 0.40, 0.45, 0.50, 0.70, 0.90]  # Include high churn rates

    # Alpha values to test for optimal penalty/R_select ratio
    # Include higher values to account for trained model's higher R_select (~4-5x random)
    alpha_values_to_test = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    print(f"\n[SIMULATION] Running {n_batches} batches × {n_simulations_per_batch} simulations...")
    print("-" * 60)

    results = {}
    batch_results = []  # Store results from each batch

    # Run 3 independent batches
    for batch_idx in range(n_batches):
        print(f"\n[BATCH {batch_idx + 1}/{n_batches}]")
        np.random.seed(2022 + batch_idx * 100)  # Different seed per batch

        batch_data = {}

        for target_churn in target_churn_rates:
            # Initialize collectors
            all_R_select = []
            all_baseline = []
            all_r_hold = []
            all_turnover = []
            all_symdiff = []
            all_penalty = []
            all_R_net = []
            all_advantage = []

            # Simulate rebalance cycles
            valid_dates = returns_df.index[horizon:-horizon]
            sample_dates = np.random.choice(len(valid_dates), min(n_simulations_per_batch, len(valid_dates)), replace=False)

            prev_topk = None

            for i, date_idx in enumerate(sample_dates):
                date = valid_dates[date_idx]

                # Select Top-K randomly (simulating Observer selection)
                available_stocks = list(returns_df.columns)
                curr_topk = set(np.random.choice(available_stocks, K, replace=False))

                # === R_SELECT: Future compounded return of selected stocks ===
                future_rets = returns_df.loc[date:].iloc[1:horizon+1]
                if len(future_rets) < horizon:
                    continue

                topk_returns = future_rets[list(curr_topk)].mean(axis=1)
                R_unscaled = (1 + topk_returns).prod() - 1
                R_select = scale_factor * R_unscaled

                # Baseline (market)
                market_fut = market_returns.loc[date:].iloc[1:horizon+1]
                baseline = scale_factor * ((1 + market_fut).prod() - 1)

                all_R_select.append(R_select)
                all_baseline.append(baseline)

                # === TURNOVER & SYMDIFF ===
                if prev_topk is not None:
                    held = curr_topk & prev_topk
                    n_held = len(held)
                    n_changed = K - n_held

                    # SymDiff (normalized 0-2)
                    symdiff = 2 * (K - n_held) / K

                    # Turnover (weight change, 0-1)
                    turnover = n_changed / K * 0.5

                    all_turnover.append(turnover)
                    all_symdiff.append(symdiff)

                    # Penalty with current alpha values
                    penalty = alpha_turnover * turnover + alpha_change * symdiff
                    all_penalty.append(penalty)

                    # R_hold - use CUMULATIVE profit over ACTUAL holding duration
                    # Simulate mix of scheduled vs triggered rebalances:
                    # - ~60% scheduled: duration = rebalance_interval (21 days)
                    # - ~40% triggered (DC event/regime/vol): duration = 5-15 days
                    if np.random.random() < 0.6:
                        actual_duration = rebalance_interval  # Scheduled
                    else:
                        actual_duration = np.random.randint(5, 16)  # Triggered early

                    if n_held > 0:
                        held_list = list(held)
                        # Use ACTUAL duration, not fixed rebalance_interval
                        held_returns = future_rets[held_list].iloc[:actual_duration].mean(axis=1)
                        cumulative_profit = (1 + held_returns).prod() - 1
                        duration_bonus = np.log1p(actual_duration)
                        r_hold = alpha_exit * cumulative_profit * duration_bonus * scale_factor
                    else:
                        r_hold = 0.0
                    all_r_hold.append(r_hold)

                    # R_net and Advantage
                    R_net = R_select + r_hold - penalty
                    all_R_net.append(R_net)
                    advantage = R_net - baseline
                    all_advantage.append(advantage)

                # Simulate realistic churn
                n_to_change = int(K * target_churn) + np.random.randint(-1, 2)
                n_to_change = max(0, min(K, n_to_change))

                if n_to_change > 0:
                    keep = set(np.random.choice(list(curr_topk), K - n_to_change, replace=False))
                    new_stocks = set(np.random.choice(
                        [s for s in available_stocks if s not in curr_topk],
                        n_to_change,
                        replace=False
                    ))
                    prev_topk = keep | new_stocks
                else:
                    prev_topk = curr_topk.copy()

            batch_data[target_churn] = {
                "R_select": np.array(all_R_select),
                "baseline": np.array(all_baseline),
                "r_hold": np.array(all_r_hold) if all_r_hold else np.array([0]),
                "turnover": np.array(all_turnover) if all_turnover else np.array([0]),
                "symdiff": np.array(all_symdiff) if all_symdiff else np.array([0]),
                "penalty": np.array(all_penalty) if all_penalty else np.array([0]),
                "R_net": np.array(all_R_net) if all_R_net else np.array([0]),
                "advantage": np.array(all_advantage) if all_advantage else np.array([0]),
            }

        batch_results.append(batch_data)
        print(f"  Churn 40%: R_select={batch_data[0.40]['R_select'].mean():.3f}, Penalty={batch_data[0.40]['penalty'].mean():.3f}")

    # Aggregate across batches for final results
    for target_churn in target_churn_rates:
        agg = {}
        for key in ["R_select", "baseline", "r_hold", "turnover", "symdiff", "penalty", "R_net", "advantage"]:
            agg[key] = np.concatenate([b[target_churn][key] for b in batch_results])
        results[target_churn] = agg

    # === ALPHA SENSITIVITY ANALYSIS ===
    print("\n" + "=" * 60)
    print("ALPHA SENSITIVITY ANALYSIS (Finding Optimal Penalty/R_select)")
    print("=" * 60)
    print(f"\nTarget: Penalty/R_select = 40-60% (strong enough to encourage holding)")
    print(f"Testing alpha values: {alpha_values_to_test}")
    print("-" * 70)

    # Use 40% churn as reference
    ref_churn = 0.40
    ref_data = results[ref_churn]
    avg_turnover = ref_data["turnover"].mean()
    avg_symdiff = ref_data["symdiff"].mean()
    avg_R_select = abs(ref_data["R_select"].mean())

    print(f"\nReference (40% churn): turnover={avg_turnover:.4f}, symdiff={avg_symdiff:.4f}, |R_select|={avg_R_select:.4f}")
    print(f"\n{'Alpha':>8} | {'Turnover Pen':>12} | {'SymDiff Pen':>12} | {'Total Pen':>10} | {'Pen/R%':>8} | {'Status':>10}")
    print("-" * 75)

    optimal_alpha = None
    optimal_ratio = None

    for test_alpha in alpha_values_to_test:
        turnover_pen = test_alpha * avg_turnover
        symdiff_pen = test_alpha * avg_symdiff
        total_pen = turnover_pen + symdiff_pen
        pen_ratio = total_pen / (avg_R_select + 1e-8) * 100

        if 40 <= pen_ratio <= 60:
            status = "✅ OPTIMAL"
            if optimal_alpha is None:
                optimal_alpha = test_alpha
                optimal_ratio = pen_ratio
        elif pen_ratio < 40:
            status = "⬆️ Too Low"
        else:
            status = "⬇️ Too High"

        print(f"{test_alpha:>8.2f} | {turnover_pen:>12.4f} | {symdiff_pen:>12.4f} | {total_pen:>10.4f} | {pen_ratio:>7.1f}% | {status:>10}")

    if optimal_alpha is None:
        # Find closest to 50% (middle of 40-60% range)
        target_pen = 0.50 * avg_R_select
        optimal_alpha = target_pen / (avg_turnover + avg_symdiff + 1e-8)
        optimal_ratio = 50.0
        print(f"\n⚠️  No alpha in test range achieves 40-60%")
        print(f"   Calculated optimal alpha for 50%: {optimal_alpha:.4f}")
    else:
        print(f"\n✅ Optimal alpha found: {optimal_alpha:.2f} (Penalty/R_select = {optimal_ratio:.1f}%)")

    # === CORRELATION ANALYSIS ===
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS (r_hold vs R_select)")
    print("=" * 60)

    r_select_arr = ref_data["R_select"]
    r_hold_arr = ref_data["r_hold"]
    penalty_arr = ref_data["penalty"]
    r_net_arr = ref_data["R_net"]
    baseline_arr = ref_data["baseline"]

    # Compute correlations (filter NaN and ensure same length)
    min_len = min(len(r_select_arr), len(r_hold_arr), len(penalty_arr), len(baseline_arr))
    r_select_arr = r_select_arr[:min_len]
    r_hold_arr = r_hold_arr[:min_len]
    penalty_arr = penalty_arr[:min_len]
    baseline_arr = baseline_arr[:min_len]

    # Remove NaN values
    valid_mask = ~(np.isnan(r_select_arr) | np.isnan(r_hold_arr) | np.isnan(penalty_arr) | np.isnan(baseline_arr))
    r_select_arr = r_select_arr[valid_mask]
    r_hold_arr = r_hold_arr[valid_mask]
    penalty_arr = penalty_arr[valid_mask]
    baseline_arr = baseline_arr[valid_mask]

    if len(r_select_arr) > 1 and len(r_hold_arr) > 1:
        corr_hold_select = np.corrcoef(r_hold_arr, r_select_arr)[0, 1]
        corr_hold_penalty = np.corrcoef(r_hold_arr, penalty_arr)[0, 1]
        corr_select_penalty = np.corrcoef(r_select_arr, penalty_arr)[0, 1]
        corr_select_baseline = np.corrcoef(r_select_arr, baseline_arr)[0, 1]

        print(f"\n📈 Correlation Matrix:")
        print(f"   r_hold ↔ R_select:   {corr_hold_select:>7.4f}")
        print(f"   r_hold ↔ Penalty:    {corr_hold_penalty:>7.4f}")
        print(f"   R_select ↔ Penalty:  {corr_select_penalty:>7.4f}")
        print(f"   R_select ↔ Baseline: {corr_select_baseline:>7.4f}")

        # Interpretation
        print(f"\n💡 Interpretation:")
        if abs(corr_hold_select) < 0.3:
            print(f"   ✅ r_hold has LOW correlation with R_select ({corr_hold_select:.2f})")
            print(f"      → r_hold provides COMPLEMENTARY signal (good!)")
        elif abs(corr_hold_select) < 0.6:
            print(f"   ⚠️  r_hold has MODERATE correlation with R_select ({corr_hold_select:.2f})")
            print(f"      → r_hold adds some unique information")
        else:
            print(f"   ❌ r_hold has HIGH correlation with R_select ({corr_hold_select:.2f})")
            print(f"      → r_hold may be REDUNDANT, consider reducing alpha_exit")

        # Contribution analysis
        r_hold_contribution = abs(r_hold_arr.mean()) / (abs(r_select_arr.mean()) + 1e-8) * 100
        print(f"\n📊 Contribution Analysis:")
        print(f"   |r_hold| / |R_select|: {r_hold_contribution:.1f}%")

        if r_hold_contribution < 30:
            print(f"   ⚠️  r_hold contribution TOO LOW → increase alpha_exit")
        elif r_hold_contribution > 50:
            print(f"   ⚠️  r_hold contribution TOO HIGH → decrease alpha_exit")
        else:
            print(f"   ✅ r_hold contribution in target range (30-50%) for 40:60 balance")

    # === PRINT RESULTS ===
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS (by Target Churn Rate)")
    print("=" * 60)

    def print_stats(arr, name, indent="  "):
        if len(arr) == 0:
            return f"{indent}{name}: No data"
        return (
            f"{indent}{name}:\n"
            f"{indent}  Mean: {arr.mean():.4f} ± {arr.std():.4f}\n"
            f"{indent}  Min: {arr.min():.4f}, Max: {arr.max():.4f}\n"
            f"{indent}  P5: {np.percentile(arr, 5):.4f}, P50: {np.percentile(arr, 50):.4f}, P95: {np.percentile(arr, 95):.4f}"
        )

    # Focus on target churn = 40% (moderate, as user selected)
    target = 0.4
    r = results[target]

    print(f"\n*** TARGET CHURN RATE: {target*100:.0f}% (Moderate - User Selected) ***")
    print("-" * 50)

    print("\n--- R_SELECT (Prospective Return) ---")
    print(print_stats(r["R_select"], "R_select"))

    print("\n--- BASELINE (Market Return) ---")
    print(print_stats(r["baseline"], "baseline"))

    print("\n--- R_HOLD (Retrospective Hold Reward) ---")
    print(print_stats(r["r_hold"], "r_hold"))
    if len(r["R_select"]) > 0 and len(r["r_hold"]) > 0:
        r_hold_pct = abs(r["r_hold"].mean()) / (abs(r["R_select"].mean()) + 1e-8) * 100
        print(f"    r_hold / |R_select|: {r_hold_pct:.1f}%")

    print("\n--- TURNOVER ---")
    print(print_stats(r["turnover"], "turnover"))

    print("\n--- SYMDIFF (Membership Change) ---")
    print(print_stats(r["symdiff"], "symdiff"))

    print("\n--- PENALTY (α_turn × turnover + α_change × symdiff) ---")
    print(print_stats(r["penalty"], "penalty"))
    if len(r["R_select"]) > 0 and len(r["penalty"]) > 0:
        penalty_pct = r["penalty"].mean() / (abs(r["R_select"].mean()) + 1e-8) * 100
        print(f"    Penalty / |R_select|: {penalty_pct:.1f}%")

    print("\n--- R_NET (R_select + r_hold - penalty) ---")
    print(print_stats(r["R_net"], "R_net"))

    print("\n--- ADVANTAGE (R_net - baseline) ---")
    print(print_stats(r["advantage"], "Advantage"))
    if len(r["advantage"]) > 1:
        skew = scipy_stats.skew(r["advantage"])
        print(f"    Skewness: {skew:.4f}")

    # === COMPARISON ACROSS CHURN RATES ===
    print("\n" + "=" * 60)
    print("COMPARISON ACROSS CHURN RATES")
    print("=" * 60)
    print(f"\n{'Churn%':>8} | {'R_select':>10} | {'Penalty':>10} | {'Pen/R%':>8} | {'R_net':>10} | {'Adv_std':>8}")
    print("-" * 70)

    for churn, r in sorted(results.items()):
        r_sel = r["R_select"].mean() if len(r["R_select"]) > 0 else 0
        pen = r["penalty"].mean() if len(r["penalty"]) > 0 else 0
        pen_pct = pen / (abs(r_sel) + 1e-8) * 100
        r_net = r["R_net"].mean() if len(r["R_net"]) > 0 else 0
        adv_std = r["advantage"].std() if len(r["advantage"]) > 0 else 0

        print(f"{churn*100:>7.0f}% | {r_sel:>10.4f} | {pen:>10.4f} | {pen_pct:>7.1f}% | {r_net:>10.4f} | {adv_std:>8.4f}")

    # === RECOMMENDATIONS ===
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATIONS FOR BALANCED REWARD PARAMETERS")
    print("=" * 60)

    r = results[0.4]  # Use 40% churn target
    r_sel_mean = r["R_select"].mean() if len(r["R_select"]) > 0 else 1.0
    pen_mean = r["penalty"].mean() if len(r["penalty"]) > 0 else 0
    adv_std = r["advantage"].std() if len(r["advantage"]) > 0 else 0
    r_hold_mean = r["r_hold"].mean() if len(r["r_hold"]) > 0 else 0

    penalty_ratio = pen_mean / (abs(r_sel_mean) + 1e-8)

    print(f"\n📊 Current Analysis (40% Churn, {n_batches} batches, {n_batches * n_simulations_per_batch} total simulations):")
    print(f"   |R_select| mean: {abs(r_sel_mean):.4f}")
    print(f"   Penalty mean (current α={alpha_turnover}): {pen_mean:.4f}")
    print(f"   Penalty/|R_select|: {penalty_ratio*100:.1f}% (target: 40-60%)")
    print(f"   r_hold mean: {r_hold_mean:.4f}")
    print(f"   Advantage std: {adv_std:.4f}")

    # Use optimal alpha from sensitivity analysis
    if optimal_alpha is not None:
        rec_alpha = optimal_alpha
    else:
        rec_alpha = 0.20 * avg_R_select / (avg_turnover + avg_symdiff + 1e-8)

    # Calculate expected penalty with new alpha
    new_pen = rec_alpha * (avg_turnover + avg_symdiff)
    new_ratio = new_pen / (avg_R_select + 1e-8) * 100

    print(f"\n🎯 Optimal Alpha from Sensitivity Analysis: {rec_alpha:.4f}")
    print(f"   Expected Penalty with α={rec_alpha:.2f}: {new_pen:.4f}")
    print(f"   Expected Penalty/|R_select|: {new_ratio:.1f}%")

    # Final recommendations
    print("\n" + "-" * 60)
    print("📝 FINAL RECOMMENDED PARAMETERS (Based on 3-Batch Analysis):")
    print("-" * 60)

    # Round to reasonable values
    rec_alpha_turn = round(rec_alpha, 2)
    rec_alpha_change = round(rec_alpha, 2)
    rec_alpha_exit = alpha_exit  # Keep current value for 40:60 r_hold:R_select balance

    print(f"   scale_factor_reward:     {scale_factor:.0f} → {scale_factor:.0f} (keep)")
    print(f"   mafia_pg_alpha_turnover: {alpha_turnover:.1f} → {rec_alpha_turn:.2f}")
    print(f"   mafia_pg_alpha_change:   {alpha_change:.1f} → {rec_alpha_change:.2f}")
    print(f"   mafia_reward_alpha_exit: {alpha_exit:.1f} → {rec_alpha_exit:.2f}")
    print(f"   entropy_coef:            {entropy_coef:.2f} → 0.15")

    # Validation
    print(f"\n📋 VALIDATION:")
    print(f"   - At 40% churn: Penalty = {rec_alpha_turn} × {avg_turnover:.4f} + {rec_alpha_change} × {avg_symdiff:.4f}")
    print(f"                         = {rec_alpha_turn * avg_turnover:.4f} + {rec_alpha_change * avg_symdiff:.4f}")
    print(f"                         = {new_pen:.4f}")
    print(f"   - Penalty/|R_select|  = {new_pen:.4f} / {avg_R_select:.4f} = {new_ratio:.1f}% ✅")

    # Save results
    results_file = os.path.join(output_dir, "reward_analysis_results.txt")
    with open(results_file, "w") as f:
        f.write("REWARD BALANCE ANALYSIS RESULTS (3-BATCH QUANTITATIVE ANALYSIS)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Simulation: {n_batches} batches × {n_simulations_per_batch} simulations = {n_batches * n_simulations_per_batch} total\n\n")
        f.write(f"Current Parameters:\n")
        f.write(f"  scale_factor_reward: {scale_factor}\n")
        f.write(f"  alpha_turnover: {alpha_turnover}\n")
        f.write(f"  alpha_change: {alpha_change}\n")
        f.write(f"  alpha_exit: {alpha_exit}\n\n")
        f.write(f"Analysis (40% Churn):\n")
        f.write(f"  |R_select| mean: {abs(r_sel_mean):.4f}\n")
        f.write(f"  turnover mean: {avg_turnover:.4f}\n")
        f.write(f"  symdiff mean: {avg_symdiff:.4f}\n")
        f.write(f"  Penalty mean: {pen_mean:.4f}\n")
        f.write(f"  Penalty/|R_select|: {penalty_ratio*100:.1f}% (target: 40-60%)\n")
        f.write(f"  Advantage std: {adv_std:.4f}\n\n")
        f.write(f"Optimal Alpha from Sensitivity Analysis: {rec_alpha:.4f}\n")
        f.write(f"Expected Penalty with new alpha: {new_pen:.4f}\n")
        f.write(f"Expected Penalty/|R_select|: {new_ratio:.1f}%\n\n")
        f.write(f"RECOMMENDED PARAMETERS:\n")
        f.write(f"  scale_factor_reward: {scale_factor:.0f} (unchanged)\n")
        f.write(f"  mafia_pg_alpha_turnover: {rec_alpha_turn:.2f}\n")
        f.write(f"  mafia_pg_alpha_change: {rec_alpha_change:.2f}\n")
        f.write(f"  mafia_reward_alpha_exit: {rec_alpha_exit:.2f}\n")
        f.write(f"  entropy_coef: 0.15\n")

    print(f"\n[SAVED] Results saved to: {results_file}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Reward Balance Analysis")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reward_analysis/",
        help="Directory to save analysis results",
    )

    args = parser.parse_args()

    analyze_reward_balance(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
