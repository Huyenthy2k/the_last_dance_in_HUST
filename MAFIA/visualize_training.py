#!/usr/bin/env python3
"""
Visualize walk-forward training metrics with trend analysis.

Features:
- Aggregates metrics across multiple walk-forward windows
- Shows train/valid/test phases with distinct colors
- Adds trend lines (moving average) for better visualization
- Organizes metrics into logical groups (Performance, Risk, Loss)
- Supports both single-window and multi-window visualization
"""
import argparse
import glob
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# ============================================================================
# Metric Groups for organized visualization
# ============================================================================
PERFORMANCE_METRICS = ["ces_score", "reward_sum", "final_capital", "sharpeRatio", "annualReturn_pct", "netProfit_pct"]
RISK_METRICS = ["risk_mse", "risk_correlation", "topk_volatility"]
LOSS_METRICS = ["mafia_loss", "mafia_pg_loss", "mafia_risk_loss", "mafia_direction_loss", "td3_actor_loss", "td3_critic_loss"]

# Phase styling
PHASE_COLORS = {
    "train": "#2196F3",   # Blue
    "valid": "#4CAF50",   # Green
    "test": "#FF5722",    # Orange-Red
}
PHASE_MARKERS = {
    "train": "o",
    "valid": "s",
    "test": "^",
}
PHASE_ALPHA = {
    "train": 0.6,
    "valid": 0.8,
    "test": 1.0,
}


def find_all_metrics_files(base_dir: str = "res", pattern: str = "**/metrics_history.csv") -> List[str]:
    """Find all metrics_history.csv files in directory tree."""
    search_path = os.path.join(base_dir, pattern)
    files = glob.glob(search_path, recursive=True)
    # Sort by modification time (newest first)
    return sorted(files, key=lambda f: os.path.getmtime(f), reverse=True)


def find_walkforward_metrics(base_dir: str) -> Dict[int, str]:
    """
    Find metrics files for walk-forward windows.
    Returns dict mapping window_idx -> metrics_file_path.
    """
    all_files = find_all_metrics_files(base_dir)
    window_files = {}

    for f in all_files:
        # Extract window index from path like "..._win0_seed..." or "..._win1_..."
        dirname = os.path.dirname(f)
        parts = os.path.basename(dirname).split("_")
        for part in parts:
            if part.startswith("win"):
                try:
                    win_idx = int(part[3:])
                    if win_idx not in window_files:
                        window_files[win_idx] = f
                except ValueError:
                    continue

    return dict(sorted(window_files.items()))


def load_and_merge_windows(window_files: Dict[int, str]) -> pd.DataFrame:
    """Load and merge metrics from multiple walk-forward windows."""
    dfs = []
    global_epoch = 0

    for win_idx, filepath in sorted(window_files.items()):
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                continue

            df["window"] = win_idx
            # Create global epoch numbering across windows
            max_epoch_in_window = df["epoch"].max()
            df["global_epoch"] = df["epoch"] + global_epoch
            global_epoch += max_epoch_in_window

            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def add_trend_line(ax, x: np.ndarray, y: np.ndarray, color: str, window: int = 3, label: str = None):
    """Add smoothed trend line using moving average."""
    if len(y) < window:
        return

    # Simple moving average
    y_smooth = pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().values
    ax.plot(x, y_smooth, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)


def plot_metric_group(
    df: pd.DataFrame,
    metrics: List[str],
    group_title: str,
    output_path: str,
    show_trend: bool = True,
    trend_window: int = 3,
    exclude_phases: Optional[List[str]] = None,
):
    """Plot a group of related metrics with walk-forward window annotations."""
    # Filter to metrics that exist in dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        print(f"⚠️  No metrics found for group '{group_title}', skipping.")
        return

    n = len(available_metrics)
    cols = min(2, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    phases = sorted(df["phase"].unique())
    windows = sorted(df["window"].unique()) if "window" in df.columns else [0]
    use_global_epoch = "global_epoch" in df.columns and len(windows) > 1
    x_col = "global_epoch" if use_global_epoch else "epoch"

    for idx, metric in enumerate(available_metrics):
        ax = axes_flat[idx]

        for phase in phases:
            if exclude_phases and phase in exclude_phases:
                continue
            phase_df = df[df["phase"] == phase].dropna(subset=[metric])
            if phase_df.empty:
                continue

            x_vals = phase_df[x_col].values
            y_vals = phase_df[metric].values

            # Main scatter plot
            ax.scatter(
                x_vals, y_vals,
                color=PHASE_COLORS.get(phase, "gray"),
                marker=PHASE_MARKERS.get(phase, "o"),
                alpha=PHASE_ALPHA.get(phase, 0.7),
                s=40,
                label=phase,
                edgecolors="white",
                linewidths=0.5,
            )

            # Trend line
            if show_trend and len(y_vals) >= trend_window:
                add_trend_line(ax, x_vals, y_vals, PHASE_COLORS.get(phase, "gray"), trend_window)

        # Add vertical lines for window boundaries
        if use_global_epoch and len(windows) > 1:
            for win_idx in windows[1:]:
                win_start = df[df["window"] == win_idx][x_col].min()
                if pd.notna(win_start):
                    ax.axvline(x=win_start, color="gray", linestyle=":", alpha=0.5, linewidth=1)
                    ax.text(
                        win_start, ax.get_ylim()[1], f"W{win_idx}",
                        fontsize=8, color="gray", ha="left", va="top", rotation=90
                    )

        # Format axis
        ax.set_title(_format_metric_name(metric), fontsize=11, fontweight="bold")
        ax.set_xlabel("Global Epoch" if use_global_epoch else "Epoch")
        ax.set_ylabel(_format_metric_name(metric))
        ax.grid(True, linestyle="--", alpha=0.3)

        # Y-axis formatting for specific metrics
        if metric in ["final_capital"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.2f}M"))
        elif metric in ["mdd", "volatility", "annualReturn_pct", "netProfit_pct", "topk_volatility"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.1f}%"))

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.axis("off")

    # Legend
    legend_elements = [
        Line2D([0], [0], marker=PHASE_MARKERS[p], color="w", markerfacecolor=PHASE_COLORS[p],
               markersize=10, label=p.capitalize())
        for p in phases if p in PHASE_COLORS
    ]
    if show_trend:
        legend_elements.append(Line2D([0], [0], linestyle="--", color="gray", label="Trend"))

    fig.legend(handles=legend_elements, loc="upper center", ncol=len(legend_elements), fontsize=10)
    fig.suptitle(f"{group_title} - Walk-Forward Training", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Saved {group_title} plot to {output_path}")


def _format_metric_name(metric: str) -> str:
    """Format metric name for display."""
    replacements = {
        "reward_sum": "Cumulative Reward",
        "final_capital": "Final Capital",
        "ces_score": "Composite Efficiency Score (CES)",
        "sharpeRatio": "Sharpe Ratio",
        "annualReturn_pct": "Annual Return",
        "netProfit_pct": "Net Profit",
        "volatility": "Volatility (σ)",
        "topk_volatility": "Portfolio Volatility",
        "mdd": "Max Drawdown",
        "mafia_loss": "Observer Total Loss",
        "mafia_direction_loss": "Observer Direction Loss",
        "td3_actor_loss": "TD3 Actor Loss",
        "td3_critic_loss": "TD3 Critic Loss",
        "risk_mse": "Risk MSE",
        "risk_correlation": "Risk Correlation",
    }
    return replacements.get(metric, metric.replace("_", " ").title())


def plot_combined_dashboard(df: pd.DataFrame, output_path: str):
    """Create a single dashboard with all key metrics."""
    # Select key metrics for dashboard
    key_metrics = ["ces_score", "sharpeRatio", "final_capital", "mdd", "topk_volatility", "mafia_loss"]
    available = [m for m in key_metrics if m in df.columns]

    if len(available) < 2:
        print("⚠️  Not enough metrics for dashboard")
        return

    n = len(available)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    phases = sorted(df["phase"].unique())
    windows = sorted(df["window"].unique()) if "window" in df.columns else [0]
    use_global_epoch = "global_epoch" in df.columns and len(windows) > 1
    x_col = "global_epoch" if use_global_epoch else "epoch"

    for idx, metric in enumerate(available):
        ax = axes_flat[idx]

        for phase in phases:
            # [CUSTOMIZATION] Exclude validation loss from dashboard
            if metric == "mafia_loss" and phase == "valid":
                continue
                
            phase_df = df[df["phase"] == phase].dropna(subset=[metric])
            if phase_df.empty:
                continue

            x_vals = phase_df[x_col].values
            y_vals = phase_df[metric].values

            ax.plot(
                x_vals, y_vals,
                color=PHASE_COLORS.get(phase, "gray"),
                marker=PHASE_MARKERS.get(phase, "o"),
                alpha=0.7,
                markersize=5,
                linewidth=1.5,
                label=phase,
            )

        # Window boundaries
        if use_global_epoch and len(windows) > 1:
            for win_idx in windows[1:]:
                win_start = df[df["window"] == win_idx][x_col].min()
                if pd.notna(win_start):
                    ax.axvline(x=win_start, color="red", linestyle=":", alpha=0.4, linewidth=1)

        ax.set_title(_format_metric_name(metric), fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.3)

        # Formatting
        if metric == "final_capital":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
        elif metric in ["mdd", "volatility", "topk_volatility"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))

    for ax in axes_flat[n:]:
        ax.axis("off")

    # Legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(phases), fontsize=10)
    fig.suptitle("Walk-Forward Training Dashboard", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved dashboard to {output_path}")


def plot_window_comparison(df: pd.DataFrame, output_path: str):
    """Compare test performance across walk-forward windows."""
    if "window" not in df.columns:
        return

    test_df = df[df["phase"] == "test"]
    if test_df.empty:
        print("⚠️  No test data for window comparison")
        return

    # Get final epoch metrics for each window
    window_metrics = []
    for win in sorted(test_df["window"].unique()):
        win_df = test_df[test_df["window"] == win]
        last_epoch = win_df["epoch"].max()
        final_row = win_df[win_df["epoch"] == last_epoch].iloc[-1] if len(win_df) > 0 else None
        if final_row is not None:
            window_metrics.append({
                "window": win,
                "sharpeRatio": final_row.get("sharpeRatio", np.nan),
                "final_capital": final_row.get("final_capital", np.nan),
                "mdd": final_row.get("mdd", np.nan),
                "annualReturn_pct": final_row.get("annualReturn_pct", np.nan),
            })

    if not window_metrics:
        return

    wm_df = pd.DataFrame(window_metrics)
    metrics_to_plot = ["sharpeRatio", "annualReturn_pct", "mdd"]
    available = [m for m in metrics_to_plot if m in wm_df.columns and wm_df[m].notna().any()]

    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(wm_df)))

    for idx, metric in enumerate(available):
        ax = axes[idx]
        bars = ax.bar(wm_df["window"].astype(str), wm_df[metric], color=colors, edgecolor="white")
        ax.set_title(_format_metric_name(metric), fontsize=11, fontweight="bold")
        ax.set_xlabel("Window")
        ax.set_ylabel(_format_metric_name(metric))
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, wm_df[metric]):
            if pd.notna(val):
                if metric in ["mdd", "annualReturn_pct"]:
                    label = f"{val*100:.1f}%"
                else:
                    label = f"{val:.2f}"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label,
                       ha="center", va="bottom", fontsize=9)

    fig.suptitle("Test Performance by Walk-Forward Window", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved window comparison to {output_path}")


def plot_trajectory_dynamics(traj_file: str, output_path: str):
    """
    Plot detailed timestep-level dynamics from trajectory_details.csv.
    Shows the hierarchy: Timestep < Trajectory < Batch.
    """
    try:
        df = pd.read_csv(traj_file)
        if df.empty:
            return
    except Exception as e:
        print(f"⚠️  Could not read trajectory log: {e}")
        return

    # Check for required columns
    required_cols = ["grad_norm", "pg_loss", "risk_loss", "dir_loss"]
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️  Missing required columns in trajectory log. Have: {df.columns.tolist()}")
        return

    # Create figure with GridSpec for hierarchical layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])

    # 1. Training Stability (Gradient Norm & Skipped Updates)
    ax_grad = fig.add_subplot(gs[0, 0])
    ax_grad.plot(df.index, df["grad_norm"], color="purple", alpha=0.6, linewidth=1, label="Grad Norm")
    
    # Highlight skipped updates
    if "skipped_update" in df.columns:
        skipped = df[df["skipped_update"] > 0]
        if not skipped.empty:
            ax_grad.scatter(skipped.index, skipped["grad_norm"], color="red", marker="x", s=50, label="Skipped Update")
    
    ax_grad.set_title("Training Stability (Gradient Norm)", fontsize=10, fontweight="bold")
    ax_grad.set_ylabel("Norm")
    ax_grad.grid(True, alpha=0.3)
    ax_grad.legend(loc="upper right")

    # 2. Loss Dynamics (Step-wise)
    ax_loss = fig.add_subplot(gs[0, 1])
    
    # Compute Total Loss (Approximation for visualization)
    # Note: Real total loss involves weights, but simple sum shows the trend
    df["total_loss_est"] = df["pg_loss"].fillna(0) + df["risk_loss"].fillna(0) + df["dir_loss"].fillna(0)
    
    # smoothed losses
    window = min(50, len(df))
    if window > 1:
        ax_loss.plot(df["total_loss_est"].rolling(window).mean(), label="Total Loss", color="black", linewidth=1.5)
        ax_loss.plot(df["pg_loss"].rolling(window).mean(), label="PG Loss", color="blue", alpha=0.6, linestyle="--")
        ax_loss.plot(df["risk_loss"].rolling(window).mean(), label="Risk Loss", color="orange", alpha=0.6, linestyle="--")
        ax_loss.plot(df["dir_loss"].rolling(window).mean(), label="Dir Loss", color="green", alpha=0.6, linestyle="--")
    else:
        ax_loss.plot(df["total_loss_est"], label="Total Loss", color="black", linewidth=1.5)
        ax_loss.plot(df["pg_loss"], label="PG Loss", color="blue", alpha=0.6, linestyle="--")
        
    ax_loss.set_title(f"Loss Dynamics (Total & Components | Rolling Mean W={window})", fontsize=10, fontweight="bold")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    # 3. Market Awareness (Context Diff & Norm)
    ax_ctx = fig.add_subplot(gs[1, 0])
    ax_ctx.plot(df.index, df["c_mkt_norm"], color="teal", alpha=0.6, label="Context Norm")
    ax_ctx2 = ax_ctx.twinx()
    ax_ctx2.plot(df.index, df["c_mkt_diff"], color="gray", alpha=0.4, linestyle=":", label="Context Diff")
    
    ax_ctx.set_title("Market Context Awareness", fontsize=10, fontweight="bold")
    ax_ctx.set_ylabel("Norm")
    ax_ctx2.set_ylabel("Diff (Step-to-Step)")
    
    lines, labels = ax_ctx.get_legend_handles_labels()
    lines2, labels2 = ax_ctx2.get_legend_handles_labels()
    ax_ctx.legend(lines + lines2, labels + labels2, loc="upper right")

    # 4. Risk Adaptation (Risk Eta vs Market Volatility)
    ax_risk = fig.add_subplot(gs[1, 1])
    if "risk_eta" in df.columns:
        ax_risk.plot(df.index, df["risk_eta"], color="purple", linewidth=1.5, label="Risk Tolerance (η)")
        ax_risk.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        
        # Overlay Market Context Norm
        ax_risk2 = ax_risk.twinx()
        ax_risk2.plot(df.index, df["c_mkt_norm"], color="red", linestyle=":", alpha=0.5, label="Market Context Norm")
        
        ax_risk.set_title("Risk Adaptation (Tolerance vs Market State)", fontsize=10, fontweight="bold")
        ax_risk.set_ylabel("Risk Tolerance (η)")
        ax_risk2.set_ylabel("Context Norm")
        
        lns, lbs = ax_risk.get_legend_handles_labels()
        lns2, lbs2 = ax_risk2.get_legend_handles_labels()
        ax_risk.legend(lns+lns2, lbs+lbs2, loc="lower left")
    else:
        ax_risk.text(0.5, 0.5, "Risk Eta not found", ha='center')

    # 5. Prediction Confidence (Probabilities)
    ax_pred = fig.add_subplot(gs[2, :])
    
    import torch
    if "dir_logit_bear" in df.columns:
        logits = torch.tensor(df[["dir_logit_bear", "dir_logit_side", "dir_logit_bull"]].values)
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()
        
        ax_pred.stackplot(df.index, probs[:, 0], probs[:, 1], probs[:, 2], 
                         labels=["Bear", "Side", "Bull"], 
                         colors=["#ffcccc", "#e0e0e0", "#ccffcc"], alpha=0.8)
        ax_pred.set_title("Direction Prediction Confidence", fontsize=10, fontweight="bold")
        ax_pred.set_ylabel("Probability")
        ax_pred.legend(loc="lower left")
    
    # 6. Reward vs Baseline (Advantage)
    ax_adv = fig.add_subplot(gs[3, :])
    ax_adv.plot(df.index, df["net_reward"], color="blue", label="Net Reward", alpha=0.4)
    ax_adv.plot(df.index, df["baseline"], color="black", linestyle="--", label="Baseline", alpha=0.4)
    ax_adv.fill_between(df.index, df["net_reward"], df["baseline"], 
                       where=(df["net_reward"] > df["baseline"]), 
                       interpolate=True, color="green", alpha=0.1, label="Positive Advantage")
    ax_adv.fill_between(df.index, df["net_reward"], df["baseline"], 
                       where=(df["net_reward"] <= df["baseline"]), 
                       interpolate=True, color="red", alpha=0.1, label="Negative Advantage")
    
    ax_adv.set_title("Reward vs Baseline (Advantage Generation)", fontsize=10, fontweight="bold")
    ax_adv.set_xlabel("Training Steps (Continuous)")
    ax_adv.legend(loc="upper left")
    ax_adv.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"📉 Saved detailed trajectory analysis to {output_path}")


def generate_epoch_report(res_dir: str, epoch: int, min_best_epoch: int = 0):
    """
    Generate MAFIA Observer insight charts for the current epoch.

    Charts generated:
    1. Dashboard - Key metrics overview
    2. Loss history - Loss components over epochs
    3. Dynamics - Step-level trajectory dynamics (single file, updated each epoch)
    4. CES breakdown - CES score components analysis (NEW)
    5. Direction breakdown - Per-class F1 analysis (NEW)
    6. Risk calibration - Risk prediction quality (NEW)
    7. Turnover trade-off - Turnover vs Sharpe analysis (NEW)
    
    Args:
        res_dir: Result directory
        epoch: Current epoch
        min_best_epoch: Minimum epoch to consider as 'Best Checkpoint' (Curriculum Learning)
    """
    # Import new plotting functions
    try:
        from RL_controller.plotting_utils import (
            plot_ces_components_breakdown,
            plot_direction_breakdown,
            plot_risk_calibration,
            plot_turnover_sharpe_tradeoff,
            plot_validation_metrics_grid
        )
    except ImportError:
        # Fallback for different import paths
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from RL_controller.plotting_utils import (
            plot_ces_components_breakdown,
            plot_direction_breakdown,
            plot_risk_calibration,
            plot_turnover_sharpe_tradeoff,
            plot_validation_metrics_grid
        )

    # Put plots in the iteration folder
    plots_dir = os.path.join(res_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Find metrics file
    metrics_file = os.path.join(res_dir, "valid_metrics.csv")
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(res_dir, "metrics_history.csv")
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(root_dir, "valid_metrics.csv")

    # 1. Update Epoch-Level Charts (Dashboard + Loss History)
    # 1. Update Epoch-Level Charts (Dashboard + Loss History)
    try:
        df_merged = pd.DataFrame()

        # Load Validation Metrics
        if os.path.exists(metrics_file):
            df_valid = pd.read_csv(metrics_file)
            if "phase" not in df_valid.columns:
                df_valid["phase"] = "valid"
            df_merged = df_valid

        # Load Training Metrics (if available)
        train_metrics_file = os.path.join(res_dir, "train_metrics.csv")
        if os.path.exists(train_metrics_file):
            try:
                df_train = pd.read_csv(train_metrics_file)
                if "phase" not in df_train.columns:
                    df_train["phase"] = "train"
                
                # Align columns (fill missing with NaN if necessary)
                if not df_merged.empty:
                    df_merged = pd.concat([df_merged, df_train], ignore_index=True)
                else:
                    df_merged = df_train
            except Exception as e:
                print(f"[WARN] Failed to load train_metrics.csv: {e}")

        if not df_merged.empty:
            df = df_merged

            # Map columns if valid_metrics.csv format
            if "topk_sharpe_ratio" in df.columns:
                rename_map = {
                    "topk_sharpe_ratio": "sharpeRatio",
                    "loss_total": "mafia_loss",
                    "loss_dir": "mafia_direction_loss",
                    "loss_risk": "mafia_risk_loss",
                    "loss_pg": "mafia_pg_loss"
                }
                df.rename(columns=rename_map, inplace=True)

            df["window"] = 0
            df["global_epoch"] = df["epoch"]

            # Core charts (always generate)
            plot_combined_dashboard(df, os.path.join(plots_dir, "dashboard_latest.png"))
            # [CUSTOMIZATION] Exclude 'valid' phase for loss metrics (User Request)
            plot_metric_group(df, LOSS_METRICS, "Loss History", os.path.join(plots_dir, "loss_history.png"), exclude_phases=["valid"])
            
            # Additional Performance/Risk charts for combined view
            plot_metric_group(df, PERFORMANCE_METRICS, "Performance Metrics", os.path.join(plots_dir, "metrics_performance.png"))
            plot_metric_group(df, RISK_METRICS, "Risk Metrics", os.path.join(plots_dir, "metrics_risk.png"))

            # NEW: Insight charts (generate from valid_metrics.csv ONLY)
            # These use the original file format/path, specific to validation analysis
            if os.path.exists(metrics_file):
                plot_ces_components_breakdown(metrics_file, os.path.join(plots_dir, "ces_breakdown.png"), min_best_epoch=min_best_epoch)
                plot_direction_breakdown(metrics_file, os.path.join(plots_dir, "direction_breakdown.png"))
                plot_risk_calibration(metrics_file, os.path.join(plots_dir, "risk_calibration.png"))
                plot_turnover_sharpe_tradeoff(metrics_file, os.path.join(plots_dir, "turnover_tradeoff.png"))
                
                # Plot detailed grid metrics (This was missing)
                plot_validation_metrics_grid(metrics_file, os.path.join(plots_dir, "validation_metrics.png"), min_best_epoch=min_best_epoch)

    except Exception as e:
        print(f"[WARN] Error generating epoch-level charts: {e}")

    # 2. Generate Micro-Dynamics Chart (Step-level) - SINGLE FILE ONLY
    # No longer creating per-epoch files (dynamics_epoch_N.png) to reduce clutter
    traj_file = os.path.join(res_dir, "trajectory_details.csv")
    if not os.path.exists(traj_file):
        traj_file_parent = os.path.join(res_dir, "../trajectory_details.csv")
        if os.path.exists(traj_file_parent):
            traj_file = traj_file_parent

    if os.path.exists(traj_file):
        # Only generate single dynamics.png (overwritten each epoch)
        plot_trajectory_dynamics(traj_file, os.path.join(plots_dir, "dynamics.png"))



def main():
    parser = argparse.ArgumentParser(description="Visualize walk-forward training metrics with trend analysis.")
    parser.add_argument("--res-dir", default="res_smoke", help="Base directory containing walk-forward results")
    parser.add_argument("--metrics-file", help="Path to single metrics_history.csv (overrides --res-dir)")
    parser.add_argument("--output-dir", help="Output directory for plots (defaults to res-dir)")
    parser.add_argument("--no-trend", action="store_true", help="Disable trend lines")
    parser.add_argument("--trend-window", type=int, default=3, help="Moving average window for trend (default: 3)")
    args = parser.parse_args()

    # Load data
    if args.metrics_file:
        df = pd.read_csv(args.metrics_file)
        df["window"] = 0
        df["global_epoch"] = df["epoch"]
        output_dir = args.output_dir or os.path.dirname(args.metrics_file)
    else:
        window_files = find_walkforward_metrics(args.res_dir)
        if not window_files:
            # Fallback to single file
            all_files = find_all_metrics_files(args.res_dir)
            if all_files:
                df = pd.read_csv(all_files[0])
                df["window"] = 0
                df["global_epoch"] = df["epoch"]
            else:
                raise FileNotFoundError(f"No metrics_history.csv found in {args.res_dir}")
        else:
            print(f"📂 Found {len(window_files)} walk-forward windows")
            df = load_and_merge_windows(window_files)
        output_dir = args.output_dir or args.res_dir

    if df.empty:
        raise ValueError("No data to visualize")

    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    plot_metric_group(
        df, PERFORMANCE_METRICS, "Performance Metrics",
        os.path.join(output_dir, "metrics_performance.png"),
        show_trend=not args.no_trend,
        trend_window=args.trend_window,
    )

    plot_metric_group(
        df, RISK_METRICS, "Risk Metrics",
        os.path.join(output_dir, "metrics_risk.png"),
        show_trend=not args.no_trend,
        trend_window=args.trend_window,
    )

    plot_metric_group(
        df, LOSS_METRICS, "Training Loss",
        os.path.join(output_dir, "metrics_loss.png"),
        show_trend=not args.no_trend,
        trend_window=args.trend_window,
        exclude_phases=["valid"],  # [CUSTOMIZATION] User request
    )

    # Combined dashboard
    plot_combined_dashboard(df, os.path.join(output_dir, "metrics_history.png"))

    # Window comparison (only for multi-window)
    if "window" in df.columns and df["window"].nunique() > 1:
        plot_window_comparison(df, os.path.join(output_dir, "metrics_window_comparison.png"))

    print(f"\n✅ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
