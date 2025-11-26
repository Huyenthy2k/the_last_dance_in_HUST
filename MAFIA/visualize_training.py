#!/usr/bin/env python3
import argparse
import math
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_METRICS = ["reward_sum", "final_capital", "sharpeRatio", "volatility", "mdd"]
LOSS_METRIC_PREFIXES = ("td3_", "mafia_")

def find_latest_metrics(base_dir="res"):
    candidates = []
    for root, _, files in os.walk(base_dir):
        if "metrics_history.csv" in files:
            path = os.path.join(root, "metrics_history.csv")
            candidates.append((os.path.getmtime(path), path))
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1] if candidates else None

def _prepare_epoch_axis(df):
    """Return tuple of (epoch_offset, rel_epoch_series)."""
    if df.empty or "epoch" not in df.columns:
        return 0, df.get("epoch", pd.Series(dtype=float))
    epoch_offset = int(df["epoch"].min())
    rel_epoch = df["epoch"] - epoch_offset + 1
    return epoch_offset, rel_epoch

def plot_metrics(df, metrics, output_path):
    phases = sorted(df["phase"].unique())
    n = len(metrics)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()
    epoch_offset, rel_epoch = _prepare_epoch_axis(df)
    rel_max = rel_epoch.max() if not rel_epoch.empty else None

    for idx, metric in enumerate(metrics):
        if metric not in df.columns:
            print(f"⚠️  Metric '{metric}' not found in metrics_history.csv, skipping plot.")
            continue
        ax = axes_flat[idx]
        for phase in phases:
            subset = df[df["phase"] == phase]
            if subset.empty:
                continue
            x_vals = subset["epoch"] - epoch_offset + 1
            ax.plot(x_vals, subset[metric], marker="o", label=phase)
        ax.set_title(metric)
        xlabel = "Epoch"
        if epoch_offset > 1:
            xlabel += " (relative to resume)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        if rel_max is not None:
            ax.set_xlim(1, rel_max)
        ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes_flat[n:]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(phases))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    print(f"📈 Saved metric plots to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize training/validation/test metrics.")
    parser.add_argument("--res-dir", help="res/... directory containing metrics_history.csv")
    parser.add_argument("--metrics-file", help="Path to metrics_history.csv (overrides --res-dir)")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="List of metrics to plot")
    parser.add_argument("--output", help="Output image path (defaults to <res-dir>/metrics_history.png)")
    args = parser.parse_args()

    metrics_file = args.metrics_file
    if not metrics_file:
        if args.res_dir:
            metrics_file = os.path.join(args.res_dir, "metrics_history.csv")
        if not metrics_file or not os.path.exists(metrics_file):
            metrics_file = find_latest_metrics()
    if metrics_file is None or not os.path.exists(metrics_file):
        raise FileNotFoundError("Cannot locate metrics_history.csv. Run training first to generate it.")

    df = pd.read_csv(metrics_file)
    if df.empty:
        raise ValueError("metrics_history.csv is empty; nothing to plot.")

    res_dir = os.path.dirname(metrics_file) if args.output is None else os.path.dirname(args.output)
    output_path = args.output or os.path.join(res_dir, "metrics_history.png")
    metrics = args.metrics
    if not metrics:
        metrics = list(DEFAULT_METRICS)
        # Auto-include loss metrics if present
        for col in df.columns:
            if col.startswith(LOSS_METRIC_PREFIXES) and col not in metrics:
                metrics.append(col)
    plot_metrics(df, metrics, output_path)

if __name__ == "__main__":
    main()
