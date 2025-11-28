#!/usr/bin/env python3
"""
Generate a lightweight HTML dashboard for walk-forward runs.

Example:
  python scripts/walk_forward_dashboard.py \
    --metrics agents/MAFIA/res/.../walk_forward_metrics.csv \
    --summary agents/MAFIA/res/.../walk_forward_summary.csv \
    --last-summary agents/MAFIA/res/.../walk_forward_last_window_summary.csv \
    --output wf_dashboard.html

Only depends on pandas (no extra viz libs). Produces tables and simple stats for:
  - Per-window metrics (raw rows)
  - Per-window mean/std
  - Last-window mean/std across seeds
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[DASH] Failed to read {path}: {e}", flush=True)
            return None
    return None


def render_table(df: Optional[pd.DataFrame], title: str) -> str:
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No data available.</p>"
    return (
        f"<h3>{title}</h3>"
        + df.to_html(index=False, border=0, justify="center", classes="data-table")
    )


def render_summary_table(df: Optional[pd.DataFrame], title: str) -> str:
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No data available.</p>"
    # Flatten multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df_flat = df.copy()
        df_flat.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in df_flat.columns]
        df_flat.reset_index(inplace=True)
    else:
        df_flat = df.reset_index()
    return (
        f"<h3>{title}</h3>"
        + df_flat.to_html(index=False, border=0, justify="center", classes="data-table")
    )


def main():
    parser = argparse.ArgumentParser(description="Generate walk-forward HTML dashboard")
    parser.add_argument("--metrics", required=True, help="Path to walk_forward_metrics.csv")
    parser.add_argument("--summary", default=None, help="Path to walk_forward_summary.csv")
    parser.add_argument(
        "--last-summary",
        dest="last_summary",
        default=None,
        help="Path to walk_forward_last_window_summary.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML file (default: metrics directory / wf_dashboard.html)",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else None
    last_summary_path = Path(args.last_summary).expanduser().resolve() if args.last_summary else None

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else metrics_path.parent / "wf_dashboard.html"
    )

    df_metrics = load_csv(str(metrics_path))
    df_summary = load_csv(str(summary_path)) if summary_path else None
    df_last_summary = load_csv(str(last_summary_path)) if last_summary_path else None

    if df_metrics is None or df_metrics.empty:
        raise SystemExit(f"[DASH] Metrics CSV is empty or missing: {metrics_path}")

    num_windows = df_metrics["window"].nunique()
    num_seeds = df_metrics["seed"].nunique()
    last_window = df_metrics["window"].max()

    overview_html = f"""
    <h2>Walk-forward Overview</h2>
    <ul>
        <li>Metrics file: {metrics_path}</li>
        <li>Summary file: {summary_path or 'n/a'}</li>
        <li>Last-window summary: {last_summary_path or 'n/a'}</li>
        <li>Windows: {num_windows}</li>
        <li>Seeds: {num_seeds}</li>
        <li>Last window index: {last_window}</li>
    </ul>
    """

    html = f"""
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>Walk-forward Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
            h1 {{ margin-bottom: 0; }}
            h2 {{ margin-top: 30px; }}
            h3 {{ margin-top: 25px; }}
            .data-table {{ border-collapse: collapse; width: 100%; }}
            .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 6px; font-size: 13px; }}
            .data-table th {{ background: #f2f2f2; }}
            ul {{ line-height: 1.6; }}
        </style>
    </head>
    <body>
        <h1>Walk-forward Dashboard</h1>
        {overview_html}
        {render_table(df_metrics, "Per-run (per window x seed) metrics")}
        {render_summary_table(df_summary, "Per-window mean/std")}
        {render_summary_table(df_last_summary, "Last window mean/std across seeds")}
    </body>
    </html>
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[DASH] Dashboard written to {output_path}")


if __name__ == "__main__":
    main()
