# -*- coding: utf-8 -*-
"""
Quick script to generate Top-K recommendations from a saved actions file.

Usage example:
python agents/MAFIA/scripts/generate_topk_manual.py \
  --run-dir agents/MAFIA/res/RLcontroller/TD3/VNINDEX-10/2025-11-22-18-16-20 \
  --actions train_actions.csv \
  --phase train \
  --k 10 --interval 15 --periods 6 --alpha 0.5 --max-cap 0.25

This reads the actions CSV (keeps it intact), drops the `date` column if present,
forces numeric, runs postprocess_topk.aggregate_topk, and writes
`topk_{phase}_manual.csv` into the run directory.
If stock_list file is missing, it will be auto-created from the CSV header.
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# Ensure postprocess_topk import works when running from repo root
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
import postprocess_topk as pp  # noqa: E402


def _ensure_stock_list(stock_list_path: str, columns: List[str]) -> List[str]:
    if os.path.exists(stock_list_path):
        with open(stock_list_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    # Auto-create if missing
    with open(stock_list_path, "w", encoding="utf-8") as f:
        for col in columns:
            f.write(str(col).strip() + "\n")
    return columns


def main():
    p = argparse.ArgumentParser(description="Generate Top-K from actions history.")
    p.add_argument("--run-dir", required=True, help="Run directory containing actions file.")
    p.add_argument("--actions", default="train_actions.csv", help="Actions CSV path (relative to run-dir or absolute).")
    p.add_argument("--stock-list", default=None, help="Stock list file (txt). If missing, will be created from CSV header.")
    p.add_argument("--phase", default="train", help="Phase name for output filename.")
    p.add_argument("--k", type=int, default=10, help="Top-K to return.")
    p.add_argument("--interval", type=int, default=15, help="Rebalance interval in steps (days).")
    p.add_argument("--periods", type=int, default=6, help="Number of past intervals to aggregate.")
    p.add_argument("--alpha", type=float, default=0.5, help="EMA alpha to favor recent periods.")
    p.add_argument("--max-cap", type=float, default=0.25, help="Per-stock cap in absolute weight.")
    p.add_argument("--max-step", type=float, default=0.1, help="Max per-stock change vs previous portfolio.")
    args = p.parse_args()

    run_dir = args.run_dir
    actions_path = args.actions if os.path.isabs(args.actions) else os.path.join(run_dir, args.actions)
    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"Actions file not found: {actions_path}")

    # Load actions CSV, drop date if exists, force numeric
    df = pd.read_csv(actions_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    actions_array = df.to_numpy(dtype=float)
    stock_cols = df.columns.tolist()

    stock_list_path = args.stock_list
    if stock_list_path is None:
        stock_list_path = os.path.join(run_dir, "stock_list_from_actions.txt")
    if not os.path.isabs(stock_list_path):
        stock_list_path = os.path.join(run_dir, os.path.basename(stock_list_path))
    stock_list = _ensure_stock_list(stock_list_path, stock_cols)

    # Compute Top-K
    top_idx, w_final = pp.aggregate_topk(
        actions_history=actions_array,
        k=args.k,
        interval=args.interval,
        periods=args.periods,
        alpha=args.alpha,
        max_cap=args.max_cap,
        vol=None,
        w_prev=None,
        max_step=args.max_step,
    )

    symbols = np.array(stock_list)
    out_df = pd.DataFrame({"symbol": symbols[top_idx], "weight": w_final})
    out_path = os.path.join(run_dir, f"topk_{args.phase}_manual.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(out_df)


if __name__ == "__main__":
    main()
