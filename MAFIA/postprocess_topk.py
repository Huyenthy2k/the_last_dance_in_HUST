"""
Post-process MAFIA/TD3 actions to produce a stable Top-K portfolio by aggregating
rebalance decisions across multiple 15-day cycles (or a custom interval).

Core ideas:
- Stability selection: favor stocks that repeatedly appear with consistent weights.
- Turnover control: only act on rebalance dates and optionally limit per-stock deltas.
- Risk budgeting: optional inverse-vol scaling and weight caps.

Usage example:
python agents/MAFIA/postprocess_topk.py \
  --actions agents/MAFIA/res/RLcontroller/TD3/VNINDEX-10/2025-11-21-15-54-47/actions.npy \
  --stock-list agents/MAFIA/res/RLcontroller/TD3/VNINDEX-10/stock_list.txt \
  --k 10 --interval 15 --periods 6 --alpha 0.5 --max-cap 0.25
"""

import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _load_array(path: str) -> np.ndarray:
    """Load numpy array from .npy or CSV (numeric)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    elif ext in {".csv", ".txt"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required to load CSV weight files") from exc
        df = pd.read_csv(path, header=None)
        return df.values
    else:
        raise ValueError(f"Unsupported file extension for array: {ext}")


def _load_list(path: str) -> List[str]:
    """Load list of symbols from .txt/.csv/.json/.npy."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return [str(x) for x in arr.tolist()]
    if ext in {".txt", ".csv"}:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("JSON stock list must be a list")
    raise ValueError(f"Unsupported file extension for list: {ext}")


def bucket_periods(actions_history: np.ndarray, interval: int) -> List[np.ndarray]:
    """Slice actions_history into consecutive intervals."""
    return [
        actions_history[i : i + interval]
        for i in range(0, len(actions_history), interval)
        if len(actions_history[i : i + interval]) == interval
    ]


def topk_from_period(period_weights: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pick top-K by absolute weight from the last day (or mean of period)."""
    if period_weights.ndim != 2:
        raise ValueError("period_weights must be 2D (days x stocks)")
    w = period_weights[-1]  # could swap to period_weights.mean(axis=0) if desired
    w = w / (np.sum(np.abs(w)) + 1e-8)
    top_idx = np.argsort(-np.abs(w))[:k]
    return top_idx, w


def aggregate_topk(
    actions_history: np.ndarray,
    k: int,
    interval: int,
    periods: int,
    alpha: float,
    max_cap: float,
    vol: Optional[np.ndarray] = None,
    w_prev: Optional[np.ndarray] = None,
    max_step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate Top-K over multiple rebalance periods.

    Returns:
        top_idx: indices of selected stocks
        w_final: normalized weights for those indices
    """
    per = bucket_periods(actions_history, interval)
    if len(per) == 0:
        raise ValueError("No full periods found in actions_history")
    per = per[-periods:]  # take most recent periods

    n = actions_history.shape[1]
    freq = np.zeros(n)
    w_sum = np.zeros(n)
    w_sq = np.zeros(n)
    w_ema = np.zeros(n)
    ema_init = True

    for p in per:
        top_idx, w = topk_from_period(p, k)
        mask = np.zeros(n)
        mask[top_idx] = 1.0
        freq += mask
        w_sum += w * mask
        w_sq += (w ** 2) * mask
        if ema_init:
            w_ema = w
            ema_init = False
        else:
            w_ema = alpha * w + (1 - alpha) * w_ema

    freq /= len(per)
    # Avoid division by zero by adding small epsilon where freq is zero
    denom = (freq * len(per)) + 1e-8
    w_mean = w_sum / denom
    w_var = np.maximum(w_sq / denom - (w_mean ** 2), 0.0)
    w_std = np.sqrt(w_var)

    score = (freq * w_mean) / (w_std + 1e-6)
    w_agg = 0.5 * w_mean + 0.5 * w_ema
    w_agg = w_agg / (1.0 + w_std)  # penalize unstable weights

    top_idx = np.argsort(-np.abs(score))[:k]
    w_final = w_agg[top_idx]

    if vol is not None:
        vol_sel = np.take(vol, top_idx)
        w_final = w_final / (vol_sel + 1e-6)

    w_final = np.clip(w_final, -max_cap, max_cap)

    if w_prev is not None:
        w_prev_sel = np.take(w_prev, top_idx)
        delta = np.clip(w_final - w_prev_sel, -max_step, max_step)
        w_final = w_prev_sel + delta

    w_final = w_final / (np.sum(np.abs(w_final)) + 1e-8)
    return top_idx, w_final


def _portfolio_metrics(portfolio_returns: np.ndarray) -> dict:
    """Compute simple Sharpe/Calmar/MDD on a return series."""
    if portfolio_returns.size == 0:
        return {"sharpe": 0.0, "calmar": 0.0, "mdd": 0.0}
    mean_r = np.mean(portfolio_returns)
    std_r = np.std(portfolio_returns) + 1e-8
    sharpe = mean_r / std_r
    curve = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    mdd = np.min(dd) if dd.size > 0 else 0.0
    calmar = mean_r / (abs(mdd) + 1e-8)
    return {"sharpe": sharpe, "calmar": calmar, "mdd": mdd}


def backtest_static_weights(
    returns: np.ndarray,
    weights: np.ndarray,
    transaction_cost: float = 0.0,
    turnover_ref: Optional[np.ndarray] = None,
) -> dict:
    """
    Apply static weights to per-step per-stock returns and compute metrics.
    turnover_ref: previous weights for initial turnover cost; if None, assume zero.
    """
    weights = np.array(weights, dtype=float)
    weights = weights / (np.sum(np.abs(weights)) + 1e-8)
    if turnover_ref is not None:
        turnover = np.sum(np.abs(weights - turnover_ref))
    else:
        turnover = np.sum(np.abs(weights))
    cost0 = turnover * transaction_cost
    step_ret = returns @ weights
    if step_ret.size > 0:
        step_ret[0] = step_ret[0] - cost0
    metrics = _portfolio_metrics(step_ret)
    metrics["turnover"] = turnover
    metrics["return_mean"] = float(np.mean(step_ret)) if step_ret.size > 0 else 0.0
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate Top-K recommendations over rebalance periods.")
    p.add_argument("--actions", required=True, help="Path to actions_history (.npy or CSV) shape (T, N).")
    p.add_argument("--stock-list", required=True, help="Path to stock list (.txt/.csv/.json/.npy).")
    p.add_argument("--vol", help="Optional path to per-stock vol array (.npy/.csv).")
    p.add_argument("--prev-weights", help="Optional path to previous portfolio weights (.npy/.csv) length N.")
    p.add_argument("--returns", help="Optional path to per-step per-stock returns (.npy/.csv) shape (T, N) for backtest.")
    p.add_argument("--transaction-cost", type=float, default=0.0, help="Linear transaction cost rate for backtest (per unit turnover).")
    p.add_argument("--k", type=int, default=10, help="Top-K to return.")
    p.add_argument("--interval", type=int, default=15, help="Rebalance interval in steps (days).")
    p.add_argument("--periods", type=int, default=6, help="Number of past intervals to aggregate.")
    p.add_argument("--alpha", type=float, default=0.5, help="EMA alpha to favor recent periods.")
    p.add_argument("--max-cap", type=float, default=0.25, help="Per-stock cap in absolute weight.")
    p.add_argument("--max-step", type=float, default=0.1, help="Max per-stock change vs previous portfolio.")
    p.add_argument("--bt-periods", type=int, default=None, help="Backtest on last M periods (default: same as --periods).")
    return p.parse_args()


def main():
    args = _parse_args()
    actions_history = _load_array(args.actions)
    stock_lst = _load_list(args.stock_list)

    if actions_history.ndim != 2:
        raise ValueError("actions_history must be 2D (T, N)")
    if actions_history.shape[1] != len(stock_lst):
        raise ValueError(
            f"actions_history width ({actions_history.shape[1]}) "
            f"does not match stock_list length ({len(stock_lst)})"
        )

    vol = _load_array(args.vol) if args.vol else None
    w_prev = _load_array(args.prev_weights) if args.prev_weights else None
    if vol is not None and len(vol) != len(stock_lst):
        raise ValueError("vol length must match stock list length")
    if w_prev is not None and len(w_prev) != len(stock_lst):
        raise ValueError("prev-weights length must match stock list length")
    returns = _load_array(args.returns) if args.returns else None
    if returns is not None and returns.shape[1] != len(stock_lst):
        raise ValueError("returns width must match stock list length")

    top_idx, w_final = aggregate_topk(
        actions_history=actions_history,
        k=args.k,
        interval=args.interval,
        periods=args.periods,
        alpha=args.alpha,
        max_cap=args.max_cap,
        vol=vol,
        w_prev=w_prev,
        max_step=args.max_step,
    )

    print("Top-K recommendations (index, symbol, weight):")
    for idx, w in zip(top_idx, w_final):
        symbol = stock_lst[idx] if idx < len(stock_lst) else f"IDX_{idx}"
        print(f"{idx:4d}  {symbol:15s}  {w:+.4f}")

    if returns is not None:
        bt_periods = args.bt_periods if args.bt_periods is not None else args.periods
        horizon = bt_periods * args.interval
        returns_slice = returns[-horizon:] if returns.shape[0] >= horizon else returns
        # Baseline: last-period weights
        last_period = actions_history[-args.interval:]
        _, w_last = topk_from_period(last_period, args.k)
        baseline_idx = np.argsort(-np.abs(w_last))[:args.k]
        w_baseline = w_last[baseline_idx]
        w_baseline = w_baseline / (np.sum(np.abs(w_baseline)) + 1e-8)
        w_final_norm = w_final / (np.sum(np.abs(w_final)) + 1e-8)

        metrics_rec = backtest_static_weights(
            returns_slice[:, top_idx], w_final_norm, transaction_cost=args.transaction_cost
        )
        metrics_base = backtest_static_weights(
            returns_slice[:, baseline_idx], w_baseline, transaction_cost=args.transaction_cost
        )
        print("\n[Backtest] Last horizon vs. recommended")
        print(f"  Horizon steps: {returns_slice.shape[0]} (periods ~ {returns_slice.shape[0]/args.interval:.1f})")
        print(f"  Recommended -> Sharpe: {metrics_rec['sharpe']:.3f}, Calmar: {metrics_rec['calmar']:.3f}, MDD: {metrics_rec['mdd']:.3f}, Turnover: {metrics_rec['turnover']:.3f}")
        print(f"  Baseline    -> Sharpe: {metrics_base['sharpe']:.3f}, Calmar: {metrics_base['calmar']:.3f}, MDD: {metrics_base['mdd']:.3f}, Turnover: {metrics_base['turnover']:.3f}")
        better = "recommended" if metrics_rec['sharpe'] >= metrics_base['sharpe'] and metrics_rec['calmar'] >= metrics_base['calmar'] else "baseline"
        print(f"  Selected: {better}")


if __name__ == "__main__":
    main()
