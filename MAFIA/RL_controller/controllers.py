# ！/usr/bin/python
# -*- coding: utf-8 -*-#

"""
---------------------------------
 Name:         controllers.py
 Description:  Implement the solver-based agent of the proposed MASA framework.
 Author:       MASA
---------------------------------
"""

import numpy as np
import cvxpy as cp
import sys

# Import LiveDisplay integration
try:
    from utils.display_integration import (
        update_controller as display_update_controller,
        get_display,
        smart_print,
    )

    LIVE_DISPLAY_AVAILABLE = True
except ImportError:
    LIVE_DISPLAY_AVAILABLE = False

    def get_display():
        return None

    smart_print = print  # Fallback to normal print


def _is_live_display_active() -> bool:
    """Check if LiveDisplay is currently active and rendering."""
    if not LIVE_DISPLAY_AVAILABLE:
        return False
    display = get_display()
    return display is not None and display.enabled


def _select_topk_from_scores(market_scores_full, method="topk", k=10, threshold=0.01):
    """
    Select Top-K stocks from market_scores_full based on method.

    Args:
        market_scores_full: (N,) array - Full market scores from MAFIA
        method: 'topk' (select top K) or 'threshold' (scores > threshold)
        k: Number of stocks to select (for 'topk' method)
        threshold: Minimum score threshold (for 'threshold' method)

    Returns:
        selected_indices: Indices of selected stocks
        selected_mask: Boolean mask (True for selected stocks)
    """
    if market_scores_full is None or len(market_scores_full) == 0:
        return np.array([]), np.zeros(
            len(market_scores_full) if market_scores_full is not None else 0, dtype=bool
        )

    market_scores_full = np.nan_to_num(
        market_scores_full, nan=0.0, posinf=0.0, neginf=0.0
    )

    if method == "topk":
        k = min(k, len(market_scores_full))
        topk_idx = np.argpartition(market_scores_full, -k)[-k:]
        topk_sorted = topk_idx[np.argsort(market_scores_full[topk_idx])[::-1]]
        mask = np.zeros(len(market_scores_full), dtype=bool)
        mask[topk_sorted] = True
        return topk_sorted, mask
    elif method == "threshold":
        mask = market_scores_full > threshold
        selected_indices = np.where(mask)[0]
        return selected_indices, mask
    else:
        raise ValueError(f"Unknown selection method: {method}")


def _expand_topk_action_to_full(a_rl, env):
    """
    Expand a Top-K action (length K) to full-universe length (N) using Observer Top-K indices.
    Returns (full_action, mask) where mask marks the selected Top-K positions.
    """
    a_rl = np.array(a_rl, dtype=float)
    n = env.stock_num
    k = min(len(a_rl), n)
    full = np.zeros(n, dtype=float)
    topk_idx = getattr(env, "observer_topk_indices", None)
    if topk_idx is None or len(np.atleast_1d(topk_idx)) == 0:
        topk_idx = np.arange(k)
    topk_idx = np.array(topk_idx, dtype=int)
    topk_idx = topk_idx[:k]
    topk_idx = np.clip(topk_idx, 0, n - 1)
    for i, idx in enumerate(topk_idx):
        full[idx] = a_rl[i]
    mask = np.zeros(n, dtype=bool)
    mask[topk_idx] = True
    if np.sum(np.abs(full)) > 1e-8:
        full = full / np.sum(np.abs(full))
    else:
        if len(topk_idx) > 0:
            full[topk_idx] = 1.0 / len(topk_idx)
        else:
            full = np.ones(n) / n
    return full, mask


def _apply_market_scores_to_rl_action(a_rl, market_scores_full, config, env=None):
    """
    Legacy helper (unused now): previously handled full-universe RL outputs.
    With Top-K-only flow, RL already operates on Observer Top-K; returns normalized a_rl.

    Args:
        a_rl: (N,) array - RL action
        market_scores_full: (N,) array - Full market scores from MAFIA
        config: Config object
        env: Environment object (required for accessing observer_topk_indices)

    Returns:
        a_rl_modified: (N,) array - Same as input, normalized.
        rl_selected_mask: None (no selection applied)
    """
    a_full = np.array(a_rl, dtype=float)
    a_full = np.nan_to_num(a_full, nan=0.0, posinf=0.0, neginf=0.0)
    if np.sum(np.abs(a_full)) > 1e-8:
        a_full = a_full / np.sum(np.abs(a_full))
    else:
        a_full = np.ones(len(a_full)) / len(a_full)
    return a_full, None


def _boost_topk_weights(a_final, market_scores_full, config, rl_selected_mask=None):
    """
    Boost weights using market_scores_full as a signal to adjust weights proportionally.

    Mode-specific behavior:
    - 'compact': Not applicable (boost disabled in compact mode)
    - 'full-score': Boost stocks selected by RL (if rl_selected_mask provided), or boost all stocks

    Two methods:
    1. 'blend': Blend current weights with market_scores_full distribution
       a_boosted = (1 - boost_factor) * a_final + boost_factor * market_scores_full
    2. 'proportional': Boost weights proportionally based on market_scores_full
       - Stocks with higher market_scores get more boost
       - Boost is proportional to the ratio of market_scores

    Args:
        a_final: (N,) array - Final action (after CBF)
        market_scores_full: (N,) array - Full market scores from MAFIA (used as signal)
        config: Config object
        rl_selected_mask: (N,) bool array - Mask of stocks selected by RL (None if not applicable)

    Returns:
        a_final_boosted: (N,) array - Action with boosted weights based on market_scores_full
    """
    # Check mode: only allow boost in 'full-score' mode
    state_mode = getattr(config, "mafia_state_mode", "compact")
    if state_mode == "compact":
        # In compact mode: no boost (RL uses Top-K from Observer)
        return a_final

    # In 'full-score' mode: check if Solver boost is enabled
    if not getattr(config, "mafia_solver_boost_enabled", False):
        return a_final

    if market_scores_full is None or len(market_scores_full) != len(a_final):
        return a_final

    boost_factor = getattr(config, "mafia_solver_boost_factor", 0.3)
    boost_method = getattr(config, "mafia_solver_boost_method", "blend")
    boost_factor = np.clip(boost_factor, 0.0, 1.0)  # Ensure in [0, 1]

    # Clean market_scores_full
    market_scores_clean = np.nan_to_num(
        market_scores_full, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Normalize market_scores_full to ensure it's a valid distribution
    if np.sum(market_scores_clean) > 1e-8:
        market_scores_normalized = market_scores_clean / np.sum(market_scores_clean)
    else:
        # If all zeros, use uniform distribution
        market_scores_normalized = np.ones(len(market_scores_clean)) / len(
            market_scores_clean
        )

    # If RL has selected stocks, only boost those stocks
    if rl_selected_mask is not None and np.any(rl_selected_mask):
        # Only boost stocks selected by RL
        if boost_method == "blend":
            # Blend only for selected stocks
            a_final_boosted = a_final.copy()
            a_final_boosted[rl_selected_mask] = (1.0 - boost_factor) * a_final[
                rl_selected_mask
            ] + boost_factor * market_scores_normalized[rl_selected_mask]
            # Renormalize selected stocks
            if np.sum(a_final_boosted[rl_selected_mask]) > 1e-8:
                a_final_boosted[rl_selected_mask] = (
                    a_final_boosted[rl_selected_mask]
                    / np.sum(a_final_boosted[rl_selected_mask])
                    * np.sum(a_final[rl_selected_mask])
                )
        elif boost_method == "proportional":
            # Proportional boost only for selected stocks
            market_scores_selected = market_scores_normalized[rl_selected_mask]
            market_scores_min = np.min(market_scores_selected)
            market_scores_max = np.max(market_scores_selected)
            if market_scores_max - market_scores_min > 1e-8:
                market_scores_scaled = (market_scores_selected - market_scores_min) / (
                    market_scores_max - market_scores_min
                )
            else:
                market_scores_scaled = np.ones(len(market_scores_selected))

            boost_per_stock = 1.0 + boost_factor * market_scores_scaled
            a_final_boosted = a_final.copy()
            a_final_boosted[rl_selected_mask] = (
                a_final[rl_selected_mask] * boost_per_stock
            )
        else:
            raise ValueError(
                f"Unknown boost method: {boost_method}. Must be 'blend' or 'proportional'"
            )
    else:
        # Boost all stocks (RL hasn't selected specific stocks, or boost all)
        if boost_method == "blend":
            a_final_boosted = (
                1.0 - boost_factor
            ) * a_final + boost_factor * market_scores_normalized
        elif boost_method == "proportional":
            market_scores_min = np.min(market_scores_normalized)
            market_scores_max = np.max(market_scores_normalized)
            if market_scores_max - market_scores_min > 1e-8:
                market_scores_scaled = (
                    market_scores_normalized - market_scores_min
                ) / (market_scores_max - market_scores_min)
            else:
                market_scores_scaled = np.ones(len(market_scores_normalized))
            boost_per_stock = 1.0 + boost_factor * market_scores_scaled
            a_final_boosted = a_final * boost_per_stock
        else:
            raise ValueError(
                f"Unknown boost method: {boost_method}. Must be 'blend' or 'proportional'"
            )

    # Renormalize to ensure sum = 1
    if np.sum(np.abs(a_final_boosted)) > 1e-8:
        a_final_boosted = a_final_boosted / np.sum(np.abs(a_final_boosted))
    else:
        # Fallback: use market_scores_normalized if all zeros
        a_final_boosted = market_scores_normalized

    # Ensure non-negative
    a_final_boosted = np.maximum(a_final_boosted, 0.0)

    # Final renormalize after clipping
    if np.sum(a_final_boosted) > 1e-8:
        a_final_boosted = a_final_boosted / np.sum(a_final_boosted)
    else:
        a_final_boosted = np.ones(len(a_final)) / len(a_final)

    if env is not None and hasattr(env, "is_last_ctrl_solvable"):
        smart_print(
            f"[SOLVER] Boost result | solvable={env.is_last_ctrl_solvable} | action_sum={np.sum(a_final_boosted):.4f}"
        )
    return a_final_boosted


def RL_withoutController(a_rl, env=None):
    a_rl = np.array(a_rl, dtype=float)
    # Ensure action length matches Top-K (observer selection)
    if env is not None and a_rl.shape[0] != env.rl_stock_num:
        idx = getattr(env, "observer_topk_indices", None)
        if idx is None or len(np.atleast_1d(idx)) == 0:
            idx = np.arange(min(env.rl_stock_num, a_rl.shape[0]))
        idx = np.clip(np.array(idx, dtype=int), 0, a_rl.shape[0] - 1)
        a_rl = a_rl[idx[: env.rl_stock_num]]
    rl_selected_mask = None
    a_cbf_topk = np.zeros_like(a_rl)

    if env is not None and hasattr(env, "_record_action_memory"):
        env._record_action_memory(a_rl, a_cbf_topk)
    return a_rl


def RL_withController(a_rl, env=None):
    a_rl = np.array(a_rl, dtype=float)
    # Force action to Top-K length using observer indices or fallback slice
    if env is not None and a_rl.shape[0] != env.rl_stock_num:
        idx = getattr(env, "observer_topk_indices", None)
        if idx is None or len(np.atleast_1d(idx)) == 0:
            idx = np.arange(min(env.rl_stock_num, a_rl.shape[0]))
        idx = np.clip(np.array(idx, dtype=int), 0, a_rl.shape[0] - 1)
        a_rl = a_rl[idx[: env.rl_stock_num]]

    # ===== Check training mode: Skip CBF in Phase 1 (Observer-Only) =====
    # During Phase 1, TD3 is frozen and uses uniform weights.
    # CBF optimization is unnecessary - just pass through the action.
    training_mode = (
        getattr(env.config, "training_mode", "RL_ONLY")
        if env and hasattr(env, "config")
        else "RL_ONLY"
    )
    if training_mode == "OBSERVER_ONLY":
        # Phase 1: Skip CBF, return normalized action directly
        a_final = np.maximum(a_rl, 0.0)
        a_final_sum = np.sum(a_final)
        if a_final_sum > 1e-8:
            a_final = a_final / a_final_sum
        else:
            a_final = np.ones(len(a_final)) / len(a_final)
        # Record action without CBF adjustment
        if env is not None and hasattr(env, "_record_action_memory"):
            env._record_action_memory(
                a_final,
                np.zeros_like(a_final),  # No CBF adjustment
                topk_indices=getattr(env, "observer_topk_indices", None),
            )
        return a_final

    # Top-K only flow; no full-universe masking
    rl_selected_mask = None
    a_cbf_topk = np.zeros_like(a_rl)

    # NOTE: Do NOT manually append to action_rl_memory or action_cbf_memeory here.
    # The _record_action_memory() call at the end handles all memory recording
    # in TopK format with proper indices for later expansion.

    if env.config.pricePredModel == "MA":
        pred_prices_change = get_pred_price_change(env=env)
        pred_dict = {"shortterm": pred_prices_change}
    else:
        raise ValueError(
            "Cannot find the price prediction model [{}]..".format(
                env.config.pricePredModel
            )
        )
    optimized_action, is_solvable_status = cbf_opt(
        env=env, a_rl=a_rl, pred_dict=pred_dict, rl_selected_mask=rl_selected_mask
    )
    if is_solvable_status and optimized_action is not None:
        a_final = optimized_action  # length K (Top-K)
        a_cbf_topk = a_final - a_rl
        # NOTE: Do NOT append to action_cbf_memeory here - _record_action_memory handles it
    else:
        a_final = a_rl
        # a_cbf_topk stays as zeros (initialized above)
    env.is_last_ctrl_solvable = bool(is_solvable_status)

    # Normalize in Top-K space to ensure constraints: sum = 1, all >= 0
    a_final = np.maximum(a_final, 0.0)
    a_final_sum = np.sum(a_final)
    if a_final_sum > 1e-8:
        a_final = a_final / a_final_sum
    else:
        a_final = np.ones(len(a_final)) / len(a_final)

    # Return Top-K weights; env will map to full universe internally
    if env is not None and hasattr(env, "_record_action_memory"):
        env._record_action_memory(
            a_final,
            a_cbf_topk,
            topk_indices=getattr(env, "observer_topk_indices", None),
        )

    # Update LiveDisplay with CBF controller metrics
    if LIVE_DISPLAY_AVAILABLE and env is not None:
        try:
            # Get CBF parameters from config
            sigma_base = getattr(env.config, "risk_market", 0.15)
            risk_eta = (
                env.risk_adj_lst[-1]
                if hasattr(env, "risk_adj_lst") and len(env.risk_adj_lst) > 0
                else 1.0
            )
            sigma_current = sigma_base * risk_eta

            # Count interventions (non-zero CBF adjustments)
            cbf_intervention_count = getattr(env, "_cbf_intervention_count", 0)
            if np.any(np.abs(a_cbf_topk) > 1e-6):
                cbf_intervention_count += 1
                env._cbf_intervention_count = cbf_intervention_count

            # Determine adjustment direction
            cbf_adj_norm = np.linalg.norm(a_cbf_topk)
            if cbf_adj_norm > 1e-6:
                # Check if action norm decreased (risk reduction)
                adjust_direction = (
                    "↓" if np.linalg.norm(a_final) < np.linalg.norm(a_rl) else "↑"
                )
            else:
                adjust_direction = ""

            # Format action vectors for display (first 5 elements)
            def fmt_action(arr, n=5):
                if arr is None or len(arr) == 0:
                    return ""
                vals = [f"{v:.2f}" for v in arr[:n]]
                suffix = ", ..." if len(arr) > n else ""
                return "[" + ", ".join(vals) + suffix + "]"

            display_update_controller(
                cbf_enabled=True,
                cbf_alpha=getattr(env.config, "cbf_alpha", 0.1),
                cbf_interventions=cbf_intervention_count,
                cbf_last_intervention="" if cbf_adj_norm < 1e-6 else "Risk adjustment",
                cbf_safety_margin=cbf_adj_norm,
                cbf_constraint_active=is_solvable_status,
                cbf_sigma_base=sigma_base,
                cbf_sigma_current=sigma_current,
                cbf_min_variance=getattr(env.config, "cbf_min_variance", 0.01),
                cbf_max_position=getattr(env.config, "cbf_max_position", 0.25),
                cbf_adjust_direction=adjust_direction,
                cbf_adjust_magnitude=cbf_adj_norm,
                cbf_action_before=fmt_action(a_rl),
                cbf_action_after=fmt_action(a_final),
                cbf_action_norm_before=float(np.linalg.norm(a_rl)),
                cbf_action_norm_after=float(np.linalg.norm(a_final)),
            )
        except Exception:
            pass

    return a_final


def get_pred_price_change(env):
    ma_lst = env.ctl_state["MA-{}".format(env.config.otherRef_indicator_ma_window)]
    pred_prices = ma_lst
    cur_close_price = np.array(env.curData["close"].values)

    # Ensure pred_prices and cur_close_price have the same shape
    if len(pred_prices) != len(cur_close_price):
        # If shapes don't match, align based on stock order in curData
        # Create a mapping from stock to index in curData
        stock_to_idx = {
            stock: idx for idx, stock in enumerate(env.curData["stock"].values)
        }

        # If ma_lst is indexed by stock, we need to reorder it
        # For now, pad or truncate to match cur_close_price length
        if len(pred_prices) < len(cur_close_price):
            # Pad with last value or use cur_close_price as fallback
            pred_prices = np.pad(
                pred_prices,
                (0, len(cur_close_price) - len(pred_prices)),
                mode="constant",
                constant_values=(pred_prices[-1] if len(pred_prices) > 0 else 1.0),
            )
        else:
            # Truncate to match
            pred_prices = pred_prices[: len(cur_close_price)]

    # Clean any NaNs/infs in pred_prices and cur_close_price
    pred_prices = np.nan_to_num(pred_prices, nan=0.0, posinf=0.0, neginf=0.0)
    cur_close_price = np.nan_to_num(cur_close_price, nan=1.0, posinf=1.0, neginf=1.0)

    # Safe division: avoid division by zero using epsilon
    epsilon = 1e-8
    # Replace zero or very small values with epsilon to avoid division by zero
    cur_close_price_safe = np.where(
        np.abs(cur_close_price) < epsilon, epsilon, cur_close_price
    )

    pred_prices_change = (pred_prices - cur_close_price) / cur_close_price_safe

    # Clean any remaining NaNs/infs in the result
    pred_prices_change = np.nan_to_num(
        pred_prices_change, nan=0.0, posinf=0.0, neginf=0.0
    )

    return pred_prices_change


def _get_daily_return_matrix(env):
    """
    Returns matrix shaped (num_stocks, lookback) with cleaned daily returns.
    Fetches historical daily returns from rawdata to compute proper covariance.
    """
    lookback = getattr(env.config, "dailyRetun_lookback", 30)

    # Try to get historical returns from rawdata (proper method)
    daily_returns = _get_historical_daily_returns(env, lookback)
    if daily_returns is not None:
        return daily_returns

    # Fallback to ctl_state (may be single-day vector)
    key = "DAILYRETURNS-{}".format(env.config.dailyRetun_lookback)
    daily_returns = env.ctl_state.get(key, None)
    if daily_returns is None:
        return None
    daily_returns = np.array(daily_returns, dtype=float)
    daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
    if daily_returns.ndim == 1:
        daily_returns = daily_returns.reshape(-1, 1)  # (num_stocks, 1)
    # Ensure stocks are rows
    if (
        daily_returns.shape[0] != env.stock_num
        and daily_returns.shape[1] == env.stock_num
    ):
        daily_returns = daily_returns.T
    if daily_returns.shape[0] != env.stock_num:
        # Pad or truncate rows to match stock_num
        if daily_returns.shape[0] < env.stock_num:
            pad = env.stock_num - daily_returns.shape[0]
            daily_returns = np.pad(
                daily_returns, ((0, pad), (0, 0)), mode="constant", constant_values=0.0
            )
        else:
            daily_returns = daily_returns[: env.stock_num, :]
    return daily_returns


def _get_historical_daily_returns(env, lookback: int):
    """
    Get historical daily returns matrix from rawdata.
    Returns: np.ndarray of shape (num_stocks, lookback) or None if unavailable.
    Uses vectorized pandas operations for efficiency.
    """
    if not hasattr(env, "rawdata") or env.rawdata is None:
        return None
    if not hasattr(env, "curTradeDay"):
        return None

    try:
        import pandas as pd

        rawdata = env.rawdata
        cur_day = env.curTradeDay

        # Get all unique dates sorted
        all_dates = sorted(rawdata["date"].unique())
        if cur_day >= len(all_dates):
            cur_day = len(all_dates) - 1

        # Get window of dates (need lookback+1 to compute lookback returns)
        start_day = max(0, cur_day - lookback)
        end_day = cur_day + 1
        window_dates = all_dates[start_day:end_day]

        if len(window_dates) < 2:
            return None

        # Filter data for window dates
        window_data = rawdata[rawdata["date"].isin(window_dates)].copy()

        # Pivot to get close prices: rows=dates, columns=stocks
        if "close" not in window_data.columns:
            return None

        close_pivot = window_data.pivot_table(
            index="date", columns="stock", values="close", aggfunc="first"
        ).sort_index()

        # Compute daily returns: (close_t - close_{t-1}) / close_{t-1}
        returns_pivot = close_pivot.pct_change(fill_method=None).iloc[
            1:
        ]  # Drop first NaN row

        # Transpose to get (num_stocks, num_days) and fill NaN with 0
        returns_matrix = returns_pivot.T.values
        returns_matrix = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure correct number of stocks (pad if needed)
        stocks = getattr(env, "stock_lst", None)
        if stocks is not None and len(stocks) > returns_matrix.shape[0]:
            pad_rows = len(stocks) - returns_matrix.shape[0]
            returns_matrix = np.pad(
                returns_matrix,
                ((0, pad_rows), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        return returns_matrix
    except Exception:
        return None


def _build_covariance_matrix(env, target_size=None, topk_indices=None):
    """
    Build PSD covariance matrix for the latest lookback window.
    If target_size/topk_indices provided, slice to those assets (Top-K mode).
    """
    daily_returns = _get_daily_return_matrix(env)
    size = target_size if target_size is not None else env.stock_num
    if daily_returns is None or daily_returns.shape[1] < 2:
        return np.eye(size) * env.config.risk_market
    if topk_indices is not None and len(topk_indices) > 0:
        idx = np.array(topk_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < daily_returns.shape[0])]
        if idx.size > 0:
            daily_returns = daily_returns[idx, :]
    try:
        cov_matrix = np.cov(daily_returns)
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        smart_print(
            f"[Controller] Warning: covariance computation failed ({exc}), using identity fallback"
        )
        cov_matrix = np.eye(size) * env.config.risk_market

    # Ensure correct shape and symmetry
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])
    elif cov_matrix.ndim == 1:
        cov_matrix = np.diag(cov_matrix)
    if cov_matrix.shape[0] != size or cov_matrix.shape[1] != size:
        cov_matrix = np.pad(
            cov_matrix,
            (
                (0, max(0, size - cov_matrix.shape[0])),
                (0, max(0, size - cov_matrix.shape[1])),
            ),
            mode="constant",
            constant_values=0.0,
        )
        cov_matrix = cov_matrix[:size, :size]
    # Symmetrize and add small jitter for numerical stability
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    return cov_matrix


def _compute_observer_bias(env, target_size=None):
    """
    Convert observer signals into linear bias vector q.
    Positive entries in q penalize corresponding weights; negative entries attract.
    """
    bias_weight = getattr(env.config, "controller_observer_bias_weight", 0.0)
    size = target_size if target_size is not None else env.stock_num
    if bias_weight <= 0 or not getattr(env.config, "enable_market_observer", False):
        return np.zeros(size)
    scores = getattr(env, "market_scores_full", None)
    if scores is None:
        return np.zeros(size)
    scores = np.array(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.shape[0] != size:
        if scores.shape[0] < size:
            scores = np.pad(
                scores,
                (0, size - scores.shape[0]),
                mode="constant",
                constant_values=0.0,
            )
        else:
            scores = scores[:size]
    # Shift to non-negative and normalize
    min_score = np.min(scores)
    scores = scores - min_score
    total = np.sum(scores)
    if total <= 1e-8:
        scores = np.ones(size) / size
    else:
        scores = scores / total
    return -bias_weight * scores


def _estimate_min_variance_risk(cov_matrix: np.ndarray):
    """
    Approximate minimum achievable portfolio risk (volatility) under sum(x)=1.
    Used to quickly detect risk bounds that are too tight to satisfy.
    """
    try:
        ones = np.ones(cov_matrix.shape[0])
        inv_cov = np.linalg.pinv(cov_matrix)
        weights = inv_cov @ ones
        if np.sum(weights) <= 1e-10:
            return None
        weights = weights / np.sum(weights)
        risk_val = float(np.matmul(weights, np.matmul(cov_matrix, weights.T)))
        return float(np.sqrt(max(risk_val, 0.0)))
    except Exception:
        return None


def _min_variance_weights(cov_matrix: np.ndarray):
    """
    Compute non-negative min-variance weights under sum(x)=1.
    """
    try:
        n = cov_matrix.shape[0]
        ones = np.ones(n)
        inv_cov = np.linalg.pinv(cov_matrix)
        w = inv_cov @ ones
        # Enforce non-negativity and renormalize
        w = np.maximum(w, 0.0)
        if np.sum(w) <= 1e-10:
            w = np.ones(n)
        w = w / np.sum(w)
        return w
    except Exception:
        return None


def cbf_opt(env, a_rl, pred_dict, rl_selected_mask=None):
    """
    Solve-based agent (spec-aligned):
    minimize 0.5 (a_RL + Δ)^T Σ (a_RL + Δ) + bias^T (a_RL + Δ) + λ_reg ||Δ||^2
    s.t. sum(Δ)=0, a_RL+Δ>=0, and risk bound from Market Observer.
    """
    del pred_dict  # Price predictions are unused in the deterministic QP objective
    a_rl = np.array(a_rl, dtype=float)
    if np.sum(a_rl) <= 1e-8:
        a_rl = np.ones(len(a_rl)) / len(a_rl)
    else:
        a_rl = a_rl / np.sum(a_rl)

    target_size = len(a_rl)
    topk_idx = getattr(env, "observer_topk_indices", None)
    use_topk_only = bool(getattr(env.config, "mafia_solver_use_rl_topk_only", False))
    if use_topk_only and rl_selected_mask is not None:
        mask_arr = np.array(rl_selected_mask, dtype=bool)
        active_indices = np.nonzero(mask_arr)[0]
    elif use_topk_only and topk_idx is not None:
        active_indices = np.array(topk_idx, dtype=int)
    else:
        active_indices = np.arange(target_size)
    if active_indices.size == 0:
        active_indices = np.arange(target_size)

    try:
        cov_matrix_full = _build_covariance_matrix(
            env, target_size=target_size, topk_indices=topk_idx
        )
    except TypeError:
        cov_matrix_full = _build_covariance_matrix(env)
    try:
        bias_vec_full = _compute_observer_bias(env, target_size=target_size)
    except TypeError:
        bias_vec_full = _compute_observer_bias(env)
    lambda_reg = max(getattr(env.config, "controller_reg_lambda", 0.0), 0.0)

    # Operate directly on Top-K subset when configured; otherwise use full universe
    if use_topk_only and active_indices.size < target_size:
        a_rl_opt = a_rl[active_indices]
        cov_matrix = cov_matrix_full[np.ix_(active_indices, active_indices)]
        bias_vec = bias_vec_full[active_indices] if bias_vec_full is not None else None
    else:
        active_indices = np.arange(target_size)
        a_rl_opt = a_rl
        cov_matrix = cov_matrix_full
        bias_vec = bias_vec_full

    # Risk tolerance factor (eta_t) from observer (relative, not absolute sigma)
    risk_eta_raw = None
    if hasattr(env, "risk_adj_lst") and len(env.risk_adj_lst) > 0:
        risk_eta_raw = env.risk_adj_lst[-1]
    risk_eta = None
    if risk_eta_raw is not None:
        try:
            risk_eta = float(risk_eta_raw)
            if risk_eta <= 0 or np.isnan(risk_eta) or np.isinf(risk_eta):
                risk_eta = None
        except Exception:
            risk_eta = None
    if risk_eta is None:
        risk_eta = getattr(
            env.config, "risk_eta_default", getattr(env.config, "risk_default", 1.0)
        )
    risk_eta = max(float(risk_eta), 1e-6)

    # Baseline risk of RL action on active subset
    sigma_base = None
    try:
        base_var = float(np.matmul(a_rl_opt, np.matmul(cov_matrix, a_rl_opt.T)))
        sigma_base = float(np.sqrt(max(base_var, 0.0)))
    except Exception:
        sigma_base = None

    # Helper to emit single-line status updates (avoid log spam)
    def _emit_status_line(msg: str, end_newline: bool):
        """Emit status on a single updating line; newline only on failures."""
        # Skip when LiveDisplay is active
        if _is_live_display_active():
            return
        try:
            pad = getattr(_emit_status_line, "prev_len", 0)
            clear_pad = " " * max(0, pad - len(msg))
            sys.stdout.write("\r" + msg + clear_pad + ("\n" if end_newline else ""))
            sys.stdout.flush()
            _emit_status_line.prev_len = len(msg)
        except Exception:
            smart_print(msg, flush=True)

    n_opt = len(a_rl_opt)

    installed = list(cp.installed_solvers())
    solver_priority = [
        "ECOS",
        "SCS",
        "CLARABEL",
        "ECOS_BB",
        "MOSEK",
        "GUROBI",
        "OSQP",
        "SCIPY",
    ]
    solver_pool = [s for s in solver_priority if s in installed]
    # Drop QP-only solvers when quadratic risk constraint is present
    sigma_target = sigma_base * risk_eta if sigma_base is not None else None

    # Compute risk_violation for state feedback (spec: Risk Violation Diagnostic)
    # δ_risk = max(0, σ(a^RL) - σ_target)
    # δ_risk = 0: RL compliant, δ_risk > 0: RL needs risk reduction
    if sigma_base is not None and sigma_target is not None:
        env.last_risk_violation = float(max(0.0, sigma_base - sigma_target))
    else:
        env.last_risk_violation = 0.0

    if sigma_target is not None:
        qp_only = {"OSQP", "SCIPY"}
        solver_pool = [s for s in solver_pool if s not in qp_only]
    if not solver_pool:
        solver_pool = [
            s for s in installed if (sigma_target is None or s not in {"OSQP", "SCIPY"})
        ]
    if not solver_pool:
        solver_pool = installed
    if not solver_pool:
        smart_print(
            "[Controller] Warning: No suitable QP solver available in cvxpy installation."
        )
        return a_rl, False

    delta_var = cp.Variable(n_opt)
    a_final_var = a_rl_opt + delta_var
    constraints = [
        cp.sum(delta_var) == 0.0,  # spec: sum adjustment = 0
        a_final_var >= 0,
    ]
    if sigma_target is not None:
        constraints.append(cp.quad_form(a_final_var, cov_matrix) <= (sigma_target**2))

    objective_expr = 0.5 * cp.quad_form(a_final_var, cov_matrix)
    if bias_vec is not None and np.any(np.abs(bias_vec) > 0):
        objective_expr += bias_vec.T @ a_final_var
    if lambda_reg > 0:
        objective_expr += lambda_reg * cp.sum_squares(delta_var)

    problem = cp.Problem(cp.Minimize(objective_expr), constraints)

    solved = False
    solution = None
    solver_used = None
    solver_errors = []
    min_var_risk = _estimate_min_variance_risk(cov_matrix)

    # Adaptive risk boundary: sigma_target = sigma_base * eta
    risk_bound_effective = (
        sigma_target if (sigma_target is not None and sigma_target > 0) else None
    )

    infeasible_risk = False
    infeasible_reason = None
    if (risk_bound_effective is not None) and (min_var_risk is not None):
        if min_var_risk > risk_bound_effective * (1.0 + 1e-5):
            infeasible_risk = True
            infeasible_reason = f"risk_bound {risk_bound_effective:.4f} below min-variance {min_var_risk:.4f}"

    # If no solver available or risk bound infeasible, fall back to min-variance weights
    if ((risk_bound_effective is not None) and (not solver_pool)) or infeasible_risk:
        minvar_w = _min_variance_weights(cov_matrix)
        a_final_opt = minvar_w if minvar_w is not None else a_rl_opt
        solved = True
        solver_used = "fallback-minvar" if minvar_w is not None else "-"
    else:
        for solver in solver_pool:
            try:
                if risk_bound_effective is not None:
                    if len(constraints) < 3:
                        constraints.append(
                            cp.quad_form(a_final_var, cov_matrix)
                            <= (risk_bound_effective**2)
                        )
                    else:
                        constraints[-1] = cp.quad_form(a_final_var, cov_matrix) <= (
                            risk_bound_effective**2
                        )
                    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except Exception as exc:
                solver_errors.append((solver, str(exc)))
                continue
            if (
                problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                and delta_var.value is not None
            ):
                solved = True
                solver_used = solver
                solution = np.array(delta_var.value).flatten()
                break

    if not solved and solver_errors:
        first_solver, first_err = solver_errors[0]
        smart_print(
            f"[Controller] Solver attempts failed. First error ({first_solver}): {first_err}"
        )

    if solved and solution is not None:
        a_final_opt = a_rl_opt + solution
    else:
        # Heuristic fallback: lean toward minimum-variance weights when solver cannot satisfy constraints
        minvar_w = _min_variance_weights(cov_matrix)
        if minvar_w is not None and len(minvar_w) == n_opt:
            a_final_opt = minvar_w
        else:
            a_final_opt = a_rl_opt

    a_final_opt = np.nan_to_num(a_final_opt, nan=0.0, posinf=0.0, neginf=0.0)
    a_final_opt = np.maximum(a_final_opt, 0.0)
    if np.sum(a_final_opt) > 1e-8:
        a_final_opt = a_final_opt / np.sum(a_final_opt)
    else:
        a_final_opt = np.ones(n_opt) / n_opt
    # Top-K output only (no expansion to full universe)
    a_final = a_final_opt

    if solved:
        env.solver_stat["solvable"] = env.solver_stat.get("solvable", 0) + 1
        env.solvable_flag.append(0)
    else:
        env.solver_stat["insolvable"] = env.solver_stat.get("insolvable", 0) + 1
        env.solvable_flag.append(1)

    # Track predicted risk for logging
    try:
        risk_value = float(np.matmul(np.matmul(a_final, cov_matrix), a_final.T))
        risk_value = np.sqrt(max(risk_value, 0.0))
    except Exception:
        risk_value = env.config.risk_market
    env.risk_pred_lst.append(risk_value)

    # Track QP Risk Constraint Enforcement (significant adjustment from RL action)
    delta_l1 = float(np.sum(np.abs(a_final - a_rl_opt)))
    enforcement_threshold = getattr(env.config, "controller_pullback_threshold", 0.1)
    if delta_l1 >= enforcement_threshold:
        if not hasattr(env, "controller_pullback_count"):
            env.controller_pullback_count = 0
        env.controller_pullback_count += 1

        # Format values safely
        sigma_base_str = f"{sigma_base:.4f}" if sigma_base else "N/A"
        sigma_target_str = (
            f"{risk_bound_effective:.4f}" if risk_bound_effective else "N/A"
        )
        eta_str = f"{risk_eta:.4f}" if risk_eta is not None else "1.0"

        # Log QP Risk Constraint Enforcement (spec: solve-agent-spec.md Section 4.2)
        # Use smart_print to route to log file when display is active
        smart_print(
            f"🛡️ [QP RISK CONSTRAINT] Controller thực thi ràng buộc rủi ro | "
            f"‖Δa‖₁={delta_l1:.4f} | σ_base={sigma_base_str} | η={eta_str} | "
            f"σ_target={sigma_target_str} | σ_final={risk_value:.4f} | "
            f"count={env.controller_pullback_count}"
        )

    # Emit compact single-line status (avoid multi-line spam)
    try:
        cur_day = getattr(env, "curTradeDay", None)
        cur_date = None
        if (
            hasattr(env, "curData")
            and env.curData is not None
            and "date" in env.curData
        ):
            try:
                cur_date = env.curData["date"].iloc[0]
            except Exception:
                cur_date = None
        eta_str = f"{risk_eta:.4f}" if risk_eta is not None else "None"
        sigma_base_str = f"{sigma_base:.4f}" if sigma_base is not None else "None"
        sigma_target_str = (
            f"{risk_bound_effective:.4f}"
            if risk_bound_effective is not None
            else "None"
        )
        status = "OK" if solved else "FAIL"
        delta_used = float(np.sum(np.abs(a_final - a_rl)))
        msg = (
            f"[Controller] {status} | day={cur_day} | date={cur_date} | "
            f"sigma_base={sigma_base_str} | eta={eta_str} | sigma_target={sigma_target_str} | "
            f"bound_min_var={min_var_risk if min_var_risk is not None else 'n/a'} | "
            f"solver={solver_used or '-'} | risk_val={risk_value:.4f} | "
            f"l1_used={delta_used:.4f}"
        )
        if infeasible_reason is not None:
            msg += f" | note={infeasible_reason}"
        log_interval = max(1, int(getattr(env.config, "controller_log_interval", 5)))
        cur_day_int = cur_day if isinstance(cur_day, (int, np.integer)) else 0
        if (cur_day_int % log_interval == 0) or (not solved):
            # Update in-place when solved; print newline on failures for visibility
            _emit_status_line(msg, end_newline=not solved)
    except Exception:
        pass

    return a_final, solved
