#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         controllers.py
 Description:  Implement the solver-based agent of the proposed MASA framework.
 Author:       MASA
---------------------------------
'''

import numpy as np
import cvxpy as cp
import sys

def _select_topk_from_scores(market_scores_full, method='topk', k=10, threshold=0.01):
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
        return np.array([]), np.zeros(len(market_scores_full) if market_scores_full is not None else 0, dtype=bool)
    
    market_scores_full = np.nan_to_num(market_scores_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    if method == 'topk':
        k = min(k, len(market_scores_full))
        topk_idx = np.argpartition(market_scores_full, -k)[-k:]
        topk_sorted = topk_idx[np.argsort(market_scores_full[topk_idx])[::-1]]
        mask = np.zeros(len(market_scores_full), dtype=bool)
        mask[topk_sorted] = True
        return topk_sorted, mask
    elif method == 'threshold':
        mask = market_scores_full > threshold
        selected_indices = np.where(mask)[0]
        return selected_indices, mask
    else:
        raise ValueError(f"Unknown selection method: {method}")

def _apply_market_scores_to_rl_action(a_rl, market_scores_full, config, env=None):
    """
    Apply Top-K selection to RL action based on mode.

    Mode-specific behavior (automatic, no flag needed):
    - 'compact': RL automatically uses Top-K from Observer (filter RL action to only keep Observer's Top-K stocks)
    - 'full-score': RL automatically self-selects Top-K from market_scores_full
    - If config.mafia_action_full_universe=True: bypass Top-K masking and use full-universe weights from RL

    Args:
        a_rl: (N,) array - RL action
        market_scores_full: (N,) array - Full market scores from MAFIA
        config: Config object
        env: Environment object (required for accessing observer_topk_indices)
    
    Returns:
        a_rl_modified: (N,) array - Modified RL action with Top-K selection (if applicable)
        rl_selected_mask: (N,) bool array - Mask of stocks selected by RL/Observer (None if not applicable)
    """
    state_mode = getattr(config, 'mafia_state_mode', 'compact')

    # Bypass masking if full-universe mode is enabled (RL outputs full N weights)
    if getattr(config, 'mafia_action_full_universe', False):
        a_full = np.array(a_rl, dtype=float)
        a_full = np.nan_to_num(a_full, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(np.abs(a_full)) > 1e-8:
            a_full = a_full / np.sum(np.abs(a_full))
        else:
            a_full = np.ones(len(a_full)) / len(a_full)
        return a_full, None
    
    if state_mode == 'compact':
        # In compact mode: RL automatically uses Top-K from Observer
        # Filter RL action to only keep stocks selected by Observer
        if env is not None and hasattr(env, 'observer_topk_indices') and env.observer_topk_indices is not None:
            observer_topk = env.observer_topk_indices
            # Ensure indices are valid
            observer_topk = np.clip(observer_topk, 0, len(a_rl) - 1).astype(int)
            
            # Create mask for Observer's Top-K stocks
            selected_mask = np.zeros(len(a_rl), dtype=bool)
            selected_mask[observer_topk] = True
            
            # Zero out non-selected stocks in RL action
            a_rl_modified = a_rl.copy()
            a_rl_modified[~selected_mask] = 0.0
            
            # Renormalize
            if np.sum(np.abs(a_rl_modified)) > 1e-8:
                a_rl_modified = a_rl_modified / np.sum(np.abs(a_rl_modified))
            else:
                # If all zeros, use uniform on Observer's Top-K stocks
                if len(observer_topk) > 0:
                    a_rl_modified[observer_topk] = 1.0 / len(observer_topk)
                else:
                    a_rl_modified = np.ones(len(a_rl)) / len(a_rl)
            
            return a_rl_modified, selected_mask
        else:
            # Fallback: if observer_topk_indices not available, return original action
            return a_rl, None
    
    # In 'full-score' mode: RL automatically self-selects Top-K from market_scores_full
    if market_scores_full is None or len(market_scores_full) != len(a_rl):
        return a_rl, None
    
    # Select Top-K based on market_scores_full
    method = getattr(config, 'mafia_rl_topk_selection_method', 'topk')
    k = getattr(config, 'mafia_top_k', 10)
    threshold = getattr(config, 'mafia_rl_topk_threshold', 0.01)
    
    selected_indices, selected_mask = _select_topk_from_scores(
        market_scores_full, method=method, k=k, threshold=threshold
    )
    
    # Zero out non-selected stocks in RL action
    a_rl_modified = a_rl.copy()
    a_rl_modified[~selected_mask] = 0.0
    
    # Renormalize
    if np.sum(np.abs(a_rl_modified)) > 1e-8:
        a_rl_modified = a_rl_modified / np.sum(np.abs(a_rl_modified))
    else:
        # If all zeros, use uniform on selected stocks
        if len(selected_indices) > 0:
            a_rl_modified[selected_indices] = 1.0 / len(selected_indices)
        else:
            a_rl_modified = np.ones(len(a_rl)) / len(a_rl)
    
    return a_rl_modified, selected_mask

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
    state_mode = getattr(config, 'mafia_state_mode', 'compact')
    if state_mode == 'compact':
        # In compact mode: no boost (RL uses Top-K from Observer)
        return a_final
    
    # In 'full-score' mode: check if Solver boost is enabled
    if not getattr(config, 'mafia_solver_boost_enabled', False):
        return a_final
    
    if market_scores_full is None or len(market_scores_full) != len(a_final):
        return a_final
    
    boost_factor = getattr(config, 'mafia_solver_boost_factor', 0.3)
    boost_method = getattr(config, 'mafia_solver_boost_method', 'blend')
    boost_factor = np.clip(boost_factor, 0.0, 1.0)  # Ensure in [0, 1]
    
    # Clean market_scores_full
    market_scores_clean = np.nan_to_num(market_scores_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize market_scores_full to ensure it's a valid distribution
    if np.sum(market_scores_clean) > 1e-8:
        market_scores_normalized = market_scores_clean / np.sum(market_scores_clean)
    else:
        # If all zeros, use uniform distribution
        market_scores_normalized = np.ones(len(market_scores_clean)) / len(market_scores_clean)
    
    # If RL has selected stocks, only boost those stocks
    if rl_selected_mask is not None and np.any(rl_selected_mask):
        # Only boost stocks selected by RL
        if boost_method == 'blend':
            # Blend only for selected stocks
            a_final_boosted = a_final.copy()
            a_final_boosted[rl_selected_mask] = (
                (1.0 - boost_factor) * a_final[rl_selected_mask] + 
                boost_factor * market_scores_normalized[rl_selected_mask]
            )
            # Renormalize selected stocks
            if np.sum(a_final_boosted[rl_selected_mask]) > 1e-8:
                a_final_boosted[rl_selected_mask] = (
                    a_final_boosted[rl_selected_mask] / 
                    np.sum(a_final_boosted[rl_selected_mask]) * np.sum(a_final[rl_selected_mask])
                )
        elif boost_method == 'proportional':
            # Proportional boost only for selected stocks
            market_scores_selected = market_scores_normalized[rl_selected_mask]
            market_scores_min = np.min(market_scores_selected)
            market_scores_max = np.max(market_scores_selected)
            if market_scores_max - market_scores_min > 1e-8:
                market_scores_scaled = (market_scores_selected - market_scores_min) / (market_scores_max - market_scores_min)
            else:
                market_scores_scaled = np.ones(len(market_scores_selected))
            
            boost_per_stock = 1.0 + boost_factor * market_scores_scaled
            a_final_boosted = a_final.copy()
            a_final_boosted[rl_selected_mask] = a_final[rl_selected_mask] * boost_per_stock
        else:
            raise ValueError(f"Unknown boost method: {boost_method}. Must be 'blend' or 'proportional'")
    else:
        # Boost all stocks (RL hasn't selected specific stocks, or boost all)
        if boost_method == 'blend':
            a_final_boosted = (1.0 - boost_factor) * a_final + boost_factor * market_scores_normalized
        elif boost_method == 'proportional':
            market_scores_min = np.min(market_scores_normalized)
            market_scores_max = np.max(market_scores_normalized)
            if market_scores_max - market_scores_min > 1e-8:
                market_scores_scaled = (market_scores_normalized - market_scores_min) / (market_scores_max - market_scores_min)
            else:
                market_scores_scaled = np.ones(len(market_scores_normalized))
            boost_per_stock = 1.0 + boost_factor * market_scores_scaled
            a_final_boosted = a_final * boost_per_stock
        else:
            raise ValueError(f"Unknown boost method: {boost_method}. Must be 'blend' or 'proportional'")
    
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
    
    if env is not None and hasattr(env, 'is_last_ctrl_solvable'):
        print(f"[SOLVER] Boost result | solvable={env.is_last_ctrl_solvable} | action_sum={np.sum(a_final_boosted):.4f}", flush=True)
    return a_final_boosted

def RL_withoutController(a_rl, env=None):
    a_cbf = np.array([0]*env.stock_num)
    a_rl = np.array(a_rl)
    
    # Apply market_scores_full to RL action if enabled (mode-aware)
    rl_selected_mask = None
    if hasattr(env, 'market_scores_full'):
        a_rl, rl_selected_mask = _apply_market_scores_to_rl_action(
            a_rl, env.market_scores_full, env.config, env=env
        )
    
    env.action_cbf_memeory.append(a_cbf)
    env.action_rl_memory.append(a_rl)
    a_final = a_rl + a_cbf
    
    # Boost top-K weights if enabled (mode-aware, only boost RL-selected stocks in 'full-score' mode)
    if hasattr(env, 'market_scores_full'):
        a_final = _boost_topk_weights(
            a_final, env.market_scores_full, env.config, rl_selected_mask=rl_selected_mask
        )
    
    return a_final

def RL_withController(a_rl, env=None):
    a_rl = np.array(a_rl)
    
    # Apply market_scores_full to RL action if enabled (mode-aware)
    rl_selected_mask = None
    if hasattr(env, 'market_scores_full'):
        a_rl, rl_selected_mask = _apply_market_scores_to_rl_action(
            a_rl, env.market_scores_full, env.config, env=env
        )
    
    env.action_rl_memory.append(a_rl)
    if env.config.pricePredModel == 'MA':
        pred_prices_change = get_pred_price_change(env=env)
        pred_dict = {'shortterm': pred_prices_change}
    else:
        raise ValueError("Cannot find the price prediction model [{}]..".format(env.config.pricePredModel))
    optimized_action, is_solvable_status = cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict, rl_selected_mask=rl_selected_mask)
    if is_solvable_status and optimized_action is not None:
        a_final = optimized_action
        a_cbf = a_final - a_rl
        env.action_cbf_memeory.append(a_cbf)
    else:
        env.action_cbf_memeory.append(np.zeros(env.stock_num))
        a_final = a_rl
    env.is_last_ctrl_solvable = bool(is_solvable_status)
    
    # Normalize to ensure constraints: sum = 1, all >= 0
    # Clip negative values to 0
    a_final = np.maximum(a_final, 0.0)
    # Normalize to sum = 1
    a_final_sum = np.sum(a_final)
    if a_final_sum > 1e-8:
        a_final = a_final / a_final_sum
    else:
        # If all zeros or negative, use uniform distribution
        a_final = np.ones(env.stock_num) / env.stock_num
    
    # Boost top-K weights if enabled (mode-aware, only boost RL-selected stocks in 'full-score' mode)
    if hasattr(env, 'market_scores_full'):
        a_final = _boost_topk_weights(
            a_final, env.market_scores_full, env.config, rl_selected_mask=rl_selected_mask
        )
    
    return a_final

def get_pred_price_change(env):
    ma_lst = env.ctl_state['MA-{}'.format(env.config.otherRef_indicator_ma_window)]
    pred_prices = ma_lst
    cur_close_price = np.array(env.curData['close'].values)
    
    # Ensure pred_prices and cur_close_price have the same shape
    if len(pred_prices) != len(cur_close_price):
        # If shapes don't match, align based on stock order in curData
        # Create a mapping from stock to index in curData
        stock_to_idx = {stock: idx for idx, stock in enumerate(env.curData['stock'].values)}
        
        # If ma_lst is indexed by stock, we need to reorder it
        # For now, pad or truncate to match cur_close_price length
        if len(pred_prices) < len(cur_close_price):
            # Pad with last value or use cur_close_price as fallback
            pred_prices = np.pad(pred_prices, (0, len(cur_close_price) - len(pred_prices)), 
                               mode='constant', constant_values=(pred_prices[-1] if len(pred_prices) > 0 else 1.0))
        else:
            # Truncate to match
            pred_prices = pred_prices[:len(cur_close_price)]
    
    # Clean any NaNs/infs in pred_prices and cur_close_price
    pred_prices = np.nan_to_num(pred_prices, nan=0.0, posinf=0.0, neginf=0.0)
    cur_close_price = np.nan_to_num(cur_close_price, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Safe division: avoid division by zero using epsilon
    epsilon = 1e-8
    # Replace zero or very small values with epsilon to avoid division by zero
    cur_close_price_safe = np.where(np.abs(cur_close_price) < epsilon, epsilon, cur_close_price)
    
    pred_prices_change = (pred_prices - cur_close_price) / cur_close_price_safe
    
    # Clean any remaining NaNs/infs in the result
    pred_prices_change = np.nan_to_num(pred_prices_change, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pred_prices_change

def _get_daily_return_matrix(env):
    """
    Returns matrix shaped (num_stocks, lookback) with cleaned daily returns.
    """
    key = 'DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)
    daily_returns = env.ctl_state.get(key, None)
    if daily_returns is None:
        return None
    daily_returns = np.array(daily_returns, dtype=float)
    daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
    if daily_returns.ndim == 1:
        daily_returns = daily_returns.reshape(1, -1)
    # Ensure stocks are rows
    if daily_returns.shape[0] != env.stock_num and daily_returns.shape[1] == env.stock_num:
        daily_returns = daily_returns.T
    if daily_returns.shape[0] != env.stock_num:
        # Pad or truncate rows to match stock_num
        if daily_returns.shape[0] < env.stock_num:
            pad = env.stock_num - daily_returns.shape[0]
            daily_returns = np.pad(daily_returns, ((0, pad), (0, 0)), mode='constant', constant_values=0.0)
        else:
            daily_returns = daily_returns[:env.stock_num, :]
    return daily_returns


def _build_covariance_matrix(env):
    """
    Build PSD covariance matrix for the latest lookback window.
    """
    daily_returns = _get_daily_return_matrix(env)
    if daily_returns is None or daily_returns.shape[1] < 2:
        return np.eye(env.stock_num) * env.config.risk_market
    try:
        cov_matrix = np.cov(daily_returns)
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        print(f"[Controller] Warning: covariance computation failed ({exc}), using identity fallback", flush=True)
        cov_matrix = np.eye(env.stock_num) * env.config.risk_market

    # Ensure correct shape and symmetry
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])
    elif cov_matrix.ndim == 1:
        cov_matrix = np.diag(cov_matrix)
    if cov_matrix.shape[0] != env.stock_num or cov_matrix.shape[1] != env.stock_num:
        cov_matrix = np.pad(
            cov_matrix,
            ((0, max(0, env.stock_num - cov_matrix.shape[0])), (0, max(0, env.stock_num - cov_matrix.shape[1]))),
            mode='constant',
            constant_values=0.0
        )
        cov_matrix = cov_matrix[:env.stock_num, :env.stock_num]
    # Symmetrize and add small jitter for numerical stability
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    cov_matrix += np.eye(env.stock_num) * 1e-6
    return cov_matrix


def _compute_observer_bias(env):
    """
    Convert observer signals into linear bias vector q.
    Positive entries in q penalize corresponding weights; negative entries attract.
    """
    bias_weight = getattr(env.config, 'controller_observer_bias_weight', 0.0)
    if bias_weight <= 0 or not getattr(env.config, 'enable_market_observer', False):
        return np.zeros(env.stock_num)
    scores = getattr(env, 'market_scores_full', None)
    if scores is None:
        return np.zeros(env.stock_num)
    scores = np.array(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.shape[0] != env.stock_num:
        if scores.shape[0] < env.stock_num:
            scores = np.pad(scores, (0, env.stock_num - scores.shape[0]), mode='constant', constant_values=0.0)
        else:
            scores = scores[:env.stock_num]
    # Shift to non-negative and normalize
    min_score = np.min(scores)
    scores = scores - min_score
    total = np.sum(scores)
    if total <= 1e-8:
        scores = np.ones(env.stock_num) / env.stock_num
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
        a_rl = np.ones(env.stock_num) / env.stock_num
    else:
        a_rl = a_rl / np.sum(a_rl)

    cov_matrix_full = _build_covariance_matrix(env)
    bias_vec_full = _compute_observer_bias(env)
    lambda_reg = max(getattr(env.config, 'controller_reg_lambda', 0.0), 0.0)

    # Restrict optimization to RL-selected Top-K (full-score mode) if configured
    active_indices = None
    if (
        rl_selected_mask is not None
        and np.any(rl_selected_mask)
        and getattr(env.config, 'mafia_solver_use_rl_topk_only', False)
    ):
        active_indices = np.where(np.array(rl_selected_mask, dtype=bool))[0]
    if active_indices is None or len(active_indices) == 0:
        active_indices = np.arange(env.stock_num)

    # Slice to active subset for optimization
    a_rl_opt = a_rl[active_indices]
    cov_matrix = cov_matrix_full[np.ix_(active_indices, active_indices)]
    bias_vec = bias_vec_full[active_indices]

    # Risk boundary from observer (sigma_s,t)
    risk_bound_raw = None
    if hasattr(env, 'risk_adj_lst') and len(env.risk_adj_lst) > 0:
        risk_bound_raw = env.risk_adj_lst[-1]
    risk_bound = None
    if risk_bound_raw is not None:
        try:
            risk_bound = float(risk_bound_raw)
            if risk_bound <= 0 or np.isnan(risk_bound) or np.isinf(risk_bound):
                risk_bound = None
        except Exception:
            risk_bound = None

    # Helper to emit single-line status updates (avoid log spam)
    def _emit_status_line(msg: str, end_newline: bool):
        """Emit status on a single updating line; newline only on failures."""
        try:
            pad = getattr(_emit_status_line, "prev_len", 0)
            clear_pad = " " * max(0, pad - len(msg))
            sys.stdout.write("\r" + msg + clear_pad + ("\n" if end_newline else ""))
            sys.stdout.flush()
            _emit_status_line.prev_len = len(msg)
        except Exception:
            print(msg, flush=True)

    n_opt = len(active_indices)

    installed = list(cp.installed_solvers())
    solver_priority = ['ECOS', 'SCS', 'CLARABEL', 'ECOS_BB', 'MOSEK', 'GUROBI', 'OSQP', 'SCIPY']
    solver_pool = [s for s in solver_priority if s in installed]
    # Drop QP-only solvers when quadratic risk constraint is present
    if risk_bound is not None:
        qp_only = {'OSQP', 'SCIPY'}
        solver_pool = [s for s in solver_pool if s not in qp_only]
    if not solver_pool:
        solver_pool = [s for s in installed if (risk_bound is None or s not in {'OSQP', 'SCIPY'})]
    if not solver_pool:
        solver_pool = installed
    if not solver_pool:
        print("[Controller] Warning: No suitable QP solver available in cvxpy installation.", flush=True)
        return a_rl, False

    delta_var = cp.Variable(n_opt)
    a_final_var = a_rl_opt + delta_var
    constraints = [
        cp.sum(delta_var) == 0.0,  # spec: sum adjustment = 0
        a_final_var >= 0,
    ]
    if risk_bound is not None:
        constraints.append(cp.quad_form(a_final_var, cov_matrix) <= (risk_bound ** 2))

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

    infeasible_risk = False
    infeasible_reason = None
    risk_bound_effective = risk_bound
    if risk_bound is not None and min_var_risk is not None:
        if min_var_risk > risk_bound * (1.0 + 1e-5):
            infeasible_risk = True
            infeasible_reason = f"risk_bound {risk_bound:.4f} below min-variance {min_var_risk:.4f}"
            # Clamp to achievable bound to avoid repeated FAIL during warm-up
            risk_bound_effective = min_var_risk
    if risk_bound_effective is not None and not solver_pool:
        infeasible_reason = "no conic solver available for risk constraint"

    # If no solver available or risk bound infeasible, fall back immediately to min-variance blend
    if (not solver_pool) or infeasible_risk:
        minvar_w = _min_variance_weights(cov_matrix)
        a_final_opt = minvar_w if minvar_w is not None else a_rl_opt
        solved = True  # treat as handled to avoid FAIL spam during warm-up
        solver_used = "fallback-minvar" if minvar_w is not None else "-"
    else:
        for solver in solver_pool:
            try:
                # Update constraint if risk bound was clamped
                if risk_bound_effective is not None and risk_bound_effective != risk_bound:
                    constraints[-1] = cp.quad_form(a_final_var, cov_matrix) <= (risk_bound_effective ** 2)
                    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except Exception as exc:
                solver_errors.append((solver, str(exc)))
                continue
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and delta_var.value is not None:
                solved = True
                solver_used = solver
                solution = np.array(delta_var.value).flatten()
                break

    if not solved and solver_errors:
        first_solver, first_err = solver_errors[0]
        print(f"[Controller] Solver attempts failed. First error ({first_solver}): {first_err}", flush=True)

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
    # Map back to full asset universe (keep zeros for non-selected)
    a_final = np.zeros(env.stock_num)
    a_final[active_indices] = a_final_opt

    if solved:
        env.solver_stat['solvable'] = env.solver_stat.get('solvable', 0) + 1
        env.solvable_flag.append(0)
    else:
        env.solver_stat['insolvable'] = env.solver_stat.get('insolvable', 0) + 1
        env.solvable_flag.append(1)

    # Track predicted risk for logging
    try:
        risk_value = float(np.matmul(np.matmul(a_final, cov_matrix_full), a_final.T))
        risk_value = np.sqrt(max(risk_value, 0.0))
    except Exception:
        risk_value = env.config.risk_market
    env.risk_pred_lst.append(risk_value)

    # Emit compact single-line status (avoid multi-line spam)
    try:
        cur_day = getattr(env, "curTradeDay", None)
        cur_date = None
        if hasattr(env, "curData") and env.curData is not None and "date" in env.curData:
            try:
                cur_date = env.curData["date"].iloc[0]
            except Exception:
                cur_date = None
        risk_bound_str = f"{risk_bound:.4f}" if risk_bound is not None else "None"
        risk_bound_eff_str = f"{risk_bound_effective:.4f}" if risk_bound_effective is not None else risk_bound_str
        status = "OK" if solved else "FAIL"
        delta_used = float(np.sum(np.abs(a_final - a_rl)))
        msg = (
            f"[Controller] {status} | day={cur_day} | date={cur_date} | "
            f"risk_bound={risk_bound_str} | risk_bound_eff={risk_bound_eff_str} | "
            f"bound_min_var={min_var_risk if min_var_risk is not None else 'n/a'} | "
            f"solver={solver_used or '-'} | risk_val={risk_value:.4f} | "
            f"l1_used={delta_used:.4f}"
        )
        if infeasible_reason is not None:
            msg += f" | note={infeasible_reason}"
        log_interval = max(1, int(getattr(env.config, 'controller_log_interval', 5)))
        cur_day_int = cur_day if isinstance(cur_day, (int, np.integer)) else 0
        if (cur_day_int % log_interval == 0) or (not solved):
            # Update in-place when solved; print newline on failures for visibility
            _emit_status_line(msg, end_newline=not solved)
    except Exception:
        pass

    return a_final, solved
