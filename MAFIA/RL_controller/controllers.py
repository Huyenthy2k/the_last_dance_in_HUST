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
    optimized_action, is_solvable_status = cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict)
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


def cbf_opt(env, a_rl, pred_dict):
    """
    Solve QP: minimize 0.5 x^T Σ x + q^T x + λ_reg ||x - a_RL||^2
    subject to sum(x)=1, x>=0, and optional risk boundary.
    """
    del pred_dict  # Price predictions are unused in the deterministic QP objective
    a_rl = np.array(a_rl, dtype=float)
    if np.sum(a_rl) <= 1e-8:
        a_rl = np.ones(env.stock_num) / env.stock_num
    else:
        a_rl = a_rl / np.sum(a_rl)

    cov_matrix = _build_covariance_matrix(env)
    q_vec = _compute_observer_bias(env)
    lambda_reg = max(getattr(env.config, 'controller_reg_lambda', 0.0), 0.0)

    # Risk boundary from observer (sigma)
    risk_bound = None
    if hasattr(env, 'risk_adj_lst') and len(env.risk_adj_lst) > 0:
        risk_bound = env.risk_adj_lst[-1]
    if risk_bound is not None:
        try:
            risk_bound = float(risk_bound)
            if risk_bound <= 0 or np.isnan(risk_bound) or np.isinf(risk_bound):
                risk_bound = None
        except Exception:
            risk_bound = None

    x = cp.Variable(env.stock_num)

    def _build_objective():
        obj = 0.5 * cp.quad_form(x, cov_matrix) + q_vec @ x
        if lambda_reg > 0:
            obj += lambda_reg * cp.sum_squares(x - a_rl)
        return obj

    installed = set(cp.installed_solvers())
    conic_solvers = [s for s in ('ECOS', 'SCS') if s in installed]
    qp_solvers = [s for s in ('OSQP',) if s in installed]
    if not (conic_solvers or qp_solvers):
        print("[Controller] Warning: No suitable QP solver available in cvxpy installation.", flush=True)
        return False, None

    def _solve_problem(include_risk_constraint):
        constraints = [cp.sum(x) == 1, x >= 0]
        if include_risk_constraint and (risk_bound is not None):
            constraints.append(cp.quad_form(x, cov_matrix) <= (risk_bound ** 2))
            constraints.append(cp.norm1(x - a_rl) <= (risk_bound * 2.0))
        problem = cp.Problem(cp.Minimize(_build_objective()), constraints)
        solver_pool = []
        if include_risk_constraint:
            solver_pool.extend(conic_solvers)
        else:
            solver_pool.extend(conic_solvers)
            solver_pool.extend(qp_solvers)
        if not solver_pool:
            return False, None
        for solver in solver_pool:
            try:
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except Exception as solver_error:
                print(f"[Controller] Solver {solver} failed: {solver_error}", flush=True)
                continue
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return True, np.array(x.value).flatten()
        return False, None

    solved, solution = _solve_problem(include_risk_constraint=True)
    if not solved:
        solved, solution = _solve_problem(include_risk_constraint=False)

    if solved and solution is not None:
        a_final = np.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
        a_final = np.maximum(a_final, 0.0)
        if np.sum(a_final) > 1e-8:
            a_final = a_final / np.sum(a_final)
        else:
            a_final = np.ones(env.stock_num) / env.stock_num
        env.solver_stat['solvable'] = env.solver_stat.get('solvable', 0) + 1
        env.solvable_flag.append(0)
    else:
        a_final = a_rl
        solved = False
        env.solver_stat['insolvable'] = env.solver_stat.get('insolvable', 0) + 1
        env.solvable_flag.append(1)

    # Track predicted risk for logging
    try:
        risk_value = float(np.matmul(np.matmul(a_final, cov_matrix), a_final.T))
        risk_value = np.sqrt(max(risk_value, 0.0))
    except Exception:
        risk_value = env.config.risk_market
    env.risk_pred_lst.append(risk_value)

    return a_final, solved
