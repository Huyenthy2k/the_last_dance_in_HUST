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
import pandas as pd
import time
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import cvxpy as cp
from scipy.linalg import sqrtm
import scipy.stats as spstats
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
    a_cbf, is_solvable_status = cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict) 
    cur_dcm_weight= 1.0 
    cur_rl_weight = 1.0
    if is_solvable_status:   
        a_cbf_weighted = a_cbf * cur_dcm_weight
        env.action_cbf_memeory.append(a_cbf_weighted)
        a_rl_weighted = a_rl * cur_rl_weight 
        a_final = a_rl_weighted + a_cbf_weighted
    else:
        env.action_cbf_memeory.append(np.array([0]*env.stock_num))
        a_final = a_rl
    
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

def cbf_opt(env, a_rl, pred_dict):
    """
    The risk constraint is based on controller barrier function (CBF) method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk. 
    
    If mafia_cbf_use_prior is True, uses market_scores_full as prior distribution to guide CBF optimization.
    """
    pred_prices_change = pred_dict['shortterm']
    
    # Clean pred_prices_change to remove any NaNs/infs
    pred_prices_change = np.nan_to_num(pred_prices_change, nan=0.0, posinf=0.0, neginf=0.0)

    a_rl = np.array(a_rl)
    
    # Apply market_scores_full as prior distribution if enabled
    prior_distribution = None
    if getattr(env.config, 'mafia_cbf_use_prior', False) and hasattr(env, 'market_scores_full'):
        if env.market_scores_full is not None and len(env.market_scores_full) == env.stock_num:
            prior_distribution = np.nan_to_num(env.market_scores_full.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize prior
            if np.sum(prior_distribution) > 1e-8:
                prior_distribution = prior_distribution / np.sum(prior_distribution)
            else:
                prior_distribution = np.ones(env.stock_num) / env.stock_num
    
    # Blend a_rl with prior distribution if prior is available
    if prior_distribution is not None:
        prior_weight = getattr(env.config, 'mafia_cbf_prior_weight', 0.3)
        prior_weight = np.clip(prior_weight, 0.0, 1.0)
        # Blend: a_rl = (1 - prior_weight) * a_rl + prior_weight * prior
        a_rl = (1.0 - prior_weight) * a_rl + prior_weight * prior_distribution
    
    # Clean NaN/Inf from a_rl before validation
    a_rl = np.nan_to_num(a_rl, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If a_rl is invalid (all zeros or contains NaN/Inf), use uniform distribution
    if np.any(np.isnan(a_rl)) or np.any(np.isinf(a_rl)) or np.sum(np.abs(a_rl)) < 1e-8:
        a_rl = np.ones(env.stock_num) / env.stock_num
    else:
        # Normalize to ensure sum = 1
        a_rl = a_rl / np.sum(np.abs(a_rl))
    
    # Validate after cleaning
    a_rl_sum = np.sum(np.abs(a_rl))
    if not (0.9999 < a_rl_sum < 1.0001):
        # If still invalid, use uniform distribution
        a_rl = np.ones(env.stock_num) / env.stock_num
    N = env.stock_num
    # Past N days daily return rate of each stock, (num_of_stocks, lookback_days), [[t-N+1, t-N+2, .., t-1, t]]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    
    # OPTIMIZATION: Only compute covariance for stocks with non-zero weights in a_rl
    # This reduces computation from O(288^2) to O(K^2) where K is typically 10-30
    nonzero_mask = np.abs(a_rl) > 1e-6
    nonzero_indices = np.where(nonzero_mask)[0]
    num_nonzero = len(nonzero_indices)
    
    # If too few stocks selected, use all stocks (fallback)
    if num_nonzero < 2:
        nonzero_indices = np.arange(N)
        num_nonzero = N
        nonzero_mask = np.ones(N, dtype=bool)
    
    # Validate and clean daily_return_ay
    daily_return_ay = np.nan_to_num(daily_return_ay, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure daily_return_ay is 2D for consistent processing
    if daily_return_ay.ndim == 1:
        daily_return_ay = daily_return_ay.reshape(-1, 1)
    
    # Check if daily_return_ay has sufficient valid data
    if daily_return_ay.size == 0 or daily_return_ay.shape[0] < 2:
        # Fallback: use identity matrix scaled by risk_market
        cov_r_t0 = np.eye(N) * env.config.risk_market
        risk_stg_t0 = env.config.risk_market
    else:
        try:
            # OPTIMIZED: Only compute covariance for selected stocks
            if num_nonzero < N:
                daily_return_selected = daily_return_ay[nonzero_indices, :]
                cov_selected = np.cov(daily_return_selected)
                
                # Expand to full size with zeros for non-selected stocks
                cov_r_t0 = np.zeros((N, N))
                cov_r_t0[np.ix_(nonzero_indices, nonzero_indices)] = cov_selected
            else:
                # Compute full covariance if all stocks selected
                cov_r_t0 = np.cov(daily_return_ay)
            
            # Clean covariance matrix
            cov_r_t0 = np.nan_to_num(cov_r_t0, nan=0.0, posinf=0.0, neginf=0.0)
            # Ensure covariance matrix has correct shape
            if cov_r_t0.ndim == 0:
                # Scalar case, convert to matrix
                cov_r_t0 = np.array([[cov_r_t0]])
            elif cov_r_t0.ndim == 1:
                # 1D case, convert to 2D
                cov_r_t0 = np.diag(cov_r_t0)
            # Ensure square matrix matches N
            if cov_r_t0.shape[0] != N or cov_r_t0.shape[1] != N:
                # Resize or pad to match N
                if cov_r_t0.shape[0] < N:
                    # Pad with zeros
                    pad_size = N - cov_r_t0.shape[0]
                    cov_r_t0 = np.pad(cov_r_t0, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0.0)
                elif cov_r_t0.shape[0] > N:
                    # Truncate
                    cov_r_t0 = cov_r_t0[:N, :N]
        except Exception as e:
            print(f"Warning: Error computing cov_r_t0: {e}, using fallback")
            cov_r_t0 = np.eye(N) * env.config.risk_market
    
    w_t0 = np.array([env.actions_memory[-1]])
    try:
        risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
        if np.isnan(risk_stg_t0) or np.isinf(risk_stg_t0):
            risk_stg_t0 = env.config.risk_market
    except Exception as error:
        print("Risk-(MV model variance): {}".format(error))
        risk_stg_t0 = env.config.risk_market
    
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        if env.is_last_ctrl_solvable:
            risk_safe_t0 = env.risk_adj_lst[-2]
        else:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0

    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market  
    risk_safe_t1 = env.risk_adj_lst[-1] 

    pred_prices_change_reshape = np.reshape(pred_prices_change, (-1, 1))
    
    # Ensure daily_return_ay has the right shape for appending
    if daily_return_ay.ndim == 1:
        daily_return_ay = daily_return_ay.reshape(-1, 1)
    
    # Check if we can slice daily_return_ay
    if daily_return_ay.shape[1] < 2:
        # Not enough columns, use the whole array
        r_t1 = np.append(daily_return_ay, pred_prices_change_reshape, axis=1)
    else:
        r_t1 = np.append(daily_return_ay[:, 1:], pred_prices_change_reshape, axis=1)
    
    # Clean r_t1 before computing covariance
    r_t1 = np.nan_to_num(r_t1, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute covariance with error handling
    try:
        cov_r_t1 = np.cov(r_t1)
        # Clean covariance matrix
        cov_r_t1 = np.nan_to_num(cov_r_t1, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure covariance matrix has correct shape
        if cov_r_t1.ndim == 0:
            # Scalar case, convert to matrix
            cov_r_t1 = np.array([[cov_r_t1]])
        elif cov_r_t1.ndim == 1:
            # 1D case, convert to 2D
            cov_r_t1 = np.diag(cov_r_t1)
        
        # Ensure square matrix matches N
        if cov_r_t1.shape[0] != N or cov_r_t1.shape[1] != N:
            # Resize or pad to match N
            if cov_r_t1.shape[0] < N:
                # Pad with zeros
                pad_size = N - cov_r_t1.shape[0]
                cov_r_t1 = np.pad(cov_r_t1, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0.0)
            elif cov_r_t1.shape[0] > N:
                # Truncate
                cov_r_t1 = cov_r_t1[:N, :N]
        
        # Check if covariance matrix is valid before sqrtm
        if np.any(np.isnan(cov_r_t1)) or np.any(np.isinf(cov_r_t1)):
            raise ValueError("Covariance matrix contains NaNs or Infs")
        
        # Compute matrix square root with error handling
        try:
            cov_sqrt_t1 = sqrtm(cov_r_t1)
            cov_sqrt_t1 = cov_sqrt_t1.real
            # Clean result
            cov_sqrt_t1 = np.nan_to_num(cov_sqrt_t1, nan=0.0, posinf=0.0, neginf=0.0)
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Warning: sqrtm failed: {e}, using identity matrix fallback")
            # Fallback: use identity matrix scaled by risk_market
            cov_sqrt_t1 = np.eye(N) * np.sqrt(env.config.risk_market)
    except Exception as e:
        print(f"Warning: Error computing cov_r_t1: {e}, using identity matrix fallback")
        # Fallback: use identity matrix scaled by risk_market
        cov_r_t1 = np.eye(N) * env.config.risk_market
        cov_sqrt_t1 = np.eye(N) * np.sqrt(env.config.risk_market)
    G_ay = np.array([]).reshape(-1, N)

    h_0 = np.array([])
    
    use_cvxopt_threshold = 10 # using cvxopt tool will be faster when the size of portfolio is less than or equal to 10. Otherwise, the cvxpy tool will be faster.
    w_lb = 0
    w_ub = 1

    if env.config.topK <= use_cvxopt_threshold:
        # Implemented by cvxopt
        A_eq = np.array([]).reshape(-1, N)
        linear_g1 = np.array([[1.0] * N]) # (1, N)
        A_eq = np.append(A_eq, linear_g1, axis=0)
        A_eq = matrix(A_eq)
        b_eq = np.array([0.0])
        b_eq = matrix(b_eq)

        h_0 = np.append(h_0, a_rl, axis=0) # linear_h3, 0 <= (a_RL + a_cbf)
        h_0 = np.append(h_0, 1-a_rl, axis=0) # linear_h4 (a_RL + a_cbf) <= 1

        linear_g3 = np.diag([-1.0] * N)
        G_ay = np.append(G_ay, linear_g3, axis=0) # 0 <= (a_RL + a_cbf)
        linear_g4 = np.diag([1.0] * N) 
        G_ay = np.append(G_ay, linear_g4, axis=0) # (a_RL + a_cbf) <= 1
    
    else:
        a_rl_re_sign = np.reshape(a_rl, (-1, 1))
        sign_mul = np.ones((1, N))
        w_lb_sign = w_lb
        w_ub_sign = w_ub
        
    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk

    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.005, 0.005, 0.005]
    cnt = 1
    if env.config.is_enable_dynamic_risk_bound:
        cnt_th = env.config.ars_trial # Iterative risk relaxation
    else:
        cnt_th = 1 

    if env.config.topK <= use_cvxopt_threshold:
        # Implemented by cvxopt
        socp_b = np.matmul(cov_sqrt_t1, a_rl)
        h = np.append(h_0, [socp_d], axis=0) # socp_d
        h = np.append(h, socp_b, axis=0) # socp_b
        h = matrix(h)
        socp_cx = np.array([[0.0] * N])
        G_ay = np.append(G_ay, -socp_cx, axis=0)
        G_ay = np.append(G_ay, -cov_sqrt_t1, axis=0) # socp_ax
        G = matrix(G_ay) # G = matrix(np.transpose(np.transpose(G_ay)))

        linear_eq_num = 2*N
        dims = {'l': linear_eq_num, 'q': [N+1], 's': []}
        QP_P = matrix(np.eye(N)) * 2 # (1/2) xP'x
        QP_Q = matrix(np.zeros((N, 1))) # q'x
        while cnt <= cnt_th:
            try:
                sol = solvers.coneqp(QP_P, QP_Q, G, h, dims, A_eq, b_eq)
                if sol['status'] == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt-2]
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
                h = np.append(h_0, [socp_d], axis=0) # socp_d
                h = np.append(h, socp_b, axis=0) # socp_b 
                h = matrix(h)

        if solver_flag:
            if sol['status'] == 'optimal':
                a_cbf = np.reshape(np.array(sol['x']), -1)
                env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
                is_solvable_status = True
                env.risk_adj_lst[-1] = risk_safe_t1
                # Check the solution whether satisfy the risk constraint.
                try:
                    risk_product = np.matmul(np.matmul((a_rl+a_cbf), cov_r_t1), (a_rl+a_cbf).T)
                    cur_alpha_risk = np.sqrt(risk_product) if risk_product >= 0 else 0.0
                    if np.isnan(cur_alpha_risk) or np.isinf(cur_alpha_risk):
                        cur_alpha_risk = env.config.risk_market
                except Exception as e:
                    print(f"Warning: Error computing cur_alpha_risk: {e}, using risk_market fallback")
                    cur_alpha_risk = env.config.risk_market
                assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
                # Note: The constraint is sum(a_cbf) = 0 (adjustment sum), not sum(a_rl + a_cbf) = 0
                # a_rl is already normalized (sum = 1), so sum(a_rl + a_cbf) ≈ 1, not 0
                # The final action will be normalized later in RL_withController
                assert np.abs(np.sum(a_cbf)) <= 0.00001, 'sum of adjustment a_cbf: {} (should be 0) \na_rl: {} \na_cbf: {}'.format(np.sum(a_cbf), a_rl, a_cbf)
                env.solvable_flag.append(0)
            else:
                a_cbf = np.zeros(N)
                env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
                is_solvable_status = False
                try:
                    risk_product = np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T)
                    cur_alpha_risk = np.sqrt(risk_product) if risk_product >= 0 else 0.0
                    if np.isnan(cur_alpha_risk) or np.isinf(cur_alpha_risk):
                        cur_alpha_risk = env.config.risk_market
                except Exception as e:
                    print(f"Warning: Error computing cur_alpha_risk: {e}, using risk_market fallback")
                    cur_alpha_risk = env.config.risk_market
                env.solvable_flag.append(1)           
        else:
            a_cbf = np.zeros(N)
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            try:
                risk_product = np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T)
                cur_alpha_risk = np.sqrt(risk_product) if risk_product >= 0 else 0.0
                if np.isnan(cur_alpha_risk) or np.isinf(cur_alpha_risk):
                    cur_alpha_risk = env.config.risk_market
            except Exception as e:
                print(f"Warning: Error computing cur_alpha_risk: {e}, using risk_market fallback")
                cur_alpha_risk = env.config.risk_market
            env.solvable_flag.append(1)
            # print("Failed to solve the problem.")

    else:
        # Complete solver
        # ++ Implemented by cvxpy
        cp_x = cp.Variable((N, 1))
        a_rl_re = np.reshape(a_rl, (-1, 1))
        cp_constraint = []
        cp_constraint.append(cp.sum(sign_mul@cp_x) + cp.sum(a_rl_re_sign) == 1)
        cp_constraint.append(a_rl_re + cp_x >= w_lb_sign) 
        cp_constraint.append(a_rl_re + cp_x <= w_ub_sign) 
        cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x))) 
        # cvxpy
        cp_prob = None
        while cnt <= cnt_th:
            try:
                obj_f2 = cp.sum_squares(cp_x)
                cp_obj = cp.Minimize(obj_f2)
                cp_prob = cp.Problem(cp_obj, cp_constraint)
                cp_prob.solve(solver=cp.ECOS, verbose=False) 

                if cp_prob.status == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                if cnt <= cnt_th:
                    risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt-2]
                    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x))

        if cp_prob is not None and (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            try:
                risk_product = np.matmul(np.matmul((a_rl+a_cbf), cov_r_t1), (a_rl+a_cbf).T)
                cur_alpha_risk = np.sqrt(risk_product) if risk_product >= 0 else 0.0
                if np.isnan(cur_alpha_risk) or np.isinf(cur_alpha_risk):
                    cur_alpha_risk = env.config.risk_market
            except Exception as e:
                print(f"Warning: Error computing cur_alpha_risk: {e}, using risk_market fallback")
                cur_alpha_risk = env.config.risk_market
            assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d) 
            assert np.abs(np.sum(np.abs(a_rl+a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(np.sum(np.abs((a_rl+a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            a_cbf = np.zeros(N)
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            try:
                risk_product = np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T)
                cur_alpha_risk = np.sqrt(risk_product) if risk_product >= 0 else 0.0
                if np.isnan(cur_alpha_risk) or np.isinf(cur_alpha_risk):
                    cur_alpha_risk = env.config.risk_market
            except Exception as e:
                print(f"Warning: Error computing cur_alpha_risk: {e}, using risk_market fallback")
                cur_alpha_risk = env.config.risk_market
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)    
    env.is_last_ctrl_solvable = is_solvable_status
    if cnt > 1:
        env.stepcount = env.stepcount + 1

    return a_cbf, is_solvable_status
