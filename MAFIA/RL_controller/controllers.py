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
def RL_withoutController(a_rl, env=None):
    a_cbf = np.array([0]*env.stock_num)
    a_rl = np.array(a_rl)
    env.action_cbf_memeory.append(a_cbf)
    env.action_rl_memory.append(a_rl)
    a_final = a_rl + a_cbf
    return a_final

def RL_withController(a_rl, env=None):
    a_rl = np.array(a_rl)
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
    """
    pred_prices_change = pred_dict['shortterm']
    
    # Clean pred_prices_change to remove any NaNs/infs
    pred_prices_change = np.nan_to_num(pred_prices_change, nan=0.0, posinf=0.0, neginf=0.0)

    a_rl = np.array(a_rl)
    
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