import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

controllers = importlib.import_module("RL_controller.controllers")


class DummyEnv:
    def __init__(
        self,
        stock_num=5,
        risk_eta=1.0,
        market_direction=1,
        noise_scale=0.01,
        alpha_up=1.2,
        alpha_hold=1.0,
        alpha_down=0.7,
        market_vec=None,
    ):
        self.stock_num = stock_num
        # Generate a stable return matrix to build covariance
        rng = np.random.default_rng(0)
        returns = rng.normal(0, noise_scale, size=(stock_num, 60))
        self.ctl_state = {f"DAILYRETURNS-30": returns}
        self.totalTradeDay = 252
        # risk_adj_lst now holds relative tolerance factor eta
        self.risk_adj_lst = [risk_eta]
        self.risk_pred_lst = []
        self.solver_stat = {}
        self.solvable_flag = []
        self.last_mkt_direction_pred = market_direction
        self.market_scores_full = market_vec
        self.config = SimpleNamespace(
            dailyRetun_lookback=30,
            risk_market=0.01,
            controller_reg_lambda=0.1,
            controller_observer_bias_weight=0.0,
            enable_market_observer=False,
            solver_alpha_up=alpha_up,
            solver_alpha_hold=alpha_hold,
            solver_alpha_down=alpha_down,
            solver_alpha_relax_factor=1.05,
            solver_alpha_max=2.0,
            solver_alpha_hard_cap=2.0,
            # risk_eta_* kept for compatibility
            risk_eta_up=alpha_up,
            risk_eta_hold=alpha_hold,
            risk_eta_down=alpha_down,
            risk_eta_default=risk_eta,
        )


def _portfolio_risk(weights, cov):
    return float(weights.T @ cov @ weights)


def test_cbf_succeeds_and_respects_constraints():
    env = DummyEnv(stock_num=6, risk_eta=1.0, market_direction=2)
    a_rl = np.array([0.4, 0.1, 0.1, 0.2, 0.1, 0.1])
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved)), "CBF should solve with feasible covariance/risk bound"
    assert a_final.shape == a_rl.shape
    np.testing.assert_allclose(np.sum(a_final), 1.0, atol=1e-6)
    assert np.all(a_final >= -1e-8)

    delta = a_final - (a_rl / np.sum(a_rl))
    np.testing.assert_allclose(np.sum(delta), 0.0, atol=1e-6)

    cov = controllers._build_covariance_matrix(env)
    eta = env.risk_adj_lst[-1]
    risk_value = _portfolio_risk(a_final, cov)
    normalized_rl = a_rl / np.sum(a_rl)
    sigma_base = np.sqrt(_portfolio_risk(normalized_rl, cov))
    sigma_target = sigma_base * eta
    assert risk_value <= (sigma_target**2) + 1e-6, "Risk constraint should be satisfied"


def test_cbf_handles_tight_risk_bound():
    # Tight risk factor (<1) should reduce risk relative to RL action
    env = DummyEnv(stock_num=4, risk_eta=0.8, market_direction=2, noise_scale=0.02)
    a_rl = np.array([0.7, 0.1, 0.1, 0.1])
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    cov = controllers._build_covariance_matrix(env)
    risk_value = _portfolio_risk(a_final, cov)
    risk_rl = _portfolio_risk(a_rl / np.sum(a_rl), cov)

    np.testing.assert_allclose(np.sum(a_final), 1.0, atol=1e-6)
    assert np.all(a_final >= -1e-8)
    if bool(np.all(solved)):
        eta = env.risk_adj_lst[-1]
        sigma_base = np.sqrt(risk_rl)
        sigma_target = sigma_base * eta
        assert risk_value <= (sigma_target**2) + 1e-6, (
            "Risk bound should hold when solved"
        )
    assert risk_value <= risk_rl + 1e-6, (
        "Optimized portfolio should not worsen risk relative to RL action"
    )


def test_tighter_risk_bound_drives_lower_risk():
    a_rl = np.array([0.3, 0.2, 0.1, 0.25, 0.15])
    env_loose = DummyEnv(
        stock_num=5, risk_eta=1.2, market_direction=1, noise_scale=0.015
    )
    env_tight = DummyEnv(
        stock_num=5, risk_eta=0.8, market_direction=1, noise_scale=0.015
    )

    a_loose, solved_loose = controllers.cbf_opt(env_loose, a_rl, pred_dict={})
    a_tight, solved_tight = controllers.cbf_opt(env_tight, a_rl, pred_dict={})

    cov = controllers._build_covariance_matrix(
        env_loose
    )  # Deterministic given fixed seed/noise
    risk_loose = _portfolio_risk(a_loose, cov)
    risk_tight = _portfolio_risk(a_tight, cov)

    assert bool(np.all(solved_loose))
    assert bool(np.all(solved_tight))
    # Tighter eta should yield lower or equal risk
    assert risk_tight <= risk_loose + 1e-6

    # Compute min-variance risk to check if target is feasible
    # Min-variance portfolio for uniform weights: equal allocation
    n = len(a_rl)
    w_minvar = np.ones(n) / n
    min_var_risk = _portfolio_risk(w_minvar, cov)

    sigma_base = np.sqrt(_portfolio_risk(a_rl / np.sum(a_rl), cov))
    sigma_target_tight = sigma_base * env_tight.risk_adj_lst[-1]
    target_risk_bound = sigma_target_tight**2

    # If target is feasible (above min-variance), check against target
    # If target is infeasible (below min-variance), solver falls back to min-variance
    if target_risk_bound >= min_var_risk:
        assert risk_tight <= target_risk_bound + 1e-6
    else:
        # Solver fell back to min-variance; risk should be at or near min-variance
        assert risk_tight <= min_var_risk + 1e-6


def test_sum_constraint_with_unnormalized_rl_action():
    env = DummyEnv(stock_num=4, risk_eta=1.0, market_direction=1, noise_scale=0.01)
    a_rl = np.array([2.0, 1.0, 0.5, 0.5])  # sums to 4.0, not 1.0
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved))
    np.testing.assert_allclose(np.sum(a_final), 1.0, atol=1e-6)

    normalized_rl = a_rl / np.sum(a_rl)
    delta = a_final - normalized_rl
    np.testing.assert_allclose(np.sum(delta), 0.0, atol=1e-6)
    assert np.all(a_final >= -1e-8)


def test_spec_observer_bias_influences_solution(monkeypatch):
    env = DummyEnv(
        stock_num=5, risk_eta=1.2, market_direction=1, noise_scale=0.02, alpha_hold=1.2
    )
    a_rl = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

    # With zero bias
    monkeypatch.setattr(
        controllers, "_compute_observer_bias", lambda _env: np.zeros(_env.stock_num)
    )
    a_no_bias, solved_base = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved_base))

    # Penalize first asset
    monkeypatch.setattr(
        controllers,
        "_compute_observer_bias",
        lambda _env: np.array([0.2] + [0.0] * (_env.stock_num - 1)),
    )
    a_with_bias, solved_bias = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved_bias))

    assert a_with_bias[0] < a_no_bias[0] - 1e-4, (
        "Positive bias should push weight away from penalized asset"
    )


def test_market_vector_biases_when_enabled():
    # Market vector should attract weight toward higher-scoring assets when bias is enabled
    market_vec = np.array([0.05, 0.1, 0.2, 0.1, 0.05])
    env = DummyEnv(
        stock_num=5,
        risk_eta=1.1,
        market_direction=1,
        noise_scale=0.01,
        market_vec=market_vec,
    )
    env.config.enable_market_observer = True
    env.config.controller_observer_bias_weight = 0.5

    a_rl = np.ones(env.stock_num) / env.stock_num
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved))
    top_idx = np.argmax(market_vec)
    assert a_final[top_idx] >= (1.0 / env.stock_num) + 1e-3


def test_risk_bound_blend_with_topk(monkeypatch):
    """
    Verify adaptive risk bound on Top-K keeps inactive weights at zero and active weights summing to 1.
    """
    N = 6
    K = 3
    sigma = 0.2  # per-asset std used in synthetic cov
    # Force deterministic covariance (sigma^2 * I) regardless of env
    monkeypatch.setattr(
        controllers,
        "_build_covariance_matrix",
        lambda _env: np.eye(_env.stock_num) * (sigma**2),
    )

    env = DummyEnv(stock_num=N, risk_eta=1.0, market_direction=1, noise_scale=0.0)
    env.config.mafia_solver_use_rl_topk_only = True

    a_rl = np.ones(N) / N
    rl_mask = np.array([1] * K + [0] * (N - K), dtype=bool)

    a_final, solved = controllers.cbf_opt(
        env, a_rl, pred_dict={}, rl_selected_mask=rl_mask
    )
    assert bool(np.all(solved))

    # Inactive weights should remain zero
    assert np.allclose(a_final[K:], 0.0, atol=1e-8)
    # Active weights should sum to 1 (within subset)
    assert np.isclose(np.sum(a_final[:K]), 1.0, atol=1e-6)

    # Risk should match min-variance on subset: sigma / sqrt(K)
    expected_risk = sigma / np.sqrt(K)
    risk_value = np.sqrt((a_final[:K] @ (np.eye(K) * (sigma**2)) @ a_final[:K]))
    assert np.isclose(risk_value, expected_risk, atol=1e-6)


def test_min_var_risk_changes_with_different_covariance():
    """
    Test that bound_min_var changes when the covariance matrix changes.
    This verifies the fix for the issue where bound_min_var was always constant.
    """
    stock_num = 10
    rng = np.random.default_rng(42)

    min_var_values = []

    # Test with different covariance matrices (simulating different time steps)
    for seed in [0, 1, 2, 3, 4]:
        rng_step = np.random.default_rng(seed)
        # Generate different return matrices for each "step"
        returns = rng_step.normal(0, 0.02, size=(stock_num, 60))

        env = DummyEnv(stock_num=stock_num, risk_eta=0.7, noise_scale=0.02)
        env.ctl_state = {f"DAILYRETURNS-30": returns}

        # Build covariance and estimate min-variance risk
        cov = controllers._build_covariance_matrix(env)
        min_var = controllers._estimate_min_variance_risk(cov)
        min_var_values.append(min_var)

        print(f"Step {seed}: min_var_risk = {min_var:.10f}")

    # Check that min_var values are different for different covariance matrices
    unique_values = set(round(v, 8) for v in min_var_values if v is not None)
    print(f"Unique min_var values: {unique_values}")

    # With different random seeds, we should get different min_var values
    assert len(unique_values) > 1, (
        f"bound_min_var should vary with different covariance matrices, "
        f"but got only {len(unique_values)} unique value(s): {unique_values}"
    )


def test_historical_daily_returns_matrix_shape():
    """
    Test that _get_historical_daily_returns returns proper matrix shape
    when rawdata is available.
    """
    import pandas as pd

    # Create a mock env with rawdata
    stock_num = 5
    num_days = 60

    # Generate mock rawdata
    dates = pd.date_range("2017-01-01", periods=num_days, freq="D")
    stocks = [f"STOCK_{i}" for i in range(stock_num)]

    data_rows = []
    rng = np.random.default_rng(123)
    for date in dates:
        for stock in stocks:
            data_rows.append(
                {
                    "date": date,
                    "stock": stock,
                    "close": 100 + rng.normal(0, 5),
                    "open": 100 + rng.normal(0, 5),
                    "high": 105 + rng.normal(0, 5),
                    "low": 95 + rng.normal(0, 5),
                    "volume": 1000000,
                }
            )

    rawdata = pd.DataFrame(data_rows)
    rawdata.sort_values(["date", "stock"], inplace=True)
    rawdata.index = rawdata["date"].factorize()[0]

    env = DummyEnv(stock_num=stock_num, risk_eta=0.8)
    env.rawdata = rawdata
    env.curTradeDay = 50  # Simulate being at day 50
    env.stock_lst = stocks

    # Get historical returns
    lookback = 30
    returns = controllers._get_historical_daily_returns(env, lookback)

    assert returns is not None, "Should return a matrix when rawdata is available"
    print(f"Returns matrix shape: {returns.shape}")

    # Shape should be (num_stocks, lookback_days)
    assert returns.shape[0] == stock_num, (
        f"Expected {stock_num} stocks, got {returns.shape[0]}"
    )
    assert returns.shape[1] <= lookback, (
        f"Expected <= {lookback} days, got {returns.shape[1]}"
    )

    # Values should be reasonable daily returns (not all zeros or constants)
    assert not np.allclose(returns, 0.0), "Returns should not be all zeros"
    assert returns.std() > 0, "Returns should have some variance"

    # Check that different days have different returns (matrix is properly built)
    col_stds = np.std(returns, axis=0)
    assert np.any(col_stds > 0), "Different days should have different stock returns"
