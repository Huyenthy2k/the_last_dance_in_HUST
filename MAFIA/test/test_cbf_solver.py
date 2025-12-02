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
        risk_bound=0.05,
        market_direction=1,
        noise_scale=0.01,
        alpha_up=0.05,
        alpha_hold=0.08,
        alpha_down=0.12,
        market_vec=None,
    ):
        self.stock_num = stock_num
        # Generate a stable return matrix to build covariance
        rng = np.random.default_rng(0)
        returns = rng.normal(0, noise_scale, size=(stock_num, 60))
        self.ctl_state = {f"DAILYRETURNS-30": returns}
        self.totalTradeDay = 252
        self.risk_adj_lst = [risk_bound]
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
            risk_up_bound=0.025,
            risk_hold_bound=0.013,
            risk_down_bound=0.009,
        )


def _portfolio_risk(weights, cov):
    return float(weights.T @ cov @ weights)


def test_cbf_succeeds_and_respects_constraints():
    env = DummyEnv(stock_num=6, risk_bound=0.05, market_direction=2)
    a_rl = np.array([0.4, 0.1, 0.1, 0.2, 0.1, 0.1])
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved)), "CBF should solve with feasible covariance/risk bound"
    assert a_final.shape == a_rl.shape
    np.testing.assert_allclose(np.sum(a_final), 1.0, atol=1e-6)
    assert np.all(a_final >= -1e-8)

    delta = a_final - (a_rl / np.sum(a_rl))
    np.testing.assert_allclose(np.sum(delta), 0.0, atol=1e-6)

    cov = controllers._build_covariance_matrix(env)
    risk_bound = env.risk_adj_lst[-1]
    risk_value = _portfolio_risk(a_final, cov)
    assert risk_value <= (risk_bound ** 2) + 1e-6, "Risk constraint should be satisfied"


def test_cbf_handles_tight_risk_bound():
    # Tight risk bound should reduce risk relative to RL action
    env = DummyEnv(stock_num=4, risk_bound=0.02, market_direction=2, noise_scale=0.02)
    a_rl = np.array([0.7, 0.1, 0.1, 0.1])
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    cov = controllers._build_covariance_matrix(env)
    risk_value = _portfolio_risk(a_final, cov)
    risk_rl = _portfolio_risk(a_rl / np.sum(a_rl), cov)

    np.testing.assert_allclose(np.sum(a_final), 1.0, atol=1e-6)
    assert np.all(a_final >= -1e-8)
    if bool(np.all(solved)):
        assert risk_value <= (env.risk_adj_lst[-1] ** 2) + 1e-6, "Risk bound should hold when solved"
    assert risk_value <= risk_rl + 1e-6, "Optimized portfolio should not worsen risk relative to RL action"


def test_tighter_risk_bound_drives_lower_risk():
    a_rl = np.array([0.3, 0.2, 0.1, 0.25, 0.15])
    env_loose = DummyEnv(stock_num=5, risk_bound=0.06, market_direction=1, noise_scale=0.015)
    env_tight = DummyEnv(stock_num=5, risk_bound=0.025, market_direction=1, noise_scale=0.015)

    a_loose, solved_loose = controllers.cbf_opt(env_loose, a_rl, pred_dict={})
    a_tight, solved_tight = controllers.cbf_opt(env_tight, a_rl, pred_dict={})

    cov = controllers._build_covariance_matrix(env_loose)  # Deterministic given fixed seed/noise
    risk_loose = _portfolio_risk(a_loose, cov)
    risk_tight = _portfolio_risk(a_tight, cov)

    assert bool(np.all(solved_loose))
    assert bool(np.all(solved_tight))
    assert risk_tight <= risk_loose + 1e-6
    assert risk_tight <= (env_tight.risk_adj_lst[-1] ** 2) + 1e-6


def test_sum_constraint_with_unnormalized_rl_action():
    env = DummyEnv(stock_num=4, risk_bound=0.04, market_direction=1, noise_scale=0.01)
    a_rl = np.array([2.0, 1.0, 0.5, 0.5])  # sums to 4.0, not 1.0
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved))
    np.testing.assert_allclose(np.sum(a_final), 1.0, atol=1e-6)

    normalized_rl = a_rl / np.sum(a_rl)
    delta = a_final - normalized_rl
    np.testing.assert_allclose(np.sum(delta), 0.0, atol=1e-6)
    assert np.all(a_final >= -1e-8)


def test_spec_observer_bias_influences_solution(monkeypatch):
    env = DummyEnv(stock_num=5, risk_bound=0.04, market_direction=1, noise_scale=0.02, alpha_hold=0.2)
    a_rl = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

    # With zero bias
    monkeypatch.setattr(controllers, "_compute_observer_bias", lambda _env: np.zeros(_env.stock_num))
    a_no_bias, solved_base = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved_base))

    # Penalize first asset
    monkeypatch.setattr(controllers, "_compute_observer_bias", lambda _env: np.array([0.2] + [0.0] * (_env.stock_num - 1)))
    a_with_bias, solved_bias = controllers.cbf_opt(env, a_rl, pred_dict={})
    assert bool(np.all(solved_bias))

    assert a_with_bias[0] < a_no_bias[0] - 1e-4, "Positive bias should push weight away from penalized asset"


def test_market_vector_biases_when_enabled():
    # Market vector should attract weight toward higher-scoring assets when bias is enabled
    market_vec = np.array([0.05, 0.1, 0.2, 0.1, 0.05])
    env = DummyEnv(
        stock_num=5,
        risk_bound=0.04,
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
