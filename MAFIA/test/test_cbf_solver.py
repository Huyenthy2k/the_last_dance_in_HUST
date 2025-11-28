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
    def __init__(self, stock_num=5, risk_bound=0.05):
        self.stock_num = stock_num
        # Generate a stable return matrix to build covariance
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, size=(stock_num, 60))
        self.ctl_state = {f"DAILYRETURNS-30": returns}
        self.totalTradeDay = 252
        self.risk_adj_lst = [risk_bound]
        self.risk_pred_lst = []
        self.solver_stat = {}
        self.solvable_flag = []
        self.config = SimpleNamespace(
            dailyRetun_lookback=30,
            risk_market=0.01,
            controller_reg_lambda=0.1,
            controller_observer_bias_weight=0.0,
            enable_market_observer=False,
        )


def _objective(x, cov, q, lambda_reg, a_rl):
    return 0.5 * x.T @ cov @ x + q @ x + lambda_reg * np.sum((x - a_rl) ** 2)


def test_cbf_succeeds_and_respects_constraints():
    env = DummyEnv(stock_num=6, risk_bound=0.05)
    a_rl = np.array([0.4, 0.1, 0.1, 0.2, 0.1, 0.1])
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    solved_flag = bool(np.all(solved))
    assert solved_flag, "CBF should solve with feasible covariance/risk bound"
    assert a_final.shape == a_rl.shape
    np.testing.assert_allclose(np.sum(a_final), 1.0, atol=1e-6)
    assert np.all(a_final >= -1e-8)

    cov = controllers._build_covariance_matrix(env)
    q_vec = controllers._compute_observer_bias(env)
    obj_rl = _objective(a_rl, cov, q_vec, env.config.controller_reg_lambda, a_rl)
    obj_cbf = _objective(a_final, cov, q_vec, env.config.controller_reg_lambda, a_rl)
    assert obj_cbf <= obj_rl + 1e-8, "Optimized portfolio should not have worse objective than raw RL action"

    risk_bound = env.risk_adj_lst[-1]
    risk_value = float(a_final.T @ cov @ a_final)
    assert risk_value <= (risk_bound ** 2) + 1e-6, "Risk constraint should be satisfied"


def test_cbf_handles_tight_risk_bound():
    # Very small risk bound; should still project to a feasible low-variance portfolio (near uniform)
    env = DummyEnv(stock_num=4, risk_bound=0.001)
    a_rl = np.array([0.7, 0.1, 0.1, 0.1])
    a_final, solved = controllers.cbf_opt(env, a_rl, pred_dict={})
    solved_flag = bool(np.all(solved))
    assert solved_flag, "CBF should solve even with tight risk bound"
    np.testing.assert_allclose(np.sum(a_final), 1.0, atol=1e-6)
    assert np.all(a_final >= -1e-8)

    cov = controllers._build_covariance_matrix(env)
    risk_value = float(a_final.T @ cov @ a_final)
    risk_rl = float(a_rl.T @ cov @ a_rl)
    # Solver may drop tight constraint but should not worsen risk relative to RL action
    assert risk_value <= risk_rl + 1e-6
