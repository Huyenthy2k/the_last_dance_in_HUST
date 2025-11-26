# -*- coding: utf-8 -*-
"""
Test that callback can produce Top-K files when actions are present.
"""

import os
import sys
import numpy as np

import types

# Add agents/MAFIA to path for callback imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.callback_func import PoCallback


class DummyEnv:
    def __init__(self, actions, stock_lst):
        self.actions_memory = actions
        self.stock_lst = stock_lst


class DummyConfig:
    def __init__(self, res_dir, rebalance_interval=5, topK=2):
        self.res_dir = res_dir
        self.rebalance_interval = rebalance_interval
        self.topK = topK
        # placeholders required by callback init
        self.mode = "RLcontroller"
        self.enable_market_observer = True
        self.enable_controller = True
        self.mktobs_algo = "mafia_1"
        self.rl_model_name = "TD3"
        self.run_manifest_path = None
        self.metrics_history_path = None
        self.res_img_dir = res_dir


def test_topk_file_created(tmp_path):
    # 10 steps, 3 stocks, interval=5 -> 2 periods
    actions = [np.array([0.4, 0.3, 0.3]) for _ in range(10)]
    env = DummyEnv(actions=actions, stock_lst=["AAA", "BBB", "CCC"])
    cfg = DummyConfig(res_dir=str(tmp_path), rebalance_interval=5, topK=2)
    cb = PoCallback(config=cfg, train_env=env)

    out_path = cb._build_topk_recommendation(env, phase="train", epoch=1)
    assert out_path is not None, "Top-K path should be returned"
    assert os.path.exists(out_path), "Top-K file should be created"
    # Validate content
    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) >= 2  # header + rows


def test_topk_from_actions_file(tmp_path):
    # Create actions CSV fallback (with date column)
    actions = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
    df_path = tmp_path / "train_actions.csv"
    import pandas as pd
    pd.DataFrame(actions, columns=["AAA", "BBB"]).assign(date=["d1", "d2", "d3", "d4", "d5"]).to_csv(df_path, index=False)

    env = DummyEnv(actions=None, stock_lst=["AAA", "BBB"])
    cfg = DummyConfig(res_dir=str(tmp_path), rebalance_interval=2, topK=2)
    cb = PoCallback(config=cfg, train_env=env)
    out_path = cb._build_topk_recommendation(env, phase="train", epoch=1)
    assert out_path is not None
    assert os.path.exists(out_path)
