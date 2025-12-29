import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer


class DummyConfig:
    """Lightweight config stub for cleaning tests."""

    def __init__(self):
        self.mafia_trajectory_length = 8
        self.mafia_batch_size = 2
        self.mafia_T_w = 2
        self.mafia_sampling_strategy = "random_trajectory"
        self.mafia_top_k = 2
        self.mafia_pg_reward_horizon = 2
        self.mafia_topk_rebalance_interval = 2
        self.rebalance_interval = 2
        self.mafia_lambda_pg = 1.0
        self.mafia_lambda_risk = 1.0
        self.mafia_lambda_dir = 1.0
        self.mafia_pg_alpha_turnover = 0.01
        self.mafia_pg_alpha_change = 0.01
        self.mafia_direction_threshold = 0.02
        self.regime_vol_k = 3.0
        self.regime_vol_window = 60
        self.mafia_max_grad_norm = 1.0
        self.use_mixed_precision = False
        self.gradient_accumulation_steps = 1
        self.mafia_return_clip_min = -0.5
        self.mafia_return_clip_max = 1.0
        self.mafia_extreme_return_threshold = 1.0
        self.mafia_price_floor = 1e-6


def make_trainer():
    cfg = DummyConfig()
    return ObserverOfflineBatchTrainer(cfg, observer=None, device=torch.device("cpu"))


def test_zero_close_is_ffilled_and_returns_clipped():
    trainer = make_trainer()
    df = pd.DataFrame(
        [
            {"date": pd.Timestamp("2020-01-01"), "stock": "AAA", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1000},
            {"date": pd.Timestamp("2020-01-02"), "stock": "AAA", "open": 10.0, "high": 10.0, "low": 10.0, "close": 0.0, "volume": 1000},
            {"date": pd.Timestamp("2020-01-03"), "stock": "AAA", "open": 12.0, "high": 12.0, "low": 12.0, "close": 12.0, "volume": 1000},
        ]
    )

    cleaned = trainer._clean_price_anomalies(df, price_floor=1e-6, extreme_threshold=1.0)
    assert (cleaned[["open", "high", "low", "close"]] > 0).all().all()

    returns = (
        cleaned.sort_values(["stock", "date"])
        .groupby("stock")["close"]
        .pct_change()
        .fillna(0)
    )
    assert returns.abs().max() <= trainer.config.mafia_return_clip_max + 1e-9


def test_extreme_positive_return_is_capped():
    trainer = make_trainer()
    df = pd.DataFrame(
        [
            {"date": pd.Timestamp("2020-01-01"), "stock": "BBB", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1000},
            {"date": pd.Timestamp("2020-01-02"), "stock": "BBB", "open": 300.0, "high": 300.0, "low": 300.0, "close": 300.0, "volume": 1000},
        ]
    )

    cleaned = trainer._clean_price_anomalies(df, price_floor=1e-6, extreme_threshold=1.0)
    returns = (
        cleaned.sort_values(["stock", "date"])
        .groupby("stock")["close"]
        .pct_change()
        .fillna(0)
    )
    assert returns.iloc[-1] <= trainer.config.mafia_return_clip_max + 1e-9
