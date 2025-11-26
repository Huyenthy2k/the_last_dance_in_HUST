# -*- coding: utf-8 -*-
"""
Unit tests for postprocess_topk utilities.
Focus on period bucketing, Top-K aggregation, and constraints (cap/step).
"""

import numpy as np
import os
import sys

# Ensure repo root and agents/MAFIA are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import postprocess_topk as pp


def test_bucket_periods_skips_partial():
    actions = np.arange(8 * 2).reshape(8, 2)  # 8 steps, 2 stocks
    buckets = pp.bucket_periods(actions, interval=3)
    # Expect only full buckets of size 3 => floor(8/3)=2
    assert len(buckets) == 2
    assert all(b.shape == (3, 2) for b in buckets)
    # First bucket should be rows 0:3, second rows 3:6
    np.testing.assert_array_equal(buckets[0], actions[0:3])
    np.testing.assert_array_equal(buckets[1], actions[3:6])


def test_topk_from_period_normalizes_and_sorts():
    period = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.2, 0.3, 0.5],  # last day
        ]
    )
    top_idx, w = pp.topk_from_period(period, k=2)
    # Weights normalized by L1
    assert abs(np.sum(np.abs(w)) - 1.0) < 1e-8
    # Top stocks should be indices with largest absolute weights (2 then 1)
    np.testing.assert_array_equal(top_idx, np.array([2, 1]))


def test_aggregate_topk_basic_flow():
    # 2 periods, interval=2, 4 stocks
    actions = np.array(
        [
            [0.6, 0.4, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0],  # period 1 -> top {0,1}
            [0.1, 0.7, 0.2, 0.0],
            [0.1, 0.7, 0.2, 0.0],  # period 2 -> top {1,2}
        ]
    )
    top_idx, w_final = pp.aggregate_topk(
        actions_history=actions,
        k=2,
        interval=2,
        periods=2,
        alpha=0.5,
        max_cap=0.5,
    )
    assert len(top_idx) == 2
    # Indices should come from stocks that appeared with positive weights
    assert set(top_idx).issubset({0, 1, 2})
    # Weights normalized
    assert abs(np.sum(np.abs(w_final)) - 1.0) < 1e-6


def test_aggregate_topk_respects_max_step_with_prev_weights():
    actions = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.9, 0.1],
            [0.9, 0.1],
        ]
    )
    w_prev = np.array([0.2, 0.8])
    top_idx, w_final = pp.aggregate_topk(
        actions_history=actions,
        k=2,
        interval=2,
        periods=2,
        alpha=0.5,
        max_cap=1.0,
        w_prev=w_prev,
        max_step=0.1,
    )
    # Moves from prev weights by at most max_step
    clipped_diff = np.abs(w_final - w_prev[top_idx])
    assert np.all(clipped_diff <= 0.1 + 1e-8)
