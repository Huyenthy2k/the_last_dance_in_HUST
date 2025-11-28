import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

# Ensure project root on path so modules like config/entrance/walk_forward resolve
auto_pipeline = importlib.import_module("scripts.auto_pipeline")
walk_forward = importlib.import_module("walk_forward")
entrance = importlib.import_module("entrance")


def _make_fake_rl(call_log):
    def _fake_rl(cfg):
        """Minimal stub for RLcontroller that writes required artifacts."""
        call_log.append(
            {
                "res_dir": cfg.res_dir,
                "resume_from_checkpoint": cfg.resume_from_checkpoint,
                "num_epochs": getattr(cfg, "num_epochs", None),
            }
        )
        ckpt_dir = Path(cfg.res_dir) / "checkpoints" / "checkpoint_final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_info = ckpt_dir / "checkpoint_info.json"
        with ckpt_info.open("w") as f:
            json.dump({"replay_buffer_path": "replay.buf"}, f)

        test_profile = Path(cfg.res_dir) / "test_profile.csv"
        pd.DataFrame(
            [
                {
                    "sharpeRatio": 1.0,
                    "mdd": -0.1,
                    "annualReturn_pct": 10.0,
                    "netProfit_pct": 12.0,
                    "final_capital": 1234,
                    "reward_sum": 5.0,
                }
            ]
        ).to_csv(test_profile, index=False)

    return _fake_rl


@pytest.fixture(autouse=True)
def set_env(tmp_path, monkeypatch):
    # Keep outputs contained inside a temp directory
    monkeypatch.setenv("MAFIA_RES_ROOT", str(tmp_path / "res"))
    yield


def test_walk_forward_resume_overlap(monkeypatch):
    call_log: List[Dict[str, Any]] = []
    fake_rl = _make_fake_rl(call_log)
    monkeypatch.setattr(walk_forward, "RLcontroller", fake_rl)
    monkeypatch.setattr(entrance, "RLcontroller", fake_rl)

    df, summary, metrics_csv, summary_csv = auto_pipeline.run_walk_forward_stage(
        start_date="2017-01-01",
        num_windows=2,
        train_years=3,
        valid_years=1,
        test_years=1,
        step_years=1,  # Overlaps windows; resume requires resume_overlap=True
        seeds=[1],
        hparam_overrides={"num_epochs": 2},
        resume_overlap=True,
        log_dir=str(Path(os.environ["MAFIA_RES_ROOT"]) / "logs"),
    )

    assert len(call_log) == 2, "Should run one call per window"
    assert call_log[0]["num_epochs"] == 2

    first_ckpt = (
        Path(call_log[0]["res_dir"]) / "checkpoints" / "checkpoint_final" / "checkpoint_info.json"
    )
    assert first_ckpt.exists()
    assert call_log[1]["resume_from_checkpoint"] == str(first_ckpt), "Should reuse checkpoint when resume_overlap=1"

    assert not df.empty
    assert metrics_csv and Path(metrics_csv).exists()
    assert summary_csv and Path(summary_csv).exists()
    assert "sharpeRatio" in df.columns
    assert not summary.empty


def test_walk_forward_overlap_without_resume(monkeypatch):
    call_log: List[Dict[str, Any]] = []
    fake_rl = _make_fake_rl(call_log)
    monkeypatch.setattr(walk_forward, "RLcontroller", fake_rl)
    monkeypatch.setattr(entrance, "RLcontroller", fake_rl)

    df, summary, _, _ = auto_pipeline.run_walk_forward_stage(
        start_date="2017-01-01",
        num_windows=2,
        train_years=3,
        valid_years=1,
        test_years=1,
        step_years=1,  # Overlaps windows
        seeds=[42],
        hparam_overrides=None,
        resume_overlap=False,
        log_dir=str(Path(os.environ["MAFIA_RES_ROOT"]) / "logs_no_resume"),
    )

    assert len(call_log) == 2
    assert call_log[1]["resume_from_checkpoint"] is None, "Should not reuse ckpt when resume_overlap=0 and windows overlap"
    assert not df.empty
    assert not summary.empty
