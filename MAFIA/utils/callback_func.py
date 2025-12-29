# ！/usr/bin/python
# -*- coding: utf-8 -*-#
"""
---------------------------------
 Name: callback_func.py
 Author: MASA
--------------------------------
"""

import numpy as np
import os
import pandas as pd
import time
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFrequencyUnit
from .model_pool import model_select
from . import run_tracker
from .training_logger import TrainingLogger, TrainingPhase, get_logger
from .simple_logger import get_simple_logger, init_simple_logger


def _is_quiet_mode() -> bool:
    """Check if we're in quiet mode (--no-display)."""
    return os.environ.get("MAFIA_NO_LIVE_DISPLAY", "").lower() in ("1", "true", "yes")


# Import LiveDisplay integration
try:
    from .display_integration import (
        update_observer as display_update_observer,
        update_td3 as display_update_td3,
        update_returns as display_update_returns,
        set_phase as display_set_phase,
        log_epoch_end as display_log_epoch_end,
        get_display,
        # 2-Phase Training support
        set_training_mode as display_set_training_mode,
        update_walkforward_iteration as display_update_walkforward_iteration,
        update_walkforward_score as display_update_walkforward_score,
    )

    LIVE_DISPLAY_AVAILABLE = True
except ImportError:
    LIVE_DISPLAY_AVAILABLE = False

import sys
import json
import shutil
import random
import pickle
import torch as th
import math
import importlib
import gzip
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
from RL_controller.controllers import RL_withoutController, RL_withController

# Robust import of postprocess_topk
try:
    import postprocess_topk  # type: ignore
except Exception:
    postprocess_topk = None


def _safe_print(msg: str, level: str = "INFO", **kwargs):
    """Print to log file only when LiveDisplay is active.

    When LiveDisplay is enabled, all print statements would interfere with
    the fixed-layout display. This function routes prints to the log file instead.

    Args:
        msg: Message to print/log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        **kwargs: Additional print arguments (e.g., flush=True) - ignored when logging
    """
    if LIVE_DISPLAY_AVAILABLE:
        display = get_display()
        if display and display.enabled:
            display.log(msg, level)
            return
    print(msg, flush=True)


class PoCallback(BaseCallback):
    def __init__(self, config, train_env, valid_env=None, test_env=None, verbose=0):
        super(PoCallback, self).__init__(verbose)
        self.train_env = train_env
        self.valid_env = valid_env
        self.test_env = test_env
        self.config = config
        if self.config.mode == "RLonly":
            self.risk_controller = RL_withoutController
        elif self.config.mode == "RLcontroller":
            self.risk_controller = RL_withController
        else:
            raise ValueError("Unexpected mode [{}]..".format(self.config.mode))
        # Track last checkpoint epoch
        self.last_checkpoint_epoch = -1
        self.partial_checkpoint_steps = getattr(
            self.config, "partial_checkpoint_steps", 0
        )
        self.last_step_checkpoint = -1
        # Track progress
        self.last_logged_epoch = -1
        self.step_counter = 0
        self.start_time = time.time()
        self._last_speed_time = self.start_time
        self._last_speed_steps = 0
        self.last_known_portfolio = None
        # Checkpoint / replay-buffer housekeeping configuration
        self.enable_checkpoint_cleanup = getattr(
            config, "enable_checkpoint_cleanup", False
        )
        self.max_checkpoints_to_keep = getattr(config, "max_checkpoints_to_keep", 2)
        self.save_replay_buffer_on_step_checkpoints = getattr(
            config, "save_replay_buffer_on_step_checkpoints", False
        )
        self.save_replay_buffer_on_epoch_checkpoints = getattr(
            config, "save_replay_buffer_on_epoch_checkpoints", True
        )
        self.early_stop_patience = getattr(config, "early_stop_patience", 0) or 0
        self.early_stop_min_delta = getattr(config, "early_stop_min_delta", 0.0) or 0.0
        self.early_stop_warmup = getattr(config, "early_stop_warmup", 0) or 0
        self.early_stop_metric = (
            getattr(config, "early_stop_metric", "reward_sum") or "reward_sum"
        )
        # Metrics logging
        default_metrics_path = getattr(self.config, "metrics_history_path", None)
        if not default_metrics_path:
            default_metrics_path = os.path.join(
                self.config.res_dir, "metrics_history.csv"
            )
        self.metrics_file = default_metrics_path
        self.manifest_path = getattr(self.config, "run_manifest_path", None)
        self.metric_fields = [
            "reward_sum",
            "final_capital",
            "annualReturn_pct",
            "netProfit_pct",
            "sharpeRatio",
            "volatility",
            "mdd",
        ]
        # TD3 training diagnostics (populated from TD3_controller.mafia_config)
        self.td3_loss_fields = [
            "td3_actor_loss",
            "td3_critic_loss",
            "td3_mean_reward",
        ]
        # MAFIA observer training diagnostics
        self.mafia_loss_fields = [
            "mafia_loss",
            "mafia_direction_loss",
        ]
        self._rollout_debug_logged = False
        # Skip rollout debug logs in no-display mode
        if os.environ.get("MAFIA_NO_LIVE_DISPLAY", "").lower() in ("1", "true", "yes"):
            self._rollout_debug_logged = True  # Skip debug logs
        self._training_ready = False
        self._warmup_notice_printed = False
        # Track best validation metrics for checkpointing / early stop
        self.best_valid_reward_sum = -np.inf
        self.best_valid_metric = -np.inf
        self.valid_no_improve_epochs = 0

        # Initialize training logger
        self.training_logger = get_logger()
        self.training_logger.set_phase(TrainingPhase.INIT)
        self._early_stop = False
        self.best_valid_checkpoint_path = None

        # Initialize TensorBoard writer
        tb_log_dir = getattr(config, "tensorboard_log", None) or "./tb_logs"
        # Create unique run directory based on res_dir name
        run_name = os.path.basename(self.config.res_dir)
        self.tb_log_path = os.path.join(tb_log_dir, run_name)
        os.makedirs(self.tb_log_path, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=self.tb_log_path)
        if not _is_quiet_mode():
            _safe_print(f"[TensorBoard] Logging to: {self.tb_log_path}")
        # Log frequency for real-time step metrics (every N steps)
        self.tb_log_freq = getattr(config, "tb_log_freq", 10)
        self._last_tb_log_step = 0

    def _on_training_start(self) -> None:
        # Initialize speed baseline to current timestep (handles resume from checkpoint)
        self._last_speed_steps = self.num_timesteps
        self._last_speed_time = time.time()

    def _build_topk_recommendation(self, env, phase, epoch):
        """
        Aggregate actions over recent rebalance periods to produce a stable Top-K.
        Saves CSV in res_dir with symbols and weights.
        """
        # Ensure we can import postprocess_topk even if top-level import failed
        pp_mod = postprocess_topk
        if pp_mod is None:
            try:
                import importlib

                # Try relative to this file's directory (agents/MAFIA)
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if base_dir not in sys.path:
                    sys.path.append(base_dir)
                pp_mod = importlib.import_module("postprocess_topk")
            except Exception as e:
                _safe_print(
                    f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: cannot import postprocess_topk ({e})",
                    flush=True,
                )
                return None

        actions_array = None
        stock_lst = None

        # Primary source: env actions_memory
        if (
            env is not None
            and hasattr(env, "actions_memory")
            and hasattr(env, "stock_lst")
        ):
            stock_lst = getattr(env, "stock_lst")
            actions = getattr(env, "actions_memory", None)
            if actions is not None:
                actions_array = np.array(actions)
                if actions_array.ndim != 2:
                    try:
                        actions_array = actions_array.reshape(len(actions_array), -1)
                    except Exception as e:
                        _safe_print(
                            f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: cannot reshape actions ({e})",
                            flush=True,
                        )
                        actions_array = None
                if (
                    actions_array is not None
                    and actions_array.shape[1] == len(stock_lst) + 1
                ):
                    actions_array = actions_array[:, 1:]  # drop cash

        # Fallback: load actions CSV from res_dir
        if actions_array is None:
            actions_file = os.path.join(
                getattr(self.config, "res_dir", "."), f"{phase}_actions.csv"
            )
            if os.path.exists(actions_file):
                try:
                    df = pd.read_csv(actions_file)
                    if "date" in df.columns:
                        df = df.drop(columns=["date"])
                    actions_array = df.to_numpy()
                    stock_lst = df.columns.tolist() if stock_lst is None else stock_lst
                    _safe_print(
                        f"[POSTPROCESS_TOPK] Using actions from file {actions_file}",
                        flush=True,
                    )
                except Exception as e:
                    _safe_print(
                        f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: failed to load {actions_file} ({e})",
                        flush=True,
                    )
            else:
                _safe_print(
                    f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: actions_memory/file not available",
                    flush=True,
                )
                return None

        if stock_lst is None:
            _safe_print(
                f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: stock list unavailable",
                flush=True,
            )
            return None

        interval = getattr(self.config, "rebalance_interval", 15)
        if actions_array.shape[0] < interval:
            _safe_print(
                f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: only {actions_array.shape[0]} steps < interval {interval}",
                flush=True,
            )
            return None
        periods = max(1, min(actions_array.shape[0] // max(interval, 1), 6))
        k = getattr(self.config, "topK", min(len(stock_lst), 10))
        try:
            top_idx, w_final = pp_mod.aggregate_topk(
                actions_history=actions_array,
                k=k,
                interval=interval,
                periods=periods,
                alpha=0.5,
                max_cap=0.25,
                vol=None,
                w_prev=None,
                max_step=0.1,
            )
        except Exception as e:
            _safe_print(
                f"[POSTPROCESS_TOPK] Skip {phase} epoch {epoch}: {e}", "WARNING"
            )
            return None
        symbols = np.array(env.stock_lst)
        out_dir = getattr(self.config, "res_dir", ".")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"topk_{phase}_epoch{epoch}.csv")
        try:
            pd.DataFrame(
                {
                    "symbol": symbols[top_idx] if len(symbols) > 0 else top_idx,
                    "weight": w_final,
                }
            ).to_csv(out_path, index=False)
            _safe_print(
                f"[POSTPROCESS_TOPK] Saved {phase} Top-K epoch {epoch} to {out_path}"
            )
            return out_path
        except Exception as e:
            _safe_print(
                f"[POSTPROCESS_TOPK] Failed to save {phase} Top-K: {e}", "ERROR"
            )
            return None

    def _run_topk_postprocess_all(self, epoch, run_validation=False, run_test=False):
        if not getattr(self.config, "enable_topk_postprocess", True):
            return
        self._build_topk_recommendation(self.train_env, phase="train", epoch=epoch)
        if run_validation and self.valid_env is not None:
            self._build_topk_recommendation(self.valid_env, phase="valid", epoch=epoch)
        if run_test and self.test_env is not None:
            self._build_topk_recommendation(self.test_env, phase="test", epoch=epoch)

    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only the N most recent ones (based on timesteps).
        Can be disabled via self.enable_checkpoint_cleanup.
        """
        # Skip cleanup if disabled
        if not getattr(self, "enable_checkpoint_cleanup", True):
            # Just log checkpoint count, don't delete anything
            if os.path.exists(self.config.checkpoint_dir):
                checkpoint_dirs = []
                try:
                    items = os.listdir(self.config.checkpoint_dir)
                    for item in items:
                        checkpoint_path = os.path.join(self.config.checkpoint_dir, item)
                        if os.path.isdir(checkpoint_path) and item.startswith(
                            "checkpoint_"
                        ):
                            info_path = os.path.join(
                                checkpoint_path, "checkpoint_info.json"
                            )
                            if os.path.exists(info_path):
                                try:
                                    with open(info_path, "r") as f:
                                        info = json.load(f)
                                        checkpoint_dirs.append(
                                            {
                                                "name": item,
                                                "timesteps": info.get("timesteps", 0),
                                            }
                                        )
                                except Exception:
                                    pass
                except Exception as e:
                    _safe_print(
                        f"[Checkpoint Storage] Error listing checkpoint directory: {e}",
                        flush=True,
                    )
                    return

                if len(checkpoint_dirs) > 0:
                    checkpoint_dirs.sort(key=lambda x: x["timesteps"], reverse=True)
                    _safe_print(
                        f"[Checkpoint Storage] Total checkpoints stored: {len(checkpoint_dirs)} (cleanup disabled)",
                        flush=True,
                    )
                    _safe_print(
                        f"[Checkpoint Storage] Latest checkpoint: {checkpoint_dirs[0]['name']} (timesteps: {checkpoint_dirs[0]['timesteps']})",
                        flush=True,
                    )
                else:
                    _safe_print(
                        f"[Checkpoint Storage] No checkpoints found in {self.config.checkpoint_dir}",
                        flush=True,
                    )
            return

        # Original cleanup logic
        verbose_cleanup = not _is_quiet_mode()
        if not os.path.exists(self.config.checkpoint_dir):
            if verbose_cleanup:
                _safe_print(
                    f"[Checkpoint Cleanup] Checkpoint directory does not exist: {self.config.checkpoint_dir}",
                    flush=True,
                )
            return

        # Get all checkpoint directories
        checkpoint_dirs = []
        try:
            items = os.listdir(self.config.checkpoint_dir)
        except Exception as e:
            if verbose_cleanup:
                _safe_print(
                    f"[Checkpoint Cleanup] Error listing checkpoint directory: {e}",
                    flush=True,
                )
            return

        for item in items:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, item)
            if os.path.isdir(checkpoint_path) and item.startswith("checkpoint_"):
                info_path = os.path.join(checkpoint_path, "checkpoint_info.json")
                if os.path.exists(info_path):
                    try:
                        with open(info_path, "r") as f:
                            info = json.load(f)
                            timesteps = info.get("timesteps", 0)
                            checkpoint_dirs.append(
                                {
                                    "path": checkpoint_path,
                                    "name": item,
                                    "timesteps": timesteps,
                                    "timestamp": info.get("timestamp", ""),
                                }
                            )
                    except Exception as e:
                        if verbose_cleanup:
                            _safe_print(
                                f"[Checkpoint Cleanup] Warning: Could not read checkpoint info from {info_path}: {e}",
                                flush=True,
                            )

        if len(checkpoint_dirs) == 0:
            if verbose_cleanup:
                _safe_print(
                    f"[Checkpoint Cleanup] No checkpoints found in {self.config.checkpoint_dir}",
                    flush=True,
                )
            return

        # Protect best-valid checkpoints from cleanup
        protected = [
            c for c in checkpoint_dirs if c["name"].startswith("checkpoint_best_valid")
        ]
        checkpoint_dirs = [
            c
            for c in checkpoint_dirs
            if not c["name"].startswith("checkpoint_best_valid")
        ]

        # Sort by timesteps (descending - newest first) for non-protected
        checkpoint_dirs.sort(key=lambda x: x["timesteps"], reverse=True)

        if verbose_cleanup:
            _safe_print(
                f"[Checkpoint Cleanup] Found {len(checkpoint_dirs)} checkpoints, keeping {self.max_checkpoints_to_keep} most recent",
                flush=True,
            )

        # Keep only the N most recent checkpoints
        if len(checkpoint_dirs) > self.max_checkpoints_to_keep:
            checkpoints_to_delete = checkpoint_dirs[self.max_checkpoints_to_keep :]
            if verbose_cleanup:
                _safe_print(
                    f"[Checkpoint Cleanup] Will delete {len(checkpoints_to_delete)} old checkpoint(s):",
                    flush=True,
                )
                for checkpoint in checkpoints_to_delete:
                    _safe_print(
                        f"  - {checkpoint['name']} (timesteps: {checkpoint['timesteps']}, timestamp: {checkpoint['timestamp']})",
                        flush=True,
                    )

            deleted_count = 0
            for checkpoint in checkpoints_to_delete:
                try:
                    shutil.rmtree(checkpoint["path"])
                    deleted_count += 1
                    if verbose_cleanup:
                        _safe_print(
                            f"[Checkpoint Cleanup] ✓ Deleted: {checkpoint['name']} (timesteps: {checkpoint['timesteps']})",
                            flush=True,
                        )
                except Exception as e:
                    if verbose_cleanup:
                        _safe_print(
                            f"[Checkpoint Cleanup] ✗ Error deleting {checkpoint['name']}: {e}",
                            flush=True,
                        )

            if verbose_cleanup:
                _safe_print(
                    f"[Checkpoint Cleanup] Cleanup complete: deleted {deleted_count}/{len(checkpoints_to_delete)} checkpoint(s)",
                    flush=True,
                )
        else:
            if verbose_cleanup:
                _safe_print(
                    f"[Checkpoint Cleanup] No cleanup needed: {len(checkpoint_dirs)} checkpoint(s) <= {self.max_checkpoints_to_keep} (max to keep)",
                    flush=True,
                )

        if protected and verbose_cleanup:
            _safe_print(
                f"[Checkpoint Cleanup] Protected best-valid checkpoints: {[c['name'] for c in protected]}",
                flush=True,
            )

    def _save_checkpoint(
        self, checkpoint_name, current_epoch, current_day, checkpoint_type
    ):
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        rl_checkpoint_path = os.path.join(checkpoint_dir, "rl_model.zip")
        self.model.save(rl_checkpoint_path)

        replay_buffer_path = None
        should_save_replay_buffer = True
        if (
            checkpoint_type.startswith("step")
            and not self.save_replay_buffer_on_step_checkpoints
        ):
            should_save_replay_buffer = False
        elif (
            checkpoint_type.startswith("epoch")
            and not self.save_replay_buffer_on_epoch_checkpoints
        ):
            should_save_replay_buffer = False

        if (
            should_save_replay_buffer
            and hasattr(self.model, "replay_buffer")
            and getattr(self.model, "replay_buffer", None) is not None
        ):
            replay_buffer_path = os.path.join(checkpoint_dir, "replay_buffer.pkl")
            try:
                self.model.save_replay_buffer(replay_buffer_path)
                # Compress replay buffer to save disk space
                compressed_path = replay_buffer_path + ".gz"
                try:
                    with (
                        open(replay_buffer_path, "rb") as f_in,
                        gzip.open(compressed_path, "wb") as f_out,
                    ):
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(replay_buffer_path)
                    replay_buffer_path = compressed_path
                    _safe_print(
                        f"[Checkpoint Storage] Replay buffer compressed to {compressed_path}",
                        flush=True,
                    )
                except Exception as e:
                    _safe_print(
                        f"Warning: Replay buffer compression failed (keeping uncompressed): {e}",
                        flush=True,
                    )
            except Exception as e:
                _safe_print(f"Warning: Failed to save replay buffer: {e}", "WARNING")
                replay_buffer_path = None
        elif not should_save_replay_buffer:
            _safe_print(
                f"[Checkpoint Storage] Skipping replay buffer save for {checkpoint_type} checkpoint",
                flush=True,
            )

        mafia_checkpoint_path = None
        if (
            hasattr(self.train_env, "mkt_observer")
            and self.train_env.mkt_observer is not None
            and hasattr(self.train_env.mkt_observer, "save_checkpoint")
        ):
            mafia_checkpoint_path = os.path.join(checkpoint_dir, "mafia_observer.pth")
            try:
                self.train_env.mkt_observer.save_checkpoint(
                    mafia_checkpoint_path, current_epoch
                )
            except Exception as e:
                _safe_print(
                    f"Warning: Failed to save MAFIA observer checkpoint: {e}",
                    flush=True,
                )
                mafia_checkpoint_path = None

        # Save environment state for mid-epoch resume
        env_state_path = None
        if hasattr(self.train_env, "save_state"):
            env_state_path = os.path.join(checkpoint_dir, "env_state.pkl")
            try:
                self.train_env.save_state(env_state_path)
            except Exception as e:
                _safe_print(
                    f"Warning: Failed to save environment state: {e}", "WARNING"
                )
                env_state_path = None

        # Save RNG state for deterministic resume
        rng_state_path = None
        try:
            rng_state_path = os.path.join(checkpoint_dir, "rng_state.pkl")
            rng_state = {
                "seed_num": getattr(self.config, "seed_num", None),
                "python_random": random.getstate(),
                "numpy_random": np.random.get_state(),
                "torch_cpu": th.get_rng_state(),
            }
            if th.cuda.is_available():
                rng_state["torch_cuda"] = th.cuda.get_rng_state_all()
            with open(rng_state_path, "wb") as f:
                pickle.dump(rng_state, f)
        except Exception as e:
            _safe_print(f"Warning: Failed to save RNG state: {e}", "WARNING")
            rng_state_path = None

        checkpoint_info = {
            "type": checkpoint_type,
            "epoch": int(current_epoch),
            "day_in_epoch": int(current_day),
            "timesteps": int(self.num_timesteps),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rl_model_path": rl_checkpoint_path,
            "mafia_observer_path": mafia_checkpoint_path,
            "replay_buffer_path": replay_buffer_path,
            "env_state_path": env_state_path,
            "rng_state_path": rng_state_path,
            "seed_num": getattr(self.config, "seed_num", None),
        }
        info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
        with open(info_path, "w") as f:
            json.dump(checkpoint_info, f, indent=2)
        run_tracker.record_checkpoint_save(
            self.manifest_path, checkpoint_name, checkpoint_info
        )

        # ============================================================
        # Walk-forward Observer checkpoint: Save to walkforward_checkpoint_dir
        # with naming convention: observer_best_{valid_year}.pth
        # This is used by train_observer_walkforward.py to chain iterations
        # ============================================================
        if (
            checkpoint_type == "epoch_best_valid"
            and mafia_checkpoint_path is not None
            and getattr(self.config, "observer_only_training", False)
        ):
            walkforward_ckpt_dir = getattr(
                self.config, "walkforward_checkpoint_dir", None
            )
            valid_year = getattr(self.config, "walkforward_valid_year", None)

            if walkforward_ckpt_dir is not None and valid_year is not None:
                os.makedirs(walkforward_ckpt_dir, exist_ok=True)
                walkforward_ckpt_path = os.path.join(
                    walkforward_ckpt_dir, f"observer_best_{valid_year}.pth"
                )
                try:
                    # Copy the observer checkpoint to walk-forward directory
                    shutil.copy2(mafia_checkpoint_path, walkforward_ckpt_path)
                    _safe_print(
                        f"[Walk-Forward] Observer checkpoint saved: {walkforward_ckpt_path}"
                    )
                    # Update checkpoint_info with walk-forward path
                    checkpoint_info["walkforward_observer_path"] = walkforward_ckpt_path
                    # Re-save info with updated path
                    with open(info_path, "w") as f:
                        json.dump(checkpoint_info, f, indent=2)
                except Exception as e:
                    _safe_print(
                        f"[Walk-Forward] Warning: Failed to save observer checkpoint: {e}",
                        "WARNING",
                    )

        # Clean up old checkpoints after saving new one
        self._cleanup_old_checkpoints()

        return checkpoint_dir

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # ===== Set training mode for dashboard (Phase 1 vs Phase 2) =====
        if LIVE_DISPLAY_AVAILABLE:
            training_mode = getattr(self.config, "training_mode", "RL_ONLY")
            # Use observer_pretrained_path (consistent with entrance.py)
            observer_checkpoint = (
                getattr(self.config, "observer_pretrained_path", "") or ""
            )
            display_set_training_mode(
                mode=training_mode,
                observer_checkpoint=observer_checkpoint,
            )
            # Only log in verbose mode (not --no-display)
            if not _is_quiet_mode():
                _safe_print(f"[Dashboard] Training mode set: {training_mode}")
        
        # ===== Set Walk-Forward context (Phase 1 or 2) =====
        wf_info = getattr(self.config, "walkforward_info", None)
        if wf_info and LIVE_DISPLAY_AVAILABLE:
            display_update_walkforward_iteration(**wf_info)
            if not _is_quiet_mode():
                _safe_print(f"[Dashboard] Walk-Forward context set: {wf_info.get('iteration', '?')}/{wf_info.get('total_iterations', '?')}")

        # Clean up old checkpoints at training start (in case there are old checkpoints from previous runs)
        if not _is_quiet_mode():
            _safe_print(f"[Checkpoint Cleanup] Starting cleanup at training start...")
        self._cleanup_old_checkpoints()
        run_tracker.print_manifest_summary(
            self.manifest_path, heading="RUN STATE SNAPSHOT"
        )
        # Refresh metric visualization immediately so reruns always regenerate the PNG
        self._update_metric_plot()

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        This is called AFTER reset() is called by stable-baselines3, so we can restore state here.
        """
        # Restore environment state if resuming from mid-epoch checkpoint
        if hasattr(self.train_env, "_resume_env_state_path"):
            env_state_path = self.train_env._resume_env_state_path
            if (
                env_state_path
                and os.path.exists(env_state_path)
                and hasattr(self.train_env, "restore_state")
            ):
                try:
                    _safe_print(
                        f"[RESUME] Restoring environment state from {env_state_path}...",
                        flush=True,
                    )
                    self.train_env.restore_state(env_state_path)
                    _safe_print(
                        f"[RESUME] Environment state restored successfully!", flush=True
                    )
                    # Clear the resume path after restoring
                    delattr(self.train_env, "_resume_env_state_path")
                except Exception as e:
                    _safe_print(
                        f"[RESUME] Warning: Failed to restore environment state: {e}",
                        flush=True,
                    )
                    import traceback

                    traceback.print_exc()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # Log training progress
        self.step_counter += 1
        env_epoch = self.train_env.epoch if hasattr(self.train_env, "epoch") else 0
        env_day = getattr(self.train_env, "curTradeDay", 0)
        total_days = (
            self.train_env.totalTradeDay
            if hasattr(self.train_env, "totalTradeDay")
            else 1
        )
        if total_days <= 0:
            total_days = 1

        training_mode = getattr(self.config, "training_mode", "RL_ONLY")
        obs_in_warmup = False
        obs_buffer_size = None
        obs_warmup_target = getattr(self.config, "observer_warmup_samples", 300)

        # Simple terminal logging every 100 steps (less noisy)
        slogger = get_simple_logger()
        if self.step_counter % 100 == 0:
            date_str = ""
            if hasattr(self.train_env, "curData"):
                try:
                    date_str = str(self.train_env.curData["date"].unique()[0])[:10]
                except Exception:
                    pass

            # Get Observer info for OBSERVER_ONLY mode
            extra = ""
            if training_mode == "OBSERVER_ONLY":
                mkt_obs = getattr(self.train_env, "mkt_observer", None)
                if mkt_obs:
                    buf_size = getattr(mkt_obs, "_buffer_size", 0)
                    warmup = getattr(mkt_obs, "warmup_samples", 300)
                    if buf_size < warmup:
                        extra = f"Buffer: {buf_size}/{warmup}"
                    else:
                        loss_dir = getattr(mkt_obs, "_last_loss_direction", 0)
                        loss_eta = getattr(mkt_obs, "_last_loss_eta", 0)
                        loss_pg = getattr(mkt_obs, "_last_loss_pg", 0)
                        if loss_dir or loss_eta or loss_pg:
                            extra = f"L_dir={loss_dir:.3f} L_eta={loss_eta:.3f} L_pg={loss_pg:.3f}"

            slogger.step_progress(
                step=env_day,
                total_steps=total_days,
                epoch=env_epoch,
                total_epochs=self.config.num_epochs,
                date=date_str,
                extra=extra,
            )

        # Derive global epoch/day from global timestep so it stays accurate across resume
        if self.num_timesteps > 0:
            global_step_index = self.num_timesteps - 1  # zero-based index
            completed_epochs_global = global_step_index // total_days
            day_in_epoch_display = (global_step_index % total_days) + 1
        else:
            completed_epochs_global = 0
            day_in_epoch_display = env_day

        max_epochs = max(1, self.config.num_epochs)
        current_epoch_global = min(completed_epochs_global + 1, max_epochs)
        total_timesteps_for_training = total_days * max_epochs

        # Handle case where num_timesteps exceeds total (e.g., when resuming from checkpoint)
        # Calculate progress based on current epoch position, not absolute timesteps
        if self.num_timesteps > total_timesteps_for_training:
            # If we've exceeded total, we're in a later epoch than expected
            # Calculate progress based on epoch completion
            progress_pct = min((current_epoch_global / max_epochs) * 100.0, 100.0)
            # Cap display timesteps to total
            display_timesteps = min(self.num_timesteps, total_timesteps_for_training)
        else:
            progress_pct = (
                (self.num_timesteps / total_timesteps_for_training * 100.0)
                if total_timesteps_for_training > 0
                else 0.0
            )
            progress_pct = min(progress_pct, 100.0)
            display_timesteps = self.num_timesteps

        # Determine training mode (2-phase training)
        training_mode = getattr(self.config, "training_mode", "RL_ONLY")

        # Compute speed based on delta since last log (robust to resume)
        now = time.time()
        delta_steps = self.num_timesteps - self._last_speed_steps
        delta_time = now - self._last_speed_time
        steps_per_sec_delta = delta_steps / delta_time if delta_time > 0 else 0.0
        self._last_speed_steps = self.num_timesteps
        self._last_speed_time = now

        if training_mode == "OBSERVER_ONLY":
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1: Observer Training (TD3 frozen with uniform weights)
            # ═══════════════════════════════════════════════════════════════════
            # Observer warmup is handled in mafia_observer.py
            # Here we update LiveDisplay phase and skip TD3 warmup logic
            mkt_observer = getattr(self.train_env, "mkt_observer", None)
            obs_warmup_complete = (
                getattr(mkt_observer, "_warmup_complete", False)
                if mkt_observer
                else True
            )

            # Track warmup buffer size for progress display
            if mkt_observer:
                obs_buffer_size = max(
                    len(getattr(mkt_observer, "topk_scores_lst", [])),
                    len(getattr(mkt_observer, "risk_eta_pred_tensor_lst", [])),
                )
                try:
                    mkt_observer._buffer_size = obs_buffer_size
                    mkt_observer.warmup_samples = obs_warmup_target
                except Exception:
                    pass

            if not obs_warmup_complete:
                # Observer warmup phase - collecting samples before gradient updates
                obs_in_warmup = True
                if LIVE_DISPLAY_AVAILABLE:
                    display_set_phase(
                        "OBSERVER_PRETRAIN",
                        epoch=current_epoch_global,
                        total_steps=total_days,  # Per-epoch days, not total training steps
                    )
                self.training_logger.set_phase(TrainingPhase.PRETRAIN)
            else:
                # Observer training phase - gradient updates active
                if LIVE_DISPLAY_AVAILABLE:
                    display_set_phase(
                        "OBSERVER_PRETRAIN",
                        epoch=current_epoch_global,
                        total_steps=total_days,  # Per-epoch days, not total training steps
                    )
                self.training_logger.set_phase(TrainingPhase.TRAIN)

            # Skip TD3 warmup logic for Phase 1; gate training-ready until warmup done
            training_ready = not obs_in_warmup
            buffer_size = obs_buffer_size
            learning_starts = obs_warmup_target if obs_in_warmup else 0
        else:
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2: TD3 Training (Observer frozen as Static Expert)
            # ═══════════════════════════════════════════════════════════════════
            # Determine whether warm-up (buffer filling) is still happening
            learning_starts = getattr(self.model, "learning_starts", 0)
            buffer_size = None
            replay_buffer = getattr(self.model, "replay_buffer", None)
            if replay_buffer is not None:
                buffer_size = (
                    replay_buffer.size()
                    if hasattr(replay_buffer, "size")
                    else len(replay_buffer)
                )

            training_ready = self._training_ready
            # If resuming from checkpoint, force-skip warm-up
            if getattr(self.config, "resume_from_checkpoint", None):
                learning_starts = 0
                if hasattr(self.model, "learning_starts"):
                    self.model.learning_starts = 0
                training_ready = True
            elif learning_starts is None or learning_starts <= 0:
                training_ready = True
            elif buffer_size is not None:
                training_ready = buffer_size >= learning_starts

            # Handle warm-up to training transition (Phase 2 only)
            if training_ready and not self._training_ready and learning_starts:
                self.training_logger.print_phase_transition(
                    TrainingPhase.WARMUP,
                    TrainingPhase.TRAIN,
                    f"Replay buffer đạt learning_starts ({buffer_size}/{learning_starts})",
                )
                # Sync LiveDisplay phase
                if LIVE_DISPLAY_AVAILABLE:
                    display_set_phase(
                        "RL_TRAIN",
                        epoch=current_epoch_global,
                        total_steps=total_days,  # Per-epoch days, not total training steps
                    )
            elif not training_ready and buffer_size is not None:
                # Real-time warmup progress on single line
                self.training_logger.set_phase(TrainingPhase.WARMUP)
                if LIVE_DISPLAY_AVAILABLE:
                    # Show actual epoch during warmup (buffer filling can span multiple epochs)
                    display_set_phase(
                        "WARMUP",
                        epoch=current_epoch_global,
                        total_steps=total_days,  # Per-epoch days
                    )
                self.training_logger.print_warmup_status(
                    buffer_size, learning_starts, steps_per_sec_delta
                )

        self._training_ready = training_ready

        # Get portfolio value for display
        portfolio_value = None
        portfolio_return = None
        if hasattr(self.train_env, "cur_capital"):
            pv = self.train_env.cur_capital
            if pv is not None and np.isfinite(pv):
                portfolio_value = pv
                self.last_known_portfolio = pv
                initial = getattr(self.train_env, "initial_asset", 1.0)
                portfolio_return = ((pv / initial) - 1) * 100 if initial else 0.0
            elif self.last_known_portfolio is not None:
                portfolio_value = self.last_known_portfolio
                initial = getattr(self.train_env, "initial_asset", 1.0)
                portfolio_return = (
                    ((portfolio_value / initial) - 1) * 100 if initial else 0.0
                )

        # Log at the start of each new epoch with enhanced banner
        if self._training_ready and current_epoch_global != self.last_logged_epoch:
            self.last_logged_epoch = current_epoch_global
            self.training_logger.set_phase(TrainingPhase.TRAIN)
            self.training_logger.print_epoch_start(
                epoch=current_epoch_global,
                total_epochs=self.config.num_epochs,
                phase=TrainingPhase.TRAIN,
                )

        # Real-time progress update on single line (every step, rate-limited by logger)
        buffer_target = None
        if obs_in_warmup:
            buffer_target = obs_warmup_target
        elif learning_starts:
            buffer_target = learning_starts

        should_update_progress = self._training_ready or obs_in_warmup

        if should_update_progress:
            if self._training_ready:
                self.training_logger.set_phase(TrainingPhase.TRAIN)
            self.training_logger.update_progress(
                step=display_timesteps,
                total_steps=total_timesteps_for_training,
                epoch=current_epoch_global,
                total_epochs=self.config.num_epochs,
                day_in_epoch=day_in_epoch_display,
                total_days=total_days,
                speed=steps_per_sec_delta,
                buffer_size=buffer_size,
                buffer_target=buffer_target,
                portfolio_value=portfolio_value,
                portfolio_return=portfolio_return,
            )
            # Sync LiveDisplay/Dashboard with the same canonical progress numbers
            if LIVE_DISPLAY_AVAILABLE:
                    display = get_display()
                    if display:
                        display.update(
                            # Per-epoch progress
                            step=day_in_epoch_display,
                            total_steps=total_days,
                            epoch_day=day_in_epoch_display,
                            epoch_total_days=total_days,
                            # Global progress
                            total_train_steps=self.num_timesteps,
                            epoch=current_epoch_global,
                            total_epochs=self.config.num_epochs,
                            speed=steps_per_sec_delta,
                        )
                        if display.enabled:
                            display.render()

        # Real-time TensorBoard logging (every tb_log_freq steps)
        if (
            self.tb_writer is not None
            and self.num_timesteps > 0
            and self.num_timesteps - self._last_tb_log_step >= self.tb_log_freq
        ):
            self._log_step_to_tensorboard(
                step=self.num_timesteps,
                buffer_size=buffer_size,
                portfolio_value=portfolio_value,
                portfolio_return=portfolio_return,
                speed=steps_per_sec_delta,
            )
            self._last_tb_log_step = self.num_timesteps

        # Update LiveDisplay with TD3 and Observer metrics
        if LIVE_DISPLAY_AVAILABLE and self._training_ready:
            self._update_live_display_metrics(
                epoch=current_epoch_global,
                buffer_size=buffer_size,
                portfolio_return=portfolio_return,
            )

        # Step-based checkpointing
        current_day = env_day

        if (
            self.partial_checkpoint_steps
            and self.num_timesteps > 0
            and self.num_timesteps % self.partial_checkpoint_steps == 0
            and self.num_timesteps != self.last_step_checkpoint
            and self._training_ready
        ):
            checkpoint_name = f"checkpoint_step_{self.num_timesteps}"
            checkpoint_dir = self._save_checkpoint(
                checkpoint_name=checkpoint_name,
                current_epoch=current_epoch_global,
                current_day=current_day,
                checkpoint_type="step",
            )
            _safe_print(
                f"Checkpoint saved: {checkpoint_dir} (step {self.num_timesteps})",
                flush=True,
            )
            self.last_step_checkpoint = self.num_timesteps

        # Save model
        if self.train_env.model_save_flag:
            exclusive_start_cputime = time.process_time()
            exclusive_start_systime = time.perf_counter()
            curmpath = os.path.join(self.config.res_model_dir, "current_model")
            self.model.save(curmpath)
            validation_freq = getattr(self.config, "validation_freq", None)
            should_validate = (
                validation_freq is not None
                and validation_freq > 0
                and current_epoch_global % validation_freq == 0
            ) or (current_epoch_global >= self.config.num_epochs)
            epoch_metrics = {"train": None, "valid": None, "test": None}
            train_profile = getattr(self.train_env, "last_epoch_profile", None)
            if train_profile is None:
                train_profile = getattr(self.train_env, "latest_invest_profile", None)
            if train_profile is None:
                try:
                    train_profile = self.train_env.get_results()
                except Exception:
                    train_profile = None
            epoch_metrics["train"] = train_profile

            # Save checkpoint if enabled and epoch matches frequency
            if (
                self.config.checkpoint_freq > 0
                and completed_epochs_global > 0
                and completed_epochs_global % self.config.checkpoint_freq == 0
                and completed_epochs_global != self.last_checkpoint_epoch
            ):
                checkpoint_epoch = completed_epochs_global
                checkpoint_name = f"checkpoint_epoch_{checkpoint_epoch}"
                checkpoint_dir = self._save_checkpoint(
                    checkpoint_name=checkpoint_name,
                    current_epoch=checkpoint_epoch,
                    current_day=current_day,
                    checkpoint_type="epoch",
                )

                self.training_logger.print_checkpoint_saved(
                    checkpoint_name, "epoch", checkpoint_epoch
                )
                self.last_checkpoint_epoch = checkpoint_epoch
            # Evaluate model in validation/test only after the final epoch
            ModelCls = model_select(
                model_name=self.config.rl_model_name, mode=self.config.mode
            )
            trained_model = None
            # Run validation every validation_freq; run test only on final epoch
            run_validation = should_validate and self.valid_env is not None
            # Run test after each epoch (can be costly)
            run_test = should_validate and self.test_env is not None
            if run_validation or run_test:
                trained_model = ModelCls.load(curmpath)

            valid_profile = None
            if run_validation:
                # Phase transition: TRAIN -> VALID
                self.training_logger.print_phase_transition(
                    TrainingPhase.TRAIN,
                    TrainingPhase.VALID,
                    f"Bắt đầu đánh giá validation cho Epoch {current_epoch_global}",
                )
                self.training_logger.set_phase(TrainingPhase.VALID)
                # Sync LiveDisplay phase
                if LIVE_DISPLAY_AVAILABLE:
                    display_set_phase("VALID", epoch=current_epoch_global)

                if hasattr(self.valid_env, "validation_mode"):
                    self.valid_env.validation_mode = True
                try:
                    obs_valid = self.valid_env.reset()
                    if isinstance(obs_valid, tuple):
                        obs_valid = obs_valid[0]
                    valid_step = 0
                    valid_total_days = getattr(self.valid_env, "totalTradeDay", 252)
                    while True:
                        a_rlonly, _ = trained_model.predict(obs_valid)
                        a_rlonly = np.reshape(a_rlonly, (-1))
                        a_rl = a_rlonly
                        if np.sum(np.abs(a_rl)) == 0:
                            a_rl = np.array([1 / len(a_rl)] * len(a_rl))
                        else:
                            a_rl = a_rl / np.sum(np.abs(a_rl))
                        a_final = self.risk_controller(a_rl=a_rl, env=self.valid_env)
                        a_final = a_final / np.sum(np.abs(a_final))
                        a_final = np.array([a_final])
                        step_result = self.valid_env.step(a_final)
                        if len(step_result) == 5:
                            obs_valid, rewards, terminal_flag, _, _ = step_result
                        else:
                            obs_valid, rewards, terminal_flag, _ = step_result
                        valid_step += 1
                        # Real-time validation progress
                        self.training_logger.update_progress(
                            step=valid_step,
                            total_steps=valid_total_days,
                            epoch=current_epoch_global,
                            total_epochs=self.config.num_epochs,
                            day_in_epoch=valid_step,
                            total_days=valid_total_days,
                            speed=0,
                            force=valid_step % 50 == 0,
                        )
                        if terminal_flag:
                            break
                    valid_profile = self.valid_env.get_results()
                    epoch_metrics["valid"] = valid_profile
                    # Persist validation profile to CSV (validation_mode skips save_profile inside env)
                    try:
                        if valid_profile is not None:
                            self.valid_env.save_profile(valid_profile)
                    except Exception as e:
                        self.training_logger.print_error(
                            "Lưu validation profile thất bại", e
                        )
                except Exception as e:
                    self.training_logger.print_error("Validation thất bại", e)
                    valid_profile = None
                finally:
                    if hasattr(self.valid_env, "validation_mode"):
                        self.valid_env.validation_mode = False
                # Early stopping / best-valid tracking
                is_best = False
                if valid_profile is not None:
                    reward_val = valid_profile.get("reward_sum", None)
                    metric_val = valid_profile.get(self.early_stop_metric, None)
                    if metric_val is None or not np.isfinite(metric_val):
                        metric_val = (
                            reward_val
                            if reward_val is not None and np.isfinite(reward_val)
                            else None
                        )
                    if reward_val is not None and np.isfinite(reward_val):
                        self.best_valid_reward_sum = max(
                            self.best_valid_reward_sum, reward_val
                        )
                    if metric_val is not None:
                        improved = metric_val > (
                            self.best_valid_metric + self.early_stop_min_delta
                        )
                        if improved:
                            self.best_valid_metric = metric_val
                            self.valid_no_improve_epochs = 0
                            is_best = True
                            # Use completed_epochs_global (the epoch that just finished)
                            # NOT current_epoch_global (which points to the next epoch)
                            # This fixes off-by-one bug where checkpoint was labeled epoch N+1
                            # but actually contained model from epoch N
                            best_valid_epoch = max(1, completed_epochs_global)

                            # Expanding Window mode: Use custom checkpoint naming (Ckpt_Best_YYYY)
                            expanding_ckpt_name = getattr(
                                self.config, "expanding_checkpoint_name", None
                            )
                            if expanding_ckpt_name and getattr(
                                self.config, "expanding_window_mode", False
                            ):
                                ckpt_name = expanding_ckpt_name
                            else:
                                ckpt_name = "checkpoint_best_valid"

                            try:
                                checkpoint_dir = self._save_checkpoint(
                                    checkpoint_name=ckpt_name,
                                    current_epoch=best_valid_epoch,
                                    current_day=current_day,
                                    checkpoint_type="epoch_best_valid",
                                )
                                self.best_valid_checkpoint_path = checkpoint_dir
                                self.training_logger.print_checkpoint_saved(
                                    ckpt_name,
                                    "best_valid",
                                    best_valid_epoch,
                                )
                            except Exception as e:
                                self.training_logger.print_error(
                                    "Lưu best-valid checkpoint thất bại", e
                                )
                        else:
                            self.valid_no_improve_epochs += 1
                            if (
                                self.early_stop_patience
                                and current_epoch_global >= self.early_stop_warmup
                                and self.valid_no_improve_epochs
                                >= self.early_stop_patience
                            ):
                                self.training_logger.print_banner(
                                    f"EARLY STOPPING - Không cải thiện {self.valid_no_improve_epochs} epochs",
                                    TrainingPhase.ERROR,
                                )
                                self._early_stop = True

                # Print validation completion with metrics
                slogger = get_simple_logger()
                slogger.validation_end(
                    current_epoch_global,
                    valid_profile if valid_profile else {},
                    is_best=is_best,
                )
                self.training_logger.print_validation_complete(
                    current_epoch_global,
                    valid_profile if valid_profile else {},
                    is_best=is_best,
                )

            if run_test:
                # Phase transition: VALID -> TEST
                self.training_logger.print_phase_transition(
                    TrainingPhase.VALID,
                    TrainingPhase.TEST,
                    f"Bắt đầu đánh giá trên tập test",
                )
                self.training_logger.set_phase(TrainingPhase.TEST)
                # Sync LiveDisplay phase
                if LIVE_DISPLAY_AVAILABLE:
                    display_set_phase("TEST", epoch=current_epoch_global)

                obs_test = self.test_env.reset()
                if isinstance(obs_test, tuple):
                    obs_test = obs_test[0]
                test_step = 0
                test_total_days = getattr(self.test_env, "totalTradeDay", 252)
                while True:
                    a_rlonly, _ = trained_model.predict(obs_test)
                    a_rlonly = np.reshape(a_rlonly, (-1))
                    a_rl = a_rlonly
                    if np.sum(np.abs(a_rl)) == 0:
                        a_rl = np.array([1 / len(a_rl)] * len(a_rl))
                    else:
                        a_rl = a_rl / np.sum(np.abs(a_rl))
                    a_final = self.risk_controller(a_rl=a_rl, env=self.test_env)
                    a_final = a_final / np.sum(np.abs(a_final))
                    a_final = np.array([a_final])
                    step_result = self.test_env.step(a_final)
                    if len(step_result) == 5:
                        obs_test, rewards, terminal_flag, _, _ = step_result
                    else:
                        obs_test, rewards, terminal_flag, _ = step_result
                    test_step += 1
                    # Real-time test progress
                    self.training_logger.update_progress(
                        step=test_step,
                        total_steps=test_total_days,
                        epoch=current_epoch_global,
                        total_epochs=self.config.num_epochs,
                        day_in_epoch=test_step,
                        total_days=test_total_days,
                        speed=0,
                        force=test_step % 50 == 0,
                    )
                    if terminal_flag:
                        break
                try:
                    test_profile = self.test_env.get_results()
                    epoch_metrics["test"] = test_profile
                    try:
                        if test_profile is not None:
                            self.test_env.save_profile(test_profile)
                    except Exception as e:
                        self.training_logger.print_error("Lưu test profile thất bại", e)

                    # Print test completion with metrics (simple logger)
                    slogger = get_simple_logger()
                    slogger.test_end(
                        current_epoch_global, test_profile if test_profile else {}
                    )

                    self.training_logger.print_test_complete(
                        test_profile if test_profile else {}
                    )
                except Exception as e:
                    self.training_logger.print_error("Đánh giá test thất bại", e)
            if not run_validation and not run_test:
                self.training_logger.newline()
                _safe_print(
                    "[CALLBACK] Bỏ qua validation/test cho epoch này (sẽ chạy sau epoch cuối).",
                    flush=True,
                )

            # Post-process Top-K recommendations for train/valid/test
            try:
                self._run_topk_postprocess_all(
                    current_epoch_global,
                    run_validation=run_validation,
                    run_test=run_test,
                )
            except Exception as e:
                _safe_print(f"[POSTPROCESS_TOPK] Aggregation failed: {e}", "ERROR")

            if trained_model is not None:
                del trained_model
            # delete the current model file
            current_model_path = os.path.join(
                self.config.res_model_dir, "current_model.zip"
            )
            if os.path.exists(current_model_path):
                try:
                    os.remove(current_model_path)
                except Exception as e:
                    _safe_print(
                        f"Warning: Could not remove {current_model_path}: {e}",
                        flush=True,
                    )
            exclusive_end_cputime = time.process_time()
            exclusive_end_systime = time.perf_counter()
            self.train_env.exclusive_cputime = (
                exclusive_end_cputime - exclusive_start_cputime
            )
            self.train_env.exclusive_systime = (
                exclusive_end_systime - exclusive_start_systime
            )

            # Transition back to TRAIN phase for next epoch
            self.training_logger.set_phase(TrainingPhase.TRAIN)
            # Sync LiveDisplay phase - use appropriate phase based on training mode
            if LIVE_DISPLAY_AVAILABLE:
                training_mode = getattr(self.config, "training_mode", "RL_ONLY")
                if training_mode == "OBSERVER_ONLY":
                    display_set_phase(
                        "OBSERVER_PRETRAIN",
                        epoch=current_epoch_global,
                        total_steps=total_days,
                    )
                else:
                    display_set_phase(
                        "RL_TRAIN",
                        epoch=current_epoch_global,
                        total_steps=total_days,
                    )

            # ===== ENHANCED: Comprehensive epoch training summary =====
            self._print_epoch_training_summary(current_epoch_global, epoch_metrics)

            # Log epoch completion with enhanced output
            epoch_complete_metrics = {}
            train_metrics = epoch_metrics.get("train") if epoch_metrics else None
            if isinstance(train_metrics, dict):
                for key in ["final_capital", "sharpeRatio", "annualReturn_pct", "mdd"]:
                    if key in train_metrics:
                        epoch_complete_metrics[key] = train_metrics[key]

            # Use completed_epochs_global for recording metrics (the epoch that just finished)
            # NOT current_epoch_global (which points to the next epoch)
            recording_epoch = max(1, completed_epochs_global)

            # Simple terminal logging for epoch end
            slogger = get_simple_logger()
            if train_metrics:
                slogger.epoch_end(recording_epoch, train_metrics)

            self.training_logger.print_epoch_complete(
                recording_epoch, epoch_complete_metrics
            )
            self._record_metrics(recording_epoch, epoch_metrics)

        self.train_env.model_save_flag = False
        if self._early_stop:
            self.training_logger.print_banner(
                f"EARLY STOPPING - Không cải thiện {self.valid_no_improve_epochs} epochs",
                TrainingPhase.ERROR,
            )
            _safe_print(
                f"   Best {self.early_stop_metric}: {self.best_valid_metric:.6f} | "
                f"Best reward_sum: {self.best_valid_reward_sum:.6f}",
                flush=True,
            )
            if self.best_valid_checkpoint_path:
                _safe_print(
                    f"   ✅ Best validation checkpoint saved at: {self.best_valid_checkpoint_path}",
                    flush=True,
                )
            return False

        # Check if we've reached total timesteps - stop training to prevent extra steps
        if self.num_timesteps >= total_timesteps_for_training:
            self.training_logger.print_banner(
                f"TRAINING HOÀN THÀNH - Đã đạt {self.num_timesteps:,} timesteps",
                TrainingPhase.COMPLETE,
            )
            return False  # Stop training

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        This is called AFTER collect_rollouts() returns, but BEFORE train() is called by base class.
        """
        # Debug: Check if training should be triggered
        if hasattr(self.model, "replay_buffer"):
            buffer_size = (
                self.model.replay_buffer.size()
                if hasattr(self.model.replay_buffer, "size")
                else len(self.model.replay_buffer)
            )
            learning_starts = getattr(self.model, "learning_starts", 100)
            n_updates_before = getattr(self.model, "_n_updates", 0)
            train_freq = getattr(self.model, "train_freq", None)

            if not self._rollout_debug_logged:
                _safe_print(
                    f"[ROLLOUT_END] Before training check | Buffer: {buffer_size}/{learning_starts} | _n_updates: {n_updates_before} | train_freq: {train_freq}",
                    flush=True,
                )

            # Check if training should happen (base class will check this)
            if train_freq is not None:
                # Extract unit from train_freq (could be TrainFreq object or tuple)
                if hasattr(train_freq, "unit"):
                    train_freq_unit = train_freq.unit
                elif isinstance(train_freq, (tuple, list)) and len(train_freq) == 2:
                    train_freq_unit = (
                        TrainFrequencyUnit.EPISODE
                        if train_freq[1] == "episode"
                        else TrainFrequencyUnit.STEP
                    )
                else:
                    train_freq_unit = None

                if not self._rollout_debug_logged:
                    if train_freq_unit and hasattr(train_freq_unit, "name"):
                        _safe_print(
                            f"[ROLLOUT_END] train_freq unit: {train_freq_unit.name}",
                            flush=True,
                        )
                    else:
                        _safe_print(
                            f"[ROLLOUT_END] train_freq unit: {train_freq_unit}",
                            flush=True,
                        )

                # Check the actual conditions for training
                should_train = buffer_size >= learning_starts and (
                    (
                        train_freq_unit == TrainFrequencyUnit.EPISODE
                        and hasattr(self.model, "_episode_num")
                        and self.model._episode_num > 0
                    )
                    or (
                        train_freq_unit == TrainFrequencyUnit.STEP
                        and self.num_timesteps % train_freq.frequency == 0
                    )
                )
                if not self._rollout_debug_logged:
                    _safe_print(
                        f"[ROLLOUT_END] Should train: {should_train} (buffer_size >= learning_starts: {buffer_size >= learning_starts}, train_freq_unit: {train_freq_unit})",
                        flush=True,
                    )
                    self._rollout_debug_logged = True

    def _print_epoch_training_summary(self, epoch, epoch_metrics):
        """
        Print comprehensive training summary at epoch end.
        Phase-aware: Shows different content for Phase 1 (OBSERVER_ONLY) vs Phase 2 (RL_ONLY).
        """
        config = self.config
        env = self.train_env

        # Determine training mode (2-phase training)
        training_mode = getattr(config, "training_mode", "RL_ONLY")

        # Format helper
        def fmt(val, decimals=6):
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                return "N/A"
            return f"{val:.{decimals}f}"

        # Get train profile metrics (common for both phases)
        train_profile = epoch_metrics.get("train", {}) if epoch_metrics else {}
        if train_profile is None:
            train_profile = {}
        reward_sum = train_profile.get("reward_sum", None)
        final_capital = train_profile.get("final_capital", None)
        sharpe = train_profile.get("sharpeRatio", None)
        mdd = train_profile.get("mdd", None)
        annual_return = train_profile.get("annualReturn_pct", None)

        if training_mode == "OBSERVER_ONLY":
            # ═══════════════════════════════════════════════════════════════════════════════
            # PHASE 1: Observer Walk-Forward Training (TD3 frozen)
            # ═══════════════════════════════════════════════════════════════════════════════
            # Gather MAFIA Observer losses
            mafia_total = getattr(config, "last_mafia_loss", None)
            mafia_eta_loss = getattr(config, "last_mafia_eta_loss", None)
            mafia_direction_loss = getattr(config, "last_mafia_direction_loss", None)
            mafia_eta_mae = getattr(config, "last_mafia_eta_mae", None)
            mafia_pg_loss = getattr(config, "last_mafia_pg_loss", None)

            # PG reward components
            pg_mean_return = 0.0
            pg_turnover_penalty = 0.0
            pg_change_penalty = 0.0
            pg_shaped_return = 0.0
            if hasattr(env, "mkt_observer") and env.mkt_observer is not None:
                obs = env.mkt_observer
                pg_mean_return = getattr(obs, "last_pg_mean_return", 0.0)
                pg_turnover_penalty = getattr(obs, "last_pg_turnover_penalty", 0.0)
                pg_change_penalty = getattr(obs, "last_pg_change_penalty", 0.0)
                pg_shaped_return = getattr(obs, "last_pg_shaped_return", 0.0)

            # Walk-forward info
            wf_iteration = getattr(config, "walkforward_iteration", 0)
            wf_total = getattr(config, "walkforward_total_iterations", 5)
            wf_train_years = getattr(config, "walkforward_train_years", "")
            wf_valid_year = getattr(config, "walkforward_valid_year", 0)

            _safe_print(
                f"\n{'═' * 100}\n"
                f"🔮 [PHASE 1 - EPOCH {epoch}] Observer Walk-Forward Training Summary\n"
                f"{'═' * 100}\n"
                f"\n┌─ WALK-FORWARD ITERATION {wf_iteration + 1}/{wf_total} ───────────────────────────────────────────┐\n"
                f"│  📅 Train Years: {wf_train_years:20s}  │  Valid Year: {wf_valid_year:>10}            │\n"
                f"└─────────────────────────────────────────────────────────────────────────────────┘\n"
                f"\n┌─ MAFIA OBSERVER (Quan sát & Dự đoán thị trường) ── ● TRAINING ───────────────┐\n"
                f"│  📊 Loss Tổng (Total Loss):                              {fmt(mafia_total):>15}  │\n"
                f"│  ├── 1️⃣  Loss Stock Selection (Policy Gradient):          {fmt(mafia_pg_loss):>15}  │\n"
                f"│  ├── 2️⃣  Loss Market Direction (Cross-Entropy):          {fmt(mafia_direction_loss):>15}  │\n"
                f"│  └── 3️⃣  Loss Risk Tolerance η (MSE Regression):         {fmt(mafia_eta_loss):>15}  │\n"
                f"│        └── Risk η MAE (Mean Absolute Error):            {fmt(mafia_eta_mae):>15}  │\n"
                f"├─────────────────────────────────────────────────────────────────────────────────┤\n"
                f"│  📈 PG Reward Breakdown:                                                        │\n"
                f"│  └── R_t = mean_return - α_turnover×turnover - α_change×symdiff                │\n"
                f"│      ├── mean_return:      {fmt(pg_mean_return, 4):>12}                                     │\n"
                f"│      ├── turnover_penalty: {fmt(pg_turnover_penalty, 4):>12}                                     │\n"
                f"│      ├── change_penalty:   {fmt(pg_change_penalty, 4):>12}                                     │\n"
                f"│      └── R_t (shaped):     {fmt(pg_shaped_return, 4):>12}                                     │\n"
                f"└─────────────────────────────────────────────────────────────────────────────────┘\n"
                f"\n┌─ TD3 PORTFOLIO ALLOCATOR ── ❄️ FROZEN (Uniform Weights) ───────────────────────┐\n"
                f"│  📊 Status: NOT TRAINING (Phase 1 - Observer only)                             │\n"
                f"│  📊 Actions: Using uniform weights [1/K, 1/K, ..., 1/K]                        │\n"
                f"└─────────────────────────────────────────────────────────────────────────────────┘\n"
                f"\n┌─ PORTFOLIO PERFORMANCE (Simulated with Uniform Weights) ───────────────────────┐\n"
                f"│  💰 Vốn cuối kỳ (Final Capital):                        ${fmt(final_capital, 2) if final_capital else 'N/A':>14}  │\n"
                f"│  📈 Lợi nhuận năm (Annual Return):                       {fmt(annual_return, 2) if annual_return else 'N/A':>14}%  │\n"
                f"│  📊 Sharpe Ratio:                                        {fmt(sharpe, 4) if sharpe else 'N/A':>15}  │\n"
                f"│  📉 Max Drawdown (MDD):                                  {fmt(mdd, 4) if mdd else 'N/A':>15}  │\n"
                f"└─────────────────────────────────────────────────────────────────────────────────┘\n"
                f"{'═' * 100}",
                flush=True,
            )
        else:
            # ═══════════════════════════════════════════════════════════════════════════════
            # PHASE 2: TD3 Training (Observer frozen as Static Expert)
            # ═══════════════════════════════════════════════════════════════════════════════
            # Gather TD3 losses
            td3_actor = getattr(config, "last_td3_actor_loss", None)
            td3_critic = getattr(config, "last_td3_critic_loss", None)
            td3_reward = getattr(config, "last_td3_mean_reward", None)
            td3_updates = getattr(config, "last_td3_updates", None)
            td3_reward_sum = getattr(config, "td3_reward_sum", None)

            # TD3 reward components
            td3_r_return = getattr(config, "last_td3_r_return", None)
            td3_r_js = getattr(config, "last_td3_r_js", None)
            td3_js_div = getattr(config, "last_td3_js_divergence", None)

            # Gather risk management stats from environment
            regime_shift_count = getattr(env, "regime_shift_count", 0)
            controller_pullback_count = getattr(env, "controller_pullback_count", 0)
            cbf_interventions = getattr(env, "cbf_interventions", 0)

            # Observer checkpoint source
            obs_checkpoint = getattr(config, "observer_checkpoint_source", "N/A")
            if len(obs_checkpoint) > 50:
                obs_checkpoint = "..." + obs_checkpoint[-47:]

            _safe_print(
                f"\n{'═' * 100}\n"
                f"🎯 [PHASE 2 - EPOCH {epoch}] TD3 Training Summary\n"
                f"{'═' * 100}\n"
                f"\n┌─ MAFIA OBSERVER ── ❄️ FROZEN (Static Expert) ──────────────────────────────────┐\n"
                f"│  📊 Status: NOT TRAINING (Phase 2 - TD3 only)                                  │\n"
                f"│  📁 Checkpoint: {obs_checkpoint:60s}  │\n"
                f"└─────────────────────────────────────────────────────────────────────────────────┘\n"
                f"\n┌─ TD3 PORTFOLIO ALLOCATOR ── ● TRAINING ─────────────────────────────────────────┐\n"
                f"│  📊 Actor Loss (Policy Network):                        {fmt(td3_actor):>15}  │\n"
                f"│  📊 Critic Loss (Q-Network):                            {fmt(td3_critic):>15}  │\n"
                f"│  📈 Mean Reward per Update (trung bình):                {fmt(td3_reward):>15}  │\n"
                f"│  📈 Sum Reward (tổng trong epoch):                      {fmt(td3_reward_sum, 4):>15}  │\n"
                f"│  🔢 Số lần Gradient Updates:                            {td3_updates if td3_updates else 'N/A':>15}  │\n"
                f"├─────────────────────────────────────────────────────────────────────────────────┤\n"
                f"│  📈 TD3 Reward Breakdown:                                                       │\n"
                f"│  └── r_total = (r_return + r_js) * reward_scale                                │\n"
                f"│      ├── r_return:         {fmt(td3_r_return, 4):>12}                                     │\n"
                f"│      ├── r_js:             {fmt(td3_r_js, 4):>12}                                     │\n"
                f"│      └── JS_divergence:    {fmt(td3_js_div, 4):>12}                                     │\n"
                f"└─────────────────────────────────────────────────────────────────────────────────┘\n"
                f"\n┌─ RISK MANAGEMENT & CBF (Quản lý rủi ro) ──────────────────────────────────────┐\n"
                f"│  🚨 Số lần Regime Shift (thay đổi chế độ thị trường):   {regime_shift_count:>15}  │\n"
                f"│  🛡️  Số lần Controller kéo về mức an toàn:               {controller_pullback_count:>15}  │\n"
                f"│  ⚠️  CBF Safety Interventions:                           {cbf_interventions:>15}  │\n"
                f"└─────────────────────────────────────────────────────────────────────────────────┘\n"
                f"\n┌─ PORTFOLIO PERFORMANCE (Hiệu suất danh mục) ───────────────────────────────────┐\n"
                f"│  💰 Vốn cuối kỳ (Final Capital):                        ${fmt(final_capital, 2) if final_capital else 'N/A':>14}  │\n"
                f"│  📈 Lợi nhuận năm (Annual Return):                       {fmt(annual_return, 2) if annual_return else 'N/A':>14}%  │\n"
                f"│  📊 Sharpe Ratio:                                        {fmt(sharpe, 4) if sharpe else 'N/A':>15}  │\n"
                f"│  📉 Max Drawdown (MDD):                                  {fmt(mdd, 4) if mdd else 'N/A':>15}  │\n"
                f"│  🎁 Tổng Reward (Sum Rewards):                           {fmt(reward_sum, 4) if reward_sum else 'N/A':>15}  │\n"
                f"└─────────────────────────────────────────────────────────────────────────────────┘\n"
                f"{'═' * 100}",
                flush=True,
            )

            # Reset accumulators for next epoch (only for Phase 2)
            if hasattr(config, "td3_reward_sum"):
                config.td3_reward_sum = 0.0
            if hasattr(env, "regime_shift_count"):
                env.regime_shift_count = 0
            if hasattr(env, "controller_pullback_count"):
                env.controller_pullback_count = 0

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        try:
            current_epoch = getattr(self.train_env, "epoch", self.config.num_epochs)
            current_day = getattr(self.train_env, "curTradeDay", 0)
            checkpoint_dir = self._save_checkpoint(
                checkpoint_name="checkpoint_final",
                current_epoch=current_epoch,
                current_day=current_day,
                checkpoint_type="epoch_final",
            )
            _safe_print(f"[Checkpoint] Final checkpoint saved: {checkpoint_dir}")
        except Exception as e:
            _safe_print(f"[Checkpoint] Failed to save final checkpoint: {e}", "ERROR")

        # ============================================================
        # Walk-forward: Ensure Observer checkpoint is saved at training end
        # If no best_valid was saved (e.g., early stopping or no improvement),
        # use the final checkpoint as the walk-forward checkpoint
        # ============================================================
        if getattr(self.config, "observer_only_training", False):
            walkforward_ckpt_dir = getattr(
                self.config, "walkforward_checkpoint_dir", None
            )
            valid_year = getattr(self.config, "walkforward_valid_year", None)

            if walkforward_ckpt_dir is not None and valid_year is not None:
                walkforward_ckpt_path = os.path.join(
                    walkforward_ckpt_dir, f"observer_best_{valid_year}.pth"
                )

                # Only save if no best_valid checkpoint was saved yet
                if not os.path.exists(walkforward_ckpt_path):
                    try:
                        # Get Observer checkpoint from final checkpoint
                        final_mafia_path = os.path.join(
                            self.config.checkpoint_dir,
                            "checkpoint_final",
                            "mafia_observer.pth",
                        )
                        if os.path.exists(final_mafia_path):
                            os.makedirs(walkforward_ckpt_dir, exist_ok=True)
                            shutil.copy2(final_mafia_path, walkforward_ckpt_path)
                            _safe_print(
                                f"[Walk-Forward] Final Observer checkpoint saved: {walkforward_ckpt_path}"
                            )
                    except Exception as e:
                        _safe_print(
                            f"[Walk-Forward] Warning: Failed to save final observer checkpoint: {e}",
                            "WARNING",
                        )

        # Close TensorBoard writer
        if hasattr(self, "tb_writer") and self.tb_writer is not None:
            try:
                self.tb_writer.close()
                _safe_print(
                    f"[TensorBoard] Writer closed. Logs saved to: {self.tb_log_path}"
                )
            except Exception as e:
                _safe_print(
                    f"[TensorBoard] Warning: Failed to close writer: {e}", "WARNING"
                )

    def _record_metrics(self, epoch, metrics_payload):
        rows = []
        for phase, profile in metrics_payload.items():
            if profile is None:
                continue
            rows.append(self._build_metric_row(epoch, phase, profile))
        if not rows:
            return
        df = pd.DataFrame(rows)
        # Append latest TD3 loss snapshots (same across phases for a given epoch)
        td3_snapshot = {
            "td3_actor_loss": getattr(self.config, "last_td3_actor_loss", np.nan),
            "td3_critic_loss": getattr(self.config, "last_td3_critic_loss", np.nan),
            "td3_mean_reward": getattr(self.config, "last_td3_mean_reward", np.nan),
        }
        mafia_snapshot = {
            "mafia_loss": getattr(self.config, "last_mafia_loss", np.nan),
            "mafia_direction_loss": getattr(
                self.config, "last_mafia_direction_loss", np.nan
            ),
            "mafia_eta_loss": getattr(self.config, "last_mafia_eta_loss", np.nan),
            "mafia_pg_loss": getattr(self.config, "last_mafia_pg_loss", np.nan),
        }
        for key, val in td3_snapshot.items():
            df[key] = val
        for key, val in mafia_snapshot.items():
            df[key] = val
        header = not os.path.exists(self.metrics_file)
        df.to_csv(self.metrics_file, mode="a", header=header, index=False)
        run_tracker.record_metrics_update(
            self.manifest_path, epoch, df["phase"].tolist()
        )
        _safe_print(
            f"[VISUALIZER] Logged metrics for epoch {epoch} phases: {', '.join(df['phase'].tolist())}",
            flush=True,
        )

        # Log metrics to TensorBoard
        self._log_to_tensorboard(epoch, metrics_payload, td3_snapshot, mafia_snapshot)

        self._update_metric_plot()

    def _log_step_to_tensorboard(
        self, step, buffer_size, portfolio_value, portfolio_return, speed
    ):
        """Log real-time step metrics to TensorBoard."""
        if self.tb_writer is None:
            return

        # Log buffer size
        if buffer_size is not None:
            self.tb_writer.add_scalar("realtime/buffer_size", buffer_size, step)

        # Log portfolio metrics
        if portfolio_value is not None and np.isfinite(portfolio_value):
            self.tb_writer.add_scalar("realtime/portfolio_value", portfolio_value, step)
        if portfolio_return is not None and np.isfinite(portfolio_return):
            self.tb_writer.add_scalar(
                "realtime/portfolio_return_pct", portfolio_return, step
            )

        # Log speed
        if speed is not None and np.isfinite(speed):
            self.tb_writer.add_scalar("realtime/steps_per_sec", speed, step)

        # Log latest reward from environment
        if hasattr(self.train_env, "last_reward"):
            reward = getattr(self.train_env, "last_reward", None)
            if reward is not None and np.isfinite(reward):
                self.tb_writer.add_scalar("realtime/reward", reward, step)

        # Log TD3 losses if available (updated during training)
        td3_actor = getattr(self.config, "last_td3_actor_loss", None)
        td3_critic = getattr(self.config, "last_td3_critic_loss", None)
        if td3_actor is not None and np.isfinite(td3_actor):
            self.tb_writer.add_scalar("realtime/td3_actor_loss", td3_actor, step)
        if td3_critic is not None and np.isfinite(td3_critic):
            self.tb_writer.add_scalar("realtime/td3_critic_loss", td3_critic, step)

        # Log MAFIA losses if available
        mafia_loss = getattr(self.config, "last_mafia_loss", None)
        if mafia_loss is not None and np.isfinite(mafia_loss):
            self.tb_writer.add_scalar("realtime/mafia_loss", mafia_loss, step)

        # Log MAFIA component losses for detailed monitoring
        mafia_eta_loss = getattr(self.config, "last_mafia_eta_loss", None)
        if mafia_eta_loss is not None and np.isfinite(mafia_eta_loss):
            self.tb_writer.add_scalar("realtime/mafia_eta_loss", mafia_eta_loss, step)

        mafia_direction_loss = getattr(self.config, "last_mafia_direction_loss", None)
        if mafia_direction_loss is not None and np.isfinite(mafia_direction_loss):
            self.tb_writer.add_scalar(
                "realtime/mafia_direction_loss", mafia_direction_loss, step
            )

        mafia_pg_loss = getattr(self.config, "last_mafia_pg_loss", None)
        if mafia_pg_loss is not None and np.isfinite(mafia_pg_loss):
            self.tb_writer.add_scalar("realtime/mafia_pg_loss", mafia_pg_loss, step)

    def _update_live_display_metrics(
        self,
        epoch: int = 0,
        buffer_size: int = 0,
        portfolio_return: float = 0.0,
    ):
        """Update LiveDisplay with TD3 and Observer metrics."""
        if not LIVE_DISPLAY_AVAILABLE:
            return

        try:
            # Get TD3 metrics from config
            td3_actor = getattr(self.config, "last_td3_actor_loss", 0.0)
            td3_critic = getattr(self.config, "last_td3_critic_loss", 0.0)
            td3_mean_q = getattr(self.config, "last_td3_mean_q", 0.0)
            buffer_max = getattr(self.config, "buffer_size", 156000)
            learning_starts = getattr(self.model, "learning_starts", 1000)
            lr = getattr(self.config, "learning_rate", 0.0001)
            noise_sigma = getattr(self.config, "action_noise_sigma", 0.15)

            # Get reward stats
            if (
                hasattr(self.train_env, "reward_lst")
                and len(self.train_env.reward_lst) > 0
            ):
                rewards = self.train_env.reward_lst
                reward_sum = sum(rewards)
                reward_mean = np.mean(rewards)
                reward_std = np.std(rewards) if len(rewards) > 1 else 0.0
            else:
                reward_sum = 0.0
                reward_mean = 0.0
                reward_std = 0.0

            # Get TD3 reward from config (set by TD3_controller train())
            td3_reward = getattr(self.config, "last_td3_mean_reward", 0.0)
            td3_updates = getattr(
                self.config, "last_td3_updates", 0
            )  # Set by TD3_controller.py

            display_update_td3(
                actor_loss=td3_actor if td3_actor else 0.0,
                critic_loss=td3_critic if td3_critic else 0.0,
                mean_q=td3_mean_q if td3_mean_q else 0.0,
                buffer_size=buffer_size or 0,
                buffer_max=buffer_max,
                learning_starts=learning_starts,
                updates=td3_updates,
                reward=td3_reward
                if td3_reward
                else reward_mean,  # Use TD3 reward or fallback to mean
                lr=lr,
                noise_sigma=noise_sigma,
            )

            # Get Observer metrics from config or train_env
            # NOTE: Attribute names must match what mafia_observer.py sets:
            # - last_mafia_loss (total loss)
            # - last_mafia_eta_loss (risk/eta MSE loss)
            # - last_mafia_direction_loss (direction CE loss)
            # - last_mafia_pg_loss (PG/selection loss) - may not be set
            obs_loss = getattr(self.config, "last_mafia_loss", 0.0)
            obs_eta_loss = getattr(self.config, "last_mafia_eta_loss", 0.0)
            obs_dir_loss = getattr(
                self.config, "last_mafia_direction_loss", 0.0
            )  # Fixed: was 'last_mafia_dir_loss'
            obs_sel_loss = getattr(
                self.config, "last_mafia_pg_loss", 0.0
            )  # Fixed: was 'last_mafia_sel_loss'
            obs_eta_pred = getattr(self.config, "last_mafia_eta_pred", 1.0)
            obs_dir_pred = getattr(self.config, "last_mafia_dir_pred", "FLAT")
            obs_dir_conf = getattr(self.config, "last_mafia_dir_conf", 0.0)

            # Get sample counts from observer's actual buffer names
            # According to mafia_observer.py train() method:
            # - mkt_direction_lst: direction samples (daily)
            # - risk_eta_pred_tensor_lst: eta/risk samples (daily)
            # - topk_scores_lst: selection/PG samples (on rebalance only)
            samples_dir = 0
            samples_risk = 0
            samples_sel = 0
            # PG reward components: R_t = mean_return - α_turnover×turnover - α_change×symdiff
            pg_mean_return = 0.0
            pg_turnover_penalty = 0.0
            pg_change_penalty = 0.0
            pg_shaped_return = 0.0

            if (
                hasattr(self.train_env, "mkt_observer")
                and self.train_env.mkt_observer is not None
            ):
                obs = self.train_env.mkt_observer
                # Direction samples (daily)
                if hasattr(obs, "mkt_direction_lst"):
                    samples_dir = len(getattr(obs, "mkt_direction_lst", []))
                # Risk/Eta samples (daily)
                if hasattr(obs, "risk_eta_pred_tensor_lst"):
                    samples_risk = len(getattr(obs, "risk_eta_pred_tensor_lst", []))
                # Selection/PG samples (rebalance only)
                if hasattr(obs, "topk_scores_lst"):
                    samples_sel = len(getattr(obs, "topk_scores_lst", []))
                # PG reward components
                pg_mean_return = getattr(obs, "last_pg_mean_return", 0.0)
                pg_turnover_penalty = getattr(obs, "last_pg_turnover_penalty", 0.0)
                pg_change_penalty = getattr(obs, "last_pg_change_penalty", 0.0)
                pg_shaped_return = getattr(obs, "last_pg_shaped_return", 0.0)

            display_update_observer(
                loss=obs_loss if obs_loss else 0.0,
                loss_eta=obs_eta_loss if obs_eta_loss else 0.0,
                loss_dir=obs_dir_loss if obs_dir_loss else 0.0,
                loss_sel=obs_sel_loss if obs_sel_loss else 0.0,
                eta_pred=obs_eta_pred if obs_eta_pred else 1.0,
                dir_pred=obs_dir_pred if obs_dir_pred else "FLAT",
                dir_conf=obs_dir_conf if obs_dir_conf else 0.0,
                samples_dir=samples_dir,
                samples_risk=samples_risk,
                samples_sel=samples_sel,
                # PG reward components
                pg_r_return=pg_mean_return,
                pg_r_turnover=pg_turnover_penalty,
                pg_r_change=pg_change_penalty,
                pg_r_total=pg_shaped_return,
            )

            # Update returns
            cumul_return = portfolio_return if portfolio_return else 0.0

            # Calculate Sharpe ratio from profit_lst
            sharpe = 0.0
            if (
                hasattr(self.train_env, "profit_lst")
                and len(self.train_env.profit_lst) > 1
            ):
                profits = np.array(self.train_env.profit_lst)
                if np.std(profits) > 1e-8:
                    sharpe = np.mean(profits) / np.std(profits) * np.sqrt(252)

            # Calculate Max Drawdown from asset_lst
            mdd = 0.0
            if (
                hasattr(self.train_env, "asset_lst")
                and len(self.train_env.asset_lst) > 1
            ):
                assets = np.array(self.train_env.asset_lst)
                peak = np.maximum.accumulate(assets)
                drawdowns = (peak - assets) / np.where(peak > 0, peak, 1)
                mdd = np.max(drawdowns) * 100  # Convert to percentage

            # Calculate additional return metrics
            win_rate = 0.5
            daily_return = 0.0
            net_profit = 0.0
            annual_return_pct = 0.0

            if (
                hasattr(self.train_env, "profit_lst")
                and len(self.train_env.profit_lst) > 0
            ):
                profits = np.array(self.train_env.profit_lst)
                win_rate = (
                    np.sum(profits > 0) / len(profits) if len(profits) > 0 else 0.5
                )
                daily_return = (
                    profits[-1] * 100 if len(profits) > 0 else 0.0
                )  # Last day return in %

            # BUG FIX: Use initial_asset (not initial_capital) to match tradeEnv
            if hasattr(self.train_env, "cur_capital") and hasattr(
                self.train_env, "initial_asset"
            ):
                initial = getattr(self.train_env, "initial_asset", 1_000_000)
                current = getattr(self.train_env, "cur_capital", 1_000_000)
                net_profit = current - initial
                # Annualized return (assuming 252 trading days)
                if (
                    hasattr(self.train_env, "curTradeDay")
                    and self.train_env.curTradeDay > 0
                ):
                    holding_period = self.train_env.curTradeDay / 252
                    if holding_period > 0 and initial > 0:
                        total_return = (current / initial) - 1
                        annual_return_pct = (
                            (1 + total_return) ** (1 / holding_period) - 1
                        ) * 100

            display_update_returns(
                epoch_return=reward_mean,
                cumul_return=cumul_return,
                sharpe=sharpe if sharpe else 0.0,
                mdd=mdd if mdd else 0.0,
                win_rate=win_rate,
                daily_return=daily_return,
                net_profit=net_profit,
                annual_return_pct=annual_return_pct,
            )

            # ===== Walk-Forward Updates (Phase 1) =====
            training_mode = getattr(self.config, "training_mode", "RL_ONLY")
            if training_mode == "OBSERVER_ONLY":
                # Get walk-forward iteration info from config
                wf_iteration = getattr(self.config, "walkforward_iteration", 0)
                wf_total_iters = getattr(self.config, "walkforward_total_iterations", 5)
                wf_train_start = getattr(
                    self.config, "walkforward_train_start_year", 2015
                )
                wf_train_end = getattr(self.config, "walkforward_train_end_year", 2018)
                wf_valid_year = getattr(self.config, "walkforward_valid_year", 2018)
                wf_infer_year = getattr(self.config, "walkforward_infer_year", 2019)
                wf_is_finetune = getattr(self.config, "walkforward_is_finetune", False)

                train_years = (
                    f"{wf_train_start}→{wf_train_end}"
                    if wf_train_end
                    else f"{wf_train_start}→"
                )

                # Get full date ranges from config (for Expanding Window display)
                train_date_start = getattr(self.config, "train_date_start", None)
                train_date_end = getattr(self.config, "train_date_end", None)
                valid_date_start = getattr(self.config, "valid_date_start", None)
                valid_date_end = getattr(self.config, "valid_date_end", None)

                # Format full date ranges (MM/YYYY format)
                def fmt_date(dt):
                    if dt is None:
                        return ""
                    try:
                        return dt.strftime("%m/%Y")
                    except Exception:
                        return str(dt)[:7] if dt else ""

                train_range = ""
                valid_range = ""
                infer_range = ""
                if train_date_start and train_date_end:
                    train_range = (
                        f"{fmt_date(train_date_start)} → {fmt_date(train_date_end)}"
                    )
                if valid_date_start and valid_date_end:
                    valid_range = (
                        f"{fmt_date(valid_date_start)} → {fmt_date(valid_date_end)}"
                    )
                if wf_infer_year:
                    infer_range = f"01/{wf_infer_year} → 12/{wf_infer_year}"

                display_update_walkforward_iteration(
                    iteration=wf_iteration,
                    total_iterations=wf_total_iters,
                    train_years=train_years,
                    valid_year=wf_valid_year or 0,
                    infer_year=wf_infer_year or 0,
                    train_range=train_range,
                    valid_range=valid_range,
                    infer_range=infer_range,
                    is_finetune=wf_is_finetune,
                )

                # Get best validation score from observer tracker
                best_score = getattr(self.config, "walkforward_best_score", 0.0)
                best_epoch = getattr(self.config, "walkforward_best_epoch", 0)
                current_score = getattr(self.config, "walkforward_current_score", 0.0)

                display_update_walkforward_score(
                    current_score=current_score,
                    best_score=best_score,
                    best_epoch=best_epoch,
                )

        except Exception as e:
            # Silently ignore display errors to not disrupt training
            pass

    def _log_to_tensorboard(self, epoch, metrics_payload, td3_snapshot, mafia_snapshot):
        """Log all metrics to TensorBoard for visualization.

        Phase-aware: Uses different prefixes for Phase 1 (Observer) vs Phase 2 (TD3).
        """
        if self.tb_writer is None:
            return

        # Determine training mode (2-phase training)
        training_mode = getattr(self.config, "training_mode", "RL_ONLY")

        if training_mode == "OBSERVER_ONLY":
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1: Observer Walk-Forward Training
            # ═══════════════════════════════════════════════════════════════════
            phase_prefix = "Phase1_Observer"

            # Log MAFIA losses with Phase 1 prefix
            for key, val in mafia_snapshot.items():
                if val is not None and np.isfinite(val):
                    self.tb_writer.add_scalar(f"{phase_prefix}/{key}", val, epoch)

            # Log additional MAFIA metrics from config
            mafia_eta_loss = getattr(self.config, "last_mafia_eta_loss", None)
            mafia_eta_mae = getattr(self.config, "last_mafia_eta_mae", None)
            mafia_pg_loss = getattr(self.config, "last_mafia_pg_loss", None)
            if mafia_eta_loss is not None and np.isfinite(mafia_eta_loss):
                self.tb_writer.add_scalar(
                    f"{phase_prefix}/eta_loss", mafia_eta_loss, epoch
                )
            if mafia_eta_mae is not None and np.isfinite(mafia_eta_mae):
                self.tb_writer.add_scalar(
                    f"{phase_prefix}/eta_mae", mafia_eta_mae, epoch
                )
            if mafia_pg_loss is not None and np.isfinite(mafia_pg_loss):
                self.tb_writer.add_scalar(
                    f"{phase_prefix}/pg_loss", mafia_pg_loss, epoch
                )

            # Log PG reward components
            if (
                hasattr(self.train_env, "mkt_observer")
                and self.train_env.mkt_observer is not None
            ):
                obs = self.train_env.mkt_observer
                pg_return = getattr(obs, "last_pg_mean_return", None)
                pg_turnover = getattr(obs, "last_pg_turnover_penalty", None)
                pg_change = getattr(obs, "last_pg_change_penalty", None)
                pg_shaped = getattr(obs, "last_pg_shaped_return", None)
                if pg_return is not None and np.isfinite(pg_return):
                    self.tb_writer.add_scalar(
                        f"{phase_prefix}/pg_mean_return", pg_return, epoch
                    )
                if pg_turnover is not None and np.isfinite(pg_turnover):
                    self.tb_writer.add_scalar(
                        f"{phase_prefix}/pg_turnover_penalty", pg_turnover, epoch
                    )
                if pg_change is not None and np.isfinite(pg_change):
                    self.tb_writer.add_scalar(
                        f"{phase_prefix}/pg_change_penalty", pg_change, epoch
                    )
                if pg_shaped is not None and np.isfinite(pg_shaped):
                    self.tb_writer.add_scalar(
                        f"{phase_prefix}/pg_shaped_return", pg_shaped, epoch
                    )

            # Log walk-forward iteration info
            wf_iteration = getattr(self.config, "walkforward_iteration", 0)
            wf_best_score = getattr(self.config, "walkforward_best_score", None)
            wf_current_score = getattr(self.config, "walkforward_current_score", None)
            self.tb_writer.add_scalar(
                f"{phase_prefix}/walkforward_iteration", wf_iteration, epoch
            )
            if wf_best_score is not None and np.isfinite(wf_best_score):
                self.tb_writer.add_scalar(
                    f"{phase_prefix}/walkforward_best_score", wf_best_score, epoch
                )
            if wf_current_score is not None and np.isfinite(wf_current_score):
                self.tb_writer.add_scalar(
                    f"{phase_prefix}/walkforward_current_score", wf_current_score, epoch
                )

        else:
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2: TD3 Training (Observer frozen)
            # ═══════════════════════════════════════════════════════════════════
            phase_prefix = "Phase2_TD3"

            # Log TD3 losses with Phase 2 prefix
            for key, val in td3_snapshot.items():
                if val is not None and np.isfinite(val):
                    self.tb_writer.add_scalar(f"{phase_prefix}/{key}", val, epoch)

            # Log TD3 reward components
            td3_r_return = getattr(self.config, "last_td3_r_return", None)
            td3_r_js = getattr(self.config, "last_td3_r_js", None)
            td3_js_div = getattr(self.config, "last_td3_js_divergence", None)
            if td3_r_return is not None and np.isfinite(td3_r_return):
                self.tb_writer.add_scalar(
                    f"{phase_prefix}/r_return", td3_r_return, epoch
                )
            if td3_r_js is not None and np.isfinite(td3_r_js):
                self.tb_writer.add_scalar(f"{phase_prefix}/r_js", td3_r_js, epoch)
            if td3_js_div is not None and np.isfinite(td3_js_div):
                self.tb_writer.add_scalar(
                    f"{phase_prefix}/js_divergence", td3_js_div, epoch
                )

            # Log CBF interventions
            cbf_interventions = getattr(self.train_env, "cbf_interventions", 0)
            self.tb_writer.add_scalar(
                f"{phase_prefix}/cbf_interventions", cbf_interventions, epoch
            )

        # Log metrics for each data phase (train, valid, test) - common for both training modes
        for data_phase, profile in metrics_payload.items():
            if profile is None:
                continue

            # Log performance metrics
            for field in self.metric_fields:
                val = profile.get(field, None)
                if val is not None:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        val = val[-1] if len(val) > 0 else None
                    if val is not None and np.isfinite(val):
                        self.tb_writer.add_scalar(f"{data_phase}/{field}", val, epoch)

        # Log best validation metric
        if np.isfinite(self.best_valid_metric):
            self.tb_writer.add_scalar(
                "validation/best_metric", self.best_valid_metric, epoch
            )

        # Flush to ensure data is written
        self.tb_writer.flush()

    def _build_metric_row(self, epoch, phase, profile):
        row = {"epoch": epoch, "phase": phase}
        for field in self.metric_fields:
            value = profile.get(field, np.nan)
            if isinstance(value, (list, tuple, np.ndarray)):
                value = value[-1] if len(value) > 0 else np.nan
            row[field] = value
        return row

    def _update_metric_plot(self):
        if not os.path.exists(self.metrics_file):
            return
        try:
            df = pd.read_csv(self.metrics_file)
        except Exception as e:
            _safe_print(
                f"[VISUALIZER] Warning: Could not read metrics file: {e}", "WARNING"
            )
            return
        if df.empty:
            return
        metrics = list(
            dict.fromkeys(
                self.metric_fields
                + [m for m in self.td3_loss_fields if m in df.columns]
                + [m for m in self.mafia_loss_fields if m in df.columns]
            )
        )
        epoch_offset = int(df["epoch"].min()) if "epoch" in df.columns else 0
        relative_epoch = (
            df["epoch"] - epoch_offset + 1 if epoch_offset else df.get("epoch")
        )
        rel_max = relative_epoch.max() if relative_epoch is not None else None
        phases = sorted(df["phase"].unique())
        n = len(metrics)
        cols = 2 if n > 1 else 1
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False
        )
        axes_flat = axes.flatten()
        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            if metric not in df.columns:
                ax.axis("off")
                continue
            for phase in phases:
                subset = df[df["phase"] == phase]
                if subset.empty:
                    continue
                x_vals = (
                    subset["epoch"] - epoch_offset + 1
                    if epoch_offset
                    else subset["epoch"]
                )
                ax.plot(x_vals, subset[metric], marker="o", label=phase)
            ax.set_title(metric)
            xlabel = "Epoch"
            if epoch_offset > 1:
                xlabel += " (relative to resume)"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            if rel_max is not None:
                ax.set_xlim(1, rel_max)
            ax.grid(True, linestyle="--", alpha=0.4)
        for ax in axes_flat[n:]:
            ax.axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(phases))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        output_dir = getattr(self.config, "res_img_dir", self.config.res_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "metrics_history.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        _safe_print(f"[VISUALIZER] Updated metric plot at {output_path}")
