# ！/usr/bin/python
# -*- coding: utf-8 -*-#

"""
---------------------------------
 Name: entrance.py
 Author: MASA
--------------------------------
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Only set CUDA_VISIBLE_DEVICES if CUDA is available
import random
import numpy as np
import torch as th
import datetime
import copy

DEFAULT_RUN_SEED = 2022

# LiveDisplay smart_print for terminal-safe logging (must be imported early)
try:
    from utils.display_integration import smart_print
except ImportError:
    smart_print = print  # Fallback

# Set device: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
if th.backends.mps.is_available():
    smart_print("MPS (Apple Silicon GPU) available, using MPS", flush=True)
elif th.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU or change to '1' if needed
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        th.use_deterministic_algorithms(True)
    except Exception:
        # Fallback if deterministic algorithms not available
        th.use_deterministic_algorithms(False, warn_only=True)
    smart_print("CUDA available, using GPU", flush=True)
else:
    smart_print("No GPU available, using CPU", flush=True)

import pandas as pd
import time
import json
import gzip
import os
import pickle
from config import Config
from utils.featGen import FeatureProcesser
from utils.tradeEnv import StockPortfolioEnv, StockPortfolioEnv_cash
from utils.model_pool import model_select, benchmark_algo_select
from utils.callback_func import PoCallback
from utils.data_validator import get_stock_data_file
from RL_controller.mafia_observer import MAFIAObserver
from stable_baselines3.common.noise import NormalActionNoise

from utils import run_tracker
from utils.training_logger import TrainingLogger, TrainingPhase, get_logger
from utils.display_integration import (
    setup_display,
    set_phase,
    log_message,
    get_display,
    # 2-Phase Training support
    set_training_mode,
    update_walkforward_iteration,
    update_walkforward_score,
    reset_performance_metrics,
)
from utils.live_display import get_display as get_live_display
import timeit

# Web Dashboard (optional, replaces terminal display for stable layout)
try:
    from utils.dashboard_server import start_dashboard_server, is_dashboard_running

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

    def start_dashboard_server(*args, **kwargs):
        return False

    def is_dashboard_running():
        return False

# Legacy RLonly function removed - MAFIA-only codebase now uses RLcontroller exclusively


def RLcontroller(config):
    """
    MAFIA training pipeline (MASA framework with MAFIA observer).
    This is the only supported training mode.
    """
    # Initialize logger
    logger = get_logger()
    logger.print_banner("MAFIA TRAINING PIPELINE", TrainingPhase.INIT)

    # ============================================================
    # PRE-DETERMINE training mode BEFORE display initialization
    # This ensures the display shows the correct layout from the start
    # ============================================================
    observer_pretrained_path = getattr(config, "observer_pretrained_path", None)
    freeze_observer = getattr(config, "freeze_observer_during_rl", False)
    is_observer_only = getattr(config, "observer_only_training", False) or getattr(
        config, "observer_only_mode", False
    )

    # Determine training mode
    if (
        observer_pretrained_path
        and os.path.exists(observer_pretrained_path)
        and freeze_observer
    ):
        # Phase 2: TD3 training with frozen Observer
        initial_training_mode = "RL_ONLY"
        config.mafia_allow_observer_training = False  # Observer frozen
    elif is_observer_only:
        # Phase 1: Observer-only training
        initial_training_mode = "OBSERVER_ONLY"
        config.mafia_allow_observer_training = (
            True  # CRITICAL: Allow Observer to train and collect samples
        )
    else:
        # Default: Phase 2 (RL_ONLY)
        initial_training_mode = "RL_ONLY"
        config.mafia_allow_observer_training = False  # Observer frozen by default

    # ============================================================
    # CRITICAL: Training Mode Assertions - Prevent Invalid States
    # ============================================================
    # Assert 1: Prevent combined training (Observer + TD3 simultaneously)
    # Two-phase separation requires exactly one component training at a time
    assert not (
        config.mafia_allow_observer_training and initial_training_mode == "RL_ONLY"
    ), (
        f"INVALID STATE: mafia_allow_observer_training=True conflicts with RL_ONLY mode.\n"
        f"Two-phase training requires Observer frozen during RL training."
    )

    # Assert 2: Observer-only mode must have Observer training enabled
    assert not (
        initial_training_mode == "OBSERVER_ONLY"
        and not config.mafia_allow_observer_training
    ), (
        f"INVALID STATE: OBSERVER_ONLY mode requires mafia_allow_observer_training=True.\n"
        f"Observer cannot train when mafia_allow_observer_training=False."
    )

    # Assert 3: Frozen Observer cannot be used with Observer-only mode
    if (
        observer_pretrained_path
        and os.path.exists(observer_pretrained_path)
        and freeze_observer
    ):
        assert initial_training_mode != "OBSERVER_ONLY", (
            f"INVALID STATE: Cannot run OBSERVER_ONLY with frozen pretrained Observer.\n"
            f"Remove observer_pretrained_path or set freeze_observer=False."
        )

    # Log training mode configuration
    smart_print(f"\n🎯 [Training Mode] {initial_training_mode}")
    smart_print(
        f"   • mafia_allow_observer_training: {config.mafia_allow_observer_training}"
    )
    smart_print(f"   • freeze_observer: {freeze_observer}")
    smart_print(f"   • is_observer_only: {is_observer_only}")
    smart_print(
        f"   • observer_pretrained_path: {observer_pretrained_path or 'None'}\n"
    )

    # Optional override: force simple logging (no LiveDisplay/dashboard)
    simple_logging = str(os.environ.get("MAFIA_SIMPLE_LOGGING", "0")).lower() in (
        "1",
        "true",
        "yes",
    ) or getattr(config, "simple_logging", False)
    if simple_logging:
        os.environ["MAFIA_SIMPLE_LOGGING"] = "1"
        config.use_live_display = False
        config.use_web_dashboard = False
        os.environ["MAFIA_NO_LIVE_DISPLAY"] = "1"
        smart_print(
            "ℹ️  SIMPLE LOGGING enabled: LiveDisplay/Dashboard disabled.", flush=True
        )

    # Initialize LiveDisplay for real-time terminal updates
    # Disable if MAFIA_NO_LIVE_DISPLAY env var is set
    use_live_display = getattr(config, "use_live_display", True)
    if os.environ.get("MAFIA_NO_LIVE_DISPLAY", "0") == "1":
        use_live_display = False
    log_file_path = (
        os.path.join(config.res_dir, "training.log")
        if hasattr(config, "res_dir")
        else None
    )

    # In simple logging mode, skip display setup entirely to avoid table layout
    if use_live_display:
        display = setup_display(
            enabled=use_live_display,
            log_file=log_file_path,
            run_id=getattr(config, "cur_datetime", ""),
            total_epochs=getattr(config, "num_epochs", 50),
            window=getattr(config, "wf_window_index", 0),
            seed=getattr(config, "seed_num", 2025),
        )
    else:
        display = None

    # Store display reference in config for other modules to access
    config._live_display = display

    # Clear QUIET_STARTUP now that display is initialized
    # (subsequent logs will be routed through display properly)
    if "MAFIA_QUIET_STARTUP" in os.environ:
        del os.environ["MAFIA_QUIET_STARTUP"]

    # Set the training mode IMMEDIATELY after display init (before any rendering)
    # Note: enable_display_mode() is already called inside setup_display()
    if use_live_display:
        set_training_mode(initial_training_mode)
        log_message(
            f"LiveDisplay initialized: run_id={config.cur_datetime}, mode={initial_training_mode}"
        )

    # Start web dashboard if enabled (provides stable layout without terminal issues)
    # Disable if MAFIA_NO_LIVE_DISPLAY env var is set
    use_web_dashboard = getattr(config, "use_web_dashboard", False)
    if os.environ.get("MAFIA_NO_LIVE_DISPLAY", "0") == "1":
        use_web_dashboard = False
    if use_web_dashboard and DASHBOARD_AVAILABLE:
        dashboard_port = getattr(config, "dashboard_port", 5050)
        open_browser = getattr(config, "dashboard_open_browser", True)
        start_dashboard_server(port=dashboard_port, open_browser=open_browser)

    # Get dataset
    fpath, error_msg = get_stock_data_file(config)
    if fpath is None:
        raise ValueError(f"Cannot load the data file. {error_msg}")
    smart_print(f"📂 Đang tải dữ liệu từ: {fpath}", flush=True)
    data = pd.DataFrame(pd.read_csv(fpath, header=0))

    # MAFIA lightweight data loading
    from utils.mafia_data_loader import MAFIADataLoader
    import gc

    smart_print("🔄 [MAFIA] Sử dụng lightweight data loader...", flush=True)
    mafia_loader = MAFIADataLoader(config=config)
    data_dict = mafia_loader.load_and_split_data(data=data)
    
    # CRITICAL: Free raw dataframe memory immediately
    del data
    gc.collect()
    
    tech_indicator_lst = []  # MAFIA doesn't use legacy tech indicators
    stock_num = data_dict["train"]["stock"].nunique()
    smart_print(f"✅ Tải dữ liệu hoàn tất: {stock_num} cổ phiếu", flush=True)

    # Initialize MAFIA observer (always enabled)
    mkt_observer = MAFIAObserver(config=config, action_dim=stock_num)

    # Enable feature caching for faster training
    # Pre-compute SMA, RSI, ATR once instead of every forward pass
    stock_list = data_dict["train"]["stock"].unique().tolist()
    mkt_observer.enable_feature_caching(data_dict["train"], stock_list)

    # ============================================================
    # Walk-Forward Phase 2: Load pre-trained observer and freeze
    # (observer_pretrained_path, freeze_observer, is_observer_only already determined above)
    # ============================================================
    if observer_pretrained_path and os.path.exists(observer_pretrained_path):
        smart_print(f"\n{'=' * 70}")
        smart_print("🔄 [PHASE 2] Loading pre-trained Observer (Static Expert)")
        smart_print(f"{'=' * 70}")
        smart_print(f"  Checkpoint: {observer_pretrained_path}")

        # Load pre-trained weights
        loaded_epoch = mkt_observer.load_checkpoint(observer_pretrained_path)
        smart_print(f"  ✅ Loaded from epoch {loaded_epoch}")

        if freeze_observer:
            # Freeze observer - disable training
            config.mafia_allow_observer_training = False
            mkt_observer.mafia_model.eval()
            for param in mkt_observer.mafia_model.parameters():
                param.requires_grad = False
            smart_print("  🔒 Observer FROZEN - Static Expert mode")
            smart_print("     • No gradient updates during RL training")
            smart_print("     • Only provides Top-K, risk_eta, direction signals")

            # Set 2-phase training mode: Phase 2 (RL_ONLY)
            config.training_phase = 2
            config.training_mode = "RL_ONLY"
            config.observer_checkpoint_source = observer_pretrained_path
            set_training_mode("RL_ONLY", observer_checkpoint=observer_pretrained_path)
        else:
            smart_print("  ⚠️  Observer loaded but NOT frozen (will continue training)")

        smart_print(f"{'=' * 70}\n")
    else:
        # No pre-trained Observer loaded
        # Determine training mode based on config flags (is_observer_only already set above)
        if is_observer_only:
            # Phase 1: Observer-only training (TD3 frozen with uniform weights)
            config.training_phase = 1
            config.training_mode = "OBSERVER_ONLY"
            # Note: set_training_mode already called at line 182
            smart_print("🔮 [PHASE 1] Observer-only training mode (TD3 frozen)")
        else:
            # Default: Phase 2 (RL_ONLY) - TD3 training
            config.training_phase = 2
            config.training_mode = "RL_ONLY"
            # Note: set_training_mode already called at line 182

    # Initialize environment
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para
        env_valid = StockPortfolioEnv(
            config=config,
            rawdata=data_dict["valid"],
            mode="valid",
            stock_num=stock_num,
            action_dim=stock_num,
            tech_indicator_lst=tech_indicator_lst,
            extra_data=data_dict["extra_valid"],
            mkt_observer=mkt_observer,
            **validInvest_env_para,
        )
    else:
        env_valid = None
        raise ValueError("No validation set is provided for training")
    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para
        env_test = StockPortfolioEnv(
            config=config,
            rawdata=data_dict["test"],
            mode="test",
            stock_num=stock_num,
            action_dim=stock_num,
            tech_indicator_lst=tech_indicator_lst,
            extra_data=data_dict["extra_test"],
            mkt_observer=mkt_observer,
            **testInvest_env_para,
        )
    else:
        env_test = None
        raise ValueError("No test set is provided for training")

    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    # Initialize environment
    trainInvest_env_para = config.invest_env_para
    env_train = StockPortfolioEnv(
        config=config,
        rawdata=data_dict["train"],
        mode="train",
        stock_num=stock_num,
        action_dim=stock_num,
        tech_indicator_lst=tech_indicator_lst,
        extra_data=data_dict["extra_train"],
        mkt_observer=mkt_observer,
        **trainInvest_env_para,
    )

    # Track if we're resuming from checkpoint (set later when checkpoint is loaded)
    _is_resuming_from_checkpoint = False

    def pretrain_market_observer_if_needed(env):
        """
        Run a lightweight observer-only warmup before TD3 starts.
        Uses uniform weights to roll through the training window and trains the observer at epoch end.
        Skipped when resuming from checkpoint if skip_pretrain_on_resume=True.
        """
        nonlocal _is_resuming_from_checkpoint
        if env is None or not getattr(config, "enable_market_observer", False):
            return

        # Skip pretrain if resuming from checkpoint and config says to skip
        if _is_resuming_from_checkpoint and getattr(
            config, "skip_pretrain_on_resume", True
        ):
            logger.newline()
            smart_print(
                "⏭️  [PRETRAIN] Bỏ qua observer warmup (đang resume từ checkpoint)",
                flush=True,
            )
            return

        # Get pretrain config
        pretrain_mini_epochs = int(
            getattr(config, "mafia_pretrain_mini_epochs", 0) or 0
        )
        mini_epoch_steps = int(getattr(config, "observer_mini_epoch_steps", 126) or 126)

        if pretrain_mini_epochs <= 0:
            return

        # Force-enable observer training during the warmup
        prev_allow_training = getattr(config, "mafia_allow_observer_training", True)
        config.mafia_allow_observer_training = True
        original_validation_mode = getattr(env, "validation_mode", False)
        env.validation_mode = True  # avoid writing train profiles during warmup

        # Mini-epoch based pretrain (2 mini-epochs × 126 steps = 252 steps by default)
        total_pretrain_steps = pretrain_mini_epochs * mini_epoch_steps

        # Print pretrain banner
        logger.print_banner(
            f"PHASE 1: PRETRAIN - Observer-only warmup", TrainingPhase.PRETRAIN
        )
        smart_print(
            f"  ╭─────────────────────────────────────────────────────────────────────╮\n"
            f"  │  📋 PRETRAIN CONFIGURATION (Cấu hình Pretrain)                      │\n"
            f"  ├─────────────────────────────────────────────────────────────────────┤\n"
            f"  │  📊 Số Mini-epochs: {pretrain_mini_epochs:<48}│\n"
            f"  │  📊 Bước mỗi mini-epoch: {mini_epoch_steps:<43}│\n"
            f"  │  📊 Tổng bước pretrain: {total_pretrain_steps:<44}│\n"
            f"  │  📊 Chiến lược action: Uniform weights + noise (không dùng TD3)     │\n"
            f"  ├─────────────────────────────────────────────────────────────────────┤\n"
            f"  │  🎯 MỤC ĐÍCH:                                                       │\n"
            f"  │     - Warmup Observer trước khi TD3 bắt đầu                         │\n"
            f"  │     - Thu thập dữ liệu để huấn luyện 3 task của Observer:           │\n"
            f"  │       1️⃣  Stock Selection (Chọn cổ phiếu Top-K)                      │\n"
            f"  │       2️⃣  Market Direction (Dự đoán hướng thị trường)                │\n"
            f"  │       3️⃣  Risk Tolerance η (Mức chấp nhận rủi ro)                    │\n"
            f"  ╰─────────────────────────────────────────────────────────────────────╯",
            flush=True,
        )
        smart_print("", flush=True)

        # Set 2-phase training mode: Phase 1 (OBSERVER_ONLY) during pretrain
        config.training_phase = 1
        config.training_mode = "OBSERVER_ONLY"
        # Note: set_training_mode already called at line 182

        # Set display phase to PRETRAIN (now OBSERVER_PRETRAIN)
        set_phase("OBSERVER_PRETRAIN", epoch=0, total_steps=total_pretrain_steps)

        try:
            env.reset()
        except Exception:
            _ = env.reset()
        steps = 0
        current_mini_epoch = 0
        while steps < total_pretrain_steps:
            action_dim = int(np.prod(env.action_space.shape))
            if action_dim <= 0:
                break
            base_action = np.ones(action_dim, dtype=np.float32)
            base_action = base_action / (np.sum(np.abs(base_action)) + 1e-8)
            noise = np.random.normal(scale=0.01, size=base_action.shape)
            action = np.clip(base_action + noise, 0.0, 1.0)
            step_out = env.step(action)
            terminated = False
            truncated = False
            if isinstance(step_out, (list, tuple)) and len(step_out) >= 5:
                terminated = bool(step_out[2])
                truncated = bool(step_out[3])
            elif isinstance(step_out, (list, tuple)) and len(step_out) == 4:
                terminated = bool(step_out[2])
                truncated = False
            elif isinstance(step_out, (list, tuple)) and len(step_out) >= 3:
                terminated = bool(step_out[2])
            done = bool(terminated or truncated)
            steps += 1

            # Update LiveDisplay for pretrain progress
            display = get_display()
            if display:
                # Always update state (for dashboard API)
                display.update(
                    step=steps,
                    total_steps=total_pretrain_steps,
                    phase="PRETRAIN",
                )
                if display.enabled:
                    display.render()
                else:
                    # Fallback: Real-time pretrain progress on single line (with Observer loss info)
                    logger.print_pretrain_progress(
                        steps,
                        total_pretrain_steps,
                        current_mini_epoch + 1,
                        pretrain_mini_epochs,
                        config=config,
                    )
            else:
                # No display at all - just print
                logger.print_pretrain_progress(
                    steps,
                    total_pretrain_steps,
                    current_mini_epoch + 1,
                    pretrain_mini_epochs,
                    config=config,
                )

            # Log progress at each mini-epoch boundary
            if steps % mini_epoch_steps == 0:
                current_mini_epoch += 1
                if not display or not display.enabled:
                    logger.newline()
                    smart_print(
                        f"  ✅ Mini-epoch {current_mini_epoch}/{pretrain_mini_epochs} hoàn thành ({steps}/{total_pretrain_steps} bước)",
                        flush=True,
                    )

            if done:
                # Reset env if episode ends before pretrain is complete
                if steps < total_pretrain_steps:
                    try:
                        env.reset()
                    except Exception:
                        _ = env.reset()

        logger.newline()

        env.validation_mode = original_validation_mode

        # After pretrain: ALWAYS freeze observer (Static Expert mode)
        config.mafia_allow_observer_training = False
        smart_print(
            "❄️  [PRETRAIN] Observer frozen - Static Expert mode (no further training)",
            flush=True,
        )

        # Switch to Phase 2 (RL_ONLY) after pretrain completes
        config.training_phase = 2
        config.training_mode = "RL_ONLY"
        set_training_mode("RL_ONLY")

        # Hard reset environment so Epoch 1 starts clean (no warmup carry-over)
        try:
            env.reset()
            if hasattr(env, "epoch"):
                env.epoch = 0  # SB3 reset will bump to 1
            if hasattr(env, "curTradeDay"):
                env.curTradeDay = 0
            # Reset display counters so Epoch 1 starts clean on dashboard
            display = get_display()
            if display:
                total_days = getattr(env, "totalTradeDay", 0) or total_pretrain_steps
                display.update(
                    phase="RL_TRAIN",
                    epoch=0,
                    total_epochs=getattr(config, "num_epochs", 0),
                    step=0,
                    total_steps=total_days,
                    epoch_day=1,
                    epoch_total_days=total_days,
                    total_train_steps=0,
                )
                if display.enabled:
                    display.render(force=True)
        except Exception:
            pass
        # Sync LiveDisplay/Dashboard to Phase 2 baseline
        try:
            set_phase(
                "RL_TRAIN",
                epoch=0,
                total_steps=getattr(env, "totalTradeDay", 0) or total_pretrain_steps,
            )
        except Exception:
            pass

        # Clear warmup performance metrics before official Epoch 1
        try:
            reset_performance_metrics(initial_capital=getattr(env, "initial_asset", 0))
        except Exception:
            pass

        # Print phase transition
        logger.print_phase_transition(
            TrainingPhase.OBSERVER_PRETRAIN,
            TrainingPhase.RL_TRAIN,
            f"Pretrain hoàn thành với {steps} bước. Chuyển sang Phase 2: TD3 Training.",
        )

        # Reset epoch counter so TD3 starts cleanly
        try:
            env.epoch = 0
            env._observer_mini_step_counter = 0
        except Exception:
            pass

    # Train Market Observer once before TD3 rollout (Top-K ready for RL/solver)
    pretrain_market_observer_if_needed(env_train)

    # Ensure run manifest exists early so resume/reporting metadata is always present
    run_tracker.ensure_manifest(
        getattr(config, "run_manifest_path", None),
        config.cur_datetime,
        config.num_epochs,
    )

    # Load RL model
    def build_action_noise(env):
        sigma = getattr(config, "action_noise_sigma", 0.0)
        if sigma is None or sigma <= 0:
            return None
        action_dim = int(np.prod(env.action_space.shape))
        if action_dim <= 0:
            return None
        return NormalActionNoise(
            mean=np.zeros(action_dim), sigma=np.ones(action_dim) * sigma
        )

    def build_model_params():
        mp = dict(config.model_para)
        if prefilled_buffer_capacity is not None:
            mp["buffer_size"] = max(
                mp.get("buffer_size", 0), int(prefilled_buffer_capacity)
            )
        action_noise_obj_inner = build_action_noise(env_train)
        if action_noise_obj_inner is not None:
            mp["action_noise"] = action_noise_obj_inner
        return mp

    def disable_learning_starts(po_model_obj):
        """Force-skip warm-up when resuming from checkpoint."""
        config.learning_starts = 0
        if hasattr(po_model_obj, "learning_starts"):
            po_model_obj.learning_starts = 0
        smart_print("[RESUME] learning_starts forced to 0 (resume path)", flush=True)

    def load_replay_buffer_file(path: str):
        """Load replay buffer from file, trying pickle then gzip."""
        # First try plain pickle
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # If gzip, fallback
            try:
                with gzip.open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                raise

    def manual_load_replay_buffer(path: str, po_model_obj):
        """Fallback: manually unpickle replay buffer and attach to model."""
        rb_obj = load_replay_buffer_file(path)
        po_model_obj.replay_buffer = rb_obj
        smart_print(
            f"[RESUME] Replay buffer manually loaded and attached from {path}",
            flush=True,
        )
        # Align learning_starts to zero because buffer is prefilled
        disable_learning_starts(po_model_obj)

    # Auto-detect latest checkpoint if auto_resume is enabled and no checkpoint specified
    checkpoint_to_resume = config.resume_from_checkpoint
    # Ensure checkpoint_to_resume is either None or a valid string path
    if checkpoint_to_resume is not None and not isinstance(checkpoint_to_resume, str):
        smart_print(
            f"Warning: resume_from_checkpoint is not a string (got {type(checkpoint_to_resume)}), resetting to None",
            flush=True,
        )
        checkpoint_to_resume = None

    if checkpoint_to_resume is None and config.auto_resume_from_latest:
        # Find latest checkpoint - search in current checkpoint_dir and all parent res directories
        search_dirs = [config.checkpoint_dir]  # Start with current checkpoint_dir

        # Also search in parent res directories (for previous runs)
        base_res_dir = os.path.dirname(config.checkpoint_dir)  # Remove 'checkpoints'
        if os.path.exists(base_res_dir):
            parent_dir = os.path.dirname(
                base_res_dir
            )  # res/RLcontroller/TD3/VNINDEX-10
            if os.path.exists(parent_dir):
                # Search in all timestamp directories
                for timestamp_dir in os.listdir(parent_dir):
                    timestamp_path = os.path.join(parent_dir, timestamp_dir)
                    if os.path.isdir(timestamp_path):
                        checkpoint_path = os.path.join(timestamp_path, "checkpoints")
                        if (
                            os.path.exists(checkpoint_path)
                            and checkpoint_path != config.checkpoint_dir
                        ):
                            search_dirs.append(checkpoint_path)

        checkpoint_records = []
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            checkpoint_dirs = [
                d
                for d in os.listdir(search_dir)
                if os.path.isdir(os.path.join(search_dir, d))
            ]
            for dirname in checkpoint_dirs:
                info_path = os.path.join(search_dir, dirname, "checkpoint_info.json")
                if os.path.exists(info_path):
                    try:
                        with open(info_path, "r") as f:
                            info = json.load(f)
                        checkpoint_records.append(
                            {
                                "info_path": info_path,
                                "info": info,
                                "mtime": os.path.getmtime(info_path),
                            }
                        )
                    except Exception as e:
                        smart_print(
                            f"Warning: Failed to read checkpoint info from {info_path}: {e}",
                            flush=True,
                        )

        if checkpoint_records:
            # Sort by timesteps, then epoch, then latest modification time
            checkpoint_records.sort(
                key=lambda item: (
                    item["info"].get("timesteps", 0),
                    item["info"].get("epoch", 0),
                    item["mtime"],
                ),
                reverse=True,
            )
            latest_record = checkpoint_records[0]
            checkpoint_to_resume = latest_record["info_path"]
            smart_print(
                f"Auto-detected latest checkpoint: {checkpoint_to_resume}", flush=True
            )
            smart_print(
                f"  Epoch: {latest_record['info'].get('epoch', 0)}, Timesteps: {latest_record['info'].get('timesteps', 0)}",
                flush=True,
            )

    # Refresh manifest after any potential directory redirection
    run_tracker.ensure_manifest(
        getattr(config, "run_manifest_path", None),
        config.cur_datetime,
        config.num_epochs,
    )

    # Resume from checkpoint if specified or auto-detected
    start_epoch = 0
    checkpoint_timesteps = 0
    checkpoint_dir = None
    resume_loaded = False
    incompatible_checkpoint = False
    replay_buffer_path = None
    original_learning_starts = getattr(config, "learning_starts", 0)
    prefilled_buffer_size = 0
    prefilled_buffer_capacity = None
    buffer_loaded = False
    reset_rb_on_resume = getattr(config, "reset_replay_buffer_on_resume", False)
    last_loaded_buffer_size = 0
    if (
        checkpoint_to_resume is not None
        and isinstance(checkpoint_to_resume, str)
        and os.path.exists(checkpoint_to_resume)
    ):
        smart_print(
            f"Resuming training from checkpoint: {checkpoint_to_resume}", flush=True
        )

        # Load checkpoint info
        checkpoint_dir = os.path.dirname(checkpoint_to_resume)
        info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")

        # Ensure new artifacts continue inside the original run directory
        # BUT: detect cross-window resume (checkpoint from different res_dir)
        # In cross-window resume, keep the NEW res_dir to avoid overwriting previous window
        checkpoint_root_dir = os.path.dirname(checkpoint_dir)
        previous_run_dir = os.path.dirname(checkpoint_root_dir)

        # Detect cross-window resume: checkpoint belongs to a different run directory
        current_res_dir = os.path.normpath(config.res_dir)
        previous_run_dir_norm = (
            os.path.normpath(previous_run_dir)
            if os.path.exists(previous_run_dir)
            else None
        )
        is_cross_window_resume = (
            previous_run_dir_norm is not None
            and current_res_dir != previous_run_dir_norm
        )

        if is_cross_window_resume:
            # Cross-window resume: keep new res_dir, only load weights from checkpoint
            smart_print(
                f"[RESUME] Cross-window resume detected: checkpoint from {previous_run_dir}",
                flush=True,
            )
            smart_print(
                f"[RESUME] Outputs will go to NEW directory: {config.res_dir}",
                flush=True,
            )
            # Ensure new directories exist
            os.makedirs(config.res_model_dir, exist_ok=True)
            os.makedirs(config.res_img_dir, exist_ok=True)
            os.makedirs(config.checkpoint_dir, exist_ok=True)
        elif os.path.exists(previous_run_dir):
            # Same-window resume: continue in original directory
            config.cur_datetime = os.path.basename(previous_run_dir)
            config.res_dir = previous_run_dir
            config.res_model_dir = os.path.join(previous_run_dir, "model")
            config.res_img_dir = os.path.join(previous_run_dir, "graph")
            config.checkpoint_dir = os.path.join(previous_run_dir, "checkpoints")
            config.metrics_history_path = os.path.join(
                config.res_dir, "metrics_history.csv"
            )
            config.run_manifest_path = os.path.join(config.res_dir, "run_manifest.json")
            os.makedirs(config.res_model_dir, exist_ok=True)
            os.makedirs(config.res_img_dir, exist_ok=True)
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            smart_print(
                f"[RESUME] Continuing outputs inside existing run directory: {config.res_dir}",
                flush=True,
            )

        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                checkpoint_info = json.load(f)
            run_tracker.record_resume_event(
                getattr(config, "run_manifest_path", None),
                checkpoint_to_resume,
                checkpoint_info,
            )
            start_epoch = checkpoint_info.get("epoch", 0)
            checkpoint_timesteps = checkpoint_info.get("timesteps", 0)
            checkpoint_type = checkpoint_info.get("type", "epoch")
            day_in_epoch = checkpoint_info.get("day_in_epoch", 0)
            replay_buffer_path = checkpoint_info.get("replay_buffer_path", None)
            if reset_rb_on_resume and replay_buffer_path:
                smart_print(
                    "[RESUME] reset_replay_buffer_on_resume=1 -> skip loading replay buffer from checkpoint",
                    flush=True,
                )
                replay_buffer_path = None
            # Peek replay buffer size/capacity to align model buffer before init
            if replay_buffer_path and os.path.exists(replay_buffer_path):
                try:
                    rb_obj = load_replay_buffer_file(replay_buffer_path)
                    # Determine how many samples are stored and the capacity
                    if hasattr(rb_obj, "size"):
                        prefilled_buffer_size = rb_obj.size()
                    elif hasattr(rb_obj, "pos"):
                        prefilled_buffer_size = int(rb_obj.pos)
                    if hasattr(rb_obj, "buffer_size"):
                        prefilled_buffer_capacity = rb_obj.buffer_size
                    elif hasattr(rb_obj, "max_size"):
                        prefilled_buffer_capacity = rb_obj.max_size
                    smart_print(
                        f"[RESUME] Detected replay buffer file: size={prefilled_buffer_size}, capacity={prefilled_buffer_capacity}",
                        flush=True,
                    )
                except Exception as e:
                    smart_print(
                        f"[RESUME] Warning: failed to peek replay buffer ({e})",
                        flush=True,
                    )

            smart_print(f"Checkpoint type: {checkpoint_type}", flush=True)
            smart_print(
                f"Resuming from epoch {start_epoch}, day {day_in_epoch}, timestep {checkpoint_timesteps}",
                flush=True,
            )

            # Check if this is a mid-epoch resume (day_in_epoch > 0 means we're in the middle of an epoch)
            env_state_path = checkpoint_info.get("env_state_path", None)
            is_mid_epoch_resume = (
                (day_in_epoch > 0)
                and (env_state_path is not None)
                and os.path.exists(env_state_path)
            )

            if is_mid_epoch_resume:
                # For mid-epoch resume: set epoch to start_epoch - 1 so after reset() it becomes start_epoch
                # Then we'll restore the full state (including curTradeDay) after reset()
                if hasattr(env_train, "epoch"):
                    env_train.epoch = start_epoch - 1
                    smart_print(
                        f"Mid-epoch resume detected: set env.epoch to {start_epoch - 1} (will become {start_epoch} after reset)",
                        flush=True,
                    )
                    smart_print(
                        f"Will restore environment state from {env_state_path} after reset()",
                        flush=True,
                    )
                # Store env_state_path in environment for callback to restore after reset()
                env_train._resume_env_state_path = env_state_path
            else:
                # For epoch-end resume: set epoch to start_epoch so after reset() it becomes start_epoch + 1 (new epoch)
                if hasattr(env_train, "epoch"):
                    env_train.epoch = start_epoch
                    smart_print(
                        f"Epoch-end resume: set env.epoch to {start_epoch} (will become {start_epoch + 1} after reset)",
                        flush=True,
                    )
                # Clear any previous resume state path
                if hasattr(env_train, "_resume_env_state_path"):
                    delattr(env_train, "_resume_env_state_path")

            # Restore RNG state if available
            rng_state_path = checkpoint_info.get("rng_state_path", None)
            if rng_state_path and os.path.exists(rng_state_path):
                try:
                    with open(rng_state_path, "rb") as f:
                        rng_state = pickle.load(f)
                    seed_from_rng = rng_state.get("seed_num", None)
                    if seed_from_rng is not None:
                        config.seed_num = seed_from_rng
                    py_state = rng_state.get("python_random")
                    if py_state is not None:
                        random.setstate(py_state)
                    np_state = rng_state.get("numpy_random")
                    if np_state is not None:
                        np.random.set_state(np_state)
                    torch_state = rng_state.get("torch_cpu")
                    if torch_state is not None:
                        th.set_rng_state(torch_state)
                    torch_cuda_state = rng_state.get("torch_cuda")
                    if torch_cuda_state is not None and th.cuda.is_available():
                        th.cuda.set_rng_state_all(torch_cuda_state)
                    smart_print(
                        f"[RESUME] RNG state restored from {rng_state_path}", flush=True
                    )
                except Exception as e:
                    smart_print(
                        f"[RESUME] Warning: Failed to restore RNG state: {e}",
                        flush=True,
                    )
            else:
                if rng_state_path:
                    smart_print(
                        f"[RESUME] RNG state file not found at {rng_state_path}",
                        flush=True,
                    )

        # Build model params (align buffer_size if needed)
        model_para_dict = build_model_params()

        # Load RL model from checkpoint
        rl_checkpoint_path = os.path.join(checkpoint_dir, "rl_model.zip")
        if os.path.exists(rl_checkpoint_path):
            try:
                po_model = ModelCls.load(rl_checkpoint_path, env=env_train)
                po_model.mafia_config = config
                po_model.verbose = 1
                # Reapply action noise for resumed training
                action_noise_loaded = build_action_noise(env_train)
                if action_noise_loaded is not None:
                    po_model.action_noise = action_noise_loaded
                resume_loaded = True
                _is_resuming_from_checkpoint = True  # Set flag for skip_pretrain logic
                smart_print(f"RL model loaded from {rl_checkpoint_path}", flush=True)
                disable_learning_starts(po_model)
            except ValueError as e:
                msg = str(e)
                mismatch_signatures = [
                    "Action spaces do not match",
                    "Observation spaces do not match",
                ]
                if any(signature in msg for signature in mismatch_signatures):
                    incompatible_checkpoint = True
                    smart_print(
                        f"Warning: Checkpoint at {rl_checkpoint_path} is incompatible with current environment ({msg}). Starting fresh training.",
                        flush=True,
                    )
                else:
                    raise
        else:
            smart_print(
                f"Warning: RL checkpoint not found at {rl_checkpoint_path}, starting fresh",
                flush=True,
            )

        if not resume_loaded:
            po_model = ModelCls(env=env_train, **model_para_dict)
            po_model.mafia_config = config
            po_model.verbose = 1
            # Reset resume-specific metadata when checkpoint cannot be loaded
            start_epoch = 0
            checkpoint_timesteps = 0
            checkpoint_dir = None
            checkpoint_to_resume = None
            if hasattr(env_train, "_resume_env_state_path"):
                delattr(env_train, "_resume_env_state_path")
            if hasattr(env_train, "epoch"):
                env_train.epoch = 0
        else:
            # Restore replay buffer if the checkpoint saved it
            if replay_buffer_path and os.path.exists(replay_buffer_path):
                try:
                    po_model.load_replay_buffer(replay_buffer_path)
                    buffer_obj = getattr(po_model, "replay_buffer", None)
                    buffer_size = 0
                    if buffer_obj is not None:
                        buffer_size = (
                            buffer_obj.size()
                            if hasattr(buffer_obj, "size")
                            else len(buffer_obj)
                        )
                    if buffer_obj is None or buffer_size <= 0:
                        raise RuntimeError(
                            f"Replay buffer empty after SB3 load (size={buffer_size})"
                        )
                    smart_print(
                        f"[RESUME] Replay buffer loaded from {replay_buffer_path} (size: {buffer_size})",
                        flush=True,
                    )
                    buffer_loaded = True
                except Exception as e:
                    smart_print(
                        f"[RESUME] Warning: Failed to load replay buffer via SB3: {e}",
                        flush=True,
                    )
                    if prefilled_buffer_size > 0:
                        try:
                            manual_load_replay_buffer(replay_buffer_path, po_model)
                            buffer_obj = getattr(po_model, "replay_buffer", None)
                            buffer_size = (
                                buffer_obj.size()
                                if hasattr(buffer_obj, "size")
                                else len(buffer_obj)
                            )
                            smart_print(
                                f"[RESUME] Manual replay buffer load succeeded (size: {buffer_size})",
                                flush=True,
                            )
                            buffer_loaded = True
                        except Exception as e2:
                            smart_print(
                                f"[RESUME] Manual replay buffer load failed: {e2}",
                                flush=True,
                            )
                buffer_obj = getattr(po_model, "replay_buffer", None)
                buffer_size = (
                    buffer_obj.size()
                    if buffer_obj is not None and hasattr(buffer_obj, "size")
                    else (len(buffer_obj) if buffer_obj is not None else 0)
                )
                last_loaded_buffer_size = buffer_size
                if buffer_loaded:
                    smart_print(
                        f"[RESUME] learning_starts set to 0 (buffer size now {buffer_size})",
                        flush=True,
                    )

                    # Apply walk-forward buffer filtering if enabled
                    filter_rb_on_resume = getattr(
                        config, "filter_replay_buffer_on_resume", False
                    )
                    if filter_rb_on_resume and hasattr(
                        buffer_obj, "filter_by_date_range"
                    ):
                        # Convert train dates to YYYYMMDD int format
                        train_start_ts = getattr(config, "train_date_start", None)
                        train_end_ts = getattr(config, "train_date_end", None)
                        if train_start_ts is not None and train_end_ts is not None:
                            try:
                                # Convert pd.Timestamp/datetime to YYYYMMDD int
                                if hasattr(train_start_ts, "strftime"):
                                    train_start_int = int(
                                        train_start_ts.strftime("%Y%m%d")
                                    )
                                    train_end_int = int(train_end_ts.strftime("%Y%m%d"))
                                else:
                                    train_start_int = int(
                                        str(train_start_ts).replace("-", "")[:8]
                                    )
                                    train_end_int = int(
                                        str(train_end_ts).replace("-", "")[:8]
                                    )

                                # Calculate prev_train_end for regime shift filtering
                                # Regime shifts after prev_train_end are from Valid/Test = leakage!
                                # We want to preserve ALL regime shifts from previous window's TRAIN period
                                #
                                # Example: Window 0 Train[2017-2019], Window 1 Train[2018-2020]
                                # - prev_train_end should be 20191231 (end of Window 0's TRAIN)
                                # - This keeps regime shifts from 2017-2019 (entire prev train)
                                # - Filters out regime shifts from 2020+ (Valid/Test of Window 0)
                                #
                                # Formula: prev_train_end = new_train_end - step_years
                                # With train=6, step=1: prev ends 1 year before new train ends
                                step_years = getattr(config, "wf_step_years", 1)

                                # Calculate prev_train_end = train_end - step_years
                                # This ensures we keep ALL regime shifts from previous TRAIN period
                                from datetime import datetime, timedelta
                                from dateutil.relativedelta import relativedelta

                                train_end_dt = datetime.strptime(
                                    str(train_end_int), "%Y%m%d"
                                )
                                prev_train_end_dt = train_end_dt - relativedelta(
                                    years=step_years
                                )
                                prev_train_end_int = int(
                                    prev_train_end_dt.strftime("%Y%m%d")
                                )

                                # Get buffer stats before filtering
                                if hasattr(buffer_obj, "get_buffer_stats"):
                                    pre_stats = buffer_obj.get_buffer_stats()
                                    smart_print(
                                        f"[RESUME] Buffer pre-filter: {pre_stats}",
                                        flush=True,
                                    )

                                # Filter buffer to retain only experiences within new train window
                                # Regime shifts are ONLY preserved if from prev train period (not Valid/Test)
                                original_count, retained_count, regime_preserved = (
                                    buffer_obj.filter_by_date_range(
                                        train_start_int,
                                        train_end_int,
                                        prev_train_end=prev_train_end_int,
                                    )
                                )

                                smart_print(
                                    f"[RESUME] Walk-forward buffer filter: {original_count} -> {retained_count} experiences "
                                    f"(train period: {train_start_int}-{train_end_int}, "
                                    f"prev_train_end: {prev_train_end_int}, regime_shift_preserved: {regime_preserved})",
                                    flush=True,
                                )

                                # Get buffer stats after filtering
                                if hasattr(buffer_obj, "get_buffer_stats"):
                                    post_stats = buffer_obj.get_buffer_stats()
                                    smart_print(
                                        f"[RESUME] Buffer post-filter: {post_stats}",
                                        flush=True,
                                    )

                                # Update buffer size tracking
                                last_loaded_buffer_size = retained_count

                            except Exception as e:
                                smart_print(
                                    f"[RESUME] Warning: Buffer filtering failed: {e}",
                                    flush=True,
                                )
            elif replay_buffer_path:
                smart_print(
                    f"[RESUME] Replay buffer file not found at {replay_buffer_path}",
                    flush=True,
                )
            else:
                smart_print(
                    "[RESUME] Replay buffer load skipped (reset_replay_buffer_on_resume=1 or no path present)",
                    flush=True,
                )

            # Load MAFIA observer from checkpoint if exists
            if (
                config.enable_market_observer
                and hasattr(env_train, "mkt_observer")
                and env_train.mkt_observer is not None
                and hasattr(env_train.mkt_observer, "load_checkpoint")
            ):
                mafia_checkpoint_path = os.path.join(
                    checkpoint_dir, "mafia_observer.pth"
                )
                if os.path.exists(mafia_checkpoint_path):
                    loaded_epoch = env_train.mkt_observer.load_checkpoint(
                        mafia_checkpoint_path
                    )
                    smart_print(
                        f"MAFIA observer loaded from {mafia_checkpoint_path} (epoch {loaded_epoch})",
                        flush=True,
                    )

                    # Reset LR scheduler if config says to (recommended for walk-forward)
                    if getattr(config, "reset_lr_scheduler_on_resume", True):
                        if hasattr(env_train.mkt_observer, "reset_lr_scheduler"):
                            env_train.mkt_observer.reset_lr_scheduler()
                            smart_print(
                                "[RESUME] LR scheduler reset to initial state (reset_lr_scheduler_on_resume=True)",
                                flush=True,
                            )
                else:
                    smart_print(
                        f"Warning: MAFIA observer checkpoint not found, starting fresh",
                        flush=True,
                    )
    else:
        model_para_dict = build_model_params()
        po_model = ModelCls(env=env_train, **model_para_dict)
        po_model.mafia_config = config
        po_model.verbose = 1
    # Ensure learning_starts is zeroed when resuming (skip warm-up)
    if checkpoint_to_resume is not None and hasattr(po_model, "learning_starts"):
        if buffer_loaded:
            config.learning_starts = 0
            po_model.learning_starts = 0
        else:
            # Keep original warm-up when buffer is reset/not loaded
            config.learning_starts = original_learning_starts
            po_model.learning_starts = original_learning_starts
        if replay_buffer_path and buffer_loaded and last_loaded_buffer_size <= 0:
            raise RuntimeError(
                f"Replay buffer at {replay_buffer_path} failed to load (size={last_loaded_buffer_size})"
            )

    # Calculate remaining timesteps and epochs
    # If resuming from checkpoint, calculate from checkpoint timesteps
    if checkpoint_timesteps > 0:
        total_timesteps_for_training = int(config.num_epochs * env_train.totalTradeDay)
        total_timesteps = max(0, total_timesteps_for_training - checkpoint_timesteps)
        # Calculate remaining epochs from checkpoint timesteps
        remaining_epochs = max(0, config.num_epochs - start_epoch)
        # More accurate: calculate from timesteps
        completed_epochs = checkpoint_timesteps // env_train.totalTradeDay
        remaining_epochs = max(0, config.num_epochs - completed_epochs)
        smart_print(
            f"Resuming from timestep {checkpoint_timesteps}/{total_timesteps_for_training}",
            flush=True,
        )
        smart_print(
            f"Completed epochs: {completed_epochs}, Remaining epochs: {remaining_epochs}",
            flush=True,
        )
    else:
        remaining_epochs = max(0, config.num_epochs - start_epoch)
        total_timesteps = int(remaining_epochs * env_train.totalTradeDay)

    # Print training information
    smart_print(f"\n{'=' * 100}")
    smart_print(f"{'TRAINING CONFIGURATION':^100}")
    # Print training information with enhanced formatting based on training mode
    training_mode = getattr(config, "training_mode", "RL_ONLY")
    if training_mode == "OBSERVER_ONLY":
        logger.print_banner(
            "PHASE 1: OBSERVER WALK-FORWARD - TD3 Frozen",
            TrainingPhase.OBSERVER_PRETRAIN,
        )
        smart_print(f"  {'─' * 70}", flush=True)
        smart_print(f"  🔮 Mode: Observer Training với uniform weights", flush=True)
        smart_print(f"  ❄️ TD3: FROZEN (không gradient updates)", flush=True)
    else:
        logger.print_banner(
            "PHASE 2: MAIN TRAINING - TD3 + Observer đồng thời", TrainingPhase.TRAIN
        )
    smart_print(f"  {'─' * 70}", flush=True)
    if start_epoch > 0 or checkpoint_timesteps > 0:
        smart_print(f"  🔄 Tiếp tục training từ epoch {start_epoch}", flush=True)
        smart_print(
            f"  📊 Epochs còn lại: {remaining_epochs}/{config.num_epochs}", flush=True
        )
    else:
        smart_print(f"  🆕 Bắt đầu training mới", flush=True)
        smart_print(f"  📊 Tổng epochs: {config.num_epochs}", flush=True)
    smart_print(f"  📊 Bước mỗi epoch: {env_train.totalTradeDay}", flush=True)
    smart_print(f"  📊 Tổng timesteps cho run này: {total_timesteps:,}", flush=True)
    smart_print(f"  📊 Batch size: {config.batch_size}", flush=True)
    smart_print(f"  📊 Learning rate: {config.learning_rate}", flush=True)
    smart_print(
        f"  📊 Checkpoint frequency: mỗi {config.checkpoint_freq} epochs"
        if config.checkpoint_freq > 0
        else "  📊 Checkpoint saving: TẮT",
        flush=True,
    )
    smart_print(f"  📂 Thư mục kết quả: {config.res_dir}", flush=True)
    smart_print(f"  {'─' * 70}", flush=True)
    smart_print("", flush=True)

    run_tracker.print_manifest_summary(
        getattr(config, "run_manifest_path", None), heading="RUN STATE SNAPSHOT"
    )

    logger.set_phase(TrainingPhase.TRAIN)
    
    log_interval = 10
    callback1 = PoCallback(
        config=config, train_env=env_train, valid_env=env_valid, test_env=env_test
    )
    cpt_start = time.process_time()
    perft_start = time.perf_counter()
    timeit_default = timeit.default_timer()

    # Set reset_num_timesteps=False when resuming from checkpoint to preserve timestep count
    reset_num_timesteps = checkpoint_timesteps == 0

    my_globals = globals()
    my_globals.update(
        {
            "po_model": po_model,
            "total_timesteps": total_timesteps,
            "callback1": callback1,
            "log_interval": log_interval,
            "reset_num_timesteps": reset_num_timesteps,
        }
    )
    t = timeit.Timer(
        stmt="po_model.learn(total_timesteps=total_timesteps, callback=callback1, log_interval=log_interval, reset_num_timesteps=reset_num_timesteps)",
        globals=my_globals,
    )
    time_usage = t.timeit(number=1)
    cpt_usgae = time.process_time() - cpt_start
    perf_usgae = time.perf_counter() - perft_start
    timeit_usgae = timeit.default_timer() - timeit_default

    # Print training completion with enhanced formatting
    logger.print_training_complete(
        total_time=time_usage,
        final_metrics={
            "Tổng epochs": config.num_epochs,
            "Thời gian thực thi": f"{time_usage:.2f}s",
            "CPU time": f"{cpt_usgae:.2f}s",
        },
    )

    smart_print(
        "Thời gian chạy {} epochs: {}s, cpu time: {}s, perf_counter: {}s".format(
            config.num_epochs,
            np.round(time_usage, 2),
            np.round(cpt_usgae, 2),
            np.round(perf_usgae, 2),
            np.round(timeit_usgae, 2),
        )
    )
    smart_print("-*" * 20)
    del po_model
    smart_print("Training Done...", flush=True)


def entrance():
    """
    Main entry point for MAFIA training.
    Legacy models (RLonly, Benchmark) have been removed.
    This codebase now exclusively runs MAFIA (RLcontroller with MAFIA observer).
    """
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    seed_env = os.environ.get("MAFIA_SEED")
    try:
        rand_seed = int(seed_env) if seed_env is not None else DEFAULT_RUN_SEED
    except ValueError:
        rand_seed = DEFAULT_RUN_SEED

    random.seed(rand_seed)
    os.environ["PYTHONHASHSEED"] = str(rand_seed)
    np.random.seed(rand_seed)
    th.manual_seed(rand_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(rand_seed)
        th.cuda.manual_seed_all(rand_seed)

    start_cputime = time.process_time()
    start_systime = time.perf_counter()
    config = Config(seed_num=rand_seed, current_date=current_date)

    smart_print("=" * 60)
    smart_print("MAFIA - Multi-Agent Framework with Integrated Attention")
    smart_print("=" * 60)
    config.print_config()
    smart_print("=" * 60)

    # Only MAFIA (RLcontroller mode) is supported
    RLcontroller(config=config)

    end_cputime = time.process_time()
    end_systime = time.perf_counter()
    smart_print(
        "[Done] Total cputime: {} s, system time: {} s".format(
            np.round(end_cputime - start_cputime, 2),
            np.round(end_systime - start_systime, 2),
        )
    )


def main():
    entrance()


if __name__ == "__main__":
    main()
