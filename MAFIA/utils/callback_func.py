#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: callback_func.py  
 Author: MASA
--------------------------------
'''
import numpy as np
import os
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFrequencyUnit
from .model_pool import model_select
from . import run_tracker
import sys
import json
import shutil
import random
import pickle
import torch as th
import math
import importlib
import gzip
sys.path.append('..')
from RL_controller.controllers import RL_withoutController, RL_withController
# Robust import of postprocess_topk
try:
    import postprocess_topk  # type: ignore
except Exception:
    postprocess_topk = None

class PoCallback(BaseCallback):

    def __init__(self, config, train_env, valid_env=None, test_env=None, verbose=0):
        super(PoCallback, self).__init__(verbose)
        self.train_env = train_env
        self.valid_env = valid_env
        self.test_env = test_env
        self.config = config
        if self.config.mode == 'RLonly': 
            self.risk_controller = RL_withoutController
        elif self.config.mode == 'RLcontroller':
            self.risk_controller = RL_withController
        else:
            raise ValueError("Unexpected mode [{}]..".format(self.config.mode))
        # Track last checkpoint epoch
        self.last_checkpoint_epoch = -1
        self.partial_checkpoint_steps = getattr(self.config, 'partial_checkpoint_steps', 0)
        self.last_step_checkpoint = -1
        # Track progress
        self.last_logged_epoch = -1
        self.step_counter = 0
        self.start_time = time.time()
        self._last_speed_time = self.start_time
        self._last_speed_steps = 0
        self.last_known_portfolio = None
        # Checkpoint / replay-buffer housekeeping configuration
        self.enable_checkpoint_cleanup = getattr(config, 'enable_checkpoint_cleanup', False)
        self.max_checkpoints_to_keep = getattr(config, 'max_checkpoints_to_keep', 2)
        self.save_replay_buffer_on_step_checkpoints = getattr(
            config, 'save_replay_buffer_on_step_checkpoints', False
        )
        self.save_replay_buffer_on_epoch_checkpoints = getattr(
            config, 'save_replay_buffer_on_epoch_checkpoints', True
        )
        self.early_stop_patience = getattr(config, 'early_stop_patience', 0)
        # Metrics logging
        default_metrics_path = getattr(self.config, 'metrics_history_path', None)
        if not default_metrics_path:
            default_metrics_path = os.path.join(self.config.res_dir, 'metrics_history.csv')
        self.metrics_file = default_metrics_path
        self.manifest_path = getattr(self.config, 'run_manifest_path', None)
        self.metric_fields = [
            'reward_sum',
            'final_capital',
            'annualReturn_pct',
            'netProfit_pct',
            'sharpeRatio',
            'volatility',
            'mdd'
        ]
        # TD3 training diagnostics (populated from TD3_controller.mafia_config)
        self.td3_loss_fields = [
            'td3_actor_loss',
            'td3_critic_loss',
            'td3_mean_reward',
        ]
        # MAFIA observer training diagnostics
        self.mafia_loss_fields = [
            'mafia_loss',
            'mafia_direction_loss',
        ]
        self._rollout_debug_logged = False
        self._training_ready = False
        self._warmup_notice_printed = False
        self.best_valid_sharpe = -np.inf
        self.valid_no_improve_epochs = 0
        self._early_stop = False
        self.best_valid_checkpoint_path = None

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
                pp_mod = importlib.import_module('postprocess_topk')
            except Exception as e:
                print(f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: cannot import postprocess_topk ({e})", flush=True)
                return None

        actions_array = None
        stock_lst = None

        # Primary source: env actions_memory
        if env is not None and hasattr(env, 'actions_memory') and hasattr(env, 'stock_lst'):
            stock_lst = getattr(env, 'stock_lst')
            actions = getattr(env, 'actions_memory', None)
            if actions is not None:
                actions_array = np.array(actions)
                if actions_array.ndim != 2:
                    try:
                        actions_array = actions_array.reshape(len(actions_array), -1)
                    except Exception as e:
                        print(f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: cannot reshape actions ({e})", flush=True)
                        actions_array = None
                if actions_array is not None and actions_array.shape[1] == len(stock_lst) + 1:
                    actions_array = actions_array[:, 1:]  # drop cash

        # Fallback: load actions CSV from res_dir
        if actions_array is None:
            actions_file = os.path.join(getattr(self.config, 'res_dir', '.'), f"{phase}_actions.csv")
            if os.path.exists(actions_file):
                try:
                    df = pd.read_csv(actions_file)
                    if 'date' in df.columns:
                        df = df.drop(columns=['date'])
                    actions_array = df.to_numpy()
                    stock_lst = df.columns.tolist() if stock_lst is None else stock_lst
                    print(f"[POSTPROCESS_TOPK] Using actions from file {actions_file}", flush=True)
                except Exception as e:
                    print(f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: failed to load {actions_file} ({e})", flush=True)
            else:
                print(f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: actions_memory/file not available", flush=True)
                return None

        if stock_lst is None:
            print(f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: stock list unavailable", flush=True)
            return None

        interval = getattr(self.config, 'rebalance_interval', 15)
        if actions_array.shape[0] < interval:
            print(f"[POSTPROCESS_TOPK] Skipped {phase} epoch {epoch}: only {actions_array.shape[0]} steps < interval {interval}", flush=True)
            return None
        periods = max(1, min(actions_array.shape[0] // max(interval, 1), 6))
        k = getattr(self.config, 'topK', min(len(stock_lst), 10))
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
            print(f"[POSTPROCESS_TOPK] Skip {phase} epoch {epoch}: {e}", flush=True)
            return None
        symbols = np.array(env.stock_lst)
        out_dir = getattr(self.config, 'res_dir', '.')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'topk_{phase}_epoch{epoch}.csv')
        try:
            pd.DataFrame({
                'symbol': symbols[top_idx] if len(symbols) > 0 else top_idx,
                'weight': w_final,
            }).to_csv(out_path, index=False)
            print(f"[POSTPROCESS_TOPK] Saved {phase} Top-K epoch {epoch} to {out_path}", flush=True)
            return out_path
        except Exception as e:
            print(f"[POSTPROCESS_TOPK] Failed to save {phase} Top-K: {e}", flush=True)
            return None

    def _run_topk_postprocess_all(self, epoch, run_validation=False, run_test=False):
        if not getattr(self.config, 'enable_topk_postprocess', True):
            return
        self._build_topk_recommendation(self.train_env, phase='train', epoch=epoch)
        if run_validation and self.valid_env is not None:
            self._build_topk_recommendation(self.valid_env, phase='valid', epoch=epoch)
        if run_test and self.test_env is not None:
            self._build_topk_recommendation(self.test_env, phase='test', epoch=epoch)
    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only the N most recent ones (based on timesteps).
        Can be disabled via self.enable_checkpoint_cleanup.
        """
        # Skip cleanup if disabled
        if not getattr(self, 'enable_checkpoint_cleanup', True):
            # Just log checkpoint count, don't delete anything
            if os.path.exists(self.config.checkpoint_dir):
                checkpoint_dirs = []
                try:
                    items = os.listdir(self.config.checkpoint_dir)
                    for item in items:
                        checkpoint_path = os.path.join(self.config.checkpoint_dir, item)
                        if os.path.isdir(checkpoint_path) and item.startswith('checkpoint_'):
                            info_path = os.path.join(checkpoint_path, 'checkpoint_info.json')
                            if os.path.exists(info_path):
                                try:
                                    with open(info_path, 'r') as f:
                                        info = json.load(f)
                                        checkpoint_dirs.append({
                                            'name': item,
                                            'timesteps': info.get('timesteps', 0),
                                        })
                                except Exception:
                                    pass
                except Exception as e:
                    print(f"[Checkpoint Storage] Error listing checkpoint directory: {e}", flush=True)
                    return

                if len(checkpoint_dirs) > 0:
                    checkpoint_dirs.sort(key=lambda x: x['timesteps'], reverse=True)
                    print(f"[Checkpoint Storage] Total checkpoints stored: {len(checkpoint_dirs)} (cleanup disabled)", flush=True)
                    print(f"[Checkpoint Storage] Latest checkpoint: {checkpoint_dirs[0]['name']} (timesteps: {checkpoint_dirs[0]['timesteps']})", flush=True)
                else:
                    print(f"[Checkpoint Storage] No checkpoints found in {self.config.checkpoint_dir}", flush=True)
            return

        # Original cleanup logic
        if not os.path.exists(self.config.checkpoint_dir):
            print(f"[Checkpoint Cleanup] Checkpoint directory does not exist: {self.config.checkpoint_dir}", flush=True)
            return

        # Get all checkpoint directories
        checkpoint_dirs = []
        try:
            items = os.listdir(self.config.checkpoint_dir)
        except Exception as e:
            print(f"[Checkpoint Cleanup] Error listing checkpoint directory: {e}", flush=True)
            return

        for item in items:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, item)
            if os.path.isdir(checkpoint_path) and item.startswith('checkpoint_'):
                info_path = os.path.join(checkpoint_path, 'checkpoint_info.json')
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                            timesteps = info.get('timesteps', 0)
                            checkpoint_dirs.append({
                                'path': checkpoint_path,
                                'name': item,
                                'timesteps': timesteps,
                                'timestamp': info.get('timestamp', '')
                            })
                    except Exception as e:
                        print(f"[Checkpoint Cleanup] Warning: Could not read checkpoint info from {info_path}: {e}", flush=True)

        if len(checkpoint_dirs) == 0:
            print(f"[Checkpoint Cleanup] No checkpoints found in {self.config.checkpoint_dir}", flush=True)
            return

        # Protect best-valid checkpoints from cleanup
        protected = [c for c in checkpoint_dirs if c['name'].startswith('checkpoint_best_valid')]
        checkpoint_dirs = [c for c in checkpoint_dirs if not c['name'].startswith('checkpoint_best_valid')]

        # Sort by timesteps (descending - newest first) for non-protected
        checkpoint_dirs.sort(key=lambda x: x['timesteps'], reverse=True)

        print(f"[Checkpoint Cleanup] Found {len(checkpoint_dirs)} checkpoints, keeping {self.max_checkpoints_to_keep} most recent", flush=True)

        # Keep only the N most recent checkpoints
        if len(checkpoint_dirs) > self.max_checkpoints_to_keep:
            checkpoints_to_delete = checkpoint_dirs[self.max_checkpoints_to_keep:]
            print(f"[Checkpoint Cleanup] Will delete {len(checkpoints_to_delete)} old checkpoint(s):", flush=True)
            for checkpoint in checkpoints_to_delete:
                print(f"  - {checkpoint['name']} (timesteps: {checkpoint['timesteps']}, timestamp: {checkpoint['timestamp']})", flush=True)

            deleted_count = 0
            for checkpoint in checkpoints_to_delete:
                try:
                    shutil.rmtree(checkpoint['path'])
                    deleted_count += 1
                    print(f"[Checkpoint Cleanup] ✓ Deleted: {checkpoint['name']} (timesteps: {checkpoint['timesteps']})", flush=True)
                except Exception as e:
                    print(f"[Checkpoint Cleanup] ✗ Error deleting {checkpoint['name']}: {e}", flush=True)

            print(f"[Checkpoint Cleanup] Cleanup complete: deleted {deleted_count}/{len(checkpoints_to_delete)} checkpoint(s)", flush=True)
        else:
            print(f"[Checkpoint Cleanup] No cleanup needed: {len(checkpoint_dirs)} checkpoint(s) <= {self.max_checkpoints_to_keep} (max to keep)", flush=True)

        if protected:
            print(f"[Checkpoint Cleanup] Protected best-valid checkpoints: {[c['name'] for c in protected]}", flush=True)

    def _save_checkpoint(self, checkpoint_name, current_epoch, current_day, checkpoint_type):
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        rl_checkpoint_path = os.path.join(checkpoint_dir, 'rl_model.zip')
        self.model.save(rl_checkpoint_path)

        replay_buffer_path = None
        should_save_replay_buffer = True
        if checkpoint_type.startswith('step') and not self.save_replay_buffer_on_step_checkpoints:
            should_save_replay_buffer = False
        elif checkpoint_type.startswith('epoch') and not self.save_replay_buffer_on_epoch_checkpoints:
            should_save_replay_buffer = False

        if (should_save_replay_buffer and
            hasattr(self.model, 'replay_buffer') and
            getattr(self.model, 'replay_buffer', None) is not None):
            replay_buffer_path = os.path.join(checkpoint_dir, 'replay_buffer.pkl')
            try:
                self.model.save_replay_buffer(replay_buffer_path)
                # Compress replay buffer to save disk space
                compressed_path = replay_buffer_path + ".gz"
                try:
                    with open(replay_buffer_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(replay_buffer_path)
                    replay_buffer_path = compressed_path
                    print(f"[Checkpoint Storage] Replay buffer compressed to {compressed_path}", flush=True)
                except Exception as e:
                    print(f"Warning: Replay buffer compression failed (keeping uncompressed): {e}", flush=True)
            except Exception as e:
                print(f"Warning: Failed to save replay buffer: {e}", flush=True)
                replay_buffer_path = None
        elif not should_save_replay_buffer:
            print(f"[Checkpoint Storage] Skipping replay buffer save for {checkpoint_type} checkpoint", flush=True)

        mafia_checkpoint_path = None
        if (hasattr(self.train_env, 'mkt_observer') and
            self.train_env.mkt_observer is not None and
            hasattr(self.train_env.mkt_observer, 'save_checkpoint')):
            mafia_checkpoint_path = os.path.join(checkpoint_dir, 'mafia_observer.pth')
            try:
                self.train_env.mkt_observer.save_checkpoint(mafia_checkpoint_path, current_epoch)
            except Exception as e:
                print(f"Warning: Failed to save MAFIA observer checkpoint: {e}", flush=True)
                mafia_checkpoint_path = None

        # Save environment state for mid-epoch resume
        env_state_path = None
        if hasattr(self.train_env, 'save_state'):
            env_state_path = os.path.join(checkpoint_dir, 'env_state.pkl')
            try:
                self.train_env.save_state(env_state_path)
            except Exception as e:
                print(f"Warning: Failed to save environment state: {e}", flush=True)
                env_state_path = None

        # Save RNG state for deterministic resume
        rng_state_path = None
        try:
            rng_state_path = os.path.join(checkpoint_dir, 'rng_state.pkl')
            rng_state = {
                'seed_num': getattr(self.config, 'seed_num', None),
                'python_random': random.getstate(),
                'numpy_random': np.random.get_state(),
                'torch_cpu': th.get_rng_state(),
            }
            if th.cuda.is_available():
                rng_state['torch_cuda'] = th.cuda.get_rng_state_all()
            with open(rng_state_path, 'wb') as f:
                pickle.dump(rng_state, f)
        except Exception as e:
            print(f"Warning: Failed to save RNG state: {e}", flush=True)
            rng_state_path = None

        checkpoint_info = {
            'type': checkpoint_type,
            'epoch': int(current_epoch),
            'day_in_epoch': int(current_day),
            'timesteps': int(self.num_timesteps),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'rl_model_path': rl_checkpoint_path,
            'mafia_observer_path': mafia_checkpoint_path,
            'replay_buffer_path': replay_buffer_path,
            'env_state_path': env_state_path,
            'rng_state_path': rng_state_path,
            'seed_num': getattr(self.config, 'seed_num', None),
        }
        info_path = os.path.join(checkpoint_dir, 'checkpoint_info.json')
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        run_tracker.record_checkpoint_save(self.manifest_path, checkpoint_name, checkpoint_info)

        # Clean up old checkpoints after saving new one
        self._cleanup_old_checkpoints()

        return checkpoint_dir

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Clean up old checkpoints at training start (in case there are old checkpoints from previous runs)
        print(f"[Checkpoint Cleanup] Starting cleanup at training start...", flush=True)
        self._cleanup_old_checkpoints()
        run_tracker.print_manifest_summary(self.manifest_path, heading="RUN STATE SNAPSHOT")
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
        if hasattr(self.train_env, '_resume_env_state_path'):
            env_state_path = self.train_env._resume_env_state_path
            if env_state_path and os.path.exists(env_state_path) and hasattr(self.train_env, 'restore_state'):
                try:
                    print(f"[RESUME] Restoring environment state from {env_state_path}...", flush=True)
                    self.train_env.restore_state(env_state_path)
                    print(f"[RESUME] Environment state restored successfully!", flush=True)
                    # Clear the resume path after restoring
                    delattr(self.train_env, '_resume_env_state_path')
                except Exception as e:
                    print(f"[RESUME] Warning: Failed to restore environment state: {e}", flush=True)
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
        env_epoch = self.train_env.epoch if hasattr(self.train_env, 'epoch') else 0
        env_day = getattr(self.train_env, 'curTradeDay', 0)
        total_days = self.train_env.totalTradeDay if hasattr(self.train_env, 'totalTradeDay') else 1
        if total_days <= 0:
            total_days = 1

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
            progress_pct = (self.num_timesteps / total_timesteps_for_training * 100.0) if total_timesteps_for_training > 0 else 0.0
            progress_pct = min(progress_pct, 100.0)
            display_timesteps = self.num_timesteps

        # Determine whether warm-up (buffer filling) is still happening
        learning_starts = getattr(self.model, 'learning_starts', 0)
        buffer_size = None
        replay_buffer = getattr(self.model, 'replay_buffer', None)
        if replay_buffer is not None:
            buffer_size = replay_buffer.size() if hasattr(replay_buffer, 'size') else len(replay_buffer)

        # Compute speed based on delta since last log (robust to resume)
        now = time.time()
        delta_steps = self.num_timesteps - self._last_speed_steps
        delta_time = now - self._last_speed_time
        steps_per_sec_delta = delta_steps / delta_time if delta_time > 0 else 0.0
        self._last_speed_steps = self.num_timesteps
        self._last_speed_time = now

        training_ready = self._training_ready
        if learning_starts is None or learning_starts <= 0:
            training_ready = True
        elif buffer_size is not None:
            training_ready = buffer_size >= learning_starts

        if training_ready and not self._training_ready and learning_starts:
            print(f"[WARM-UP COMPLETE] Replay buffer reached learning_starts ({buffer_size}/{learning_starts}). Enabling epoch progress + checkpoints.", flush=True)
        elif (not training_ready and not self._warmup_notice_printed and
              learning_starts and buffer_size is not None):
            print(f"[WARM-UP] Collecting experience | Buffer: {buffer_size}/{learning_starts} | Speed: {steps_per_sec_delta:.2f} steps/s | Epoch/Checkpoint logs will start after warm-up.", flush=True)
            self._warmup_notice_printed = True

        if training_ready:
            self._warmup_notice_printed = False

        self._training_ready = training_ready
        
        # Log at the start of each new epoch
        if self._training_ready and current_epoch_global != self.last_logged_epoch:
            self.last_logged_epoch = current_epoch_global
            elapsed = time.time() - self.start_time
            print(f"\n{'='*100}", flush=True)
            print(f"📊 EPOCH {current_epoch_global}/{self.config.num_epochs} ({progress_pct:.1f}%)", flush=True)
            print(f"{'='*100}", flush=True)
            print(f"⏱️  Elapsed: {elapsed:.1f}s", flush=True)
            print(f"🔢 Global Timesteps: {self.num_timesteps} (total across all epochs)", flush=True)
            print(f"📈 Day in Epoch: {day_in_epoch_display}/{total_days} (computed from global steps)", flush=True)
            if env_epoch != current_epoch_global or env_day != day_in_epoch_display:
                epoch_offset = env_epoch - current_epoch_global
                print(f"ℹ️  Sync note: env.epoch={env_epoch}, env.day={env_day}", flush=True)
                if epoch_offset == 1:
                    print(f"ℹ️  Note: env.epoch has +1 offset due to SB3's initial reset() call", flush=True)
                elif epoch_offset > 1:
                    print(f"⚠️  Warning: env.epoch offset is {epoch_offset} (expected +1). Check for unexpected resets!", flush=True)
            updates_attr = getattr(self.model, '_n_updates', None)
            if updates_attr is None:
                print(f"🔄 Network Updates: N/A", flush=True)
            else:
                updates_note = ""
                if updates_attr == 0 and env_day < (total_days - 1):
                    updates_note = " (next update runs at epoch end)"
                print(f"🔄 Network Updates: {updates_attr}{updates_note}", flush=True)
            if hasattr(self.train_env, 'cur_capital'):
                pv = self.train_env.cur_capital
                pv_note = ""
                if pv is None or not np.isfinite(pv):
                    if self.last_known_portfolio is not None:
                        pv = self.last_known_portfolio
                        pv_note = " (using last known value)"
                    else:
                        pv = getattr(self.train_env, 'initial_asset', 0.0)
                        pv_note = " (initial value)"
                else:
                    self.last_known_portfolio = pv
                initial = getattr(self.train_env, 'initial_asset', 1.0)
                return_pct = ((pv / initial) - 1) * 100 if initial else 0.0
                print(f"💰 Portfolio Value: ${pv:,.2f} ({return_pct:+.2f}%){pv_note}", flush=True)
            print(f"{'='*100}", flush=True)
            sys.stdout.flush()
        
        # Print progress every 100 steps
        if self.num_timesteps % 100 == 0:
            steps_per_sec = steps_per_sec_delta
            # Calculate remaining timesteps (accounting for resume from checkpoint)
            # If num_timesteps exceeds total, we're past the expected end, so remaining is 0
            if self.num_timesteps > total_timesteps_for_training:
                remaining_timesteps = 0
            else:
                remaining_timesteps = max(0, total_timesteps_for_training - self.num_timesteps)
            eta_seconds = remaining_timesteps / steps_per_sec if steps_per_sec > 0 else 0
            # Show both global timesteps and local day in epoch
            # Use display_timesteps to avoid showing values > total when resuming
            display_timesteps = min(self.num_timesteps, total_timesteps_for_training) if self.num_timesteps > total_timesteps_for_training else self.num_timesteps
            if self._training_ready:
                print(f"   🔄 Step {display_timesteps:4d}/{total_timesteps_for_training} | Epoch {current_epoch_global}/{self.config.num_epochs} Day {day_in_epoch_display:4d}/{total_days} | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_seconds/60:.1f}m", flush=True)
            else:
                warmup_target = learning_starts if isinstance(learning_starts, (int, float)) else '?'
                buffer_display = buffer_size if buffer_size is not None else '?'
                msg = f"   [WARM-UP] Global step {display_timesteps:4d} | Buffer {buffer_display}/{warmup_target} | Speed: {steps_per_sec:.2f} steps/s"
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()
            sys.stdout.flush()

        # Step-based checkpointing
        current_day = env_day

        if (self.partial_checkpoint_steps and
            self.num_timesteps > 0 and
            self.num_timesteps % self.partial_checkpoint_steps == 0 and
            self.num_timesteps != self.last_step_checkpoint and
            self._training_ready):
            checkpoint_name = f'checkpoint_step_{self.num_timesteps}'
            checkpoint_dir = self._save_checkpoint(
                checkpoint_name=checkpoint_name,
                current_epoch=current_epoch_global,
                current_day=current_day,
                checkpoint_type='step'
            )
            print(f"Checkpoint saved: {checkpoint_dir} (step {self.num_timesteps})", flush=True)
            self.last_step_checkpoint = self.num_timesteps
        
        # Save model
        if self.train_env.model_save_flag:
            exclusive_start_cputime = time.process_time()
            exclusive_start_systime = time.perf_counter()
            curmpath = os.path.join(self.config.res_model_dir, 'current_model')
            self.model.save(curmpath)
            validation_freq = getattr(self.config, 'validation_freq', None)
            should_validate = (
                (validation_freq is not None and validation_freq > 0 and current_epoch_global % validation_freq == 0)
                or (current_epoch_global >= self.config.num_epochs)
            )
            epoch_metrics = {'train': None, 'valid': None, 'test': None}
            train_profile = getattr(self.train_env, 'last_epoch_profile', None)
            if train_profile is None:
                train_profile = getattr(self.train_env, 'latest_invest_profile', None)
            if train_profile is None:
                try:
                    train_profile = self.train_env.get_results()
                except Exception:
                    train_profile = None
            epoch_metrics['train'] = train_profile
            
            # Save checkpoint if enabled and epoch matches frequency
            if (self.config.checkpoint_freq > 0 and 
                completed_epochs_global > 0 and 
                completed_epochs_global % self.config.checkpoint_freq == 0 and
                completed_epochs_global != self.last_checkpoint_epoch):

                checkpoint_epoch = completed_epochs_global
                checkpoint_name = f'checkpoint_epoch_{checkpoint_epoch}'
                checkpoint_dir = self._save_checkpoint(
                    checkpoint_name=checkpoint_name,
                    current_epoch=checkpoint_epoch,
                    current_day=current_day,
                    checkpoint_type='epoch'
                )

                print(f"Checkpoint saved: {checkpoint_dir} (epoch {checkpoint_epoch})", flush=True)
                self.last_checkpoint_epoch = checkpoint_epoch
            # Evaluate model in validation/test only after the final epoch
            ModelCls = model_select(model_name=self.config.rl_model_name,  mode=self.config.mode)
            trained_model = None
            # Run validation every validation_freq; run test only on final epoch
            run_validation = should_validate and self.valid_env is not None
            # Run test after each epoch (can be costly)
            run_test = should_validate and self.test_env is not None
            if run_validation or run_test:
                trained_model = ModelCls.load(curmpath)

            valid_profile = None
            if run_validation:
                if hasattr(self.valid_env, 'validation_mode'):
                    self.valid_env.validation_mode = True
                try:
                    obs_valid = self.valid_env.reset()
                    if isinstance(obs_valid, tuple):
                        obs_valid = obs_valid[0]
                    while True:
                        a_rlonly, _ = trained_model.predict(obs_valid)
                        a_rlonly = np.reshape(a_rlonly, (-1))
                        a_rl = a_rlonly
                        if np.sum(np.abs(a_rl)) == 0:
                            a_rl = np.array([1/len(a_rl)]*len(a_rl))
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
                        if terminal_flag:
                            break
                    valid_profile = self.valid_env.get_results()
                    epoch_metrics['valid'] = valid_profile
                    # Persist validation profile to CSV (validation_mode skips save_profile inside env)
                    try:
                        if valid_profile is not None:
                            self.valid_env.save_profile(valid_profile)
                    except Exception as e:
                        print(f"❌ Validation profile save failed: {e}", flush=True)
                except Exception as e:
                    print(f"❌ Validation failed: {e}", flush=True)
                    valid_profile = None
                finally:
                    if hasattr(self.valid_env, 'validation_mode'):
                        self.valid_env.validation_mode = False
                # Early stopping tracking
                if self.early_stop_patience and valid_profile is not None:
                    sharpe_val = valid_profile.get('sharpeRatio', None)
                    if sharpe_val is not None and np.isfinite(sharpe_val):
                        if sharpe_val > self.best_valid_sharpe + 1e-6:
                            self.best_valid_sharpe = sharpe_val
                            self.valid_no_improve_epochs = 0
                            try:
                                checkpoint_dir = self._save_checkpoint(
                                    checkpoint_name='checkpoint_best_valid',
                                    current_epoch=current_epoch_global,
                                    current_day=current_day,
                                    checkpoint_type='epoch_best_valid'
                                )
                                self.best_valid_checkpoint_path = checkpoint_dir
                                print(f"[BEST VALID] Sharpe improved to {sharpe_val:.4f}. Saved best-valid checkpoint: {checkpoint_dir}", flush=True)
                            except Exception as e:
                                print(f"[BEST VALID] Failed to save best-valid checkpoint: {e}", flush=True)
                        else:
                            self.valid_no_improve_epochs += 1
                            if self.valid_no_improve_epochs >= self.early_stop_patience:
                                print(f"[EARLY STOP] Validation Sharpe did not improve for {self.early_stop_patience} epochs (best={self.best_valid_sharpe:.4f}). Stopping training.", flush=True)
                                self._early_stop = True

            if run_test:
                obs_test = self.test_env.reset()
                if isinstance(obs_test, tuple):
                    obs_test = obs_test[0]
                while True:
                    a_rlonly, _ = trained_model.predict(obs_test)
                    a_rlonly = np.reshape(a_rlonly, (-1))
                    a_rl = a_rlonly
                    if np.sum(np.abs(a_rl)) == 0:
                        a_rl = np.array([1/len(a_rl)]*len(a_rl))
                    else:
                        a_rl = a_rl / np.sum(np.abs(a_rl))
                    a_final  = self.risk_controller(a_rl=a_rl, env=self.test_env)
                    a_final = a_final / np.sum(np.abs(a_final))
                    a_final = np.array([a_final])
                    step_result = self.test_env.step(a_final)
                    if len(step_result) == 5:
                        obs_test, rewards, terminal_flag, _, _ = step_result
                    else:
                        obs_test, rewards, terminal_flag, _ = step_result 
                    if terminal_flag:
                        break
                try:
                    test_profile = self.test_env.get_results()
                    epoch_metrics['test'] = test_profile
                    try:
                        if test_profile is not None:
                            self.test_env.save_profile(test_profile)
                    except Exception as e:
                        print(f"❌ Test profile save failed: {e}", flush=True)
                except Exception as e:
                    print(f"❌ Test evaluation metrics unavailable: {e}", flush=True)
            if not run_validation and not run_test:
                print("[CALLBACK] Skipping validation/test for this epoch (will run after final epoch).", flush=True)

            # Post-process Top-K recommendations for train/valid/test
            try:
                self._run_topk_postprocess_all(
                    current_epoch_global,
                    run_validation=run_validation,
                    run_test=run_test,
                )
            except Exception as e:
                print(f"[POSTPROCESS_TOPK] Aggregation failed: {e}", flush=True)

            if trained_model is not None:
                del trained_model
            # delete the current model file
            current_model_path = os.path.join(self.config.res_model_dir, 'current_model.zip')
            if os.path.exists(current_model_path):
                try:
                    os.remove(current_model_path)
                except Exception as e:
                    print(f"Warning: Could not remove {current_model_path}: {e}", flush=True)
            exclusive_end_cputime = time.process_time()
            exclusive_end_systime = time.perf_counter()
            self.train_env.exclusive_cputime = exclusive_end_cputime - exclusive_start_cputime
            self.train_env.exclusive_systime = exclusive_end_systime - exclusive_start_systime
            
            # Log epoch completion
            eval_time = exclusive_end_systime - exclusive_start_systime
            print(f"\n{'='*100}", flush=True)
            print(f"[EPOCH COMPLETE] Epoch {current_epoch_global} finished", flush=True)
            print(f"[EPOCH COMPLETE] Evaluation time: {eval_time:.2f}s", flush=True)
            if should_validate:
                print(f"[EPOCH COMPLETE] Model saved and evaluated on validation/test sets", flush=True)
            else:
                print(f"[EPOCH COMPLETE] Model saved (validation/test deferred to final epoch)", flush=True)
            if hasattr(self.train_env, 'cur_capital'):
                # Use final_capital from training metrics if available, otherwise fallback gracefully
                train_metrics = epoch_metrics.get('train') if epoch_metrics else None
                if not isinstance(train_metrics, dict):
                    train_metrics = {}
                pv = train_metrics.get('final_capital')
                if pv is None or not np.isfinite(pv):
                    pv = self.train_env.cur_capital
                if pv is None or not np.isfinite(pv):
                    if self.last_known_portfolio is not None:
                        pv = self.last_known_portfolio
                    else:
                        pv = getattr(self.train_env, 'initial_asset', 0.0)
                else:
                    self.last_known_portfolio = pv
                initial = getattr(self.train_env, 'initial_asset', 1.0)
                return_pct = ((pv / initial) - 1) * 100 if initial else 0.0
                print(f"[EPOCH COMPLETE] Final portfolio value: ${pv:,.2f} ({return_pct:+.2f}%)", flush=True)
            print(f"{'='*100}\n", flush=True)
            self._record_metrics(current_epoch_global, epoch_metrics)
            
        self.train_env.model_save_flag = False
        if self._early_stop:
            print(f"\n{'='*100}", flush=True)
            print(f"🛑 Early stopping triggered (validation). Best valid Sharpe: {self.best_valid_sharpe:.4f}", flush=True)
            print(f"{'='*100}\n", flush=True)
            return False
        
        # Check if we've reached total timesteps - stop training to prevent extra steps
        if self.num_timesteps >= total_timesteps_for_training:
            print(f"\n{'='*100}", flush=True)
            print(f"✅ Reached total timesteps: {self.num_timesteps}/{total_timesteps_for_training}", flush=True)
            print(f"🛑 Stopping training to prevent extra steps", flush=True)
            print(f"{'='*100}\n", flush=True)
            return False  # Stop training
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        This is called AFTER collect_rollouts() returns, but BEFORE train() is called by base class.
        """
        # Debug: Check if training should be triggered
        if hasattr(self.model, 'replay_buffer'):
            buffer_size = self.model.replay_buffer.size() if hasattr(self.model.replay_buffer, 'size') else len(self.model.replay_buffer)
            learning_starts = getattr(self.model, 'learning_starts', 100)
            n_updates_before = getattr(self.model, '_n_updates', 0)
            train_freq = getattr(self.model, 'train_freq', None)

            if not self._rollout_debug_logged:
                print(f"[ROLLOUT_END] Before training check | Buffer: {buffer_size}/{learning_starts} | _n_updates: {n_updates_before} | train_freq: {train_freq}", flush=True)

            # Check if training should happen (base class will check this)
            if train_freq is not None:
                # Extract unit from train_freq (could be TrainFreq object or tuple)
                if hasattr(train_freq, 'unit'):
                    train_freq_unit = train_freq.unit
                elif isinstance(train_freq, (tuple, list)) and len(train_freq) == 2:
                    train_freq_unit = TrainFrequencyUnit.EPISODE if train_freq[1] == 'episode' else TrainFrequencyUnit.STEP
                else:
                    train_freq_unit = None

                if not self._rollout_debug_logged:
                    if train_freq_unit and hasattr(train_freq_unit, 'name'):
                        print(f"[ROLLOUT_END] train_freq unit: {train_freq_unit.name}", flush=True)
                    else:
                        print(f"[ROLLOUT_END] train_freq unit: {train_freq_unit}", flush=True)

                # Check the actual conditions for training
                should_train = (buffer_size >= learning_starts and
                              ((train_freq_unit == TrainFrequencyUnit.EPISODE and hasattr(self.model, '_episode_num') and self.model._episode_num > 0) or
                               (train_freq_unit == TrainFrequencyUnit.STEP and self.num_timesteps % train_freq.frequency == 0)))
                if not self._rollout_debug_logged:
                    print(f"[ROLLOUT_END] Should train: {should_train} (buffer_size >= learning_starts: {buffer_size >= learning_starts}, train_freq_unit: {train_freq_unit})", flush=True)
                    self._rollout_debug_logged = True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        try:
            current_epoch = getattr(self.train_env, 'epoch', self.config.num_epochs)
            current_day = getattr(self.train_env, 'curTradeDay', 0)
            checkpoint_dir = self._save_checkpoint(
                checkpoint_name='checkpoint_final',
                current_epoch=current_epoch,
                current_day=current_day,
                checkpoint_type='epoch_final'
            )
            print(f"[Checkpoint] Final checkpoint saved: {checkpoint_dir}", flush=True)
        except Exception as e:
            print(f"[Checkpoint] Failed to save final checkpoint: {e}", flush=True)

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
            'td3_actor_loss': getattr(self.config, 'last_td3_actor_loss', np.nan),
            'td3_critic_loss': getattr(self.config, 'last_td3_critic_loss', np.nan),
            'td3_mean_reward': getattr(self.config, 'last_td3_mean_reward', np.nan),
        }
        mafia_snapshot = {
            'mafia_loss': getattr(self.config, 'last_mafia_loss', np.nan),
            'mafia_direction_loss': getattr(self.config, 'last_mafia_direction_loss', np.nan),
        }
        for key, val in td3_snapshot.items():
            df[key] = val
        for key, val in mafia_snapshot.items():
            df[key] = val
        header = not os.path.exists(self.metrics_file)
        df.to_csv(self.metrics_file, mode='a', header=header, index=False)
        run_tracker.record_metrics_update(self.manifest_path, epoch, df['phase'].tolist())
        print(f"[VISUALIZER] Logged metrics for epoch {epoch} phases: {', '.join(df['phase'].tolist())}", flush=True)
        self._update_metric_plot()

    def _build_metric_row(self, epoch, phase, profile):
        row = {'epoch': epoch, 'phase': phase}
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
            print(f"[VISUALIZER] Warning: Could not read metrics file: {e}", flush=True)
            return
        if df.empty:
            return
        metrics = list(dict.fromkeys(
            self.metric_fields
            + [m for m in self.td3_loss_fields if m in df.columns]
            + [m for m in self.mafia_loss_fields if m in df.columns]
        ))
        epoch_offset = int(df['epoch'].min()) if 'epoch' in df.columns else 0
        relative_epoch = df['epoch'] - epoch_offset + 1 if epoch_offset else df.get('epoch')
        rel_max = relative_epoch.max() if relative_epoch is not None else None
        phases = sorted(df['phase'].unique())
        n = len(metrics)
        cols = 2 if n > 1 else 1
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
        axes_flat = axes.flatten()
        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            if metric not in df.columns:
                ax.axis("off")
                continue
            for phase in phases:
                subset = df[df['phase'] == phase]
                if subset.empty:
                    continue
                x_vals = subset['epoch'] - epoch_offset + 1 if epoch_offset else subset['epoch']
                ax.plot(x_vals, subset[metric], marker='o', label=phase)
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
        output_dir = getattr(self.config, 'res_img_dir', self.config.res_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'metrics_history.png')
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[VISUALIZER] Updated metric plot at {output_path}", flush=True)
