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
import sys
import json
import shutil
import random
import pickle
import torch as th
import math
sys.path.append('..')
from RL_controller.controllers import RL_withoutController, RL_withController

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
        self.last_known_portfolio = None
        # Checkpoint cleanup: keep all checkpoints (disable cleanup)
        self.enable_checkpoint_cleanup = False  # Set to False to keep all checkpoints
        self.max_checkpoints_to_keep = 2  # Only used if cleanup is enabled
        # Metrics logging
        self.metrics_file = os.path.join(self.config.res_dir, 'metrics_history.csv')
        self.metric_fields = [
            'reward_sum',
            'final_capital',
            'annualReturn_pct',
            'netProfit_pct',
            'sharpeRatio',
            'volatility',
            'mdd'
        ]
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

        # Sort by timesteps (descending - newest first)
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

    def _save_checkpoint(self, checkpoint_name, current_epoch, current_day, checkpoint_type):
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        rl_checkpoint_path = os.path.join(checkpoint_dir, 'rl_model.zip')
        self.model.save(rl_checkpoint_path)

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
            'env_state_path': env_state_path,
            'rng_state_path': rng_state_path,
            'seed_num': getattr(self.config, 'seed_num', None),
        }
        info_path = os.path.join(checkpoint_dir, 'checkpoint_info.json')
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

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
        
        # Log at the start of each new epoch
        if current_epoch_global != self.last_logged_epoch:
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
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
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
            print(f"   🔄 Step {display_timesteps:4d}/{total_timesteps_for_training} | Epoch {current_epoch_global}/{self.config.num_epochs} Day {day_in_epoch_display:4d}/{total_days} | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_seconds/60:.1f}m", flush=True)
            sys.stdout.flush()

        # Step-based checkpointing
        current_day = env_day

        if (self.partial_checkpoint_steps and
            self.num_timesteps > 0 and
            self.num_timesteps % self.partial_checkpoint_steps == 0 and
            self.num_timesteps != self.last_step_checkpoint):
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
            if should_validate:
                trained_model = ModelCls.load(curmpath)
                if self.valid_env is not None:
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
                    except Exception as e:
                        print(f"❌ Validation failed: {e}", flush=True)
                        valid_profile = None
                    finally:
                        if hasattr(self.valid_env, 'validation_mode'):
                            self.valid_env.validation_mode = False

                if self.test_env is not None:
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
                    except Exception as e:
                        print(f"❌ Test evaluation metrics unavailable: {e}", flush=True)
            else:
                print("[CALLBACK] Skipping validation/test for this epoch (will run after final epoch).", flush=True)

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

                if train_freq_unit and hasattr(train_freq_unit, 'name'):
                    print(f"[ROLLOUT_END] train_freq unit: {train_freq_unit.name}", flush=True)
                else:
                    print(f"[ROLLOUT_END] train_freq unit: {train_freq_unit}", flush=True)

                # Check the actual conditions for training
                should_train = (buffer_size >= learning_starts and
                              ((train_freq_unit == TrainFrequencyUnit.EPISODE and hasattr(self.model, '_episode_num') and self.model._episode_num > 0) or
                               (train_freq_unit == TrainFrequencyUnit.STEP and self.num_timesteps % train_freq.frequency == 0)))
                print(f"[ROLLOUT_END] Should train: {should_train} (buffer_size >= learning_starts: {buffer_size >= learning_starts}, train_freq_unit: {train_freq_unit})", flush=True)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def _record_metrics(self, epoch, metrics_payload):
        rows = []
        for phase, profile in metrics_payload.items():
            if profile is None:
                continue
            rows.append(self._build_metric_row(epoch, phase, profile))
        if not rows:
            return
        df = pd.DataFrame(rows)
        header = not os.path.exists(self.metrics_file)
        df.to_csv(self.metrics_file, mode='a', header=header, index=False)
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
        metrics = self.metric_fields
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
                ax.plot(subset['epoch'], subset[metric], marker='o', label=phase)
            ax.set_title(metric)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
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
