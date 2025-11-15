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
from .model_pool import model_select
import sys
import json
import shutil
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
        # Checkpoint cleanup: keep only the 2 most recent checkpoints
        self.max_checkpoints_to_keep = 2
    
    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only the N most recent ones (based on timesteps).
        """
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

        checkpoint_info = {
            'type': checkpoint_type,
            'epoch': int(current_epoch),
            'day_in_epoch': int(current_day),
            'timesteps': int(self.num_timesteps),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'rl_model_path': rl_checkpoint_path,
            'mafia_observer_path': mafia_checkpoint_path,
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
        """
        pass

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
        progress_pct = (self.num_timesteps / total_timesteps_for_training * 100.0) if total_timesteps_for_training > 0 else 0.0
        progress_pct = min(progress_pct, 100.0)
        
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
            remaining_timesteps = max(0, total_timesteps_for_training - self.num_timesteps)
            eta_seconds = remaining_timesteps / steps_per_sec if steps_per_sec > 0 else 0
            # Show both global timesteps and local day in epoch
            print(f"   🔄 Step {self.num_timesteps:4d}/{total_timesteps_for_training} | Epoch {current_epoch_global}/{self.config.num_epochs} Day {day_in_epoch_display:4d}/{total_days} | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_seconds/60:.1f}m", flush=True)
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
            # Evaluate model in validation set and test set
            ModelCls = model_select(model_name=self.config.rl_model_name,  mode=self.config.mode)
            trained_model = ModelCls.load(curmpath)
            if self.valid_env is not None:
                obs_valid = self.valid_env.reset()
                # Handle gymnasium return format: (obs, info) vs gym/VecEnv: obs
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
                    # Handle gymnasium return format: (obs, reward, terminated, truncated, info) vs gym: (obs, reward, done, info)
                    if len(step_result) == 5:
                        obs_valid, rewards, terminal_flag, _, _ = step_result
                    else:
                        obs_valid, rewards, terminal_flag, _ = step_result 
                    if terminal_flag:
                        break    
            
                cur_ep = self.valid_env.epoch # self.valid_env.epoch is the epoch number before reset().
                env_type = 'valid'
                fpath = os.path.join(self.config.res_dir, '{}_bestmodel.csv'.format(env_type))
                # Handle case when file doesn't exist or is empty
                if os.path.exists(fpath):
                    try:
                        model_records = pd.read_csv(fpath, header=0)
                        if len(model_records) > 0:
                            ep_col = '{}_ep'.format(self.config.trained_best_model_type)
                            if ep_col in model_records.columns:
                                if cur_ep == int(model_records[ep_col].iloc[0]):
                                    mpath = os.path.join(self.config.res_model_dir, '{}_{}'.format(env_type, self.config.trained_best_model_type))
                                    trained_model.save(mpath)
                    except Exception as e:
                        print(f"Warning: Could not read or process {fpath}: {e}", flush=True)
                
            if self.test_env is not None:
                obs_test = self.test_env.reset()
                # Handle gymnasium return format: (obs, info) vs gym/VecEnv: obs
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
                    # Handle gymnasium return format: (obs, reward, terminated, truncated, info) vs gym: (obs, reward, done, info)
                    if len(step_result) == 5:
                        obs_test, rewards, terminal_flag, _, _ = step_result
                    else:
                        obs_test, rewards, terminal_flag, _ = step_result 
                    if terminal_flag:
                        break

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
            print(f"[EPOCH COMPLETE] Model saved and evaluated on validation/test sets", flush=True)
            if hasattr(self.train_env, 'cur_capital'):
                pv = self.train_env.cur_capital
                initial = getattr(self.train_env, 'initial_asset', 1.0)
                return_pct = ((pv / initial) - 1) * 100 if initial else 0.0
                print(f"[EPOCH COMPLETE] Final portfolio value: ${pv:,.2f} ({return_pct:+.2f}%)", flush=True)
            print(f"{'='*100}\n", flush=True)
            
        self.train_env.model_save_flag = False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
