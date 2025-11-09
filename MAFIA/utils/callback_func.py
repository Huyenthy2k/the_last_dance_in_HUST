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
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

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
        # Save model
        if self.train_env.model_save_flag:
            exclusive_start_cputime = time.process_time()
            exclusive_start_systime = time.perf_counter()
            curmpath = os.path.join(self.config.res_model_dir, 'current_model')
            self.model.save(curmpath)
            
            # Save checkpoint if enabled and epoch matches frequency
            current_epoch = self.train_env.epoch
            if (self.config.checkpoint_freq > 0 and 
                current_epoch > 0 and 
                current_epoch % self.config.checkpoint_freq == 0 and
                current_epoch != self.last_checkpoint_epoch):
                
                checkpoint_epoch = current_epoch
                checkpoint_name = f'checkpoint_epoch_{checkpoint_epoch}'
                checkpoint_dir = os.path.join(self.config.checkpoint_dir, checkpoint_name)
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save RL model checkpoint
                rl_checkpoint_path = os.path.join(checkpoint_dir, 'rl_model.zip')
                self.model.save(rl_checkpoint_path)
                
                # Save MAFIA observer checkpoint if exists
                mafia_checkpoint_path = None
                if (hasattr(self.train_env, 'mkt_observer') and 
                    self.train_env.mkt_observer is not None and
                    hasattr(self.train_env.mkt_observer, 'save_checkpoint')):
                    mafia_checkpoint_path = os.path.join(checkpoint_dir, 'mafia_observer.pth')
                    self.train_env.mkt_observer.save_checkpoint(mafia_checkpoint_path, checkpoint_epoch)
                
                # Save checkpoint info
                checkpoint_info = {
                    'epoch': checkpoint_epoch,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'rl_model_path': rl_checkpoint_path,
                    'mafia_observer_path': mafia_checkpoint_path,
                }
                import json
                info_path = os.path.join(checkpoint_dir, 'checkpoint_info.json')
                with open(info_path, 'w') as f:
                    json.dump(checkpoint_info, f, indent=2)
                
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
