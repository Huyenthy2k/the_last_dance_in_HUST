#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test with UNBUFFERED output - logs hiển thị ngay lập tức
"""

import os
import sys
import warnings

# FORCE UNBUFFERED OUTPUT - Quan trọng!
sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# SUPPRESS WARNINGS to keep logs clean
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Degrees of freedom.*')
warnings.filterwarnings('ignore', message='.*Downcasting object dtype.*')

import random 
import numpy as np
import torch as th
import datetime
import time

if th.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
else:
    print("CUDA not available, using CPU", flush=True)

import pandas as pd
from config import Config
from utils.tradeEnv import StockPortfolioEnv
from utils.model_pool import model_select
from utils.data_validator import get_stock_data_file
from RL_controller.mafia_observer import MAFIAObserver
from utils.mafia_data_loader import MAFIADataLoader
from stable_baselines3.common.callbacks import BaseCallback


class SimpleMonitorCallback(BaseCallback):
    """Simple callback với output rõ ràng, real-time."""
    
    def __init__(self, config, train_env, verbose=0):
        super().__init__(verbose)
        self.config = config
        self.train_env = train_env
        self.last_logged_epoch = -1
        self.start_time = time.time()
        
    def _on_training_start(self) -> None:
        """Called at start of training."""
        print("\n" + "="*100, flush=True)
        print("🚀 TRAINING STARTED 🚀".center(100), flush=True)
        print("="*100, flush=True)
        print(f"⏰ Time: {datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)
        print("="*100 + "\n", flush=True)
    
    def _on_step(self) -> bool:
        """Called after each step - print progress clearly."""
        current_epoch = getattr(self.train_env, 'epoch', 0)
        current_day = getattr(self.train_env, 'curTradeDay', 0)
        total_days = getattr(self.train_env, 'totalTradeDay', 1)
        
        # New epoch started
        if current_epoch != self.last_logged_epoch:
            elapsed = time.time() - self.start_time
            
            print("\n" + "="*100, flush=True)
            print(f"📊 EPOCH {current_epoch}/{self.config.num_epochs}", flush=True)
            print("="*100, flush=True)
            print(f"⏱️  Elapsed: {elapsed:.1f}s", flush=True)
            print(f"🔢 Total Timesteps: {self.num_timesteps}", flush=True)
            print(f"📈 Day in Epoch: {current_day}/{total_days}", flush=True)
            
            if hasattr(self.model, '_n_updates'):
                print(f"🔄 Network Updates: {self.model._n_updates}", flush=True)
            
            # Get portfolio value
            if hasattr(self.train_env, 'cur_capital'):
                pv = self.train_env.cur_capital
                initial = self.train_env.initial_asset
                return_pct = ((pv / initial) - 1) * 100
                print(f"💰 Portfolio Value: ${pv:,.2f} ({return_pct:+.2f}%)", flush=True)
            
            print("="*100, flush=True)
            sys.stdout.flush()
            
            self.last_logged_epoch = current_epoch
        
        # DEBUG: Print every 10 steps to see progress
        if self.num_timesteps % 10 == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            eta_seconds = (total_days * self.config.num_epochs - self.num_timesteps) / steps_per_sec if steps_per_sec > 0 else 0
            print(f"   🔄 Step {self.num_timesteps:4d} | Day {current_day:4d}/{total_days} | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_seconds/60:.1f}m", flush=True)
        
        # Print progress every 5% of epoch (more frequent updates)
        progress_interval = max(1, total_days // 20)  # 20 updates = every 5%
        if current_day > 0 and current_day % progress_interval == 0:
            progress = (current_day / total_days) * 100
            
            # Get current portfolio info
            if hasattr(self.train_env, 'cur_capital'):
                pv = self.train_env.cur_capital
                initial = self.train_env.initial_asset
                return_pct = ((pv / initial) - 1) * 100
                print(f"   ⏳ Progress: {progress:.0f}% | Day {current_day}/{total_days} | Portfolio: ${pv:,.0f} ({return_pct:+.2f}%)", flush=True)
            else:
                print(f"   ⏳ Progress: {progress:.0f}% ({current_day}/{total_days} days)", flush=True)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of rollout."""
        print(f"\n" + "="*100, flush=True)
        print(f"✅ ROLLOUT COMPLETE", flush=True)
        print("="*100, flush=True)
        
        # Try to get loss values from model
        actor_loss = None
        critic_loss = None
        
        if hasattr(self.model, 'logger'):
            try:
                if 'train/actor_loss' in self.model.logger.name_to_value:
                    actor_loss = self.model.logger.name_to_value['train/actor_loss']
                
                if 'train/critic_loss' in self.model.logger.name_to_value:
                    critic_loss = self.model.logger.name_to_value['train/critic_loss']
            except:
                pass
        
        # Also try to get from model attributes
        if actor_loss is None and hasattr(self.model, 'actor'):
            try:
                if hasattr(self.model, '_n_updates') and self.model._n_updates > 0:
                    # Loss values should be available after training
                    pass
            except:
                pass
        
        # Print available metrics
        print(f"🔄 Network Updates: {getattr(self.model, '_n_updates', 0)}", flush=True)
        
        if actor_loss is not None:
            print(f"📉 Actor Loss: {actor_loss:.6f}", flush=True)
        else:
            print(f"📉 Actor Loss: (Training in progress...)", flush=True)
            
        if critic_loss is not None:
            print(f"📉 Critic Loss: {critic_loss:.6f}", flush=True)
        else:
            print(f"📉 Critic Loss: (Training in progress...)", flush=True)
        
        # Portfolio summary
        if hasattr(self.train_env, 'cur_capital'):
            pv = self.train_env.cur_capital
            initial = self.train_env.initial_asset
            return_pct = ((pv / initial) - 1) * 100
            print(f"💰 Current Portfolio: ${pv:,.2f} ({return_pct:+.2f}%)", flush=True)
        
        print("="*100, flush=True)
        sys.stdout.flush()
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*100, flush=True)
        print("✅ TRAINING COMPLETED ✅".center(100), flush=True)
        print("="*100, flush=True)
        print(f"⏱️  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)", flush=True)
        
        if hasattr(self.train_env, 'cur_capital'):
            pv = self.train_env.cur_capital
            initial = self.train_env.initial_asset
            return_pct = ((pv / initial) - 1) * 100
            print(f"💰 Final Portfolio Value: ${pv:,.2f} ({return_pct:+.2f}%)", flush=True)
        
        print("="*100 + "\n", flush=True)


def main():
    """Main training function with simple monitoring."""
    print("\n" + "="*100, flush=True)
    print("🎯 SIMPLE TRAINING MONITOR - Real-time Logs".center(100), flush=True)
    print("="*100, flush=True)
    print(flush=True)
    
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    rand_seed = 2022
    
    # Set seeds
    random.seed(rand_seed)
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    th.manual_seed(rand_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(rand_seed)
        th.cuda.manual_seed_all(rand_seed)
    
    # Create config
    config = Config(seed_num=rand_seed, current_date=current_date)
    config.benchmark_algo = 'MASA-mafia'
    config.enable_market_observer = True
    config.num_epochs = 3  # 3 epochs for demo
    config.checkpoint_freq = 1
    
    print(f"📋 Configuration:", flush=True)
    print(f"   Mode: {config.mode}", flush=True)
    print(f"   Epochs: {config.num_epochs}", flush=True)
    print(f"   Market: {config.market_name}", flush=True)
    print(f"   Top K stocks: {config.topK}", flush=True)
    print(flush=True)
    
    # Load data
    fpath, error_msg = get_stock_data_file(config)
    if fpath is None:
        raise ValueError(f"Cannot load data: {error_msg}")
    print(f"📂 Loading data from: {fpath}", flush=True)
    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Process data
    print("🔄 Processing data...", flush=True)
    mafia_loader = MAFIADataLoader(config=config)
    data_dict = mafia_loader.load_and_split_data(data=data)
    tech_indicator_lst = []
    stock_num = data_dict['train']['stock'].nunique()
    print(f"✅ Loaded {stock_num} stocks", flush=True)
    print(flush=True)
    
    # Initialize MAFIA observer
    print("🤖 Initializing MAFIA Observer...", flush=True)
    mkt_observer = MAFIAObserver(config=config, action_dim=stock_num)
    
    # Initialize environments
    print("🌍 Initializing environments...", flush=True)
    trainInvest_env_para = config.invest_env_para
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train',
        stock_num=stock_num, action_dim=stock_num,
        tech_indicator_lst=tech_indicator_lst,
        extra_data=data_dict['extra_train'],
        mkt_observer=mkt_observer, **trainInvest_env_para
    )
    
    # Load RL model
    print("🧠 Initializing TD3 model...", flush=True)
    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    model_para_dict = config.model_para
    po_model = ModelCls(env=env_train, **model_para_dict)
    
    # Calculate timesteps
    total_timesteps = int(config.num_epochs * env_train.totalTradeDay)
    
    print(flush=True)
    print("="*100, flush=True)
    print(f"🎯 Ready to train!", flush=True)
    print(f"   Total timesteps: {total_timesteps:,}", flush=True)
    print(f"   Steps per epoch: {env_train.totalTradeDay}", flush=True)
    print("="*100, flush=True)
    print(flush=True)
    
    # Create simple callback
    callback = SimpleMonitorCallback(config=config, train_env=env_train)
    
    # Start training
    print("🏁 STARTING TRAINING NOW...\n", flush=True)
    start_time = time.time()
    
    po_model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10  # Log every 10 steps
    )
    
    end_time = time.time()
    
    print(flush=True)
    print("="*100, flush=True)
    print(f"✅ Training finished in {end_time - start_time:.1f}s", flush=True)
    print(f"📁 Results: {config.res_dir}", flush=True)
    print("="*100, flush=True)
    
    del po_model


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user", flush=True)
        exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

