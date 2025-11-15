#!/usr/bin/env python3
"""
Test script to verify TD3 training is working correctly.

This script:
1. Creates a minimal environment
2. Initializes TD3 model
3. Runs a few steps
4. Checks if training is triggered
5. Verifies buffer size and _n_updates
"""

import os
import sys
import numpy as np
import torch as th

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_td3_training():
    """Test TD3 training with minimal setup."""
    print("="*80)
    print("TD3 TRAINING TEST")
    print("="*80)
    print()
    
    try:
        from config import Config
        from RL_controller.TD3_controller import TD3Controller
        from utils.tradeEnv import StockPortfolioEnv
        from utils.mafia_data_loader import MAFIADataLoader
        import datetime
        import pandas as pd
        
        # Create config
        print("1. Creating config...")
        config = Config(seed_num=123, current_date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        print(f"   ✅ Config created")
        print(f"   learning_starts: {config.model_para.get('learning_starts', 'N/A')}")
        print(f"   train_freq: {config.model_para.get('train_freq', 'N/A')}")
        print(f"   verbose: {config.model_para.get('verbose', 'N/A')}")
        print()
        
        # Load minimal data
        print("2. Loading data...")
        data_file = os.path.join(config.dataDir, config.stock_data_file)
        if not os.path.exists(data_file):
            print(f"   ❌ Data file not found: {data_file}")
            return False
        
        data = pd.read_csv(data_file, header=0)
        mafia_loader = MAFIADataLoader(config=config)
        data_dict = mafia_loader.load_and_split_data(data=data)
        stock_num = data_dict['train']['stock'].nunique()
        print(f"   ✅ Data loaded: {stock_num} stocks")
        print()
        
        # Initialize MAFIA observer
        print("3. Initializing MAFIA observer...")
        from RL_controller.mafia_observer import MAFIAObserver
        mkt_observer = MAFIAObserver(config=config, action_dim=stock_num)
        print(f"   ✅ MAFIA observer initialized")
        print()
        
        # Create environment
        print("4. Creating environment...")
        trainInvest_env_para = config.invest_env_para
        env_train = StockPortfolioEnv(
            config=config, 
            rawdata=data_dict['train'], 
            mode='train', 
            stock_num=stock_num, 
            action_dim=stock_num, 
            tech_indicator_lst=[], 
            extra_data=data_dict['extra_train'], 
            mkt_observer=mkt_observer, 
            **trainInvest_env_para
        )
        print(f"   ✅ Environment created")
        print(f"   totalTradeDay: {env_train.totalTradeDay}")
        print()
        
        # Initialize TD3 model
        print("5. Initializing TD3 model...")
        model_para_dict = config.model_para
        model = TD3Controller(env=env_train, **model_para_dict)
        print(f"   ✅ TD3 model initialized")
        
        # Check initial state
        initial_buffer_size = model.replay_buffer.size() if hasattr(model.replay_buffer, 'size') else len(model.replay_buffer)
        initial_n_updates = getattr(model, '_n_updates', 0)
        learning_starts = getattr(model, 'learning_starts', 100)
        
        print(f"   Initial buffer size: {initial_buffer_size}")
        print(f"   Initial _n_updates: {initial_n_updates}")
        print(f"   learning_starts: {learning_starts}")
        print()
        
        # Run a few steps
        print("6. Running training steps...")
        print("   (This will collect data and trigger training if buffer >= learning_starts)")
        print()
        
        # Run for enough steps to trigger training
        num_steps_to_run = max(200, learning_starts + 50)  # Run enough to fill buffer and trigger training
        print(f"   Running {num_steps_to_run} steps...")
        
        # Wrap in Monitor for compatibility
        from stable_baselines3.common.monitor import Monitor
        env_train = Monitor(env_train)
        
        # Create a simple callback to track training
        from stable_baselines3.common.callbacks import BaseCallback
        
        class TrainingTracker(BaseCallback):
            def __init__(self):
                super().__init__()
                self.training_triggered = False
                self.buffer_sizes = []
                self.n_updates_list = []
            
            def _on_step(self) -> bool:
                if hasattr(self.model, 'replay_buffer'):
                    buffer_size = self.model.replay_buffer.size() if hasattr(self.model.replay_buffer, 'size') else len(self.model.replay_buffer)
                    n_updates = getattr(self.model, '_n_updates', 0)
                    self.buffer_sizes.append(buffer_size)
                    self.n_updates_list.append(n_updates)
                    
                    if n_updates > 0 and not self.training_triggered:
                        self.training_triggered = True
                        print(f"   ✅ Training triggered! _n_updates: {n_updates}, buffer_size: {buffer_size}")
                return True
        
        tracker = TrainingTracker()
        
        # Run training
        try:
            model.learn(
                total_timesteps=num_steps_to_run,
                callback=tracker,
                log_interval=10,
                reset_num_timesteps=True
            )
        except KeyboardInterrupt:
            print("   ⚠️  Training interrupted by user")
        except Exception as e:
            print(f"   ❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print()
        
        # Check final state
        print("7. Checking final state...")
        final_buffer_size = model.replay_buffer.size() if hasattr(model.replay_buffer, 'size') else len(model.replay_buffer)
        final_n_updates = getattr(model, '_n_updates', 0)
        
        print(f"   Final buffer size: {final_buffer_size}")
        print(f"   Final _n_updates: {final_n_updates}")
        print()
        
        # Results
        print("="*80)
        print("TEST RESULTS")
        print("="*80)
        
        if final_buffer_size >= learning_starts:
            print(f"✅ Buffer size ({final_buffer_size}) >= learning_starts ({learning_starts})")
        else:
            print(f"❌ Buffer size ({final_buffer_size}) < learning_starts ({learning_starts})")
            print(f"   Training should be disabled until buffer reaches {learning_starts}")
        
        if final_n_updates > 0:
            print(f"✅ Training occurred! _n_updates increased from {initial_n_updates} to {final_n_updates}")
            print(f"   Total gradient updates: {final_n_updates}")
            if tracker.training_triggered:
                print(f"✅ Training was triggered during the run")
            else:
                print(f"⚠️  Training occurred but tracker didn't catch it")
        else:
            print(f"❌ Training did NOT occur! _n_updates still {final_n_updates}")
            print(f"   Possible reasons:")
            print(f"   1. Buffer size ({final_buffer_size}) < learning_starts ({learning_starts})")
            print(f"   2. train_freq not configured correctly")
            print(f"   3. Training not triggered by stable-baselines3")
        
        if len(tracker.buffer_sizes) > 0:
            print(f"\nBuffer size progression: {tracker.buffer_sizes[0]} -> {tracker.buffer_sizes[-1]}")
            print(f"Max buffer size reached: {max(tracker.buffer_sizes)}")
        
        print()
        
        return final_n_updates > 0
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_td3_training()
    sys.exit(0 if success else 1)

