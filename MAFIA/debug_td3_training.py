#!/usr/bin/env python3
"""
Script to debug why TD3 model is not updating during training.

This script checks:
1. Replay buffer size and learning_starts threshold
2. train_freq configuration
3. Whether TD3.train() is being called
4. Training logs and messages
5. Parameter changes between checkpoints
"""

import os
import sys
import json
import glob
import torch as th
import numpy as np
from pathlib import Path
from datetime import datetime

def check_training_config():
    """Check TD3 training configuration."""
    print("="*80)
    print("CHECKING TD3 TRAINING CONFIGURATION")
    print("="*80)
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from config import Config
        import datetime as dt
        
        config = Config(seed_num=123, current_date=dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        
        print(f"✅ Config loaded successfully")
        print(f"\nTraining Configuration:")
        print(f"   learning_starts: {getattr(config, 'learning_starts', 100)}")
        print(f"   train_freq: {getattr(config, 'train_freq', (1, 'episode'))}")
        print(f"   gradient_steps: {config.gradient_steps}")
        print(f"   batch_size: {config.batch_size}")
        print(f"   buffer_size: {getattr(config, 'buffer_size', 1000000)}")
        
        # Check if train_freq is episode-based
        train_freq = getattr(config, 'train_freq', (1, 'episode'))
        if isinstance(train_freq, (tuple, list)) and len(train_freq) == 2:
            freq, unit = train_freq
            print(f"\n   train_freq: {freq} {unit}")
            if unit == 'episode':
                print(f"   ✅ Training should occur after each episode")
            else:
                print(f"   ⚠️  Training occurs every {freq} {unit}, not after each episode")
        else:
            print(f"   ⚠️  train_freq format unexpected: {train_freq}")
            print(f"   Expected: (frequency, 'episode') or [frequency, 'episode']")
        
        print()
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_logs_for_training_messages(res_dir=None):
    """Check logs for TD3 training messages."""
    print("="*80)
    print("CHECKING LOGS FOR TD3 TRAINING MESSAGES")
    print("="*80)
    
    log_files = []
    if os.path.exists('nohup.out'):
        log_files.append('nohup.out')
    
    if res_dir:
        log_pattern = os.path.join(res_dir, "*.log")
        log_files.extend(glob.glob(log_pattern))
    
    train_messages = []
    rollout_messages = []
    buffer_messages = []
    error_messages = []
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            continue
        
        print(f"Checking: {log_file}")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    # Check for training messages
                    if '[TRAIN]' in line or 'Starting gradient updates' in line or 'Completed | Updates:' in line:
                        train_messages.append((i+1, line.strip()))
                    
                    # Check for rollout messages
                    if '[ROLLOUT]' in line or 'ROLLOUT COMPLETE' in line:
                        rollout_messages.append((i+1, line.strip()))
                    
                    # Check for buffer size messages
                    if 'buffer_size' in line.lower() or 'replay buffer' in line.lower():
                        buffer_messages.append((i+1, line.strip()))
                    
                    # Check for errors
                    if '[ERROR]' in line or 'Exception' in line or 'Traceback' in line:
                        error_messages.append((i+1, line.strip()))
        except Exception as e:
            print(f"   ❌ Error reading {log_file}: {e}")
    
    print(f"\nTraining Messages Found: {len(train_messages)}")
    if train_messages:
        print("   Recent training messages:")
        for line_num, msg in train_messages[-10:]:
            print(f"      Line {line_num}: {msg[:100]}")
    else:
        print("   ❌ NO TRAINING MESSAGES FOUND!")
        print("   This indicates TD3.train() may not be called")
    
    print(f"\nRollout Messages Found: {len(rollout_messages)}")
    if rollout_messages:
        print("   Recent rollout messages:")
        for line_num, msg in rollout_messages[-5:]:
            print(f"      Line {line_num}: {msg[:100]}")
    
    print(f"\nBuffer Messages Found: {len(buffer_messages)}")
    if buffer_messages:
        print("   Recent buffer messages:")
        for line_num, msg in buffer_messages[-5:]:
            print(f"      Line {line_num}: {msg[:100]}")
    
    print(f"\nError Messages Found: {len(error_messages)}")
    if error_messages:
        print("   Recent errors (first 10):")
        for line_num, msg in error_messages[:10]:
            print(f"      Line {line_num}: {msg[:150]}")
    
    print()
    return train_messages, rollout_messages, error_messages

def check_replay_buffer_from_checkpoint(checkpoint_path):
    """Check replay buffer state from checkpoint."""
    print("="*80)
    print("CHECKING REPLAY BUFFER FROM CHECKPOINT")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print()
        return None
    
    try:
        from RL_controller.TD3_controller import TD3Controller
        from utils.tradeEnv import StockPortfolioEnv
        from config import Config
        import datetime as dt
        
        # Create dummy env for loading
        config = Config(seed_num=123, current_date=dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        
        # Try to load model
        model = TD3Controller.load(checkpoint_path, device='cpu')
        
        print(f"✅ Model loaded from checkpoint")
        
        # Check replay buffer
        if hasattr(model, 'replay_buffer'):
            buffer = model.replay_buffer
            buffer_size = buffer.size() if hasattr(buffer, 'size') else len(buffer)
            max_size = getattr(buffer, 'max_size', None)
            
            print(f"\nReplay Buffer State:")
            print(f"   Current size: {buffer_size}")
            print(f"   Max size: {max_size}")
            
            # Check learning_starts
            learning_starts = getattr(model, 'learning_starts', 100)
            print(f"   learning_starts threshold: {learning_starts}")
            
            if buffer_size >= learning_starts:
                print(f"   ✅ Buffer size ({buffer_size}) >= learning_starts ({learning_starts})")
                print(f"   ✅ Training should be enabled")
            else:
                print(f"   ❌ Buffer size ({buffer_size}) < learning_starts ({learning_starts})")
                print(f"   ❌ Training is DISABLED until buffer reaches {learning_starts}")
            
            # Check train_freq
            train_freq = getattr(model, 'train_freq', None)
            print(f"\nTraining Frequency:")
            print(f"   train_freq: {train_freq}")
            
            # Check _n_updates
            n_updates = getattr(model, '_n_updates', None)
            print(f"\nTraining Updates:")
            print(f"   _n_updates: {n_updates}")
            
            if n_updates == 0:
                print(f"   ❌ Model has NOT been updated!")
            else:
                print(f"   ✅ Model has been updated {n_updates} times")
            
            return {
                'buffer_size': buffer_size,
                'max_size': max_size,
                'learning_starts': learning_starts,
                'train_freq': train_freq,
                'n_updates': n_updates
            }
        else:
            print(f"   ❌ Model has no replay_buffer attribute")
            return None
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_latest_checkpoint(res_dir):
    """Find the latest checkpoint."""
    if res_dir is None:
        return None
    
    checkpoint_dir = os.path.join(res_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(checkpoint_path):
            info_path = os.path.join(checkpoint_path, 'checkpoint_info.json')
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    checkpoints.append({
                        'path': os.path.join(checkpoint_path, 'rl_model.zip'),
                        'name': item,
                        'timesteps': info.get('timesteps', 0),
                        'epoch': info.get('epoch', 0)
                    })
                except:
                    pass
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x['timesteps'], reverse=True)
        return checkpoints[0]['path']
    return None

def find_latest_res_dir():
    """Find the latest res directory."""
    res_base = "res/RLcontroller/TD3"
    if not os.path.exists(res_base):
        return None
    
    timestamp_dirs = []
    for market_dir in os.listdir(res_base):
        market_path = os.path.join(res_base, market_dir)
        if not os.path.isdir(market_path):
            continue
        
        for timestamp_dir in os.listdir(market_path):
            timestamp_path = os.path.join(market_path, timestamp_dir)
            if os.path.isdir(timestamp_path):
                try:
                    datetime.strptime(timestamp_dir, '%Y-%m-%d-%H-%M-%S')
                    mtime = os.path.getmtime(timestamp_path)
                    timestamp_dirs.append({
                        'path': timestamp_path,
                        'mtime': mtime
                    })
                except ValueError:
                    pass
    
    if not timestamp_dirs:
        return None
    
    timestamp_dirs.sort(key=lambda x: x['mtime'], reverse=True)
    return timestamp_dirs[0]['path']

def check_parameter_changes(checkpoint1_path, checkpoint2_path):
    """Compare parameters between two checkpoints."""
    print("="*80)
    print("COMPARING PARAMETERS BETWEEN CHECKPOINTS")
    print("="*80)
    
    if not os.path.exists(checkpoint1_path) or not os.path.exists(checkpoint2_path):
        print("❌ One or both checkpoints not found")
        print()
        return
    
    try:
        from RL_controller.TD3_controller import TD3Controller
        
        print(f"Loading checkpoint 1: {os.path.basename(checkpoint1_path)}")
        model1 = TD3Controller.load(checkpoint1_path, device='cpu')
        
        print(f"Loading checkpoint 2: {os.path.basename(checkpoint2_path)}")
        model2 = TD3Controller.load(checkpoint2_path, device='cpu')
        
        # Compare _n_updates
        n1 = getattr(model1, '_n_updates', 0)
        n2 = getattr(model2, '_n_updates', 0)
        print(f"\n_n_updates: {n1} -> {n2} (diff: {n2 - n1})")
        
        if n2 > n1:
            print("   ✅ _n_updates increased - training occurred")
        elif n2 == n1:
            print("   ❌ _n_updates unchanged - NO training occurred")
        else:
            print("   ⚠️  _n_updates decreased - unexpected!")
        
        # Compare actor parameters
        if hasattr(model1, 'actor') and hasattr(model2, 'actor'):
            actor1_params = {name: param.data.clone() for name, param in model1.actor.named_parameters()}
            actor2_params = {name: param.data.clone() for name, param in model2.actor.named_parameters()}
            
            actor_changed = False
            max_diff = 0.0
            for key in actor1_params.keys():
                if key in actor2_params:
                    diff = th.abs(actor1_params[key] - actor2_params[key]).max().item()
                    if diff > 1e-6:
                        actor_changed = True
                        max_diff = max(max_diff, diff)
            
            if actor_changed:
                print(f"   ✅ Actor parameters changed (max diff: {max_diff:.2e})")
            else:
                print(f"   ❌ Actor parameters unchanged")
        
        # Compare critic parameters
        if hasattr(model1, 'critic') and hasattr(model2, 'critic'):
            critic1_params = {name: param.data.clone() for name, param in model1.critic.named_parameters()}
            critic2_params = {name: param.data.clone() for name, param in model2.critic.named_parameters()}
            
            critic_changed = False
            max_diff = 0.0
            for key in critic1_params.keys():
                if key in critic2_params:
                    diff = th.abs(critic1_params[key] - critic2_params[key]).max().item()
                    if diff > 1e-6:
                        critic_changed = True
                        max_diff = max(max_diff, diff)
            
            if critic_changed:
                print(f"   ✅ Critic parameters changed (max diff: {max_diff:.2e})")
            else:
                print(f"   ❌ Critic parameters unchanged")
        
        print()
        
    except Exception as e:
        print(f"❌ Error comparing checkpoints: {e}")
        import traceback
        traceback.print_exc()
        print()

def main():
    print("\n" + "="*80)
    print("TD3 TRAINING DEBUG SCRIPT")
    print("="*80 + "\n")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Check configuration
    config = check_training_config()
    
    # Find latest res directory
    latest_res_dir = find_latest_res_dir()
    if latest_res_dir:
        print(f"Using res directory: {latest_res_dir}\n")
    else:
        print("⚠️  Could not find latest res directory\n")
    
    # Check logs
    train_msgs, rollout_msgs, error_msgs = check_logs_for_training_messages(latest_res_dir)
    
    # Check latest checkpoint
    buffer_info = None
    if latest_res_dir:
        latest_checkpoint = find_latest_checkpoint(latest_res_dir)
        if latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}\n")
            buffer_info = check_replay_buffer_from_checkpoint(latest_checkpoint)
            
            # Find another checkpoint for comparison
            checkpoint_dir = os.path.join(latest_res_dir, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                checkpoints = []
                for item in os.listdir(checkpoint_dir):
                    checkpoint_path = os.path.join(checkpoint_dir, item, 'rl_model.zip')
                    if os.path.exists(checkpoint_path):
                        info_path = os.path.join(checkpoint_dir, item, 'checkpoint_info.json')
                        if os.path.exists(info_path):
                            try:
                                with open(info_path, 'r') as f:
                                    info = json.load(f)
                                checkpoints.append({
                                    'path': checkpoint_path,
                                    'timesteps': info.get('timesteps', 0)
                                })
                            except:
                                pass
                
                if len(checkpoints) >= 2:
                    checkpoints.sort(key=lambda x: x['timesteps'])
                    check_parameter_changes(checkpoints[0]['path'], checkpoints[-1]['path'])
    
    # Summary
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if not train_msgs:
        print("\n❌ CRITICAL: No TD3 training messages found in logs!")
        print("   This indicates TD3.train() is NOT being called.")
        print("\nPossible causes:")
        print("   1. Replay buffer size < learning_starts threshold")
        print("   2. train_freq not configured correctly")
        print("   3. Training is disabled or blocked")
        print("   4. Exception during training (check error messages)")
    else:
        print(f"\n✅ Found {len(train_msgs)} training messages - TD3.train() is being called")
    
    if buffer_info:
        if buffer_info['buffer_size'] < buffer_info['learning_starts']:
            print(f"\n❌ CRITICAL: Buffer size ({buffer_info['buffer_size']}) < learning_starts ({buffer_info['learning_starts']})")
            print("   Training is DISABLED until buffer reaches threshold!")
        else:
            print(f"\n✅ Buffer size ({buffer_info['buffer_size']}) >= learning_starts ({buffer_info['learning_starts']})")
    
    if error_msgs:
        print(f"\n⚠️  Found {len(error_msgs)} error messages - check logs for details")
    
    print()

if __name__ == '__main__':
    main()

