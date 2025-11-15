#!/usr/bin/env python3
"""
Script to verify if model parameters are being updated during training.

This script:
1. Loads checkpoints and compares parameters between epochs
2. Checks _n_updates counter in TD3 model
3. Checks optimizer state (step count, learning rate)
4. Compares parameters before/after training steps
"""

import os
import sys
import json
import glob
import torch as th
import numpy as np
from pathlib import Path
from datetime import datetime

def load_checkpoint_info(checkpoint_dir):
    """Load checkpoint info from JSON file."""
    info_path = os.path.join(checkpoint_dir, 'checkpoint_info.json')
    if not os.path.exists(info_path):
        return None
    
    try:
        with open(info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"   ❌ Error reading checkpoint info: {e}")
        return None

def get_checkpoint_parameters(checkpoint_path, device='cpu'):
    """Extract parameters from a checkpoint."""
    try:
        # Load the model
        from RL_controller.TD3_controller import TD3Controller
        from utils.tradeEnv import StockPortfolioEnv
        from config import Config
        import datetime
        
        # We need to create a dummy env to load the model
        # This is a simplified version - in practice you'd need the actual env
        config = Config(seed_num=123, current_date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        
        # Try to load model
        model = TD3Controller.load(checkpoint_path, device=device)
        
        # Extract parameters
        params = {}
        
        # Actor parameters
        if hasattr(model, 'actor') and model.actor is not None:
            actor_params = {}
            for name, param in model.actor.named_parameters():
                if param.requires_grad:
                    actor_params[name] = param.data.clone().cpu()
            params['actor'] = actor_params
        
        # Critic parameters
        if hasattr(model, 'critic') and model.critic is not None:
            critic_params = {}
            for name, param in model.critic.named_parameters():
                if param.requires_grad:
                    critic_params[name] = param.data.clone().cpu()
            params['critic'] = critic_params
        
        # Training state
        if hasattr(model, '_n_updates'):
            params['_n_updates'] = model._n_updates
        
        # Optimizer states
        if hasattr(model, 'actor') and hasattr(model.actor, 'optimizer'):
            params['actor_optimizer_step'] = model.actor.optimizer.state_dict().get('state', {})
        
        if hasattr(model, 'critic') and hasattr(model.critic, 'optimizer'):
            params['critic_optimizer_step'] = model.critic.optimizer.state_dict().get('state', {})
        
        return params, model
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_parameters(params1, params2, name1="Checkpoint 1", name2="Checkpoint 2"):
    """Compare parameters between two checkpoints."""
    print(f"\n   Comparing {name1} vs {name2}:")
    
    differences = []
    
    # Compare _n_updates
    if '_n_updates' in params1 and '_n_updates' in params2:
        n1 = params1['_n_updates']
        n2 = params2['_n_updates']
        if n1 != n2:
            print(f"      ✅ _n_updates changed: {n1} -> {n2} (diff: {n2 - n1})")
        else:
            print(f"      ⚠️  _n_updates unchanged: {n1}")
    
    # Compare actor parameters
    if 'actor' in params1 and 'actor' in params2:
        actor_diff = compare_param_dict(params1['actor'], params2['actor'], 'actor')
        if actor_diff:
            differences.extend(actor_diff)
            print(f"      ✅ Actor parameters changed")
        else:
            print(f"      ⚠️  Actor parameters unchanged")
    
    # Compare critic parameters
    if 'critic' in params1 and 'critic' in params2:
        critic_diff = compare_param_dict(params1['critic'], params2['critic'], 'critic')
        if critic_diff:
            differences.extend(critic_diff)
            print(f"      ✅ Critic parameters changed")
        else:
            print(f"      ⚠️  Critic parameters unchanged")
    
    return differences

def compare_param_dict(dict1, dict2, prefix=""):
    """Compare two parameter dictionaries."""
    differences = []
    
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        if key not in dict1:
            differences.append(f"{prefix}.{key}: missing in first")
            continue
        if key not in dict2:
            differences.append(f"{prefix}.{key}: missing in second")
            continue
        
        p1 = dict1[key]
        p2 = dict2[key]
        
        if p1.shape != p2.shape:
            differences.append(f"{prefix}.{key}: shape mismatch {p1.shape} vs {p2.shape}")
            continue
        
        # Compute difference
        diff = th.abs(p1 - p2).max().item()
        if diff > 1e-6:
            differences.append(f"{prefix}.{key}: max diff = {diff:.2e}")
        elif diff > 0:
            differences.append(f"{prefix}.{key}: tiny diff = {diff:.2e}")
    
    return differences

def check_checkpoints_for_updates(res_dir):
    """Check checkpoints to see if parameters are updating."""
    print("="*80)
    print("CHECKING CHECKPOINTS FOR PARAMETER UPDATES")
    print("="*80)
    
    if res_dir is None:
        print("❌ res_dir is None")
        return
    
    checkpoint_dir = os.path.join(res_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        print()
        return
    
    # Find all checkpoints
    checkpoint_dirs = []
    for item in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(checkpoint_path) and item.startswith('checkpoint_'):
            info = load_checkpoint_info(checkpoint_path)
            if info:
                checkpoint_dirs.append({
                    'path': checkpoint_path,
                    'name': item,
                    'info': info,
                    'epoch': info.get('epoch', 0),
                    'timesteps': info.get('timesteps', 0)
                })
    
    if len(checkpoint_dirs) < 2:
        print(f"⚠️  Need at least 2 checkpoints to compare, found {len(checkpoint_dirs)}")
        print()
        return
    
    # Sort by timesteps
    checkpoint_dirs.sort(key=lambda x: x['timesteps'])
    
    print(f"✅ Found {len(checkpoint_dirs)} checkpoint(s)")
    print(f"   Comparing first and last checkpoints...\n")
    
    # Load first and last checkpoints
    first_cp = checkpoint_dirs[0]
    last_cp = checkpoint_dirs[-1]
    
    print(f"Loading checkpoint 1: {first_cp['name']} (Epoch {first_cp['epoch']}, Timesteps {first_cp['timesteps']})")
    params1, model1 = get_checkpoint_parameters(
        os.path.join(first_cp['path'], 'rl_model.zip')
    )
    
    print(f"Loading checkpoint 2: {last_cp['name']} (Epoch {last_cp['epoch']}, Timesteps {last_cp['timesteps']})")
    params2, model2 = get_checkpoint_parameters(
        os.path.join(last_cp['path'], 'rl_model.zip')
    )
    
    if params1 is None or params2 is None:
        print("❌ Failed to load checkpoint parameters")
        print()
        return
    
    # Compare parameters
    differences = compare_parameters(
        params1, params2,
        name1=f"Epoch {first_cp['epoch']}",
        name2=f"Epoch {last_cp['epoch']}"
    )
    
    if differences:
        print(f"\n   ✅ Parameters ARE updating (found {len(differences)} differences)")
    else:
        print(f"\n   ❌ Parameters are NOT updating!")
    
    print()

def check_n_updates_from_logs(res_dir):
    """Check _n_updates from log files."""
    print("="*80)
    print("CHECKING _n_updates FROM LOGS")
    print("="*80)
    
    log_files = []
    if os.path.exists('nohup.out'):
        log_files.append('nohup.out')
    
    if res_dir:
        log_pattern = os.path.join(res_dir, "*.log")
        log_files.extend(glob.glob(log_pattern))
    
    n_updates_found = []
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            continue
        
        print(f"Checking: {log_file}")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if '_n_updates' in line or 'Network Updates' in line or '[TRAIN]' in line:
                        n_updates_found.append(line.strip())
        except Exception as e:
            print(f"   ❌ Error reading {log_file}: {e}")
    
    if n_updates_found:
        print(f"\n   Found {len(n_updates_found)} relevant log lines:")
        for line in n_updates_found[-10:]:  # Show last 10
            print(f"      {line[:100]}")
        
        # Try to extract _n_updates values
        updates_values = []
        for line in n_updates_found:
            if 'Updates:' in line:
                try:
                    # Try to extract number after "Updates:"
                    parts = line.split('Updates:')
                    if len(parts) > 1:
                        num_str = parts[1].split()[0]
                        updates_values.append(int(num_str))
                except:
                    pass
        
        if updates_values:
            print(f"\n   _n_updates progression: {updates_values[-10:]}")
            if len(updates_values) > 1 and updates_values[-1] > updates_values[0]:
                print(f"   ✅ _n_updates is increasing: {updates_values[0]} -> {updates_values[-1]}")
            else:
                print(f"   ⚠️  _n_updates may not be increasing")
    else:
        print("   ⚠️  No _n_updates information found in logs")
    
    print()

def check_mafia_observer_updates(res_dir):
    """Check MAFIA observer checkpoint for updates."""
    print("="*80)
    print("CHECKING MAFIA OBSERVER UPDATES")
    print("="*80)
    
    if res_dir is None:
        print("❌ res_dir is None")
        print()
        return
    
    checkpoint_dir = os.path.join(res_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        print()
        return
    
    # Find checkpoints with MAFIA observer files
    mafia_checkpoints = []
    for item in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(checkpoint_path):
            mafia_path = os.path.join(checkpoint_path, 'mafia_observer.pth')
            if os.path.exists(mafia_path):
                info = load_checkpoint_info(checkpoint_path)
                if info:
                    mafia_checkpoints.append({
                        'path': mafia_path,
                        'name': item,
                        'epoch': info.get('epoch', 0),
                        'timesteps': info.get('timesteps', 0)
                    })
    
    if len(mafia_checkpoints) < 2:
        print(f"⚠️  Need at least 2 MAFIA checkpoints to compare, found {len(mafia_checkpoints)}")
        print()
        return
    
    mafia_checkpoints.sort(key=lambda x: x['timesteps'])
    
    print(f"✅ Found {len(mafia_checkpoints)} MAFIA checkpoint(s)")
    print(f"   Comparing first and last...\n")
    
    # Load first and last
    first_path = mafia_checkpoints[0]['path']
    last_path = mafia_checkpoints[-1]['path']
    
    try:
        first_state = th.load(first_path, map_location='cpu')
        last_state = th.load(last_path, map_location='cpu')
        
        # Compare model state
        if 'model_state_dict' in first_state and 'model_state_dict' in last_state:
            first_params = first_state['model_state_dict']
            last_params = last_state['model_state_dict']
            
            differences = []
            for key in first_params.keys():
                if key in last_params:
                    p1 = first_params[key]
                    p2 = last_params[key]
                    if p1.shape == p2.shape:
                        diff = th.abs(p1 - p2).max().item()
                        if diff > 1e-6:
                            differences.append(f"{key}: max diff = {diff:.2e}")
            
            if differences:
                print(f"   ✅ MAFIA parameters ARE updating (found {len(differences)} differences)")
                for diff in differences[:5]:  # Show first 5
                    print(f"      {diff}")
            else:
                print(f"   ⚠️  MAFIA parameters appear unchanged")
            
            # Check optimizer step
            if 'optimizer_state_dict' in first_state and 'optimizer_state_dict' in last_state:
                first_opt = first_state['optimizer_state_dict']
                last_opt = last_state['optimizer_state_dict']
                
                first_step = first_opt.get('state', {}).get(0, {}).get('step', 0) if isinstance(first_opt.get('state'), dict) else 0
                last_step = last_opt.get('state', {}).get(0, {}).get('step', 0) if isinstance(last_opt.get('state'), dict) else 0
                
                if last_step > first_step:
                    print(f"   ✅ Optimizer step increased: {first_step} -> {last_step}")
                else:
                    print(f"   ⚠️  Optimizer step unchanged: {first_step}")
        
    except Exception as e:
        print(f"   ❌ Error comparing MAFIA checkpoints: {e}")
        import traceback
        traceback.print_exc()
    
    print()

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

def main():
    print("\n" + "="*80)
    print("MAFIA PARAMETER UPDATE VERIFICATION SCRIPT")
    print("="*80 + "\n")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Find latest res directory
    latest_res_dir = find_latest_res_dir()
    if latest_res_dir:
        print(f"Using res directory: {latest_res_dir}\n")
    else:
        print("⚠️  Could not find latest res directory\n")
    
    # Run checks
    check_n_updates_from_logs(latest_res_dir)
    check_checkpoints_for_updates(latest_res_dir)
    check_mafia_observer_updates(latest_res_dir)
    
    # Summary
    print("="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print("\nTo verify parameter updates:")
    print("   1. Check _n_updates counter in logs (should increase)")
    print("   2. Compare parameters between checkpoints (should differ)")
    print("   3. Check optimizer step counts (should increase)")
    print("   4. Monitor training logs for [TRAIN] messages with loss values")
    print()

if __name__ == '__main__':
    main()

