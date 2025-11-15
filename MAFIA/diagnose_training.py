#!/usr/bin/env python3
"""
Diagnostic script to check why train_profile.csv is not being created.

This script checks:
1. If res_dir exists and is writable
2. If terminal condition is being triggered
3. If save_profile is being called
4. If there are any errors in the save process
"""

import os
import sys
import glob
import json
from datetime import datetime
from pathlib import Path

def check_profile_files():
    """Check if train_profile.csv exists anywhere in res directory."""
    print("="*80)
    print("CHECKING FOR train_profile.csv FILES")
    print("="*80)
    
    res_dir = "res"
    if not os.path.exists(res_dir):
        print(f"❌ res directory not found at {os.path.abspath(res_dir)}")
        return []
    
    # Find all profile CSV files
    pattern = os.path.join(res_dir, "**", "*profile*.csv")
    profile_files = glob.glob(pattern, recursive=True)
    
    if profile_files:
        print(f"✅ Found {len(profile_files)} profile file(s):")
        for f in sorted(profile_files):
            size = os.path.getsize(f)
            mtime = os.path.getmtime(f)
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   - {f}")
            print(f"     Size: {size} bytes, Modified: {mtime_str}")
    else:
        print("❌ No profile CSV files found!")
        print(f"   Searched in: {os.path.abspath(res_dir)}")
    
    print()
    return profile_files

def check_latest_res_dir():
    """Find the latest res directory based on timestamp."""
    print("="*80)
    print("CHECKING LATEST RES DIRECTORY")
    print("="*80)
    
    res_base = "res/RLcontroller/TD3"
    if not os.path.exists(res_base):
        print(f"❌ Base res directory not found: {os.path.abspath(res_base)}")
        return None
    
    # Find all timestamp directories
    timestamp_dirs = []
    for market_dir in os.listdir(res_base):
        market_path = os.path.join(res_base, market_dir)
        if not os.path.isdir(market_path):
            continue
        
        for timestamp_dir in os.listdir(market_path):
            timestamp_path = os.path.join(market_path, timestamp_dir)
            if os.path.isdir(timestamp_path):
                try:
                    # Try to parse as timestamp
                    datetime.strptime(timestamp_dir, '%Y-%m-%d-%H-%M-%S')
                    mtime = os.path.getmtime(timestamp_path)
                    timestamp_dirs.append({
                        'path': timestamp_path,
                        'timestamp': timestamp_dir,
                        'mtime': mtime
                    })
                except ValueError:
                    # Not a timestamp directory, skip
                    pass
    
    if not timestamp_dirs:
        print("❌ No timestamp directories found!")
        return None
    
    # Sort by modification time (newest first)
    timestamp_dirs.sort(key=lambda x: x['mtime'], reverse=True)
    
    latest = timestamp_dirs[0]
    print(f"✅ Latest res directory: {latest['path']}")
    print(f"   Timestamp: {latest['timestamp']}")
    print(f"   Modified: {datetime.fromtimestamp(latest['mtime']).strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    return latest['path']

def check_res_dir_permissions(res_dir):
    """Check if res_dir exists and is writable."""
    print("="*80)
    print("CHECKING RES_DIR PERMISSIONS")
    print("="*80)
    
    if res_dir is None:
        print("❌ res_dir is None")
        return False
    
    if not os.path.exists(res_dir):
        print(f"❌ res_dir does not exist: {os.path.abspath(res_dir)}")
        print(f"   Attempting to create...")
        try:
            os.makedirs(res_dir, exist_ok=True)
            print(f"   ✅ Created successfully")
        except Exception as e:
            print(f"   ❌ Failed to create: {e}")
            return False
    else:
        print(f"✅ res_dir exists: {os.path.abspath(res_dir)}")
    
    # Check write permissions
    test_file = os.path.join(res_dir, '.write_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"✅ res_dir is writable")
    except Exception as e:
        print(f"❌ res_dir is NOT writable: {e}")
        return False
    
    print()
    return True

def check_logs_for_epoch_end(res_dir):
    """Check log files for epoch end messages."""
    print("="*80)
    print("CHECKING LOGS FOR EPOCH END MESSAGES")
    print("="*80)
    
    if res_dir is None:
        print("❌ res_dir is None, cannot check logs")
        return
    
    # Check for nohup.out in current directory
    log_files = ['nohup.out']
    if os.path.exists('nohup.out'):
        log_files.append('nohup.out')
    
    # Also check for any .log files in res_dir
    log_pattern = os.path.join(res_dir, "*.log")
    log_files.extend(glob.glob(log_pattern))
    
    epoch_end_count = 0
    profile_save_count = 0
    terminal_count = 0
    error_count = 0
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            continue
        
        print(f"Checking: {log_file}")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if '[EPOCH END]' in line:
                        epoch_end_count += 1
                        if epoch_end_count <= 3:  # Show first 3
                            print(f"   Line {i+1}: [EPOCH END] found")
                    if '[Profile Save]' in line:
                        profile_save_count += 1
                        if profile_save_count <= 3:  # Show first 3
                            print(f"   Line {i+1}: [Profile Save] found")
                    if '[TERMINAL]' in line:
                        terminal_count += 1
                        if terminal_count <= 3:  # Show first 3
                            print(f"   Line {i+1}: [TERMINAL] found")
                    if '[ERROR]' in line or 'Exception' in line:
                        error_count += 1
                        if error_count <= 5:  # Show first 5 errors
                            print(f"   Line {i+1}: ERROR - {line.strip()[:100]}")
        except Exception as e:
            print(f"   ❌ Error reading {log_file}: {e}")
    
    print(f"\nSummary:")
    print(f"   [EPOCH END] messages: {epoch_end_count}")
    print(f"   [Profile Save] messages: {profile_save_count}")
    print(f"   [TERMINAL] messages: {terminal_count}")
    print(f"   ERROR messages: {error_count}")
    
    if epoch_end_count == 0:
        print("   ⚠️  WARNING: No [EPOCH END] messages found - epochs may not be completing!")
    if profile_save_count == 0:
        print("   ⚠️  WARNING: No [Profile Save] messages found - save_profile() may not be called!")
    if terminal_count == 0:
        print("   ⚠️  WARNING: No [TERMINAL] messages found - terminal condition may not be triggered!")
    
    print()

def check_config():
    """Check training configuration."""
    print("="*80)
    print("CHECKING TRAINING CONFIGURATION")
    print("="*80)
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from config import Config
        import datetime
        
        config = Config(seed_num=123, current_date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        
        print(f"✅ Config loaded successfully")
        print(f"   num_epochs: {config.num_epochs}")
        print(f"   train_freq: {config.train_freq}")
        print(f"   learning_starts: {getattr(config, 'learning_starts', 100)}")
        print(f"   gradient_steps: {config.gradient_steps}")
        print(f"   batch_size: {config.batch_size}")
        print(f"   res_dir: {config.res_dir}")
        print(f"   mode: {config.mode}")
        
        if hasattr(config, 'res_dir') and os.path.exists(config.res_dir):
            print(f"   ✅ res_dir exists: {config.res_dir}")
            
            # Check for train_profile.csv
            train_profile_path = os.path.join(config.res_dir, 'train_profile.csv')
            if os.path.exists(train_profile_path):
                size = os.path.getsize(train_profile_path)
                mtime = os.path.getmtime(train_profile_path)
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"   ✅ train_profile.csv exists: {size} bytes, modified: {mtime_str}")
            else:
                print(f"   ❌ train_profile.csv does NOT exist in res_dir")
        else:
            print(f"   ❌ res_dir does not exist: {getattr(config, 'res_dir', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def check_checkpoints(res_dir):
    """Check checkpoint files for training progress."""
    print("="*80)
    print("CHECKING CHECKPOINTS FOR TRAINING PROGRESS")
    print("="*80)
    
    if res_dir is None:
        print("❌ res_dir is None, cannot check checkpoints")
        return
    
    checkpoint_dir = os.path.join(res_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        print()
        return
    
    checkpoint_dirs = []
    for item in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(checkpoint_path) and item.startswith('checkpoint_'):
            info_path = os.path.join(checkpoint_path, 'checkpoint_info.json')
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    checkpoint_dirs.append({
                        'name': item,
                        'epoch': info.get('epoch', 0),
                        'timesteps': info.get('timesteps', 0),
                        'timestamp': info.get('timestamp', '')
                    })
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not read {info_path}: {e}")
    
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: x['timesteps'], reverse=True)
        print(f"✅ Found {len(checkpoint_dirs)} checkpoint(s):")
        for cp in checkpoint_dirs[:5]:  # Show top 5
            print(f"   - {cp['name']}: Epoch {cp['epoch']}, Timesteps {cp['timesteps']}, {cp['timestamp']}")
        
        latest = checkpoint_dirs[0]
        print(f"\n   Latest checkpoint: Epoch {latest['epoch']}, Timesteps {latest['timesteps']}")
    else:
        print("❌ No checkpoints found!")
    
    print()

def main():
    print("\n" + "="*80)
    print("MAFIA TRAINING DIAGNOSTIC SCRIPT")
    print("="*80 + "\n")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Run diagnostics
    profile_files = check_profile_files()
    latest_res_dir = check_latest_res_dir()
    
    if latest_res_dir:
        check_res_dir_permissions(latest_res_dir)
        check_logs_for_epoch_end(latest_res_dir)
        check_checkpoints(latest_res_dir)
    
    check_config()
    
    # Summary
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if profile_files:
        print(f"✅ Found {len(profile_files)} profile file(s)")
    else:
        print("❌ No profile files found")
        print("\nPossible issues:")
        print("   1. Terminal condition may not be triggered")
        print("   2. save_profile() may not be called")
        print("   3. res_dir may not be set correctly")
        print("   4. File write permissions may be missing")
        print("   5. Exception may be occurring during save")
        print("\nRecommendations:")
        print("   1. Check logs for [EPOCH END] and [Profile Save] messages")
        print("   2. Verify res_dir exists and is writable")
        print("   3. Check for exceptions in logs")
        print("   4. Run training with verbose logging enabled")
    
    print()

if __name__ == '__main__':
    main()

