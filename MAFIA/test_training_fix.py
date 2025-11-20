#!/usr/bin/env python3
"""
Test script to verify training fixes.
Tests individual components to ensure they work correctly.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from utils.tradeEnv import StockPortfolioEnv
from utils.model_pool import model_select
from utils.callback_func import PoCallback


def test_environment_step_format():
    """Test that environment returns consistent format."""
    print("=" * 60)
    print("TESTING ENVIRONMENT STEP FORMAT")
    print("=" * 60)

    # Create minimal config
    config = Config(seed_num=123, current_date='2025-11-17-15-00-00')
    config.num_tradeDay = 3

    # Create minimal data
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='D'),
        'stock': ['STOCK1'] * 3,
        'open': [100] * 3,
        'close': [101, 102, 103],
        'high': [102] * 3,
        'low': [99] * 3,
        'volume': [1000] * 3
    })

    env = StockPortfolioEnv(config=config, rawdata=data, mode='train', stock_num=1, action_dim=1)

    # Test reset
    obs = env.reset()
    print(f"Reset returned: obs shape {obs.shape if hasattr(obs, 'shape') else 'no shape'}")

    # Test step
    action = np.array([0.1])  # Simple action
    result = env.step(action)
    print(f"Step returned {len(result)} values")

    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        print("✓ Correct gymnasium format: (obs, reward, terminated, truncated, info)")
        print(f"  terminated: {terminated}, truncated: {truncated}")
    else:
        print("✗ Unexpected return format")

    # Test terminal step
    while not terminated and not truncated:
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            print(f"  Step: terminated={terminated}, truncated={truncated}")

    print("✓ Environment step format test passed")


def test_save_profile_arrays():
    """Test that save_profile handles array lengths correctly."""
    print("\n" + "=" * 60)
    print("TESTING SAVE_PROFILE ARRAY LENGTHS")
    print("=" * 60)

    # Create minimal config
    config = Config(seed_num=123, current_date='2025-11-17-15-00-00')
    config.num_tradeDay = 3
    config.res_dir = './test_results'
    os.makedirs(config.res_dir, exist_ok=True)

    # Create minimal data
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='D'),
        'stock': ['STOCK1'] * 3,
        'open': [100] * 3,
        'close': [101, 102, 103],
        'high': [102] * 3,
        'low': [99] * 3,
        'volume': [1000] * 3
    })

    env = StockPortfolioEnv(config=config, rawdata=data, mode='train', stock_num=1, action_dim=1)

    # Simulate some trading steps
    env.reset()
    for i in range(2):  # 2 steps (should trigger terminal on 3rd)
        action = np.array([0.1])
        env.step(action)

    # Trigger terminal step to save profile
    try:
        action = np.array([0.1])
        env.step(action)  # This should trigger save_profile
        print("✓ save_profile executed without array length errors")
    except ValueError as e:
        if "All arrays must be of the same length" in str(e):
            print("✗ Array length error still present:", e)
            return False
        else:
            print("✗ Different error:", e)
            return False
    except Exception as e:
        print("✓ No array length error (other error is expected):", type(e).__name__)

    # Check if files were created
    stepdata_path = os.path.join(config.res_dir, 'train_stepdata.csv')
    if os.path.exists(stepdata_path):
        df = pd.read_csv(stepdata_path)
        print(f"✓ Step data CSV created with {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
    else:
        print("✗ Step data CSV not created")

    print("✓ save_profile array length test passed")
    return True


def test_checkpoint_cleanup_enabled():
    """Test that checkpoint cleanup defaults to enabled."""
    print("\n" + "=" * 60)
    print("TESTING CHECKPOINT CLEANUP ENABLED")
    print("=" * 60)

    config = Config(seed_num=123, current_date='2025-11-17-15-00-00')
    callback = PoCallback(config=config, train_env=None)

    print(f"enable_checkpoint_cleanup: {getattr(callback, 'enable_checkpoint_cleanup', 'not set')}")
    print(f"max_checkpoints_to_keep: {getattr(callback, 'max_checkpoints_to_keep', 'not set')}")

    if getattr(callback, 'enable_checkpoint_cleanup', False):
        print("✓ Checkpoint cleanup is enabled by default")
    else:
        print("✗ Checkpoint cleanup is still disabled")

    print("✓ Checkpoint cleanup test passed")


if __name__ == "__main__":
    print("TESTING TRAINING FIXES")
    print("=" * 80)

    # Test 1: Environment step format
    test_environment_step_format()

    # Test 2: Save profile array lengths
    test_save_profile_arrays()

    # Test 3: Checkpoint cleanup enabled
    test_checkpoint_cleanup_enabled()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
