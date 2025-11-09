#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify MASA environment setup
"""

import sys
import os

print("=" * 60)
print("MASA Framework Environment Test")
print("=" * 60)

# Test Python version
print(f"\n✓ Python version: {sys.version}")

# Test critical imports
print("\n[1/8] Testing critical imports...")
try:
    import numpy as np
    import pandas as pd
    import torch as th
    import gym
    import stable_baselines3
    import cvxopt
    import cvxpy
    import scipy
    import matplotlib
    print("  ✓ All core dependencies imported")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test MASA modules
print("\n[2/8] Testing MASA modules...")
try:
    from config import Config
    print("  ✓ config.py")
except Exception as e:
    print(f"  ✗ config.py: {e}")
    sys.exit(1)

try:
    from utils.featGen import FeatureProcesser
    print("  ✓ utils.featGen")
except Exception as e:
    print(f"  ✗ utils.featGen: {e}")
    sys.exit(1)

try:
    from utils.tradeEnv import StockPortfolioEnv
    print("  ✓ utils.tradeEnv")
except Exception as e:
    print(f"  ✗ utils.tradeEnv: {e}")
    sys.exit(1)

try:
    from utils.model_pool import model_select
    print("  ✓ utils.model_pool")
except Exception as e:
    print(f"  ✗ utils.model_pool: {e}")
    sys.exit(1)

try:
    from utils.callback_func import PoCallback
    print("  ✓ utils.callback_func")
except Exception as e:
    print(f"  ✗ utils.callback_func: {e}")
    sys.exit(1)

try:
    from RL_controller.TD3_controller import TD3Controller
    print("  ✓ RL_controller.TD3_controller")
except Exception as e:
    print(f"  ✗ RL_controller.TD3_controller: {e}")
    sys.exit(1)

try:
    from RL_controller.market_obs import MarketObserver
    print("  ✓ RL_controller.market_obs")
except Exception as e:
    print(f"  ✗ RL_controller.market_obs: {e}")
    sys.exit(1)

try:
    from RL_controller.controllers import RL_withController
    print("  ✓ RL_controller.controllers")
except Exception as e:
    print(f"  ✗ RL_controller.controllers: {e}")
    sys.exit(1)

# Test Config initialization
print("\n[3/8] Testing Config initialization...")
try:
    import datetime
    import time
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))
    config = Config(seed_num=rand_seed, current_date=current_date)
    print(f"  ✓ Config initialized")
    print(f"    - Mode: {config.mode}")
    print(f"    - Market: {config.market_name}")
    print(f"    - TopK: {config.topK}")
except Exception as e:
    print(f"  ✗ Config initialization failed: {e}")
    sys.exit(1)

# Test entrance.py
print("\n[4/8] Testing entrance.py...")
try:
    import entrance
    print("  ✓ entrance.py imported")
except Exception as e:
    print(f"  ✗ entrance.py import failed: {e}")
    sys.exit(1)

# Test GPU availability
print("\n[5/8] Testing GPU availability...")
if th.cuda.is_available():
    print(f"  ✓ CUDA available: {th.cuda.get_device_name(0)}")
else:
    print("  ℹ CUDA not available, will use CPU")

# Test data directory
print("\n[6/8] Testing data directory...")
data_dir = "./data"
if os.path.exists(data_dir):
    print(f"  ✓ Data directory exists: {data_dir}")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if csv_files:
        print(f"    - Found {len(csv_files)} CSV files")
        for f in csv_files[:3]:
            print(f"      * {f}")
    else:
        print("    ⚠ No CSV files found (data files needed for training)")
else:
    print(f"  ⚠ Data directory not found: {data_dir}")

# Test optional dependencies
print("\n[7/8] Testing optional dependencies...")
try:
    from talib import abstract
    print("  ✓ TA-Lib installed")
except ImportError:
    print("  ⚠ TA-Lib not installed (optional, will use fallback)")

try:
    import empyrical
    print("  ✓ Empirically installed")
except ImportError:
    print("  ⚠ Empirically not installed (optional)")

# Final summary
print("\n[8/8] Environment Summary")
print("=" * 60)
print("✓ All critical modules imported successfully")
print("✓ Config initialization works")
print("✓ Code is ready to run")
print("\n⚠ Note: Data files are required for training")
print("   Place your data files in ./data/ directory")
print("=" * 60)
print("\n✅ Environment test PASSED!")

