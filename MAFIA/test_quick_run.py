#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify MAFIA pipeline runs without errors
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

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
from utils.featGen import FeatureProcesser
from utils.tradeEnv import StockPortfolioEnv
from RL_controller.mafia_observer import MAFIAObserver

def quick_test():
    """Quick test with minimal epochs"""
    print("=" * 60)
    print("Quick MAFIA Pipeline Test")
    print("=" * 60)
    
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    rand_seed = 2022  # Fixed seed for reproducibility
    
    random.seed(rand_seed)
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    th.manual_seed(rand_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(rand_seed)
        th.cuda.manual_seed_all(rand_seed)
    
    config = Config(seed_num=rand_seed, current_date=current_date)
    config.benchmark_algo = 'MASA-mafia'
    config.enable_market_observer = True
    config.num_epochs = 1  # Just 1 epoch for quick test
    
    # Get dataset
    mkt_name = config.market_name
    fpath = os.path.join(config.dataDir, '{}_{}_{}.csv'.format(mkt_name, config.topK, config.freq))
    if not os.path.exists(fpath):
        raise ValueError("Cannot load the data from {}".format(fpath))
    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst
    stock_num = data_dict['train']['stock'].nunique()
    print(f"✓ Data processed: {stock_num} stocks")
    
    # Initialize MAFIA observer
    mkt_observer = MAFIAObserver(config=config, action_dim=stock_num)
    print("✓ MAFIA Observer initialized")
    
    # Initialize environment
    trainInvest_env_para = config.invest_env_para 
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train', stock_num=stock_num, action_dim=stock_num, 
        tech_indicator_lst=tech_indicator_lst, extra_data=data_dict['extra_train'], 
        mkt_observer=mkt_observer, **trainInvest_env_para
    )
    print("✓ Training environment initialized")
    
    # Test a few steps
    print("\nTesting a few steps...")
    obs = env_train.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Unwrap if tuple
    print(f"✓ Reset successful, obs shape: {np.array(obs).shape if not isinstance(obs, np.ndarray) else obs.shape}")
    
    for i in range(5):
        # Random action for testing
        action = np.random.dirichlet(np.ones(stock_num))
        result = env_train.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        print(f"  Step {i+1}: reward={reward:.6f}, done={done}")
        if done:
            break
    
    print("\n" + "=" * 60)
    print("✓ Quick test completed successfully!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

