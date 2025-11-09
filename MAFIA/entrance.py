#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name: entrance.py  
 Author: MASA
--------------------------------
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# Only set CUDA_VISIBLE_DEVICES if CUDA is available
import random 
import numpy as np
import torch as th
import datetime
import copy

# Set CUDA device if available, otherwise use CPU
if th.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU or change to '1' if needed
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        th.use_deterministic_algorithms(True)
    except Exception:
        # Fallback if deterministic algorithms not available
        th.use_deterministic_algorithms(False, warn_only=True)
else:
    print("CUDA not available, using CPU", flush=True)

import pandas as pd
import time
import json
import os
from config import Config
from utils.featGen import FeatureProcesser
from utils.tradeEnv import StockPortfolioEnv, StockPortfolioEnv_cash
from utils.model_pool import model_select, benchmark_algo_select
from utils.callback_func import PoCallback
from utils.data_validator import get_stock_data_file
from RL_controller.market_obs import MarketObserver, MarketObserver_Algorithmic
from RL_controller.mafia_observer import MAFIAObserver
import timeit

def RLonly(config):
    # For running the single-agent RL-based framework (TD3-Profit, TD3-PR, TD3-SR)
    # Get dataset
    fpath, error_msg = get_stock_data_file(config)
    if fpath is None:
        raise ValueError(f"Cannot load the data file. {error_msg}")
    print(f"Loading stock data from: {fpath}", flush=True)
    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst 
    stock_num = data_dict['train']['stock'].nunique()
    print("Data has been processed..")

    # Initialize environment
    trainInvest_env_para = config.invest_env_para 
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train', stock_num=stock_num, action_dim=stock_num, 
        tech_indicator_lst=tech_indicator_lst, **trainInvest_env_para
    )
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para 
        env_valid = StockPortfolioEnv(
            config=config, rawdata=data_dict['valid'], mode='valid', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, **validInvest_env_para
        )
    else:
        env_valid = None

    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para 
        env_test = StockPortfolioEnv(
            config=config, rawdata=data_dict['test'], mode='test', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, **testInvest_env_para
        )
    else:
        env_test = None

    # Load RL model
    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    model_para_dict = config.model_para
    po_model = ModelCls(env=env_train, **model_para_dict) # Create instance 
    total_timesteps = int(config.num_epochs * env_train.totalTradeDay)
    print('Training Start', flush=True)
    log_interval = 10
    callback1 =PoCallback(config=config, train_env=env_train, valid_env=env_valid, test_env=env_test)
    cpt_start = time.process_time()
    perft_start = time.perf_counter()
    timeit_default = timeit.default_timer()
    my_globals = globals()
    my_globals.update({'po_model': po_model, 'total_timesteps': total_timesteps, 'callback1': callback1, 'log_interval': log_interval})
    t = timeit.Timer(stmt='po_model.learn(total_timesteps=total_timesteps, callback=callback1, log_interval=log_interval)', globals=my_globals)
    time_usage = t.timeit(number=1)
    cpt_usgae = time.process_time() - cpt_start
    perf_usgae = time.perf_counter() - perft_start
    timeit_usgae = timeit.default_timer() - timeit_default
    print("Time usgae for {} epochs: {}s, cpu time: {}s, perf_couter: {}s, timeit_default: {}s".format(config.num_epochs, np.round(time_usage, 2), np.round(cpt_usgae, 2), np.round(perf_usgae, 2), np.round(timeit_usgae, 2)))
    print("-*"*20)

    del po_model
    print("Training Done...", flush=True)


def RLcontroller(config):
    # For running the MASA framework
    # Get dataset
    fpath, error_msg = get_stock_data_file(config)
    if fpath is None:
        raise ValueError(f"Cannot load the data file. {error_msg}")
    print(f"Loading stock data from: {fpath}", flush=True)
    data = pd.DataFrame(pd.read_csv(fpath, header=0))

    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst
    stock_num = data_dict['train']['stock'].nunique()
    print("Data has been processed..")

    if config.enable_market_observer:
        if config.mktobs_algo == 'mafia_1':
            mkt_observer = MAFIAObserver(config=config, action_dim=stock_num)
        elif ('ma' in config.mktobs_algo) or ('dc' in config.mktobs_algo):
            mkt_observer = MarketObserver_Algorithmic(config=config, action_dim=stock_num)
        else:
            mkt_observer = MarketObserver(config=config, action_dim=stock_num) 
    else:
        mkt_observer = None

    # Initialize environment
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para 
        env_valid = StockPortfolioEnv(
            config=config, rawdata=data_dict['valid'], mode='valid', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, extra_data=data_dict['extra_valid'], 
            mkt_observer=mkt_observer, **validInvest_env_para
        )
    else:
        env_valid = None
        raise ValueError("No validation set is provided for training")
    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para 
        env_test = StockPortfolioEnv(
            config=config, rawdata=data_dict['test'], mode='test', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, extra_data=data_dict['extra_test'], 
            mkt_observer=mkt_observer, **testInvest_env_para
        )
    else:
        env_test = None
        raise ValueError("No test set is provided for training")

    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    # Initialize environment
    trainInvest_env_para = config.invest_env_para 
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train', stock_num=stock_num, action_dim=stock_num, 
        tech_indicator_lst=tech_indicator_lst, extra_data=data_dict['extra_train'], 
        mkt_observer=mkt_observer, **trainInvest_env_para
    )

    # Load RL model
    model_para_dict = config.model_para
    start_epoch = 0
    
    # Auto-detect latest checkpoint if auto_resume is enabled and no checkpoint specified
    checkpoint_to_resume = config.resume_from_checkpoint
    if checkpoint_to_resume is None and config.auto_resume_from_latest:
        # Find latest checkpoint in checkpoint_dir
        if os.path.exists(config.checkpoint_dir):
            checkpoint_dirs = [d for d in os.listdir(config.checkpoint_dir) 
                             if d.startswith('checkpoint_epoch_') and os.path.isdir(os.path.join(config.checkpoint_dir, d))]
            if checkpoint_dirs:
                # Sort by epoch number (extract number from directory name)
                def get_epoch_num(dirname):
                    try:
                        return int(dirname.split('_')[-1])
                    except:
                        return -1
                checkpoint_dirs.sort(key=get_epoch_num, reverse=True)
                latest_checkpoint_dir = checkpoint_dirs[0]
                latest_checkpoint_path = os.path.join(config.checkpoint_dir, latest_checkpoint_dir, 'checkpoint_info.json')
                if os.path.exists(latest_checkpoint_path):
                    checkpoint_to_resume = latest_checkpoint_path
                    print(f"Auto-detected latest checkpoint: {latest_checkpoint_path}", flush=True)
    
    # Resume from checkpoint if specified or auto-detected
    if checkpoint_to_resume is not None and os.path.exists(checkpoint_to_resume):
        print(f"Resuming training from checkpoint: {checkpoint_to_resume}", flush=True)
        
        # Load checkpoint info
        checkpoint_dir = os.path.dirname(checkpoint_to_resume)
        info_path = os.path.join(checkpoint_dir, 'checkpoint_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                checkpoint_info = json.load(f)
            start_epoch = checkpoint_info.get('epoch', 0)
            print(f"Resuming from epoch {start_epoch}", flush=True)
        
        # Load RL model from checkpoint
        rl_checkpoint_path = os.path.join(checkpoint_dir, 'rl_model.zip')
        if os.path.exists(rl_checkpoint_path):
            po_model = ModelCls.load(rl_checkpoint_path, env=env_train)
            print(f"RL model loaded from {rl_checkpoint_path}", flush=True)
        else:
            po_model = ModelCls(env=env_train, **model_para_dict)
            print(f"Warning: RL checkpoint not found, starting fresh", flush=True)
        
        # Load MAFIA observer from checkpoint if exists
        if (config.enable_market_observer and 
            hasattr(env_train, 'mkt_observer') and 
            env_train.mkt_observer is not None and
            hasattr(env_train.mkt_observer, 'load_checkpoint')):
            mafia_checkpoint_path = os.path.join(checkpoint_dir, 'mafia_observer.pth')
            if os.path.exists(mafia_checkpoint_path):
                loaded_epoch = env_train.mkt_observer.load_checkpoint(mafia_checkpoint_path)
                print(f"MAFIA observer loaded from {mafia_checkpoint_path} (epoch {loaded_epoch})", flush=True)
            else:
                print(f"Warning: MAFIA observer checkpoint not found, starting fresh", flush=True)
    else:
        po_model = ModelCls(env=env_train, **model_para_dict)
    
    # Calculate remaining timesteps
    remaining_epochs = max(0, config.num_epochs - start_epoch)
    total_timesteps = int(remaining_epochs * env_train.totalTradeDay)
    
    if start_epoch > 0:
        print(f'Resuming training from epoch {start_epoch} ({remaining_epochs} epochs remaining)', flush=True)
    else:
        print('Training Start', flush=True)
    log_interval = 10
    callback1 = PoCallback(config=config, train_env=env_train, valid_env=env_valid, test_env=env_test)
    cpt_start = time.process_time()
    perft_start = time.perf_counter()
    timeit_default = timeit.default_timer()
    my_globals = globals()
    my_globals.update({'po_model': po_model, 'total_timesteps': total_timesteps, 'callback1': callback1, 'log_interval': log_interval})
    t = timeit.Timer(stmt='po_model.learn(total_timesteps=total_timesteps, callback=callback1, log_interval=log_interval)', globals=my_globals)
    time_usage = t.timeit(number=1)
    cpt_usgae = time.process_time() - cpt_start
    perf_usgae = time.perf_counter() - perft_start
    timeit_usgae = timeit.default_timer() - timeit_default

    print("Time usgae for {} epochs: {}s, cpu time: {}s, perf_couter: {}s, timeit_default: {}s".format(config.num_epochs, np.round(time_usage, 2), np.round(cpt_usgae, 2), np.round(perf_usgae, 2), np.round(timeit_usgae, 2)))
    print("-*"*20)
    del po_model
    print("Training Done...", flush=True)

def entrance():
    """
    Entrance function for running the MASA framework
    """
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))

    random.seed(rand_seed)
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    th.manual_seed(rand_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(rand_seed)
        th.cuda.manual_seed_all(rand_seed)

    start_cputime = time.process_time()
    start_systime = time.perf_counter()
    config = Config(seed_num=rand_seed, current_date=current_date) 
    config.print_config()
    if config.mode == 'RLonly':
        RLonly(config=config)
    elif config.mode == 'RLcontroller':
        RLcontroller(config=config)
    elif config.mode == 'Benchmark':
        raise NotImplementedError("Please refer to the corresponding papers and their provided implementations.")
    else:
        raise ValueError('Unexpected mode {}'.format(config.mode))

    end_cputime = time.process_time()
    end_systime = time.perf_counter()
    print("[Done] Total cputime: {} s, system time: {} s".format(np.round(end_cputime - start_cputime, 2), np.round(end_systime - start_systime, 2)))
    
def main():
    entrance()

if __name__ == '__main__':
    main()
