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
DEFAULT_RUN_SEED = 2022

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
import pickle
from config import Config
from utils.featGen import FeatureProcesser
from utils.tradeEnv import StockPortfolioEnv, StockPortfolioEnv_cash
from utils.model_pool import model_select, benchmark_algo_select
from utils.callback_func import PoCallback
from utils.data_validator import get_stock_data_file
from RL_controller.mafia_observer import MAFIAObserver
import timeit

# Legacy RLonly function removed - MAFIA-only codebase now uses RLcontroller exclusively


def RLcontroller(config):
    """
    MAFIA training pipeline (MASA framework with MAFIA observer).
    This is the only supported training mode.
    """
    # Get dataset
    fpath, error_msg = get_stock_data_file(config)
    if fpath is None:
        raise ValueError(f"Cannot load the data file. {error_msg}")
    print(f"Loading stock data from: {fpath}", flush=True)
    data = pd.DataFrame(pd.read_csv(fpath, header=0))

    # MAFIA lightweight data loading 
    from utils.mafia_data_loader import MAFIADataLoader
    print("[MAFIA] Using lightweight data loader (no legacy preprocessing)...", flush=True)
    mafia_loader = MAFIADataLoader(config=config)
    data_dict = mafia_loader.load_and_split_data(data=data)
    tech_indicator_lst = []  # MAFIA doesn't use legacy tech indicators
    stock_num = data_dict['train']['stock'].nunique()
    print("Data loading complete.")

    # Initialize MAFIA observer (always enabled)
    mkt_observer = MAFIAObserver(config=config, action_dim=stock_num)

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
    # Ensure checkpoint_to_resume is either None or a valid string path
    if checkpoint_to_resume is not None and not isinstance(checkpoint_to_resume, str):
        print(f"Warning: resume_from_checkpoint is not a string (got {type(checkpoint_to_resume)}), resetting to None", flush=True)
        checkpoint_to_resume = None
    
    if checkpoint_to_resume is None and config.auto_resume_from_latest:
        # Find latest checkpoint - search in current checkpoint_dir and all parent res directories
        search_dirs = [config.checkpoint_dir]  # Start with current checkpoint_dir
        
        # Also search in parent res directories (for previous runs)
        base_res_dir = os.path.dirname(config.checkpoint_dir)  # Remove 'checkpoints'
        if os.path.exists(base_res_dir):
            parent_dir = os.path.dirname(base_res_dir)  # res/RLcontroller/TD3/VNINDEX-10
            if os.path.exists(parent_dir):
                # Search in all timestamp directories
                for timestamp_dir in os.listdir(parent_dir):
                    timestamp_path = os.path.join(parent_dir, timestamp_dir)
                    if os.path.isdir(timestamp_path):
                        checkpoint_path = os.path.join(timestamp_path, 'checkpoints')
                        if os.path.exists(checkpoint_path) and checkpoint_path != config.checkpoint_dir:
                            search_dirs.append(checkpoint_path)
        
        checkpoint_records = []
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            checkpoint_dirs = [
                d for d in os.listdir(search_dir)
                if os.path.isdir(os.path.join(search_dir, d))
            ]
            for dirname in checkpoint_dirs:
                info_path = os.path.join(search_dir, dirname, 'checkpoint_info.json')
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        checkpoint_records.append({
                            'info_path': info_path,
                            'info': info,
                            'mtime': os.path.getmtime(info_path)
                        })
                    except Exception as e:
                        print(f"Warning: Failed to read checkpoint info from {info_path}: {e}", flush=True)
        
        if checkpoint_records:
            # Sort by timesteps, then epoch, then latest modification time
            checkpoint_records.sort(
                key=lambda item: (
                    item['info'].get('timesteps', 0),
                    item['info'].get('epoch', 0),
                    item['mtime']
                ),
                reverse=True
            )
            latest_record = checkpoint_records[0]
            checkpoint_to_resume = latest_record['info_path']
            print(f"Auto-detected latest checkpoint: {checkpoint_to_resume}", flush=True)
            print(f"  Epoch: {latest_record['info'].get('epoch', 0)}, Timesteps: {latest_record['info'].get('timesteps', 0)}", flush=True)
    
    # Resume from checkpoint if specified or auto-detected
    start_epoch = 0
    checkpoint_timesteps = 0
    checkpoint_dir = None
    resume_loaded = False
    incompatible_checkpoint = False
    if checkpoint_to_resume is not None and isinstance(checkpoint_to_resume, str) and os.path.exists(checkpoint_to_resume):
        print(f"Resuming training from checkpoint: {checkpoint_to_resume}", flush=True)
        
        # Load checkpoint info
        checkpoint_dir = os.path.dirname(checkpoint_to_resume)
        info_path = os.path.join(checkpoint_dir, 'checkpoint_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                checkpoint_info = json.load(f)
            start_epoch = checkpoint_info.get('epoch', 0)
            checkpoint_timesteps = checkpoint_info.get('timesteps', 0)
            checkpoint_type = checkpoint_info.get('type', 'epoch')
            day_in_epoch = checkpoint_info.get('day_in_epoch', 0)
            
            print(f"Checkpoint type: {checkpoint_type}", flush=True)
            print(f"Resuming from epoch {start_epoch}, day {day_in_epoch}, timestep {checkpoint_timesteps}", flush=True)
            
            # Check if this is a mid-epoch resume (day_in_epoch > 0 means we're in the middle of an epoch)
            env_state_path = checkpoint_info.get('env_state_path', None)
            is_mid_epoch_resume = (day_in_epoch > 0) and (env_state_path is not None) and os.path.exists(env_state_path)
            
            if is_mid_epoch_resume:
                # For mid-epoch resume: set epoch to start_epoch - 1 so after reset() it becomes start_epoch
                # Then we'll restore the full state (including curTradeDay) after reset()
                if hasattr(env_train, 'epoch'):
                    env_train.epoch = start_epoch - 1
                    print(f"Mid-epoch resume detected: set env.epoch to {start_epoch - 1} (will become {start_epoch} after reset)", flush=True)
                    print(f"Will restore environment state from {env_state_path} after reset()", flush=True)
                # Store env_state_path in environment for callback to restore after reset()
                env_train._resume_env_state_path = env_state_path
            else:
                # For epoch-end resume: set epoch to start_epoch so after reset() it becomes start_epoch + 1 (new epoch)
                if hasattr(env_train, 'epoch'):
                    env_train.epoch = start_epoch
                    print(f"Epoch-end resume: set env.epoch to {start_epoch} (will become {start_epoch + 1} after reset)", flush=True)
                # Clear any previous resume state path
                if hasattr(env_train, '_resume_env_state_path'):
                    delattr(env_train, '_resume_env_state_path')
            
            # Restore RNG state if available
            rng_state_path = checkpoint_info.get('rng_state_path', None)
            if rng_state_path and os.path.exists(rng_state_path):
                try:
                    with open(rng_state_path, 'rb') as f:
                        rng_state = pickle.load(f)
                    seed_from_rng = rng_state.get('seed_num', None)
                    if seed_from_rng is not None:
                        config.seed_num = seed_from_rng
                    py_state = rng_state.get('python_random')
                    if py_state is not None:
                        random.setstate(py_state)
                    np_state = rng_state.get('numpy_random')
                    if np_state is not None:
                        np.random.set_state(np_state)
                    torch_state = rng_state.get('torch_cpu')
                    if torch_state is not None:
                        th.set_rng_state(torch_state)
                    torch_cuda_state = rng_state.get('torch_cuda')
                    if torch_cuda_state is not None and th.cuda.is_available():
                        th.cuda.set_rng_state_all(torch_cuda_state)
                    print(f"[RESUME] RNG state restored from {rng_state_path}", flush=True)
                except Exception as e:
                    print(f"[RESUME] Warning: Failed to restore RNG state: {e}", flush=True)
            else:
                if rng_state_path:
                    print(f"[RESUME] RNG state file not found at {rng_state_path}", flush=True)
        
        # Load RL model from checkpoint
        rl_checkpoint_path = os.path.join(checkpoint_dir, 'rl_model.zip')
        if os.path.exists(rl_checkpoint_path):
            try:
                po_model = ModelCls.load(rl_checkpoint_path, env=env_train)
                po_model.mafia_config = config
                po_model.verbose = 1
                resume_loaded = True
                print(f"RL model loaded from {rl_checkpoint_path}", flush=True)
            except ValueError as e:
                msg = str(e)
                mismatch_signatures = [
                    "Action spaces do not match",
                    "Observation spaces do not match",
                ]
                if any(signature in msg for signature in mismatch_signatures):
                    incompatible_checkpoint = True
                    print(f"Warning: Checkpoint at {rl_checkpoint_path} is incompatible with current environment ({msg}). Starting fresh training.", flush=True)
                else:
                    raise
        else:
            print(f"Warning: RL checkpoint not found at {rl_checkpoint_path}, starting fresh", flush=True)

        if not resume_loaded:
            po_model = ModelCls(env=env_train, **model_para_dict)
            po_model.mafia_config = config
            po_model.verbose = 1
            # Reset resume-specific metadata when checkpoint cannot be loaded
            start_epoch = 0
            checkpoint_timesteps = 0
            checkpoint_dir = None
            checkpoint_to_resume = None
            if hasattr(env_train, '_resume_env_state_path'):
                delattr(env_train, '_resume_env_state_path')
            if hasattr(env_train, 'epoch'):
                env_train.epoch = 0
        else:
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
        po_model.mafia_config = config
        po_model.verbose = 1
    
    # Calculate remaining timesteps and epochs
    # If resuming from checkpoint, calculate from checkpoint timesteps
    if checkpoint_timesteps > 0:
        total_timesteps_for_training = int(config.num_epochs * env_train.totalTradeDay)
        total_timesteps = max(0, total_timesteps_for_training - checkpoint_timesteps)
        # Calculate remaining epochs from checkpoint timesteps
        remaining_epochs = max(0, config.num_epochs - start_epoch)
        # More accurate: calculate from timesteps
        completed_epochs = checkpoint_timesteps // env_train.totalTradeDay
        remaining_epochs = max(0, config.num_epochs - completed_epochs)
        print(f"Resuming from timestep {checkpoint_timesteps}/{total_timesteps_for_training}", flush=True)
        print(f"Completed epochs: {completed_epochs}, Remaining epochs: {remaining_epochs}", flush=True)
    else:
        remaining_epochs = max(0, config.num_epochs - start_epoch)
        total_timesteps = int(remaining_epochs * env_train.totalTradeDay)
    
    # Print training information
    print(f"\n{'='*100}")
    print(f"{'TRAINING CONFIGURATION':^100}")
    print(f"{'='*100}")
    if start_epoch > 0 or checkpoint_timesteps > 0:
        print(f'[INFO] Resuming training from epoch {start_epoch}')
        print(f'[INFO] Remaining epochs: {remaining_epochs}/{config.num_epochs}')
    else:
        print(f'[INFO] Starting fresh training')
        print(f'[INFO] Total epochs to train: {config.num_epochs}')
    print(f'[INFO] Steps per epoch: {env_train.totalTradeDay}')
    print(f'[INFO] Total timesteps for this run: {total_timesteps}')
    print(f'[INFO] Batch size: {config.batch_size}')
    print(f'[INFO] Learning rate: {config.learning_rate}')
    print(f'[INFO] Checkpoint frequency: every {config.checkpoint_freq} epochs' if config.checkpoint_freq > 0 else '[INFO] Checkpoint saving disabled')
    print(f'[INFO] Results directory: {config.res_dir}')
    print(f"{'='*100}\n")
    print('Training Start', flush=True)
    log_interval = 10
    callback1 = PoCallback(config=config, train_env=env_train, valid_env=env_valid, test_env=env_test)
    cpt_start = time.process_time()
    perft_start = time.perf_counter()
    timeit_default = timeit.default_timer()
    
    # Set reset_num_timesteps=False when resuming from checkpoint to preserve timestep count
    reset_num_timesteps = (checkpoint_timesteps == 0)
    
    my_globals = globals()
    my_globals.update({
        'po_model': po_model, 
        'total_timesteps': total_timesteps, 
        'callback1': callback1, 
        'log_interval': log_interval,
        'reset_num_timesteps': reset_num_timesteps
    })
    t = timeit.Timer(stmt='po_model.learn(total_timesteps=total_timesteps, callback=callback1, log_interval=log_interval, reset_num_timesteps=reset_num_timesteps)', globals=my_globals)
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
    Main entry point for MAFIA training.
    Legacy models (RLonly, Benchmark) have been removed.
    This codebase now exclusively runs MAFIA (RLcontroller with MAFIA observer).
    """
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    seed_env = os.environ.get('MAFIA_SEED')
    try:
        rand_seed = int(seed_env) if seed_env is not None else DEFAULT_RUN_SEED
    except ValueError:
        rand_seed = DEFAULT_RUN_SEED

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
    
    print("="*60)
    print("MAFIA - Multi-Agent Framework with Integrated Attention")
    print("="*60)
    config.print_config()
    print("="*60)

    # Only MAFIA (RLcontroller mode) is supported
    RLcontroller(config=config)

    end_cputime = time.process_time()
    end_systime = time.perf_counter()
    print("[Done] Total cputime: {} s, system time: {} s".format(np.round(end_cputime - start_cputime, 2), np.round(end_systime - start_systime, 2)))
    
def main():
    entrance()

if __name__ == '__main__':
    main()
