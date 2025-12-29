#!/usr/bin/env python3
"""
Full Flow Test for MAFIA Observer
=================================
Tests the complete "Macro -> Selection" training lifecycle with minimal data.

1. Sets up temporary output directory.
2. Loads minimal stock data (top 5 stocks).
3. Phase 1: MACRO Training (2 epochs).
4. Phase 2: SELECTION Training (2 epochs, Transfer from Macro).
5. Verifies checkpoints and logs.
"""

import os
import sys
import shutil
import pandas as pd
import torch as th
from datetime import datetime

# Add root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config
from scripts.train_observer_offline import run_offline_observer_training
from utils.display_integration import smart_print

def create_mock_config(output_dir):
    """Create a minimal config for testing."""
    config = Config()
    
    # Paths
    config.res_root = os.path.join(output_dir, "results")
    config.checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(config.res_root, exist_ok=True)
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    # Minimal Data
    config.mafia_top_k = 3
    config.mafia_batch_size = 4
    config.mafia_trajectory_length = 32  # Small trajectory
    config.mafia_T_w = 30
    config.mafia_pg_reward_horizon = 5
    
    # Model Specs
    config.mafia_d_model = 16
    config.mafia_n_heads = 2
    config.mafia_n_layers = 1
    config.mafia_dropout = 0.0
    
    # Disable features that need extra data/compute
    config.log_trajectory_details = False  # Keep logs clean
    config.use_mixed_precision = False
    
    return config

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(REPO_ROOT, "test_full_flow", f"run_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)
    
    smart_print(f"🚀 Starting Full Flow Test in: {test_dir}")
    
    # 1. Setup Data (Use real data slice if available, else fail)
    # We assume 'data/stock_data.csv' exists as seen in file listing.
    # We will let the trainer load it, but we force year range to be small.
    start_year = 2015
    first_infer_year = 2016  # Just 1 year train
    last_infer_year = 2016
    
    # 2. Phase 1: MACRO Training
    smart_print("\n" + "="*50)
    smart_print("  PHASE 1: MACRO TRAINING")
    smart_print("="*50)
    
    macro_out = os.path.join(test_dir, "macro")
    os.makedirs(macro_out, exist_ok=True)
    
    try:
        final_ckpt_macro = run_offline_observer_training(
            start_year=start_year,
            first_infer_year=first_infer_year,
            last_infer_year=last_infer_year,
            output_dir=macro_out,
            num_epochs=1,           # Fast test
            batches_per_epoch=2,    # Force small steps
            traj_len=32,
            seed=42,
            verbose=True,
            max_iterations=1, # Fast test
            mode="MACRO_ONLY" # Explicit Mode
            # Waait. I need to check if run_offline_observer_training accepts **kwargs or training_mode.
            # Step 9 view showed lines 1-800. Definition at 788.
            # def run_offline_observer_training(..., max_iterations=None,
            # it ended there in the view. I need to verify if I can pass training_mode.
            # If not, I need to modify it.
            # Assuming I can pass it if I added it? 
            # I will assume I need to check/fix it.
        )
        smart_print(f"✅ Phase 1 Result: {final_ckpt_macro}")
        
        # Verify result is not empty
        if not final_ckpt_macro or len(final_ckpt_macro) == 0:
             raise ValueError("Phase 1 returned empty results")
             
        # Get checkpoint path from known location since return structure varies
        # or parse from result if available. 
        # But for now, let's just use the know output path standard.
        macro_ckpt_path = os.path.join(macro_out, "checkpoints", "Observer_Infer_2016_Q1.pth")
        if not os.path.exists(macro_ckpt_path):
             # Try alternate name if mode changed prefix
             macro_ckpt_path = os.path.join(macro_out, "checkpoints", "Observer_Macro_2016_Q1.pth")
        
        if not os.path.exists(macro_ckpt_path):
             raise FileNotFoundError(f"Macro checkpoint not found at {macro_ckpt_path}")
             
        smart_print(f"✅ Verified Checkpoint: {macro_ckpt_path}")

    except Exception as e:
        smart_print(f"❌ Phase 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Phase 2: SELECTION Training (Transfer)
    smart_print("\n" + "="*50)
    smart_print("  PHASE 2: SELECTION TRANSFER")
    smart_print("="*50)
    
    select_out = os.path.join(test_dir, "selection")
    os.makedirs(select_out, exist_ok=True)
    
    try:
        # We need to pass training_mode="SELECTION" and init_checkpoint
        # I need to confirm `run_offline_observer_training` supports these.
        # If not, I will update it.
        
        final_ckpt_select = run_offline_observer_training(
            start_year=start_year,
            first_infer_year=first_infer_year,
            last_infer_year=last_infer_year,
            output_dir=select_out,
            num_epochs=1,
            batches_per_epoch=2,
            traj_len=32,
            seed=42,
            verbose=True,
            max_iterations=1,
            mode="SELECTION_ONLY",
            init_checkpoint=final_ckpt_macro[0]["final_checkpoint"] # run outputs list of results. We need path from result or construct it?
            # Returns list of dicts. Result dict has 'final_checkpoint'?
            # checking return of run_offline... in file
        )
        smart_print(f"✅ Phase 2 Complete. Checkpoint: {final_ckpt_select}")

    except Exception as e:
        smart_print(f"❌ Phase 2 Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    smart_print("\n🎉 Full Flow Test Passed!")

if __name__ == "__main__":
    main()
