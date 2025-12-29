#!/usr/bin/env python3
"""
Test TensorBoard Logger Integration

Quick test to verify TensorBoard logger works correctly.
"""
import os
import sys
import numpy as np
import torch as th

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.tensorboard_logger import TensorBoardLogger, create_tensorboard_logger
from config import Config

def test_tensorboard_logger():
    """Test basic TensorBoard logger functionality."""
    print("=" * 70)
    print("Testing TensorBoard Logger")
    print("=" * 70)
    
    # Create test directory
    test_dir = "./test_tensorboard"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test 1: Basic initialization
    print("\n[TEST 1] Initialization...")
    logger = TensorBoardLogger(
        log_dir=os.path.join(test_dir, "run1"),
        enabled=True,
        log_images=True,
        log_histograms=True,
        histogram_freq=5,
    )
    print("✓ Logger initialized successfully")
    
    # Test 2: Scalar logging
    print("\n[TEST 2] Scalar logging...")
    for step in range(10):
        logger.log_scalar("loss/total", np.random.rand(), step, phase="train")
        logger.log_scalar("loss/pg", np.random.rand(), step, phase="train")
        logger.log_scalar("metrics/sharpe", np.random.rand() * 2, step, phase="valid")
    print("✓ Scalars logged successfully")
    
    # Test 3: Histogram logging
    print("\n[TEST 3] Histogram logging...")
    weights = th.randn(100, 50)
    logger.log_histogram("weights/layer1", weights, step=5, phase="train")
    print("✓ Histogram logged successfully")
    
    # Test 4: Multiple scalars
    print("\n[TEST 4] Multiple scalars...")
    logger.log_scalars(
        "losses",
        {"pg": 0.5, "risk": 0.3, "direction": 0.2},
        step=10,
        phase="train"
    )
    print("✓ Multiple scalars logged successfully")
    
    # Test 5: Text logging
    print("\n[TEST 5] Text logging...")
    logger.log_text("config", "Test configuration: batch_size=32, lr=0.001", step=0)
    print("✓ Text logged successfully")
    
    # Test 6: Hyperparameters
    print("\n[TEST 6] Hyperparameters...")
    hparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
    }
    metrics = {
        "final_sharpe": 1.5,
        "final_ces": 0.8,
    }
    logger.log_hparams(hparams, metrics)
    print("✓ Hyperparameters logged successfully")
    
    # Test 7: Flush and close
    print("\n[TEST 7] Flush and close...")
    logger.flush()
    logger.close()
    print("✓ Logger closed successfully")
    
    # Test 8: Factory function with config
    print("\n[TEST 8] Factory function...")
    config = Config(create_dirs=False)
    logger2 = create_tensorboard_logger(
        base_dir=test_dir,
        run_name="test_run",
        config=config,
        enabled=True,
    )
    logger2.log_scalar("test/metric", 1.0, 0)
    logger2.close()
    print("✓ Factory function works correctly")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print(f"\nTensorBoard logs saved to: {test_dir}")
    print(f"To view, run: tensorboard --logdir {test_dir}")
    print("=" * 70)

if __name__ == "__main__":
    test_tensorboard_logger()
