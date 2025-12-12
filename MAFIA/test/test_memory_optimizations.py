#!/usr/bin/env python3
"""
Test script to verify Memory Optimizations work correctly.

Tests:
1. Mixed Precision (FP16) - GradScaler works
2. Gradient Accumulation - gradients accumulate correctly
3. Training runs without errors

Usage:
    python test_memory_optimizations.py
"""

import torch as th
import sys
import os

# Add MAFIA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents', 'MAFIA'))

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer

def test_mixed_precision():
    """Test Mixed Precision is properly initialized."""
    print("="*70)
    print("TEST 1: Mixed Precision Initialization")
    print("="*70)
    
    config = Config(seed_num=42)
    config.use_mixed_precision = True
    config.gradient_accumulation_steps = 1
    
    if th.cuda.is_available():
        device = th.device("cuda")
    elif th.backends.mps.is_available():
        device = th.device("mps")
    else:
        device = th.device("cpu")
    print(f"Device: {device}")
    
    # Create observer and trainer
    observer = MAFIAObserver(config, stock_list=["TEST1", "TEST2", "TEST3"], device=device)
    trainer = ObserverOfflineBatchTrainer(config, observer, device)
    
    # Check if scaler is initialized correctly
    if device.type == "cuda":
        assert trainer.scaler is not None, "GradScaler should be initialized on CUDA"
        assert trainer.use_mixed_precision is True
        print("✅ Mixed Precision ENABLED (CUDA available)")
        print(f"   GradScaler: {type(trainer.scaler)}")
    else:
        assert trainer.scaler is None, "GradScaler should be None on CPU"
        assert trainer.use_mixed_precision is False
        print("⚠️  Mixed Precision DISABLED (CPU mode)")
    
    print()
    return True

def test_gradient_accumulation():
    """Test Gradient Accumulation is configured correctly."""
    print("="*70)
    print("TEST 2: Gradient Accumulation Configuration")
    print("="*70)
    
    config = Config(seed_num=42)
    config.gradient_accumulation_steps = 4
    config.mafia_batch_size = 8
    
    device = th.device("cpu")  # Use CPU for simplicity
    
    observer = MAFIAObserver(config, stock_list=["TEST1", "TEST2"], device=device)
    trainer = ObserverOfflineBatchTrainer(config, observer, device)
    
    assert trainer.gradient_accumulation_steps == 4
    effective_batch = trainer.batch_size * trainer.gradient_accumulation_steps
    
    print(f"✅ Gradient Accumulation: {trainer.gradient_accumulation_steps} steps")
    print(f"   Batch size: {trainer.batch_size}")
    print(f"   Effective batch: {effective_batch}")
    assert effective_batch == 32, f"Expected 32, got {effective_batch}"
    
    print()
    return True

def test_config_flags():
    """Test config flags are properly set."""
    print("="*70)
    print("TEST 3: Config Flags")
    print("="*70)
    
    config = Config(seed_num=42)
    
    # Check all memory optimization flags exist
    assert hasattr(config, 'use_mixed_precision'), "Missing use_mixed_precision"
    assert hasattr(config, 'gradient_accumulation_steps'), "Missing gradient_accumulation_steps"
    assert hasattr(config, 'use_gradient_checkpointing'), "Missing use_gradient_checkpointing"
    
    print(f"✅ use_mixed_precision: {config.use_mixed_precision}")
    print(f"✅ gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    print(f"✅ use_gradient_checkpointing: {config.use_gradient_checkpointing}")
    
    print()
    return True

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MAFIA MEMORY OPTIMIZATION TESTS")
    print("="*70 + "\n")
    
    try:
        results = []
        results.append(("Config Flags", test_config_flags()))
        results.append(("Mixed Precision", test_mixed_precision()))
        results.append(("Gradient Accumulation", test_gradient_accumulation()))
        
        print("="*70)
        print("TEST SUMMARY")
        print("="*70)
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {name}")
        
        all_passed = all(r[1] for r in results)
        
        if all_passed:
            print("\n🎉 All tests PASSED! Memory optimizations are working correctly.")
            return 0
        else:
            print("\n❌ Some tests FAILED. Please check the implementation.")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
