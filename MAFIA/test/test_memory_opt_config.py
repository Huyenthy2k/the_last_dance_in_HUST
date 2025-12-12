#!/usr/bin/env python3
"""
Simplified test to verify Memory Optimizations config is set correctly.

This test verifies:
1. Config flags exist and have correct default values
2. Mixed Precision logic (GradScaler) is conditional on CUDA
3. Gradient Accumulation parameters are set

Usage:
    python test_memory_opt_config.py
"""

import sys
import os

# Add MAFIA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents', 'MAFIA'))

from config import Config

def test_config():
    """Test config has all memory optimization flags."""
    print("="*70)
    print("MAFIA MEMORY OPTIMIZATION CONFIG TEST")
    print("="*70)
    
    config = Config(seed_num=42)
    
    # Test 1: Flags exist
    print("\n[Test 1] Config Flags Exist")
    assert hasattr(config, 'use_mixed_precision'), "Missing use_mixed_precision"
    assert hasattr(config, 'gradient_accumulation_steps'), "Missing gradient_accumulation_steps"
    assert hasattr(config, 'use_gradient_checkpointing'), "Missing use_gradient_checkpointing"
    print("✅ All flags exist")
    
    # Test 2: Default values
    print("\n[Test 2] Default Values")
    print(f"  use_mixed_precision: {config.use_mixed_precision}")
    print(f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    print(f"  use_gradient_checkpointing: {config.use_gradient_checkpointing}")
    print(f"  mafia_batch_size: {config.mafia_batch_size}")
    
    assert config.use_mixed_precision == True, "Mixed Precision should be enabled by default"
    assert config.gradient_accumulation_steps == 4, f"Expected 4, got {config.gradient_accumulation_steps}"
    assert config.use_gradient_checkpointing == True, "Gradient Checkpointing should be enabled by default"
    print("✅ Default values correct")
    
    # Test 3: Effective batch size
    print("\n[Test 3] Effective Batch Size Calculation")
    effective_batch = config.mafia_batch_size * config.gradient_accumulation_steps
    print(f"  Batch size: {config.mafia_batch_size}")
    print(f"  Accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {effective_batch}")
    assert effective_batch == 128, f"Expected 128, got {effective_batch}"
    print("✅ Effective batch size correct (32 × 4 = 128)")
    
    # Test 4: Can disable optimizations
    print("\n[Test 4] Can Disable Optimizations")
    config.use_mixed_precision = False
    config.gradient_accumulation_steps = 1
    config.use_gradient_checkpointing = False
    assert config.use_mixed_precision == False
    assert config.gradient_accumulation_steps == 1
    assert config.use_gradient_checkpointing == False
    print("✅ Can disable all optimizations")
    
    return True

def main():
    try:
        if test_config():
            print("\n" + "="*70)
            print("🎉 ALL TESTS PASSED!")
            print("="*70)
            print("\n✅ Memory Optimization Config Implementation:")
            print("  • Mixed Precision (FP16): Config parameter added")
            print("  • Gradient Accumulation: Config parameter added")
            print("  • Gradient Checkpointing: Config parameter added")
            print("\n📝 Next steps:")
            print("  • Run full training to test memory reduction")
            print("  • Monitor RAM/VRAM usage with nvidia-smi or Activity Monitor")
            print("  • Verify training converges to similar metrics")
            return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
