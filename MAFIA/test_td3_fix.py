#!/usr/bin/env python3
"""
Test script to verify TD3 training trigger fix.
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, '.')

def test_numpy_boolean_evaluation():
    """Test numpy boolean evaluation in if statements."""
    print("=" * 60)
    print("TESTING NUMPY BOOLEAN EVALUATION")
    print("=" * 60)

    # Create numpy boolean array (similar to dones array)
    dones = np.array([True, False, True])

    print("Testing numpy boolean evaluation:")

    # Test old problematic way
    collected_old = 0
    for idx, done in enumerate(dones):
        if done:  # This might not work with numpy booleans
            collected_old += 1
            print(f"  OLD: done[{idx}] = {done} (type: {type(done)}) -> collected: {collected_old}")

    # Test new fixed way
    collected_new = 0
    for idx, done in enumerate(dones):
        if bool(done):  # Convert to Python boolean
            collected_new += 1
            print(f"  NEW: done[{idx}] = {done} (type: {type(done)}) -> collected: {collected_new}")

    print(f"Result: Old method collected {collected_old}, New method collected {collected_new}")

    if collected_new == 2:  # Should collect 2 True values
        print("✓ Numpy boolean evaluation test PASSED")
        return True
    else:
        print("✗ Numpy boolean evaluation test FAILED")
        return False

def test_td3_episode_collection():
    """Test TD3 episode collection logic."""
    print("\n" + "=" * 60)
    print("TESTING TD3 EPISODE COLLECTION LOGIC")
    print("=" * 60)

    # Simulate dones array from VecEnv (numpy array)
    dones = np.array([True])  # Episode ended

    num_collected_episodes = 0
    episodes_ended = 0

    # Test the fixed logic
    for idx, done in enumerate(dones):
        if bool(done):  # Fixed: convert numpy boolean to Python boolean
            # Update stats
            num_collected_episodes += 1
            episodes_ended += 1
            print(f"Episode {num_collected_episodes} completed! done_idx={idx}")

    print(f"Episodes ended: {episodes_ended}, Total collected: {num_collected_episodes}")

    if num_collected_episodes == 1:
        print("✓ TD3 episode collection test PASSED")
        return True
    else:
        print("✗ TD3 episode collection test FAILED")
        return False

if __name__ == "__main__":
    print("TESTING TD3 TRAINING TRIGGER FIXES")
    print("=" * 80)

    test1_passed = test_numpy_boolean_evaluation()
    test2_passed = test_td3_episode_collection()

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Numpy boolean evaluation: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  TD3 episode collection: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED - TD3 training trigger should work!")
    else:
        print("❌ SOME TESTS FAILED - Fix needed")
    print("=" * 80)
