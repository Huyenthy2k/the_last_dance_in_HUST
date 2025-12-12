"""
Quick test to verify vol_window_size_20 bug fix.
Tests that the variable is defined before use in sample_trajectory_batch().
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_vol_window_size_20_defined():
    """
    Verify vol_window_size_20 is defined in sample_trajectory_batch code.
    """
    print("Testing vol_window_size_20 variable definition...")

    # Read the trainer file
    trainer_path = os.path.join(
        os.path.dirname(__file__), "..", "RL_controller", "observer_offline_trainer.py"
    )

    with open(trainer_path, "r") as f:
        lines = f.readlines()

    # Find the sample_trajectory_batch method
    method_start = -1
    for i, line in enumerate(lines):
        if "def sample_trajectory_batch" in line:
            method_start = i
            break

    assert method_start != -1, "Cannot find sample_trajectory_batch method!"

    # Search for vol_window_size_20 definition and usage
    definition_line = -1
    usage_lines = []

    for i in range(method_start, min(len(lines), method_start + 500)):
        line = lines[i]

        # Check for definition
        if "vol_window_size_20 = 20" in line or "vol_window_size_20 =" in line:
            definition_line = i
            print(f"  ✅ Found definition at line {i + 1}: {line.strip()}")

        # Check for usage (exclude definition line)
        if "vol_window_size_20" in line and i != definition_line:
            # Check if this is actual usage (appears on right side of expression)
            # Look for patterns like: range(vol_window_size_20, ...) or [...vol_window_size_20...]
            stripped = line.strip()
            if "range(" in stripped or "[" in stripped or "-" in stripped:
                usage_lines.append((i, line.strip()))

    # Verify definition exists
    assert definition_line != -1, (
        "vol_window_size_20 is not defined in sample_trajectory_batch!"
    )

    # Verify all usages come AFTER definition
    for usage_line, usage_text in usage_lines:
        assert usage_line > definition_line, (
            f"vol_window_size_20 used at line {usage_line + 1} BEFORE definition at line {definition_line + 1}!"
        )
        print(f"  ✅ Usage at line {usage_line + 1} comes after definition")

    print(f"  ✅ Found {len(usage_lines)} usages, all after definition")
    print("✅ vol_window_size_20 bug is FIXED!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing vol_window_size_20 Bug Fix")
    print("=" * 70)
    print()

    try:
        test_vol_window_size_20_defined()
        print()
        print("=" * 70)
        print("✅ TEST PASSED - Bug is fixed!")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        sys.exit(1)
