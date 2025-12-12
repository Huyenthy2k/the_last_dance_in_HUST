"""
Comprehensive Tests for LiveDisplay
====================================
Ensures:
1. Layout is FIXED at exactly FIXED_LINES (dynamically checked)
2. All values update correctly and reflect real-time data
3. No line jumping during training simulation
4. All display sections render properly
"""

import os
import sys
import time
import random
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.live_display import LiveDisplay, DisplayState, ANSI
from utils.display_integration import (
    setup_display,
    update_step,
    update_selection,
    update_observer,
    update_td3,
    update_returns,
    update_controller,
    update_reward_components,
    update_regime_shift,
    set_phase,
    get_display,
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name: str, passed: bool, error: str = ""):
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {error}")
            print(f"  ✗ {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nErrors:")
            for e in self.errors:
                print(f"  - {e}")
        return self.failed == 0


def test_fixed_layout():
    """Test that layout is exactly FIXED_LINES."""
    display = LiveDisplay(enabled=False)
    EXPECTED_LINES = display.FIXED_LINES

    print("\n" + "="*60)
    print(f"TEST 1: Fixed Layout ({EXPECTED_LINES} lines)")
    print("="*60)

    results = TestResults()

    # Test with default state
    lines = display._build_fixed_display()
    results.record(
        f"Default state produces {EXPECTED_LINES} lines",
        len(lines) == EXPECTED_LINES,
        f"Got {len(lines)} lines instead of {EXPECTED_LINES}"
    )

    # Test with various states
    test_cases = [
        ("Empty events", {"event_log": []}),
        ("Full events", {"event_log": ["Event1", "Event2", "Event3", "Event4"]}),
        ("CBF active", {"cbf_constraint_active": True, "cbf_action_before": "[0.1,0.2]", "cbf_action_after": "[0.08,0.18]"}),
        ("CBF inactive", {"cbf_constraint_active": False, "cbf_action_before": "", "cbf_action_after": ""}),
        ("Rebalance day", {"is_rebalance": True, "n_added": 5, "n_removed": 3}),
        ("Hold day", {"is_rebalance": False, "n_added": 0, "n_removed": 0}),
        ("High PG reward", {"obs_r_sel_total": 0.5, "obs_r_sel_return": 0.4, "obs_r_sel_div": 0.1}),
        ("Negative PG reward", {"obs_r_sel_total": -0.3, "obs_r_sel_return": -0.2, "obs_r_sel_div": -0.1}),
    ]

    for name, state_updates in test_cases:
        display = LiveDisplay(enabled=False)
        for key, value in state_updates.items():
            setattr(display.state, key, value)
        lines = display._build_fixed_display()
        results.record(
            f"{name} produces {EXPECTED_LINES} lines",
            len(lines) == EXPECTED_LINES,
            f"Got {len(lines)} lines"
        )

    return results.summary()


def test_line_width():
    """Test that all lines have consistent width."""
    print("\n" + "="*60)
    print("TEST 2: Consistent Line Width")
    print("="*60)

    results = TestResults()
    display = LiveDisplay(enabled=False)

    # Set various states
    display.state.phase = "TRAIN"
    display.state.cbf_action_before = "[0.10, 0.15, 0.08, 0.12, 0.18, 0.09, 0.11, 0.14, 0.16, 0.13]"
    display.state.cbf_action_after = "[0.08, 0.12, 0.06, 0.10, 0.15, 0.07, 0.09, 0.11, 0.13, 0.10]"
    display.state.cbf_action_norm_before = 0.4567
    display.state.cbf_action_norm_after = 0.3654

    lines = display._build_fixed_display()

    # Check each line width (visible characters)
    for i, line in enumerate(lines):
        visible = display._strip_ansi(line)
        if visible.strip():  # Non-empty lines
            width = len(visible)
            # Allow empty padding lines and box borders
            is_valid = width <= display.FIXED_WIDTH or visible == ""
            results.record(
                f"Line {i+1} width <= {display.FIXED_WIDTH}",
                is_valid,
                f"Width is {width}: '{visible[:50]}...'" if not is_valid else ""
            )

    return results.summary()


def test_value_updates():
    """Test that all values update correctly."""
    print("\n" + "="*60)
    print("TEST 3: Value Updates")
    print("="*60)

    results = TestResults()
    display = LiveDisplay(enabled=False)

    # Test numeric updates
    test_values = {
        "epoch": 25,
        "step": 150,
        "total_steps": 252,
        "direction": 2,  # Bull
        "trend_z": 1.567,
        "eta_risk": 0.789,
        "volatility": 0.0234,
        "regime_shift_count": 5,
        "sharpe": 1.234,
        "mdd": -8.567,
        "win_rate": 0.612,
        "obs_loss_total": 0.0345,
        "obs_loss_dir": 0.0123,
        "obs_loss_eta": 0.0089,
        "obs_loss_pg": 0.0133,
        "obs_r_sel_total": 0.0456,
        "obs_r_sel_return": 0.0312,
        "obs_r_sel_div": 0.0144,
        "td3_actor_loss": 0.000234,
        "td3_critic_loss": 0.001567,
        "td3_reward": 15.678,
        "td3_buffer_size": 98765,
        "cbf_alpha": 0.123,
        "cbf_interventions": 42,
        "cbf_sigma_base": 0.15,
        "cbf_sigma_current": 0.11,
        "cbf_action_norm_before": 0.3456,
        "cbf_action_norm_after": 0.2789,
    }

    display.update(**test_values)

    for key, expected in test_values.items():
        actual = getattr(display.state, key)
        results.record(
            f"{key} = {expected}",
            actual == expected,
            f"Got {actual}"
        )

    return results.summary()


def test_display_integration():
    """Test display_integration functions update state correctly."""
    print("\n" + "="*60)
    print("TEST 4: Display Integration Functions")
    print("="*60)

    results = TestResults()

    # Create display and manually enable (but don't initialize to avoid terminal writes)
    display = LiveDisplay(enabled=False)
    display.enabled = True  # Enable to allow integration functions to work
    display._initialized = False  # Don't initialize terminal
    display.render = lambda force=False: None  # Mock render to avoid terminal output

    # Manually set as global display
    import utils.live_display as ld_module
    ld_module._display = display

    # Test update_observer with PG rewards
    update_observer(
        loss=0.0345,
        loss_eta=0.0089,
        loss_dir=0.0123,
        loss_sel=0.0133,
        eta_pred=0.85,
        dir_pred="BULL",
        dir_conf=0.78,
        samples_dir=1250,
        samples_risk=980,
        samples_sel=450,
        pg_r_return=0.0312,
        pg_r_diversity=0.0089,
        pg_r_total=0.0401,
    )

    results.record("Observer loss_total", display.state.obs_loss_total == 0.0345, f"Got {display.state.obs_loss_total}")
    results.record("Observer loss_eta", display.state.obs_loss_eta == 0.0089, f"Got {display.state.obs_loss_eta}")
    results.record("Observer loss_dir", display.state.obs_loss_dir == 0.0123, f"Got {display.state.obs_loss_dir}")
    results.record("Observer loss_pg", display.state.obs_loss_pg == 0.0133, f"Got {display.state.obs_loss_pg}")
    results.record("Observer eta_pred", display.state.obs_eta_pred == 0.85, f"Got {display.state.obs_eta_pred}")
    results.record("Observer dir_pred", display.state.obs_dir_pred == "BULL", f"Got {display.state.obs_dir_pred}")
    results.record("Observer dir_conf", display.state.obs_dir_conf == 0.78, f"Got {display.state.obs_dir_conf}")
    results.record("PG r_sel_return", display.state.obs_r_sel_return == 0.0312, f"Got {display.state.obs_r_sel_return}")
    results.record("PG r_sel_div", display.state.obs_r_sel_div == 0.0089, f"Got {display.state.obs_r_sel_div}")
    results.record("PG r_sel_total", display.state.obs_r_sel_total == 0.0401, f"Got {display.state.obs_r_sel_total}")

    # Test update_controller with from→to values
    update_controller(
        cbf_enabled=True,
        cbf_alpha=0.1,
        cbf_interventions=15,
        cbf_last_intervention="High volatility",
        cbf_safety_margin=0.0234,
        cbf_constraint_active=True,
        cbf_sigma_base=0.15,
        cbf_sigma_current=0.12,
        cbf_min_variance=0.01,
        cbf_max_position=0.25,
        cbf_adjust_direction="↓",
        cbf_adjust_magnitude=-0.03,
        cbf_action_before="[0.10, 0.15, 0.08]",
        cbf_action_after="[0.08, 0.12, 0.06]",
        cbf_action_norm_before=0.2892,
        cbf_action_norm_after=0.2314,
    )

    results.record("CBF enabled", display.state.cbf_enabled == True, f"Got {display.state.cbf_enabled}")
    results.record("CBF alpha", display.state.cbf_alpha == 0.1, f"Got {display.state.cbf_alpha}")
    results.record("CBF interventions", display.state.cbf_interventions == 15, f"Got {display.state.cbf_interventions}")
    results.record("CBF sigma_base", display.state.cbf_sigma_base == 0.15, f"Got {display.state.cbf_sigma_base}")
    results.record("CBF sigma_current", display.state.cbf_sigma_current == 0.12, f"Got {display.state.cbf_sigma_current}")
    results.record("CBF action_before", display.state.cbf_action_before == "[0.10, 0.15, 0.08]", f"Got {display.state.cbf_action_before}")
    results.record("CBF action_after", display.state.cbf_action_after == "[0.08, 0.12, 0.06]", f"Got {display.state.cbf_action_after}")
    results.record("CBF norm_before", display.state.cbf_action_norm_before == 0.2892, f"Got {display.state.cbf_action_norm_before}")
    results.record("CBF norm_after", display.state.cbf_action_norm_after == 0.2314, f"Got {display.state.cbf_action_norm_after}")

    # Test update_td3
    update_td3(
        actor_loss=0.000234,
        critic_loss=0.001567,
        buffer_size=85000,
        buffer_max=156000,
        updates=3500,
        reward=12.34,
        mean_q=5.67,
        lr=0.0001,
        noise_sigma=0.15,
    )

    results.record("TD3 actor_loss", display.state.td3_actor_loss == 0.000234, f"Got {display.state.td3_actor_loss}")
    results.record("TD3 critic_loss", display.state.td3_critic_loss == 0.001567, f"Got {display.state.td3_critic_loss}")
    results.record("TD3 buffer_size", display.state.td3_buffer_size == 85000, f"Got {display.state.td3_buffer_size}")
    results.record("TD3 reward", display.state.td3_reward == 12.34, f"Got {display.state.td3_reward}")

    # Test update_reward_components
    update_reward_components(
        reward_return=0.0856,
        reward_js=-0.0123,
        reward_total=7.33,
        reward_unscaled=0.0733,
        js_divergence=0.0245,
        w_return=1.0,
        lambda_js=0.1,
        reward_scale=100.0,
    )

    results.record("TD3 r_return", display.state.td3_r_return == 0.0856, f"Got {display.state.td3_r_return}")
    results.record("TD3 r_js", display.state.td3_r_js == -0.0123, f"Got {display.state.td3_r_js}")
    results.record("TD3 js_divergence", display.state.td3_js_divergence == 0.0245, f"Got {display.state.td3_js_divergence}")

    return results.summary()


def test_training_simulation():
    """Simulate a full training run and verify layout stays fixed."""
    print("\n" + "="*60)
    print("TEST 5: Training Simulation (Layout Stability)")
    print("="*60)

    results = TestResults()
    display = LiveDisplay(enabled=False)

    # Simulate training phases
    phases = ["PRETRAIN", "TRAIN", "VALID", "TEST"]
    epochs_per_phase = 3
    steps_per_epoch = 50

    line_counts = []

    for phase in phases:
        display.state.phase = phase

        for epoch in range(epochs_per_phase):
            display.state.epoch = epoch + 1
            display.state.total_epochs = epochs_per_phase * len(phases)

            for step in range(steps_per_epoch):
                # Simulate random state updates
                display.state.step = step
                display.state.total_steps = steps_per_epoch
                display.state.date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                display.state.direction = random.randint(0, 2)
                display.state.trend_z = random.uniform(-2, 2)
                display.state.eta_risk = random.uniform(0.5, 1.5)
                display.state.volatility = random.uniform(0.01, 0.05)
                display.state.is_rebalance = random.random() < 0.1
                display.state.n_added = random.randint(0, 5) if display.state.is_rebalance else 0
                display.state.n_removed = random.randint(0, 5) if display.state.is_rebalance else 0
                display.state.n_kept = 10 - display.state.n_removed

                # Observer updates
                display.state.obs_loss_total = random.uniform(0, 0.1)
                display.state.obs_loss_dir = random.uniform(0, 0.05)
                display.state.obs_loss_eta = random.uniform(0, 0.03)
                display.state.obs_loss_pg = random.uniform(0, 0.02)
                display.state.obs_r_sel_total = random.uniform(-0.1, 0.2)
                display.state.obs_r_sel_return = random.uniform(-0.05, 0.15)
                display.state.obs_r_sel_div = random.uniform(-0.02, 0.05)

                # TD3 updates
                display.state.td3_buffer_size = min(step * 100 + epoch * 5000, 156000)
                display.state.td3_actor_loss = random.uniform(0, 0.001)
                display.state.td3_critic_loss = random.uniform(0, 0.005)
                display.state.td3_reward = random.uniform(-10, 50)

                # CBF updates - sometimes with adjustment, sometimes without
                if random.random() < 0.3:
                    display.state.cbf_constraint_active = True
                    display.state.cbf_action_before = f"[{','.join([f'{random.uniform(0.05, 0.2):.2f}' for _ in range(5)])}]"
                    display.state.cbf_action_after = f"[{','.join([f'{random.uniform(0.03, 0.18):.2f}' for _ in range(5)])}]"
                    display.state.cbf_action_norm_before = random.uniform(0.2, 0.4)
                    display.state.cbf_action_norm_after = random.uniform(0.15, 0.35)
                    display.state.cbf_adjust_direction = random.choice(["↑", "↓"])
                    display.state.cbf_adjust_magnitude = random.uniform(-0.05, 0.05)
                else:
                    display.state.cbf_constraint_active = False
                    display.state.cbf_action_before = ""
                    display.state.cbf_action_after = ""
                    display.state.cbf_adjust_direction = ""

                # Build display and count lines
                lines = display._build_fixed_display()
                line_counts.append(len(lines))

    # Verify all renders produced exactly FIXED_LINES
    expected = display.FIXED_LINES
    all_expected = all(c == expected for c in line_counts)
    unique_counts = set(line_counts)

    results.record(
        f"All {len(line_counts)} renders produced {expected} lines",
        all_expected,
        f"Unique line counts: {unique_counts}"
    )

    # Verify no variation
    results.record(
        "Zero line count variation",
        len(unique_counts) == 1,
        f"Found {len(unique_counts)} different line counts: {unique_counts}"
    )

    return results.summary()


def test_ansi_stripping():
    """Test that ANSI codes are properly stripped for width calculation."""
    print("\n" + "="*60)
    print("TEST 6: ANSI Code Stripping")
    print("="*60)

    results = TestResults()
    display = LiveDisplay(enabled=False)

    test_cases = [
        (f"{ANSI.RED}Hello{ANSI.RESET}", "Hello"),
        (f"{ANSI.BOLD}{ANSI.GREEN}Test{ANSI.RESET}", "Test"),
        (f"Normal text", "Normal text"),
        (f"{ANSI.DIM}Dimmed{ANSI.RESET} and normal", "Dimmed and normal"),
        (f"{ANSI.BRIGHT_YELLOW}Bright{ANSI.RESET}", "Bright"),
    ]

    for ansi_text, expected in test_cases:
        stripped = display._strip_ansi(ansi_text)
        results.record(
            f"Strip '{expected}'",
            stripped == expected,
            f"Got '{stripped}'"
        )

    return results.summary()


def test_extreme_values():
    """Test display with extreme values."""
    print("\n" + "="*60)
    print("TEST 7: Extreme Values")
    print("="*60)

    results = TestResults()
    display = LiveDisplay(enabled=False)

    # Test with extreme values
    extreme_cases = [
        ("Very high reward", {"td3_reward": 99999.9999}),
        ("Very negative reward", {"td3_reward": -99999.9999}),
        ("Zero everything", {"td3_reward": 0, "sharpe": 0, "mdd": 0}),
        ("Large buffer", {"td3_buffer_size": 999999999}),
        ("Long intervention text", {"cbf_last_intervention": "A" * 100}),
        ("Long action vector", {"cbf_action_before": "[" + ",".join(["0.12345"] * 50) + "]"}),
        ("High precision floats", {"obs_loss_total": 0.123456789012345}),
        ("Many events", {"event_log": [f"Event {i}" for i in range(100)]}),
    ]

    expected_lines = LiveDisplay(enabled=False).FIXED_LINES

    for name, state_updates in extreme_cases:
        display = LiveDisplay(enabled=False)
        for key, value in state_updates.items():
            setattr(display.state, key, value)

        try:
            lines = display._build_fixed_display()
            results.record(
                f"{name} renders {expected_lines} lines",
                len(lines) == expected_lines,
                f"Got {len(lines)} lines"
            )
        except Exception as e:
            results.record(f"{name} renders without error", False, str(e))

    return results.summary()


def test_section_content():
    """Test that each section contains expected content."""
    print("\n" + "="*60)
    print("TEST 8: Section Content Verification")
    print("="*60)

    results = TestResults()
    display = LiveDisplay(enabled=False)

    # Set identifiable values
    display.state.seed = 9999
    display.state.epoch = 42
    display.state.step = 123
    display.state.date = "2025-12-25"
    display.state.obs_r_sel_total = 0.1234
    display.state.cbf_interventions = 777
    display.state.td3_buffer_size = 88888
    display.state.sharpe = 2.345

    lines = display._build_fixed_display()
    full_text = "\n".join([display._strip_ansi(l) for l in lines])

    # Check for key values in output
    checks = [
        ("Seed:9999", "seed value"),
        ("42/", "epoch number"),
        ("123", "step number"),
        ("2025-12-25", "date"),
        ("0.1234", "PG reward"),
        ("777", "CBF interventions"),
        ("88888", "TD3 buffer size"),
        ("2.345", "sharpe ratio"),
    ]

    for expected, description in checks:
        found = expected in full_text
        results.record(
            f"Contains {description} ({expected})",
            found,
            "Not found in output"
        )

    return results.summary()


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("LIVE DISPLAY COMPREHENSIVE TESTS")
    print("="*60)

    all_passed = True

    all_passed &= test_fixed_layout()
    all_passed &= test_line_width()
    all_passed &= test_value_updates()
    all_passed &= test_display_integration()
    all_passed &= test_training_simulation()
    all_passed &= test_ansi_stripping()
    all_passed &= test_extreme_values()
    all_passed &= test_section_content()

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
