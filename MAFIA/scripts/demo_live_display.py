#!/usr/bin/env python3
"""
Demo script for LiveDisplay - Run this to test fixed layout behavior.

Usage:
    python agents/MAFIA/scripts/demo_live_display.py

This will show the LiveDisplay updating in real-time for 30 seconds,
demonstrating that the layout stays fixed without jumping.
"""
import sys
import os
import time
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.live_display import LiveDisplay, init_display
from utils.display_integration import (
    setup_display, cleanup_display,
    update_step, update_observer, update_td3, update_returns,
    update_controller, update_reward_components, update_regime_shift,
    set_phase,
)


def run_demo():
    """Run a demo of the LiveDisplay."""
    print("\n" + "=" * 60)
    print("MAFIA LiveDisplay Demo")
    print("=" * 60)
    print("\nThis demo will show the display updating for 30 seconds.")
    print("The layout should remain FIXED at 64 lines without jumping.\n")
    print("Press Ctrl+C to stop early.\n")
    time.sleep(2)

    # Initialize display
    log_file = "/tmp/mafia_display_demo.log"
    display = setup_display(
        enabled=True,
        log_file=log_file,
        run_id="demo",
        total_epochs=50,
        window=0,
        seed=2025,
    )

    try:
        start_time = time.time()
        total_steps = 252
        phases = ["PRETRAIN", "TRAIN", "VALID", "TEST"]
        current_phase_idx = 0

        for step in range(1, total_steps + 1):
            # Simulate phase changes
            if step == 50:
                current_phase_idx = 1  # TRAIN
                set_phase("TRAIN", epoch=1, total_steps=total_steps)
            elif step == 180:
                current_phase_idx = 2  # VALID
                set_phase("VALID", epoch=1, total_steps=total_steps)
            elif step == 220:
                current_phase_idx = 3  # TEST
                set_phase("TEST", epoch=1, total_steps=total_steps)

            # Update step info
            update_step(
                step=step,
                date=f"2024-{(step % 12) + 1:02d}-{(step % 28) + 1:02d}",
                direction=random.choice([0, 1, 2]),
                trend_z=random.uniform(-2, 2),
                eta_risk=random.uniform(0.5, 1.5),
                volatility=random.uniform(0.01, 0.05),
                regime_shift_count=step // 50,
                is_rebalance=(step % 15 == 0),
                days_since_rebal=step % 15,
                rebal_interval=15,
                top_k=10,
                capital=1_000_000 * (1 + random.uniform(-0.1, 0.2)),
                daily_return=random.uniform(-2, 3),
                cumul_return=random.uniform(-5, 20),
                n_kept=8,
                n_added=random.randint(0, 2),
                n_removed=random.randint(0, 2),
                rebalance_count=step // 15,
                sharpe=random.uniform(-0.5, 2.5),
                mdd=random.uniform(-15, 0),
                win_rate=random.uniform(0.4, 0.65),
                td3_reward=random.uniform(-1, 1),
                total_steps=total_steps,
                start_time=start_time,
            )

            # Update Observer
            update_observer(
                loss=random.uniform(0.1, 2.0),
                loss_eta=random.uniform(0.01, 0.5),
                loss_dir=random.uniform(0.1, 1.0),
                loss_sel=random.uniform(0.01, 0.3),
                rebalance_ratio=10.0,
                eta_pred=random.uniform(0.8, 1.2),
                dir_pred=random.choice(["BEAR", "FLAT", "BULL"]),
                dir_conf=random.uniform(0.5, 0.95),
                samples_dir=step * 10,
                samples_risk=step * 8,
                samples_sel=step * 5,
                pg_r_return=random.uniform(-0.5, 0.5),
                pg_r_diversity=random.uniform(0, 0.2),
                pg_r_total=random.uniform(-0.3, 0.7),
            )

            # Update TD3
            update_td3(
                actor_loss=random.uniform(0.001, 0.1),
                critic_loss=random.uniform(0.01, 0.5),
                buffer_size=min(step * 100, 156000),
                buffer_max=156000,
                updates=step * 2,
                reward=random.uniform(-1, 2),
                mean_q=random.uniform(-10, 10),
                lr=0.0001,
                noise_sigma=0.15,
            )

            # Update Returns
            update_returns(
                epoch_return=random.uniform(-5, 15),
                cumul_return=random.uniform(-10, 30),
                sharpe=random.uniform(-0.5, 2.5),
                mdd=random.uniform(-20, 0),
                win_rate=random.uniform(0.45, 0.6),
                daily_return=random.uniform(-2, 3),
                net_profit=random.uniform(-50000, 200000),
                vol_max=random.uniform(10, 35),
                annual_return_pct=random.uniform(-10, 40),
            )

            # Update Controller CBF
            update_controller(
                cbf_enabled=True,
                cbf_alpha=0.1,
                cbf_interventions=step // 30,
                cbf_last_intervention="Risk limit" if step % 30 == 0 else "",
                cbf_safety_margin=random.uniform(-0.1, 0.1),
                cbf_constraint_active=(step % 20 < 5),
                cbf_sigma_base=0.15,
                cbf_sigma_current=0.15 + random.uniform(-0.02, 0.02),
                cbf_min_variance=0.01,
                cbf_max_position=0.25,
                cbf_adjust_direction=random.choice(["↑", "↓", ""]),
                cbf_adjust_magnitude=random.uniform(-0.01, 0.01),
                cbf_action_before="[0.10, 0.12, 0.08, ...]",
                cbf_action_after="[0.09, 0.11, 0.08, ...]",
                cbf_action_norm_before=0.45,
                cbf_action_norm_after=0.43,
            )

            # Update reward components
            r_return = random.uniform(-0.5, 1.0)
            r_js = random.uniform(-0.3, 0)
            update_reward_components(
                reward_return=r_return,
                reward_js=r_js,
                reward_total=(r_return + r_js) * 100,
                reward_unscaled=r_return + r_js,
                js_divergence=random.uniform(0, 0.1),
                w_return=1.0,
                lambda_js=0.1,
                reward_scale=100.0,
            )

            # Simulate regime shifts
            if step % 50 == 0:
                update_regime_shift(
                    date=f"2024-{(step % 12) + 1:02d}-{(step % 28) + 1:02d}",
                    reason="Trend reversal detected",
                    from_regime=random.choice(["BEAR", "FLAT"]),
                    to_regime=random.choice(["FLAT", "BULL"]),
                )

            # Short delay to see updates
            time.sleep(0.1)

            # Stop after 30 seconds
            if time.time() - start_time > 30:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cleanup_display()
        print("\n\nDemo complete!")
        print(f"Log file saved to: {log_file}")


if __name__ == "__main__":
    run_demo()
