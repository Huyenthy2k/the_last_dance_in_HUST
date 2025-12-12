"""
Display Integration Layer (Simplified)
======================================
Integrates simplified LiveDisplay with MAFIA training components.
"""

import os
import sys
import time
import atexit
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from utils.live_display import LiveDisplay, init_display, get_display, ANSI


# Check if LiveDisplay should be disabled via environment variable
_LIVE_DISPLAY_DISABLED = os.environ.get("MAFIA_NO_LIVE_DISPLAY", "0") == "1"

# Global state
_display_enabled = True and not _LIVE_DISPLAY_DISABLED
_log_file_path: Optional[str] = None
_suppress_prints = False

# Walk-forward iteration tracking (to prevent event log spam)
_last_logged_wf_iteration = -1


def setup_display(
    enabled: bool = True,
    log_file: Optional[str] = None,
    run_id: str = "",
    total_epochs: int = 50,
    window: int = 0,
    seed: int = 2025,
) -> LiveDisplay:
    """Setup and initialize the live display."""
    global _display_enabled, _log_file_path, _suppress_prints

    # Check env var override
    if _LIVE_DISPLAY_DISABLED:
        enabled = False

    _display_enabled = enabled
    _log_file_path = log_file
    _suppress_prints = enabled

    display = init_display(enabled=enabled, log_file=log_file)
    display.update(
        total_epochs=total_epochs,
        window=window,
        seed=seed,
    )

    if enabled:
        display.initialize()
        # Capture stdout/stderr to prevent breaking the display
        display.capture_stdout()
        # Enable print suppression
        enable_display_mode()
        atexit.register(_cleanup_display_at_exit, display)

    return display


def _cleanup_display_at_exit(display):
    """Cleanup function for atexit.

    Order is important:
    1. cleanup() - Clear display and position cursor correctly (while still capturing stdout)
    2. release_stdout() - Restore original stdout/stderr
    3. disable_display_mode() - Restore normal print behavior
    """
    display.cleanup()  # First: cleanup display while stdout is still captured
    display.release_stdout()  # Second: restore original stdout
    disable_display_mode()  # Last: restore normal print


def cleanup_display():
    """Cleanup the display on exit."""
    display = get_display()
    if display and display.enabled:
        display.cleanup()


def update_step(
    step: int,
    date: str,
    # Market
    direction: int = 1,  # 0=Bear, 1=Flat, 2=Bull
    trend_z: float = 0.0,
    eta_risk: float = 1.0,
    volatility: float = 0.0,
    regime_shift_count: int = 0,
    # Portfolio
    is_rebalance: bool = False,
    days_since_rebal: int = 0,
    rebal_interval: int = 10,
    top_k: int = 10,
    capital: float = 1_000_000,
    daily_return: float = 0.0,
    cumul_return: float = 0.0,
    # Selection
    n_kept: int = 10,
    n_added: int = 0,
    n_removed: int = 0,
    rebalance_count: int = 0,
    # Returns
    sharpe: float = 0.0,
    mdd: float = 0.0,
    win_rate: float = 0.5,
    # TD3
    td3_reward: float = 0.0,
    # Timing
    total_steps: int = 0,
    start_time: float = 0.0,
    **kwargs,  # Ignore extra args for compatibility
):
    """Update display with step information."""
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    # Calculate speed
    elapsed = time.time() - start_time if start_time > 0 else 0
    speed = step / elapsed if elapsed > 0 else 0

    # Build update dict - skip total_steps if 0 to preserve callback-set value
    update_kwargs = dict(
        date=date,
        direction=direction,
        trend_z=trend_z,
        eta_risk=eta_risk,
        volatility=volatility,
        regime_shift_count=regime_shift_count,
        is_rebalance=is_rebalance,
        days_since_rebal=days_since_rebal,
        rebal_interval=rebal_interval,
        top_k=top_k,
        capital=capital,
        daily_return=daily_return,
        cumul_return=cumul_return,
        n_kept=n_kept,
        n_added=n_added,
        n_removed=n_removed,
        rebalance_count=rebalance_count,
        sharpe=sharpe,
        mdd=mdd,
        win_rate=win_rate,
        td3_reward=td3_reward,
        speed=speed,
        elapsed_seconds=elapsed,
    )
    # Epoch-level progress: always track day-in-epoch here
    update_kwargs["epoch_day"] = max(0, step) + 1  # 1-based for display
    if total_steps > 0:
        update_kwargs["epoch_total_days"] = total_steps
    # Only update global step/total_steps when caller passes total_steps>0
    if total_steps > 0:
        update_kwargs["step"] = step
        update_kwargs["total_steps"] = total_steps

    display.update(**update_kwargs)
    if display.enabled:
        display.render()


def update_selection(
    step: int,
    date: str,
    trigger: str,
    kept: List[str],
    added: List[str],
    removed: List[str],
):
    """Update display with stock selection event."""
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    display.update(
        n_kept=len(kept),
        n_added=len(added),
        n_removed=len(removed),
        is_rebalance=True,
        last_selection_date=date,
        last_selection_trigger=trigger,
        last_kept_tickers=",".join(kept[:5]) + ("..." if len(kept) > 5 else ""),
        last_added_tickers=",".join(added[:3]) + ("..." if len(added) > 3 else ""),
        last_removed_tickers=",".join(removed[:3])
        + ("..." if len(removed) > 3 else ""),
    )

    # Add event to log
    added_str = f"+{','.join(added[:3])}" if added else ""
    removed_str = f"-{','.join(removed[:3])}" if removed else ""
    changes = " ".join(filter(None, [added_str, removed_str]))
    display.add_event("REBALANCE", f"{date} {trigger}: kept={len(kept)} {changes}")

    if display.enabled:
        display.render()

    # Log to file
    display.log(
        f"Selection: {date} | Trigger: {trigger} | "
        f"Kept: {len(kept)}, Added: {len(added)}, Removed: {len(removed)}"
    )
    if added:
        display.log(f"  Added: {', '.join(added)}")
    if removed:
        display.log(f"  Removed: {', '.join(removed)}")


def update_observer(
    loss: float = 0.0,
    loss_eta: float = 0.0,
    loss_dir: float = 0.0,
    loss_sel: float = 0.0,
    rebalance_ratio: float = 10.0,
    eta_pred: float = 1.0,
    dir_pred: str = "FLAT",
    dir_conf: float = 0.0,
    samples_dir: int = 0,
    samples_risk: int = 0,
    samples_sel: int = 0,
    # PG (Stock Selection) reward components
    # R_t = mean_return - α_turnover×turnover - α_change×symdiff
    pg_r_return: float = 0.0,  # mean_return component
    pg_r_turnover: float = 0.0,  # α_turnover × turnover penalty
    pg_r_change: float = 0.0,  # α_change × symdiff penalty
    pg_r_total: float = 0.0,  # Total shaped return (R_t)
    pg_r_diversity: float = 0.0,  # Legacy (kept for compat)
    **kwargs,  # Ignore extra args
):
    """Update display with Observer metrics."""
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    display.update(
        obs_loss_total=loss,
        obs_loss_eta=loss_eta,
        obs_loss_dir=loss_dir,
        obs_loss_pg=loss_sel,  # Map loss_sel to obs_loss_pg
        rebalance_ratio=rebalance_ratio,
        obs_eta_pred=eta_pred,
        obs_dir_pred=dir_pred,
        obs_dir_conf=dir_conf,
        obs_samples_dir=samples_dir,
        obs_samples_risk=samples_risk,
        obs_samples_sel=samples_sel,
        # PG reward components: R_t = mean_return - turnover - change
        obs_r_sel_return=pg_r_return,
        obs_r_sel_turnover=pg_r_turnover,
        obs_r_sel_change=pg_r_change,
        obs_r_sel_total=pg_r_total,
        obs_r_sel_div=pg_r_diversity,  # Legacy
    )
    if display.enabled:
        display.render()


# Track warmup progress across episodes (persistent)
_warmup_episode_count = 0
_warmup_samples_at_episode_start = 0


def update_observer_warmup(
    buffer_size: int,
    target_samples: int = 300,
    warmup_complete: bool = False,
    new_episode: bool = False,  # Set True when starting new episode
):
    """Update display with Observer warmup progress.

    Called during Phase 1 (OBSERVER_ONLY) while collecting initial samples
    before Observer training starts.

    Cross-episode tracking: Warmup samples persist across episode resets.
    Use new_episode=True to log episode boundary events.

    Args:
        buffer_size: Current number of samples in buffer
        target_samples: Minimum samples needed to start training (default 300)
        warmup_complete: True when warmup phase is done
        new_episode: True when this is first call after episode reset
    """
    global _warmup_episode_count, _warmup_samples_at_episode_start

    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    # Track episode boundaries for warmup
    if new_episode and not warmup_complete:
        _warmup_episode_count += 1
        samples_added = buffer_size - _warmup_samples_at_episode_start
        if _warmup_episode_count > 1 and samples_added > 0:
            display.log(
                f"[OBSERVER-WARMUP] Episode {_warmup_episode_count}: Carried over {buffer_size} samples from previous episode"
            )
        _warmup_samples_at_episode_start = buffer_size

    display.update(
        obs_warmup_samples=buffer_size,
        obs_warmup_target=target_samples,
        obs_warmup_complete=warmup_complete,
    )
    if display.enabled:
        display.render()

    # Log warmup progress to file
    if not warmup_complete:
        pct = (buffer_size / target_samples * 100) if target_samples > 0 else 0
        display.log(
            f"[OBSERVER-WARMUP] Buffer: {buffer_size}/{target_samples} ({pct:.1f}%)"
        )
    else:
        display.log(
            f"[OBSERVER-WARMUP] ✅ Complete! Buffer has {buffer_size} samples (across {_warmup_episode_count} episodes). Training started."
        )
        # Reset episode counter for next warmup phase
        _warmup_episode_count = 0
        _warmup_samples_at_episode_start = 0


def reset_performance_metrics(initial_capital: float = 0.0):
    """
    Clear performance metrics (capital/returns) when transitioning out of warmup.

    This prevents warmup noise (uniform/random actions) from polluting the
    performance board at the start of official training.
    """
    display = get_display()
    if not display:
        return
    display.update(
        capital=initial_capital,
        daily_return=0.0,
        cumul_return=0.0,
        sharpe=0.0,
        mdd=0.0,
        win_rate=0.0,
        annual_return_pct=0.0,
        net_profit=0.0,
    )
    if display.enabled:
        display.render(force=True)
    display.log(
        f"[RESET] Performance metrics cleared (capital set to {initial_capital:.2f})"
    )


def update_td3(
    actor_loss: float = 0.0,
    critic_loss: float = 0.0,
    buffer_size: int = 0,
    buffer_max: int = 156000,
    learning_starts: int = 1000,
    updates: int = 0,
    reward: float = 0.0,
    mean_q: float = 0.0,
    lr: float = 0.0001,
    noise_sigma: float = 0.15,
    **kwargs,  # Ignore extra args
):
    """Update display with TD3 metrics."""
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    display.update(
        td3_actor_loss=actor_loss,
        td3_critic_loss=critic_loss,
        td3_buffer_size=buffer_size,
        td3_buffer_max=buffer_max,
        td3_learning_starts=learning_starts,
        td3_updates=updates,
        td3_reward=reward,
        td3_mean_q=mean_q,
        td3_lr=lr,
        td3_noise_sigma=noise_sigma,
    )
    if display.enabled:
        display.render()


def update_returns(
    epoch_return: float = 0.0,
    cumul_return: float = 0.0,
    sharpe: float = 0.0,
    mdd: float = 0.0,
    win_rate: float = 0.5,
    daily_return: float = 0.0,
    # New performance metrics
    net_profit: float = 0.0,
    vol_max: float = 0.0,
    annual_return_pct: float = 0.0,
    **kwargs,  # Ignore extra args
):
    """Update display with returns metrics."""
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    display.update(
        epoch_return=epoch_return,
        cumul_return=cumul_return,
        sharpe=sharpe,
        mdd=mdd,
        win_rate=win_rate,
        daily_return=daily_return,
        net_profit=net_profit,
        vol_max=vol_max,
        annual_return_pct=annual_return_pct,
    )
    if display.enabled:
        display.render()


def update_controller(
    cbf_enabled: bool = True,
    cbf_alpha: float = 0.1,
    cbf_interventions: int = 0,
    cbf_last_intervention: str = "",
    cbf_safety_margin: float = 0.0,
    cbf_constraint_active: bool = False,
    # Detailed CBF parameters
    cbf_sigma_base: float = 0.15,
    cbf_sigma_current: float = 0.15,
    cbf_min_variance: float = 0.01,
    cbf_max_position: float = 0.25,
    cbf_adjust_direction: str = "",
    cbf_adjust_magnitude: float = 0.0,
    # CBF Action adjustment "from → to" values
    cbf_action_before: str = "",  # e.g., "[0.10, 0.15, ...]"
    cbf_action_after: str = "",  # e.g., "[0.08, 0.12, ...]"
    cbf_action_norm_before: float = 0.0,
    cbf_action_norm_after: float = 0.0,
    **kwargs,  # Ignore extra args
):
    """Update display with Controller CBF metrics."""
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    display.update(
        cbf_enabled=cbf_enabled,
        cbf_alpha=cbf_alpha,
        cbf_interventions=cbf_interventions,
        cbf_last_intervention=cbf_last_intervention,
        cbf_safety_margin=cbf_safety_margin,
        cbf_constraint_active=cbf_constraint_active,
        cbf_sigma_base=cbf_sigma_base,
        cbf_sigma_current=cbf_sigma_current,
        cbf_min_variance=cbf_min_variance,
        cbf_max_position=cbf_max_position,
        cbf_adjust_direction=cbf_adjust_direction,
        cbf_adjust_magnitude=cbf_adjust_magnitude,
        cbf_action_before=cbf_action_before,
        cbf_action_after=cbf_action_after,
        cbf_action_norm_before=cbf_action_norm_before,
        cbf_action_norm_after=cbf_action_norm_after,
    )
    if display.enabled:
        display.render()


def update_reward_components(
    reward_return: float = 0.0,
    reward_js: float = 0.0,
    reward_total: float = 0.0,
    reward_unscaled: float = 0.0,
    js_divergence: float = 0.0,
    w_return: float = 1.0,
    lambda_js: float = 0.1,
    reward_scale: float = 100.0,
    **kwargs,  # Ignore extra args
):
    """Update display with TD3 reward components for hyperparameter tuning."""
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    display.update(
        td3_r_return=reward_return,
        td3_r_js=reward_js,
        td3_reward=reward_total,
        td3_r_unscaled=reward_unscaled,
        td3_js_divergence=js_divergence,
        td3_w_return=w_return,
        td3_lambda_js=lambda_js,
        td3_reward_scale=reward_scale,
    )
    if display.enabled:
        display.render()


def update_regime_shift(
    date: str,
    reason: str,
    from_regime: str = "",
    to_regime: str = "",
    trigger_type: str = "",  # "direction_reversal", "vol_shock", "dc_trigger"
):
    """Update display with regime shift event.

    Args:
        date: Trading date
        reason: Detailed reason string
        from_regime: Previous regime (BEAR/FLAT/BULL)
        to_regime: Current regime (BEAR/FLAT/BULL)
        trigger_type: Type of trigger - determines display format:
            - "direction_reversal": Shows "SHIFT BEAR→BULL"
            - "vol_shock": Shows "⚡ VOL_SHOCK [BULL]"
            - "dc_trigger": Shows "⚡ DC_REVERSAL [BULL]"
    """
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    current_count = display.state.regime_shift_count
    display.update(
        regime_shift_count=current_count + 1,
        last_regime_date=date,
        last_regime_reason=reason,
        last_regime_from=from_regime,
        last_regime_to=to_regime,
    )

    # Format event based on trigger type (per spec)
    if trigger_type == "direction_reversal":
        # Direction change: Show from→to
        event_text = f"{date} SHIFT {from_regime}→{to_regime}"
    elif trigger_type == "vol_shock":
        # Vol shock: Show current direction only, with shock indicator
        event_text = f"{date} ⚡ VOL_SHOCK [{to_regime}]"
    elif trigger_type == "dc_trigger":
        # DC reversal: Show current direction only
        event_text = f"{date} ⚡ DC_REVERSAL [{to_regime}]"
    else:
        # Legacy fallback
        direction_info = (
            f"{from_regime}→{to_regime}" if from_regime and to_regime else ""
        )
        event_text = f"{date} SHIFT {direction_info}: {reason}"

    display.add_event("REGIME", event_text)

    if display.enabled:
        display.render()
    display.log(f"Regime shift: {trigger_type or 'unknown'} - {event_text}")


def set_phase(phase: str, epoch: int = 0, total_steps: int = 0):
    """Set the current training phase."""
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    old_phase = display.state.phase
    old_epoch = display.state.epoch

    # Build update dict - only reset step if phase or epoch actually changes
    update_kwargs = dict(
        phase=phase,
        epoch=epoch,
        is_rebalance=False,  # Reset rebalance flag for new phase
    )

    # Only update total_steps if provided (>0)
    if total_steps > 0:
        update_kwargs["total_steps"] = total_steps

    # Only reset step to 0 if epoch changes (new epoch starts)
    if epoch != old_epoch:
        update_kwargs["step"] = 0

    display.update(**update_kwargs)

    # Add event for phase/epoch change
    if phase != old_phase:
        display.add_event("EPOCH", f"Phase changed: {old_phase} → {phase}")
    elif epoch != old_epoch:
        display.add_event(
            "EPOCH", f"Epoch {epoch}/{display.state.total_epochs} started ({phase})"
        )

    if display.enabled:
        display.render(force=True)
    display.log(f"Phase: {phase}, Epoch: {epoch}")


def set_training_mode(
    mode: str,
    observer_checkpoint: str = "",
):
    """Set the 2-phase training mode.

    Args:
        mode: "OBSERVER_ONLY" for Phase 1 (Observer training, TD3 frozen)
              "RL_ONLY" for Phase 2 (TD3 training, Observer frozen)
        observer_checkpoint: Path to frozen Observer checkpoint (for Phase 2)

    Note: There is NO COMBINED mode - Observer and TD3 never train together.
    """
    display = get_display()
    if not display:
        return
    # Note: We update state even if display.enabled=False (no TTY),
    # because the web dashboard reads state via API regardless of terminal display.

    if mode == "OBSERVER_ONLY":
        # Phase 1: Observer training, TD3 frozen (uniform weights)
        # CBF is disabled because TD3 actions are just uniform weights, not learned
        display.update(
            training_mode="OBSERVER_ONLY",
            observer_frozen=False,  # Observer is TRAINING
            td3_frozen=True,  # TD3 is FROZEN
            cbf_enabled=False,  # CBF disabled (TD3 frozen, uniform weights)
            observer_checkpoint_source="",
            phase="OBSERVER_PRETRAIN",  # Set appropriate phase
        )
        display.add_event("EPOCH", "🔮 Phase 1: Observer Walk-Forward Training")
    else:  # RL_ONLY
        # Phase 2: TD3 training, Observer frozen (Static Expert)
        display.update(
            training_mode="RL_ONLY",
            observer_frozen=True,  # Observer is FROZEN (Static Expert)
            td3_frozen=False,  # TD3 is TRAINING
            cbf_enabled=True,  # CBF enabled (TD3 learning real actions)
            observer_checkpoint_source=observer_checkpoint,
            phase="WARMUP",  # Start with warmup
            epoch=1,  # Phase 2 starts at epoch 1 (will be updated by callback)
            step=0,  # Reset step for new phase
        )
        obs_src = (
            observer_checkpoint[-40:]
            if len(observer_checkpoint) > 40
            else observer_checkpoint
        )
        display.add_event(
            "EPOCH", f"🎯 Phase 2: TD3 Training (Observer: {obs_src or 'N/A'})"
        )

    display.render(force=True)
    display.log(f"Training mode set: {mode}")


def update_walkforward_iteration(
    iteration: int,
    total_iterations: int,
    train_years: str = "",
    valid_year: int = 0,
    infer_year: int = 0,
    # Full date ranges for Expanding Window (optional, for detailed display)
    train_range: str = "",
    valid_range: str = "",
    infer_range: str = "",
    is_finetune: bool = False,
):
    """Update walk-forward iteration info (for Phase 1).

    Args:
        iteration: Current iteration (0-indexed)
        total_iterations: Total number of walk-forward iterations
        train_years: Training year range, e.g., "2015→2017" (legacy)
        valid_year: Validation year, e.g., 2017 (legacy)
        infer_year: Inference/test year, e.g., 2018 (legacy)
        train_range: Full train date range, e.g., "01/2015 → 06/2017"
        valid_range: Full valid date range, e.g., "07/2017 → 12/2017"
        infer_range: Full infer date range, e.g., "01/2018 → 12/2018"
        is_finetune: True if finetuning from previous checkpoint
    """
    global _last_logged_wf_iteration

    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    # Always update state (for dashboard API)
    display.update(
        walkforward_iteration=iteration,
        walkforward_total_iterations=total_iterations,
        walkforward_train_years=train_years,
        walkforward_valid_year=valid_year,
        walkforward_infer_year=infer_year,
        walkforward_train_range=train_range,
        walkforward_valid_range=valid_range,
        walkforward_infer_range=infer_range,
        walkforward_is_finetune=is_finetune,
    )

    # Only add event when iteration CHANGES (not on every step)
    # This fixes the bug where the event log was spammed every step
    if iteration != _last_logged_wf_iteration:
        _last_logged_wf_iteration = iteration
        # Use full ranges if available, otherwise fallback to year-only
        train_display = train_range or train_years
        valid_display = valid_range or str(valid_year)
        infer_display = infer_range or str(infer_year)
        finetune_tag = " [Finetune]" if is_finetune else " [Scratch]"
        display.add_event(
            "EPOCH",
            f"WF [{iteration + 1}/{total_iterations}]{finetune_tag}: Train {train_display}",
        )
        display.log(
            f"Walk-Forward iteration {iteration + 1}/{total_iterations}: "
            f"Train={train_display}, Valid={valid_display}, Infer={infer_display}"
        )

    display.render()


def update_walkforward_score(
    current_score: float,
    best_score: float = 0.0,
    best_epoch: int = 0,
):
    """Update walk-forward validation score (for Phase 1).

    Args:
        current_score: Current validation composite score
        best_score: Best score achieved so far
        best_epoch: Epoch that achieved the best score
    """
    display = get_display()
    if not display:
        return
    # Note: Update state even if display.enabled=False (for dashboard API)

    display.update(
        walkforward_current_score=current_score,
        walkforward_best_score=best_score,
        walkforward_best_epoch=best_epoch,
    )
    display.render()


def reset_walkforward_tracking():
    """Reset walk-forward iteration tracking state.

    Call this at the start of a new training run to ensure clean state.
    This prevents the first iteration event from being skipped if the
    module was already loaded from a previous run.
    """
    global _last_logged_wf_iteration
    _last_logged_wf_iteration = -1


def log_message(message: str, level: str = "INFO"):
    """Log a message to file only."""
    display = get_display()
    if display:
        display.log(message, level)


def log_epoch_end(epoch: int, metrics: Dict[str, Any]):
    """Log epoch-end summary to file."""
    display = get_display()
    if display:
        display.log(f"=== Epoch {epoch} Summary ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                display.log(f"  {key}: {value:.4f}")
            else:
                display.log(f"  {key}: {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Print Suppression System for LiveDisplay
# ═══════════════════════════════════════════════════════════════════════════════
# This system ensures ALL print statements go to the log file, not the terminal,
# when LiveDisplay is active. This is critical for maintaining the fixed layout.
# ═══════════════════════════════════════════════════════════════════════════════

# Store references to ORIGINAL print and stdout BEFORE any redirection
# Use sys.__stdout__ which Python guarantees is never redirected
import builtins

_original_print = builtins.print
_original_stdout = sys.__stdout__  # The REAL stdout, never redirected


def _display_aware_print(*args, **kwargs):
    """Print function that respects display mode.

    When LiveDisplay is active:
    - ALL prints are COMPLETELY SUPPRESSED from terminal
    - Prints are logged to the log file only (if available)
    - This ensures the display layout remains fixed

    CRITICAL: Never write to terminal when display is active!
    """
    display = get_display()
    live_display_active = display and display.enabled

    if not live_display_active:
        # Display not active, use normal print via original stdout
        # Use _original_print which goes to _original_stdout
        try:
            _original_print(*args, **kwargs)
        except Exception:
            pass
        return

    # ═══ LiveDisplay is ACTIVE - SUPPRESS ALL terminal output ═══
    # Only write to log file, NEVER to terminal

    if args:
        msg = " ".join(str(arg) for arg in args)
        end = kwargs.get("end", "\n")
        full_msg = msg + end

        # Method 1: Use display's log method (preferred)
        if display and hasattr(display, "log"):
            try:
                display.log(msg.rstrip())  # Remove trailing newline for log
                return
            except Exception:
                pass

        # Method 2: Write directly to log file if available
        if display and display.log_file:
            try:
                with open(display.log_file, "a", encoding="utf-8") as f:
                    f.write(full_msg)
                return
            except Exception:
                pass

    # If we get here, completely suppress output
    # This is intentional - we MUST keep the display fixed
    return


def enable_display_mode():
    """Enable display mode and suppress old print statements.

    This patches builtins.print to route all prints through _display_aware_print,
    which logs to file instead of terminal when LiveDisplay is active.
    """
    builtins.print = _display_aware_print


def disable_display_mode():
    """Disable display mode and restore normal printing."""
    builtins.print = _original_print


@contextmanager
def suppress_old_prints():
    """Context manager to temporarily suppress old print statements."""
    enable_display_mode()
    try:
        yield
    finally:
        disable_display_mode()


# ═══════════════════════════════════════════════════════════════════════════════
# SMART PRINT FUNCTION - To be imported by other modules for verbose logging
# ═══════════════════════════════════════════════════════════════════════════════


def smart_print(*args, verbose: bool = True, **kwargs):
    """Smart print function that routes verbose logs to file when appropriate.

    This function should be used by all modules that have verbose logging
    (tradeEnv.py, TD3_controller.py, controllers.py, callback_func.py, etc.)
    to ensure terminal layout remains fixed when LiveDisplay is active.

    Key behaviors:
    - If display.enabled=True (TTY available): Route to log file, suppress terminal
    - If display.enabled=False but log_file exists: Route to log file, suppress terminal
    - If no log file: Print normally (fallback)
    - If verbose=False: Always print (critical messages)
    - If MAFIA_QUIET_STARTUP=1: Suppress pre-training messages (LiveDisplay will be used)

    Args:
        *args: Arguments to print
        verbose: If False, always print (used for critical messages)
        **kwargs: Keyword arguments for print (flush, end, etc.)

    Usage:
        from utils.display_integration import smart_print
        smart_print("[ENV] day= 0 | η=1.0 | ...")  # Goes to log file when display active
        smart_print("[CRITICAL] Error!", verbose=False)  # Always prints
    """
    # Force flush=True for realtime logging unless explicitly disabled
    if "flush" not in kwargs:
        kwargs["flush"] = True

    # Suppress startup messages when LiveDisplay will be used (they get overwritten anyway)
    if os.environ.get("MAFIA_QUIET_STARTUP", "") == "1" and verbose:
        # During startup, suppress verbose output since LiveDisplay will overwrite it
        return

    display = get_display()

    # If verbose=False, always print (critical messages)
    if not verbose:
        _original_print(*args, **kwargs)
        return

    # Check if we should route to log file:
    # - Display is enabled (TTY mode with live display)
    # - OR display has a log file (even if TTY unavailable)
    should_log_only = (display and display.enabled) or (display and display.log_file)

    # If no display or no log file, print normally
    if not should_log_only:
        try:
            _original_print(*args, **kwargs)
        except Exception:
            pass
        return

    # ═══ Route to log file only - Suppress terminal output ═══
    if args:
        msg = " ".join(str(arg) for arg in args)
        end = kwargs.get("end", "\n")
        full_msg = msg + end

        # Method 1: Use display's log method (if enabled)
        if display and display.enabled and hasattr(display, "log"):
            try:
                display.log(msg.rstrip())
                return
            except Exception:
                pass

        # Method 2: Write directly to log file
        if display and display.log_file:
            try:
                with open(display.log_file, "a", encoding="utf-8") as f:
                    f.write(full_msg)
                return
            except Exception:
                pass


def is_display_active() -> bool:
    """Check if LiveDisplay is currently active.

    Returns True if display is enabled and running, False otherwise.
    Use this to conditionally skip expensive formatting for verbose logs.
    """
    display = get_display()
    return display is not None and display.enabled
