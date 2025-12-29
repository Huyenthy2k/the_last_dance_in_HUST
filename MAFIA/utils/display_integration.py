"""
Display Integration Layer (Unified Text Logger)
===============================================
Standardized text-based logger for MAFIA (Observer & TD3).
Replaces legacy LiveDisplay with detailed, step-by-step console logging.
"""

import os
import sys
import time
import atexit
from typing import Any, Dict, List, Optional

# Remove LiveDisplay imports

# Global state
_display_enabled = True 
_log_file_path: Optional[str] = None
_logger_instance = None
_last_logged_wf_iteration = -1  # Track Walk-Forward iteration to prevent spam


def setup_display(
    enabled: bool = True,
    log_file: Optional[str] = None,
    run_id: str = "",
    total_epochs: int = 50,
    window: int = 0,
    seed: int = 2025,
) -> "UnifiedLogger":
    """Setup and initialize the unified text logger."""
    global _display_enabled, _log_file_path, _logger_instance

    _display_enabled = enabled
    _log_file_path = log_file

    logger = UnifiedLogger(log_file=log_file)
    logger.update(
        total_epochs=total_epochs,
        window=window,
        seed=seed,
    )
    _logger_instance = logger
    return logger

def get_display():
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = UnifiedLogger()
    return _logger_instance

def smart_print(msg: str, flush: bool = False):
    """Pass-through to logger if available, else print."""
    display = get_display()
    if display:
        display.log(msg)
    else:
        print(msg, flush=flush)


class UnifiedLogger:
    """
    A unified text-based logger that prints detailed, step-by-step metrics 
    for both Observer (Batch/Epoch) and TD3 (Update/Summary).
    """
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.state = {}
        self.last_epoch = -1
        self.last_step = -1
        self.last_td3_update = -1
        self.enabled = True 

    def update(self, **kwargs):
        self.state.update(kwargs)
        
        # Observer Batch Update Logic
        step = self.state.get('step', -1)
        # Check if we have sample_details which implies Observer batch context
        if 'sample_details' in self.state and step != self.last_step and step >= 0:
            self._print_batch_update()
            self.last_step = step
            
        # TD3 Batch Update Logic
        td3_updates = self.state.get('td3_updates', -1)
        if td3_updates != self.last_td3_update and td3_updates > 0:
            self._print_td3_update()
            self.last_td3_update = td3_updates

    def initialize(self): pass
    def cleanup(self): pass
    def capture_stdout(self): pass
    def release_stdout(self): pass

    def log(self, msg, level="INFO"):
        if level == "INFO":
            print(f"{msg}")
        else:
            print(f"[{level}] {msg}")

    def add_event(self, type, msg):
        print(f"[{type}] {msg}")

    def render(self, force=False):
        # Handle End-of-Epoch Summaries
        epoch = self.state.get('epoch', 0)
        if epoch != self.last_epoch and epoch > 0:
            self._print_epoch_summary()
            self.last_epoch = epoch

    def _print_batch_update(self):
        s = self.state
        if 'sample_details' not in s: return
        d = s['sample_details']
        
        epoch = s.get('epoch', 0)
        batch = s.get('batch', 0)
        step_str = f"{d.get('step_t',0)+1}/{d.get('total_steps_in_traj',0)}"
        trig = d.get('trigger', 'N/A')
        icon = '⏸' if 'Hold' in trig else '🔄'
        
        print(f"  ⏱  [Epoch {epoch}] Batch {batch+1} | Step {step_str} | {icon}  {trig.upper()}  [Sample 1/{s.get('batch_size', '?')} trajectories]")
        print(f"     Trigger: {d.get('trigger_details', 'N/A')}")
        print(f"     📈 Market:    RawReturn={d.get('market_raw_return', 0):+.4f} | CtxDiff={d.get('ctx_diff', 0):.3f}")
        
        t_pen = d.get('cost_turn_pen', 0)
        s_pen = d.get('cost_symdiff_pen', 0)
        tp_str = f"{t_pen:.4f}" if t_pen > 0 else "N/A (HOLD)"
        sp_str = f"{s_pen:.4f}" if s_pen > 0 else "N/A (HOLD)"
        print(f"     💸 Costs:     TurnPen={tp_str} | SymDiffPen={sp_str}")
        
        print(f"     ⚖️  Outcome:   R_total={d.get('score_r_total',0):+.4f} | R_baseline={d.get('score_r_baseline',0):+.4f} | Advantage={d.get('score_advantage',0):+.4f} | R_hold={d.get('score_r_hold',0):+.4f}")
        print(f"     📉 Losses:    L_PG={d.get('loss_pg',0):.4f} | L_risk={d.get('loss_risk',0):.4f} | L_dir={d.get('loss_dir',0):.4f} | L_bal={d.get('loss_bal',0):.4f}")
        
        tickers = d.get('portfolio_tickers', [])
        ticker_str = ", ".join(tickers[:5]) + ("..." if len(tickers)>5 else "")
        print(f"     📊 Portfolio: Held={int(d.get('portfolio_held_count',0))} | {ticker_str}")
        
        dir_map = {0: "Bear", 1: "Side", 2: "Bull"}
        pred_cls = dir_map.get(d.get('forecast_dir_pred'), '?')
        gt_cls = dir_map.get(d.get('forecast_dir_gt'), '?')
        match = "✅" if pred_cls == gt_cls else "❌"
        probs = d.get('forecast_dir_probs', [0,0,0])
        prob_str = f"[Bear={probs[0]:.2f}, Side={probs[1]:.2f}, Bull={probs[2]:.2f}]"
        
        print(f"     🔮 Forecast:  Risk(η)={d['forecast_risk_eta']:.2f}(Tg={d['forecast_risk_target']:.2f}) | Dir={pred_cls} (GT: {gt_cls}){match} {prob_str}")
        
        gw = d.get('gate_weights', {})
        gw_str = ", ".join([f"{k}={v*100:.1f}%" for k,v in gw.items()]) if gw else "(No Gate Info)"
        print(f"     🧩 Gate:      {gw_str}")
        print("")

    def _print_td3_update(self):
        """Detailed TD3 Update Log matching Observer format."""
        s = self.state
        updates = s.get('td3_updates', 0)
        actor_loss = s.get('td3_actor_loss', 0.0)
        critic_loss = s.get('td3_critic_loss', 0.0)
        reward = s.get('td3_reward', 0.0)
        buffer_size = s.get('td3_buffer_size', 0)
        
        # Trying to mock "Batch" concept via updates count or similar.
        # Since TD3 updates continuously, we can treat each log as a "Batch"
        print(f"  ⏱  [TD3 Training] Update {updates} | Buffer {buffer_size:,} | 🔄 REGIMES: {s.get('last_regime_to', 'N/A')}")
        print(f"     📉 Actor Loss:  {actor_loss:+.6f} (Policy Optimization)")
        print(f"     📉 Critic Loss: {critic_loss:+.6f} (Q-Value Error)")
        print(f"     💰 Batch Reward: {reward:+.6f} (Avg sampled from Replay Buffer)")
        
        # If we had action distribution stats, we'd print them here.
        # For now, print Top-K info if available in state
        if 'td3_topk_display' in s:
             print(f"     🧩 Top-K Focus: {s['td3_topk_display']}")
             
        print("")

    def _print_epoch_summary(self):
        s = self.state
        print(f"\n{'='*100}")
        print(f"📊 EPOCH {s.get('epoch', 0)} SUMMARY")
        print(f"{'-'*100}")
        print(f"  📈 Returns:     Mean={s.get('epoch_return', 0):+.4f} | Std={s.get('volatility', 0):.4f}")
        print(f"  💰 Net Reward:  Mean={s.get('obs_r_sel_total', 0):+.4f}")
        print(f"  ⚖️  Sharpe:      {s.get('sharpe', 0):.4f}")
        dir_acc = s.get('win_rate', 0) * 100 
        print(f"  🎯 Direction:   Accuracy={dir_acc:.1f}%")
        print(f"  📉 Risk η:      Mean={s.get('obs_eta_pred', 0):.3f}")
        print(f"{'-'*100}")
        print(f"  📉 Losses:")
        print(f"    ├─ Total:     {s.get('obs_loss_total', 0):.6f}")
        print(f"    ├─ PG:        {s.get('obs_loss_pg', 0):.6f}")
        print(f"    ├─ Risk:      {s.get('obs_loss_eta', 0):.6f}")
        print(f"    └─ Direction: {s.get('obs_loss_dir', 0):.6f}")
        print(f"{'-'*100}")
        if s.get('td3_updates', 0) > 0:
             print(f"  🤖 TD3 Status: {s.get('td3_updates')} updates | Avg Reward: {s.get('td3_reward', 0):.4f}")
        print(f"{'='*100}\n")

# Placeholder for cleanup functions
def _cleanup_display_at_exit(display): pass
def cleanup_display(): pass


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
    pg_r_total: float = 0.0,  # Total shaped reward sum
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
        display.update(
            training_mode="OBSERVER_ONLY",
            observer_frozen=False,  # Observer is TRAINING
            td3_frozen=True,  # TD3 is FROZEN
            cbf_enabled=False,  # CBF disabled
            observer_checkpoint_source="",
            phase="OBSERVER_PRETRAIN",
        )
        msg = (
            "\n"
            "╔══════════════════════════════════════════════════════════════════════════════╗\n"
            "║ 🔮 PHASE 1A: MACRO SPECIALIST TRAINING (Observer Only)                       ║\n"
            "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            "║ • Goal:      Train Grid-Trader Backbone + Direction/Risk Heads               ║\n"
            "║ • Frozen:    Selection Head (Top-K) & TD3 Agent                              ║\n"
            "║ • Strategy:  Iterative Expanding Window (Quarterly Updates)                  ║\n"
            "╚══════════════════════════════════════════════════════════════════════════════╝\n"
        )
        display.log(msg)
        display.add_event("EPOCH", "🔮 Phase 1: Observer Walk-Forward Training")

    else:  # RL_ONLY / SELECTION_ONLY logic (Phase 2)
        # Phase 2: Selection Specialist (RL) or TD3
        display.update(
            training_mode="RL_ONLY",
            observer_frozen=True,  # Observer is FROZEN (Static Expert)
            td3_frozen=False,  # TD3 is TRAINING
            cbf_enabled=True,  # CBF enabled
            observer_checkpoint_source=observer_checkpoint,
            phase="WARMUP",
            epoch=1,
            step=0,
        )
        obs_src = (
            observer_checkpoint[-40:]
            if len(observer_checkpoint) > 40
            else observer_checkpoint
        )
        msg = (
            "\n"
            "╔══════════════════════════════════════════════════════════════════════════════╗\n"
            "║ 🎯 PHASE 2: SELECTION SPECIALIST (Portfolio Manager)                         ║\n"
            "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            "║ • Goal:      Train Selection Head (Top-K) via Policy Gradient / TD3          ║\n"
            "║ • Frozen:    Backbone, Direction/Risk Heads (Loaded from Phase 1)            ║\n"
            "║ • Source:    {:<63} ║\n"
            "╚══════════════════════════════════════════════════════════════════════════════╝\n"
        ).format(obs_src or "None")
        display.log(msg)
        display.add_event(
            "EPOCH", f"🎯 Phase 2: TD3 Training (Observer: {obs_src or 'N/A'})"
        )
    
    display.render(force=True)


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



