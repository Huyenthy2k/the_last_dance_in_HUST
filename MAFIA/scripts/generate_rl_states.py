#!/usr/bin/env python3
"""
Generate RL States from Frozen Observer

Runs a frozen observer through a year of data and saves pre-computed states
for use in Phase 2 RL training. Includes regime shift detection for dynamic
rebalancing signals.

Output format:
{
    'states': List[Dict],  # Each day's state with rebalance signals
    'topk_indices': List[np.array],  # Top-K per rebalance
    'risk_eta': List[float],
    'direction_logits': List[np.array],
    'is_rebalance': List[bool],  # Rebalance flag (schedule OR regime shift)
    'rebalance_reason': List[str],  # "init", "schedule", "regime_shift:*", "hold"
    'regime_shift_detected': List[bool],  # Pure regime shift flag
    'days_since_rebalance': List[int],  # Days since last rebalance
    'metadata': {
        'year': int,
        'checkpoint_path': str,
        'num_days': int,
        'rebalance_interval': int,
        'regime_shift_count': int,
        'scheduled_rebalance_count': int,
    }
}

Usage:
    python scripts/generate_rl_states.py \
        --checkpoint ./observer_walkforward/checkpoints/observer_best_2018.pth \
        --year 2019 \
        --output ./observer_walkforward/rl_states/state_2019.pkl

    # Batch generation for multiple years
    python scripts/generate_rl_states.py \
        --checkpoint-dir ./observer_walkforward/checkpoints \
        --years 2019 2020 2021 2022 2023 2024 \
        --output-dir ./observer_walkforward/rl_states
"""

import argparse
import os
import pickle
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config


class RegimeShiftDetector:
    """
    Detect regime shifts for dynamic rebalancing signals.

    Replicates the logic from tradeEnv._detect_regime_shift() for use
    in pre-computing RL states from frozen Observer.

    Detection triggers:
    1. Direction reversal: Bull→Bear or Bear→Bull for N consecutive days
    2. Volatility shock: Current vol > μ + k*σ
    3. DC trigger: Index price move > threshold
    """

    def __init__(self, config: Config):
        self.config = config

        # Direction history for reversal detection
        self.direction_history = deque(maxlen=10)

        # Volatility tracking
        self.vol_window = deque(maxlen=60)  # 60-day rolling window

        # DC (Directional Change) tracking
        self.last_index_pivot = None
        self.last_dc_direction = 0  # -1=down, 0=neutral, 1=up

        # Config parameters
        self.confirmation_window = int(getattr(config, "regime_confirmation_window", 3))
        self.trend_z_threshold = float(getattr(config, "regime_trend_z_threshold", 1.0))
        self.vol_k = float(getattr(config, "regime_vol_k", 3.0))
        self.dc_threshold_pct = float(getattr(config, "regime_dc_threshold_pct", 0.05))

    def update_and_detect(
        self,
        direction_logits: np.ndarray,
        index_close: float = None,
        daily_return: float = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Update internal state and detect regime shift.

        IMPORTANT: Detection uses data BEFORE current day to avoid look-ahead bias.
        Buffers are updated AFTER detection completes.

        Args:
            direction_logits: (3,) array from Observer [bear, side, bull]
            index_close: Current index close price (for DC detection)
            daily_return: Daily return for volatility tracking

        Returns:
            (regime_shift_detected: bool, reason: str or None)
        """
        reason = None

        # Get direction label (0=bull, 1=side, 2=bear)
        # Note: Observer outputs [bear, side, bull] so we map accordingly
        dir_label = int(np.argmax(direction_logits))
        self.direction_history.append(dir_label)

        # === Detection 1: Direction Reversal ===
        # Uses direction history which is trailing (no look-ahead)
        hist = list(self.direction_history)
        if len(hist) >= self.confirmation_window + 1:
            last_seq = hist[-self.confirmation_window:]
            prev_regime = hist[-(self.confirmation_window + 1)]
            bull, bear = 0, 2  # Based on argmax of [bear, side, bull]

            # Bear to Bull reversal
            if all(v == bull for v in last_seq) and prev_regime == bear:
                reason = f"direction_reversal_bear_to_bull_{self.confirmation_window}d"
            # Bull to Bear reversal
            elif all(v == bear for v in last_seq) and prev_regime == bull:
                reason = f"direction_reversal_bull_to_bear_{self.confirmation_window}d"

        # === Detection 2: Volatility Shock ===
        # CRITICAL: Use EXISTING buffer (before adding today's return) to avoid look-ahead
        # vol_current = realized vol from returns up to T-1
        # vol_mu, vol_sigma = historical baseline from vol_window
        if reason is None and len(self.vol_window) >= 20:
            vol_array = np.array(self.vol_window)
            # vol_current = std of existing returns (T-N...T-1), NOT including today
            vol_current = float(np.std(vol_array)) if len(vol_array) >= 2 else 0
            # vol_mu, vol_sigma from historical volatility values
            vol_mu = float(np.mean(vol_array))
            vol_sigma = float(np.std(vol_array))

            if vol_sigma > 0 and vol_current > vol_mu + self.vol_k * vol_sigma:
                reason = f"vol_shock_{vol_current:.4f}_vs_{vol_mu:.4f}+{self.vol_k}*{vol_sigma:.4f}"

        # === Detection 3: DC Trigger on Index ===
        # Uses price change from last pivot (trailing, no look-ahead)
        if reason is None and index_close is not None:
            if self.last_index_pivot is None:
                self.last_index_pivot = index_close
            else:
                pct_change = (index_close - self.last_index_pivot) / max(self.last_index_pivot, 1e-8)

                # Upward DC trigger
                if pct_change >= self.dc_threshold_pct and self.last_dc_direction <= 0:
                    reason = "dc_trigger_index_up"
                    self.last_dc_direction = 1
                    self.last_index_pivot = index_close
                # Downward DC trigger
                elif pct_change <= -self.dc_threshold_pct and self.last_dc_direction >= 0:
                    reason = "dc_trigger_index_down"
                    self.last_dc_direction = -1
                    self.last_index_pivot = index_close

        # === Update buffers AFTER detection to avoid using current-day data ===
        if daily_return is not None:
            self.vol_window.append(abs(daily_return))

        return (reason is not None), reason

    def reset(self):
        """Reset detector state for new year/period."""
        self.direction_history.clear()
        self.vol_window.clear()
        self.last_index_pivot = None
        self.last_dc_direction = 0


def load_frozen_observer(checkpoint_path: str, config: Config):
    """
    Load observer from checkpoint and freeze all parameters.

    Args:
        checkpoint_path: Path to observer checkpoint
        config: Config object

    Returns:
        Frozen observer instance
    """
    from RL_controller.mafia_observer import MAFIAObserver

    # Create observer instance
    observer = MAFIAObserver(config)

    # Load checkpoint
    loaded_epoch = observer.load_checkpoint(checkpoint_path)
    print(f"Loaded observer from epoch {loaded_epoch}")

    # Freeze all parameters
    observer.mafia_model.eval()
    for param in observer.mafia_model.parameters():
        param.requires_grad = False

    return observer


def setup_inference_environment(config: Config, year: int):
    """
    Setup environment for inference on a specific year.

    Args:
        config: Config object
        year: Year for inference

    Returns:
        Environment instance
    """
    from utils.tradeEnv import StockPortfolioEnv
    from utils.mafia_data_loader import MAFIADataLoader

    # Set date range for the year
    config.test_date_start = f"{year}-01-01 00:00:00"
    config.test_date_end = f"{year}-12-31 23:59:59"

    # Load data
    mafia_loader = MAFIADataLoader(config=config)
    data = mafia_loader.load_raw_data()
    data_dict = mafia_loader.load_and_split_data(data=data)

    # Get test data for the year
    test_data = data_dict.get("test", data_dict.get("train"))

    # Create environment
    env = StockPortfolioEnv(
        config=config,
        rawdata=test_data,
        mode="test",
        stock_num=config.topK,
        action_dim=config.topK,
        tech_indicator_lst=[],
        extra_data={},
        mkt_observer=None,  # Will be set separately
    )

    return env, test_data


def generate_states_for_year(
    observer,
    config: Config,
    year: int,
    rebalance_interval: int = 14,
    verbose: bool = True,
) -> Dict:
    """
    Generate pre-computed states for one year of data with regime shift detection.

    Saves critical Observer outputs for RL training:
    - topk_embeddings: (K, D_stock) - Feature vectors for K stocks
    - topk_scores: (K,) - Quality scores for K stocks
    - market_context: (D_macro,) - Macro context
    - risk_eta: (1,) - Risk coefficient
    - direction_logits: (3,) - Market direction
    - is_rebalance: bool - Rebalance flag (schedule OR regime shift)
    - rebalance_reason: str - "init", "schedule", "regime_shift:*", "hold"
    - regime_shift_detected: bool - Pure regime shift flag
    - days_since_rebalance: int - Days since last rebalance

    Args:
        observer: Frozen observer instance
        config: Config object
        year: Year for inference
        rebalance_interval: Days between rebalances
        verbose: Print progress

    Returns:
        Dict with states, topk data, rebalance signals, and metadata
    """
    env, test_data = setup_inference_environment(config, year)

    # Attach observer to environment
    env.mkt_observer = observer

    # Initialize regime shift detector
    regime_detector = RegimeShiftDetector(config)

    # Storage for generated states (per-day data)
    states = []
    topk_indices_list = []
    topk_embeddings_list = []
    topk_scores_list = []
    risk_eta_list = []
    direction_logits_list = []
    market_context_list = []

    # NEW: Rebalance signal storage
    is_rebalance_list = []
    rebalance_reason_list = []
    regime_shift_detected_list = []
    days_since_rebalance_list = []

    # Current values (persist between rebalances)
    current_topk = None
    current_topk_embeddings = None
    current_topk_scores = None
    current_risk_eta = 1.0
    current_direction = np.array([0.33, 0.34, 0.33])
    current_market_context = None

    # Rebalance tracking
    days_since_rebalance = 0
    last_index_close = None
    regime_shift_count = 0
    scheduled_rebalance_count = 0

    # Reset environment
    obs = env.reset()
    done = False
    day = 0

    if verbose:
        print(f"\nGenerating states for year {year}...")
        print(f"  Rebalance interval: {rebalance_interval} days")
        print(f"  Regime shift detection: ENABLED")

    while not done:
        # === Step 1: Determine rebalance trigger ===
        is_scheduled_rebalance = (day == 0) or (days_since_rebalance >= rebalance_interval)
        regime_shift = False
        regime_reason = None
        rebalance_reason = "hold"

        # Get current index close for DC detection (from environment)
        index_close = None
        daily_return = None
        try:
            if hasattr(env, "curData") and env.curData is not None:
                # Try to get market index close price
                if "close" in env.curData.columns:
                    index_close = float(env.curData["close"].mean())
                elif "mkt_close" in env.curData.columns:
                    index_close = float(env.curData["mkt_close"].iloc[0])

                # Calculate daily return for volatility tracking
                if last_index_close is not None and index_close is not None:
                    daily_return = (index_close - last_index_close) / max(last_index_close, 1e-8)
                last_index_close = index_close
        except Exception:
            pass

        # === Step 2: Get Observer prediction (always needed for direction_logits) ===
        try:
            with torch.no_grad():
                topk, risk_eta, direction = observer.predict(
                    env=env,
                    mode="test",
                    force_rebalance=is_scheduled_rebalance,
                )
                temp_direction = np.array(direction) if direction is not None else np.array([0.33, 0.34, 0.33])

                # Run regime shift detection with current direction
                regime_shift, regime_reason = regime_detector.update_and_detect(
                    direction_logits=temp_direction,
                    index_close=index_close,
                    daily_return=daily_return,
                )
        except Exception as e:
            if verbose and day == 0:
                print(f"[Warning] Observer predict failed at day {day}: {e}")
            temp_direction = np.array([0.33, 0.34, 0.33])

        # === Step 3: Determine final rebalance decision ===
        is_rebalance = False
        if day == 0:
            is_rebalance = True
            rebalance_reason = "init"
        elif is_scheduled_rebalance:
            is_rebalance = True
            rebalance_reason = "schedule"
            scheduled_rebalance_count += 1
        elif regime_shift and getattr(config, "mafia_enable_regime_force_rebalance", True):
            is_rebalance = True
            rebalance_reason = f"regime_shift:{regime_reason}"
            regime_shift_count += 1
            if verbose:
                print(f"  📊 Day {day}: Regime shift detected - {regime_reason}")

        # === Step 4: Update Top-K on rebalance ===
        if is_rebalance:
            days_since_rebalance = 0
            try:
                with torch.no_grad():
                    topk, risk_eta, direction = observer.predict(
                        env=env,
                        mode="test",
                        force_rebalance=True,
                    )
                    current_topk = np.array(topk) if topk is not None else np.arange(config.topK)
                    current_risk_eta = float(risk_eta) if risk_eta is not None else 1.0
                    current_direction = np.array(direction) if direction is not None else np.array([0.33, 0.34, 0.33])

                    # Extract topk_embeddings
                    if hasattr(observer, "last_topk_embeddings"):
                        current_topk_embeddings = np.array(observer.last_topk_embeddings)
                    elif hasattr(observer, "mafia_model") and hasattr(observer.mafia_model, "last_topk_embeddings"):
                        current_topk_embeddings = np.array(observer.mafia_model.last_topk_embeddings.cpu().numpy())
                    else:
                        d_stock = getattr(config, "mafia_stock_embed_dim", 64)
                        current_topk_embeddings = np.zeros((config.topK, d_stock))

                    # Extract topk_scores
                    if hasattr(observer, "last_topk_scores"):
                        current_topk_scores = np.array(observer.last_topk_scores)
                    elif hasattr(observer, "mafia_model") and hasattr(observer.mafia_model, "last_topk_scores"):
                        current_topk_scores = np.array(observer.mafia_model.last_topk_scores.cpu().numpy())
                    else:
                        current_topk_scores = np.ones(config.topK) / config.topK

                    # Extract market_context
                    if hasattr(observer, "last_market_context"):
                        current_market_context = np.array(observer.last_market_context)
                    elif hasattr(observer, "mafia_model") and hasattr(observer.mafia_model, "last_market_context"):
                        current_market_context = np.array(observer.mafia_model.last_market_context.cpu().numpy())
                    else:
                        d_macro = getattr(config, "mafia_market_embed_dim", 64)
                        current_market_context = np.zeros(d_macro)

            except Exception as e:
                if verbose:
                    print(f"[Warning] Observer predict failed at day {day}: {e}")
                current_topk = np.arange(config.topK)
                current_risk_eta = 1.0
                current_direction = np.array([0.33, 0.34, 0.33])
                d_stock = getattr(config, "mafia_stock_embed_dim", 64)
                d_macro = getattr(config, "mafia_market_embed_dim", 64)
                current_topk_embeddings = np.zeros((config.topK, d_stock))
                current_topk_scores = np.ones(config.topK) / config.topK
                current_market_context = np.zeros(d_macro)
        else:
            # Update direction_logits daily (even on non-rebalance days)
            current_direction = temp_direction

        # === Step 5: Store values ===
        topk_indices_list.append(current_topk.copy() if current_topk is not None else np.arange(config.topK))
        topk_embeddings_list.append(current_topk_embeddings.copy() if current_topk_embeddings is not None else None)
        topk_scores_list.append(current_topk_scores.copy() if current_topk_scores is not None else None)
        risk_eta_list.append(current_risk_eta)
        direction_logits_list.append(current_direction.copy())
        market_context_list.append(current_market_context.copy() if current_market_context is not None else None)

        # Store rebalance signals
        is_rebalance_list.append(is_rebalance)
        rebalance_reason_list.append(rebalance_reason)
        regime_shift_detected_list.append(regime_shift)
        days_since_rebalance_list.append(days_since_rebalance)

        # Compose full state for this day
        state = {
            "day": day,
            "is_rebalance": is_rebalance,
            "rebalance_reason": rebalance_reason,
            "regime_shift_detected": regime_shift,
            "days_since_rebalance": days_since_rebalance,
            "topk_indices": topk_indices_list[-1],
            "topk_embeddings": topk_embeddings_list[-1],
            "topk_scores": topk_scores_list[-1],
            "risk_eta": risk_eta_list[-1],
            "direction_logits": direction_logits_list[-1],
            "market_context": market_context_list[-1],
            "observation": obs if isinstance(obs, dict) else {"flat": obs},
        }
        states.append(state)

        # === Step 6: Advance environment ===
        action = np.ones(config.topK) / config.topK
        try:
            step_result = env.step(action)
            if len(step_result) >= 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, done = step_result[:3]
                info = {}
        except Exception as e:
            if verbose:
                print(f"[Warning] Environment step failed at day {day}: {e}")
            done = True

        day += 1
        days_since_rebalance += 1

        if verbose and day % 50 == 0:
            print(f"  Processed {day} days... (regime shifts: {regime_shift_count})")

    # === Summary ===
    if verbose:
        print(f"\n  Generated {len(states)} states for year {year}")
        print(f"  Rebalance summary:")
        print(f"    - Scheduled rebalances: {scheduled_rebalance_count}")
        print(f"    - Regime shift rebalances: {regime_shift_count}")
        print(f"    - Total rebalances: {sum(is_rebalance_list)}")

        if topk_embeddings_list[0] is not None:
            d_stock = topk_embeddings_list[0].shape[1] if len(topk_embeddings_list[0].shape) > 1 else 0
        else:
            d_stock = 0
        if market_context_list[0] is not None:
            d_macro = len(market_context_list[0])
        else:
            d_macro = 0
        print(f"  State dimensions:")
        print(f"    - topk_embeddings: ({config.topK}, {d_stock})")
        print(f"    - topk_scores: ({config.topK},)")
        print(f"    - market_context: ({d_macro},)")
        print(f"    - direction_logits: (3,)")
        print(f"    - risk_eta: (1,)")

    # Compile output with all critical Observer outputs and rebalance signals
    result = {
        "states": states,
        # Per-day lists (for direct access)
        "topk_indices": topk_indices_list,
        "topk_embeddings": topk_embeddings_list,
        "topk_scores": topk_scores_list,
        "risk_eta": risk_eta_list,
        "direction_logits": direction_logits_list,
        "market_context": market_context_list,
        # NEW: Rebalance signals for Portfolio Allocator
        "is_rebalance": is_rebalance_list,
        "rebalance_reason": rebalance_reason_list,
        "regime_shift_detected": regime_shift_detected_list,
        "days_since_rebalance": days_since_rebalance_list,
        "metadata": {
            "year": year,
            "checkpoint_path": config.observer_pretrained_path,
            "num_days": len(states),
            "rebalance_interval": rebalance_interval,
            "topk": config.topK,
            "market_name": config.market_name,
            "d_stock": topk_embeddings_list[0].shape[1] if topk_embeddings_list[0] is not None and len(topk_embeddings_list[0].shape) > 1 else 64,
            "d_macro": len(market_context_list[0]) if market_context_list[0] is not None else 64,
            # NEW: Regime shift statistics
            "regime_shift_count": regime_shift_count,
            "scheduled_rebalance_count": scheduled_rebalance_count,
            "total_rebalances": sum(is_rebalance_list),
        },
    }

    return result


def save_states(states: Dict, output_path: str):
    """Save states to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(states, f)
    print(f"Saved states to: {output_path}")


def load_states(input_path: str) -> Dict:
    """Load states from pickle file."""
    with open(input_path, "rb") as f:
        return pickle.load(f)


def generate_single_year(
    checkpoint_path: str,
    year: int,
    output_path: str,
    config: Optional[Config] = None,
    rebalance_interval: int = 14,
    verbose: bool = True,
):
    """
    Generate and save states for a single year.

    Args:
        checkpoint_path: Path to observer checkpoint
        year: Year for inference
        output_path: Path to save output pickle
        config: Optional config object (created if None)
        rebalance_interval: Days between rebalances
        verbose: Print progress
    """
    if config is None:
        config = Config()

    config.observer_pretrained_path = checkpoint_path

    # Load frozen observer
    observer = load_frozen_observer(checkpoint_path, config)

    # Generate states
    states = generate_states_for_year(
        observer=observer,
        config=config,
        year=year,
        rebalance_interval=rebalance_interval,
        verbose=verbose,
    )

    # Save
    save_states(states, output_path)


def generate_batch(
    checkpoint_dir: str,
    years: List[int],
    output_dir: str,
    config: Optional[Config] = None,
    rebalance_interval: int = 14,
    verbose: bool = True,
):
    """
    Generate states for multiple years in batch.

    Expects checkpoint files named: observer_best_{valid_year}.pth
    where valid_year = infer_year - 1

    Args:
        checkpoint_dir: Directory containing observer checkpoints
        years: List of inference years
        output_dir: Directory to save output pickles
        config: Optional config object
        rebalance_interval: Days between rebalances
        verbose: Print progress
    """
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        valid_year = year - 1
        checkpoint_path = os.path.join(checkpoint_dir, f"observer_best_{valid_year}.pth")

        if not os.path.exists(checkpoint_path):
            print(f"[Warning] Checkpoint not found: {checkpoint_path}, skipping year {year}")
            continue

        output_path = os.path.join(output_dir, f"state_{year}.pkl")

        if verbose:
            print(f"\n{'='*50}")
            print(f"Year {year}: Using checkpoint from {valid_year}")
            print(f"{'='*50}")

        try:
            generate_single_year(
                checkpoint_path=checkpoint_path,
                year=year,
                output_path=output_path,
                config=config,
                rebalance_interval=rebalance_interval,
                verbose=verbose,
            )
        except Exception as e:
            print(f"[Error] Failed to generate states for year {year}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Generate RL States from Frozen Observer"
    )

    # Single year mode
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to observer checkpoint (single year mode)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year for inference (single year mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for states pickle (single year mode)",
    )

    # Batch mode
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing observer checkpoints (batch mode)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="List of inference years (batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for states pickles (batch mode)",
    )

    # Common options
    parser.add_argument(
        "--rebalance-interval",
        type=int,
        default=14,
        help="Days between rebalances (default: 14)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Determine mode
    if args.checkpoint and args.year and args.output:
        # Single year mode
        generate_single_year(
            checkpoint_path=args.checkpoint,
            year=args.year,
            output_path=args.output,
            rebalance_interval=args.rebalance_interval,
            verbose=not args.quiet,
        )
    elif args.checkpoint_dir and args.years and args.output_dir:
        # Batch mode
        generate_batch(
            checkpoint_dir=args.checkpoint_dir,
            years=args.years,
            output_dir=args.output_dir,
            rebalance_interval=args.rebalance_interval,
            verbose=not args.quiet,
        )
    else:
        parser.print_help()
        print("\nError: Must specify either single year or batch mode arguments")
        sys.exit(1)


if __name__ == "__main__":
    main()
