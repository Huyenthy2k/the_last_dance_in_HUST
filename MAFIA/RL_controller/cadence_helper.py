"""
Helper utilities for Training Cadence Alignment.

Generates cadence masks for different trading operational schedules:
- Selection cadence: Top-K rebalance trigger (e.g., every N days)
- Risk cadence: Daily adaptive risk estimation
- Direction cadence: Daily market regime detection
"""

import numpy as np
import torch as th
from typing import Optional, Union, Tuple


def generate_rebalance_schedule(
    total_steps: int,
    rebalance_interval: int = 5,
    start_offset: int = 0,
) -> np.ndarray:
    """
    Generate binary rebalance schedule (cadence mask).

    Args:
        total_steps: Total number of steps in episode
        rebalance_interval: Rebalance every N steps (e.g., 5 = weekly if daily steps)
        start_offset: Offset to first rebalance (0 = rebalance on step 0)

    Returns:
        mask: (total_steps,) binary array where 1 = rebalance, 0 = holding

    Example:
        >>> mask = generate_rebalance_schedule(total_steps=20, rebalance_interval=5)
        >>> mask
        array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    """
    mask = np.zeros(total_steps, dtype=np.float32)

    # Mark rebalance days
    idx = start_offset
    while idx < total_steps:
        mask[idx] = 1.0
        idx += rebalance_interval

    return mask


def generate_adaptive_cadence(
    total_steps: int,
    base_interval: int = 5,
    regime_shift_indices: Optional[np.ndarray] = None,
    rebalance_on_regime_shift: bool = True,
) -> np.ndarray:
    """
    Generate adaptive rebalance cadence with regime-triggered events.

    Args:
        total_steps: Total number of steps in episode
        base_interval: Base rebalance interval (e.g., 5 = every 5 days)
        regime_shift_indices: Indices where market regime shifts occur
        rebalance_on_regime_shift: If True, force rebalance on regime shifts

    Returns:
        mask: (total_steps,) binary array where 1 = rebalance, 0 = holding

    Example:
        >>> regimes = np.array([0, 10, 15])  # Regime shifts at steps 0, 10, 15
        >>> mask = generate_adaptive_cadence(
        ...     total_steps=20,
        ...     base_interval=5,
        ...     regime_shift_indices=regimes,
        ...     rebalance_on_regime_shift=True
        ... )
        >>> mask[10], mask[15]  # Should be 1.0
        (1.0, 1.0)
    """
    mask = np.zeros(total_steps, dtype=np.float32)

    # Regular interval-based rebalances
    idx = 0
    while idx < total_steps:
        mask[idx] = 1.0
        idx += base_interval

    # Force rebalance on regime shifts if enabled
    if rebalance_on_regime_shift and regime_shift_indices is not None:
        regime_shift_indices = np.asarray(regime_shift_indices).astype(int)
        regime_shift_indices = regime_shift_indices[
            (regime_shift_indices >= 0) & (regime_shift_indices < total_steps)
        ]
        mask[regime_shift_indices] = 1.0

    return mask


def create_batch_cadence_masks(
    batch_size: int,
    total_steps: int,
    rebalance_interval: int = 5,
    noise_level: float = 0.0,
) -> th.Tensor:
    """
    Create cadence masks for a batch of trajectories.

    Args:
        batch_size: Number of trajectories in batch
        total_steps: Number of steps per trajectory
        rebalance_interval: Rebalance every N steps
        noise_level: Add random noise to mask (0.0-1.0)

    Returns:
        masks: (total_steps, batch_size) tensor of masks

    Example:
        >>> masks = create_batch_cadence_masks(batch_size=32, total_steps=100, rebalance_interval=10)
        >>> masks.shape
        torch.Size([100, 32])
        >>> masks[:, 0].sum()  # Number of rebalances in first trajectory
        tensor(10.)
    """
    # Generate base mask
    base_mask = generate_rebalance_schedule(total_steps, rebalance_interval)

    # Replicate for batch
    masks = np.tile(base_mask[:, np.newaxis], (1, batch_size))  # (T, B)

    # Optional: add small random noise to create diverse cadences
    if noise_level > 0:
        noise = np.random.binomial(1, noise_level, size=(total_steps, batch_size))
        masks = np.logical_xor(masks, noise).astype(np.float32)

    return th.from_numpy(masks).float()


def align_cadence_with_env_schedule(
    env_rebalance_days: Optional[np.ndarray],
    total_steps: int,
) -> np.ndarray:
    """
    Convert environment rebalance schedule to cadence mask.

    Args:
        env_rebalance_days: Array of (batch_idx, step_idx) tuples indicating rebalances
        total_steps: Total steps in episode

    Returns:
        mask: (total_steps,) binary array where 1 = rebalance, 0 = holding
    """
    mask = np.zeros(total_steps, dtype=np.float32)

    if env_rebalance_days is not None:
        for step_idx in np.asarray(env_rebalance_days):
            if 0 <= step_idx < total_steps:
                mask[int(step_idx)] = 1.0

    return mask


def compute_cadence_coverage(
    masks: Union[np.ndarray, th.Tensor],
) -> Tuple[float, float]:
    """
    Compute coverage statistics of cadence masks.

    Args:
        masks: (T,) or (T, B) mask array/tensor

    Returns:
        (rebalance_ratio, holding_ratio): Fraction of rebalance vs holding days

    Example:
        >>> mask = generate_rebalance_schedule(20, rebalance_interval=5)
        >>> rebal_ratio, hold_ratio = compute_cadence_coverage(mask)
        >>> rebal_ratio, hold_ratio
        (0.2, 0.8)  # 20% rebalance, 80% holding
    """
    if isinstance(masks, th.Tensor):
        masks = masks.cpu().numpy()

    masks = np.asarray(masks)
    total_days = masks.size
    rebalance_days = np.sum(masks > 0.5)

    rebalance_ratio = rebalance_days / total_days if total_days > 0 else 0.0
    holding_ratio = 1.0 - rebalance_ratio

    return float(rebalance_ratio), float(holding_ratio)


# Example configuration strings for common cadences
CADENCE_CONFIGS = {
    "daily": {
        "rebalance_interval": 1,  # Every day is rebalance
        "description": "Selection updated every day (full training cadence)",
    },
    "weekly": {
        "rebalance_interval": 5,  # Assuming 5 trading days/week
        "description": "Selection updated weekly, hold days between rebalances",
    },
    "biweekly": {
        "rebalance_interval": 10,
        "description": "Selection updated every 2 weeks",
    },
    "monthly": {
        "rebalance_interval": 21,  # Approx 21 trading days/month
        "description": "Selection updated monthly",
    },
}


def get_cadence_schedule(
    cadence_name: str = "weekly",
    total_steps: int = 252,  # 1 trading year
    adaptive: bool = False,
    regime_shifts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Get predefined cadence schedule.

    Args:
        cadence_name: One of 'daily', 'weekly', 'biweekly', 'monthly'
        total_steps: Total steps in episode
        adaptive: If True, incorporate regime shifts
        regime_shifts: Indices of regime shift events

    Returns:
        mask: (total_steps,) binary array

    Example:
        >>> mask = get_cadence_schedule("weekly", total_steps=252)
        >>> mask.sum()  # ~51 rebalance days in a year
        51.0
    """
    if cadence_name not in CADENCE_CONFIGS:
        raise ValueError(
            f"Unknown cadence: {cadence_name}. Choose from {list(CADENCE_CONFIGS.keys())}"
        )

    config = CADENCE_CONFIGS[cadence_name]
    interval = config["rebalance_interval"]

    if adaptive and regime_shifts is not None:
        return generate_adaptive_cadence(
            total_steps=total_steps,
            base_interval=interval,
            regime_shift_indices=regime_shifts,
            rebalance_on_regime_shift=True,
        )
    else:
        return generate_rebalance_schedule(
            total_steps=total_steps,
            rebalance_interval=interval,
        )


if __name__ == "__main__":
    # Example usage
    print("=== Cadence Masking Examples ===\n")

    # 1. Weekly rebalance cadence
    weekly_mask = generate_rebalance_schedule(total_steps=100, rebalance_interval=5)
    print(f"Weekly cadence (first 20 steps): {weekly_mask[:20]}")
    print(f"Rebalance coverage: {compute_cadence_coverage(weekly_mask)}\n")

    # 2. Adaptive cadence with regime shifts
    regime_shifts = np.array([0, 30, 60, 90])
    adaptive_mask = generate_adaptive_cadence(
        total_steps=100,
        base_interval=10,
        regime_shift_indices=regime_shifts,
        rebalance_on_regime_shift=True,
    )
    print(f"Adaptive cadence (regime at steps {regime_shifts}): {adaptive_mask[:20]}")
    print(f"Rebalance coverage: {compute_cadence_coverage(adaptive_mask)}\n")

    # 3. Batch cadence masks
    batch_masks = create_batch_cadence_masks(
        batch_size=4, total_steps=50, rebalance_interval=5
    )
    print(f"Batch masks shape: {batch_masks.shape}")
    print(f"First trajectory rebalances: {batch_masks[:, 0].sum()}\n")

    # 4. Predefined schedules
    for cadence_name in ["daily", "weekly", "monthly"]:
        mask = get_cadence_schedule(cadence_name, total_steps=252)
        rebal_ratio, _ = compute_cadence_coverage(mask)
        print(
            f"{cadence_name.capitalize()}: {rebal_ratio * 100:.1f}% rebalance days/year"
        )
