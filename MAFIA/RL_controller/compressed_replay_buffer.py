import warnings
from typing import Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.buffers import (
    DictReplayBuffer,
    DictReplayBufferSamples,
    ReplayBuffer,
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import smart_print
except ImportError:
    smart_print = print  # Fallback


class CompressedReplayBuffer(ReplayBuffer):
    """
    Replay buffer variant that stores observations in a lower-precision dtype
    to save memory (acts like a lightweight `compress_obs`).

    Notes:
    - Compressing to float16 reduces memory ~2x but adds minor CPU cost
      for casting back to float32 when sampling.
    - Only intended for Box observation spaces.
    - Stores timestamps for each experience to enable filtering by date range.
    - Supports priority sampling for regime shift experiences.
    """

    def __init__(
        self,
        *args: Any,
        compress_obs: bool = True,
        storage_dtype: np.dtype = np.float16,
        recency_bias_sigma: Optional[float] = None,
        recency_bias_fraction: float = 0.6,
        regime_shift_sample_ratio: float = 0.15,
        **kwargs: Any,
    ) -> None:
        self.compress_obs = compress_obs
        self.storage_dtype = storage_dtype
        # If sigma is provided and >0, use it directly.
        # If sigma is None, we can derive a "flat" bias from recency_bias_fraction * max_index.
        # If sigma <= 0, fall back to uniform sampling.
        self.recency_bias_sigma = recency_bias_sigma
        self.recency_bias_fraction = recency_bias_fraction
        # Fraction of batch to sample from regime shift experiences (priority sampling)
        # Default 15%: if batch_size=256 and enough regime shifts exist, ~38 will be regime shifts
        self.regime_shift_sample_ratio = regime_shift_sample_ratio
        super().__init__(*args, **kwargs)

        # Allocate timestamp buffer (store as int64 nanoseconds for efficiency)
        # This stores the trade date for each experience
        self.timestamps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

        # Regime shift flag: mark experiences at regime shift events for priority retention
        # These experiences are preserved during buffer filtering across walk-forward windows
        self.regime_shift_flags = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.bool_
        )

        # Reallocate buffers with the compressed dtype if enabled and supported
        if self.compress_obs:
            if not isinstance(self.observation_space, spaces.Box):
                warnings.warn(
                    "compress_obs=True is only supported for Box observation spaces. Disabling."
                )
                self.compress_obs = False
            else:
                self.observations = np.zeros(
                    (self.buffer_size, self.n_envs, *self.obs_shape),
                    dtype=self.storage_dtype,
                )
                if not self.optimize_memory_usage:
                    self.next_observations = np.zeros(
                        (self.buffer_size, self.n_envs, *self.obs_shape),
                        dtype=self.storage_dtype,
                    )

    def _cast_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.compress_obs:
            return np.array(obs)
        return np.array(obs, dtype=self.storage_dtype)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = self._cast_obs(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = self._cast_obs(
                next_obs
            )
        else:
            self.next_observations[self.pos] = self._cast_obs(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        # Store timestamps from infos for buffer filtering across walk-forward windows
        # Extract curTradeDay from each env's info dict (format: YYYYMMDD as int64)
        self.timestamps[self.pos] = np.array(
            [info.get("curTradeDay", 0) for info in infos], dtype=np.int64
        )

        # Store regime shift flags for priority retention during buffer filtering
        # Experiences at regime shift events are preserved even when outside date range
        self.regime_shift_flags[self.pos] = np.array(
            [info.get("regime_shift_event", False) for info in infos], dtype=np.bool_
        )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """
        Sample indices with optional Gaussian recency bias and regime shift priority.

        Sampling Strategy (Two-Phase):
        1. Priority Phase: Sample `regime_shift_sample_ratio` of batch from regime shift
           experiences (if available). Uses uniform sampling within regime shift pool.
        2. Recency Phase: Sample remaining from all experiences using Gaussian recency bias
           centered at most recent experience.

        This ensures rare but important regime shift experiences are adequately represented
        in training batches, preventing "catastrophic forgetting" of crisis patterns.
        """
        max_index = self.buffer_size if self.full else self.pos
        if max_index == 0:
            return np.array([], dtype=np.int64)

        idxs = []

        # Phase 1: Priority sampling from regime shift experiences
        if self.regime_shift_sample_ratio > 0:
            # Find all regime shift indices in active buffer
            regime_shift_indices = np.where(
                self.regime_shift_flags[:max_index].any(axis=1)
            )[0]

            if len(regime_shift_indices) > 0:
                # Calculate how many to sample from regime shifts
                n_regime = min(
                    int(batch_size * self.regime_shift_sample_ratio),
                    len(regime_shift_indices),
                )

                if n_regime > 0:
                    # Uniform sample from regime shift pool (with replacement if needed)
                    regime_samples = np.random.choice(
                        regime_shift_indices,
                        size=n_regime,
                        replace=(n_regime > len(regime_shift_indices)),
                    )
                    idxs.extend(regime_samples.tolist())

        # Phase 2: Recency-biased sampling for remaining slots
        remaining = batch_size - len(idxs)
        if remaining <= 0:
            return np.array(idxs, dtype=np.int64)

        # Determine effective sigma for Gaussian recency bias
        sigma = self.recency_bias_sigma
        if sigma is None:
            sigma = max(1, int(self.recency_bias_fraction * max_index))

        if sigma <= 0:
            # Fallback to uniform sampling
            uniform_samples = np.random.randint(0, max_index, size=remaining)
            idxs.extend(uniform_samples.tolist())
        else:
            # Gaussian recency bias: center at most recent experience
            center = max_index - 1
            max_trials = remaining * 10
            trials = 0
            while len(idxs) < batch_size and trials < max_trials:
                candidate = int(np.random.normal(loc=center, scale=sigma))
                trials += 1
                if 0 <= candidate < max_index:
                    idxs.append(candidate)

            # Fill remaining with uniform if Gaussian didn't produce enough
            while len(idxs) < batch_size:
                idxs.append(np.random.randint(0, max_index))

        return np.array(idxs, dtype=np.int64)

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        batch_inds = self._sample_indices(batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs_raw = self.observations[
                (batch_inds + 1) % self.buffer_size, env_indices, :
            ]
        else:
            next_obs_raw = self.next_observations[batch_inds, env_indices, :]

        # Cast back to float32 for stability during training
        obs = self._normalize_obs(
            self.observations[batch_inds, env_indices, :].astype(np.float32), env
        )
        next_obs = self._normalize_obs(next_obs_raw.astype(np.float32), env)

        data = (
            obs,
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def filter_by_date_range(
        self,
        train_start: int,
        train_end: int,
        inplace: bool = True,
        preserve_regime_shifts: bool = True,
        prev_train_end: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """
        Filter buffer to keep only experiences within the specified date range.
        Used for walk-forward window transitions to retain overlapping experiences.

        Regime shift experiences from PREVIOUS TRAIN PERIOD are preserved because
        they contain critical information about market dynamics. Regime shifts
        from Valid/Test periods are NOT preserved to avoid data leakage.

        Args:
            train_start: Start date of new training window (YYYYMMDD format, e.g., 20180101)
            train_end: End date of new training window (YYYYMMDD format, e.g., 20201231)
            inplace: If True, modify buffer in-place. If False, return counts only.
            preserve_regime_shifts: If True, keep regime shift experiences from prev train period.
            prev_train_end: End date of PREVIOUS window's train period (YYYYMMDD).
                           Regime shifts are only preserved if timestamp <= prev_train_end.
                           If None, defaults to train_start - 1 day (conservative).

        Returns:
            Tuple[int, int, int]: (original_count, retained_count, regime_shift_preserved) for logging

        Example:
            Window 0: Train[2017-2019], Valid[2020], Test[2021]
            Window 1: Train[2018-2020], Valid[2021], Test[2022]

            Call: filter_by_date_range(20180101, 20201231, prev_train_end=20191231)

            - Experiences from 2018-2019: KEEP (in new train range)
            - Regime shift from 2017: KEEP (in prev train, before new train_start)
            - Regime shift from 2020 (Valid): FILTERED OUT (after prev_train_end = leakage!)
            - Regime shift from 2021 (Test): FILTERED OUT (after prev_train_end = leakage!)
        """
        if self.pos == 0 and not self.full:
            return (0, 0, 0)  # Buffer is empty

        max_idx = self.buffer_size if self.full else self.pos

        # Default prev_train_end: assume previous train ended 1 day before new train starts
        # This is conservative - only keeps regime shifts strictly before new window
        if prev_train_end is None:
            prev_train_end = train_start - 1  # e.g., 20171231 if train_start=20180101

        # Get mask for experiences within date range OR regime shift events (from prev train only)
        # timestamps shape: (buffer_size, n_envs), we check env 0
        valid_mask = np.zeros(max_idx, dtype=bool)
        regime_shift_preserved = 0
        regime_shift_filtered_leakage = 0

        for i in range(max_idx):
            ts = self.timestamps[i, 0]  # All envs at same pos have same date
            is_in_range = train_start <= ts <= train_end
            is_regime_shift = self.regime_shift_flags[i].any()  # Any env flagged

            if is_in_range:
                valid_mask[i] = True
                # Also count regime shifts that happen to be in range
                if is_regime_shift:
                    regime_shift_preserved += 1
            elif preserve_regime_shifts and is_regime_shift:
                # Only preserve regime shift if it's from PREVIOUS TRAIN period
                # NOT from Valid/Test (which would be data leakage!)
                if ts <= prev_train_end:
                    valid_mask[i] = True
                    regime_shift_preserved += 1
                else:
                    # Regime shift from Valid/Test period - DO NOT KEEP (leakage!)
                    regime_shift_filtered_leakage += 1

        if regime_shift_filtered_leakage > 0:
            smart_print(
                f"[BUFFER] Filtered {regime_shift_filtered_leakage} regime shift experiences "
                f"from Valid/Test period (after prev_train_end={prev_train_end}) to prevent leakage",
                flush=True,
            )

        original_count = max_idx
        retained_count = valid_mask.sum()

        if not inplace:
            return (original_count, int(retained_count), regime_shift_preserved)

        if retained_count == 0:
            # No valid experiences, reset buffer
            self.pos = 0
            self.full = False
            return (original_count, 0, 0)

        if retained_count == original_count:
            # All experiences are valid, no filtering needed
            return (original_count, int(retained_count), regime_shift_preserved)

        # Compact buffer: move valid experiences to front
        valid_indices = np.where(valid_mask)[0]

        new_observations = self.observations[valid_indices].copy()
        new_actions = self.actions[valid_indices].copy()
        new_rewards = self.rewards[valid_indices].copy()
        new_dones = self.dones[valid_indices].copy()
        new_timeouts = self.timeouts[valid_indices].copy()
        new_timestamps = self.timestamps[valid_indices].copy()
        new_regime_shift_flags = self.regime_shift_flags[valid_indices].copy()

        if not self.optimize_memory_usage:
            new_next_observations = self.next_observations[valid_indices].copy()

        # Reset and repopulate
        self.observations[:retained_count] = new_observations
        self.actions[:retained_count] = new_actions
        self.rewards[:retained_count] = new_rewards
        self.dones[:retained_count] = new_dones
        self.timeouts[:retained_count] = new_timeouts
        self.timestamps[:retained_count] = new_timestamps
        self.regime_shift_flags[:retained_count] = new_regime_shift_flags

        if not self.optimize_memory_usage:
            self.next_observations[:retained_count] = new_next_observations

        # Update position and full flag
        self.pos = int(retained_count)
        self.full = False  # Buffer is no longer circular after compaction

        return (original_count, int(retained_count), regime_shift_preserved)

    def get_buffer_stats(self) -> dict:
        """Get buffer statistics for logging/debugging."""
        max_idx = self.buffer_size if self.full else self.pos
        if max_idx == 0:
            return {
                "size": 0,
                "min_date": None,
                "max_date": None,
                "regime_shift_count": 0,
                "memory_mb": 0,
            }

        # Get date range from timestamps
        active_timestamps = self.timestamps[:max_idx, 0]  # Use env 0
        valid_timestamps = active_timestamps[active_timestamps > 0]

        # Count regime shift experiences
        regime_shift_count = self.regime_shift_flags[:max_idx].any(axis=1).sum()

        return {
            "size": max_idx,
            "min_date": int(valid_timestamps.min())
            if len(valid_timestamps) > 0
            else None,
            "max_date": int(valid_timestamps.max())
            if len(valid_timestamps) > 0
            else None,
            "regime_shift_count": int(regime_shift_count),
            "memory_mb": round(self.observations[:max_idx].nbytes / (1024 * 1024), 2),
        }


class CompressedDictReplayBuffer(DictReplayBuffer):
    """
    DictReplayBuffer that stores Box sub-observations in a lower-precision dtype
    to save memory. optimize_memory_usage is not supported for dict buffers.

    Also stores timestamps for each experience to support filtering across
    walk-forward windows. Supports priority sampling for regime shift experiences.
    """

    def __init__(
        self,
        *args: Any,
        compress_obs: bool = True,
        storage_dtype: np.dtype = np.float16,
        recency_bias_sigma: Optional[float] = None,
        recency_bias_fraction: float = 0.6,
        regime_shift_sample_ratio: float = 0.15,
        **kwargs: Any,
    ) -> None:
        self.compress_obs = compress_obs
        self.storage_dtype = storage_dtype
        self.recency_bias_sigma = recency_bias_sigma
        self.recency_bias_fraction = recency_bias_fraction
        # Fraction of batch to sample from regime shift experiences (priority sampling)
        self.regime_shift_sample_ratio = regime_shift_sample_ratio
        optimize_memory_usage = kwargs.pop("optimize_memory_usage", False)
        if optimize_memory_usage:
            warnings.warn(
                "optimize_memory_usage is not supported for Dict replay buffers; forcing False."
            )
        super().__init__(*args, optimize_memory_usage=False, **kwargs)

        # Timestamp storage for walk-forward buffer filtering
        self.timestamps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

        # Regime shift flag: mark experiences at regime shift events for priority retention
        self.regime_shift_flags = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.bool_
        )

        if self.compress_obs:
            compressed_keys = []
            for key, space in self.observation_space.spaces.items():
                if isinstance(space, spaces.Box):
                    self.observations[key] = np.zeros(
                        (self.buffer_size, self.n_envs, *self.obs_shape[key]),
                        dtype=self.storage_dtype,
                    )
                    self.next_observations[key] = np.zeros(
                        (self.buffer_size, self.n_envs, *self.obs_shape[key]),
                        dtype=self.storage_dtype,
                    )
                    compressed_keys.append(key)
            if len(compressed_keys) == 0:
                warnings.warn(
                    "compress_obs=True set but no Box observations to compress. Disabling."
                )
                self.compress_obs = False

    def _cast_obs(self, obs: np.ndarray, key: str) -> np.ndarray:
        if not self.compress_obs:
            return np.array(obs)
        space = self.observation_space.spaces[key]
        if isinstance(space, spaces.Box):
            return np.array(obs, dtype=self.storage_dtype)
        return np.array(obs)

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """
        Sample indices with optional Gaussian recency bias and regime shift priority.

        Sampling Strategy (Two-Phase):
        1. Priority Phase: Sample `regime_shift_sample_ratio` of batch from regime shift
           experiences (if available). Uses uniform sampling within regime shift pool.
        2. Recency Phase: Sample remaining from all experiences using Gaussian recency bias
           centered at most recent experience.
        """
        max_index = self.buffer_size if self.full else self.pos
        if max_index == 0:
            return np.array([], dtype=np.int64)

        idxs = []

        # Phase 1: Priority sampling from regime shift experiences
        if self.regime_shift_sample_ratio > 0:
            regime_shift_indices = np.where(
                self.regime_shift_flags[:max_index].any(axis=1)
            )[0]

            if len(regime_shift_indices) > 0:
                n_regime = min(
                    int(batch_size * self.regime_shift_sample_ratio),
                    len(regime_shift_indices),
                )

                if n_regime > 0:
                    regime_samples = np.random.choice(
                        regime_shift_indices,
                        size=n_regime,
                        replace=(n_regime > len(regime_shift_indices)),
                    )
                    idxs.extend(regime_samples.tolist())

        # Phase 2: Recency-biased sampling for remaining slots
        remaining = batch_size - len(idxs)
        if remaining <= 0:
            return np.array(idxs, dtype=np.int64)

        sigma = self.recency_bias_sigma
        if sigma is None:
            sigma = max(1, int(self.recency_bias_fraction * max_index))

        if sigma <= 0:
            uniform_samples = np.random.randint(0, max_index, size=remaining)
            idxs.extend(uniform_samples.tolist())
        else:
            center = max_index - 1
            max_trials = remaining * 10
            trials = 0
            while len(idxs) < batch_size and trials < max_trials:
                candidate = int(np.random.normal(loc=center, scale=sigma))
                trials += 1
                if 0 <= candidate < max_index:
                    idxs.append(candidate)

            while len(idxs) < batch_size:
                idxs.append(np.random.randint(0, max_index))

        return np.array(idxs, dtype=np.int64)

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        for key in self.observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = self._cast_obs(obs[key], key)

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape(
                    (self.n_envs,) + self.obs_shape[key]
                )
            self.next_observations[key][self.pos] = self._cast_obs(next_obs[key], key)

        action = action.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        # Store timestamps from infos for buffer filtering across walk-forward windows
        self.timestamps[self.pos] = np.array(
            [info.get("curTradeDay", 0) for info in infos], dtype=np.int64
        )

        # Store regime shift flags for priority retention during buffer filtering
        self.regime_shift_flags[self.pos] = np.array(
            [info.get("regime_shift_event", False) for info in infos], dtype=np.bool_
        )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        obs_ = self._normalize_obs(
            {
                key: obs[batch_inds, env_indices, :].astype(np.float32)
                for key, obs in self.observations.items()
            },
            env,
        )
        next_obs_ = self._normalize_obs(
            {
                key: obs[batch_inds, env_indices, :].astype(np.float32)
                for key, obs in self.next_observations.items()
            },
            env,
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        observations = {key: self.to_torch(o) for key, o in obs_.items()}
        next_observations = {key: self.to_torch(o) for key, o in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1), env
                )
            ),
        )

    def sample(  # type: ignore[override]
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        batch_inds = self._sample_indices(batch_size)
        return self._get_samples(batch_inds, env=env)

    def filter_by_date_range(
        self, train_start: int, train_end: int, inplace: bool = True
    ) -> Tuple[int, int]:
        """
        Filter buffer to keep only experiences within the specified date range.
        Used for walk-forward window transitions to retain overlapping experiences.

        Args:
            train_start: Start date of new training window (YYYYMMDD format)
            train_end: End date of new training window (YYYYMMDD format)
            inplace: If True, modify buffer in-place. If False, return counts only.

        Returns:
            Tuple[int, int]: (original_count, retained_count) for logging
        """
        if self.pos == 0 and not self.full:
            return (0, 0)

        max_idx = self.buffer_size if self.full else self.pos

        # Get mask for experiences within date range
        valid_mask = np.zeros(max_idx, dtype=bool)
        for i in range(max_idx):
            ts = self.timestamps[i, 0]
            if train_start <= ts <= train_end:
                valid_mask[i] = True

        original_count = max_idx
        retained_count = valid_mask.sum()

        if not inplace:
            return (original_count, int(retained_count))

        if retained_count == 0:
            self.pos = 0
            self.full = False
            return (original_count, 0)

        if retained_count == original_count:
            return (original_count, int(retained_count))

        # Compact buffer
        valid_indices = np.where(valid_mask)[0]

        new_observations = {
            key: obs[valid_indices].copy() for key, obs in self.observations.items()
        }
        new_next_observations = {
            key: obs[valid_indices].copy()
            for key, obs in self.next_observations.items()
        }
        new_actions = self.actions[valid_indices].copy()
        new_rewards = self.rewards[valid_indices].copy()
        new_dones = self.dones[valid_indices].copy()
        new_timeouts = self.timeouts[valid_indices].copy()
        new_timestamps = self.timestamps[valid_indices].copy()

        # Reset and repopulate
        for key in self.observations.keys():
            self.observations[key][:retained_count] = new_observations[key]
            self.next_observations[key][:retained_count] = new_next_observations[key]

        self.actions[:retained_count] = new_actions
        self.rewards[:retained_count] = new_rewards
        self.dones[:retained_count] = new_dones
        self.timeouts[:retained_count] = new_timeouts
        self.timestamps[:retained_count] = new_timestamps

        self.pos = int(retained_count)
        self.full = False

        return (original_count, int(retained_count))

    def get_buffer_stats(self) -> dict:
        """Get buffer statistics for logging/debugging."""
        max_idx = self.buffer_size if self.full else self.pos
        if max_idx == 0:
            return {"size": 0, "min_date": None, "max_date": None, "memory_mb": 0}

        active_timestamps = self.timestamps[:max_idx, 0]
        valid_timestamps = active_timestamps[active_timestamps > 0]

        # Calculate memory for dict observations
        total_bytes = sum(obs[:max_idx].nbytes for obs in self.observations.values())

        return {
            "size": max_idx,
            "min_date": int(valid_timestamps.min())
            if len(valid_timestamps) > 0
            else None,
            "max_date": int(valid_timestamps.max())
            if len(valid_timestamps) > 0
            else None,
            "memory_mb": round(total_bytes / (1024 * 1024), 2),
        }
