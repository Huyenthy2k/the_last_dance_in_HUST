import warnings
from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import (
    DictReplayBuffer,
    DictReplayBufferSamples,
    ReplayBuffer,
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize


class CompressedReplayBuffer(ReplayBuffer):
    """
    Replay buffer variant that stores observations in a lower-precision dtype
    to save memory (acts like a lightweight `compress_obs`).

    Notes:
    - Compressing to float16 reduces memory ~2x but adds minor CPU cost
      for casting back to float32 when sampling.
    - Only intended for Box observation spaces.
    """

    def __init__(
        self,
        *args: Any,
        compress_obs: bool = True,
        storage_dtype: np.dtype = np.float16,
        recency_bias_sigma: Optional[float] = None,
        recency_bias_fraction: float = 0.6,
        **kwargs: Any,
    ) -> None:
        self.compress_obs = compress_obs
        self.storage_dtype = storage_dtype
        # If sigma is provided and >0, use it directly.
        # If sigma is None, we can derive a "flat" bias from recency_bias_fraction * max_index.
        # If sigma <= 0, fall back to uniform sampling.
        self.recency_bias_sigma = recency_bias_sigma
        self.recency_bias_fraction = recency_bias_fraction
        super().__init__(*args, **kwargs)

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

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """Sample indices with optional Gaussian recency bias."""
        max_index = self.buffer_size if self.full else self.pos
        if max_index == 0:
            return np.array([], dtype=np.int64)
        # Determine effective sigma
        sigma = self.recency_bias_sigma
        if sigma is None:
            sigma = max(1, int(self.recency_bias_fraction * max_index))
        if sigma <= 0:
            return np.random.randint(0, max_index, size=batch_size, dtype=np.int64)
        center = max_index - 1
        idxs = []
        max_trials = batch_size * 10
        trials = 0
        while len(idxs) < batch_size and trials < max_trials:
            candidate = int(np.random.normal(loc=center, scale=sigma))
            trials += 1
            if 0 <= candidate < max_index:
                idxs.append(candidate)
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


class CompressedDictReplayBuffer(DictReplayBuffer):
    """
    DictReplayBuffer that stores Box sub-observations in a lower-precision dtype
    to save memory. optimize_memory_usage is not supported for dict buffers.
    """

    def __init__(
        self,
        *args: Any,
        compress_obs: bool = True,
        storage_dtype: np.dtype = np.float16,
        recency_bias_sigma: Optional[float] = None,
        recency_bias_fraction: float = 0.6,
        **kwargs: Any,
    ) -> None:
        self.compress_obs = compress_obs
        self.storage_dtype = storage_dtype
        self.recency_bias_sigma = recency_bias_sigma
        self.recency_bias_fraction = recency_bias_fraction
        optimize_memory_usage = kwargs.pop("optimize_memory_usage", False)
        if optimize_memory_usage:
            warnings.warn(
                "optimize_memory_usage is not supported for Dict replay buffers; forcing False."
            )
        super().__init__(*args, optimize_memory_usage=False, **kwargs)

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
        max_index = self.buffer_size if self.full else self.pos
        if max_index == 0:
            return np.array([], dtype=np.int64)
        sigma = self.recency_bias_sigma
        if sigma is None:
            sigma = max(1, int(self.recency_bias_fraction * max_index))
        if sigma <= 0:
            return np.random.randint(0, max_index, size=batch_size, dtype=np.int64)
        center = max_index - 1
        idxs = []
        max_trials = batch_size * 10
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
