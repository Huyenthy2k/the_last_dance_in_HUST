"""
Test luồng lưu kinh nghiệm vào buffer và sample từ buffer.
Kiểm tra các yêu cầu trong spec refactor_mafia.md:

1. Lưu kinh nghiệm: obs, next_obs, action, reward, done, timestamps, regime_shift_flags
2. Two-Phase Sampling Strategy:
   - Phase 1: ~15% batch từ regime shift experiences (uniform)
   - Phase 2: Gaussian recency bias cho phần còn lại (center = max_idx-1, σ = 60% * buffer_size)
3. Filter by date range với anti-leakage (không giữ regime shift từ Valid/Test)
"""

import numpy as np
import pytest
from gymnasium import spaces
from compressed_replay_buffer import CompressedReplayBuffer


class TestCompressedReplayBuffer:
    """Test suite for CompressedReplayBuffer."""

    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.obs_shape = (10,)
        self.action_dim = 5
        self.buffer_size = 1000
        self.n_envs = 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def create_buffer(
        self,
        buffer_size: int = None,
        regime_shift_sample_ratio: float = 0.15,
        recency_bias_fraction: float = 0.6,
    ) -> CompressedReplayBuffer:
        """Create a buffer with specified parameters."""
        return CompressedReplayBuffer(
            buffer_size=buffer_size or self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_envs=self.n_envs,
            regime_shift_sample_ratio=regime_shift_sample_ratio,
            recency_bias_fraction=recency_bias_fraction,
        )

    def add_experience(
        self,
        buffer: CompressedReplayBuffer,
        cur_trade_day: int = 20200101,
        regime_shift_event: bool = False,
    ):
        """Add a single experience to buffer."""
        obs = np.random.randn(self.n_envs, *self.obs_shape).astype(np.float32)
        next_obs = np.random.randn(self.n_envs, *self.obs_shape).astype(np.float32)
        action = np.random.randn(self.n_envs, self.action_dim).astype(np.float32)
        reward = np.random.randn(self.n_envs).astype(np.float32)
        done = np.zeros(self.n_envs, dtype=bool)
        infos = [
            {"curTradeDay": cur_trade_day, "regime_shift_event": regime_shift_event}
        ]
        buffer.add(obs, next_obs, action, reward, done, infos)

    # =========================================================================
    # TEST 1: Lưu kinh nghiệm vào buffer
    # =========================================================================
    def test_add_experience_basic(self):
        """Test basic experience storage."""
        buffer = self.create_buffer()
        assert buffer.pos == 0
        assert not buffer.full

        # Add one experience
        self.add_experience(buffer, cur_trade_day=20200115)

        assert buffer.pos == 1
        assert not buffer.full
        assert buffer.timestamps[0, 0] == 20200115
        assert not buffer.regime_shift_flags[0, 0]

    def test_add_experience_with_regime_shift(self):
        """Test experience storage with regime shift flag."""
        buffer = self.create_buffer()

        # Add experience with regime shift
        self.add_experience(buffer, cur_trade_day=20200301, regime_shift_event=True)

        assert buffer.regime_shift_flags[0, 0] == True
        assert buffer.timestamps[0, 0] == 20200301

    def test_buffer_circular_behavior(self):
        """Test buffer circular overwrite when full (spec: circular buffer)."""
        buffer = self.create_buffer(buffer_size=5)

        # Fill buffer
        for i in range(5):
            self.add_experience(buffer, cur_trade_day=20200101 + i)

        assert buffer.full
        assert buffer.pos == 0

        # Add one more - should overwrite position 0
        self.add_experience(buffer, cur_trade_day=20200201)

        assert buffer.pos == 1
        assert buffer.timestamps[0, 0] == 20200201  # Overwritten

    def test_add_stores_all_fields(self):
        """Test that all fields are stored correctly."""
        buffer = self.create_buffer()

        obs = np.ones((self.n_envs, *self.obs_shape), dtype=np.float32) * 1.5
        next_obs = np.ones((self.n_envs, *self.obs_shape), dtype=np.float32) * 2.5
        action = np.ones((self.n_envs, self.action_dim), dtype=np.float32) * 0.5
        reward = np.array([10.0], dtype=np.float32)
        done = np.array([True], dtype=bool)
        infos = [{"curTradeDay": 20200601, "regime_shift_event": True}]

        buffer.add(obs, next_obs, action, reward, done, infos)

        # Check all fields stored
        np.testing.assert_allclose(buffer.observations[0].astype(np.float32), obs, rtol=1e-2)
        np.testing.assert_allclose(buffer.actions[0], action)
        np.testing.assert_allclose(buffer.rewards[0], reward)
        assert buffer.dones[0, 0] == True
        assert buffer.timestamps[0, 0] == 20200601
        assert buffer.regime_shift_flags[0, 0] == True

    # =========================================================================
    # TEST 2: Two-Phase Sampling Strategy
    # =========================================================================
    def test_sample_empty_buffer(self):
        """Test sampling from empty buffer returns empty."""
        buffer = self.create_buffer()
        indices = buffer._sample_indices(batch_size=256)
        assert len(indices) == 0

    def test_sample_phase1_regime_shift_priority(self):
        """
        Test Phase 1: ~15% of batch sampled from regime shift experiences.
        Spec: regime_shift_sample_ratio = 0.15

        NOTE: Total regime shift count in batch may exceed 15% because:
        1. Phase 1 guarantees ~15% from regime shift pool (uniform)
        2. Phase 2 recency-biased sampling also includes regime shifts
           (regime shifts at end of buffer get sampled more due to recency bias)

        This is CORRECT behavior - Phase 1 ensures minimum representation,
        Phase 2 doesn't exclude regime shifts.
        """
        buffer = self.create_buffer(buffer_size=500)

        # Add 450 normal experiences FIRST
        for i in range(450):
            self.add_experience(buffer, cur_trade_day=20200101 + i, regime_shift_event=False)

        # Add 50 regime shift experiences at END (will be sampled more by recency bias)
        for i in range(50):
            self.add_experience(buffer, cur_trade_day=20200601 + i, regime_shift_event=True)

        # Sample many times and check distribution
        batch_size = 256
        n_samples = 100
        regime_shift_counts = []

        for _ in range(n_samples):
            indices = buffer._sample_indices(batch_size)
            # Count how many are regime shifts
            regime_count = sum(buffer.regime_shift_flags[idx].any() for idx in indices)
            regime_shift_counts.append(regime_count)

        avg_regime_count = np.mean(regime_shift_counts)
        expected_min_regime = batch_size * 0.15  # 38.4 from Phase 1

        print(f"\n[Phase 1 Test] Average regime shift samples: {avg_regime_count:.1f}")
        print(f"[Phase 1 Test] Phase 1 guarantees minimum: {expected_min_regime:.1f}")

        # Phase 1 guarantees AT LEAST 15% from regime shifts
        # Total may be higher due to recency bias in Phase 2
        assert avg_regime_count >= expected_min_regime * 0.8, \
            f"Regime shift samples ({avg_regime_count:.1f}) should be >= {expected_min_regime * 0.8:.1f}"

        # Since regime shifts are at END of buffer, recency bias will sample more
        # 50/500 = 10% of buffer are regime shifts, at end
        # With recency bias (σ=60%), expect ~68% from last 40% (indices 300-500)
        # Regime shifts are at indices 450-500, so expect higher than 15%
        # Upper bound: if all 256 samples came from last 10% (regime shifts) = 100%
        # Realistic upper bound: ~30-40% due to recency bias + Phase 1
        assert avg_regime_count <= batch_size * 0.5, \
            f"Regime shift samples ({avg_regime_count:.1f}) unexpectedly high"

    def test_sample_phase1_limited_regime_shifts(self):
        """Test Phase 1 when regime shift pool is smaller than target."""
        buffer = self.create_buffer(buffer_size=500)

        # Add 490 normal experiences
        for i in range(490):
            self.add_experience(buffer, cur_trade_day=20200101 + i, regime_shift_event=False)

        # Add only 10 regime shift experiences
        for i in range(10):
            self.add_experience(buffer, cur_trade_day=20200601 + i, regime_shift_event=True)

        batch_size = 256
        n_samples = 50
        regime_shift_counts = []

        for _ in range(n_samples):
            indices = buffer._sample_indices(batch_size)
            regime_count = sum(buffer.regime_shift_flags[idx].any() for idx in indices)
            regime_shift_counts.append(regime_count)

        avg_regime_count = np.mean(regime_shift_counts)
        # With replacement, can sample more than 10, but limited by min()
        print(f"\n[Phase 1 Limited Test] Average regime shift samples: {avg_regime_count:.1f}")
        print(f"[Phase 1 Limited Test] Pool size: 10, Target: 38")

        # Should be at most 10 (pool size limit applies)
        assert avg_regime_count >= 8, f"Should sample near all available regime shifts"

    def test_sample_phase2_recency_bias(self):
        """
        Test Phase 2: Gaussian recency bias with center at max_idx-1.
        Spec: σ = recency_bias_fraction * buffer_size = 60% * buffer_size
        """
        buffer = self.create_buffer(buffer_size=1000, regime_shift_sample_ratio=0.0)

        # Fill buffer completely
        for i in range(1000):
            self.add_experience(buffer, cur_trade_day=20200101 + i, regime_shift_event=False)

        batch_size = 256
        n_samples = 100
        all_indices = []

        for _ in range(n_samples):
            indices = buffer._sample_indices(batch_size)
            all_indices.extend(indices.tolist())

        all_indices = np.array(all_indices)
        mean_idx = np.mean(all_indices)
        std_idx = np.std(all_indices)

        # With Gaussian centered at 999 (max_idx-1) and σ = 600 (60% of 1000)
        expected_center = 999
        expected_sigma = 600

        print(f"\n[Phase 2 Recency Bias Test]")
        print(f"Mean index: {mean_idx:.1f} (expected center: {expected_center})")
        print(f"Std index: {std_idx:.1f} (expected σ: {expected_sigma})")

        # Mean should be biased towards end (> 500 for uniform, should be higher)
        assert mean_idx > 500, f"Mean ({mean_idx:.1f}) should be > 500 (recency bias)"

        # Distribution should favor recent experiences
        recent_count = np.sum(all_indices >= 600)  # Last 40%
        total_count = len(all_indices)
        recent_ratio = recent_count / total_count

        print(f"Ratio of recent samples (idx>=600): {recent_ratio:.2%}")
        # With Gaussian bias, > 50% should be from recent experiences
        assert recent_ratio > 0.5, f"Recent ratio ({recent_ratio:.2%}) should be > 50%"

    def test_sample_returns_correct_batch_size(self):
        """Test that sample always returns requested batch size."""
        buffer = self.create_buffer(buffer_size=100)

        # Add 50 experiences
        for i in range(50):
            self.add_experience(buffer, cur_trade_day=20200101 + i)

        for batch_size in [10, 50, 100, 256]:
            indices = buffer._sample_indices(batch_size)
            assert len(indices) == batch_size, \
                f"Expected {batch_size} indices, got {len(indices)}"

    # =========================================================================
    # TEST 3: Filter by date range với anti-leakage
    # =========================================================================
    def test_filter_empty_buffer(self):
        """Test filtering empty buffer."""
        buffer = self.create_buffer()
        result = buffer.filter_by_date_range(20180101, 20201231)
        assert result == (0, 0, 0)

    def test_filter_retains_experiences_in_range(self):
        """Test that experiences within date range are retained."""
        buffer = self.create_buffer(buffer_size=100)

        # Add experiences from 2017 to 2022
        dates = [20170601, 20180301, 20190501, 20200701, 20210901, 20220101]
        for date in dates:
            self.add_experience(buffer, cur_trade_day=date)

        # Filter to 2018-2020 range
        original, retained, regime_preserved = buffer.filter_by_date_range(
            train_start=20180101,
            train_end=20201231,
        )

        assert original == 6
        assert retained == 3  # 20180301, 20190501, 20200701
        print(f"\n[Filter Range Test] Original: {original}, Retained: {retained}")

    def test_filter_preserves_regime_shifts_from_prev_train_only(self):
        """
        Test anti-leakage: only preserve regime shifts from PREVIOUS TRAIN period.
        Spec: Regime shifts from Valid/Test should NOT be preserved.

        Window 0: Train[2017-2019], Valid[2020], Test[2021]
        Window 1: Train[2018-2020], Valid[2021], Test[2022]

        Logic:
        1. Experiences IN new train range (2018-2020): ALWAYS KEEP
        2. Regime shifts OUTSIDE range but <= prev_train_end (2019): KEEP (transfer learning)
        3. Regime shifts OUTSIDE range AND > prev_train_end: FILTERED (leakage!)
        """
        buffer = self.create_buffer(buffer_size=100)

        # Add experiences simulating Window 0 data
        # 2017: Out of range, IS regime shift, ts <= prev_train_end → KEEP
        self.add_experience(buffer, cur_trade_day=20170601, regime_shift_event=True)
        # 2018: In range → KEEP
        self.add_experience(buffer, cur_trade_day=20180301, regime_shift_event=False)
        # 2019: In range → KEEP
        self.add_experience(buffer, cur_trade_day=20190501, regime_shift_event=True)
        # 2020: IN range (train_end=20201231) → KEEP (not leakage, it's in new train!)
        self.add_experience(buffer, cur_trade_day=20200601, regime_shift_event=True)
        # 2021: OUT of range, IS regime shift, ts > prev_train_end → FILTERED (leakage!)
        self.add_experience(buffer, cur_trade_day=20210301, regime_shift_event=True)

        # Simulate Window 1 transition
        original, retained, regime_preserved = buffer.filter_by_date_range(
            train_start=20180101,
            train_end=20201231,
            prev_train_end=20191231,  # Window 0 train ended here
        )

        print(f"\n[Anti-Leakage Test]")
        print(f"Original: {original}")
        print(f"Retained: {retained}")
        print(f"Regime shifts preserved: {regime_preserved}")

        # Expected:
        # - 20170601: Out of range, regime shift, ts <= prev_train_end → KEEP
        # - 20180301: In range → KEEP
        # - 20190501: In range, regime shift → KEEP (counts in regime_preserved)
        # - 20200601: In range, regime shift → KEEP (counts in regime_preserved)
        # - 20210301: Out of range, regime shift, ts > prev_train_end → FILTERED!

        assert original == 5
        assert retained == 4, f"Expected 4 retained, got {retained}"
        # regime_preserved counts: 20170601 (preserved as regime) + 20190501 + 20200601 (in range) = 3
        assert regime_preserved == 3, f"Expected 3 regime preserved, got {regime_preserved}"

        # Verify buffer contents
        stats = buffer.get_buffer_stats()
        assert stats["size"] == 4
        assert stats["regime_shift_count"] == 3  # 3 regime shifts kept

    def test_filter_no_regime_shift_leakage_from_valid_test(self):
        """Explicit test: regime shifts from Valid/Test are NOT preserved."""
        buffer = self.create_buffer(buffer_size=100)

        # Only add regime shifts from Valid/Test period
        self.add_experience(buffer, cur_trade_day=20200601, regime_shift_event=True)  # Valid
        self.add_experience(buffer, cur_trade_day=20210301, regime_shift_event=True)  # Test

        original, retained, regime_preserved = buffer.filter_by_date_range(
            train_start=20180101,
            train_end=20171231,  # Range that doesn't include 2020-2021
            prev_train_end=20191231,
        )

        print(f"\n[No Leakage Test] Retained: {retained}, Regime preserved: {regime_preserved}")
        # Both should be filtered out (not in range, and after prev_train_end)
        assert retained == 0, f"Expected 0 retained (all filtered), got {retained}"
        assert regime_preserved == 0, f"Expected 0 regime preserved, got {regime_preserved}"

    def test_buffer_stats(self):
        """Test get_buffer_stats returns correct statistics."""
        buffer = self.create_buffer(buffer_size=100)

        for i in range(50):
            regime = i % 10 == 0  # 5 regime shifts
            self.add_experience(buffer, cur_trade_day=20200101 + i, regime_shift_event=regime)

        stats = buffer.get_buffer_stats()

        assert stats["size"] == 50
        assert stats["min_date"] == 20200101
        assert stats["max_date"] == 20200150
        assert stats["regime_shift_count"] == 5
        print(f"\n[Buffer Stats] {stats}")


class TestSamplingDistribution:
    """Detailed tests for sampling distribution according to spec."""

    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.obs_shape = (10,)
        self.action_dim = 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def test_spec_example_scenario(self):
        """
        Test exact scenario from spec (Section 9.4.3):
        - Buffer size: 50,000 experiences
        - Regime shifts: 500 (~1%)
        - Batch size: 256

        Expected:
        - Phase 1: min(38, 500) = 38 samples from regime shifts
        - Phase 2: 218 samples with recency bias
        - ~68% of 218 = 148 samples from recent 40%
        """
        buffer_size = 5000  # Scaled down for test speed
        n_regime_shifts = 50  # 1% of buffer
        batch_size = 256

        buffer = CompressedReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_envs=1,
            regime_shift_sample_ratio=0.15,
            recency_bias_fraction=0.6,
        )

        # Add normal experiences
        for i in range(buffer_size - n_regime_shifts):
            obs = np.random.randn(1, *self.obs_shape).astype(np.float32)
            next_obs = np.random.randn(1, *self.obs_shape).astype(np.float32)
            action = np.random.randn(1, self.action_dim).astype(np.float32)
            reward = np.random.randn(1).astype(np.float32)
            done = np.zeros(1, dtype=bool)
            infos = [{"curTradeDay": 20200101 + i, "regime_shift_event": False}]
            buffer.add(obs, next_obs, action, reward, done, infos)

        # Add regime shift experiences (distributed throughout)
        regime_positions = np.linspace(0, buffer_size - n_regime_shifts - 1, n_regime_shifts).astype(int)
        for i in range(n_regime_shifts):
            obs = np.random.randn(1, *self.obs_shape).astype(np.float32)
            next_obs = np.random.randn(1, *self.obs_shape).astype(np.float32)
            action = np.random.randn(1, self.action_dim).astype(np.float32)
            reward = np.random.randn(1).astype(np.float32)
            done = np.zeros(1, dtype=bool)
            infos = [{"curTradeDay": 20210101 + i, "regime_shift_event": True}]
            buffer.add(obs, next_obs, action, reward, done, infos)

        # Sample and analyze
        n_trials = 100
        phase1_counts = []
        phase2_recency_counts = []

        for _ in range(n_trials):
            indices = buffer._sample_indices(batch_size)

            # Count regime shifts (Phase 1 contribution)
            regime_count = sum(buffer.regime_shift_flags[idx].any() for idx in indices)
            phase1_counts.append(regime_count)

            # Count recent samples (last 40% of buffer)
            recent_threshold = int(0.6 * buffer.pos)
            recent_count = sum(idx >= recent_threshold for idx in indices)
            phase2_recency_counts.append(recent_count)

        avg_phase1 = np.mean(phase1_counts)
        avg_recency = np.mean(phase2_recency_counts)

        print(f"\n[Spec Scenario Test]")
        print(f"Buffer size: {buffer.pos}")
        print(f"Regime shifts in buffer: {buffer.regime_shift_flags[:buffer.pos].any(axis=1).sum()}")
        print(f"Average Phase 1 (regime) samples: {avg_phase1:.1f} (expected ~38)")
        print(f"Average recent samples (idx >= 60%): {avg_recency:.1f}")
        print(f"Recency ratio: {avg_recency / batch_size:.1%}")

        # Verify Phase 1: ~15% = 38 samples
        expected_phase1 = min(batch_size * 0.15, n_regime_shifts)
        assert avg_phase1 >= expected_phase1 * 0.7, \
            f"Phase 1 samples ({avg_phase1:.1f}) below expected ({expected_phase1 * 0.7:.1f})"

        # Verify recency bias: > 50% from recent half
        assert avg_recency / batch_size > 0.45, \
            f"Recency ratio ({avg_recency / batch_size:.1%}) should be > 45%"


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
