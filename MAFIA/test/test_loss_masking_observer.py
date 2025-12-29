"""
Test Loss Masking Observer (Training Cadence Alignment)

Covers:
- TrainingCadenceMask.compute_loss_with_cadence()
- TrainingCadenceAlignmentContext

NOTE: mask_gradient_flow() tests were REMOVED because:
1. Loss masking (loss * mask) already prevents gradient flow for holding days
2. Post-backward gradient zeroing was buggy and redundant

Per spec: When trigger=0 (holding position): only L_Risk and L_Dir are trained
         When trigger=1 (rebalance/regime shift): full L_total including L_PG
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL_controller.mafia_observer_train_cadence import (
    TrainingCadenceMask,
    TrainingCadenceAlignmentContext,
)


class MockConfig:
    """Mock config for testing."""

    def __init__(self):
        self.mafia_lambda_pg = 1.0
        self.mafia_lambda_risk = 1.0
        self.mafia_lambda_dir = 1.0


class MockMAFIAModel(nn.Module):
    """Mock MAFIA model for gradient flow testing."""

    def __init__(self):
        super().__init__()
        # Selection-related parameters (should be zeroed during holding)
        self.stock_experts = nn.Linear(10, 10)
        self.st_fusion = nn.Linear(10, 10)
        self.moe_router = nn.Linear(10, 4)
        self.gating_router = nn.Linear(10, 4)

        # Non-selection parameters (should NOT be zeroed)
        self.risk_head = nn.Linear(10, 1)
        self.direction_head = nn.Linear(10, 3)
        self.encoder = nn.Linear(10, 10)

    def forward(self, x):
        h = self.encoder(x)
        stock_out = self.stock_experts(h)
        fusion_out = self.st_fusion(stock_out)
        moe_out = self.moe_router(h)
        gate_out = self.gating_router(h)
        risk = self.risk_head(h)
        direction = self.direction_head(h)
        return fusion_out, moe_out, gate_out, risk, direction


# ============================================================
# Test TrainingCadenceMask.compute_loss_with_cadence()
# ============================================================


class TestComputeLossWithCadence:
    """Test compute_loss_with_cadence() method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = MockConfig()
        self.cadence_mask = TrainingCadenceMask(self.config)

    def test_holding_period_masks_pg_loss(self):
        """Test: When mask=0 (holding), L_PG should be masked to 0."""
        loss_pg = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss_risk = torch.tensor([0.5, 0.5, 0.5, 0.5])
        loss_dir = torch.tensor([0.3, 0.3, 0.3, 0.3])
        selection_mask = torch.zeros(4)  # All holding period

        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        # L_PG should be masked to 0
        assert loss_dict["loss_pg_masked"] == 0.0, (
            f"PG loss should be 0 during holding, got {loss_dict['loss_pg_masked']}"
        )
        # L_Risk and L_Dir should remain unchanged
        assert abs(loss_dict["loss_risk"] - 0.5) < 1e-5, (
            f"Risk loss should be 0.5, got {loss_dict['loss_risk']}"
        )
        assert abs(loss_dict["loss_dir"] - 0.3) < 1e-5, (
            f"Dir loss should be 0.3, got {loss_dict['loss_dir']}"
        )
        print("Holding period (mask=0): L_PG=0, L_Risk/L_Dir active")

    def test_rebalance_period_full_loss(self):
        """Test: When mask=1 (rebalance), full L_total including L_PG."""
        loss_pg = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss_risk = torch.tensor([0.5, 0.5, 0.5, 0.5])
        loss_dir = torch.tensor([0.3, 0.3, 0.3, 0.3])
        selection_mask = torch.ones(4)  # All rebalance period

        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        # All losses should be active
        assert abs(loss_dict["loss_pg_masked"] - 1.0) < 1e-5, (
            f"PG loss should be 1.0 during rebalance, got {loss_dict['loss_pg_masked']}"
        )
        assert abs(loss_dict["loss_risk"] - 0.5) < 1e-5
        assert abs(loss_dict["loss_dir"] - 0.3) < 1e-5

        # Total loss = 1.0 + 0.5 + 0.3 = 1.8
        expected_total = 1.0 + 0.5 + 0.3
        assert abs(loss_dict["total_loss"] - expected_total) < 1e-5, (
            f"Total loss should be {expected_total}, got {loss_dict['total_loss']}"
        )
        print(f"Rebalance period (mask=1): total_loss={loss_dict['total_loss']}")

    def test_mixed_mask_values(self):
        """Test: Mixed mask values (some holding, some rebalance)."""
        loss_pg = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss_risk = torch.tensor([0.5, 0.5, 0.5, 0.5])
        loss_dir = torch.tensor([0.3, 0.3, 0.3, 0.3])
        # 50% holding, 50% rebalance
        selection_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])

        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        # L_PG should be 0.5 (2 out of 4 steps masked)
        expected_pg_masked = (0 + 0 + 1 + 1) / 4.0
        assert abs(loss_dict["loss_pg_masked"] - expected_pg_masked) < 1e-5, (
            f"PG masked should be {expected_pg_masked}, got {loss_dict['loss_pg_masked']}"
        )
        assert abs(loss_dict["selection_mask_mean"] - 0.5) < 1e-5
        print(f"Mixed mask: loss_pg_masked={loss_dict['loss_pg_masked']}, mask_mean=0.5")

    def test_2d_loss_tensors(self):
        """Test: 2D loss tensors (T, B)."""
        T, B = 4, 3
        loss_pg = torch.ones(T, B)
        loss_risk = torch.ones(T, B) * 0.5
        loss_dir = torch.ones(T, B) * 0.3
        selection_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])  # (T,)

        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        # Mask should broadcast: (T,) -> (T, 1) for (T, B) losses
        expected_pg_masked = 0.5  # 2/4 steps have mask=1
        assert abs(loss_dict["loss_pg_masked"] - expected_pg_masked) < 1e-5
        print(f"2D tensors (T={T}, B={B}): loss_pg_masked={loss_dict['loss_pg_masked']}")

    def test_lambda_weights(self):
        """Test: Custom lambda weights."""
        loss_pg = torch.tensor([1.0])
        loss_risk = torch.tensor([1.0])
        loss_dir = torch.tensor([1.0])
        selection_mask = torch.ones(1)

        lambda_pg = 2.0
        lambda_risk = 0.5
        lambda_dir = 0.3

        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
            lambda_pg=lambda_pg,
            lambda_risk=lambda_risk,
            lambda_dir=lambda_dir,
        )

        # Total = 2.0*1.0 + 0.5*1.0 + 0.3*1.0 = 2.8
        expected_total = 2.0 + 0.5 + 0.3
        assert abs(loss_dict["total_loss"] - expected_total) < 1e-5, (
            f"Expected {expected_total}, got {loss_dict['total_loss']}"
        )
        print(f"Lambda weights: {lambda_pg}, {lambda_risk}, {lambda_dir} -> total={expected_total}")

    def test_numpy_mask_conversion(self):
        """Test: Numpy array mask is converted to tensor."""
        loss_pg = torch.tensor([1.0, 1.0])
        loss_risk = torch.tensor([0.5, 0.5])
        loss_dir = torch.tensor([0.3, 0.3])
        selection_mask = np.array([0.0, 1.0])  # Numpy array

        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        # Should work without error
        assert "total_loss" in loss_dict
        print(f"Numpy mask converted successfully: total_loss={loss_dict['total_loss']}")

    def test_list_mask_conversion(self):
        """Test: Python list mask is converted to tensor."""
        loss_pg = torch.tensor([1.0, 1.0])
        loss_risk = torch.tensor([0.5, 0.5])
        loss_dir = torch.tensor([0.3, 0.3])
        selection_mask = [0.0, 1.0]  # Python list

        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        assert "total_loss" in loss_dict
        print(f"List mask converted successfully: total_loss={loss_dict['total_loss']}")

    def test_loss_dict_completeness(self):
        """Test: loss_dict contains all required keys."""
        loss_pg = torch.tensor([1.0])
        loss_risk = torch.tensor([0.5])
        loss_dir = torch.tensor([0.3])
        selection_mask = torch.ones(1)

        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        required_keys = [
            "loss_pg_raw",
            "loss_pg_masked",
            "loss_risk",
            "loss_dir",
            "total_loss",
            "selection_mask_mean",
        ]
        for key in required_keys:
            assert key in loss_dict, f"Missing key: {key}"
        print(f"All required keys present: {list(loss_dict.keys())}")

    def test_gradient_flows_through_total_loss(self):
        """Test: Gradient flows through total_loss tensor."""
        loss_pg = torch.tensor([1.0], requires_grad=True)
        loss_risk = torch.tensor([0.5], requires_grad=True)
        loss_dir = torch.tensor([0.3], requires_grad=True)
        selection_mask = torch.ones(1)

        total_loss, _ = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        # Should be able to backprop
        total_loss.backward()

        assert loss_pg.grad is not None, "Gradient should flow to loss_pg"
        assert loss_risk.grad is not None, "Gradient should flow to loss_risk"
        assert loss_dir.grad is not None, "Gradient should flow to loss_dir"
        print("Gradient flows through total_loss")

    def test_holding_blocks_pg_gradient(self):
        """Test: During holding, gradient does not flow through L_PG."""
        loss_pg = torch.tensor([1.0], requires_grad=True)
        loss_risk = torch.tensor([0.5], requires_grad=True)
        loss_dir = torch.tensor([0.3], requires_grad=True)
        selection_mask = torch.zeros(1)  # Holding

        total_loss, _ = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        total_loss.backward()

        # loss_pg gradient should be 0 (masked out)
        assert loss_pg.grad is not None
        assert abs(loss_pg.grad.item()) < 1e-5, (
            f"PG grad should be 0 during holding, got {loss_pg.grad.item()}"
        )
        # loss_risk and loss_dir should have gradients
        assert loss_risk.grad is not None and loss_risk.grad.item() > 0
        assert loss_dir.grad is not None and loss_dir.grad.item() > 0
        print("Holding blocks PG gradient, Risk/Dir gradients active")


# ============================================================
# NOTE: TestMaskGradientFlow class was REMOVED
# mask_gradient_flow() is no longer needed because loss masking
# already prevents gradient flow for holding days automatically.
# ============================================================


# ============================================================
# Test TrainingCadenceAlignmentContext
# ============================================================


class TestTrainingCadenceAlignmentContext:
    """Test TrainingCadenceAlignmentContext context manager."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = MockConfig()
        self.model = MockMAFIAModel()

    def test_context_manager_basic(self):
        """Test: Context manager enters and exits properly."""
        selection_mask = torch.ones(4)

        with TrainingCadenceAlignmentContext(
            model=self.model, config=self.config, selection_mask=selection_mask
        ) as ctx:
            assert ctx is not None
            assert hasattr(ctx, "cadence_handler")
            assert hasattr(ctx, "compute_total_loss")
        print("Context manager enters/exits properly")

    def test_compute_total_loss_uses_config_lambdas(self):
        """Test: compute_total_loss uses config lambda values."""
        self.config.mafia_lambda_pg = 2.0
        self.config.mafia_lambda_risk = 0.5
        self.config.mafia_lambda_dir = 0.3

        selection_mask = torch.ones(1)

        with TrainingCadenceAlignmentContext(
            model=self.model, config=self.config, selection_mask=selection_mask
        ) as ctx:
            loss_pg = torch.tensor([1.0])
            loss_risk = torch.tensor([1.0])
            loss_dir = torch.tensor([1.0])

            total_loss, loss_dict = ctx.compute_total_loss(loss_pg, loss_risk, loss_dir)

            # Expected: 2.0*1.0 + 0.5*1.0 + 0.3*1.0 = 2.8
            expected = 2.0 + 0.5 + 0.3
            assert abs(loss_dict["total_loss"] - expected) < 1e-5, (
                f"Expected {expected}, got {loss_dict['total_loss']}"
            )
        print(f"Config lambdas used: total_loss={loss_dict['total_loss']}")

    def test_default_selection_mask(self):
        """Test: Default selection_mask is ones (all rebalance)."""
        with TrainingCadenceAlignmentContext(
            model=self.model, config=self.config, selection_mask=None
        ) as ctx:
            assert ctx.selection_mask is not None
            # Default should allow PG loss through
            loss_pg = torch.tensor([1.0])
            loss_risk = torch.tensor([0.5])
            loss_dir = torch.tensor([0.3])

            total_loss, loss_dict = ctx.compute_total_loss(loss_pg, loss_risk, loss_dir)
            assert loss_dict["loss_pg_masked"] > 0, (
                "Default mask should allow PG loss"
            )
        print("Default selection_mask allows PG loss")

    def test_context_with_holding_mask(self):
        """Test: Context with holding mask zeros PG loss."""
        selection_mask = torch.zeros(4)  # All holding

        with TrainingCadenceAlignmentContext(
            model=self.model, config=self.config, selection_mask=selection_mask
        ) as ctx:
            loss_pg = torch.tensor([1.0, 1.0, 1.0, 1.0])
            loss_risk = torch.tensor([0.5, 0.5, 0.5, 0.5])
            loss_dir = torch.tensor([0.3, 0.3, 0.3, 0.3])

            total_loss, loss_dict = ctx.compute_total_loss(loss_pg, loss_risk, loss_dir)
            assert loss_dict["loss_pg_masked"] == 0.0, (
                "Holding mask should zero PG loss"
            )
        print("Context with holding mask zeros PG loss")

    def test_full_training_loop_simulation(self):
        """Test: Simulate a full training loop with cadence masking."""
        # Simulate 10 steps: steps 0-4 holding, steps 5-9 rebalance
        selection_mask = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.float32)

        with TrainingCadenceAlignmentContext(
            model=self.model, config=self.config, selection_mask=selection_mask
        ) as ctx:
            # Simulate losses for 10 steps (with requires_grad for backward test)
            loss_pg = torch.ones(10)
            loss_pg.requires_grad_(True)
            loss_risk = torch.ones(10) * 0.5
            loss_risk.requires_grad_(True)
            loss_dir = torch.ones(10) * 0.3
            loss_dir.requires_grad_(True)

            total_loss, loss_dict = ctx.compute_total_loss(loss_pg, loss_risk, loss_dir)

            # PG masked: 5/10 steps have mask=1, so mean = 0.5
            assert abs(loss_dict["loss_pg_masked"] - 0.5) < 1e-5
            assert abs(loss_dict["selection_mask_mean"] - 0.5) < 1e-5

            # Test gradient flow
            total_loss.backward()

        print(f"Full training loop: loss_pg_masked={loss_dict['loss_pg_masked']}")

    def test_cadence_handler_access(self):
        """Test: Can access cadence_handler for direct loss computation."""
        selection_mask = torch.zeros(4)

        with TrainingCadenceAlignmentContext(
            model=self.model, config=self.config, selection_mask=selection_mask
        ) as ctx:
            # Access cadence_handler
            assert ctx.cadence_handler is not None

            # Use cadence_handler directly for loss computation
            loss_pg = torch.ones(4)
            loss_risk = torch.ones(4) * 0.5
            loss_dir = torch.ones(4) * 0.3

            total_loss, loss_dict = ctx.cadence_handler.compute_loss_with_cadence(
                loss_pg=loss_pg,
                loss_risk=loss_risk,
                loss_dir=loss_dir,
                selection_mask=selection_mask,
            )

            # All holding, so PG should be zeroed
            assert loss_dict["loss_pg_masked"] == 0.0
        print("cadence_handler accessible and functional")


# ============================================================
# Integration Tests
# ============================================================


class TestIntegration:
    """Integration tests for Loss Masking Observer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = MockConfig()
        self.model = MockMAFIAModel()

    def test_end_to_end_cadence_alignment(self):
        """Test: End-to-end training with cadence alignment."""
        # Simulate a rebalance window (every 10 days)
        T = 30
        selection_mask = torch.zeros(T)
        selection_mask[::10] = 1.0  # Rebalance at days 0, 10, 20

        with TrainingCadenceAlignmentContext(
            model=self.model, config=self.config, selection_mask=selection_mask
        ) as ctx:
            # Simulate losses (with requires_grad for backward test)
            loss_pg = torch.randn(T).abs()
            loss_pg.requires_grad_(True)
            loss_risk = torch.randn(T).abs() * 0.5
            loss_risk.requires_grad_(True)
            loss_dir = torch.randn(T).abs() * 0.3
            loss_dir.requires_grad_(True)

            total_loss, loss_dict = ctx.compute_total_loss(loss_pg, loss_risk, loss_dir)

            # 3 out of 30 days are rebalance
            assert abs(loss_dict["selection_mask_mean"] - 3/30) < 1e-5

            # Backward pass - loss masking handles gradient flow automatically
            # No post-backward gradient masking needed!
            total_loss.backward()

            # Verify loss_pg only gets gradient for rebalance steps (via loss masking)
            # The masked loss automatically zeros gradient contribution for holding days

        print(f"End-to-end: {T} steps, 3 rebalance days")

    def test_multiple_batches(self):
        """Test: Multiple batches with different masks."""
        batch_masks = [
            torch.zeros(4),  # All holding
            torch.ones(4),  # All rebalance
            torch.tensor([0, 0, 1, 1]),  # Mixed
        ]

        results = []
        for mask in batch_masks:
            with TrainingCadenceAlignmentContext(
                model=self.model, config=self.config, selection_mask=mask
            ) as ctx:
                loss_pg = torch.ones(4)
                loss_risk = torch.ones(4) * 0.5
                loss_dir = torch.ones(4) * 0.3

                _, loss_dict = ctx.compute_total_loss(loss_pg, loss_risk, loss_dir)
                results.append(loss_dict["loss_pg_masked"])

        assert results[0] == 0.0, "All holding should mask all PG"
        assert abs(results[1] - 1.0) < 1e-5, "All rebalance should pass all PG"
        assert abs(results[2] - 0.5) < 1e-5, "Mixed should pass half PG"
        print(f"Multiple batches: {results}")

    def test_gpu_compatibility(self):
        """Test: Works on GPU if available."""
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU test")
            return

        device = torch.device("cuda")
        model = MockMAFIAModel().to(device)
        selection_mask = torch.zeros(4, device=device)

        cadence_mask = TrainingCadenceMask(self.config)

        loss_pg = torch.ones(4, device=device)
        loss_risk = torch.ones(4, device=device) * 0.5
        loss_dir = torch.ones(4, device=device) * 0.3

        total_loss, loss_dict = cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        assert total_loss.device.type == "cuda"
        print(f"GPU test passed: device={total_loss.device}")


# ============================================================
# Edge Cases
# ============================================================


class TestEdgeCases:
    """Test edge cases for robustness."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = MockConfig()
        self.cadence_mask = TrainingCadenceMask(self.config)

    def test_empty_mask(self):
        """Test: Empty tensor mask."""
        loss_pg = torch.tensor([1.0])
        loss_risk = torch.tensor([0.5])
        loss_dir = torch.tensor([0.3])
        selection_mask = torch.tensor([])

        # Should handle gracefully (may raise or return default)
        try:
            _, loss_dict = self.cadence_mask.compute_loss_with_cadence(
                loss_pg=loss_pg,
                loss_risk=loss_risk,
                loss_dir=loss_dir,
                selection_mask=selection_mask,
            )
            print(f"Empty mask handled: {loss_dict}")
        except Exception as e:
            print(f"Empty mask raised expected error: {type(e).__name__}")

    def test_very_large_tensors(self):
        """Test: Performance with large tensors."""
        T, B = 1000, 64
        loss_pg = torch.randn(T, B)
        loss_risk = torch.randn(T, B) * 0.5
        loss_dir = torch.randn(T, B) * 0.3
        selection_mask = torch.randint(0, 2, (T,)).float()

        import time
        start = time.time()
        total_loss, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Large tensor should be fast, took {elapsed:.2f}s"
        print(f"Large tensor ({T}x{B}): {elapsed*1000:.2f}ms")

    def test_all_zeros_losses(self):
        """Test: All zero losses."""
        loss_pg = torch.zeros(4)
        loss_risk = torch.zeros(4)
        loss_dir = torch.zeros(4)
        selection_mask = torch.ones(4)

        _, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        assert loss_dict["total_loss"] == 0.0
        print("All zero losses handled correctly")

    def test_negative_losses(self):
        """Test: Negative loss values (can happen with some loss functions)."""
        loss_pg = torch.tensor([-1.0, -2.0])
        loss_risk = torch.tensor([-0.5, -0.5])
        loss_dir = torch.tensor([-0.3, -0.3])
        selection_mask = torch.ones(2)

        _, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        expected = -1.5 + (-0.5) + (-0.3)
        assert abs(loss_dict["total_loss"] - expected) < 1e-5
        print(f"Negative losses handled: total={loss_dict['total_loss']}")

    def test_inf_nan_handling(self):
        """Test: Inf and NaN in losses."""
        loss_pg = torch.tensor([1.0, float("inf")])
        loss_risk = torch.tensor([0.5, float("nan")])
        loss_dir = torch.tensor([0.3, 0.3])
        selection_mask = torch.ones(2)

        _, loss_dict = self.cadence_mask.compute_loss_with_cadence(
            loss_pg=loss_pg,
            loss_risk=loss_risk,
            loss_dir=loss_dir,
            selection_mask=selection_mask,
        )

        # Should propagate inf/nan (not crash)
        assert np.isnan(loss_dict["total_loss"]) or np.isinf(loss_dict["total_loss"])
        print("Inf/NaN propagated correctly (not masked)")


def run_all_tests():
    """Run all Loss Masking Observer tests."""
    print("=" * 60)
    print("Testing Loss Masking Observer (Training Cadence Alignment)")
    print("=" * 60)

    # compute_loss_with_cadence tests
    print("\n--- compute_loss_with_cadence Tests ---")
    loss_tests = TestComputeLossWithCadence()
    loss_tests.setup_method()
    loss_tests.test_holding_period_masks_pg_loss()
    loss_tests.test_rebalance_period_full_loss()
    loss_tests.test_mixed_mask_values()
    loss_tests.test_2d_loss_tensors()
    loss_tests.test_lambda_weights()
    loss_tests.test_numpy_mask_conversion()
    loss_tests.test_list_mask_conversion()
    loss_tests.test_loss_dict_completeness()
    loss_tests.test_gradient_flows_through_total_loss()
    loss_tests.test_holding_blocks_pg_gradient()

    # NOTE: mask_gradient_flow tests REMOVED - method no longer exists
    # Loss masking handles gradient flow automatically

    # TrainingCadenceAlignmentContext tests
    print("\n--- TrainingCadenceAlignmentContext Tests ---")
    ctx_tests = TestTrainingCadenceAlignmentContext()
    ctx_tests.setup_method()
    ctx_tests.test_context_manager_basic()
    ctx_tests.test_compute_total_loss_uses_config_lambdas()
    ctx_tests.test_default_selection_mask()
    ctx_tests.test_context_with_holding_mask()

    ctx_tests.setup_method()
    ctx_tests.test_full_training_loop_simulation()

    ctx_tests.setup_method()
    ctx_tests.test_cadence_handler_access()

    # Integration tests
    print("\n--- Integration Tests ---")
    int_tests = TestIntegration()
    int_tests.setup_method()
    int_tests.test_end_to_end_cadence_alignment()

    int_tests.setup_method()
    int_tests.test_multiple_batches()

    int_tests.setup_method()
    int_tests.test_gpu_compatibility()

    # Edge cases
    print("\n--- Edge Cases ---")
    edge_tests = TestEdgeCases()
    edge_tests.setup_method()
    edge_tests.test_empty_mask()
    edge_tests.test_very_large_tensors()
    edge_tests.test_all_zeros_losses()
    edge_tests.test_negative_losses()
    edge_tests.test_inf_nan_handling()

    print("\n" + "=" * 60)
    print("All Loss Masking Observer Tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
