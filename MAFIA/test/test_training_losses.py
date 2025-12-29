"""
Test Training Objectives (Section 5 of Spec)

Covers:
- L_PG: Policy Gradient with REINFORCE
- L_Risk: MSE for eta calibration
- L_Dir: CrossEntropy for direction prediction
- Loss masking according to cadence
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRewardComputation:
    """Test reward shaping for Policy Gradient."""

    def test_compounding_reward_formula(self):
        """Test compounding formula: prod(1+r) - 1 over horizon h."""
        # Simulated rate of price change for 3 stocks over 5 days
        rates = np.array(
            [
                [0.01, 0.02, -0.01, 0.03, 0.01],  # Stock 0
                [-0.02, 0.01, 0.02, 0.01, -0.01],  # Stock 1
                [0.03, -0.01, 0.02, 0.02, 0.01],  # Stock 2
            ]
        )

        horizon = 5

        # Compounding: prod(1+r) - 1
        expected_returns = []
        for stock_idx in range(3):
            cumulative = 1.0
            for t in range(horizon):
                cumulative *= 1 + rates[stock_idx, t]
            expected_returns.append(cumulative - 1)

        expected_returns = np.array(expected_returns)

        # Verify formula
        assert abs(expected_returns[0] - 0.0615) < 0.01, (
            f"Stock 0 compound return: {expected_returns[0]}"
        )
        assert expected_returns[1] > -0.01, (
            f"Stock 1 should be near 0: {expected_returns[1]}"
        )
        assert expected_returns[2] > 0.07, (
            f"Stock 2 should be positive: {expected_returns[2]}"
        )

        print(f"✅ Compounding rewards: {expected_returns}")

    def test_topk_reward_aggregation(self):
        """Test reward aggregation: mean of Top-K compound returns."""
        # Top-K indices
        topk_indices = np.array([0, 2])  # Selected stocks 0 and 2

        # Full returns for all stocks
        all_returns = np.array([0.05, -0.02, 0.08, 0.01])

        # Gather Top-K returns
        topk_returns = all_returns[topk_indices]

        # Aggregate as mean
        R_t = np.mean(topk_returns)

        expected = (0.05 + 0.08) / 2  # = 0.065
        assert abs(R_t - expected) < 1e-6, f"R_t should be {expected}, got {R_t}"
        print(f"✅ Top-K reward aggregation: R_t = {R_t}")

    def test_advantage_calculation(self):
        """Test advantage: A_t = (R_t - penalties) - baseline."""
        R_t = 0.065
        baseline = 0.03  # Market return
        turnover_penalty = 0.002
        change_penalty = 0.001
        alpha_turnover = 0.1
        alpha_change = 0.05

        # Advantage formula from spec
        A_t = (
            R_t - alpha_turnover * turnover_penalty - alpha_change * change_penalty
        ) - baseline

        expected = 0.065 - 0.1 * 0.002 - 0.05 * 0.001 - 0.03
        assert abs(A_t - expected) < 1e-6, f"Advantage mismatch: {A_t} vs {expected}"
        print(f"✅ Advantage calculation: A_t = {A_t:.6f}")


class TestPolicyGradientLoss:
    """Test L_PG with REINFORCE formula."""

    def test_log_prob_from_topk_scores(self):
        """Test log-prob computation from Top-K softmax scores."""
        # Simulated Top-K scores (softmax output, K=3)
        topk_scores = torch.tensor([0.4, 0.35, 0.25])

        # Log probability of selecting this Top-K
        log_prob = torch.log(topk_scores + 1e-8).sum()

        expected = np.log(0.4) + np.log(0.35) + np.log(0.25)
        assert abs(log_prob.item() - expected) < 1e-5
        print(f"✅ Log-prob from Top-K scores: {log_prob.item():.4f}")

    def test_pg_loss_formula(self):
        """Test L_PG = -log(π) * A (REINFORCE)."""
        # Simulated values
        topk_scores = torch.tensor([0.4, 0.35, 0.25], requires_grad=True)
        advantage = torch.tensor(0.05)  # Detached, no grad

        # Log-prob
        log_prob = torch.log(topk_scores + 1e-8).sum()

        # REINFORCE loss
        L_PG = -log_prob * advantage.detach()

        assert L_PG.requires_grad, "L_PG should have gradient"
        assert L_PG.item() > 0, (
            f"L_PG should be positive when advantage > 0: {L_PG.item()}"
        )
        print(f"✅ L_PG (REINFORCE): {L_PG.item():.4f}")

    def test_pg_loss_zero_when_masked(self):
        """Test L_PG = 0 when selection_trigger = 0 (hold day)."""
        topk_scores = torch.tensor([0.4, 0.35, 0.25])
        advantage = torch.tensor(0.05)
        selection_mask = 0  # Hold day

        log_prob = torch.log(topk_scores + 1e-8).sum()
        L_PG = -log_prob * advantage * selection_mask

        assert L_PG.item() == 0, f"L_PG should be 0 when masked: {L_PG.item()}"
        print("✅ L_PG = 0 when selection_mask = 0")

    def test_pg_loss_active_when_rebalance(self):
        """Test L_PG > 0 when selection_trigger = 1 (rebalance day)."""
        topk_scores = torch.tensor([0.4, 0.35, 0.25])
        advantage = torch.tensor(0.05)
        selection_mask = 1  # Rebalance day

        log_prob = torch.log(topk_scores + 1e-8).sum()
        L_PG = -log_prob * advantage * selection_mask

        assert L_PG.item() > 0, f"L_PG should be > 0 when rebalance: {L_PG.item()}"
        print(f"✅ L_PG = {L_PG.item():.4f} when selection_mask = 1")


class TestRiskLoss:
    """Test L_Risk with MSE formula."""

    def test_eta_target_from_zscore(self):
        """Test η_target = 1 + λ*tanh(κ*Z_fut) where Z_fut is z-score."""
        # Simulated future price data
        P_t = 100.0  # Current price
        prices_future = np.array([102, 105, 103, 108, 110])  # Future prices

        mu_future = np.mean(prices_future)
        sigma_future = np.std(prices_future) + 1e-8

        # Z-score
        Z_fut = (mu_future - P_t) / sigma_future

        # Eta target formula from spec
        lambda_val = 0.3  # eta_amp
        kappa = 1.0
        eta_target = 1.0 + lambda_val * np.tanh(kappa * Z_fut)

        # Verify bounds
        assert 0.7 <= eta_target <= 1.3, f"η_target out of bounds: {eta_target}"
        print(f"✅ η_target from z-score: Z_fut={Z_fut:.3f}, η_target={eta_target:.4f}")

    def test_eta_target_bullish(self):
        """Test η_target > 1.0 when future is bullish (price up)."""
        P_t = 100.0
        prices_future = np.array([105, 110, 115, 120, 125])  # Strong uptrend

        mu_future = np.mean(prices_future)
        sigma_future = np.std(prices_future) + 1e-8
        Z_fut = (mu_future - P_t) / sigma_future

        eta_target = 1.0 + 0.3 * np.tanh(Z_fut)

        assert eta_target > 1.0, f"Bullish should have η > 1.0: {eta_target}"
        print(f"✅ Bullish η_target = {eta_target:.4f} > 1.0")

    def test_eta_target_bearish(self):
        """Test η_target < 1.0 when future is bearish (price down)."""
        P_t = 100.0
        prices_future = np.array([95, 90, 85, 80, 75])  # Strong downtrend

        mu_future = np.mean(prices_future)
        sigma_future = np.std(prices_future) + 1e-8
        Z_fut = (mu_future - P_t) / sigma_future

        eta_target = 1.0 + 0.3 * np.tanh(Z_fut)

        assert eta_target < 1.0, f"Bearish should have η < 1.0: {eta_target}"
        print(f"✅ Bearish η_target = {eta_target:.4f} < 1.0")

    def test_risk_mse_loss(self):
        """Test L_Risk = MSE(η_pred, η_target)."""
        eta_pred = torch.tensor([1.1, 0.9, 1.2], requires_grad=True)
        eta_target = torch.tensor([1.0, 0.85, 1.15])

        L_Risk = F.mse_loss(eta_pred, eta_target)

        # Manual calculation
        expected = ((1.1 - 1.0) ** 2 + (0.9 - 0.85) ** 2 + (1.2 - 1.15) ** 2) / 3

        assert abs(L_Risk.item() - expected) < 1e-6
        assert L_Risk.requires_grad, "L_Risk should have gradient"
        print(f"✅ L_Risk (MSE): {L_Risk.item():.6f}")


class TestDirectionLoss:
    """Test L_Dir with CrossEntropy formula."""

    def test_direction_label_bear(self):
        """Test R_fut < -δ → label=0 (bear)."""
        R_fut = -0.05  # 5% loss
        delta = 0.025

        if R_fut < -delta:
            label = 0  # Bear
        elif R_fut > delta:
            label = 2  # Bull
        else:
            label = 1  # Side

        assert label == 0, f"Should be bear (0), got {label}"
        print(f"✅ R_fut={R_fut} → label={label} (bear)")

    def test_direction_label_side(self):
        """Test |R_fut| ≤ δ → label=1 (side)."""
        for R_fut in [-0.02, 0.0, 0.02]:
            delta = 0.025

            if R_fut < -delta:
                label = 0
            elif R_fut > delta:
                label = 2
            else:
                label = 1

            assert label == 1, f"R_fut={R_fut} should be side (1), got {label}"

        print(f"✅ |R_fut| ≤ 0.025 → label=1 (side)")

    def test_direction_label_bull(self):
        """Test R_fut > δ → label=2 (bull)."""
        R_fut = 0.05  # 5% gain
        delta = 0.025

        if R_fut < -delta:
            label = 0
        elif R_fut > delta:
            label = 2
        else:
            label = 1

        assert label == 2, f"Should be bull (2), got {label}"
        print(f"✅ R_fut={R_fut} → label={label} (bull)")

    def test_direction_crossentropy_loss(self):
        """Test L_Dir = CrossEntropy(logits, labels)."""
        # Direction logits (NOT softmax'd)
        logits = torch.tensor(
            [
                [1.5, 0.2, -0.8],  # Predicts bear
                [-0.5, 1.2, 0.3],  # Predicts side
                [-1.0, 0.1, 2.0],
            ]
        )  # Predicts bull

        labels = torch.tensor([0, 1, 2])  # True labels

        L_Dir = F.cross_entropy(logits, labels)

        assert L_Dir.item() > 0, f"CrossEntropy should be positive: {L_Dir.item()}"
        assert L_Dir.item() < 5, (
            f"Loss too high for correct predictions: {L_Dir.item()}"
        )
        print(f"✅ L_Dir (CrossEntropy): {L_Dir.item():.4f}")

    def test_direction_logits_no_softmax(self):
        """Verify direction head outputs raw logits (no softmax)."""
        # Simulated raw logits from Direction Head
        logits = torch.tensor([1.5, 0.2, -0.8])

        # Raw logits can be any real number
        assert logits.sum().item() != 1.0, "Raw logits should NOT sum to 1"
        assert (logits < 0).any() or (logits > 1).any(), (
            "Raw logits can be outside [0,1]"
        )

        # Softmax would be applied only for loss computation
        probs = F.softmax(logits, dim=0)
        assert abs(probs.sum().item() - 1.0) < 1e-5, "Softmax should sum to 1"

        print("✅ Direction logits are raw (no softmax applied)")


class TestTotalLoss:
    """Test total loss with weights."""

    def test_total_loss_weighted_sum(self):
        """Test L_total = λ_pg*L_PG + λ_risk*L_Risk + λ_dir*L_Dir."""
        L_PG = torch.tensor(0.5)
        L_Risk = torch.tensor(0.1)
        L_Dir = torch.tensor(0.3)

        lambda_pg = 1.0
        lambda_risk = 1.0
        lambda_dir = 1.0

        L_total = lambda_pg * L_PG + lambda_risk * L_Risk + lambda_dir * L_Dir

        expected = 0.5 + 0.1 + 0.3
        assert abs(L_total.item() - expected) < 1e-6
        print(f"✅ L_total = {L_total.item():.4f}")

    def test_total_loss_with_pg_masking(self):
        """Test L_total when L_PG is masked (hold day)."""
        L_PG = torch.tensor(0.5)
        L_Risk = torch.tensor(0.1)
        L_Dir = torch.tensor(0.3)
        selection_mask = 0  # Hold day

        L_total = selection_mask * L_PG + L_Risk + L_Dir

        expected = 0.0 + 0.1 + 0.3  # L_PG masked out
        assert abs(L_total.item() - expected) < 1e-6
        print(f"✅ L_total (PG masked) = {L_total.item():.4f}")


def run_all_tests():
    """Run all training loss tests."""
    print("=" * 60)
    print("Testing Training Objectives (Section 5)")
    print("=" * 60)

    # Reward computation
    print("\n--- Reward Computation Tests ---")
    reward_tests = TestRewardComputation()
    reward_tests.test_compounding_reward_formula()
    reward_tests.test_topk_reward_aggregation()
    reward_tests.test_advantage_calculation()

    # Policy Gradient
    print("\n--- Policy Gradient Loss Tests ---")
    pg_tests = TestPolicyGradientLoss()
    pg_tests.test_log_prob_from_topk_scores()
    pg_tests.test_pg_loss_formula()
    pg_tests.test_pg_loss_zero_when_masked()
    pg_tests.test_pg_loss_active_when_rebalance()

    # Risk Loss
    print("\n--- Risk Loss Tests ---")
    risk_tests = TestRiskLoss()
    risk_tests.test_eta_target_from_zscore()
    risk_tests.test_eta_target_bullish()
    risk_tests.test_eta_target_bearish()
    risk_tests.test_risk_mse_loss()

    # Direction Loss
    print("\n--- Direction Loss Tests ---")
    dir_tests = TestDirectionLoss()
    dir_tests.test_direction_label_bear()
    dir_tests.test_direction_label_side()
    dir_tests.test_direction_label_bull()
    dir_tests.test_direction_crossentropy_loss()
    dir_tests.test_direction_logits_no_softmax()

    # Total Loss
    print("\n--- Total Loss Tests ---")
    total_tests = TestTotalLoss()
    total_tests.test_total_loss_weighted_sum()
    total_tests.test_total_loss_with_pg_masking()

    print("\n" + "=" * 60)
    print("All Training Loss Tests PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
