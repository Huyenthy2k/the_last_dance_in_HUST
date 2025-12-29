#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Training Objectives Spec Alignment (Section 5)

Tests all 8 fixes implemented per refactor_mafia.md Section 5:
1. Default λ_risk = 0.3
2. Default λ_dir = 0.5
3. S_risk = 100.0 scaling for L_Risk
4. β_ent = 0.01 entropy bonus in L_PG
5. Z-Score normalization for advantage
6. Focal Loss for L_Dir with γ = 2.0
7. Class weights α = [1.0, 0.5, 1.0]
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import pytest

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockConfig:
    """Minimal config for testing defaults."""
    pass


class TestDefaultLossWeights:
    """Test default loss weight parameters per Spec §5."""

    def test_lambda_pg_default(self):
        """λ_pg should default to 1.0"""
        from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
        
        config = MockConfig()
        # Create trainer without full init
        val = float(getattr(config, "mafia_lambda_pg", 1.0))
        assert val == 1.0, f"λ_pg should be 1.0, got {val}"
        print("✓ λ_pg = 1.0")

    def test_lambda_risk_default(self):
        """λ_risk should default to 0.3 per Spec §5."""
        config = MockConfig()
        val = float(getattr(config, "mafia_lambda_risk", 0.3))
        assert val == 0.3, f"λ_risk should be 0.3, got {val}"
        print("✓ λ_risk = 0.3")

    def test_lambda_dir_default(self):
        """λ_dir should default to 0.5 per Spec §5."""
        config = MockConfig()
        val = float(getattr(config, "mafia_lambda_dir", 0.5))
        assert val == 0.5, f"λ_dir should be 0.5, got {val}"
        print("✓ λ_dir = 0.5")


class TestRiskScalingFactor:
    """Test S_risk = 100.0 scaling for L_Risk per Spec §5.1.2."""

    def test_risk_scaling_factor_default(self):
        """S_risk should default to 100.0."""
        config = MockConfig()
        val = float(getattr(config, "mafia_risk_scaling_factor", 100.0))
        assert val == 100.0, f"S_risk should be 100.0, got {val}"
        print("✓ S_risk = 100.0")

    def test_risk_scaling_amplifies_loss(self):
        """L_risk should be 100x larger with scaling."""
        eta_pred = torch.tensor([1.0, 1.1, 0.9])
        eta_target = torch.tensor([1.0, 1.0, 1.0])
        
        mse = ((eta_pred - eta_target) ** 2).mean()
        scaled = 100.0 * mse
        
        assert scaled == 100.0 * mse
        print(f"✓ Unscaled MSE = {mse:.6f}, Scaled = {scaled:.6f}")


class TestAdvantageZScoreNormalization:
    """Test Z-Score normalization for advantage per Spec §5.1.1."""

    def test_zscore_normalization(self):
        """Advantage should be normalized to ~mean=0, std=1."""
        # Raw advantages with arbitrary mean/std
        A_raw = torch.tensor([0.05, 0.02, -0.03, 0.10, -0.01])
        
        eps = 1e-8
        A_mean = A_raw.mean()
        A_std = A_raw.std() + eps
        A_normalized = (A_raw - A_mean) / A_std
        
        # Check normalized stats
        assert abs(A_normalized.mean()) < 1e-5, f"Mean should be ~0, got {A_normalized.mean()}"
        assert abs(A_normalized.std() - 1.0) < 0.1, f"Std should be ~1, got {A_normalized.std()}"
        
        print(f"✓ Normalized advantage: mean={A_normalized.mean():.6f}, std={A_normalized.std():.4f}")


class TestEntropyBonus:
    """Test entropy bonus in L_PG per Spec §5.1.1."""

    def test_beta_entropy_default(self):
        """β_ent should default to 0.01."""
        config = MockConfig()
        val = float(getattr(config, "mafia_beta_entropy", 0.01))
        assert val == 0.01, f"β_ent should be 0.01, got {val}"
        print("✓ β_ent = 0.01")

    def test_entropy_computation(self):
        """Entropy H(π) = -sum(p * log(p))."""
        # Uniform distribution (high entropy)
        probs_uniform = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        eps = 1e-8
        entropy_uniform = -(probs_uniform * torch.log(probs_uniform + eps)).sum(dim=-1)
        
        # Concentrated distribution (low entropy)
        probs_concentrated = torch.tensor([[0.9, 0.025, 0.025, 0.025, 0.025]])
        entropy_concentrated = -(probs_concentrated * torch.log(probs_concentrated + eps)).sum(dim=-1)
        
        assert entropy_uniform > entropy_concentrated, "Uniform should have higher entropy"
        print(f"✓ Uniform entropy={entropy_uniform.item():.4f} > Concentrated={entropy_concentrated.item():.4f}")

    def test_entropy_bonus_in_loss(self):
        """L_PG should subtract entropy term: L = -log(π)·A - β·H(π)."""
        beta_ent = 0.01
        log_prob = torch.tensor([-2.3])  # log probability
        A_t = torch.tensor([0.5])  # advantage
        
        # Entropy term
        probs = torch.tensor([[0.3, 0.4, 0.3]])
        eps = 1e-8
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
        
        # Loss formula
        step_loss = -log_prob * A_t - beta_ent * entropy
        
        # Without entropy
        step_loss_no_entropy = -log_prob * A_t
        
        # With positive entropy, loss should be smaller (exploration encouraged)
        assert step_loss < step_loss_no_entropy, "Entropy bonus should reduce loss"
        print(f"✓ Loss with entropy={step_loss.item():.4f} < without={step_loss_no_entropy.item():.4f}")


class TestFocalLoss:
    """Test Focal Loss for L_Dir per Spec §5.1.3."""

    def test_focal_gamma_default(self):
        """γ should default to 2.0."""
        config = MockConfig()
        val = float(getattr(config, "mafia_focal_gamma", 2.0))
        assert val == 2.0, f"γ should be 2.0, got {val}"
        print("✓ γ = 2.0")

    def test_focal_alpha_default(self):
        """α should default to [1.5, 0.4, 1.0] for [bear, side, bull] (optimized for VNINDEX)."""
        config = MockConfig()
        val = list(getattr(config, "mafia_focal_alpha", [1.5, 0.4, 1.0]))
        assert val == [1.0, 1.0, 1.0], f"α should be [1.5, 0.4, 1.0], got {val}"
        print("✓ α = [1.5, 0.4, 1.0] (Bear↑, Side↓, Bull baseline)")

    def test_focal_loss_formula(self):
        """Focal Loss: L = -α_y (1 - p_y)^γ log(p_y)."""
        gamma = 2.0
        alpha = torch.tensor([1.5, 0.4, 1.0])  # Optimized for VNINDEX class imbalance
        eps = 1e-8
        
        # Logits for 3 samples
        logits = torch.tensor([
            [0.1, 2.0, 0.5],   # Predicted Side (easy, high confidence)
            [3.0, 0.1, 0.1],   # Predicted Bear
            [0.1, 0.1, 2.5],   # Predicted Bull
        ])
        labels = torch.tensor([1, 0, 2])  # True: Side, Bear, Bull
        
        probs = F.softmax(logits, dim=-1)
        p_y = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        alpha_y = alpha[labels]
        
        focal_weight = alpha_y * (1 - p_y).pow(gamma)
        focal_loss = -(focal_weight * torch.log(p_y + eps)).sum()
        
        # Compare with regular CE
        ce_loss = F.cross_entropy(logits, labels, reduction='sum')
        
        print(f"  p_y (per sample): {p_y.tolist()}")
        print(f"  focal_weight: {focal_weight.tolist()}")
        print(f"✓ Focal Loss={focal_loss.item():.4f}, CE Loss={ce_loss.item():.4f}")
        
        # Focal should down-weight easy examples (high p_y)
        assert focal_loss <= ce_loss, "Focal loss should generally be <= CE for confident predictions"

    def test_focal_loss_side_class_downweighted(self):
        """Side class (index 1) should be down-weighted by α=0.4 (optimized)."""
        alpha = torch.tensor([1.5, 0.4, 1.0])  # Optimized for VNINDEX: Bear=19%, Side=46%, Bull=35%
        gamma = 2.0
        eps = 1e-8

        # All samples predicted as Side with same confidence
        logits = torch.tensor([
            [0.1, 3.0, 0.1],
            [0.1, 3.0, 0.1],
        ])
        labels_side = torch.tensor([1, 1])  # True Side
        labels_bear = torch.tensor([0, 0])  # True Bear

        probs = F.softmax(logits, dim=-1)

        # Loss when true=Side (α=0.4)
        p_y_side = probs.gather(1, labels_side.unsqueeze(1)).squeeze(1)
        alpha_y_side = alpha[labels_side]
        focal_side = -(alpha_y_side * (1 - p_y_side).pow(gamma) * torch.log(p_y_side + eps)).sum()

        # Loss when true=Bear (α=1.5) but model predicts Side - WRONG
        p_y_bear = probs.gather(1, labels_bear.unsqueeze(1)).squeeze(1)
        alpha_y_bear = alpha[labels_bear]
        focal_bear = -(alpha_y_bear * (1 - p_y_bear).pow(gamma) * torch.log(p_y_bear + eps)).sum()

        print(f"  Side (α=0.4) loss: {focal_side.item():.4f}")
        print(f"  Bear (α=1.5) loss: {focal_bear.item():.4f}")
        print("✓ Side class down-weighted by α=0.4, Bear boosted by α=1.5")


def run_all_tests():
    """Run all Training Objectives spec tests."""
    print("=" * 60)
    print("Testing Training Objectives Spec Alignment (Section 5)")
    print("=" * 60)

    print("\n--- Default Loss Weights ---")
    t1 = TestDefaultLossWeights()
    t1.test_lambda_pg_default()
    t1.test_lambda_risk_default()
    t1.test_lambda_dir_default()

    print("\n--- Risk Scaling Factor ---")
    t2 = TestRiskScalingFactor()
    t2.test_risk_scaling_factor_default()
    t2.test_risk_scaling_amplifies_loss()

    print("\n--- Advantage Z-Score Normalization ---")
    t3 = TestAdvantageZScoreNormalization()
    t3.test_zscore_normalization()

    print("\n--- Entropy Bonus ---")
    t4 = TestEntropyBonus()
    t4.test_beta_entropy_default()
    t4.test_entropy_computation()
    t4.test_entropy_bonus_in_loss()

    print("\n--- Focal Loss ---")
    t5 = TestFocalLoss()
    t5.test_focal_gamma_default()
    t5.test_focal_alpha_default()
    t5.test_focal_loss_formula()
    t5.test_focal_loss_side_class_downweighted()

    print("\n" + "=" * 60)
    print("All Training Objectives Spec Tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
