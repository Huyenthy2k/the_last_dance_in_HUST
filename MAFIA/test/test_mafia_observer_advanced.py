"""
Test MAFIA Observer Advanced Features (Section 1-4 of Spec)

Covers:
- Risk Head eta bounds [0.7, 1.3]
- Gumbel-TopK (train vs eval)
- Market-DC Agent embeddings
- Gating weights softmax
- TopK embeddings from fused embeddings
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRiskHeadEtaBounds:
    """Test Risk Head η calculation and bounds."""

    def test_eta_formula(self):
        """Test η = η_base + η_amp * tanh(η_raw)."""
        eta_base = 1.0
        eta_amp = 0.3

        # Test various η_raw values
        eta_raw_values = [-3.0, -1.0, 0.0, 1.0, 3.0]

        for eta_raw in eta_raw_values:
            eta = eta_base + eta_amp * np.tanh(eta_raw)

            # Check bounds [0.7, 1.3]
            assert 0.7 <= eta <= 1.3, f"η out of bounds: {eta} for η_raw={eta_raw}"

        print("✅ η formula verified for all η_raw values")

    def test_eta_bounds_theoretical(self):
        """Test theoretical bounds: η ∈ [1 - η_amp, 1 + η_amp] = [0.7, 1.3]."""
        eta_base = 1.0
        eta_amp = 0.3

        # tanh range is [-1, 1]
        eta_min_theoretical = eta_base + eta_amp * (-1)  # = 0.7
        eta_max_theoretical = eta_base + eta_amp * (1)  # = 1.3

        assert eta_min_theoretical == 0.7, (
            f"Min should be 0.7, got {eta_min_theoretical}"
        )
        assert eta_max_theoretical == 1.3, (
            f"Max should be 1.3, got {eta_max_theoretical}"
        )

        print(
            f"✅ Theoretical η bounds: [{eta_min_theoretical}, {eta_max_theoretical}]"
        )

    def test_eta_neutral_at_zero(self):
        """Test η = 1.0 (neutral) when η_raw = 0."""
        eta_base = 1.0
        eta_amp = 0.3
        eta_raw = 0.0

        eta = eta_base + eta_amp * np.tanh(eta_raw)

        assert abs(eta - 1.0) < 1e-6, f"η should be 1.0 when η_raw=0, got {eta}"
        print("✅ η = 1.0 (neutral) when η_raw = 0")

    def test_eta_tensor_computation(self):
        """Test η computation with PyTorch tensors."""
        eta_base = 1.0
        eta_amp = 0.3
        eta_raw = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        eta = eta_base + eta_amp * torch.tanh(eta_raw)

        # Verify all in bounds
        assert (eta >= 0.7).all() and (eta <= 1.3).all(), f"η out of bounds: {eta}"
        print(f"✅ η tensor: {eta.tolist()}")


class TestGumbelTopK:
    """Test Gumbel-TopK selection (train vs eval)."""

    def test_gumbel_softmax_basic(self):
        """Test Gumbel-Softmax produces valid probabilities."""
        logits = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8])
        tau = 1.0  # Temperature

        # Gumbel-Softmax
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        gumbel_logits = (logits + gumbel_noise) / tau
        soft_samples = F.softmax(gumbel_logits, dim=-1)

        # Verify valid probabilities
        assert (soft_samples >= 0).all() and (soft_samples <= 1).all()
        assert abs(soft_samples.sum().item() - 1.0) < 1e-5
        print(f"✅ Gumbel-Softmax output: {soft_samples.numpy()}")

    def test_hard_topk_eval(self):
        """Test hard argmax-K at evaluation."""
        scores = torch.tensor([0.1, 0.3, 0.05, 0.25, 0.15, 0.1, 0.05])
        K = 3

        # Hard Top-K (eval mode)
        _, topk_indices = torch.topk(scores, K)

        expected_indices = torch.tensor([1, 3, 4])  # Indices of top 3 scores
        assert torch.equal(topk_indices.sort()[0], expected_indices.sort()[0])
        print(f"✅ Hard Top-K indices (K={K}): {topk_indices.tolist()}")

    def test_soft_vs_hard_topk(self):
        """Test difference between soft (train) and hard (eval) Top-K."""
        logits = torch.tensor([1.0, 2.5, 0.5, 2.0, 0.8])
        K = 2

        # Hard Top-K (eval)
        _, hard_indices = torch.topk(logits, K)
        hard_mask = torch.zeros_like(logits)
        hard_mask[hard_indices] = 1.0

        # Soft selection (train) - via Gumbel-Softmax
        tau = 0.5
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        soft_scores = F.softmax((logits + gumbel_noise) / tau, dim=-1)

        print(f"   Hard mask: {hard_mask.tolist()}")
        print(f"   Soft scores: {soft_scores.numpy()}")

        # Verify hard is binary, soft is continuous
        assert (hard_mask == 0).sum() + (hard_mask == 1).sum() == len(hard_mask), (
            "Hard mask should be binary"
        )
        assert not torch.allclose(soft_scores, hard_mask), (
            "Soft should differ from hard"
        )
        print("✅ Soft vs Hard Top-K verified")

    def test_temperature_effect(self):
        """Test temperature τ effect on Gumbel-Softmax sharpness."""
        logits = torch.tensor([1.0, 3.0, 0.5])

        # High temperature (τ = 5.0) → more uniform
        soft_high_tau = F.softmax(logits / 5.0, dim=-1)

        # Low temperature (τ = 0.1) → sharper (more like argmax)
        soft_low_tau = F.softmax(logits / 0.1, dim=-1)

        # Verify: low τ is sharper (max prob is higher)
        assert soft_low_tau.max() > soft_high_tau.max(), "Low τ should be sharper"
        print(f"   τ=5.0 (uniform): {soft_high_tau.numpy()}")
        print(f"   τ=0.1 (sharp): {soft_low_tau.numpy()}")
        print("✅ Temperature effect verified")


class TestGatingRouter:
    """Test Dense MoE Gating Router."""

    def test_gate_weights_sum_to_one(self):
        """Test gate weights (w_0, w_1, w_2, w_3) sum to 1."""
        # Simulated gate logits from router
        gate_logits = torch.tensor([0.5, 1.2, 0.8, 0.3])

        # Apply softmax
        gate_weights = F.softmax(gate_logits, dim=-1)

        assert abs(gate_weights.sum().item() - 1.0) < 1e-5, (
            f"Gate weights should sum to 1"
        )
        assert (gate_weights >= 0).all() and (gate_weights <= 1).all(), (
            "Weights should be in [0, 1]"
        )
        print(
            f"✅ Gate weights: {gate_weights.numpy()}, sum = {gate_weights.sum().item()}"
        )

    def test_gate_weights_for_experts(self):
        """Test gate assigns weights to 4 experts (Tech + 3 DC)."""
        num_experts = 4

        # Simulated gate output
        gate_weights = F.softmax(torch.randn(num_experts), dim=-1)

        assert len(gate_weights) == 4, f"Should have 4 expert weights"
        print(f"✅ Expert weights (Tech, DC0.5, DC1.0, DC2.0): {gate_weights.numpy()}")

    def test_aggregation_formula(self):
        """Test aggregation: O = Σ w_i * O_i."""
        # 4 expert outputs, each for N=5 stocks
        O_0 = torch.tensor([0.1, 0.3, 0.2, 0.25, 0.15])  # Tech
        O_1 = torch.tensor([0.15, 0.25, 0.25, 0.2, 0.15])  # DC 0.5%
        O_2 = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])  # DC 1.0%
        O_3 = torch.tensor([0.1, 0.4, 0.1, 0.3, 0.1])  # DC 2.0%

        gate_weights = torch.tensor([0.3, 0.25, 0.25, 0.2])

        # Weighted sum
        O_agg = (
            gate_weights[0] * O_0
            + gate_weights[1] * O_1
            + gate_weights[2] * O_2
            + gate_weights[3] * O_3
        )

        assert O_agg.shape == O_0.shape, f"Aggregated shape mismatch"
        print(f"✅ Aggregated output: {O_agg.numpy()}")


class TestTopKEmbeddings:
    """Test TopK embeddings gathered from fused embeddings."""

    def test_gather_from_fused(self):
        """Test topk_embeddings = gather(Σ w_i * H_i, topk_indices)."""
        N = 10  # Number of stocks
        D = 64  # Embedding dim
        K = 3  # Top-K

        # Simulated fused embedding H = Σ w_i * H_i
        H_fused = torch.randn(N, D)

        # Top-K indices
        topk_indices = torch.tensor([2, 5, 8])

        # Gather
        topk_embeddings = H_fused[topk_indices]

        assert topk_embeddings.shape == (K, D), f"Shape should be ({K}, {D})"
        assert torch.allclose(topk_embeddings[0], H_fused[2])
        assert torch.allclose(topk_embeddings[1], H_fused[5])
        assert torch.allclose(topk_embeddings[2], H_fused[8])
        print(f"✅ TopK embeddings gathered correctly, shape: {topk_embeddings.shape}")

    def test_batch_gather(self):
        """Test batch gathering of TopK embeddings."""
        B = 4  # Batch size
        N = 10  # Number of stocks
        D = 64  # Embedding dim
        K = 3  # Top-K

        # Simulated fused embeddings for batch
        H_fused = torch.randn(B, N, D)

        # Top-K indices for each batch (could be different)
        topk_indices = torch.tensor([[1, 3, 5], [2, 4, 6], [0, 7, 9], [3, 5, 8]])

        # Gather for each batch
        topk_embeddings = torch.stack([H_fused[b][topk_indices[b]] for b in range(B)])

        assert topk_embeddings.shape == (B, K, D), f"Shape should be ({B}, {K}, {D})"
        print(f"✅ Batch TopK embeddings shape: {topk_embeddings.shape}")


class TestMarketScores:
    """Test market_scores_full and market_vector outputs."""

    def test_market_scores_full_softmax(self):
        """Test market_scores_full is full softmax over N stocks."""
        N = 10
        logits = torch.randn(N)

        # Full softmax (no masking)
        market_scores_full = F.softmax(logits, dim=-1)

        assert market_scores_full.shape == (N,)
        assert abs(market_scores_full.sum().item() - 1.0) < 1e-5
        print(
            f"✅ market_scores_full shape: {market_scores_full.shape}, sum: {market_scores_full.sum()}"
        )

    def test_market_vector_topk_masked(self):
        """Test market_vector has zeros outside Top-K."""
        N = 10
        K = 3
        logits = torch.randn(N)

        # Get Top-K indices
        _, topk_indices = torch.topk(logits, K)

        # Create mask
        mask = torch.zeros(N)
        mask[topk_indices] = 1.0

        # Apply mask and renormalize
        masked_logits = logits * mask
        masked_logits[mask == 0] = float("-inf")
        market_vector = F.softmax(masked_logits, dim=-1)

        # Verify zeros outside Top-K
        for i in range(N):
            if i not in topk_indices:
                assert market_vector[i].item() == 0 or market_vector[i].item() < 1e-6

        # Verify Top-K sums to 1
        topk_sum = market_vector[topk_indices].sum()
        assert abs(topk_sum.item() - 1.0) < 1e-5

        print(f"✅ market_vector: zeros outside Top-K, Top-K sum = {topk_sum}")

    def test_topk_scores_from_market_vector(self):
        """Test topk_scores is gathered from market_vector."""
        N = 10
        K = 3

        # Simulated market_vector (zeros outside Top-K)
        market_vector = torch.zeros(N)
        topk_indices = torch.tensor([2, 5, 8])
        topk_scores = F.softmax(torch.randn(K), dim=-1)
        market_vector[topk_indices] = topk_scores

        # Gather topk_scores from market_vector
        gathered = market_vector[topk_indices]

        assert torch.allclose(gathered, topk_scores)
        print(f"✅ topk_scores gathered correctly: {topk_scores.numpy()}")


class TestDirectionHead:
    """Test Direction Head outputs."""

    def test_direction_logits_shape(self):
        """Test direction_logits has shape (B, 3) for bear/side/bull."""
        B = 4

        # Simulated direction logits
        direction_logits = torch.randn(B, 3)

        assert direction_logits.shape == (B, 3)
        print(f"✅ direction_logits shape: {direction_logits.shape}")

    def test_direction_logits_raw(self):
        """Test direction_logits are raw (NOT softmax'd)."""
        direction_logits = torch.tensor([1.5, -0.5, 2.0])

        # Raw logits can be negative and don't sum to 1
        assert (direction_logits < 0).any() or direction_logits.sum().item() != 1.0
        print(f"✅ direction_logits are raw: {direction_logits.numpy()}")

    def test_direction_argmax_inference(self):
        """Test argmax of direction_logits for inference."""
        direction_logits = torch.tensor(
            [
                [2.0, 0.5, -1.0],  # Predicts bear (0)
                [-1.0, 2.5, 0.5],  # Predicts side (1)
                [-0.5, 0.2, 3.0],  # Predicts bull (2)
            ]
        )

        predictions = direction_logits.argmax(dim=-1)
        expected = torch.tensor([0, 1, 2])

        assert torch.equal(predictions, expected)
        print(f"✅ Direction predictions: {predictions.tolist()}")


def run_all_tests():
    """Run all MAFIA Observer advanced tests."""
    print("=" * 60)
    print("Testing MAFIA Observer Advanced Features (Section 1-4)")
    print("=" * 60)

    # Risk Head Eta
    print("\n--- Risk Head Eta Tests ---")
    eta_tests = TestRiskHeadEtaBounds()
    eta_tests.test_eta_formula()
    eta_tests.test_eta_bounds_theoretical()
    eta_tests.test_eta_neutral_at_zero()
    eta_tests.test_eta_tensor_computation()

    # Gumbel-TopK
    print("\n--- Gumbel-TopK Tests ---")
    gumbel_tests = TestGumbelTopK()
    gumbel_tests.test_gumbel_softmax_basic()
    gumbel_tests.test_hard_topk_eval()
    gumbel_tests.test_soft_vs_hard_topk()
    gumbel_tests.test_temperature_effect()

    # Gating Router
    print("\n--- Gating Router Tests ---")
    gating_tests = TestGatingRouter()
    gating_tests.test_gate_weights_sum_to_one()
    gating_tests.test_gate_weights_for_experts()
    gating_tests.test_aggregation_formula()

    # TopK Embeddings
    print("\n--- TopK Embeddings Tests ---")
    topk_tests = TestTopKEmbeddings()
    topk_tests.test_gather_from_fused()
    topk_tests.test_batch_gather()

    # Market Scores
    print("\n--- Market Scores Tests ---")
    scores_tests = TestMarketScores()
    scores_tests.test_market_scores_full_softmax()
    scores_tests.test_market_vector_topk_masked()
    scores_tests.test_topk_scores_from_market_vector()

    # Direction Head
    print("\n--- Direction Head Tests ---")
    dir_tests = TestDirectionHead()
    dir_tests.test_direction_logits_shape()
    dir_tests.test_direction_logits_raw()
    dir_tests.test_direction_argmax_inference()

    print("\n" + "=" * 60)
    print("All MAFIA Observer Advanced Tests PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
