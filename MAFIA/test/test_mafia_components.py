#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         test_mafia_components.py
 Description:  Unit tests for MAFIA components
 Author:       MASA
---------------------------------
'''
import numpy as np
import torch as th
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RL_controller.mafia_feature_processor import MAFIAFeatureProcessor
from RL_controller.mafia_modules import (
    CSAModule, TAModule, STFusionModule, DenseMoESignalGenerator, MAFIAModel,
    PositionalEncoding, EventDetector,
    AttentionBasedTemporalEncoder, TemporalConvolutionEncoder, UnidirectionalLSTMEncoder
)


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.mafia_T_w = 30
        self.mafia_DC_thresholds = [0.005, 0.01, 0.02]
        self.mafia_D = 64
        self.mafia_D_h = 128
        self.mafia_encoder_layers = 2
        self.mafia_encoder_heads = 4
        self.mafia_M_tech = 8
        self.mafia_M_dc = 5
        self.mafia_M_mkt = 19
        self.mafia_learning_rate = 1e-4
        self.mafia_weight_decay = 0.001
        self.topK = 10
        self.mafia_top_k = 10
        self.mafia_gumbel_temperature = 1.0
        self.mafia_hard_topk_inference = True
        # Dense MoE Gating Configuration
        self.mafia_gating_encoder_type = 'attention_based_aggregation'
        self.mafia_gating_num_heads = 4
        self.mafia_gating_dropout = 0.1
        self.mafia_gating_lstm_layers = 2
        self.mafia_gating_conv_kernels = [3, 5, 7]
        # Eta scaling defaults
        self.mafia_eta_base = 1.0
        self.mafia_eta_amplitude = 0.3
        self.mafia_eta_min = 0.7
        self.mafia_eta_max = 1.3


def test_feature_processor_technical():
    """Test Technical Agent feature processing."""
    print("Testing Technical Agent feature processing...")
    config = MockConfig()
    processor = MAFIAFeatureProcessor(config)
    
    # Create dummy OCHLV data: (N=10, M=5, T_w=30)
    N, M, T_w = 10, 5, 30
    ochlv_data = np.random.rand(N, M, T_w) * 100 + 50  # Realistic price range
    
    # Process features
    P_tech = processor.process_technical_features(ochlv_data)
    
    # Verify output shape
    assert P_tech.shape == (N, T_w, config.mafia_M_tech), \
        f"Expected shape ({N}, {T_w}, {config.mafia_M_tech}), got {P_tech.shape}"
    
    # Verify no NaN or Inf
    assert not np.isnan(P_tech).any(), "Found NaN in Technical features"
    assert not np.isinf(P_tech).any(), "Found Inf in Technical features"
    
    print(f"✓ Technical features shape: {P_tech.shape}")
    print(f"✓ Technical features range: [{P_tech.min():.2f}, {P_tech.max():.2f}]")
    print("✓ Technical Agent feature processing: PASSED\n")


def test_feature_processor_dc():
    """Test DC Agent feature processing."""
    print("Testing DC Agent feature processing...")
    config = MockConfig()
    processor = MAFIAFeatureProcessor(config)
    
    # Create dummy OCHLV data with trend
    N, M, T_w = 10, 5, 30
    ochlv_data = np.zeros((N, M, T_w))
    
    # Create upward trend
    base_price = 100.0
    for t in range(T_w):
        price = base_price * (1 + 0.01 * t)  # 1% increase per day
        ochlv_data[:, 0, t] = price * 0.99  # open
        ochlv_data[:, 1, t] = price  # close
        ochlv_data[:, 2, t] = price * 1.01  # high
        ochlv_data[:, 3, t] = price * 0.98  # low
        ochlv_data[:, 4, t] = 1000000  # volume
    
    # Process features for each DC threshold
    for threshold in config.mafia_DC_thresholds:
        P_dc = processor.process_dc_features(ochlv_data, threshold)
        
        # Verify output shape
        assert P_dc.shape == (N, T_w, config.mafia_M_dc), \
            f"Expected shape ({N}, {T_w}, {config.mafia_M_dc}), got {P_dc.shape}"
        
        # Verify State values are +1 or -1
        states = P_dc[:, :, 0]
        assert np.all(np.isin(states, [-1, 1])), \
            f"State values should be -1 or 1, got range [{states.min()}, {states.max()}]"
        
        # Verify Event_Flag values are 0.5 or 1.0
        event_flags = P_dc[:, :, 4]
        assert np.all(np.isin(event_flags, [0.5, 1.0])), \
            f"Event_Flag values should be 0.5 or 1.0, got range [{event_flags.min()}, {event_flags.max()}]"
        
        print(f"✓ DC features (threshold={threshold}) shape: {P_dc.shape}")
        print(f"✓ State range: [{states.min()}, {states.max()}]")
        print(f"✓ Event_Flag range: [{event_flags.min()}, {event_flags.max()}]")
    
    print("✓ DC Agent feature processing: PASSED\n")


def test_csa_module():
    """Test CSA Module."""
    print("Testing CSA Module...")
    config = MockConfig()
    device = th.device('cpu')
    
    # Test Technical Agent CSA
    csa_tech = CSAModule(config, agent_type='tech').to(device)
    batch_size, N, T_w, M = 1, 10, 30, 8
    P_tech = th.randn(batch_size, N, T_w, M).to(device)
    
    O_CSA = csa_tech(P_tech)
    assert O_CSA.shape == (batch_size, N, config.mafia_D), \
        f"Expected shape ({batch_size}, {N}, {config.mafia_D}), got {O_CSA.shape}"
    print(f"✓ Technical CSA output shape: {O_CSA.shape}")
    
    # Test DC Agent CSA
    csa_dc = CSAModule(config, agent_type='dc').to(device)
    batch_size, N, T_w, M = 1, 10, 30, 5
    P_dc = th.randn(batch_size, N, T_w, M).to(device)
    
    O_CSA = csa_dc(P_dc)
    assert O_CSA.shape == (batch_size, N, config.mafia_D), \
        f"Expected shape ({batch_size}, {N}, {config.mafia_D}), got {O_CSA.shape}"
    print(f"✓ DC CSA output shape: {O_CSA.shape}")
    
    print("✓ CSA Module: PASSED\n")


def test_ta_module():
    """Test TA Module."""
    print("Testing TA Module...")
    config = MockConfig()
    device = th.device('cpu')
    
    # Test Technical Agent TA
    ta_tech = TAModule(config, agent_type='tech').to(device)
    batch_size, N, T_w, M = 1, 10, 30, 8
    P_tech = th.randn(batch_size, N, T_w, M).to(device)
    
    O_TA = ta_tech(P_tech)
    assert O_TA.shape == (batch_size, T_w, config.mafia_D), \
        f"Expected shape ({batch_size}, {T_w}, {config.mafia_D}), got {O_TA.shape}"
    print(f"✓ Technical TA output shape: {O_TA.shape}")
    
    # Test DC Agent TA
    shared_mlp = th.nn.Sequential(
        th.nn.Linear(config.topK * config.mafia_M_dc, config.mafia_D_h),
        th.nn.GELU(),
        th.nn.Linear(config.mafia_D_h, config.mafia_D)
    )
    ta_dc = TAModule(config, agent_type='dc', shared_mlp=shared_mlp).to(device)
    batch_size, N, T_w, M = 1, 10, 30, 5
    P_dc = th.randn(batch_size, N, T_w, M).to(device)
    
    O_TA = ta_dc(P_dc, dc_features=P_dc)
    assert O_TA.shape == (batch_size, T_w, config.mafia_D), \
        f"Expected shape ({batch_size}, {T_w}, {config.mafia_D}), got {O_TA.shape}"
    print(f"✓ DC TA output shape: {O_TA.shape}")
    
    print("✓ TA Module: PASSED\n")


def test_st_fusion():
    """Test ST-Fusion Module."""
    print("Testing ST-Fusion Module...")
    config = MockConfig()
    device = th.device('cpu')
    
    st_fusion = STFusionModule(config.mafia_D).to(device)
    batch_size, N, T_w, D = 1, 10, 30, config.mafia_D
    
    O_CSA = th.randn(batch_size, N, D).to(device)
    O_TA = th.randn(batch_size, T_w, D).to(device)
    
    O_i, O_i_ST = st_fusion(O_CSA, O_TA)
    assert O_i.shape == (batch_size, N, 1), \
        f"Expected shape ({batch_size}, {N}, 1), got {O_i.shape}"
    assert O_i_ST.shape == (batch_size, N, config.mafia_D), \
        f"Expected fusion embedding shape ({batch_size}, {N}, {config.mafia_D}), got {O_i_ST.shape}"
    print(f"✓ ST-Fusion output shape: {O_i.shape}")
    
    print("✓ ST-Fusion Module: PASSED\n")


def test_signal_generator():
    """Test Dense MoE Signal Generator."""
    print("Testing Dense MoE Signal Generator...")
    config = MockConfig()
    device = th.device('cpu')
    
    signal_gen = DenseMoESignalGenerator(config).to(device)
    batch_size, N, T_w, D = 1, 10, 30, config.mafia_D
    
    # Create outputs from 4 stock experts (1 Tech + 3 DC)
    expert_outputs = [th.randn(batch_size, N, 1).to(device) for _ in range(4)]
    expert_ta_outputs = [th.randn(batch_size, T_w, D).to(device) for _ in range(4)]
    
    # Create market-index TA output for gating
    O_mkt_TA = th.randn(batch_size, T_w, D).to(device)
    
    # Forward pass
    market_vector, risk_eta, market_scores_full, market_context, sigma_logits, topk_indices, topk_embeddings, topk_scores = signal_gen(
        expert_outputs, expert_ta_outputs, x_mkt_seq=O_mkt_TA, O_mkt_TA=None, expert_st_embeddings=[th.randn(batch_size, N, D).to(device) for _ in range(4)]
    )
    
    # Assertions
    assert topk_indices.shape == (batch_size, config.mafia_top_k), \
        f"Expected topk_indices shape ({batch_size}, {config.mafia_top_k}), got {topk_indices.shape}"
    
    assert market_vector.shape == (batch_size, N), \
        f"Expected market_vector shape ({batch_size}, {N}), got {market_vector.shape}"
    assert risk_eta.shape == (batch_size,), \
        f"Expected risk_eta shape ({batch_size},), got {risk_eta.shape}"
    assert (risk_eta > 0).all(), "risk_eta should be positive"
    
    assert market_scores_full.shape == (batch_size, N), \
        f"Expected market_scores_full shape ({batch_size}, {N}), got {market_scores_full.shape}"
    
    # New: Test topk_scores
    assert topk_scores.shape == (batch_size, config.mafia_top_k), \
        f"Expected topk_scores shape ({batch_size}, {config.mafia_top_k}), got {topk_scores.shape}"
    
    print(f"✓ market_vector shape: {market_vector.shape}")
    print(f"✓ risk_eta shape: {risk_eta.shape}")
    print(f"✓ risk_eta range: [{risk_eta.min():.4f}, {risk_eta.max():.4f}]")
    print(f"✓ market_context shape: {market_context.shape}")
    assert sigma_logits.shape == (batch_size, 3), \
        f"Expected sigma_logits shape ({batch_size}, 3), got {sigma_logits.shape}"
    if topk_embeddings is not None:
        assert topk_embeddings.shape[1] == config.mafia_top_k
    print(f"✓ topk_scores: {topk_scores[0].tolist()}")
    print("✓ Dense MoE Signal Generator: PASSED\n")


def test_mafia_model():
    """Test complete MAFIA Model with Dense MoE."""
    print("Testing MAFIA Model with Dense MoE...")
    config = MockConfig()
    device = th.device('cpu')
    
    # Create MAFIA model
    mafia_model = MAFIAModel(config, action_dim=config.topK).to(device)
    mafia_model.eval()
    
    # Create dummy OCHLV data: (batch=1, N=10, M=5, T_w=30)
    batch_size, N, M, T_w = 1, 10, 5, 30
    ochlv_data = th.randn(batch_size, N, M, T_w).to(device) * 50 + 100  # Realistic prices
    
    # Create market-index OCHLV data: (batch=1, 1, M=5, T_w=30)
    market_index_ochlv_data = th.randn(batch_size, 1, M, T_w).to(device) * 50 + 1000  # VNINDEX prices
    
    with th.no_grad():
        market_vector, risk_eta, market_scores_full, market_context, sigma_logits, topk_indices, topk_embeddings, topk_scores = mafia_model(
            ochlv_data, market_index_ochlv_data=market_index_ochlv_data
        )
    
    # Original assertions
    assert market_vector.shape == (batch_size, N), \
        f"Expected market_vector shape ({batch_size}, {N}), got {market_vector.shape}"
    assert risk_eta.shape == (batch_size,), \
        f"Expected risk_eta shape ({batch_size},), got {risk_eta.shape}"
    assert (risk_eta > 0).all(), "risk_eta should be positive"
    assert topk_indices.shape == (batch_size, config.mafia_top_k), \
        f"Expected topk_indices shape ({batch_size}, {config.mafia_top_k}), got {topk_indices.shape}"
    
    # New assertions for Dense MoE
    assert market_scores_full.shape == (batch_size, N), \
        f"Expected market_scores_full shape ({batch_size}, {N}), got {market_scores_full.shape}"
    assert market_context.shape == (batch_size, config.mafia_D), \
        f"Expected market_context shape ({batch_size}, {config.mafia_D}), got {market_context.shape}"
    assert topk_embeddings is None or topk_embeddings.shape[1] == config.mafia_top_k
    assert topk_scores is None or topk_scores.shape[1] == config.mafia_top_k
    assert sigma_logits.shape == (batch_size, 3), \
        f"Expected sigma_logits shape ({batch_size}, 3), got {sigma_logits.shape}"
    if topk_embeddings is not None:
        assert topk_embeddings.shape[1] == config.mafia_top_k
    
    print(f"✓ market_vector shape: {market_vector.shape}")
    print(f"✓ risk_eta shape: {risk_eta.shape}")
    print(f"✓ market_vector range: [{market_vector.min():.4f}, {market_vector.max():.4f}]")
    print(f"✓ risk_eta range: [{risk_eta.min():.4f}, {risk_eta.max():.4f}]")
    print(f"✓ topk_scores: {topk_scores[0].tolist()}")
    print(f"✓ market_context shape: {market_context.shape}")
    if topk_embeddings is not None:
        print(f"✓ topk_embeddings shape: {topk_embeddings.shape}")
    print("✓ MAFIA Model with Dense MoE: PASSED\n")


def test_temporal_encoders():
    """Test all 3 temporal encoders for Dense MoE gating."""
    print("Testing Temporal Encoders...")
    config = MockConfig()
    device = th.device('cpu')
    
    batch_size, T_w, D = 2, 30, config.mafia_D
    O_mkt_TA = th.randn(batch_size, T_w, D).to(device)
    
    encoder_types = [
        ('attention_based_aggregation', AttentionBasedTemporalEncoder),
        ('temporal_convolution', TemporalConvolutionEncoder),
        ('bidirectional_lstm', UnidirectionalLSTMEncoder)
    ]
    
    for encoder_name, EncoderClass in encoder_types:
        print(f"  Testing {encoder_name}...")
        encoder = EncoderClass(config).to(device)
        encoder.eval()
        
        with th.no_grad():
            market_context, aux_output = encoder(O_mkt_TA)
        
        # Assertions
        assert market_context.shape == (batch_size, D), \
            f"Expected market_context shape ({batch_size}, {D}), got {market_context.shape}"
        assert not th.isnan(market_context).any(), f"{encoder_name}: market_context contains NaN"
        assert not th.isinf(market_context).any(), f"{encoder_name}: market_context contains Inf"
        
        print(f"    ✓ {encoder_name}: market_context shape {market_context.shape}")
        print(f"    ✓ {encoder_name}: No NaN or Inf values")
    
    print("✓ All Temporal Encoders: PASSED\n")


def test_stateful_bilstm_encoder_continuity():
    """Spec 3.6.2: BiLSTM encoder should carry and reset hidden state cleanly."""
    config = MockConfig()
    encoder = UnidirectionalLSTMEncoder(config)

    batch_size, T_w, D = 1, 8, config.mafia_D
    # First pass seeds cached state (detach from graph)
    O_mkt_TA = th.randn(batch_size, T_w, D, requires_grad=True)
    assert encoder._cached_state is None
    _ctx1, _ = encoder(O_mkt_TA)
    assert encoder._cached_state is not None
    h1, c1 = encoder._cached_state
    assert h1.requires_grad is False and c1.requires_grad is False

    # Detach API should keep state but truncate graph
    encoder.detach_state()
    h1_det, c1_det = encoder._cached_state
    assert h1_det.requires_grad is False and c1_det.requires_grad is False
    assert th.allclose(h1, h1_det) and th.allclose(c1, c1_det)

    # Cached state should be reusable for the next timestep (continuous propagation)
    prepared_state = encoder._prepare_state(batch_size, O_mkt_TA.device)
    assert prepared_state is not None
    h1_clone = h1.clone()
    O_mkt_TA_2 = th.randn(batch_size, T_w, D, requires_grad=True)
    _ctx2, _ = encoder(O_mkt_TA_2)
    h2, c2 = encoder._cached_state
    assert encoder._cached_state is not None
    # New state should reflect latest input (not identical to previous cached state)
    assert not th.equal(h1_clone, h2) or not th.equal(c1, c2)

    # Reset should drop cached state (episode boundary)
    encoder.reset_state(batch_size=batch_size, device=O_mkt_TA.device)
    assert encoder._cached_state is not None  # materialized zero state
    h0, c0 = encoder._cached_state
    assert th.count_nonzero(h0).item() == 0 and th.count_nonzero(c0).item() == 0
    encoder.reset_state()  # defer zero-init to next forward
    assert encoder._cached_state is None
    print("✓ Stateful BiLSTM encoder propagates/clears state per Spec 3.6.2\n")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("MAFIA Component Unit Tests")
    print("=" * 60 + "\n")
    
    try:
        test_feature_processor_technical()
        test_feature_processor_dc()
        test_csa_module()
        test_ta_module()
        test_st_fusion()
        test_temporal_encoders()
        test_stateful_bilstm_encoder_continuity()
        test_signal_generator()
        test_mafia_model()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
