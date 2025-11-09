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
    CSAModule, TAModule, STFusionModule, SignalGenerator, MAFIAModel,
    PositionalEncoding, EventDetector
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
        self.mafia_learning_rate = 1e-4
        self.mafia_weight_decay = 0.001
        self.topK = 10


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
    
    O_i = st_fusion(O_CSA, O_TA)
    assert O_i.shape == (batch_size, N, 1), \
        f"Expected shape ({batch_size}, {N}, 1), got {O_i.shape}"
    print(f"✓ ST-Fusion output shape: {O_i.shape}")
    
    print("✓ ST-Fusion Module: PASSED\n")


def test_signal_generator():
    """Test Signal Generator."""
    print("Testing Signal Generator...")
    config = MockConfig()
    device = th.device('cpu')
    
    signal_gen = SignalGenerator(config).to(device)
    batch_size, N, T_w, D = 1, 10, 30, config.mafia_D
    
    # Create outputs from 4 agents (1 Tech + 3 DC)
    agent_outputs = [th.randn(batch_size, N, 1).to(device) for _ in range(4)]
    agent_ta_outputs = [th.randn(batch_size, T_w, D).to(device) for _ in range(4)]
    
    market_vector, boundary_risk = signal_gen(agent_outputs, agent_ta_outputs)
    
    assert market_vector.shape == (batch_size, N), \
        f"Expected market_vector shape ({batch_size}, {N}), got {market_vector.shape}"
    assert boundary_risk.shape == (batch_size,), \
        f"Expected boundary_risk shape ({batch_size},), got {boundary_risk.shape}"
    assert (boundary_risk > 0).all(), "boundary_risk should be positive (Softplus ensures this)"
    
    print(f"✓ market_vector shape: {market_vector.shape}")
    print(f"✓ boundary_risk shape: {boundary_risk.shape}")
    print(f"✓ boundary_risk range: [{boundary_risk.min():.4f}, {boundary_risk.max():.4f}]")
    print("✓ Signal Generator: PASSED\n")


def test_mafia_model():
    """Test complete MAFIA Model."""
    print("Testing MAFIA Model...")
    config = MockConfig()
    device = th.device('cpu')
    
    # Create MAFIA model
    mafia_model = MAFIAModel(config, action_dim=config.topK).to(device)
    mafia_model.eval()
    
    # Create dummy OCHLV data: (batch=1, N=10, M=5, T_w=30)
    batch_size, N, M, T_w = 1, 10, 5, 30
    ochlv_data = th.randn(batch_size, N, M, T_w).to(device) * 50 + 100  # Realistic prices
    
    with th.no_grad():
        market_vector, boundary_risk = mafia_model(ochlv_data)
    
    assert market_vector.shape == (batch_size, N), \
        f"Expected market_vector shape ({batch_size}, {N}), got {market_vector.shape}"
    assert boundary_risk.shape == (batch_size,), \
        f"Expected boundary_risk shape ({batch_size},), got {boundary_risk.shape}"
    assert (boundary_risk > 0).all(), "boundary_risk should be positive"
    
    print(f"✓ market_vector shape: {market_vector.shape}")
    print(f"✓ boundary_risk shape: {boundary_risk.shape}")
    print(f"✓ market_vector range: [{market_vector.min():.4f}, {market_vector.max():.4f}]")
    print(f"✓ boundary_risk range: [{boundary_risk.min():.4f}, {boundary_risk.max():.4f}]")
    print("✓ MAFIA Model: PASSED\n")


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

