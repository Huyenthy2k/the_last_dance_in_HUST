
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock, patch

# Adjust path to import agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL_controller.mafia_modules import DenseMoEGatingRouter, UnidirectionalLSTMEncoder
from RL_controller.mafia_observer import MAFIAObserver

class MockConfig:
    def __init__(self):
        self.mafia_D = 10
        self.mafia_D_h = 20
        self.mafia_T_w = 30
        self.mafia_encoder_layers = 1
        self.mafia_encoder_heads = 2
        self.mafia_M_tech = 8
        self.mafia_M_dc = 5
        self.mafia_M_mkt = 10
        self.router_context_window = 5
        self.router_use_temporal_augmentation = True
        self.mafia_gumbel_temperature = 1.0
        self.mafia_top_k = 2
        self.mafia_DC_thresholds = [0.005, 0.01, 0.02]
        self.mafia_learning_rate = 0.001
        self.mafia_weight_decay = 0.0
        self.num_epochs = 10
        self.risk_market = 0.0
        self.mafia_gating_encoder_type = "bidirectional_lstm"
        self.mafia_gating_lstm_layers = 1

class MockEncoder(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D = D
        self.reset_called = False
        self.detach_called = False
        # Add parameter so next(parameters()) works
        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def forward(self, x):
        # Return constant 10.0 for context
        return torch.ones(x.size(0), self.D) * 10.0, None

    def reset_state(self):
        self.reset_called = True
        
    def detach_state(self):
        self.detach_called = True

class MockGateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None
        # Add parameter so next(parameters()) works
        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def forward(self, x):
        self.last_input = x
        return torch.zeros(x.size(0), 4)

class TestContextBuffer(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.action_dim = 10
        # Mock MAFIAModel to avoid complex init
        with patch('RL_controller.mafia_modules.MAFIAModel') as MockModel:
            # Create a simple module that has parameters so optimizer doesn't complain
            dummy_model = nn.Linear(1, 1).to(torch.device('cpu'))
            MockModel.return_value = dummy_model
            
            self.observer = MAFIAObserver(self.config, self.action_dim)
            # Ensure we can access observer methods


    def test_buffer_init(self):
        """Test lazy initialization of buffer."""
        self.assertIsNone(self.observer.context_buffer)
        self.assertFalse(self.observer._context_buffer_initialized)
        
        # Trigger init via update
        mc = torch.randn(1, self.config.mafia_D)
        self.observer._update_context_buffer(mc)
        
        self.assertTrue(self.observer._context_buffer_initialized)
        self.assertIsNotNone(self.observer.context_buffer)
        self.assertEqual(self.observer.context_buffer.shape, (self.config.router_context_window, self.config.mafia_D))
        
        # Check last element
        self.assertTrue(torch.allclose(self.observer.context_buffer[-1].cpu(), mc.cpu().squeeze(0)))

    def test_buffer_fifo(self):
        """Test FIFO logic."""
        W = self.config.router_context_window
        D = self.config.mafia_D
        device = torch.device('cpu')
        
        # Manually init buffer
        self.observer._init_context_buffer(D, device)
        
        # Push W items
        items = []
        for i in range(W):
            val = torch.full((1, D), float(i+1), device=device)
            items.append(val)
            self.observer._update_context_buffer(val)
            
        # Verify content: should be [1, 2, 3, 4, 5]
        expected = torch.cat(items, dim=0)
        self.assertTrue(torch.allclose(self.observer.context_buffer, expected))
        
        # Push one more (6)
        val6 = torch.full((1, D), 6.0, device=device)
        self.observer._update_context_buffer(val6)
        
        # Should be [2, 3, 4, 5, 6]
        expected_shifted = torch.cat([items[1], items[2], items[3], items[4], val6], dim=0)
        self.assertTrue(torch.allclose(self.observer.context_buffer, expected_shifted))

class TestAugmentedInput(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.router = DenseMoEGatingRouter(self.config)
        self.D = self.config.mafia_D
        
    def test_augmented_input_logic(self):
        """Test C_aug construction with inclusive rolling window."""
        batch_size = 2
        W = 5
        
        # Replace temporal_encoder with MockEncoder
        self.router.temporal_encoder = MockEncoder(self.D)
        
        # Replace gate_network with MockGateNetwork to inspect input
        self.router.gate_network = MockGateNetwork()
        
        # Mock buffer with sequential values [1, 2, 3, 4, 5]
        # Shape (W, D) expanded to (1, W, D) internally or (batch, W, D)
        buffer = torch.zeros(W, self.D)
        for i in range(W):
            buffer[i] = float(i + 1)
        
        # Run forward
        # x_mkt will produce market_context = 11.0 (via modified MockEncoder below)
        x_mkt = torch.randn(batch_size, 30, 10)
        
        # Override MockEncoder to return 11.0
        def forward_11(x):
            return torch.ones(x.size(0), self.D) * 11.0, None
        self.router.temporal_encoder.forward = forward_11
        
        self.router(x_mkt_seq=x_mkt, context_buffer=buffer)
        
        # Inspect input
        gate_input = self.router.gate_network.last_input
        self.assertEqual(gate_input.shape, (batch_size, 3 * self.D))
        
        # Verify values
        # C_mkt^(t) = 11.0 (from updated MockEncoder)
        # Buffer was [1, 2, 3, 4, 5]
        # Spec 3.6: "C_bar is aggregation of context window W ending at t"
        # Rolling Window = [buffer[1], buffer[2], buffer[3], buffer[4], current]
        #                = [2.0, 3.0, 4.0, 5.0, 11.0]
        # C_bar = mean([2, 3, 4, 5, 11]) = 25 / 5 = 5.0
        # Delta_C = C_mkt - C_mkt^(t-W+1) = 11.0 - 2.0 = 9.0
        
        c_mkt_part = gate_input[:, :self.D]
        c_bar_part = gate_input[:, self.D:2*self.D]
        delta_c_part = gate_input[:, 2*self.D:]
        
        self.assertTrue(torch.allclose(c_mkt_part, torch.ones_like(c_mkt_part) * 11.0), "C_mkt values incorrect")
        self.assertTrue(torch.allclose(c_bar_part, torch.ones_like(c_bar_part) * 5.0), 
                        f"C_bar values incorrect. Expected 5.0, got {c_bar_part[0,0].item()}")
        self.assertTrue(torch.allclose(delta_c_part, torch.ones_like(delta_c_part) * 9.0),
                        f"Delta_C values incorrect. Expected 9.0, got {delta_c_part[0,0].item()}")

class TestStatePropagation(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.router = DenseMoEGatingRouter(self.config)
        
    def test_reset_and_detach(self):
        """Test reset and detach delegation."""
        # Replace with MockEncoder
        self.router.temporal_encoder = MockEncoder(self.config.mafia_D)
        self.router._stateful_encoder = True
        
        self.router.reset_temporal_state()
        self.assertTrue(self.router.temporal_encoder.reset_called)
        
        self.router.detach_temporal_state()
        self.assertTrue(self.router.temporal_encoder.detach_called)
        
    def test_bilstm_state_cache(self):
        """Test BiLSTM actual state caching."""
        encoder = UnidirectionalLSTMEncoder(self.config)
        
        # Reset
        encoder.reset_state(batch_size=2)
        self.assertIsNotNone(encoder._cached_state)
        h, c = encoder._cached_state
        self.assertTrue(torch.allclose(h, torch.zeros_like(h)))
        self.assertEqual(h.size(1), 2) # batch size
        
        # Forward pass should update cache
        x = torch.randn(2, 5, self.config.mafia_D)
        out, _ = encoder(x)
        
        h_new, c_new = encoder._cached_state
        self.assertFalse(torch.allclose(h_new, h)) # Should have changed
        
        # Confirm it's detached (requires checking graph, hard to do directly, but assume code works if logic is correct)
        self.assertFalse(h_new.requires_grad)

if __name__ == "__main__":
    unittest.main()
