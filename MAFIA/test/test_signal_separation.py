import torch
import torch.nn as nn
import unittest
from types import SimpleNamespace
from RL_controller.mafia_modules import DenseMoESignalGenerator, DenseMoEGatingRouter

class TestSignalSeparation(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            mafia_D=16,
            mafia_T_w=5,
            mafia_top_k=2,
            mafia_M_mkt=16, # Match D to avoid extra projection
            mafia_encoder_layers=1,
            mafia_encoder_heads=2,
            mafia_D_h=32,
            mafia_DC_thresholds=[0.01, 0.02, 0.03],
            mafia_gating_dropout=0.1,
            router_use_temporal_augmentation=True,
            mafia_test_mode=True
        )
        self.model = DenseMoESignalGenerator(self.config)

    def test_adapters_exist(self):
        """Verify that adapters are correctly instantiated."""
        self.assertTrue(hasattr(self.model, 'macro_adapter'), "DenseMoESignalGenerator should have macro_adapter")
        self.assertTrue(hasattr(self.model.gating_router, 'selection_adapter'), "DenseMoEGatingRouter should have selection_adapter")

    def test_gradient_flow_separation(self):
        """Verify that gradients flow through the correct adapters for specific tasks."""
        batch_size = 4
        N = 10
        D = self.config.mafia_D
        T_w = self.config.mafia_T_w
        
        # Dummy Inputs
        # 4 Experts * (Batch, N, 1)
        expert_outputs = [torch.randn(batch_size, N, 1) for _ in range(4)]
        # 4 Experts * (Batch, T_w, D)
        expert_ta = [torch.randn(batch_size, T_w, D) for _ in range(4)]
        # Market sequence
        x_mkt = torch.randn(batch_size, T_w, D)
        # Explicit signals for Direction Head
        explicit_sig = torch.randn(batch_size, 4) # 4 dims for explicit signals

        # Enable grads
        x_mkt.requires_grad = True
        
        # --- Task 1: Macro Task (Risk) ---
        # Forward pass
        outputs = self.model(
            expert_outputs, 
            expert_ta, 
            x_mkt_seq=x_mkt, 
            explicit_signals=explicit_sig
        )
        
        eta = outputs[1] # (batch,)
        
        # Compute loss purely on Risk (eta)
        loss_risk = eta.mean()
        
        # Zero grads
        self.model.zero_grad()
        loss_risk.backward(retain_graph=True)
        
        # Check Gradients
        # Macro Adapter should have grad
        self.assertIsNotNone(self.model.macro_adapter.weight.grad, "Macro adapter should receive gradient from Risk Loss")
        self.assertNotEqual(self.model.macro_adapter.weight.grad.abs().sum(), 0, "Macro adapter grad should be non-zero")
        
        # Selection Adapter should NOT have grad (Risk doesn't affect gating decisions directly in this backward, 
        # unless risk affects topk? Risk affects eta, topk affects portfolio_context. 
        # But wait, portfolio_context is DETACHED in forward()! 
        # So Risk calculation is: eta = f(market_context_macro, portfolio_context.detach()).
        # So Risk depends on market_context_macro. 
        # market_context_macro = macro_adapter(raw).
        # raw = encoder(x_mkt).
        # Selection adapter is in Gating Router, used for gate weights.
        # Gate weights affect portfolio_context.
        # But portfolio_context is detached!
        # So Selection Adapter should NOT receive gradient from Risk Loss!
        self.assertIsNone(self.model.gating_router.selection_adapter.weight.grad, 
                          "Selection adapter should NOT receive gradient from Risk Loss due to detach()")
                          
        # --- Task 2: Selection Task (Gating) ---
        # To test selection gradient, we need a loss that depends on gate weights.
        # market_vector depends on gate_weights.
        market_vector = outputs[0]
        loss_selection = market_vector.mean()
        
        self.model.zero_grad()
        loss_selection.backward()
        
        # Check Gradients
        # Selection Adapter should have grad
        self.assertIsNotNone(self.model.gating_router.selection_adapter.weight.grad, "Selection adapter should receive gradient from Selection Loss")
        self.assertNotEqual(self.model.gating_router.selection_adapter.weight.grad.abs().sum(), 0, "Selection adapter grad should be non-zero")
        
        # Macro Adapter should NOT have grad (Market vector calculation doesn't use macro context)
        # Wait, market_vector = Softmax(Weighted Sum of Experts). Weights come from Gating Router.
        # Gating Router uses Selection Adapter.
        # Macro Adapter is used for eta and sigma_logits.
        # So Macro Adapter should be zero.
        if self.model.macro_adapter.weight.grad is not None:
             self.assertEqual(self.model.macro_adapter.weight.grad.abs().sum(), 0, "Macro adapter should NOT receive gradient from Selection Loss")

    def test_buffer_consistency(self):
        """Verify that the returned market_context is indeed the RAW context."""
        batch_size = 2
        D = self.config.mafia_D
        T_w = self.config.mafia_T_w
        x_mkt = torch.randn(batch_size, T_w, D)
        explicit_sig = torch.randn(batch_size, 4)
        
        self.model.eval() # Disable dropout for deterministic check
        with torch.no_grad():
             outputs = self.model(
                [torch.randn(batch_size, 10, 1) for _ in range(4)], 
                [torch.randn(batch_size, T_w, D) for _ in range(4)], 
                x_mkt_seq=x_mkt, 
                explicit_signals=explicit_sig
            )
        
        returned_context = outputs[4] # market_context / raw_context
        
        # Manually compute raw context from encoder
        raw_context_check, _ = self.model.gating_router.temporal_encoder(x_mkt)
        
        # They should be identical
        self.assertTrue(torch.allclose(returned_context, raw_context_check, atol=1e-5), 
                        "Returned context must be the RAW context for buffer consistency")

if __name__ == '__main__':
    unittest.main()
