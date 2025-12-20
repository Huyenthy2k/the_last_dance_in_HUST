
import os
import sys
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), "agents", "MAFIA"))
from config import Config
from RL_controller.mafia_modules import DirectionHead

def debug_gradients():
    print("=" * 60)
    print("🕵️‍♂️  DIRECTION HEAD GRADIENT PROBE")
    print("=" * 60)
    
    # 1. Load Config & Init Module
    config = Config()
    # Ensure config matches what we saw
    config.mafia_D = 64
    config.mafia_explicit_dim = 6
    config.mafia_focal_alpha = [1.7, 0.85, 1.15]
    config.mafia_focal_gamma = 3.0
    
    dir_head = DirectionHead(config)
    dir_head.train() # Make sure dropout is active (though not used for gradient flow check per se)
    
    # 2. Create Dummy Inputs (Batch Size 64)
    B = 64
    D = config.mafia_D
    
    # Latent Context (Deep Path)
    c_mkt = th.randn(B, D, requires_grad=True)
    delta_c_mkt = th.randn(B, D, requires_grad=True)
    
    # Explicit Signals (Wide Path)
    # Simulate realistic signal ranges
    # Vol ~ 0.015, Gap ~ -5.0, Div ~ 0.0
    vol = th.normal(0.015, 0.005, size=(B, 1))
    dc_event = (th.rand(B, 1) > 0.9).float()
    gap = th.normal(-2.0, 5.0, size=(B, 1))
    div = th.normal(0.0, 0.5, size=(B, 1))
    vpi = th.normal(0.0, 1.0, size=(B, 1))
    dd60 = th.normal(-0.1, 0.05, size=(B, 1))
    
    explicit_signals = th.cat([vol, dc_event, gap, div, vpi, dd60], dim=1).detach()
    explicit_signals.requires_grad = True # Probe sensitivity
    
    # Targets (Random labels: 0, 1, 2)
    labels = th.randint(0, 3, (B,))
    
    print("\n[Input Stats]")
    print(f"  c_mkt: (mean={c_mkt.mean():.4f}, std={c_mkt.std():.4f})")
    print(f"  explicit: (mean={explicit_signals.mean():.4f}, std={explicit_signals.std():.4f})")
    
    # 3. Forward Pass
    logits = dir_head(c_mkt, delta_c_mkt, explicit_signals)
    
    # 4. Compute Loss (Standard CE for simplicity, or we can implement Focal)
    # Let's use standard CE to check pure gradient flow, 
    # but Focal is what makes it 'hard'. Let's impl simple Focal.
    
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    p_t = th.exp(-ce_loss)
    
    # Get alpha for each target
    alphas = th.tensor(config.mafia_focal_alpha)
    alpha_t = alphas[labels]
    
    loss = (alpha_t * (1 - p_t) ** config.mafia_focal_gamma * ce_loss).mean()
    
    print(f"\n[Forward]")
    print(f"  Logits: {logits[0].detach().numpy()}")
    print(f"  Prior Bias: {dir_head.classifier.bias.detach().numpy()}")
    print(f"  Loss: {loss.item():.4f}")
    
    # 5. Backward
    loss.backward()
    
    # 6. Analyze Gradients
    print("\n[Gradient Analysis]")
    
    # Deep Path Sensitivity
    grad_c = c_mkt.grad
    grad_c_norm = grad_c.norm(dim=1).mean().item()
    print(f"  🌊 Deep Path (c_mkt) Grad Norm: {grad_c_norm:.6f}")
    
    # Wide Path Sensitivity
    grad_explicit = explicit_signals.grad
    grad_explicit_avg = grad_explicit.abs().mean(dim=0)
    
    signal_names = ["Vol_Std20", "DC_Event", "Breadth", "Div_Sig", "VPI_Z", "DD60"]
    print("\n  📢 Wide Path (Explicit) Sensitivity (dLoss/dInput):")
    for i, name in enumerate(signal_names):
        print(f"     - {name:10s}: {grad_explicit_avg[i].item():.6f}")
        
    # Classifier Weights Gradient
    print("\n  ⚖️  Classifier Weight Gradients (Last Layer):")
    # shape (3, D+6)
    W_grad = dir_head.classifier.weight.grad
    
    # Deep part of W (0..D)
    W_grad_deep = W_grad[:, :D].abs().mean().item()
    
    # Wide part of W (D..D+6)
    W_grad_wide = W_grad[:, D:].abs().mean(dim=0) # per signal
    
    print(f"     - Deep Features Weights: {W_grad_deep:.6f}")
    print("     - Wide Features Weights:")
    for i, name in enumerate(signal_names):
        print(f"       - {name:10s}: {W_grad_wide[i].item():.6f}")

    print("\n[Conclusion]")
    ratio = grad_c_norm / (grad_explicit_avg.mean().item() + 1e-9)
    print(f"  Ratio (Deep sens / Wide sens): {ratio:.2f}")
    if ratio < 0.1:
        print("  ⚠️  Deep Path is receiving very weak gradients compared to Explicit Signals!")
        print("      Model is relying almost entirely on Wide Path.")
    elif ratio > 10.0:
         print("  ⚠️  Wide Path is receiving weak gradients. Model might be ignoring explicit signals.")
    else:
        print("  ✅ Gradient flow is balanced between Wide and Deep paths.")

if __name__ == "__main__":
    debug_gradients()
