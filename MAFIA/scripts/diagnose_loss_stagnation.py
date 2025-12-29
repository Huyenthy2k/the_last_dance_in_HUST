import sys
import os
import torch as th
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from agents.MAFIA.config import Config, MafiaTrainMode
from agents.MAFIA.RL_controller.mafia_observer import MAFIAObserver
from agents.MAFIA.RL_controller.observer_offline_trainer import FocalLoss

def diagnose_stagnation():
    print("=== Diagnosing Stagnant Loss (Epoch 3-5 conditions) ===")
    
    # 1. Setup Config & Model
    config = Config()
    config.mafia_train_mode = MafiaTrainMode.MACRO_ONLY
    # Ensure relevant params match what we saw in finding
    config.mafia_lambda_dir = 1.0 # From config inspection
    config.mafia_lambda_risk = 1.0
    config.scale_factor_risk = 1.5
    
    # Force device to CPU for easy debugging
    device = th.device("cpu")
    
    observer = MAFIAObserver(config, action_dim=10) # Dummy action dim
    observer.mafia_model.to(device)
    observer.mafia_model.set_train_mode(MafiaTrainMode.MACRO_ONLY)
    
    # 2. Create Dummy Batch (Simulate inputs)
    B = 32
    D = config.mafia_D
    
    # Random inputs
    c_mkt = th.randn(B, D, device=device, requires_grad=True)
    delta_c_mkt = th.randn(B, D, device=device, requires_grad=True)
    # Explicit signals: [Vol, DC, Breadth, Div, VPI, DD]
    # Set them to random but bounded values
    explicit_signals = th.randn(B, 6, device=device).clamp(-3, 3)
    explicit_signals.requires_grad = True
    
    # Targets
    # Direction: Random classes 0, 1, 2
    dir_targets = th.randint(0, 3, (B,), device=device)
    # Risk: Random eta in [0.7, 1.3]
    risk_targets = th.rand(B, device=device) * 0.6 + 0.7
    
    # 3. Access Heads directly for granular check
    risk_head = observer.mafia_model.signal_generator.risk_head
    dir_head = observer.mafia_model.signal_generator.direction_head
    
    print("\n--- 1. Forward Pass Check ---")
    
    # Risk Head
    eta_raw = risk_head(c_mkt, delta_c_mkt, explicit_signals)
    # Apply tanh logic from code
    eta = config.mafia_eta_base + config.mafia_eta_amplitude * th.tanh(eta_raw)
    
    print(f"Risk Eta Mean: {eta.mean().item():.4f}, Std: {eta.std().item():.4f}")
    print(f"Risk Eta GradFn: {eta.grad_fn}")
    
    # Direction Head
    logits = dir_head(c_mkt, delta_c_mkt, explicit_signals)
    probs = th.softmax(logits, dim=-1)
    
    print(f"Dir Logits Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
    print(f"Dir Probs Mean: {probs.mean(dim=0)}") # Check class balance output
    
    # 4. Loss Calculation
    criterion_risk = th.nn.MSELoss()
    criterion_dir = FocalLoss(gamma=2.0, alpha=[1.5, 0.4, 1.0], device=device, label_smoothing=0.1)
    
    loss_risk = criterion_risk(eta, risk_targets) * config.scale_factor_risk
    loss_dir = criterion_dir(logits, dir_targets)
    
    total_loss = loss_risk * config.mafia_lambda_risk + loss_dir * config.mafia_lambda_dir
    
    print("\n--- 2. Loss Values ---")
    print(f"Loss Risk: {loss_risk.item():.6f}")
    print(f"Loss Dir:  {loss_dir.item():.6f}")
    print(f"Total Loss: {total_loss.item():.6f}")
    
    # 5. Backward Pass & Gradient Check
    print("\n--- 3. Gradient Flow Check ---")
    observer.optimizer.zero_grad()
    total_loss.backward()
    
    def check_grad(name, param):
        if param.grad is None:
            print(f"❌ {name}: Grad is None!")
        else:
            grad_norm = param.grad.norm().item()
            status = "✅" if grad_norm > 1e-6 else "Warning: Small/Zero"
            print(f"{status} {name}: Norm={grad_norm:.6f}")
            if hasattr(param, "_is_explicit_weight"): # Hypothetical tag
                 print(f"   Explicit Weight for {name}")

    # Inspect specific parameters of interest
    print(">> Risk Head Grads:")
    check_grad("Risk.Expand.Weight", risk_head.expand_proj.weight)
    check_grad("Risk.Compress.Weight", risk_head.compress_proj.weight)
    check_grad("Risk.Fusion.Weight", risk_head.fusion_net.weight)
    # Specifically check Wide Path weights (last 6 dims of fusion layer)
    fusion_grad = risk_head.fusion_net.weight.grad
    if fusion_grad is not None:
        wide_grads = fusion_grad[:, -6:].norm(dim=0)
        print(f"   Wide Path Signal Grads: {wide_grads}")
    
    print(">> Direction Head Grads:")
    check_grad("Dir.Expand.Weight", dir_head.expand_proj.weight)
    check_grad("Dir.Compress.Weight", dir_head.compress_proj.weight)
    check_grad("Dir.Classifier.Weight", dir_head.classifier.weight)
    
    # Inspect Wide Path grads for Direction
    cls_grad = dir_head.classifier.weight.grad
    if cls_grad is not None:
         wide_grads_dir = cls_grad[:, -6:].norm(dim=0) # Sum across classes? or norm
         print(f"   Wide Path Signal Grads (Dir): {wide_grads_dir}")
         
    # 6. Weight Initialization Check (Potential Saturation Cause)
    print("\n--- 4. Weight Initialization Analysis ---")
    print("Checking if usage of custom initialization makes weights too large...")
    print(f"Risk Fusion Weights (Mean/Std): {risk_head.fusion_net.weight.mean().item():.4f} / {risk_head.fusion_net.weight.std().item():.4f}")
    print(f"Dir Classifier Weights (Mean/Std): {dir_head.classifier.weight.mean().item():.4f} / {dir_head.classifier.weight.std().item():.4f}")

    # 7. Check Focal Loss Behavior on "Stuck" Predictions
    # Simulate the "Stuck" case: Model predicts Side (class 1) with high confidence
    print("\n--- 5. Simulation: Stuck on 'Side' Class ---")
    # Force logits to favor class 1 strongly
    stuck_logits = th.tensor([[ -1.0, 2.0, -1.0 ]], device=device) 
    # Target is Bull (2) -> Should have high loss
    target_bull = th.tensor([2], device=device)
    loss_stuck = criterion_dir(stuck_logits, target_bull)
    print(f"Stuck Logits: {stuck_logits}")
    print(f"Target: Bull (2)")
    print(f"Loss (should be high): {loss_stuck.item():.6f}")
    
    # Calculate gradient for stuck case
    stuck_logits.requires_grad = True
    loss_stuck = criterion_dir(stuck_logits, target_bull)
    loss_stuck.backward()
    print(f"Gradients on Logits: {stuck_logits.grad}")

if __name__ == "__main__":
    diagnose_stagnation()
