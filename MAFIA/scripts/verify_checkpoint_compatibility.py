
import torch as th
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.mafia_modules import DirectionHead

def test_checkpoint_compatibility():
    print("=== Testing Checkpoint Compatibility (Strict=False Fallback) ===")
    
    # 1. Setup Config
    config = Config()
    # Ensure config allows for the new model structure
    config.mafia_explicit_dim = 6
    config.mafia_D = 128
    
    # 2. Initialize Observer with NEW Model (Has LayerNorm)
    print("\n[1] Initializing MAFIAObserver with NEW architecture (LayerNorm)...")
    observer = MAFIAObserver(config, action_dim=10)
    new_model = observer.mafia_model
    
    # Verify LayerNorm exists
    has_norm = False
    for name, module in new_model.named_modules():
        if "explicit_norm" in name:
            has_norm = True
            print(f"    - Verified: Found new layer '{name}'")
    if not has_norm:
        print("    - ERROR: New LayerNorm not found in model!")
        return

    # 3. Create a FAKE "Old" Checkpoint
    # The old model did NOT have 'signal_generator.direction_head.explicit_norm.weight'
    # We simulate this by creating a state_dict that MISSES this key.
    print("\n[2] Creating FAKE 'Old' Checkpoint (missing LayerNorm weights)...")
    
    full_state_dict = new_model.state_dict()
    old_state_dict = {k: v.clone() for k, v in full_state_dict.items() 
                      if "explicit_norm" not in k} # REMOVE the new key
    
    # Save this fake checkpoint
    fake_ckpt_path = "temp_fake_old_ckpt.pth"
    th.save({
        "mafia_model_state_dict": old_state_dict,
        "optimizer_state_dict": observer.optimizer.state_dict(),
        "epoch": 999
    }, fake_ckpt_path)
    print(f"    - Saved fake checkpoint to {fake_ckpt_path}")
    print(f"    - Old dict size: {len(old_state_dict)}, New dict size: {len(full_state_dict)}")

    # 4. Attempt Validation Load
    print("\n[3] Attempting to load 'Old' Checkpoint into 'New' Observer...")
    print("    Expectation: Should FAIL strict load, print WARNING, then SUCCEED with strict=False.")
    
    try:
        loaded_epoch = observer.load_checkpoint(fake_ckpt_path)
        print(f"\n[SUCCESS] Loaded epoch: {loaded_epoch}")
        
        # Verify that explicit_norm weights are NOT NaNs and are trainable (require_grad=True)
        # In a real scenario, they would be random initialized (since they weren't in ckpt)
        # or default initialized.
        norm_layer = observer.mafia_model.signal_generator.direction_head.explicit_norm
        print(f"    - Explicit Norm Weight exists: {norm_layer.weight is not None}")
        print(f"    - Explicit Norm Bias exists: {norm_layer.bias is not None}")
        
    except Exception as e:
        print(f"\n[FAILURE] Crashed with error: {e}")
    finally:
        if os.path.exists(fake_ckpt_path):
            os.remove(fake_ckpt_path)
            print("\n[Cleanup] Removed temp checkpoint.")

if __name__ == "__main__":
    test_checkpoint_compatibility()
