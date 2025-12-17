
import torch as th
import sys
import os
import shutil

# Add root to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
mafia_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_root not in sys.path:
    sys.path.append(repo_root)
if mafia_root not in sys.path:
    sys.path.append(mafia_root)

from agents.MAFIA.config import Config
from agents.MAFIA.RL_controller.mafia_observer import MAFIAObserver

def verify_api():
    print("=== Verifying MAFIA Observer API ===")
    
    # 1. Setup Config & Observer
    config = Config()
    # Ensure critical params are set (mocking what might be missing in default)
    config.mafia_T_w = 30
    config.mafia_D = 128
    config.mafia_D_h = 64
    config.mafia_encoder_layers = 2
    config.mafia_encoder_heads = 4
    config.mafia_M_tech = 8  # 5 OCHLV + 3 Ind
    config.mafia_M_dc = 5    # 5 DC features
    config.mafia_explicit_dim = 4 # Matches _compute_explicit_signals output
    config.mafia_learning_rate = 1e-4
    config.mafia_weight_decay = 1e-5
    
    # Mock Action Dim
    N = 10
    observer = MAFIAObserver(config, action_dim=N)
    print("✅ Observer Initialized")
    
    # 2. Test Predict (The Buggy Part)
    print("\n--- Testing predict() ---")
    B = 2
    T_w = config.mafia_T_w
    import numpy as np
    # Mock input: (B, N, 5, T_w) - predict expects numpy array for raw_ochlv_data sometimes?
    # Let's check signature. The error said "expected np.ndarray".
    raw_ochlv = np.random.randn(B, N, 5, T_w).astype(np.float32)
    
    try:
        print(f"   Input shape: {raw_ochlv.shape}")
        # returns 9 values now
        results = observer.predict(
            raw_ochlv_data=raw_ochlv,
            mode='train'
        )
        
        # Unpack to verify count
        (
            market_vector,
            risk_eta,
            market_scores_full,
            market_context,
            direction_logits,
            topk_indices,
            topk_embeddings,
            topk_scores,
            market_logits
        ) = results
        
        print(f"✅ predict() returned 9 values successfully.")
        print(f"   market_logits shape: {market_logits.shape} (Expected: ({B}, {N}))")
        
        # Verify shapes
        assert market_vector.shape == (B, N)
        assert market_logits.shape == (B, N)
        assert market_scores_full.shape == (B, N)
        
    except ValueError as e:
        import traceback
        traceback.print_exc()
        print(f"❌ predict() FAILED: {e}")
        return False
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ predict() FAILED with unexpected error: {e}")
        return False

    # 3. Test Checkpoint Save/Load
    print("\n--- Testing Checkpoint ---")
    ckpt_path = "test_mafia_ckpt.pth"
    try:
        observer.save_checkpoint(ckpt_path, epoch=0)
        print("✅ save_checkpoint passed")
        
        observer.load_checkpoint(ckpt_path)
        print("✅ load_checkpoint passed")
    except Exception as e:
         print(f"❌ Checkpoint test FAILED: {e}")
         return False
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    print("\n=== All API Tests Passed ===")
    return True

if __name__ == "__main__":
    success = verify_api()
    if not success:
        sys.exit(1)
