import sys
import os
import torch as th

# Add RL_controller to path
sys.path.append(os.path.join(os.getcwd(), 'agents/MAFIA'))

from RL_controller.mafia_observer import MAFIAObserver, smart_print

# Mock config to match training parameters
class MockConfig:
    def __init__(self):
        self.mafia_T_w = 30
        self.mafia_D = 256
        self.mafia_D_h = 128
        self.mafia_encoder_layers = 2
        self.mafia_encoder_heads = 4
        self.mafia_M_tech = 8
        self.mafia_M_dc = 5
        self.mafia_M_mkt = 19
        self.mafia_learning_rate = 1e-4
        self.mafia_weight_decay = 1e-4
        self.mafia_DC_multipliers = [1.0, 1.5, 2.0] # Standard config
        self.mafia_gumbel_temperature = 1.0
        self.topK = 40 
        self.mafia_top_k = 40
        self.mafia_train_mode = "MACRO_ONLY"
        self.num_epochs = 1
        self.mafia_lr_schedule = "cosine"
        self.mafia_lr_warmup_epochs = 2
        self.mafia_lr_min = 1e-6
        self.mafia_use_market_index_agent = True
        self.mafia_eta_base = 1.0
        self.mafia_eta_amplitude = 0.3
        self.mafia_eta_min = 0.7
        self.mafia_eta_max = 1.3
        self.mafia_hard_topk_inference = True
        self.mafia_gumbel_temp_inference = 0.1
        self.mafia_lr_direction_head = 5e-4
        self.mafia_lr_risk_head = 1e-4
        self.mafia_lr_backbone = 3e-4
        self.macro_enable_l2_regularization = True
        self.mafia_weight_decay_risk_head = 0.01

def main():
    config = MockConfig()
    action_dim = 122 # Matches training data count
    
    print(f"Initializing MAFIAObserver with N={action_dim}...")
    try:
        observer = MAFIAObserver(config, action_dim)
    except Exception as e:
        print(f"Error initializing observer: {e}")
        return

    ckpt_path = "/Users/nguyensiry/Documents/the_last_dance/agents/MAFIA/observer_offline_6_2/checkpoints/macro/temp_iter_0/epoch_22.pth"
    
    print(f"\nLoading checkpoint from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print("Checkpoint file NOT FOUND!")
        return

    # verified_load_checkpoint trigger our debug prints
    observer.load_checkpoint(ckpt_path, load_optimizer=False)
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
