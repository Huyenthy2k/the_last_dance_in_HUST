
import torch
import os
import sys

# Path to the specific checkpoint
checkpoint_path = "/Users/nguyensiry/Documents/the_last_dance/agents/MAFIA/observer_offline/checkpoints/temp_iter_0/epoch_16.pth"

if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    # Try to find any epoch file
    dir_path = os.path.dirname(checkpoint_path)
    if os.path.exists(dir_path):
        files = [f for f in os.listdir(dir_path) if f.startswith("epoch_") and f.endswith(".pth")]
        if files:
            files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            checkpoint_path = os.path.join(dir_path, files[0])
            print(f"Fallback: Using latest found checkpoint: {checkpoint_path}")
        else:
            sys.exit(1)
    else:
        sys.exit(1)

print(f"Loading checkpoint: {checkpoint_path}")
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Access state dict
    if 'mafia_model_state_dict' in checkpoint:
        state_dict = checkpoint['mafia_model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint # Assume direct state dict

    # Key for holding bias
    # It is usually under 'signal_generator.holding_bias' or 'module.signal_generator.holding_bias'
    bias_key = 'signal_generator.holding_bias'
    
    val = None
    for k, v in state_dict.items():
        if 'holding_bias' in k:
            val = v.item()
            print(f"Found {k}: {val}")
            
    if val is not None:
        print("-" * 30)
        print(f"Current Holding Bias: {val:.6f}")
        print("-" * 30)
    else:
        print("Could not find 'holding_bias' in state dict keys.")
        # Debug: print keys
        # print(state_dict.keys())

except Exception as e:
    print(f"Failed to inspect: {e}")
