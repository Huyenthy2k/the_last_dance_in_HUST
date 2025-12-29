
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import sys

# File path
FILE_PATH = "/Users/nguyensiry/Documents/the_last_dance/observer_offline_1/phase1_macro/iter_0_valid_2020/trajectory_details.csv"

def analyze_trajectory(file_path):
    print(f"Reading file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Total samples: {len(df)}")
    
    # relevant columns
    logit_cols = ['dir_logit_bear', 'dir_logit_side', 'dir_logit_bull']
    
    if not all(col in df.columns for col in logit_cols):
        print(f"Missing logical columns: {logit_cols}")
        print(f"Available columns: {df.columns}")
        return

    # Turn logits into tensor
    logits = torch.tensor(df[logit_cols].values)
    probs = F.softmax(logits, dim=1)
    
    # Calculate predictions (class with max logit)
    predictions = torch.argmax(probs, dim=1).numpy()
    
    # 0: Bear, 1: Side, 2: Bull
    class_names = ['Bear', 'Side', 'Bull']
    
    # Class Distribution
    print("\n--- Predicted Class Distribution ---")
    counts = np.bincount(predictions, minlength=3)
    total = len(predictions)
    for i, name in enumerate(class_names):
        print(f"{name}: {counts[i]} ({counts[i]/total:.2%})")
        
    # Average Probabilities
    print("\n--- Average Softmax Probabilities ---")
    avg_probs = probs.mean(dim=0).numpy()
    for i, name in enumerate(class_names):
        print(f"{name}: {avg_probs[i]:.4f}")
        
    # Analyze Side Weakness
    print("\n--- Side Label Analysis ---")
    # Check if Side is ever the dominant probability
    side_dominance_count = (predictions == 1).sum()
    print(f"Count where Side is dominant: {side_dominance_count}")
    
    # Compare raw logits
    avg_logits = logits.mean(dim=0).numpy()
    print(f"Average Logits: Bear={avg_logits[0]:.4f}, Side={avg_logits[1]:.4f}, Bull={avg_logits[2]:.4f}")
    
    # Check max logit distribution
    print("\n--- Logit Stats ---")
    print(f"Max Logit Mean: {logits.max(dim=1)[0].mean():.4f}")
    print(f"Min Logit Mean: {logits.min(dim=1)[0].mean():.4f}")
    
    # Loss Analysis
    if 'dir_loss' in df.columns:
        print("\n--- Direction Loss Stats ---")
        print(f"Mean Dir Loss: {df['dir_loss'].mean():.4f}")
        print(f"Median Dir Loss: {df['dir_loss'].median():.4f}")
        print(f"Max Dir Loss: {df['dir_loss'].max():.4f}")
        print(f"Min Dir Loss: {df['dir_loss'].min():.4f}")
        
    # Analyze samples with high loss
    if 'dir_loss' in df.columns:
        high_loss_threshold = df['dir_loss'].quantile(0.9)
        print(f"\n--- High Loss Samples (> {high_loss_threshold:.4f}) ---")
        high_loss_df = df[df['dir_loss'] > high_loss_threshold]
        
        # Check distribution in high loss
        hl_logits = torch.tensor(high_loss_df[logit_cols].values)
        hl_preds = torch.argmax(hl_logits, dim=1).numpy()
        hl_counts = np.bincount(hl_preds, minlength=3)
        print("Predicted distribution in high loss samples:")
        for i, name in enumerate(class_names):
            print(f"{name}: {hl_counts[i]} ({hl_counts[i]/len(high_loss_df):.2%})")

if __name__ == "__main__":
    analyze_trajectory(FILE_PATH)
