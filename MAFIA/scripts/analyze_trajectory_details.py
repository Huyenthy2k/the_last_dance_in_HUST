
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_trajectory(file_path):
    print(f"Analyzing {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Total rows: {len(df)}")
    print("-" * 30)
    
    # 1. Trigger Analysis
    print("Trigger Counts:")
    print(df['trigger'].value_counts())
    print("-" * 30)
    
    # 2. Rebalancing Frequency
    rebal_rate = df['rebalanced'].mean()
    print(f"Rebalancing Rate: {rebal_rate:.2%}")
    print("-" * 30)
    
    # 3. Loss Analysis
    print("Loss Statistics:")
    print(df[['risk_loss', 'dir_loss', 'grad_norm']].describe())
    print("-" * 30)
    
    # 4. Risk Eta Analysis
    print("Risk Eta Statistics:")
    print(df['risk_eta'].describe())
    print("-" * 30)
    
    # 5. Direction Logits Analysis
    print("Direction Logits Statistics:")
    print(df[['dir_logit_bear', 'dir_logit_side', 'dir_logit_bull']].describe())
    
    # Predicted Class
    logits = df[['dir_logit_bear', 'dir_logit_side', 'dir_logit_bull']].values
    pred_classes = np.argmax(logits, axis=1)
    class_map = {0: 'Bear', 1: 'Side', 2: 'Bull'}
    pred_labels = [class_map[c] for c in pred_classes]
    df['pred_class'] = pred_labels
    
    print("\nPredicted Class Distribution:")
    print(df['pred_class'].value_counts(normalize=True))
    print("-" * 30)
    
    # 6. Correlations
    print("Correlations:")
    cols = ['risk_eta', 'risk_loss', 'dir_loss', 'grad_norm', 'c_mkt_norm']
    print(df[cols].corr())
    print("-" * 30)

    # 7. High Risk Analysis
    print("High Risk Eta (> 0.8) Analysis:")
    high_risk_df = df[df['risk_eta'] > 0.8]
    if not high_risk_df.empty:
        print(f"Count: {len(high_risk_df)}")
        print("Triggers in High Risk:")
        print(high_risk_df['trigger'].value_counts())
    else:
        print("No high risk eta > 0.8 found.")
    print("-" * 30)

if __name__ == "__main__":
    file_path = "/Users/nguyensiry/Documents/the_last_dance/observer_offline_1/phase1_macro/iter_0_valid_2020/trajectory_details.csv"
    if os.path.exists(file_path):
        analyze_trajectory(file_path)
    else:
        print(f"File not found: {file_path}")
