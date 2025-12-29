
import pandas as pd
import numpy as np

def analyze_trajectory(file_path):
    print(f"Analyzing {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Total Rows: {len(df)}")
    
    # 1. Direction Prediction Analysis
    if 'dir_logit_bear' in df.columns:
        logits = df[['dir_logit_bear', 'dir_logit_side', 'dir_logit_bull']].values
        preds = np.argmax(logits, axis=1)
        
        counts = pd.Series(preds).value_counts(normalize=True).sort_index()
        print("\n=== Direction Prediction Distribution ===")
        print(counts)
        
        # Check if logits are frozen (std dev near 0)
        print("\n=== Logit Standard Deviation (frozen check) ===")
        print(df[['dir_logit_bear', 'dir_logit_side', 'dir_logit_bull']].std())

        # Check mean logits
        print("\n=== Mean Logits ===")
        print(df[['dir_logit_bear', 'dir_logit_side', 'dir_logit_bull']].mean())

    # 2. Risk Eta Analysis
    if 'risk_eta' in df.columns:
        print("\n=== Risk Eta Statistics ===")
        print(df['risk_eta'].describe())

    # 3. Advantage Analysis
    if 'advantage' in df.columns and 'raw_return' in df.columns:
        print("\n=== Advantage Statistics ===")
        print(df['advantage'].describe())
        
        # Correlation
        corr = df['advantage'].corr(df['raw_return'])
        print(f"\nCorrelation(Advantage, Raw_Return): {corr:.4f}")
        
        # Check for NaNs
        nans = df['advantage'].isna().sum()
        print(f"NaNs in Advantage: {nans}")

    # 4. Penalty Analysis
    if 'turnover_penalty' in df.columns:
        print("\n=== Turnover Penalty Stats (Non-Zero) ===")
        print(df[df['turnover_penalty'] > 0]['turnover_penalty'].describe())


if __name__ == "__main__":
    import sys
    # Redirect stdout to file
    with open("analysis_report.txt", "w") as f:
        sys.stdout = f
        analyze_trajectory("/Users/nguyensiry/Documents/the_last_dance/agents/MAFIA/observer_offline/trajectory_details.csv")

