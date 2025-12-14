
import os
import sys
import pandas as pd
import numpy as np
import shutil

# Ensure repo root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from agents.MAFIA.visualize_training import generate_epoch_report

def run_demo():
    print("🚀 Starting Demo for Curriculum Phase Plotting...")
    
    # Setup directories
    demo_root = os.path.abspath(os.path.join(REPO_ROOT, "agents/MAFIA/demo_results"))
    iter_dir = os.path.join(demo_root, "iter_0_valid_2017")
    
    if os.path.exists(demo_root):
        shutil.rmtree(demo_root)
    os.makedirs(iter_dir)
    
    # Create dummy valid_metrics.csv
    print(f"📝 Creating dummy data in: {iter_dir}")
    
    epochs = [0, 1, 2]
    data = []
    for ep in epochs:
        data.append({
            "epoch": ep,
            "ces_score": 0.5 + ep * 0.1,  # Improved score
            "ces_rank_sharpe": 0.5 + ep * 0.05,
            "ces_rank_dir_f1": 0.5 + ep * 0.05,
            "ces_rank_risk_mse": 0.5 + ep * 0.05,
            
            "topk_sharpe_ratio": 1.0 + ep * 0.2,
            "direction_f1_macro": 0.4 + ep * 0.05,
            "risk_mse": 0.1 - ep * 0.01,
            
            "loss_total": 2.0 - ep * 0.5,
            "loss_pg": 1.0 - ep * 0.2,
            "loss_risk": 0.5 - ep * 0.1,
            "loss_dir": 0.5 - ep * 0.2,
            
            "direction_f1_bear": 0.3,
            "direction_f1_side": 0.4,
            "direction_f1_bull": 0.5,
            "direction_accuracy": 0.5,
            
            "risk_mae": 0.2,
            "risk_correlation": 0.1 + ep * 0.1,
            "topk_turnover": 0.2
        })
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(iter_dir, "valid_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {csv_path}")
    
    # Run generate_epoch_report with min_best_epoch > current epochs
    # This simulates being in the "warmup/rampup" phase where valid best models aren't selected yet
    min_best = 5 
    print(f"🎨 Generating charts (Current Epoch: 2, Min Best Epoch: {min_best})...")
    print("   Expectation: Plots should show 'Curriculum Phase' or 'Pending' labels.")
    
    generate_epoch_report(res_dir=iter_dir, epoch=2, min_best_epoch=min_best)
    
    plots_dir = os.path.join(demo_root, "plots")
    print(f"\n✨ Demo Complete!")
    print(f"📂 Please check the generated charts in: {plots_dir}")
    print(f"   - validation_metrics.png (CES highlighted)")
    print(f"   - ces_breakdown.png")
    
if __name__ == "__main__":
    run_demo()
