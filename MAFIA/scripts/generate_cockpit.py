import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Add repo root to path
sys.path.append(os.getcwd())

from RL_controller.plotting_utils import plot_cockpit_dashboard

def main():
    parser = argparse.ArgumentParser(description="Generate MAFIA Cockpit Dashboard")
    parser.add_argument("--csv", type=str, required=True, help="Path to valid_metrics.csv")
    parser.add_argument("--output", type=str, default="dynamic_dashboard.png", help="Output filename")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV not found at {args.csv}")
        return

    print(f"Generating dashboard from {args.csv}...")
    
    # Generate
    try:
        plot_cockpit_dashboard(args.csv, args.output)
        print(f"Dashboard saved to {args.csv}")
    except Exception as e:
        print(f"Failed to generate dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
