#!/usr/bin/env python3
"""
Generate Observer Training Visualization Charts

Usage:
    python scripts/generate_charts.py --input ./observer_test/iter_0_valid_2017/valid_metrics.csv --output ./charts

    # With trajectory details
    python scripts/generate_charts.py \
        --input ./observer_test/iter_0_valid_2017/valid_metrics.csv \
        --trajectory ./observer_test/trajectory_details.csv \
        --output ./charts

    # Walk-forward multi-iteration
    python scripts/generate_charts.py \
        --walk-forward ./observer_test/offline_training_summary.json \
        --output ./charts
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add MAFIA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from RL_controller.plotting_utils import (
    create_all_charts,
    plot_loss_components,
    plot_validation_metrics_grid,
    plot_ces_progression,
    plot_walkforward_metrics_comparison,
    plot_trigger_distribution,
)


def parse_walkforward_summary(json_path: str):
    """
    Parse offline_training_summary.json to extract iterations summary.
    
    Returns:
        List[Dict] with keys: year, ces, train_range, sharpe, dir_f1, risk_mse
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    iterations = []
    for result in data:
        if result['status'] == 'success':
            metrics = result.get('best_metrics', {})
            iterations.append({
                'year': result['valid_year'],
                'ces': metrics.get('ces_score', 0.0),
                'train_range': result['train_range'].split(' → ')[0].split('-')[0] + '-' + 
                               result['train_range'].split(' → ')[1].split('-')[0],  # "2015-2017"
                'sharpe': metrics.get('topk_sharpe_ratio', 0.0),
                'dir_f1': metrics.get('direction_f1_macro', 0.0),
                'risk_mse': metrics.get('risk_mse', 0.0),
            })
    
    return iterations


def main():
    parser = argparse.ArgumentParser(description="Generate Observer training charts")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to valid_metrics.csv (single iteration)",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        help="Optional path to trajectory_details.csv for trigger analysis",
    )
    parser.add_argument(
        "--walk-forward",
        type=str,
        help="Path to offline_training_summary.json for multi-iteration analysis",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./charts",
        help="Output directory for charts (default: ./charts)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input and not args.walk_forward:
        print("❌ Error: Must provide either --input or --walk-forward")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "="*70)
    print("OBSERVER TRAINING CHART GENERATION")
    print("="*70)
    
    # Single iteration charts
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file not found: {args.input}")
            sys.exit(1)
        
        print(f"\n📊 Generating charts from: {args.input}")
        
        iterations_summary = None
        if args.walk_forward and os.path.exists(args.walk_forward):
            iterations_summary = parse_walkforward_summary(args.walk_forward)
        
        create_all_charts(
            valid_csv=args.input,
            iterations_summary=iterations_summary,
            trajectory_csv=args.trajectory,
            output_dir=args.output,
        )
    
    # Walk-forward multi-iteration charts
    elif args.walk_forward:
        if not os.path.exists(args.walk_forward):
            print(f"❌ Error: Summary file not found: {args.walk_forward}")
            sys.exit(1)
        
        print(f"\n📊 Generating walk-forward charts from: {args.walk_forward}")
        
        iterations_summary = parse_walkforward_summary(args.walk_forward)
        
        if not iterations_summary:
            print("❌ Error: No successful iterations found in summary")
            sys.exit(1)
        
        # Generate walk-forward specific charts
        print("[1/2] CES Progression...")
        plot_ces_progression(
            iterations_summary,
            output_path=os.path.join(args.output, "walkforward_ces_progression.png")
        )
        
        print("[2/2] Metrics Comparison...")
        plot_walkforward_metrics_comparison(
            iterations_summary,
            output_path=os.path.join(args.output, "walkforward_metrics.png")
        )
    
    print(f"\n{'='*70}")
    print(f"✅ All charts saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
