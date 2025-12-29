"""
Observer Training Visualization Utilities

Provides plotting functions for Observer training analysis:
- Loss curves
- Validation metrics
- CES progression
- Walk-forward performance
- Portfolio analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_loss_components(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "Loss Components Over Epochs"
) -> plt.Figure:
    """
    Plot 4 loss components in separate subplots.
    
    Args:
        csv_path: Path to valid_metrics.csv
        output_path: Optional path to save figure
        title: Overall figure title
        
    Returns:
        matplotlib Figure object
    """
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Total Loss
    axes[0].plot(df['epoch'], df['loss_total'], 'o-', color='black', linewidth=2, label='Total Loss')
    axes[0].set_ylabel('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Policy Gradient Loss
    axes[1].plot(df['epoch'], df['loss_pg'], 'o-', color='blue', linewidth=2, label='L_PG (Selection)')
    axes[1].set_ylabel('L_PG')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Risk Loss
    axes[2].plot(df['epoch'], df['loss_risk'], 'o-', color='orange', linewidth=2, label='L_Risk (Hybrid)')
    axes[2].set_ylabel('L_Risk')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Direction Loss
    # Direction Loss
    axes[3].plot(df['epoch'], df['loss_dir'], 'o-', color='green', linewidth=2, label='L_Dir (CrossEntropy)')
    axes[3].set_ylabel('L_Dir')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Balance Loss
    if 'loss_bal' in df.columns:
        axes[4].plot(df['epoch'], df['loss_bal'], 'o-', color='purple', linewidth=2, label='L_Bal (Load Balancing)')
    else:
        axes[4].text(0.5, 0.5, "L_Bal Not Available", transform=axes[4].transAxes, ha='center')
        
    axes[4].set_ylabel('L_Bal')
    axes[4].set_xlabel('Epoch')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved loss components to {output_path}")
    
    return fig


def plot_validation_metrics_grid(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "Validation Metrics",
    min_best_epoch: int = 0
) -> plt.Figure:
    """
    Plot validation metrics in 2×2 grid with best epoch marked.
    
    Subplots:
      - Top-left: Sharpe Ratio
      - Top-right: Direction F1 (Macro)
      - Bottom-left: Risk MSE (inverted)
      - Bottom-right: CES Score (highlighted)
    """
    df = pd.read_csv(csv_path)
    
    # Filter for valid 'Best' candidates (Curriculum Logic)
    if 'ces_score' in df.columns:
        valid_candidates = df[df['epoch'] >= min_best_epoch]
        if not valid_candidates.empty:
            best_idx = valid_candidates['ces_score'].idxmax()
            best_epoch = df.loc[best_idx, 'epoch']
            best_ces = df.loc[best_idx, 'ces_score']
            best_label = f'Best: Epoch {int(best_epoch)}'
        else:
            # Fallback if no valid candidates yet (early curriculum phase)
            best_idx = df['ces_score'].idxmax()
            best_epoch = df.loc[best_idx, 'epoch']
            best_ces = df.loc[best_idx, 'ces_score']
            best_label = f'Best (Curriculum): {int(best_epoch)}'
    else:
        # Fallback for train_metrics where CES might not exist
        # Use simple "Last Epoch" or "Min Total Loss"
        best_idx = df.index[-1]
        best_epoch = df.loc[best_idx, 'epoch']
        best_ces = 0.0
        best_label = f'Latest: {int(best_epoch)}'
        valid_candidates = pd.DataFrame() # Empty to suppress "Best" star unless logic added
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Sharpe Ratio
    axes[0, 0].plot(df['epoch'], df['topk_sharpe_ratio'], 'o-', color='blue')
    axes[0, 0].axvline(best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Best CES')
    axes[0, 0].set_title('Top-K Sharpe Ratio (Higher = Better)')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Direction F1
    axes[0, 1].plot(df['epoch'], df['direction_f1_macro'], 'o-', color='green')
    axes[0, 1].axvline(best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Best CES')
    axes[0, 1].set_title('Direction F1 Macro (Higher = Better)')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Risk MSE (inverted y-axis)
    axes[1, 0].plot(df['epoch'], df['risk_mse'], 'o-', color='orange')
    axes[1, 0].axvline(best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Best CES')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title('Risk Prediction MSE (Lower = Better)')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # CES Score (highlighted)
    if 'ces_score' in df.columns:
        axes[1, 1].plot(df['epoch'], df['ces_score'], 'o-', color='purple', linewidth=2.5, markersize=6)
        
        # Only show start if it is a VALID best (passed curriculum)
        if not valid_candidates.empty:
            axes[1, 1].scatter(best_epoch, best_ces, 
                               s=300, c='gold', marker='*', edgecolor='black', linewidth=2, zorder=5,
                               label=best_label)
        else:
            # Show "Pending" indicator
            axes[1, 1].text(0.5, 0.5, "Best Checkpoint: Pending\n(Curriculum Phase)", 
                           transform=axes[1, 1].transAxes, ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.8))
        axes[1, 1].set_title('Composite Efficiency Score (CES)')
        axes[1, 1].set_ylabel('CES Score')
    else:
        # If CES missing, plot something else or leave blank?
        # Maybe Total Loss?
        axes[1, 1].plot(df['epoch'], df.get('loss_total', np.zeros_like(df['epoch'])), 'o-', color='black', linewidth=2, label='Total Loss')
        axes[1, 1].set_title('Total Loss (CES N/A)')
        axes[1, 1].set_ylabel('Loss')
        
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved validation metrics grid to {output_path}")
    
    return fig


def plot_ces_progression(
    iterations_summary: List[Dict],
    output_path: Optional[str] = None,
    title: str = "Walk-Forward Sharpe Progression"
) -> plt.Figure:
    """
    Plot Sharpe Ratio (phase_score for SELECTION_ONLY) across walk-forward iterations.

    Args:
        iterations_summary: List of dicts with keys:
            - year: Validation year
            - phase_score: Best phase score (Sharpe for SELECTION)
            - train_range: Training window (e.g., "2015-2017")
            - sharpe, dir_f1, risk_mse: Metric values
    """
    years = [it['year'] for it in iterations_summary]
    # Use phase_score (which is Sharpe for SELECTION_ONLY)
    scores = [it.get('phase_score', it.get('sharpe', 0.0)) for it in iterations_summary]

    # Normalize scores to [0, 1] for color mapping
    normalized_scores = np.array(scores)
    if normalized_scores.max() > normalized_scores.min():
        normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
    else:
        normalized_scores = np.ones_like(normalized_scores) * 0.5

    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar chart with gradient colors
    colors = plt.cm.RdYlGn(normalized_scores)
    bars = ax.bar(years, scores, color=colors, edgecolor='black', linewidth=1.5)

    # Annotate bars
    for bar, it in zip(bars, iterations_summary):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f"{it['train_range']}\nSharpe={height:.3f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Validation Year', fontsize=12)
    ax.set_ylabel('Best Sharpe Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    # Add color bar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved Sharpe progression to {output_path}")

    return fig


def plot_walkforward_metrics_comparison(
    iterations_summary: List[Dict],
    output_path: Optional[str] = None,
    title: str = "Walk-Forward Validation Metrics"
) -> plt.Figure:
    """
    Compare Sharpe, F1, and MSE across walk-forward iterations.
    """
    years = [it['year'] for it in iterations_summary]
    sharpe = [it['sharpe'] for it in iterations_summary]
    dir_f1 = [it['dir_f1'] for it in iterations_summary]
    risk_mse = [it['risk_mse'] for it in iterations_summary]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary y-axis: Sharpe Ratio
    color1 = 'tab:blue'
    ax1.set_xlabel('Validation Year', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', color=color1, fontsize=12)
    line1 = ax1.plot(years, sharpe, 'o-', color=color1, linewidth=2, markersize=8, label='Sharpe Ratio')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Secondary y-axis: F1 and MSE
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    color3 = 'tab:orange'
    ax2.set_ylabel('F1 / MSE', fontsize=12)
    line2 = ax2.plot(years, dir_f1, 's-', color=color2, linewidth=2, markersize=8, label='Direction F1')
    line3 = ax2.plot(years, risk_mse, '^-', color=color3, linewidth=2, markersize=8, label='Risk MSE')
    
    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved walk-forward comparison to {output_path}")
    
    return fig


def plot_trigger_distribution(
    trajectory_csv: str,
    output_path: Optional[str] = None,
    title: str = "Trigger Distribution"
) -> plt.Figure:
    """
    Pie chart showing trigger type distribution.
    
    Args:
        trajectory_csv: Path to trajectory_details.csv
    """
    df = pd.read_csv(trajectory_csv)
    
    # Count triggers
    trigger_counts = df['trigger'].value_counts()
    
    # Color mapping
    colors_map = {
        'NONE': '#d3d3d3',
        'SCHEDULE': '#4CAF50',
        'VOL_SHOCK': '#FF9800',
        'DIR_REVERSAL': '#2196F3',
    }
    colors = [colors_map.get(t, '#9E9E9E') for t in trigger_counts.index]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        trigger_counts,
        labels=trigger_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.05 if t != 'NONE' else 0 for t in trigger_counts.index],
        shadow=True
    )
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # Add counts
    for i, (label, count) in enumerate(zip(trigger_counts.index, trigger_counts)):
        texts[i].set_text(f"{label}\n({count:,} timesteps)")
        texts[i].set_fontsize(10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved trigger distribution to {output_path}")
    
    return fig


def plot_ces_components_breakdown(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "CES Score Components Breakdown",
    min_best_epoch: int = 0
) -> plt.Figure:
    """
    Visualize how each component (Sharpe, Direction F1, Risk MSE) contributes to CES.

    CES Formula: 0.6 * Sharpe_norm + 0.2 * Dir_F1_norm + 0.2 * (1 - Risk_MSE_norm)

    Args:
        csv_path: Path to valid_metrics.csv with ces_rank_* columns
        output_path: Optional path to save figure
        title: Figure title
        min_best_epoch: Minimum epoch index to consider for "Best" selection (Curriculum)
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ['epoch', 'ces_score', 'ces_rank_sharpe', 'ces_rank_dir_f1', 'ces_rank_risk_mse']
    if not all(col in df.columns for col in required_cols):
        print(f"[WARN] Missing CES rank columns. Available: {df.columns.tolist()}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Filter for valid 'Best' candidates (Curriculum Logic)
    valid_candidates = df[df['epoch'] >= min_best_epoch]
    if not valid_candidates.empty:
        best_idx = valid_candidates['ces_score'].idxmax()
        best_epoch = df.loc[best_idx, 'epoch']
        best_ces = df.loc[best_idx, 'ces_score']
        best_label = f'Best (Epoch {int(best_epoch)})'
        best_color = 'gold'
    else:
        # Fallback
        best_idx = df['ces_score'].idxmax()
        best_epoch = df.loc[best_idx, 'epoch']
        best_ces = df.loc[best_idx, 'ces_score']
        best_label = f'Best (Curriculum Phase)'
        best_color = 'lightgray'

    # Plot 1: Stacked bar chart showing component contributions
    ax1 = axes[0, 0]
    epochs = df['epoch']

    # Calculate weighted contributions
    sharpe_contrib = 0.5 * df['ces_rank_sharpe']
    dir_f1_contrib = 0.3 * df['ces_rank_dir_f1']
    risk_mse_contrib = 0.2 * df['ces_rank_risk_mse']  # Rank is already Goodness (High=Best)

    ax1.bar(epochs, sharpe_contrib, label='Sharpe (50%)', color='#2196F3', alpha=0.8)
    ax1.bar(epochs, dir_f1_contrib, bottom=sharpe_contrib, label='Direction F1 (30%)', color='#4CAF50', alpha=0.8)
    ax1.bar(epochs, risk_mse_contrib, bottom=sharpe_contrib + dir_f1_contrib, label='Risk (20%)', color='#FF9800', alpha=0.8)
    
    if not valid_candidates.empty:
        ax1.axvline(best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7, label=best_label)
    else:
         ax1.text(0.5, 0.9, "Curriculum Phase (No Best Yet)", transform=ax1.transAxes, ha='center', color='gray')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CES Contribution')
    ax1.set_title('Weighted Component Contributions')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Raw rank values
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['ces_rank_sharpe'], 'o-', color='#2196F3', linewidth=2, label='Sharpe Rank')
    ax2.plot(epochs, df['ces_rank_dir_f1'], 's-', color='#4CAF50', linewidth=2, label='Direction F1 Rank')
    ax2.plot(epochs, df['ces_rank_risk_mse'], '^-', color='#FF9800', linewidth=2, label='Risk MSE Rank (High=Good)')
    if not valid_candidates.empty:
        ax2.axvline(best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Normalized Rank [0-1]')
    ax2.set_title('Individual Component Ranks')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Calculate dynamic y-limit to avoid cutting off high ranks (when epochs > fixed N)
    max_rank = max(
        df['ces_rank_sharpe'].max(),
        df['ces_rank_dir_f1'].max(),
        df['ces_rank_risk_mse'].max()
    )
    upper_limit = max(1.1, max_rank + 0.1)
    ax2.set_ylim(-0.1, upper_limit)

    # Plot 3: Pie chart of best epoch composition
    ax3 = axes[1, 0]
    best_row = df.loc[best_idx]
    contributions = [
        0.5 * best_row['ces_rank_sharpe'],
        0.3 * best_row['ces_rank_dir_f1'],
        0.2 * best_row['ces_rank_risk_mse']
    ]
    labels = ['Sharpe\n(50% weight)', 'Direction F1\n(30% weight)', 'Risk\n(20% weight)']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    wedges, texts, autotexts = ax3.pie(
        contributions, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=[0.05, 0, 0]
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax3.set_title(f'Best Epoch ({int(best_epoch)}) CES Composition\nTotal CES: {best_row["ces_score"]:.3f}')

    # Plot 4: CES over epochs with trend
    ax4 = axes[1, 1]
    ax4.plot(epochs, df['ces_score'], 'o-', color='purple', linewidth=2.5, markersize=8)
    
    if not valid_candidates.empty:
        # Plot Curriculum Best (Gold Star)
        ax4.scatter(best_epoch, best_row['ces_score'], s=300, c='gold', marker='*',
                    edgecolor='black', linewidth=2, zorder=5, label=f'Best (Valid): {best_row["ces_score"]:.3f}')
        
        # Check for Global Best (if different from Curriculum Best)
        global_best_idx = df['ces_score'].idxmax()
        if global_best_idx != best_idx:
            g_epoch = df.loc[global_best_idx, 'epoch']
            g_score = df.loc[global_best_idx, 'ces_score']
            ax4.scatter(g_epoch, g_score, s=150, c='silver', marker='*', 
                       edgecolor='gray', linewidth=1, zorder=4, 
                       label=f'Global Max (Warmup): {g_score:.3f}')
    else:
        ax4.text(0.5, 0.5, "Curriculum Phase", transform=ax4.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # Add trend line
    if len(epochs) > 1 and df['ces_score'].notna().all() and np.isfinite(df['ces_score']).all():
        try:
            z = np.polyfit(epochs, df['ces_score'], 1)
            p = np.poly1d(z)
            ax4.plot(epochs, p(epochs), '--', color='gray', alpha=0.5, label=f'Trend (slope: {z[0]:.4f})')
        except Exception as e:
            print(f"[WARN] Trend line fitting failed: {e}")
            pass
    elif len(epochs) <= 1:
        pass # Not enough data for trend line

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('CES Score')
    ax4.set_title('CES Score Progression')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved CES components breakdown to {output_path}")

    return fig


def plot_direction_breakdown(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "Direction Classification Breakdown"
) -> plt.Figure:
    """
    Visualize per-class F1 scores (Bear/Side/Bull) and detect class imbalance issues.

    Args:
        csv_path: Path to valid_metrics.csv with direction_f1_* columns
        output_path: Optional path to save figure
        title: Figure title
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ['epoch', 'direction_f1_bear', 'direction_f1_side', 'direction_f1_bull', 'direction_accuracy']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    epochs = df['epoch']
    best_idx = df['direction_f1_macro'].idxmax() if 'direction_f1_macro' in df.columns else 0
    best_epoch = df.loc[best_idx, 'epoch'] if best_idx > 0 else epochs.iloc[-1]

    # Detect scale (Ratio vs Percentage)
    # If accuracy mean > 1.0, it's percentage.
    is_percentage = df['direction_accuracy'].mean() > 1.0
    scale_factor = 100.0 if not is_percentage else 1.0 # If input is ratio, scale to %. If input is %, keep.
    
    # Actually, user wants % everywhere.
    # Convert ALL to Percentage (0-100)
    
    # Deep copy to avoiding modifying original DF if passed elsewhere (though local read)
    df_plot = df.copy()
    
    # F1 columns are typically 0-1. Convert to %
    f1_cols = ['direction_f1_bear', 'direction_f1_side', 'direction_f1_bull', 'direction_f1_macro']
    for col in f1_cols:
        if col in df_plot.columns and df_plot[col].max() <= 1.0:
             df_plot[col] = df_plot[col] * 100.0
             
    # Accuracy column: if max <= 1.0, convert to %
    if df_plot['direction_accuracy'].max() <= 1.0:
        df_plot['direction_accuracy'] = df_plot['direction_accuracy'] * 100.0
        
    # Standardize Constants
    baseline_random = 33.33
    baseline_majority = 46.0
    ylim_max = 105

    # Plot 1: Per-class F1 over epochs
    ax1 = axes[0, 0]
    ax1.plot(epochs, df_plot['direction_f1_bear'], 'o-', color='#F44336', linewidth=2, markersize=6, label='Bear F1')
    ax1.plot(epochs, df_plot['direction_f1_side'], 's-', color='#9E9E9E', linewidth=2, markersize=6, label='Side F1')
    ax1.plot(epochs, df_plot['direction_f1_bull'], '^-', color='#4CAF50', linewidth=2, markersize=6, label='Bull F1')
    ax1.axvline(best_epoch, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Macro F1')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1 Score (%)')
    ax1.set_title('Per-Class F1 Over Epochs')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Plot 2: Grouped bar chart of final epoch F1 scores
    ax2 = axes[0, 1]
    final_row = df_plot.iloc[-1]
    classes = ['Bear', 'Side', 'Bull']
    f1_scores = [final_row['direction_f1_bear'], final_row['direction_f1_side'], final_row['direction_f1_bull']]
    colors = ['#F44336', '#9E9E9E', '#4CAF50']

    bars = ax2.bar(classes, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=baseline_random, color='gray', linestyle='--', alpha=0.5, label='Random (33%)')

    # Annotate bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 2, f'{score:.5f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('F1 Score (%)')
    ax2.set_title(f'Per-Class F1 (Epoch {int(final_row["epoch"])})')
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(0, 110)

    # Plot 3: Macro F1 vs Accuracy comparison
    ax3 = axes[1, 0]
    if 'direction_f1_macro' in df_plot.columns:
        ax3.plot(epochs, df_plot['direction_f1_macro'], 'o-', color='purple', linewidth=2, label='Macro F1')
    ax3.plot(epochs, df_plot['direction_accuracy'], 's-', color='blue', linewidth=2, alpha=0.7, label='Accuracy')
    ax3.axhline(y=baseline_majority, color='gray', linestyle='--', alpha=0.5, label='Majority (~46%)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score (%)')
    ax3.set_title('Macro F1 vs Accuracy')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)

    # Plot 4: Class imbalance detection heatmap
    ax4 = axes[1, 1]

    # Create heatmap data: rows = epochs, cols = classes
    # Use normalized 0-1 for heatmap color intensity but label with %
    heatmap_data_pct = df_plot[['direction_f1_bear', 'direction_f1_side', 'direction_f1_bull']].values
    heatmap_data_norm = heatmap_data_pct / 100.0

    im = ax4.imshow(heatmap_data_norm.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Bear', 'Side', 'Bull'])
    ax4.set_xlabel('Epoch')
    ax4.set_xticks(range(len(epochs)))
    ax4.set_xticklabels([int(e) for e in epochs])
    ax4.set_title('F1 Score Heatmap (Green=Good)')

    # Annotate heatmap
    for i in range(3):
        for j in range(len(epochs)):
            val = heatmap_data_pct[j, i]
            color = 'white' if val < 50 else 'black'
            ax4.text(j, i, f'{val:.5f}', ha='center', va='center', color=color, fontsize=6)

    plt.colorbar(im, ax=ax4, label='F1 Score (Normalized)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved direction breakdown to {output_path}")

    return fig


def plot_risk_calibration(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "Risk Calibration Analysis"
) -> plt.Figure:
    """
    Analyze risk prediction quality (η prediction vs target).

    Args:
        csv_path: Path to valid_metrics.csv with risk_* columns
        output_path: Optional path to save figure
        title: Figure title
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ['epoch', 'risk_mse', 'risk_mae', 'risk_correlation']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    epochs = df['epoch']

    # Plot 1: Risk MSE over epochs
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['risk_mse'], 'o-', color='#F44336', linewidth=2, markersize=8)
    ax1.fill_between(epochs, df['risk_mse'], alpha=0.3, color='#F44336')

    # Highlight best (lowest) MSE
    best_idx = df['risk_mse'].idxmin()
    best_epoch = df.loc[best_idx, 'epoch']
    best_mse = df.loc[best_idx, 'risk_mse']
    
    # Simple check: if best epoch < min_best_epoch, maybe mark it differently?
    # For Risk Calibration, we might care about best technical risk even if penalties aren't full.
    # But for consistency, let's keep it simple or update later if requested.
    # Leaving Risk Calibration as-is for now (mostly technical), or user can request update.
    # The user specifically mentioned "best checkpoint" which usually refers to CES.
    
    ax1.scatter(best_epoch, best_mse, s=200, c='gold', marker='*', edgecolor='black',
                linewidth=2, zorder=5, label=f'Best: {best_mse:.4f}')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Risk Prediction MSE (Lower = Better)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Risk MAE over epochs
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['risk_mae'], 'o-', color='#FF9800', linewidth=2, markersize=8)
    ax2.fill_between(epochs, df['risk_mae'], alpha=0.3, color='#FF9800')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Risk Prediction MAE (Lower = Better)')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # Plot 3: Risk Correlation over epochs
    ax3 = axes[1, 0]
    colors = ['#4CAF50' if c > 0 else '#F44336' for c in df['risk_correlation']]
    ax3.bar(epochs, df['risk_correlation'], color=colors, edgecolor='black', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good correlation (0.5)')
    ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Inverse correlation (-0.5)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Risk η Prediction Correlation (Higher = Better)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)

    # Plot 4: Combined metrics summary
    ax4 = axes[1, 1]

    # Normalize metrics for radar-like comparison
    final_row = df.iloc[-1]

    # Create summary text box with HybridLoss formula
    alpha = 0.7  # Default HybridLoss alpha
    mse_score = 1.0 - min(final_row['risk_mse'], 1.0)
    corr_score = max(0.0, final_row['risk_correlation'])
    risk_score = (1.0 - alpha) * mse_score + alpha * corr_score

    summary_text = f"""Risk Calibration Summary (Epoch {int(final_row['epoch'])})

MSE:  {final_row['risk_mse']:.4f}  {'(Good < 0.1)' if final_row['risk_mse'] < 0.1 else '(Needs improvement)'}
MAE:  {final_row['risk_mae']:.4f}  {'(Good < 0.3)' if final_row['risk_mae'] < 0.3 else '(Needs improvement)'}
Corr: {final_row['risk_correlation']:.4f}  {'(Good > 0.3)' if final_row['risk_correlation'] > 0.3 else '(Weak/Negative)'}

HybridLoss Risk Score (α={alpha}):
  mse_score  = 1 - MSE = {mse_score:.4f}
  corr_score = max(0, ρ) = {corr_score:.4f}
  risk_score = {int((1-alpha)*100)}%×{mse_score:.2f} + {int(alpha*100)}%×{corr_score:.2f} = {risk_score:.4f}

Interpretation:
- Positive correlation: Model learns risk dynamics
- Low MSE/MAE: Accurate η predictions
- Negative correlation: Model may be overfitting
"""

    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Calibration Assessment')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved risk calibration to {output_path}")

    return fig


def plot_macro_phase_score(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "MACRO Phase Score Breakdown",
    alpha: float = 0.7  # HybridLoss alpha for risk_score calculation
) -> plt.Figure:
    """
    Plot MACRO phase score breakdown showing direction_score and risk_score components.

    Formula:
        phase_score = 0.5 * direction_score + 0.5 * risk_score
        risk_score = (1-α) * mse_score + α * corr_score  (HybridLoss)

    Args:
        csv_path: Path to valid_metrics.csv or train_metrics.csv
        output_path: Optional path to save figure
        title: Figure title
        alpha: HybridLoss correlation weight (default 0.7)
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ['epoch', 'risk_mse', 'risk_correlation', 'direction_f1_macro']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[WARN] Missing columns for phase score plot: {missing}")
        return None

    epochs = df['epoch']

    # Compute scores using HybridLoss formula
    MAX_MSE = 1.0
    mse_norm = np.clip(df['risk_mse'] / MAX_MSE, 0, 1)
    mse_score = 1.0 - mse_norm
    corr_score = np.clip(df['risk_correlation'], 0, 1)  # Clamp negative to 0

    # HybridLoss risk_score
    risk_score = (1.0 - alpha) * mse_score + alpha * corr_score
    direction_score = df['direction_f1_macro']

    # Phase score (equal weights)
    phase_score = 0.5 * direction_score + 0.5 * risk_score

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title} (α={alpha}: {int((1-alpha)*100)}%MSE + {int(alpha*100)}%Corr)",
                 fontsize=14, fontweight='bold')

    # Plot 1: Phase Score Progression
    ax1 = axes[0, 0]
    ax1.plot(epochs, phase_score, 'o-', color='purple', linewidth=2, markersize=8, label='Phase Score')

    # Best epoch
    best_idx = phase_score.idxmax()
    best_epoch = df.loc[best_idx, 'epoch']
    best_score = phase_score[best_idx]
    ax1.scatter(best_epoch, best_score, s=200, c='gold', marker='*',
                edgecolor='black', zorder=5, label=f'Best: {best_score:.4f}')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Phase Score')
    ax1.set_title('1. MACRO Phase Score (Higher = Better)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Component Breakdown (Stacked Area)
    ax2 = axes[0, 1]
    dir_contrib = 0.5 * direction_score
    risk_contrib = 0.5 * risk_score

    ax2.stackplot(epochs, dir_contrib, risk_contrib,
                  labels=['Direction (50%)', 'Risk (50%)'],
                  colors=['#4CAF50', '#2196F3'], alpha=0.7)
    ax2.axvline(best_epoch, color='red', linestyle='--', alpha=0.8, label='Best')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Contribution')
    ax2.set_title('2. Phase Score Components')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Risk Score Breakdown (MSE vs Corr)
    ax3 = axes[1, 0]
    mse_contrib = (1.0 - alpha) * mse_score
    corr_contrib = alpha * corr_score

    ax3.stackplot(epochs, mse_contrib, corr_contrib,
                  labels=[f'MSE Score ({int((1-alpha)*100)}%)', f'Corr Score ({int(alpha*100)}%)'],
                  colors=['#F44336', '#2196F3'], alpha=0.7)
    ax3.plot(epochs, risk_score, 'k--', linewidth=1.5, label='Total Risk Score')
    ax3.axvline(best_epoch, color='gold', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Risk Score')
    ax3.set_title('3. Risk Score = (1-α)×MSE_score + α×Corr_score')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary
    ax4 = axes[1, 1]
    final_row = df.iloc[-1]
    final_phase = phase_score.iloc[-1]
    final_risk = risk_score.iloc[-1]
    final_dir = direction_score.iloc[-1]

    summary_text = f"""MACRO Phase Score Summary (Epoch {int(final_row['epoch'])})

Phase Score:     {final_phase:.4f}  (Target > 0.5)
├─ Direction:    {final_dir:.4f}  × 0.5 = {0.5*final_dir:.4f}
└─ Risk:         {final_risk:.4f}  × 0.5 = {0.5*final_risk:.4f}

Risk Score Breakdown (α={alpha}):
├─ MSE Score:    {mse_score.iloc[-1]:.4f}  × {1-alpha:.1f} = {(1-alpha)*mse_score.iloc[-1]:.4f}
└─ Corr Score:   {corr_score.iloc[-1]:.4f}  × {alpha:.1f} = {alpha*corr_score.iloc[-1]:.4f}

Best Epoch: {int(best_epoch)} (Score: {best_score:.4f})

Formula:
  risk_score = {int((1-alpha)*100)}%×(1-MSE) + {int(alpha*100)}%×max(0,ρ)
  phase_score = 50%×dir_f1 + 50%×risk_score
"""

    ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.axis('off')
    ax4.set_title('4. Score Summary')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved MACRO phase score to {output_path}")

    return fig


def plot_turnover_sharpe_tradeoff(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "Turnover vs Sharpe Trade-off"
) -> plt.Figure:
    """
    Analyze the trade-off between portfolio turnover and Sharpe ratio.

    Args:
        csv_path: Path to valid_metrics.csv
        output_path: Optional path to save figure
        title: Figure title
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ['epoch', 'topk_turnover', 'topk_sharpe_ratio']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    epochs = df['epoch']
    turnover = df['topk_turnover']
    sharpe = df['topk_sharpe_ratio']

    # Plot 1: Scatter plot with color by epoch
    ax1 = axes[0]
    scatter = ax1.scatter(turnover, sharpe, c=epochs, cmap='viridis',
                          s=150, edgecolors='black', linewidths=1, alpha=0.8)

    # Add trend line
    if len(turnover) > 1 and turnover.notna().all() and np.isfinite(turnover).all() and sharpe.notna().all() and np.isfinite(sharpe).all():
        try:
            z = np.polyfit(turnover, sharpe, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(turnover.min(), turnover.max(), 100)
            ax1.plot(x_trend, p(x_trend), '--', color='red', alpha=0.7,
                    label=f'Trend (slope: {z[0]:.2f})')
        except Exception:
            pass

    # Highlight best CES point
    if 'ces_score' in df.columns:
        best_idx = df['ces_score'].idxmax()
        ax1.scatter(turnover.iloc[best_idx], sharpe.iloc[best_idx],
                   s=300, c='gold', marker='*', edgecolor='black', linewidth=2,
                   zorder=5, label='Best CES')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Epoch')

    ax1.set_xlabel('Turnover (Higher = More Trading)')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Trade-off Scatter Plot')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dual-axis line plot
    ax2 = axes[1]

    color1 = '#2196F3'
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Turnover', color=color1)
    line1 = ax2.plot(epochs, turnover, 'o-', color=color1, linewidth=2, markersize=6, label='Turnover')
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.fill_between(epochs, turnover, alpha=0.2, color=color1)

    ax2_twin = ax2.twinx()
    color2 = '#4CAF50'
    ax2_twin.set_ylabel('Sharpe Ratio', color=color2)
    line2 = ax2_twin.plot(epochs, sharpe, 's-', color=color2, linewidth=2, markersize=6, label='Sharpe')
    ax2_twin.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=9)

    ax2.set_title('Turnover & Sharpe Over Epochs')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved turnover-sharpe trade-off to {output_path}")

    return fig


def create_all_charts(
    valid_csv: str,
    iterations_summary: Optional[List[Dict]] = None,
    trajectory_csv: Optional[str] = None,
    output_dir: str = "./charts",
    training_mode: Optional[str] = None
):
    """
    Generate all visualization charts for MAFIA Observer analysis.

    Args:
        valid_csv: Path to valid_metrics.csv
        iterations_summary: Optional list of iteration summaries for walk-forward charts
        trajectory_csv: Optional path to trajectory_details.csv for trigger analysis
        output_dir: Directory to save charts
        training_mode: Training mode (MACRO_ONLY or SELECTION_ONLY) for display
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Determine phase and label
    is_macro = training_mode == "MACRO_ONLY"
    is_selection = training_mode == "SELECTION_ONLY"

    phase_label = ""
    if is_macro:
        phase_label = " [Phase 1: MACRO]"
    elif is_selection:
        phase_label = " [Phase 2: SELECTION]"

    print(f"\n{'='*70}")
    print(f"GENERATING MAFIA OBSERVER INSIGHT CHARTS{phase_label}")
    print(f"{'='*70}\n")

    chart_num = 0

    # Chart: Loss Components (always generate)
    chart_num += 1
    print(f"[{chart_num}] Loss Components...")
    plot_loss_components(
        valid_csv,
        output_path=os.path.join(output_dir, "loss_components.png")
    )
    plt.close()

    # SELECTION_ONLY charts
    if is_selection:
        chart_num += 1
        print(f"[{chart_num}] Validation Metrics Grid...")
        plot_validation_metrics_grid(
            valid_csv,
            output_path=os.path.join(output_dir, "validation_metrics.png")
        )
        plt.close()

        chart_num += 1
        print(f"[{chart_num}] CES Components Breakdown...")
        plot_ces_components_breakdown(
            valid_csv,
            output_path=os.path.join(output_dir, "ces_breakdown.png")
        )
        plt.close()

        chart_num += 1
        print(f"[{chart_num}] Turnover-Sharpe Trade-off...")
        plot_turnover_sharpe_tradeoff(
            valid_csv,
            output_path=os.path.join(output_dir, "turnover_tradeoff.png")
        )
        plt.close()

    # MACRO_ONLY charts
    if is_macro:
        chart_num += 1
        print(f"[{chart_num}] MACRO Phase Score Breakdown...")
        plot_macro_phase_score(
            valid_csv,
            output_path=os.path.join(output_dir, "macro_phase_score.png")
        )
        plt.close()

        chart_num += 1
        print(f"[{chart_num}] Direction Classification Breakdown...")
        plot_direction_breakdown(
            valid_csv,
            output_path=os.path.join(output_dir, "direction_breakdown.png")
        )
        plt.close()

        chart_num += 1
        print(f"[{chart_num}] Risk Calibration Analysis...")
        plot_risk_calibration(
            valid_csv,
            output_path=os.path.join(output_dir, "risk_calibration.png")
        )
        plt.close()

    # Walk-forward charts (SELECTION_ONLY - uses CES metrics)
    if iterations_summary and is_selection:
        chart_num += 1
        print(f"[{chart_num}] Sharpe Progression...")
        plot_ces_progression(
            iterations_summary,
            output_path=os.path.join(output_dir, "sharpe_progression.png")
        )
        plt.close()

    # Trigger Distribution (if trajectory data provided)
    if trajectory_csv and os.path.exists(trajectory_csv):
        chart_num += 1
        print(f"[{chart_num}] Trigger Distribution...")
        plot_trigger_distribution(
            trajectory_csv,
            output_path=os.path.join(output_dir, "trigger_distribution.png")
        )
        plt.close()

    print(f"[DONE] Generated {chart_num} charts")

    print(f"\n{'='*70}")
    print(f"MAFIA OBSERVER INSIGHT CHARTS GENERATED")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")



def plot_cockpit_dashboard(
    csv_path: str,
    output_path: Optional[str] = None,
    title: str = "MAFIA Training Cockpit"
) -> plt.Figure:
    """
    Generate a high-density 'Cockpit' dashboard (3x2 grid) for monitoring training health.
    
    Panels:
    1. CES Score (North Star Metric)
    2. Component Breakdown (Sharpe, F1, Risk) - Stacked Area or Bar
    3. Direction Intelligence (F1 per class Heatmap)
    4. Risk Calibration (MSE & Correlation dual-axis)
    5. Efficiency (Turnover vs Sharpe Tradeoff)
    6. Training Stability (Loss Components)
    
    Args:
        csv_path: Path to valid_metrics.csv
        output_path: Optional path to save figure
    """
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    best_idx = df['ces_score'].idxmax()
    best_epoch = df.loc[best_idx, 'epoch']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
    fig.suptitle(f"{title} (Best Epoch: {int(best_epoch)}, CES: {df.loc[best_idx, 'ces_score']:.3f})", 
                 fontsize=16, fontweight='bold', y=0.95)

    # --- Panel 1: CES Score Progression (The North Star) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, df['ces_score'], 'o-', color='purple', linewidth=2, label='CES Score')
    ax1.scatter(best_epoch, df.loc[best_idx, 'ces_score'], s=200, c='gold', marker='*', 
               edgecolor='black', zorder=5, label='Best Model')
    # Trend line
    if len(epochs) > 1 and df['ces_score'].notna().all() and np.isfinite(df['ces_score']).all():
        try:
            z = np.polyfit(epochs, df['ces_score'], 1)
            p = np.poly1d(z)
            ax1.plot(epochs, p(epochs), '--', color='gray', alpha=0.5, label=f'Trend (slope={z[0]:.4f})')
        except Exception:
            pass
    ax1.set_ylabel('CES Score')
    ax1.set_title('1. Overall Performance (CES)', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Component Breakdown (Stacked) ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Spec weights: 0.6 Sharpe, 0.2 F1, 0.2 (1-Risk)
    w_sharpe = 0.6 * df['ces_rank_sharpe']
    w_f1 = 0.2 * df['ces_rank_dir_f1']
    w_risk = 0.2 * (1 - df['ces_rank_risk_mse'])
    
    ax2.stackplot(epochs, w_sharpe, w_f1, w_risk, 
                 labels=['Sharpe (60%)', 'Dir F1 (20%)', 'Risk (20%)'],
                 colors=['#2196F3', '#4CAF50', '#FF9800'], alpha=0.7)
    ax2.axvline(best_epoch, color='red', linestyle='--', alpha=0.8)
    ax2.set_ylabel('Weighted Contribution')
    ax2.set_title('2. CES Component Contribution', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Direction Intelligence (Heatmap) ---
    ax3 = fig.add_subplot(gs[1, 0])
    cols = ['direction_f1_bear', 'direction_f1_side', 'direction_f1_bull']
    # Check if cols exist (some validation sets might miss classes like Bear in 2017)
    valid_cols = [c for c in cols if c in df.columns]
    
    if valid_cols:
        data = df[valid_cols].values.T
        im = ax3.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax3.set_yticks(range(len(valid_cols)))
        ax3.set_yticklabels([c.replace('direction_f1_', '').title() for c in valid_cols])
        # Annotate
        for i in range(len(valid_cols)):
            for j in range(len(epochs)):
                val = data[i, j]
                color = 'white' if val < 0.5 else 'black'
                # Only label every Nth epoch if too many
                if len(epochs) < 20 or j % (len(epochs)//10) == 0:
                     ax3.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=7)
    
        ax3.set_title('3. Direction F1 by Regime (Heatmap)', fontweight='bold')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, "No Direction Data", ha='center', va='center')

    # --- Panel 4: Risk Calibration (Dual Axis) ---
    ax4 = fig.add_subplot(gs[1, 1])
    color_mse = '#F44336'
    color_corr = '#2196F3'
    
    line1 = ax4.plot(epochs, df['risk_mse'], 'o-', color=color_mse, label='MSE (Lower=Better)')
    ax4.set_ylabel('MSE', color=color_mse)
    ax4.tick_params(axis='y', labelcolor=color_mse)
    ax4.invert_yaxis() # MSE lower is better, so visually Up is Good
    
    ax4_twin = ax4.twinx()
    line2 = ax4_twin.plot(epochs, df['risk_correlation'], 's--', color=color_corr, label='Correlation (Higher=Better)')
    ax4_twin.set_ylabel('Correlation', color=color_corr)
    ax4_twin.tick_params(axis='y', labelcolor=color_corr)
    ax4_twin.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper center', fontsize=8)
    ax4.set_title('4. Risk Calibration (HybridLoss: MSE+Corr)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    # Annotation for HybridLoss balance
    ax4.text(0.02, 0.98, 'Loss = 30%MSE + 70%Corr', transform=ax4.transAxes,
             fontsize=7, verticalalignment='top', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 5: Efficiency (Turnover vs Sharpe) ---
    ax5 = fig.add_subplot(gs[2, 0])
    # Scatter trace
    sc = ax5.scatter(df['topk_turnover'], df['topk_sharpe_ratio'], c=epochs, cmap='viridis', s=100, edgecolors='k')
    # Best epoch marker
    ax5.scatter(df.loc[best_idx, 'topk_turnover'], df.loc[best_idx, 'topk_sharpe_ratio'], 
               s=250, c='gold', marker='*', edgecolors='k', label='Best Model')
    
    ax5.set_xlabel('Turnover (Avg Daily)')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.set_title('5. Efficiency Frontier (Turnover vs Sharpe)', fontweight='bold')
    plt.colorbar(sc, ax=ax5, label='Epoch')
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Training Stability (Losses) ---
    ax6 = fig.add_subplot(gs[2, 1])
    # Normalize losses to start at 1.0 for comparison? Or just log scale?
    # Let's use simple plot but with secondary axis for Total vs Components
    l1 = ax6.plot(epochs, df['loss_total'], 'k-', linewidth=2, label='Total Loss')
    l2 = ax6.plot(epochs, df['loss_pg'], '--', label='Selection (PG)')
    l3 = ax6.plot(epochs, df['loss_dir'], ':', label='Direction')
    l4 = ax6.plot(epochs, df['loss_risk'], '-.', label='Risk')
    
    ax6.set_ylabel('Loss Value')
    ax6.set_title('6. Training Convergence', fontweight='bold')
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel('Epoch')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[PLOT] Saved Cockpit Dashboard to {output_path}")

    return fig

if __name__ == "__main__":
    # Demo usage
    print("Observer Plotting Utilities")
    print("Import this module to use plotting functions")
    print("\nExample:")
    print('  from RL_controller.plotting_utils import create_all_charts')
    print('  create_all_charts("./iter_0_valid_2017/valid_metrics.csv")')
