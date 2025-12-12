#!/usr/bin/env python3
"""
Data Quality Analysis & Cleaning Script for MAFIA Stock Data

This script:
1. Loads stock price data
2. Calculates returns and identifies extreme values
3. Shows detailed statistics about bad datapoints
4. Optionally cleans the data by removing/fixing outliers
5. Exports cleaned data

Usage:
    python scripts/analyze_data_quality.py --input data/stock_data_top23.csv --threshold 1.0
    python scripts/analyze_data_quality.py --input data/stock_data_top23.csv --clean --output data/stock_data_top23_cleaned.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_returns(df: pd.DataFrame, date_col='date', stock_col='stock', price_col='close'):
    """Calculate daily returns for each stock."""
    df = df.sort_values([stock_col, date_col])
    df['return'] = df.groupby(stock_col)[price_col].pct_change()
    return df


def find_extreme_returns(df: pd.DataFrame, threshold=1.0, return_col='return'):
    """
    Find extreme returns that exceed threshold.
    
    Args:
        df: DataFrame with returns column
        threshold: Absolute return threshold (1.0 = 100% change)
        return_col: Name of return column
    
    Returns:
        DataFrame with extreme returns only
    """
    extreme_mask = df[return_col].abs() > threshold
    return df[extreme_mask].copy()


def analyze_data_quality(input_file: str, threshold: float = 1.0):
    """
    Comprehensive data quality analysis.
    
    Args:
        input_file: Path to CSV file
        threshold: Return threshold for detecting bad data
    """
    print(f"\n{'='*80}")
    print(f"📊 DATA QUALITY ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Input file: {input_file}")
    print(f"Threshold:  {threshold:.1%} (returns > this are considered extreme)\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"  ✓ Loaded {len(df):,} rows, {df['stock'].nunique()} stocks\n")
    
    # Basic info
    print("📋 Dataset Overview:")
    print(f"  Date range:   {df['date'].min()} to {df['date'].max()}")
    print(f"  Stocks:       {df['stock'].nunique()}")
    print(f"  Total rows:   {len(df):,}")
    print(f"  Columns:      {', '.join(df.columns)}\n")
    
    # Calculate returns
    print("Calculating returns...")
    df = calculate_returns(df)
    valid_returns = df['return'].dropna()
    print(f"  ✓ Calculated {len(valid_returns):,} returns\n")
    
    # Overall statistics
    print("📈 Return Statistics (All Data):")
    print(f"  Mean:         {valid_returns.mean():.4%}")
    print(f"  Median:       {valid_returns.median():.4%}")
    print(f"  Std Dev:      {valid_returns.std():.4%}")
    print(f"  Min:          {valid_returns.min():.4%}")
    print(f"  Max:          {valid_returns.max():.4%}")
    print(f"  Percentiles:")
    print(f"    1%:         {valid_returns.quantile(0.01):.4%}")
    print(f"    5%:         {valid_returns.quantile(0.05):.4%}")
    print(f"    95%:        {valid_returns.quantile(0.95):.4%}")
    print(f"    99%:        {valid_returns.quantile(0.99):.4%}\n")
    
    # Find extreme returns
    print(f"🔍 Detecting Extreme Returns (|return| > {threshold:.0%})...")
    extreme_df = find_extreme_returns(df, threshold=threshold)
    
    if len(extreme_df) == 0:
        print(f"  ✓ No extreme returns found! Data quality looks good.\n")
        return df, None
    
    # Extreme returns analysis
    print(f"  🚨 Found {len(extreme_df):,} extreme returns ({len(extreme_df)/len(valid_returns)*100:.3f}%)\n")
    
    print("📊 Extreme Returns Statistics:")
    print(f"  Count:        {len(extreme_df):,}")
    print(f"  Min:          {extreme_df['return'].min():.4f} ({extreme_df['return'].min()*100:.0f}%)")
    print(f"  Max:          {extreme_df['return'].max():.4f} ({extreme_df['return'].max()*100:.0f}%)")
    print(f"  Mean (abs):   {extreme_df['return'].abs().mean():.4f}\n")
    
    # Group by stock
    print("📌 Stocks with Extreme Returns (Top 20):")
    extreme_by_stock = extreme_df.groupby('stock').agg({
        'return': ['count', 'min', 'max', lambda x: x.abs().max()]
    }).round(2)
    extreme_by_stock.columns = ['count', 'min_return', 'max_return', 'abs_max_return']
    extreme_by_stock = extreme_by_stock.sort_values('abs_max_return', ascending=False)
    
    for i, (stock, row) in enumerate(extreme_by_stock.head(20).iterrows()):
        print(f"  {i+1:2d}. {stock:10s}  "
              f"count={int(row['count']):3d}  "
              f"range=[{row['min_return']:8.2f}, {row['max_return']:8.2f}]  "
              f"max_abs={row['abs_max_return']:10.2f}")
    
    if len(extreme_by_stock) > 20:
        print(f"  ... and {len(extreme_by_stock) - 20} more stocks\n")
    else:
        print()
    
    # Detailed listing of worst offenders
    print("🔥 Top 10 Most Extreme Returns (Sorted by Absolute Value):")
    top_extreme = extreme_df.nlargest(10, 'return', keep='all')[
        ['date', 'stock', 'open', 'close', 'high', 'low', 'return']
    ].copy()
    
    # Calculate price change for context
    top_extreme['price_change'] = top_extreme['close'] - top_extreme['open']
    
    print()
    for i, row in top_extreme.iterrows():
        print(f"  {row['date']}  {row['stock']:10s}  "
              f"return={row['return']:12.2f} ({row['return']*100:,.0f}%)  "
              f"O={row['open']:8.1f} C={row['close']:8.1f} "
              f"H={row['high']:8.1f} L={row['low']:8.1f}")
    
    print()
    
    # Diagnosis
    print("🔬 Likely Causes:")
    
    # Check for zero/near-zero prices
    zero_prices = df[(df['close'] < 1e-6) | (df['open'] < 1e-6)]
    if len(zero_prices) > 0:
        print(f"  ⚠️  Found {len(zero_prices)} rows with near-zero prices (likely data error)")
    
    # Check for massive price jumps
    massive_jumps = extreme_df[extreme_df['return'].abs() > 10]
    if len(massive_jumps) > 0:
        print(f"  ⚠️  Found {len(massive_jumps)} returns > 1000% (likely stock splits or bad data)")
    
    # Check for suspicious patterns
    negative_extreme = extreme_df[extreme_df['return'] < -0.9]
    if len(negative_extreme) > 0:
        print(f"  ⚠️  Found {len(negative_extreme)} returns < -90% (possible delisting/corporate action)")
    
    print()
    print("💡 Recommendations:")
    print("  1. Review the listed dates/stocks manually")
    print("  2. Check for stock splits that weren't adjusted")
    print("  3. Verify prices are in correct units (VND vs 1000 VND)")
    print("  4. Remove or interpolate bad datapoints")
    print("  5. Use --clean flag to auto-clean data\n")
    
    return df, extreme_df


def clean_data(df: pd.DataFrame, extreme_df: pd.DataFrame, method='clip', threshold=1.0):
    """
    Clean data by removing or fixing extreme returns.
    
    Args:
        df: Full DataFrame
        extreme_df: DataFrame with extreme returns
        method: 'clip', 'remove', or 'interpolate'
        threshold: Return threshold
    
    Returns:
        Cleaned DataFrame
    """
    print(f"\n{'='*80}")
    print(f"🧹 DATA CLEANING")
    print(f"{'='*80}\n")
    print(f"Method:    {method}")
    print(f"Threshold: {threshold:.1%}\n")
    
    df_clean = df.copy()
    
    if method == 'clip':
        # Clip extreme returns by adjusting prices
        print("Clipping extreme returns to [-0.5, 1.0] range...")
        
        # For each extreme return, adjust the current price
        # to make return = threshold (or -threshold)
        for idx in extreme_df.index:
            stock = df_clean.loc[idx, 'stock']
            date = df_clean.loc[idx, 'date']
            
            # Get previous close
            stock_df = df_clean[df_clean['stock'] == stock].sort_values('date')
            date_idx = stock_df[stock_df['date'] == date].index[0]
            prev_idx = stock_df.index[stock_df.index < date_idx].max()
            
            if pd.notna(prev_idx):
                prev_close = df_clean.loc[prev_idx, 'close']
                current_return = df_clean.loc[idx, 'return']
                
                # Clip return
                clipped_return = np.clip(current_return, -0.5, 1.0)
                
                # Recalculate close price
                new_close = prev_close * (1 + clipped_return)
                df_clean.loc[idx, 'close'] = new_close
                
                # Also adjust OHLC to be consistent
                df_clean.loc[idx, 'high'] = max(new_close, df_clean.loc[idx, 'high'])
                df_clean.loc[idx, 'low'] = min(new_close, df_clean.loc[idx, 'low'])
        
        # Recalculate returns
        df_clean = calculate_returns(df_clean)
        print(f"  ✓ Clipped {len(extreme_df)} extreme returns\n")
    
    elif method == 'remove':
        # Remove rows with extreme returns
        print("Removing rows with extreme returns...")
        df_clean = df_clean.drop(extreme_df.index)
        df_clean = calculate_returns(df_clean)
        print(f"  ✓ Removed {len(extreme_df)} rows\n")
    
    elif method == 'interpolate':
        # Replace extreme values with interpolated prices
        print("Interpolating extreme returns...")
        for stock in extreme_df['stock'].unique():
            stock_mask = df_clean['stock'] == stock
            stock_extreme_mask = stock_mask & df_clean.index.isin(extreme_df.index)
            
            # Set extreme prices to NaN
            df_clean.loc[stock_extreme_mask, ['open', 'close', 'high', 'low']] = np.nan
            
            # Interpolate
            df_clean.loc[stock_mask, ['open', 'close', 'high', 'low']] = (
                df_clean.loc[stock_mask, ['open', 'close', 'high', 'low']].interpolate(method='linear')
            )
        
        df_clean = calculate_returns(df_clean)
        print(f"  ✓ Interpolated {len(extreme_df)} extreme values\n")
    
    # Verify cleaning
    print("Verifying cleaned data...")
    df_clean_check = calculate_returns(df_clean)
    remaining_extreme = find_extreme_returns(df_clean_check, threshold=threshold)
    
    if len(remaining_extreme) == 0:
        print(f"  ✓ SUCCESS: No extreme returns remaining!\n")
    else:
        print(f"  ⚠️  WARNING: {len(remaining_extreme)} extreme returns still present\n")
        print("  Top remaining extremes:")
        for i, row in remaining_extreme.nlargest(5, 'return', keep='all').iterrows():
            print(f"    {row['date']}  {row['stock']:10s}  return={row['return']:.2f}")
        print()
    
    return df_clean


def main():
    parser = argparse.ArgumentParser(description='Analyze and clean stock data quality')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--threshold', type=float, default=1.0, help='Extreme return threshold (default: 1.0 = 100%%)')
    parser.add_argument('--clean', action='store_true', help='Clean the data')
    parser.add_argument('--method', type=str, default='clip', choices=['clip', 'remove', 'interpolate'],
                        help='Cleaning method (default: clip)')
    parser.add_argument('--output', type=str, help='Output file for cleaned data (default: input_cleaned.csv)')
    
    args = parser.parse_args()
    
    # Analyze
    df, extreme_df = analyze_data_quality(args.input, threshold=args.threshold)
    
    # Clean if requested
    if args.clean and extreme_df is not None and len(extreme_df) > 0:
        df_clean = clean_data(df, extreme_df, method=args.method, threshold=args.threshold)
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            input_path = Path(args.input)
            output_file = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
        
        # Save
        print(f"💾 Saving cleaned data to: {output_file}")
        # Drop the 'return' column before saving (it's just for analysis)
        df_clean_save = df_clean.drop(columns=['return'], errors='ignore')
        df_clean_save.to_csv(output_file, index=False)
        print(f"  ✓ Saved {len(df_clean_save):,} rows\n")
        
        print(f"\n{'='*80}")
        print(f"✅ DATA CLEANING COMPLETE!")
        print(f"{'='*80}\n")
        print(f"Summary:")
        print(f"  Original rows:     {len(df):,}")
        print(f"  Extreme returns:   {len(extreme_df):,}")
        print(f"  Cleaned rows:      {len(df_clean_save):,}")
        print(f"  Output file:       {output_file}\n")
    
    elif args.clean and (extreme_df is None or len(extreme_df) == 0):
        print("✓ No cleaning needed - data quality is good!\n")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
