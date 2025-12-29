#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Data Cleaning Script for MAFIA Observer Training

This script cleans source CSV data to ensure high-quality training:
1. Fix zero/negative prices with forward fill
2. Detect and handle stock splits (extreme returns)
3. Fill missing dates with forward fill
4. Validate OHLC constraints
5. Generate quality report

Usage:
    python scripts/clean_stock_data.py --input data/stock_data_top23.csv --output data/stock_data_top23_cleaned.csv
    python scripts/clean_stock_data.py --input data/stock_data_top23.csv --analyze-only
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    """Report of data cleaning actions and statistics."""
    input_file: str
    output_file: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Input statistics
    total_rows: int = 0
    total_stocks: int = 0
    date_range: Tuple[str, str] = ("", "")

    # Issues found
    zero_price_rows: int = 0
    negative_price_rows: int = 0
    nan_price_rows: int = 0
    inf_return_count: int = 0
    extreme_return_count: int = 0
    ohlc_violation_count: int = 0

    # Missing data
    missing_date_stocks: Dict[str, float] = field(default_factory=dict)  # stock -> missing %
    incomplete_stocks: List[str] = field(default_factory=list)  # stocks with >20% missing

    # Cleaning actions
    actions: List[str] = field(default_factory=list)
    rows_modified: int = 0
    rows_dropped: int = 0
    stocks_dropped: List[str] = field(default_factory=list)

    # Output statistics
    output_rows: int = 0
    output_nan_ratio: float = 0.0
    output_inf_count: int = 0

    def is_acceptable(self, max_nan_ratio: float = 0.01) -> bool:
        """Check if cleaned data meets quality threshold."""
        return self.output_nan_ratio < max_nan_ratio and self.output_inf_count == 0

    def to_dict(self) -> dict:
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "timestamp": self.timestamp,
            "input_stats": {
                "total_rows": int(self.total_rows),
                "total_stocks": int(self.total_stocks),
                "date_range": self.date_range,
            },
            "issues_found": {
                "zero_price_rows": int(self.zero_price_rows),
                "negative_price_rows": int(self.negative_price_rows),
                "nan_price_rows": int(self.nan_price_rows),
                "inf_return_count": int(self.inf_return_count),
                "extreme_return_count": int(self.extreme_return_count),
                "ohlc_violation_count": int(self.ohlc_violation_count),
            },
            "missing_data": {
                "stocks_with_missing": len(self.missing_date_stocks),
                "incomplete_stocks": self.incomplete_stocks,
            },
            "cleaning_actions": {
                "actions": self.actions,
                "rows_modified": int(self.rows_modified),
                "rows_dropped": int(self.rows_dropped),
                "stocks_dropped": self.stocks_dropped,
            },
            "output_stats": {
                "output_rows": int(self.output_rows),
                "output_nan_ratio": float(self.output_nan_ratio),
                "output_inf_count": int(self.output_inf_count),
                "is_acceptable": self.is_acceptable(),
            },
        }

    def save_to_file(self, path: str):
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Report saved to: {path}")

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("DATA CLEANING REPORT")
        print("=" * 70)

        print(f"\nInput: {self.input_file}")
        print(f"Output: {self.output_file or 'N/A (analyze only)'}")
        print(f"Timestamp: {self.timestamp}")

        print(f"\n--- Input Statistics ---")
        print(f"Total rows: {self.total_rows:,}")
        print(f"Total stocks: {self.total_stocks}")
        print(f"Date range: {self.date_range[0]} to {self.date_range[1]}")

        print(f"\n--- Issues Found ---")
        print(f"Zero price rows: {self.zero_price_rows}")
        print(f"Negative price rows: {self.negative_price_rows}")
        print(f"NaN price rows: {self.nan_price_rows}")
        print(f"Inf returns: {self.inf_return_count}")
        print(f"Extreme returns (>50%): {self.extreme_return_count}")
        print(f"OHLC violations: {self.ohlc_violation_count}")

        print(f"\n--- Missing Data ---")
        print(f"Stocks with missing dates: {len(self.missing_date_stocks)}")
        if self.incomplete_stocks:
            print(f"Incomplete stocks (>20% missing): {self.incomplete_stocks}")

        print(f"\n--- Cleaning Actions ---")
        for action in self.actions:
            print(f"  - {action}")
        print(f"Rows modified: {self.rows_modified}")
        print(f"Rows dropped: {self.rows_dropped}")
        if self.stocks_dropped:
            print(f"Stocks dropped: {self.stocks_dropped}")

        print(f"\n--- Output Statistics ---")
        print(f"Output rows: {self.output_rows:,}")
        print(f"NaN ratio: {self.output_nan_ratio:.4%}")
        print(f"Inf count: {self.output_inf_count}")

        status = "✅ ACCEPTABLE" if self.is_acceptable() else "❌ NEEDS ATTENTION"
        print(f"\nStatus: {status}")
        print("=" * 70)


def analyze_data(df: pd.DataFrame, report: CleaningReport) -> None:
    """Analyze data and populate report with issues found."""

    # Basic stats
    report.total_rows = len(df)
    report.total_stocks = df['stock'].nunique()
    report.date_range = (str(df['date'].min())[:10], str(df['date'].max())[:10])

    price_cols = ['open', 'high', 'low', 'close']

    # Zero prices
    for col in price_cols:
        report.zero_price_rows += (df[col] == 0).sum()

    # Negative prices
    for col in price_cols:
        report.negative_price_rows += (df[col] < 0).sum()

    # NaN prices
    for col in price_cols:
        report.nan_price_rows += df[col].isna().sum()

    # Compute returns and check for Inf/extreme
    df_sorted = df.sort_values(['stock', 'date']).copy()
    df_sorted['return'] = df_sorted.groupby('stock')['close'].pct_change()

    report.inf_return_count = np.isinf(df_sorted['return']).sum()
    report.extreme_return_count = ((df_sorted['return'].abs() > 0.5) &
                                    ~np.isinf(df_sorted['return']) &
                                    ~np.isnan(df_sorted['return'])).sum()

    # OHLC violations
    ohlc_violations = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    report.ohlc_violation_count = ohlc_violations.sum()

    # Missing dates analysis
    all_dates = pd.to_datetime(df['date']).dt.date.unique()
    total_dates = len(all_dates)

    for stock in df['stock'].unique():
        stock_dates = pd.to_datetime(df[df['stock'] == stock]['date']).dt.date.unique()
        missing_ratio = 1 - len(stock_dates) / total_dates
        if missing_ratio > 0.01:  # More than 1% missing
            report.missing_date_stocks[stock] = round(missing_ratio * 100, 2)
        if missing_ratio > 0.2:  # More than 20% missing
            report.incomplete_stocks.append(stock)


def fix_zero_negative_prices(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Fix zero and negative prices with forward fill."""
    df = df.copy()
    price_cols = ['open', 'high', 'low', 'close']

    total_fixed = 0
    for col in price_cols:
        # Mark invalid prices as NaN
        invalid_mask = (df[col] <= 0) | df[col].isna()
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            df.loc[invalid_mask, col] = np.nan
            total_fixed += invalid_count

    if total_fixed > 0:
        # Forward fill within each stock
        df = df.sort_values(['stock', 'date'])
        for col in price_cols:
            df[col] = df.groupby('stock')[col].ffill()
            df[col] = df.groupby('stock')[col].bfill()  # Backward fill for start

        report.actions.append(f"Fixed {total_fixed} zero/negative/NaN prices with forward fill")
        report.rows_modified += total_fixed

    return df


def fix_extreme_returns(df: pd.DataFrame, report: CleaningReport,
                        threshold: float = 0.5, clip_min: float = -0.5,
                        clip_max: float = 1.0) -> pd.DataFrame:
    """Detect and clip extreme returns by adjusting prices."""
    df = df.copy()
    df = df.sort_values(['stock', 'date']).reset_index(drop=True)

    total_adjusted = 0

    for stock in df['stock'].unique():
        stock_mask = df['stock'] == stock
        stock_idx = df.index[stock_mask].tolist()

        for i in range(1, len(stock_idx)):
            prev_idx = stock_idx[i - 1]
            curr_idx = stock_idx[i]

            prev_close = df.loc[prev_idx, 'close']
            curr_close = df.loc[curr_idx, 'close']

            if prev_close > 0:
                ret = (curr_close - prev_close) / prev_close

                if abs(ret) > threshold:
                    # Clip the return
                    clipped_ret = np.clip(ret, clip_min, clip_max)
                    new_close = prev_close * (1 + clipped_ret)

                    # Adjust all OHLC proportionally
                    if curr_close > 0:
                        ratio = new_close / curr_close
                        df.loc[curr_idx, 'open'] *= ratio
                        df.loc[curr_idx, 'high'] *= ratio
                        df.loc[curr_idx, 'low'] *= ratio
                        df.loc[curr_idx, 'close'] = new_close
                        total_adjusted += 1

    if total_adjusted > 0:
        report.actions.append(f"Adjusted {total_adjusted} extreme returns (>{threshold*100:.0f}%) to [{clip_min*100:.0f}%, {clip_max*100:.0f}%]")
        report.rows_modified += total_adjusted

    return df


def fix_ohlc_constraints(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Fix OHLC constraint violations."""
    df = df.copy()

    total_fixed = 0

    # High should be >= max(open, close)
    should_be_high = df[['open', 'close']].max(axis=1)
    fix_high = df['high'] < should_be_high
    if fix_high.any():
        df.loc[fix_high, 'high'] = should_be_high[fix_high]
        total_fixed += fix_high.sum()

    # Low should be <= min(open, close)
    should_be_low = df[['open', 'close']].min(axis=1)
    fix_low = df['low'] > should_be_low
    if fix_low.any():
        df.loc[fix_low, 'low'] = should_be_low[fix_low]
        total_fixed += fix_low.sum()

    if total_fixed > 0:
        report.actions.append(f"Fixed {total_fixed} OHLC constraint violations")
        report.rows_modified += total_fixed

    return df


def fill_missing_dates(df: pd.DataFrame, report: CleaningReport,
                       drop_incomplete: bool = True,
                       missing_threshold: float = 0.2) -> pd.DataFrame:
    """Fill missing dates with forward fill, optionally drop incomplete stocks."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Get full date range
    all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='B')  # Business days
    all_stocks = df['stock'].unique()

    # Create full index
    full_index = pd.MultiIndex.from_product([all_dates, all_stocks], names=['date', 'stock'])

    # Reindex
    df = df.set_index(['date', 'stock'])
    df = df.reindex(full_index)

    # Count missing before fill
    missing_before = df.isna().any(axis=1).sum()

    # Forward fill within each stock
    df = df.groupby('stock').ffill()
    df = df.groupby('stock').bfill()  # Fill start with backward fill

    df = df.reset_index()

    # Calculate missing ratio per stock
    stocks_to_drop = []
    if drop_incomplete:
        for stock in all_stocks:
            stock_data = df[df['stock'] == stock]
            missing_ratio = stock_data['close'].isna().sum() / len(stock_data)
            if missing_ratio > missing_threshold:
                stocks_to_drop.append(stock)

        if stocks_to_drop:
            df = df[~df['stock'].isin(stocks_to_drop)]
            report.stocks_dropped = stocks_to_drop
            report.actions.append(f"Dropped {len(stocks_to_drop)} stocks with >{missing_threshold*100:.0f}% missing: {stocks_to_drop}")

    filled_count = missing_before - df.isna().any(axis=1).sum()
    if filled_count > 0:
        report.actions.append(f"Forward-filled {filled_count} missing date entries")
        report.rows_modified += filled_count

    return df


def drop_incomplete_stocks(df: pd.DataFrame, report: CleaningReport,
                           missing_threshold: float = 0.2) -> pd.DataFrame:
    """Drop stocks with too many missing dates (without filling)."""
    df = df.copy()

    # Get date coverage per stock
    all_dates = df['date'].nunique()
    stock_date_counts = df.groupby('stock')['date'].nunique()

    stocks_to_drop = []
    for stock, count in stock_date_counts.items():
        missing_ratio = 1 - count / all_dates
        if missing_ratio > missing_threshold:
            stocks_to_drop.append(stock)

    if stocks_to_drop:
        rows_before = len(df)
        df = df[~df['stock'].isin(stocks_to_drop)]
        rows_dropped = rows_before - len(df)

        report.stocks_dropped = stocks_to_drop
        report.rows_dropped = rows_dropped
        report.actions.append(
            f"Dropped {len(stocks_to_drop)} stocks with >{missing_threshold*100:.0f}% missing: {stocks_to_drop}"
        )

    return df


def compute_output_stats(df: pd.DataFrame, report: CleaningReport) -> None:
    """Compute statistics for cleaned output."""
    report.output_rows = len(df)

    # NaN ratio
    total_cells = len(df) * len(['open', 'high', 'low', 'close', 'volume'])
    nan_cells = df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()
    report.output_nan_ratio = nan_cells / total_cells if total_cells > 0 else 0

    # Inf count in returns
    df_sorted = df.sort_values(['stock', 'date']).copy()
    df_sorted['return'] = df_sorted.groupby('stock')['close'].pct_change()
    report.output_inf_count = np.isinf(df_sorted['return']).sum()


def clean_stock_data(input_file: str, output_file: Optional[str] = None,
                     analyze_only: bool = False,
                     drop_incomplete: bool = True,
                     missing_threshold: float = 0.2,
                     extreme_return_threshold: float = 0.5,
                     fill_missing: bool = True) -> CleaningReport:
    """
    Main function to clean stock data.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output cleaned CSV file
        analyze_only: If True, only analyze without cleaning
        drop_incomplete: Drop stocks with >missing_threshold missing data
        missing_threshold: Threshold for incomplete stocks (default 0.2 = 20%)
        extreme_return_threshold: Threshold for extreme returns (default 0.5 = 50%)
        fill_missing: Fill missing dates with forward fill

    Returns:
        CleaningReport with details of issues and actions
    """
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)

    # Initialize report
    report = CleaningReport(input_file=input_file, output_file=output_file)

    # Analyze input data
    print("Analyzing input data...")
    analyze_data(df, report)

    if analyze_only:
        report.output_rows = report.total_rows
        compute_output_stats(df, report)
        report.print_summary()
        return report

    # Apply cleaning steps
    print("\nApplying cleaning steps...")

    # Step 1: Fix zero/negative prices
    print("  1. Fixing zero/negative prices...")
    df = fix_zero_negative_prices(df, report)

    # Step 2: Fix extreme returns
    print("  2. Fixing extreme returns...")
    df = fix_extreme_returns(df, report, threshold=extreme_return_threshold)

    # Step 3: Fix OHLC constraints
    print("  3. Fixing OHLC constraints...")
    df = fix_ohlc_constraints(df, report)

    # Step 4: Handle missing dates
    if fill_missing:
        print("  4. Filling missing dates...")
        df = fill_missing_dates(df, report, drop_incomplete=drop_incomplete,
                               missing_threshold=missing_threshold)
    elif drop_incomplete:
        # Drop incomplete stocks without filling
        print("  4. Dropping incomplete stocks...")
        df = drop_incomplete_stocks(df, report, missing_threshold=missing_threshold)

    # Compute output statistics
    compute_output_stats(df, report)

    # Save cleaned data
    if output_file:
        print(f"\nSaving cleaned data to: {output_file}")
        df.to_csv(output_file, index=False)
        report.actions.append(f"Saved cleaned data to {output_file}")

    report.print_summary()

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Clean stock data for MAFIA Observer Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze data without cleaning
  python scripts/clean_stock_data.py --input data/stock_data_top23.csv --analyze-only

  # Clean and save to new file
  python scripts/clean_stock_data.py --input data/stock_data_top23.csv --output data/stock_data_top23_cleaned.csv

  # Clean with custom thresholds
  python scripts/clean_stock_data.py --input data/stock_data_top23.csv --output data/cleaned.csv --missing-threshold 0.1 --extreme-threshold 0.3

  # Keep incomplete stocks (don't drop)
  python scripts/clean_stock_data.py --input data/stock_data_top23.csv --output data/cleaned.csv --no-drop-incomplete
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                        help='Only analyze without cleaning')
    parser.add_argument('--no-drop-incomplete', action='store_true',
                        help='Keep stocks with high missing data')
    parser.add_argument('--no-fill-missing', action='store_true',
                        help='Do not fill missing dates')
    parser.add_argument('--missing-threshold', type=float, default=0.2,
                        help='Threshold for dropping incomplete stocks (default: 0.2)')
    parser.add_argument('--extreme-threshold', type=float, default=0.5,
                        help='Threshold for extreme return detection (default: 0.5)')
    parser.add_argument('--save-report', '-r', help='Save report to JSON file')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not args.analyze_only and not args.output:
        print("Error: Output file required unless --analyze-only is specified")
        sys.exit(1)

    # Run cleaning
    report = clean_stock_data(
        input_file=args.input,
        output_file=args.output,
        analyze_only=args.analyze_only,
        drop_incomplete=not args.no_drop_incomplete,
        missing_threshold=args.missing_threshold,
        extreme_return_threshold=args.extreme_threshold,
        fill_missing=not args.no_fill_missing,
    )

    # Save report if requested
    if args.save_report:
        report.save_to_file(args.save_report)

    # Exit with appropriate code
    sys.exit(0 if report.is_acceptable() else 1)


if __name__ == "__main__":
    main()
