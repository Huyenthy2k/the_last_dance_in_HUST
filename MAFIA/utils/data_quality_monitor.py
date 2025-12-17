#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Monitoring Utilities for MAFIA Observer Training

Provides real-time quality monitoring during data loading and training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """
    Report of data quality statistics and issues.

    Used for monitoring data quality during:
    - Data loading
    - Tensor preparation
    - Training validation
    """
    # Basic stats
    total_rows: int = 0
    total_stocks: int = 0
    total_dates: int = 0

    # Price issues
    nan_price_count: int = 0
    zero_price_count: int = 0
    negative_price_count: int = 0

    # Return issues
    nan_return_count: int = 0
    inf_return_count: int = 0
    extreme_return_count: int = 0  # |return| > threshold

    # Missing data
    missing_date_ratio: float = 0.0
    stocks_with_missing: List[str] = field(default_factory=list)
    incomplete_stocks: List[str] = field(default_factory=list)  # >threshold missing

    # Cleaning actions taken
    cleaning_actions: List[str] = field(default_factory=list)
    rows_cleaned: int = 0

    def is_acceptable(self, max_nan_ratio: float = 0.01, max_inf: int = 0) -> bool:
        """
        Check if data quality meets training requirements.

        Args:
            max_nan_ratio: Maximum acceptable NaN ratio (default 1%)
            max_inf: Maximum acceptable Inf count (default 0)

        Returns:
            True if data quality is acceptable
        """
        if self.total_rows == 0:
            return False

        nan_ratio = self.nan_price_count / (self.total_rows * 4)  # 4 price columns
        return nan_ratio < max_nan_ratio and self.inf_return_count <= max_inf

    def get_summary(self) -> str:
        """Get human-readable summary string."""
        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            f"Rows: {self.total_rows:,} | Stocks: {self.total_stocks} | Dates: {self.total_dates}",
            "",
            "Price Issues:",
            f"  NaN: {self.nan_price_count} | Zero: {self.zero_price_count} | Negative: {self.negative_price_count}",
            "",
            "Return Issues:",
            f"  NaN: {self.nan_return_count} | Inf: {self.inf_return_count} | Extreme: {self.extreme_return_count}",
            "",
            f"Missing Date Ratio: {self.missing_date_ratio:.2%}",
        ]

        if self.incomplete_stocks:
            lines.append(f"Incomplete Stocks (>20% missing): {self.incomplete_stocks}")

        if self.cleaning_actions:
            lines.append("")
            lines.append("Cleaning Actions:")
            for action in self.cleaning_actions:
                lines.append(f"  - {action}")

        status = "✅ ACCEPTABLE" if self.is_acceptable() else "❌ NEEDS ATTENTION"
        lines.extend(["", f"Status: {status}", "=" * 60])

        return "\n".join(lines)

    def log_summary(self, level: int = logging.INFO):
        """Log summary to logger."""
        for line in self.get_summary().split("\n"):
            logger.log(level, line)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_rows": int(self.total_rows),
            "total_stocks": int(self.total_stocks),
            "total_dates": int(self.total_dates),
            "price_issues": {
                "nan": int(self.nan_price_count),
                "zero": int(self.zero_price_count),
                "negative": int(self.negative_price_count),
            },
            "return_issues": {
                "nan": int(self.nan_return_count),
                "inf": int(self.inf_return_count),
                "extreme": int(self.extreme_return_count),
            },
            "missing_data": {
                "ratio": float(self.missing_date_ratio),
                "incomplete_stocks": self.incomplete_stocks,
            },
            "cleaning": {
                "actions": self.cleaning_actions,
                "rows_cleaned": int(self.rows_cleaned),
            },
            "is_acceptable": self.is_acceptable(),
        }


def analyze_dataframe_quality(
    df: pd.DataFrame,
    price_cols: List[str] = None,
    extreme_return_threshold: float = 0.5,
    missing_threshold: float = 0.2,
) -> DataQualityReport:
    """
    Analyze quality of a stock DataFrame.

    Args:
        df: DataFrame with columns: date, stock, open, high, low, close, volume
        price_cols: Price columns to check (default: open, high, low, close)
        extreme_return_threshold: Threshold for extreme returns
        missing_threshold: Threshold for marking stocks as incomplete

    Returns:
        DataQualityReport with analysis results
    """
    if price_cols is None:
        price_cols = ['open', 'high', 'low', 'close']

    report = DataQualityReport()

    # Basic stats
    report.total_rows = len(df)
    report.total_stocks = df['stock'].nunique()
    report.total_dates = df['date'].nunique()

    # Price issues
    for col in price_cols:
        if col in df.columns:
            report.nan_price_count += df[col].isna().sum()
            report.zero_price_count += (df[col] == 0).sum()
            report.negative_price_count += (df[col] < 0).sum()

    # Compute returns for analysis
    if 'close' in df.columns:
        df_sorted = df.sort_values(['stock', 'date']).copy()
        df_sorted['_return'] = df_sorted.groupby('stock')['close'].pct_change()

        report.nan_return_count = df_sorted['_return'].isna().sum()
        report.inf_return_count = np.isinf(df_sorted['_return']).sum()
        report.extreme_return_count = (
            (df_sorted['_return'].abs() > extreme_return_threshold) &
            ~np.isinf(df_sorted['_return']) &
            ~np.isnan(df_sorted['_return'])
        ).sum()

    # Missing date analysis
    if 'date' in df.columns and 'stock' in df.columns:
        all_dates = df['date'].nunique()
        stock_date_counts = df.groupby('stock')['date'].nunique()

        for stock, count in stock_date_counts.items():
            missing_ratio = 1 - count / all_dates
            if missing_ratio > 0.01:
                report.stocks_with_missing.append(stock)
            if missing_ratio > missing_threshold:
                report.incomplete_stocks.append(stock)

        # Overall missing ratio
        expected_total = all_dates * df['stock'].nunique()
        actual_total = len(df)
        report.missing_date_ratio = 1 - actual_total / expected_total if expected_total > 0 else 0

    return report


def analyze_tensor_quality(
    returns: np.ndarray,
    ochlv: Optional[np.ndarray] = None,
    extreme_threshold: float = 0.5,
) -> DataQualityReport:
    """
    Analyze quality of prepared tensors.

    Args:
        returns: Returns tensor (T, N) or (B, T, N)
        ochlv: OCHLV tensor (T, N, 5) or (B, T, N, 5)
        extreme_threshold: Threshold for extreme returns

    Returns:
        DataQualityReport with tensor analysis
    """
    report = DataQualityReport()

    # Flatten for analysis
    returns_flat = returns.flatten()
    report.total_rows = returns_flat.shape[0]

    # Return issues
    report.nan_return_count = np.isnan(returns_flat).sum()
    report.inf_return_count = np.isinf(returns_flat).sum()
    valid_returns = returns_flat[~np.isnan(returns_flat) & ~np.isinf(returns_flat)]
    report.extreme_return_count = (np.abs(valid_returns) > extreme_threshold).sum()

    # OCHLV issues
    if ochlv is not None:
        ochlv_flat = ochlv.flatten()
        report.nan_price_count = np.isnan(ochlv_flat).sum()
        report.zero_price_count = (ochlv_flat == 0).sum()
        report.negative_price_count = (ochlv_flat < 0).sum()

    return report


def validate_data_for_training(
    df: pd.DataFrame,
    config: Optional[object] = None,
    strict_mode: bool = False,
) -> Tuple[bool, DataQualityReport]:
    """
    Validate data is ready for training.

    Args:
        df: Stock DataFrame
        config: Optional config object with thresholds
        strict_mode: If True, raise exception on quality issues

    Returns:
        Tuple of (is_valid, report)

    Raises:
        DataQualityError: If strict_mode and data quality is unacceptable
    """
    # Get thresholds from config or use defaults
    extreme_threshold = getattr(config, 'mafia_extreme_return_threshold', 0.5) if config else 0.5
    missing_threshold = getattr(config, 'data_missing_threshold', 0.2) if config else 0.2

    report = analyze_dataframe_quality(
        df,
        extreme_return_threshold=extreme_threshold,
        missing_threshold=missing_threshold,
    )

    is_valid = report.is_acceptable()

    if not is_valid and strict_mode:
        raise DataQualityError(f"Data quality below threshold:\n{report.get_summary()}")

    return is_valid, report


class DataQualityError(Exception):
    """Exception raised when data quality is unacceptable."""
    pass


def quick_quality_check(df: pd.DataFrame) -> bool:
    """
    Quick quality check for data.

    Returns:
        True if data passes basic quality checks
    """
    if df.empty:
        return False

    required_cols = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return False

    # Check for any Inf in returns
    df_sorted = df.sort_values(['stock', 'date'])
    returns = df_sorted.groupby('stock')['close'].pct_change()
    if np.isinf(returns).any():
        return False

    # Check for excessive NaN (>5%)
    nan_ratio = df[['open', 'high', 'low', 'close']].isna().sum().sum() / (len(df) * 4)
    if nan_ratio > 0.05:
        return False

    return True
