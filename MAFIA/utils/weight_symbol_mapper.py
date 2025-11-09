#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         weight_symbol_mapper.py
 Description:  Utility functions to map portfolio weights to stock symbols
 Author:       MASA
---------------------------------
'''

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


def map_weights_to_symbols(
    weights: Union[np.ndarray, List[float]],
    symbols: Union[np.ndarray, List[str]],
    sort_by_weight: bool = True,
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Map portfolio weights to their corresponding stock symbols.
    
    Args:
        weights: Array of portfolio weights (should sum to ~1.0)
        symbols: Array of stock symbols corresponding to weights
        sort_by_weight: If True, sort by weight descending
        threshold: If provided, only return stocks with weight >= threshold
    
    Returns:
        DataFrame with columns: ['symbol', 'weight', 'weight_pct']
    """
    weights = np.array(weights)
    symbols = np.array(symbols)
    
    if len(weights) != len(symbols):
        raise ValueError(
            f"Length mismatch: weights has {len(weights)} elements, "
            f"symbols has {len(symbols)} elements"
        )
    
    # Create mapping DataFrame
    df = pd.DataFrame({
        'symbol': symbols,
        'weight': weights,
        'weight_pct': weights * 100  # Convert to percentage
    })
    
    # Filter by threshold if provided
    if threshold is not None:
        df = df[df['weight'] >= threshold].copy()
    
    # Sort by weight if requested
    if sort_by_weight:
        df = df.sort_values('weight', ascending=False).reset_index(drop=True)
    
    return df


def print_weight_mapping(
    weights: Union[np.ndarray, List[float]],
    symbols: Union[np.ndarray, List[str]],
    title: str = "Portfolio Weights",
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    precision: int = 4
) -> None:
    """
    Print portfolio weights with their corresponding symbols in a readable format.
    
    Args:
        weights: Array of portfolio weights
        symbols: Array of stock symbols
        title: Title for the output
        top_k: If provided, only show top K stocks by weight
        threshold: If provided, only show stocks with weight >= threshold
        precision: Number of decimal places for weights
    """
    df = map_weights_to_symbols(weights, symbols, sort_by_weight=True, threshold=threshold)
    
    if top_k is not None:
        df = df.head(top_k)
    
    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Symbol':<15} {'Weight':<15} {'Weight %':<15}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        print(f"{row['symbol']:<15} {row['weight']:<15.{precision}f} {row['weight_pct']:<15.{precision}f}%")
    
    if top_k is not None or threshold is not None:
        remaining_weight = 1.0 - df['weight'].sum()
        if remaining_weight > 1e-6:
            print(f"{'... (others)':<15} {remaining_weight:<15.{precision}f} {remaining_weight*100:<15.{precision}f}%")
    
    print("=" * 60)
    print(f"Total: {df['weight'].sum():.{precision}f} ({df['weight'].sum()*100:.{precision}f}%)")
    print()


def get_top_stocks(
    weights: Union[np.ndarray, List[float]],
    symbols: Union[np.ndarray, List[str]],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Get top K stocks by weight.
    
    Args:
        weights: Array of portfolio weights
        symbols: Array of stock symbols
        top_k: Number of top stocks to return
    
    Returns:
        List of tuples (symbol, weight) sorted by weight descending
    """
    df = map_weights_to_symbols(weights, symbols, sort_by_weight=True)
    top_df = df.head(top_k)
    return [(row['symbol'], row['weight']) for _, row in top_df.iterrows()]


def get_stocks_above_threshold(
    weights: Union[np.ndarray, List[float]],
    symbols: Union[np.ndarray, List[str]],
    threshold: float = 0.05
) -> List[Tuple[str, float]]:
    """
    Get all stocks with weight above a threshold.
    
    Args:
        weights: Array of portfolio weights
        symbols: Array of stock symbols
        threshold: Minimum weight threshold (e.g., 0.05 for 5%)
    
    Returns:
        List of tuples (symbol, weight) sorted by weight descending
    """
    df = map_weights_to_symbols(weights, symbols, sort_by_weight=True, threshold=threshold)
    return [(row['symbol'], row['weight']) for _, row in df.iterrows()]


def validate_weight_symbol_mapping(
    weights: Union[np.ndarray, List[float]],
    symbols: Union[np.ndarray, List[str]]
) -> Dict[str, Union[bool, str, float]]:
    """
    Validate that weights and symbols are properly aligned.
    
    Args:
        weights: Array of portfolio weights
        symbols: Array of stock symbols
    
    Returns:
        Dictionary with validation results:
        - 'valid': bool, whether mapping is valid
        - 'error': str, error message if invalid
        - 'weight_sum': float, sum of weights
        - 'num_stocks': int, number of stocks
    """
    weights = np.array(weights)
    symbols = np.array(symbols)
    
    result = {
        'valid': True,
        'error': None,
        'weight_sum': float(np.sum(weights)),
        'num_stocks': len(symbols)
    }
    
    # Check length match
    if len(weights) != len(symbols):
        result['valid'] = False
        result['error'] = f"Length mismatch: {len(weights)} weights vs {len(symbols)} symbols"
        return result
    
    # Check for NaN or inf
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        result['valid'] = False
        result['error'] = "Weights contain NaN or Inf values"
        return result
    
    # Check for negative weights (if not allowed)
    if np.any(weights < 0):
        result['valid'] = False
        result['error'] = "Weights contain negative values"
        return result
    
    # Check weight sum (should be close to 1.0)
    weight_sum = np.sum(weights)
    if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point errors
        result['valid'] = False
        result['error'] = f"Weight sum is {weight_sum:.6f}, expected ~1.0"
        return result
    
    # Check for duplicate symbols
    if len(symbols) != len(set(symbols)):
        result['valid'] = False
        result['error'] = "Duplicate symbols found"
        return result
    
    return result

