#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         test_weight_symbol_mapping.py
 Description:  Test script to verify weight-to-symbol mapping functionality
 Author:       MASA
---------------------------------
'''

import numpy as np
import pandas as pd
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.weight_symbol_mapper import (
    map_weights_to_symbols,
    print_weight_mapping,
    get_top_stocks,
    get_stocks_above_threshold,
    validate_weight_symbol_mapping
)


def test_basic_mapping():
    """Test basic weight to symbol mapping"""
    print("=" * 60)
    print("Test 1: Basic Weight to Symbol Mapping")
    print("=" * 60)
    
    # Sample data (10 stocks)
    symbols = np.array(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                       'META', 'NVDA', 'JPM', 'V', 'JNJ'])
    weights = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 
                       0.13, 0.11, 0.09, 0.08, 0.07])
    
    # Normalize to ensure sum = 1.0
    weights = weights / np.sum(weights)
    
    # Test mapping
    df = map_weights_to_symbols(weights, symbols)
    print("\nMapped DataFrame:")
    print(df)
    print(f"\nWeight sum: {df['weight'].sum():.6f}")
    
    assert len(df) == len(symbols), "DataFrame should have same length as input"
    assert abs(df['weight'].sum() - 1.0) < 1e-6, "Weights should sum to 1.0"
    assert df.iloc[0]['symbol'] == 'AAPL', "Should be sorted by weight (AAPL has highest)"
    
    print("✓ Test 1 passed\n")


def test_print_mapping():
    """Test printing weight mapping"""
    print("=" * 60)
    print("Test 2: Print Weight Mapping")
    print("=" * 60)
    
    symbols = np.array(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    
    print_weight_mapping(weights, symbols, title="Test Portfolio Weights")
    print_weight_mapping(weights, symbols, title="Top 3 Stocks", top_k=3)
    print_weight_mapping(weights, symbols, title="Stocks >= 20%", threshold=0.20)
    
    print("✓ Test 2 passed\n")


def test_top_stocks():
    """Test getting top K stocks"""
    print("=" * 60)
    print("Test 3: Get Top K Stocks")
    print("=" * 60)
    
    symbols = np.array(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'])
    weights = np.array([0.20, 0.18, 0.15, 0.12, 0.10, 0.13, 0.12])
    weights = weights / np.sum(weights)
    
    top_3 = get_top_stocks(weights, symbols, top_k=3)
    print(f"\nTop 3 stocks: {top_3}")
    
    assert len(top_3) == 3, "Should return 3 stocks"
    assert top_3[0][0] == 'AAPL', "AAPL should be top stock"
    assert top_3[0][1] > top_3[1][1], "Should be sorted by weight"
    
    print("✓ Test 3 passed\n")


def test_threshold_filtering():
    """Test filtering by threshold"""
    print("=" * 60)
    print("Test 4: Threshold Filtering")
    print("=" * 60)
    
    symbols = np.array(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    weights = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
    weights = weights / np.sum(weights)
    
    above_10pct = get_stocks_above_threshold(weights, symbols, threshold=0.10)
    print(f"\nStocks with weight >= 10%: {above_10pct}")
    
    assert len(above_10pct) == 4, "Should have 4 stocks above 10%"
    assert all(w >= 0.10 for _, w in above_10pct), "All should be >= 10%"
    
    print("✓ Test 4 passed\n")


def test_validation():
    """Test validation function"""
    print("=" * 60)
    print("Test 5: Validation")
    print("=" * 60)
    
    # Valid case
    symbols = np.array(['AAPL', 'MSFT', 'GOOGL'])
    weights = np.array([0.4, 0.35, 0.25])
    result = validate_weight_symbol_mapping(weights, symbols)
    print(f"\nValid case: {result}")
    assert result['valid'] == True, "Should be valid"
    
    # Invalid: length mismatch
    weights2 = np.array([0.4, 0.35])
    result = validate_weight_symbol_mapping(weights2, symbols)
    print(f"Length mismatch: {result}")
    assert result['valid'] == False, "Should be invalid"
    
    # Invalid: sum not 1.0
    weights3 = np.array([0.4, 0.35, 0.15])  # Sum = 0.9
    result = validate_weight_symbol_mapping(weights3, symbols)
    print(f"Sum not 1.0: {result}")
    assert result['valid'] == False, "Should be invalid"
    
    # Invalid: negative weights
    weights4 = np.array([0.4, 0.35, -0.25])
    result = validate_weight_symbol_mapping(weights4, symbols)
    print(f"Negative weights: {result}")
    assert result['valid'] == False, "Should be invalid"
    
    print("✓ Test 5 passed\n")


def test_with_real_data_structure():
    """Test with data structure similar to trading environment"""
    print("=" * 60)
    print("Test 6: Real Data Structure Simulation")
    print("=" * 60)
    
    # Simulate environment data
    stock_lst = np.array(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                          'META', 'NVDA', 'JPM', 'V', 'JNJ'])
    
    # Simulate actions_memory (like in tradeEnv)
    actions_memory = [
        np.array([0.1] * 10),  # Uniform distribution
        np.array([0.15, 0.12, 0.10, 0.08, 0.07, 0.13, 0.11, 0.09, 0.08, 0.05]),
        np.array([0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05])
    ]
    
    # Normalize each action
    for i in range(len(actions_memory)):
        actions_memory[i] = actions_memory[i] / np.sum(actions_memory[i])
    
    print("\nSimulating portfolio weights over 3 days:")
    for day, weights in enumerate(actions_memory, 1):
        print(f"\n--- Day {day} ---")
        print_weight_mapping(weights, stock_lst, title=f"Day {day} Portfolio", top_k=5)
        
        # Validate
        validation = validate_weight_symbol_mapping(weights, stock_lst)
        print(f"Validation: {'✓ Valid' if validation['valid'] else '✗ Invalid: ' + validation['error']}")
    
    print("✓ Test 6 passed\n")


def test_edge_cases():
    """Test edge cases"""
    print("=" * 60)
    print("Test 7: Edge Cases")
    print("=" * 60)
    
    # Single stock
    symbols = np.array(['AAPL'])
    weights = np.array([1.0])
    df = map_weights_to_symbols(weights, symbols)
    assert len(df) == 1, "Should handle single stock"
    print("✓ Single stock case passed")
    
    # Very small weights
    symbols = np.array(['AAPL', 'MSFT', 'GOOGL'])
    weights = np.array([0.98, 0.01, 0.01])
    df = map_weights_to_symbols(weights, symbols, threshold=0.005)
    assert len(df) == 3, "Should include all stocks above threshold"
    print("✓ Small weights case passed")
    
    # Equal weights
    symbols = np.array(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    df = map_weights_to_symbols(weights, symbols)
    assert abs(df['weight'].sum() - 1.0) < 1e-6, "Equal weights should sum to 1.0"
    print("✓ Equal weights case passed")
    
    print("✓ Test 7 passed\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing Weight-to-Symbol Mapping Functions")
    print("=" * 60 + "\n")
    
    try:
        test_basic_mapping()
        test_print_mapping()
        test_top_stocks()
        test_threshold_filtering()
        test_validation()
        test_with_real_data_structure()
        test_edge_cases()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

