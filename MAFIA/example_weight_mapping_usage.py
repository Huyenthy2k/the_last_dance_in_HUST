#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         example_weight_mapping_usage.py
 Description:  Example of how to use weight-to-symbol mapping in trading environment
 Author:       MASA
---------------------------------
'''

import numpy as np
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.weight_symbol_mapper import (
    map_weights_to_symbols,
    print_weight_mapping,
    get_top_stocks,
    validate_weight_symbol_mapping
)


def example_usage_in_trading_env():
    """
    Example showing how to use weight mapping in the trading environment context.
    This can be integrated into tradeEnv.py or callback_func.py without affecting existing code.
    """
    
    # Simulate what you would get from the trading environment
    # In actual code, you would access: env.stock_lst and env.actions_memory[-1]
    
    # Example 1: Map current weights to symbols
    print("=" * 60)
    print("Example 1: Map Current Portfolio Weights")
    print("=" * 60)
    
    # These would come from your environment
    stock_lst = np.array(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                          'META', 'NVDA', 'JPM', 'V', 'JNJ'])
    current_weights = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 
                               0.13, 0.11, 0.09, 0.08, 0.06])
    current_weights = current_weights / np.sum(current_weights)  # Normalize
    
    # Map and print
    print_weight_mapping(
        current_weights, 
        stock_lst, 
        title="Current Portfolio Allocation"
    )
    
    
    # Example 2: Get top 5 stocks
    print("\n" + "=" * 60)
    print("Example 2: Get Top 5 Stocks")
    print("=" * 60)
    
    top_5 = get_top_stocks(current_weights, stock_lst, top_k=5)
    print("\nTop 5 holdings:")
    for i, (symbol, weight) in enumerate(top_5, 1):
        print(f"{i}. {symbol}: {weight:.2%}")
    
    
    # Example 3: Validate weights before using
    print("\n" + "=" * 60)
    print("Example 3: Validate Weights")
    print("=" * 60)
    
    validation = validate_weight_symbol_mapping(current_weights, stock_lst)
    if validation['valid']:
        print(f"✓ Weights are valid (sum: {validation['weight_sum']:.6f})")
    else:
        print(f"✗ Invalid weights: {validation['error']}")
    
    
    # Example 4: Create DataFrame for analysis
    print("\n" + "=" * 60)
    print("Example 4: Create DataFrame for Analysis")
    print("=" * 60)
    
    df = map_weights_to_symbols(current_weights, stock_lst)
    print("\nDataFrame for further analysis:")
    print(df)
    print(f"\nTotal weight: {df['weight'].sum():.6f}")
    print(f"Number of stocks: {len(df)}")
    print(f"Average weight: {df['weight'].mean():.4f}")
    print(f"Max weight: {df['weight'].max():.4f} ({df.loc[df['weight'].idxmax(), 'symbol']})")
    print(f"Min weight: {df['weight'].min():.4f} ({df.loc[df['weight'].idxmin(), 'symbol']})")


def example_integration_with_tradeEnv():
    """
    Example showing how to integrate into tradeEnv.py save_profile method.
    This is optional and won't affect existing code flow.
    """
    
    print("\n" + "=" * 60)
    print("Example: Integration with tradeEnv.save_profile()")
    print("=" * 60)
    
    print("""
    To integrate into tradeEnv.py, you can add this to the save_profile method:
    
    # Optional: Print weight mapping for debugging/analysis
    if hasattr(self, 'actions_memory') and len(self.actions_memory) > 0:
        from utils.weight_symbol_mapper import print_weight_mapping
        sample_idx = min(4, len(self.actions_memory) - 1)
        sample_weights = self.actions_memory[sample_idx]
        print_weight_mapping(
            sample_weights, 
            self.stock_lst, 
            title=f"Sample Portfolio Weights (Day {sample_idx})",
            top_k=5
        )
    
    This will print a readable mapping without affecting the existing code flow.
    """)


if __name__ == "__main__":
    example_usage_in_trading_env()
    example_integration_with_tradeEnv()
    
    print("\n" + "=" * 60)
    print("Usage Examples Complete")
    print("=" * 60)
    print("""
    To use in your code:
    
    1. Import the functions:
       from utils.weight_symbol_mapper import print_weight_mapping, map_weights_to_symbols
    
    2. Use with your environment:
       print_weight_mapping(env.actions_memory[-1], env.stock_lst)
    
    3. Or create a DataFrame:
       df = map_weights_to_symbols(weights, symbols)
    
    All functions are non-destructive and won't affect existing code flow.
    """)

