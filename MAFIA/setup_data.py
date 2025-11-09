#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to setup data files for MAFIA framework
Validates and lists available data files in the data directory
"""

import os
import shutil
from pathlib import Path
import sys
sys.path.append(".")
from utils.data_validator import validate_stock_data_file, validate_index_data_file

def setup_data_files():
    """Setup data files by validating and listing available files"""
    data_dir = Path("./data")
    
    if not data_dir.exists():
        print(f"Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MAFIA Framework - Data File Setup & Validation")
    print("=" * 60)
    
    # Find all CSV files in data directory
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("\n⚠ No CSV files found in data directory!")
        print(f"   Please add your data files to: {data_dir.absolute()}")
    else:
        print(f"\nFound {len(csv_files)} CSV file(s) in data directory:")
        
        stock_files = []
        index_files = []
        other_files = []
        
        for csv_file in csv_files:
            # Check if it's an index file
            if 'index' in csv_file.name.lower():
                is_valid, error_msg = validate_index_data_file(str(csv_file))
                if is_valid:
                    index_files.append((csv_file.name, "✓ Valid"))
                else:
                    index_files.append((csv_file.name, f"✗ Invalid: {error_msg}"))
            else:
                is_valid, error_msg = validate_stock_data_file(str(csv_file))
                if is_valid:
                    stock_files.append((csv_file.name, "✓ Valid"))
                else:
                    other_files.append((csv_file.name, f"✗ Invalid: {error_msg}"))
        
        if stock_files:
            print("\n📊 Stock Data Files (valid):")
            for filename, status in stock_files:
                print(f"  {status} {filename}")
        
        if index_files:
            print("\n📈 Index Data Files (valid):")
            for filename, status in index_files:
                print(f"  {status} {filename}")
        
        if other_files:
            print("\n⚠ Other/Invalid Files:")
            for filename, status in other_files:
                print(f"  {status} {filename}")
    
    print("\n" + "=" * 60)
    print("Data File Configuration")
    print("=" * 60)
    print("\nTo use a specific data file, set in config.py:")
    print("  config.stock_data_file = 'your_stock_file.csv'")
    print("  config.index_data_file = 'your_index_file.csv'")
    print("\nOr leave as None to use auto-detection:")
    print("  config.stock_data_file = None  # Auto-detect from pattern")
    print("  config.index_data_file = None  # Auto-detect from pattern")
    
    print("\n" + "=" * 60)
    print("Expected Data File Formats")
    print("=" * 60)
    print("\nStock Data File:")
    print("  Required columns (in order): date, stock, open, high, low, close, volume")
    print("\nIndex Data File:")
    print("  Required columns (in order): date, open, high, low, close, volume")
    print("  (May also include 'tic' or 'stock' column)")
    
    print("\n" + "=" * 60)
    print("Data setup complete!")
    print("=" * 60)

if __name__ == "__main__":
    setup_data_files()
