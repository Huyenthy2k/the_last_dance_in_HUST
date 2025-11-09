#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data file validation utilities
Validates CSV data file formats for stock and index data
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List


# Expected columns for stock data files
REQUIRED_STOCK_COLUMNS = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume']
REQUIRED_STOCK_COLUMN_ORDER = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume']

# Expected columns for index data files (can use 'tic' or 'stock' for ticker)
REQUIRED_INDEX_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']
REQUIRED_INDEX_COLUMN_ORDER = ['date', 'open', 'high', 'low', 'close', 'volume']


def validate_stock_data_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate stock data file format
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    try:
        # Read first few rows to check format
        df = pd.read_csv(file_path, nrows=5)
        
        # Check if all required columns exist
        missing_cols = set(REQUIRED_STOCK_COLUMNS) - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns)}"
        
        # Check column order (optional but recommended)
        actual_order = [col for col in df.columns if col in REQUIRED_STOCK_COLUMN_ORDER]
        expected_order = REQUIRED_STOCK_COLUMN_ORDER
        if actual_order != expected_order:
            return False, f"Column order mismatch. Expected: {expected_order}, Found: {actual_order}"
        
        # Check data types
        df_full = pd.read_csv(file_path, nrows=100)  # Read more rows for validation
        if not pd.api.types.is_datetime64_any_dtype(df_full['date']):
            try:
                pd.to_datetime(df_full['date'])
            except:
                return False, "Column 'date' cannot be converted to datetime"
        
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df_full[col]):
                return False, f"Column '{col}' must be numeric"
        
        return True, "Validation passed"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def validate_index_data_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate index data file format
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    try:
        # Read first few rows to check format
        df = pd.read_csv(file_path, nrows=5)
        
        # Index files can have 'tic' or 'stock' column, but we only need the core columns
        # Check if all required columns exist
        missing_cols = set(REQUIRED_INDEX_COLUMNS) - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns)}"
        
        # Check data types
        df_full = pd.read_csv(file_path, nrows=100)
        if not pd.api.types.is_datetime64_any_dtype(df_full['date']):
            try:
                pd.to_datetime(df_full['date'])
            except:
                return False, "Column 'date' cannot be converted to datetime"
        
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df_full[col]):
                return False, f"Column '{col}' must be numeric"
        
        return True, "Validation passed"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def find_data_file(data_dir: str, file_name: Optional[str] = None, 
                   pattern: Optional[str] = None) -> Optional[str]:
    """
    Find data file in data directory
    
    Args:
        data_dir: Directory to search in
        file_name: Specific file name to look for (if provided)
        pattern: Pattern to match (e.g., '{market}_{topK}_{freq}.csv')
        
    Returns:
        Path to file if found, None otherwise
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return None
    
    # If specific file name provided, check if it exists
    if file_name:
        file_path = data_path / file_name
        if file_path.exists():
            return str(file_path)
        return None
    
    # If pattern provided, try to find matching file
    if pattern:
        # Simple pattern matching - look for files matching the pattern
        for file_path in data_path.glob('*.csv'):
            if pattern in file_path.name:
                return str(file_path)
    
    return None


def get_stock_data_file(config) -> Tuple[str, str]:
    """
    Get stock data file path with validation
    
    Args:
        config: Config object with dataDir, stock_data_file, market_name, topK, freq
        
    Returns:
        Tuple of (file_path, error_message)
        If successful, error_message is empty string
    """
    data_dir = config.dataDir
    
    # If specific file name provided in config
    if config.stock_data_file:
        file_path = os.path.join(data_dir, config.stock_data_file)
        is_valid, error_msg = validate_stock_data_file(file_path)
        if is_valid:
            return file_path, ""
        else:
            return None, f"Validation failed for {config.stock_data_file}: {error_msg}"
    
    # Otherwise, use auto-detection pattern
    pattern = '{}_{}_{}.csv'.format(config.market_name, config.topK, config.freq)
    file_path = os.path.join(data_dir, pattern)
    
    if os.path.exists(file_path):
        is_valid, error_msg = validate_stock_data_file(file_path)
        if is_valid:
            return file_path, ""
        else:
            return None, f"Validation failed for {pattern}: {error_msg}"
    
    # Try to find any CSV file in data directory
    data_path = Path(data_dir)
    csv_files = list(data_path.glob('*.csv'))
    
    # Filter out index files (they usually have 'index' in name)
    stock_files = [f for f in csv_files if 'index' not in f.name.lower()]
    
    if not stock_files:
        return None, f"No stock data file found in {data_dir}. Expected pattern: {pattern}"
    
    # Try to validate the first stock file found
    for file_path in stock_files:
        is_valid, error_msg = validate_stock_data_file(str(file_path))
        if is_valid:
            return str(file_path), ""
    
    return None, f"Found CSV files but none passed validation. Last error: {error_msg}"


def get_index_data_file(config, freq: str = '1d') -> Tuple[str, str]:
    """
    Get index data file path with validation
    
    Args:
        config: Config object with dataDir, index_data_file, market_name, freq
        freq: Frequency string (default '1d')
        
    Returns:
        Tuple of (file_path, error_message)
        If successful, error_message is empty string
    """
    data_dir = config.dataDir
    
    # If specific file name provided in config
    if config.index_data_file:
        file_path = os.path.join(data_dir, config.index_data_file)
        is_valid, error_msg = validate_index_data_file(file_path)
        if is_valid:
            return file_path, ""
        else:
            return None, f"Validation failed for {config.index_data_file}: {error_msg}"
    
    # Otherwise, use auto-detection pattern
    pattern = '{}_{}_index.csv'.format(config.market_name, freq)
    file_path = os.path.join(data_dir, pattern)
    
    if os.path.exists(file_path):
        is_valid, error_msg = validate_index_data_file(file_path)
        if is_valid:
            return file_path, ""
        else:
            return None, f"Validation failed for {pattern}: {error_msg}"
    
    # Try to find index file with different frequency
    if freq != '1d':
        pattern_1d = '{}_{}_index.csv'.format(config.market_name, '1d')
        file_path_1d = os.path.join(data_dir, pattern_1d)
        if os.path.exists(file_path_1d):
            is_valid, error_msg = validate_index_data_file(file_path_1d)
            if is_valid:
                return file_path_1d, ""
    
    return None, f"No valid index data file found in {data_dir}. Expected pattern: {pattern}"

