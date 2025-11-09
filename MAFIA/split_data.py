#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để tách data thành training, validation, và test sets
Tách data dựa trên date ranges và lưu thành các file CSV riêng
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.data_validator import get_stock_data_file, validate_stock_data_file


def split_data_by_dates(data_file, output_dir, 
                        train_start, train_end,
                        valid_start=None, valid_end=None,
                        test_start=None, test_end=None,
                        verbose=True):
    """
    Tách data thành train/valid/test sets dựa trên date ranges
    
    Args:
        data_file: Đường dẫn đến file data gốc
        output_dir: Thư mục để lưu các file đã tách
        train_start: Ngày bắt đầu training (YYYY-MM-DD)
        train_end: Ngày kết thúc training (YYYY-MM-DD)
        valid_start: Ngày bắt đầu validation (YYYY-MM-DD, optional)
        valid_end: Ngày kết thúc validation (YYYY-MM-DD, optional)
        test_start: Ngày bắt đầu test (YYYY-MM-DD, optional)
        test_end: Ngày kết thúc test (YYYY-MM-DD, optional)
        verbose: In thông tin chi tiết
    
    Returns:
        dict: Thông tin về các file đã tạo
    """
    # Validate input file
    is_valid, error_msg = validate_stock_data_file(data_file)
    if not is_valid:
        raise ValueError(f"Data file validation failed: {error_msg}")
    
    # Read data
    if verbose:
        print(f"Reading data from: {data_file}")
    data = pd.read_csv(data_file)
    
    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Convert date strings to Timestamp
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    
    valid_start_ts = pd.Timestamp(valid_start) if valid_start else None
    valid_end_ts = pd.Timestamp(valid_end) if valid_end else None
    
    test_start_ts = pd.Timestamp(test_start) if test_start else None
    test_end_ts = pd.Timestamp(test_end) if test_end else None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = Path(data_file).stem
    
    results = {}
    
    # Split training data
    train_data = data[(data['date'] >= train_start_ts) & (data['date'] <= train_end_ts)].copy()
    train_data = train_data.sort_values(['date', 'stock'], ascending=True).reset_index(drop=True)
    
    train_file = os.path.join(output_dir, f'{base_name}_train.csv')
    train_data.to_csv(train_file, index=False)
    results['train'] = {
        'file': train_file,
        'rows': len(train_data),
        'dates': (train_data['date'].min(), train_data['date'].max()),
        'stocks': train_data['stock'].nunique()
    }
    
    if verbose:
        print(f"\n✓ Training set:")
        print(f"  File: {train_file}")
        print(f"  Rows: {len(train_data):,}")
        print(f"  Date range: {train_data['date'].min().date()} to {train_data['date'].max().date()}")
        print(f"  Stocks: {train_data['stock'].nunique()}")
    
    # Split validation data
    if valid_start and valid_end:
        valid_data = data[(data['date'] >= valid_start_ts) & (data['date'] <= valid_end_ts)].copy()
        valid_data = valid_data.sort_values(['date', 'stock'], ascending=True).reset_index(drop=True)
        
        valid_file = os.path.join(output_dir, f'{base_name}_valid.csv')
        valid_data.to_csv(valid_file, index=False)
        results['valid'] = {
            'file': valid_file,
            'rows': len(valid_data),
            'dates': (valid_data['date'].min(), valid_data['date'].max()),
            'stocks': valid_data['stock'].nunique()
        }
        
        if verbose:
            print(f"\n✓ Validation set:")
            print(f"  File: {valid_file}")
            print(f"  Rows: {len(valid_data):,}")
            print(f"  Date range: {valid_data['date'].min().date()} to {valid_data['date'].max().date()}")
            print(f"  Stocks: {valid_data['stock'].nunique()}")
    
    # Split test data
    if test_start and test_end:
        test_data = data[(data['date'] >= test_start_ts) & (data['date'] <= test_end_ts)].copy()
        test_data = test_data.sort_values(['date', 'stock'], ascending=True).reset_index(drop=True)
        
        test_file = os.path.join(output_dir, f'{base_name}_test.csv')
        test_data.to_csv(test_file, index=False)
        results['test'] = {
            'file': test_file,
            'rows': len(test_data),
            'dates': (test_data['date'].min(), test_data['date'].max()),
            'stocks': test_data['stock'].nunique()
        }
        
        if verbose:
            print(f"\n✓ Test set:")
            print(f"  File: {test_file}")
            print(f"  Rows: {len(test_data):,}")
            print(f"  Date range: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
            print(f"  Stocks: {test_data['stock'].nunique()}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total original rows: {len(data):,}")
        print(f"  Training rows: {results['train']['rows']:,} ({100*results['train']['rows']/len(data):.1f}%)")
        if 'valid' in results:
            print(f"  Validation rows: {results['valid']['rows']:,} ({100*results['valid']['rows']/len(data):.1f}%)")
        if 'test' in results:
            print(f"  Test rows: {results['test']['rows']:,} ({100*results['test']['rows']/len(data):.1f}%)")
        print("=" * 60)
    
    return results


def split_data_from_config(config=None, output_dir=None, verbose=True):
    """
    Tách data sử dụng date ranges từ config
    
    Args:
        config: Config object (nếu None, sẽ tạo mới)
        output_dir: Thư mục output (nếu None, dùng dataDir)
        verbose: In thông tin chi tiết
    
    Returns:
        dict: Thông tin về các file đã tạo
    """
    if config is None:
        config = Config()
    
    if output_dir is None:
        output_dir = config.dataDir
    
    # Get data file
    data_file, error_msg = get_stock_data_file(config)
    if data_file is None:
        raise ValueError(f"Cannot find data file: {error_msg}")
    
    # Get date ranges from config
    train_start = config.train_date_start.strftime('%Y-%m-%d')
    train_end = config.train_date_end.strftime('%Y-%m-%d')
    
    valid_start = config.valid_date_start.strftime('%Y-%m-%d') if config.valid_date_start else None
    valid_end = config.valid_date_end.strftime('%Y-%m-%d') if config.valid_date_end else None
    
    test_start = config.test_date_start.strftime('%Y-%m-%d') if config.test_date_start else None
    test_end = config.test_date_end.strftime('%Y-%m-%d') if config.test_date_end else None
    
    if verbose:
        print("=" * 60)
        print("Splitting Data from Config")
        print("=" * 60)
        print(f"Data file: {data_file}")
        print(f"Output directory: {output_dir}")
        print(f"\nDate ranges:")
        print(f"  Training: {train_start} to {train_end}")
        if valid_start and valid_end:
            print(f"  Validation: {valid_start} to {valid_end}")
        if test_start and test_end:
            print(f"  Test: {test_start} to {test_end}")
        print("=" * 60)
    
    return split_data_by_dates(
        data_file=data_file,
        output_dir=output_dir,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
        verbose=verbose
    )


def main():
    parser = argparse.ArgumentParser(
        description='Tách data thành training, validation, và test sets'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Đường dẫn đến file data gốc (nếu không có, sẽ dùng từ config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Thư mục để lưu các file đã tách (mặc định: dataDir từ config)'
    )
    
    parser.add_argument(
        '--use-config',
        action='store_true',
        help='Sử dụng date ranges từ config.py'
    )
    
    parser.add_argument(
        '--train-start',
        type=str,
        default=None,
        help='Ngày bắt đầu training (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--train-end',
        type=str,
        default=None,
        help='Ngày kết thúc training (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--valid-start',
        type=str,
        default=None,
        help='Ngày bắt đầu validation (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--valid-end',
        type=str,
        default=None,
        help='Ngày kết thúc validation (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--test-start',
        type=str,
        default=None,
        help='Ngày bắt đầu test (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--test-end',
        type=str,
        default=None,
        help='Ngày kết thúc test (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.use_config or (args.train_start is None and args.train_end is None):
            # Use config
            print("Using date ranges from config.py...")
            results = split_data_from_config(output_dir=args.output_dir, verbose=True)
        else:
            # Use command line arguments
            if args.data_file is None:
                # Try to get from config
                config = Config()
                data_file, error_msg = get_stock_data_file(config)
                if data_file is None:
                    raise ValueError(f"Cannot find data file. Please specify --data-file. Error: {error_msg}")
            else:
                data_file = args.data_file
            
            if args.output_dir is None:
                config = Config()
                output_dir = config.dataDir
            else:
                output_dir = args.output_dir
            
            if args.train_start is None or args.train_end is None:
                raise ValueError("--train-start and --train-end are required when not using --use-config")
            
            results = split_data_by_dates(
                data_file=data_file,
                output_dir=output_dir,
                train_start=args.train_start,
                train_end=args.train_end,
                valid_start=args.valid_start,
                valid_end=args.valid_end,
                test_start=args.test_start,
                test_end=args.test_end,
                verbose=True
            )
        
        print("\n✓ Data splitting completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()

