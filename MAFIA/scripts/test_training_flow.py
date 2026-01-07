#!/usr/bin/env python3
"""
Test script for MAFIA Observer Training Flow.

Runs training with minimal data to detect crashes and logic issues.
Tests all modes: MACRO_ONLY, SELECTION_ONLY, FULL.
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import torch as th

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, MafiaTrainMode
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer


def create_minimal_test_data(num_stocks: int = 10, num_days: int = 200):
    """Create minimal synthetic data for testing."""
    print(f"[TEST] Creating synthetic data: {num_stocks} stocks, {num_days} days")

    np.random.seed(42)

    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='D')

    # Generate stock data
    stock_data = []
    stock_symbols = [f"STOCK_{i:02d}" for i in range(num_stocks)]

    for symbol in stock_symbols:
        base_price = np.random.uniform(10, 100)
        returns = np.random.normal(0.001, 0.02, num_days)
        prices = base_price * np.cumprod(1 + returns)

        for i, date in enumerate(dates):
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = np.random.randint(100000, 10000000)

            stock_data.append({
                'date': date,
                'stock': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

    stock_df = pd.DataFrame(stock_data)

    # Generate market (index) data
    market_data = []
    base_index = 1000
    returns = np.random.normal(0.0005, 0.015, num_days)
    index_prices = base_index * np.cumprod(1 + returns)

    for i, date in enumerate(dates):
        close = index_prices[i]
        high = close * (1 + abs(np.random.normal(0, 0.008)))
        low = close * (1 - abs(np.random.normal(0, 0.008)))
        open_price = close * (1 + np.random.normal(0, 0.004))
        volume = np.random.randint(1000000, 100000000)

        market_data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    market_df = pd.DataFrame(market_data)

    print(f"[TEST] Stock data shape: {stock_df.shape}")
    print(f"[TEST] Market data shape: {market_df.shape}")

    return stock_df, market_df, stock_symbols


def test_training_mode(mode: str, config: Config, stock_df: pd.DataFrame,
                       market_df: pd.DataFrame, stock_list: list,
                       num_epochs: int = 2, batches_per_epoch: int = 3):
    """Test a single training mode."""
    print(f"\n{'='*60}")
    print(f"[TEST] Testing mode: {mode}")
    print(f"{'='*60}")

    errors = []
    warnings = []

    try:
        # Set training mode
        if mode == "MACRO_ONLY":
            config.mafia_train_mode = MafiaTrainMode.MACRO_ONLY
        elif mode == "SELECTION_ONLY":
            config.mafia_train_mode = MafiaTrainMode.SELECTION_ONLY
        else:
            config.mafia_train_mode = MafiaTrainMode.FULL

        # Create observer
        print(f"[TEST] Creating MAFIAObserver...")
        observer = MAFIAObserver(
            config=config,
            action_dim=len(stock_list),
        )

        # Create trainer
        print(f"[TEST] Creating trainer...")
        trainer = ObserverOfflineBatchTrainer(
            config=config,
            observer=observer,
            device=None,  # Auto-detect
        )

        print(f"[TEST] Device: {trainer.device}")
        print(f"[TEST] Batch size: {trainer.batch_size}")
        print(f"[TEST] Trajectory length: {trainer.T_m}")

        # Prepare data tensors
        print(f"[TEST] Preparing data tensors...")
        start_date = stock_df['date'].min()
        end_date = stock_df['date'].max()
        data_tensors = trainer.prepare_data_tensors(
            data=stock_df,
            stock_list=stock_list,
            start_date=start_date,
            end_date=end_date,
            market_data=market_df,
        )

        print(f"[TEST] Data tensors keys: {list(data_tensors.keys())}")
        print(f"[TEST] OCHLV shape: {data_tensors['ochlv'].shape}")

        # Run training epochs
        for epoch in range(num_epochs):
            print(f"\n[TEST] --- Epoch {epoch + 1}/{num_epochs} ---")

            try:
                result = trainer.train_epoch(
                    data_tensors=data_tensors,
                    training_mode=mode,
                    steps_per_epoch=batches_per_epoch,
                    verbose=True,
                    epoch=epoch,  # Sync epoch display
                )

                # Check result (can be dict or ObserverValidationResult)
                print(f"[TEST] Epoch {epoch + 1} completed:")
                if hasattr(result, 'get'):
                    # Dict-like result
                    print(f"       Loss Total: {result.get('loss_total', 'N/A')}")
                    print(f"       Loss PG: {result.get('loss_pg', 'N/A')}")
                    print(f"       Loss Risk: {result.get('loss_risk', 'N/A')}")
                    print(f"       Loss Dir: {result.get('loss_dir', 'N/A')}")
                    result_dict = result
                elif hasattr(result, 'loss_total'):
                    # ObserverValidationResult
                    print(f"       Loss Total: {result.loss_total:.6f}")
                    print(f"       Loss PG: {getattr(result, 'loss_pg', 0.0):.6f}")
                    print(f"       Loss Risk: {getattr(result, 'loss_risk', 0.0):.6f}")
                    print(f"       Loss Dir: {getattr(result, 'loss_dir', 0.0):.6f}")
                    result_dict = vars(result) if hasattr(result, '__dict__') else {}
                else:
                    print(f"       Result type: {type(result)}")
                    result_dict = {}

                # Check for NaN/Inf
                for key, val in result_dict.items():
                    if isinstance(val, (int, float)):
                        if np.isnan(val) or np.isinf(val):
                            warnings.append(f"Epoch {epoch}: {key} = {val} (NaN/Inf)")

            except Exception as e:
                error_msg = f"Epoch {epoch} failed: {type(e).__name__}: {e}"
                errors.append(error_msg)
                print(f"[ERROR] {error_msg}")
                traceback.print_exc()
                break

        # Test validation
        print(f"\n[TEST] Testing validation...")
        try:
            val_result = trainer.validate_epoch(
                data_tensors=data_tensors,
                steps=2,
            )
            print(f"[TEST] Validation completed successfully")
        except Exception as e:
            error_msg = f"Validation failed: {type(e).__name__}: {e}"
            errors.append(error_msg)
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()

        # Cleanup
        del trainer
        del observer
        if th.cuda.is_available():
            th.cuda.empty_cache()
        if hasattr(th, 'mps') and hasattr(th.mps, 'empty_cache'):
            th.mps.empty_cache()

    except Exception as e:
        error_msg = f"Setup failed: {type(e).__name__}: {e}"
        errors.append(error_msg)
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()

    return errors, warnings


def run_all_tests():
    """Run all training flow tests."""
    print("="*70)
    print("MAFIA Observer Training Flow Test")
    print("="*70)

    # Minimal config for testing
    config = Config(create_dirs=False)

    # Override for minimal test
    config.mafia_batch_size = 4  # Very small batch
    config.mafia_trajectory_length = 32  # Short trajectory
    config.mafia_T_w = 10  # Short window
    config.mafia_top_k = 3  # Few stocks
    config.mafia_pg_reward_horizon = 3  # Short horizon
    config.mps_max_batch_size = 4

    # Disable heavy features
    config.log_trajectory_details = False

    # Create test data
    stock_df, market_df, stock_list = create_minimal_test_data(
        num_stocks=10,  # Must be >= mafia_top_k
        num_days=100    # Must be >= T_w + T_m + horizon
    )

    all_errors = {}
    all_warnings = {}

    # Test all modes
    modes = ["MACRO_ONLY", "SELECTION_ONLY", "FULL"]

    for mode in modes:
        errors, warnings = test_training_mode(
            mode=mode,
            config=config,
            stock_df=stock_df,
            market_df=market_df,
            stock_list=stock_list,
            num_epochs=2,
            batches_per_epoch=3,
        )
        all_errors[mode] = errors
        all_warnings[mode] = warnings

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total_errors = 0
    total_warnings = 0

    for mode in modes:
        errors = all_errors[mode]
        warnings = all_warnings[mode]

        status = "PASS" if len(errors) == 0 else "FAIL"
        print(f"\n{mode}: {status}")

        if errors:
            print(f"  Errors ({len(errors)}):")
            for err in errors:
                print(f"    - {err}")
            total_errors += len(errors)

        if warnings:
            print(f"  Warnings ({len(warnings)}):")
            for warn in warnings:
                print(f"    - {warn}")
            total_warnings += len(warnings)

    print("\n" + "-"*70)
    print(f"Total: {total_errors} errors, {total_warnings} warnings")

    if total_errors > 0:
        print("\nTEST FAILED - Please fix errors before running full training")
        return False
    else:
        print("\nTEST PASSED - All training modes work correctly")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
