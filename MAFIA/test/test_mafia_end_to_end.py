#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         test_mafia_end_to_end.py
 Description:  End-to-end test for MAFIA integration with MASA framework
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import torch as th
import sys
import os
import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.featGen import FeatureProcesser
from RL_controller.mafia_observer import MAFIAObserver


def test_mafia_observer_initialization():
    """Test MAFIAObserver can be initialized."""
    print("Testing MAFIAObserver initialization...")
    
    try:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        config = Config(seed_num=2022, current_date=current_date)
        
        # Set MAFIA mode
        config.benchmark_algo = 'MASA-mafia'
        config.enable_market_observer = True
        
        # Re-initialize to apply MAFIA settings
        config.__init__(seed_num=2022, current_date=current_date)
        config.benchmark_algo = 'MASA-mafia'
        
        # Create observer
        action_dim = 10
        observer = MAFIAObserver(config=config, action_dim=action_dim)
        
        assert observer is not None, "MAFIAObserver should be created"
        assert observer.action_dim == action_dim, f"action_dim mismatch: {observer.action_dim} != {action_dim}"
        assert observer.mafia_model is not None, "MAFIA model should be initialized"
        
        print(f"✓ MAFIAObserver initialized with action_dim={action_dim}")
        print(f"✓ Device: {observer.device}")
        print("✓ MAFIAObserver initialization: PASSED\n")
        return True, config, observer
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_mafia_observer_predict():
    """Test MAFIAObserver predict method."""
    print("Testing MAFIAObserver predict...")
    
    success, config, observer = test_mafia_observer_initialization()
    if not success:
        return False
    
    try:
        # Create dummy raw OCHLV data: (N=10, M=5, T_w=30)
        N, M, T_w = 10, 5, 30
        raw_ochlv_data = np.random.rand(N, M, T_w) * 100 + 50
        
        # Test predict
        (
            market_vector,
            boundary_risk,
            _market_scores_full,
            _gate_weights,
            _market_context,
            _stock_embedding,
            sigma_val,
            sigma_log_p,
        ) = observer.predict(
            raw_ochlv_data=raw_ochlv_data,
            mode='test'
        )
        
        # Verify outputs
        assert market_vector.shape == (1, N), \
            f"Expected market_vector shape (1, {N}), got {market_vector.shape}"
        assert boundary_risk.shape == (1,), \
            f"Expected boundary_risk shape (1,), got {boundary_risk.shape}"
        assert sigma_val.shape[0] == 1, "sigma_val should have batch dimension"
        assert sigma_log_p.shape[1] == 3, "sigma_log_p should have 3 direction logits"
        assert (boundary_risk > 0).all(), "boundary_risk should be positive"
        
        print(f"✓ market_vector shape: {market_vector.shape}")
        print(f"✓ boundary_risk shape: {boundary_risk.shape}")
        print(f"✓ market_vector range: [{market_vector.min():.4f}, {market_vector.max():.4f}]")
        print(f"✓ boundary_risk value: {boundary_risk[0]:.4f}")
        print("✓ MAFIAObserver predict: PASSED\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mafia_with_real_data():
    """Test MAFIA with real data if available."""
    print("Testing MAFIA with real data...")
    
    try:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        config = Config(seed_num=2022, current_date=current_date)
        config.benchmark_algo = 'MASA-mafia'
        config.enable_market_observer = True
        
        # Check if data file exists
        data_path = os.path.join(config.dataDir, f'{config.market_name}_{config.topK}_{config.freq}.csv')
        if not os.path.exists(data_path):
            print(f"⚠ Data file not found: {data_path}")
            print("⚠ Skipping real data test (this is OK if data is not set up)")
            return True
        
        # Load and process data
        print(f"✓ Loading data from: {data_path}")
        data = pd.read_csv(data_path, header=0)
        
        # Preprocess features
        featProc = FeatureProcesser(config=config)
        data_dict = featProc.preprocess_feat(data=data)
        stock_num = data_dict['train']['stock'].nunique()
        
        print(f"✓ Data processed: {stock_num} stocks")
        
        # Create MAFIA observer
        observer = MAFIAObserver(config=config, action_dim=stock_num)
        
        # Test with a few samples from training data
        train_data = data_dict['train']
        dates = sorted(train_data['date'].unique())[:5]  # Test with first 5 dates
        
        print(f"✓ Testing with {len(dates)} dates...")
        
        for date in dates:
            date_data = train_data[train_data['date'] == date]
            
            # Extract OCHLV for this date and previous 29 days
            all_dates = sorted(train_data['date'].unique())
            date_idx = all_dates.index(date)
            start_idx = max(0, date_idx - 29)
            window_dates = all_dates[start_idx:date_idx+1]
            
            # Build OCHLV array
            stocks = sorted(date_data['stock'].unique())
            N = len(stocks)
            T_w = len(window_dates)
            
            if T_w < 30:
                print(f"⚠ Skipping {date}: insufficient history ({T_w} days)")
                continue
            
            ochlv_array = np.zeros((N, 5, T_w))
            for i, stock in enumerate(stocks):
                stock_data = train_data[train_data['stock'] == stock]
                for j, wdate in enumerate(window_dates):
                    day_data = stock_data[stock_data['date'] == wdate]
                    if len(day_data) > 0:
                        ochlv_array[i, 0, j] = day_data['open'].values[0]
                        ochlv_array[i, 1, j] = day_data['close'].values[0]
                        ochlv_array[i, 2, j] = day_data['high'].values[0]
                        ochlv_array[i, 3, j] = day_data['low'].values[0]
                        ochlv_array[i, 4, j] = day_data['volume'].values[0]
            
            # Test predict
            (
                market_vector,
                boundary_risk,
                _market_scores_full,
                _gate_weights,
                _market_context,
                _stock_embedding,
                _sigma_val,
                _sigma_log_p,
            ) = observer.predict(
                raw_ochlv_data=ochlv_array,
                mode='test'
            )
            
            assert market_vector.shape[1] == N, f"market_vector size mismatch: {market_vector.shape[1]} != {N}"
            print(f"  ✓ {date}: market_vector shape {market_vector.shape}, boundary_risk={boundary_risk[0]:.4f}")
        
        print("✓ MAFIA with real data: PASSED\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mafia_interface_compatibility():
    """Test MAFIAObserver interface compatibility with MarketObserver."""
    print("Testing interface compatibility...")
    
    try:
        current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        config = Config(seed_num=2022, current_date=current_date)
        config.benchmark_algo = 'MASA-mafia'
        config.enable_market_observer = True
        
        observer = MAFIAObserver(config=config, action_dim=10)
        
        # Test all required methods exist
        assert hasattr(observer, 'predict'), "Missing predict method"
        assert hasattr(observer, 'train'), "Missing train method"
        assert hasattr(observer, 'reset'), "Missing reset method"
        assert hasattr(observer, 'update_hidden_vec_reward'), "Missing update_hidden_vec_reward method"
        
        # Test reset
        observer.reset()
        assert len(observer.market_vector_lst) == 0, "reset() should clear buffers"
        
        # Test update_hidden_vec_reward (should not crash)
        rate_of_price_change = np.random.rand(1, 11)  # (batch, N+1) with cash
        mkt_direction = np.array([1])
        observer.update_hidden_vec_reward('train', rate_of_price_change, mkt_direction)
        
        print("✓ All required methods exist")
        print("✓ reset() works correctly")
        print("✓ update_hidden_vec_reward() works correctly")
        print("✓ Interface compatibility: PASSED\n")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_end_to_end_tests():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("MAFIA End-to-End Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Initialization
    results.append(("Initialization", test_mafia_observer_initialization()[0]))
    
    # Test 2: Predict
    results.append(("Predict", test_mafia_observer_predict()))
    
    # Test 3: Interface compatibility
    results.append(("Interface Compatibility", test_mafia_interface_compatibility()))
    
    # Test 4: Real data (optional)
    results.append(("Real Data", test_mafia_with_real_data()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("=" * 60)
    if all_passed:
        print("ALL END-TO-END TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = run_end_to_end_tests()
    sys.exit(0 if success else 1)
