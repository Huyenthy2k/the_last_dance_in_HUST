
import os
import sys
import torch as th
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath("."))

from config import Config as MAFIAConfig
from RL_controller.mafia_modules import MAFIAModel
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer

def run_test():
    print(">>> Starting Risk Head Refactor Verification...")
    
    # 1. Setup Config
    config = MAFIAConfig()
    config.mafia_explicit_dim = 6
    config.mafia_num_epochs = 1
    config.mafia_batch_size = 4
    config.mafia_T_m = 256
    config.device = "cpu" # Use CPU for simple functional test
    
    print(f">>> Configured explicit_dim: {config.mafia_explicit_dim}")
    
    # 2. Initialize Model
    action_dim = 20 # Mock number of stocks
    model = MAFIAModel(config, action_dim=action_dim).to(config.device)
    print(">>> Model initialized successfully.")
    
    # 3. Initialize Trainer
    # We need a mock observer object that has 'mafia_model'
    class MockObserver:
        def __init__(self, model):
            self.mafia_model = model
            self.model_save_path = "test_artifacts"
            self.optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
            
    observer = MockObserver(model)
    trainer = ObserverOfflineBatchTrainer(config, observer, device=th.device("cpu"))
    print(">>> Trainer initialized successfully.")
    
    # 4. Create Mock Data
    print(">>> Generaring mock data...")
    # T_total needs to be enough for T_m + horizon + windows
    T_total = 1000 
    N = action_dim
    
    # Mock Open, High, Low, Close, Volume
    # Shape: (T_total, N, 5)
    # 5 features: Open, High, Low, Close, Volume
    stock_data = np.random.rand(T_total, N, 5).astype(np.float32) * 100
    market_data = np.random.rand(T_total, 1, 5).astype(np.float32) * 1000
    
    # Ensure High >= Low, etc to avoid sanity check issues if any
    stock_data[:, :, 1] = stock_data[:, :, 3] * 1.05 # High
    stock_data[:, :, 2] = stock_data[:, :, 3] * 0.95 # Low
    
    stock_returns = np.zeros_like(stock_data[:,:,3]) # (T, N)
    market_returns = np.zeros_like(market_data[:,:,3]).squeeze() # (T,)
    
    # Mock dates
    start_date = pd.Timestamp("2023-01-01")
    dates = np.array([(start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(T_total)])
    
    data_tensors = {
        "stock_ochlv": th.from_numpy(stock_data),
        "ochlv": th.from_numpy(stock_data), 
        "market_ochlv": th.from_numpy(market_data),
        "returns": th.from_numpy(stock_returns), 
        "market_returns": th.from_numpy(market_returns),
        "dates": dates, # Added
        "T_total": T_total
    }
    
    # 5. Run Train Epoch
    print(">>> Running train_epoch() step...")
    # We bypass the full loop and call train_epoch with our mock data
    # But train_epoch expects a DataLoader or similar?
    # Actually trainer.train_epoch takes 'data_tensors' dictionary
    
    try:
        metrics = trainer.train_epoch(data_tensors)
        print(">>> Train Epoch completed successfully!")
        print(f">>> Metrics: {metrics}")
        
        if "loss_risk" in metrics and "loss_total" in metrics:
            print(">>> SUCCESS: Risk Head Refactor Verified.")
            return True
        else:
            print(">>> FAILURE: Missing metrics.")
            return False
            
    except Exception as e:
        print(f">>> FAILURE: Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    if not success:
        sys.exit(1)
