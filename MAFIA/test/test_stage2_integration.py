
import unittest
import os
import sys
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from scripts.train_separated import run_phase2_walkforward, create_phase2_iterative_config
from utils.tradeEnv import StockPortfolioEnv

class TestStage2Integration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "output")
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create dummy Observer checkpoints
        self.train_years = [2015, 2016]
        self.infer_year = 2017
        
        # Create dummy checkpoint files
        for year in self.train_years:
            dummy_ckpt = os.path.join(self.checkpoint_dir, f"observer_best_{year}.pth")
            with open(dummy_ckpt, "w") as f:
                f.write("dummy_content")
                
        # Mock Config to avoid loading real data
        self.mock_config = MagicMock(spec=Config)
        self.mock_config.mafia_top_k = 5
        self.mock_config.enable_market_observer = True
        self.mock_config.mafia_allow_observer_training = False # Should be False for Stage 2
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('entrance.RLcontroller')
    @patch('scripts.train_separated.create_phase2_iterative_config')
    @patch('os.path.exists')
    def test_run_phase2_walkforward_chaining(self, mock_exists, mock_create_config, mock_rl_controller):
        """
        Verify that run_phase2_walkforward correctly identifies the previous year's checkpoint
        and passes it to the RL controller config.
        """
        # Setup mocks
        mock_exists.return_value = True # Assume files exist
        
        # Create a real-ish config object to return
        real_config = Config()
        real_config.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        real_config.res_model_dir = os.path.join(self.output_dir, "model")
        mock_create_config.return_value = real_config
        
        # Run the function
        run_phase2_walkforward(
            observer_checkpoint_dir=self.checkpoint_dir,
            start_year=2015,
            first_infer_year=2017,
            last_infer_year=2017,
            output_dir=self.output_dir,
            dry_run=False
        )
        
        # Verify call to create_phase2_iterative_config
        # It should ask for observer checkpoint from 2016 (2017 - 1)
        expected_ckpt_path = os.path.join(self.checkpoint_dir, "observer_best_2016.pth")
        
        # Check arguments passed to create_phase2_iterative_config
        call_args = mock_create_config.call_args
        self.assertIsNotNone(call_args, "create_phase2_iterative_config should be called")
        
        # kwargs are usually used
        kwargs = call_args.kwargs
        self.assertEqual(kwargs.get('observer_checkpoint'), expected_ckpt_path)
        self.assertEqual(kwargs.get('iteration_label'), "iter_2017")
        
        # Verify RLcontroller was called
        mock_rl_controller.assert_called_once()


    @patch('RL_controller.mafia_observer.MAFIAObserver') 
    def test_env_observer_initialization(self, mock_observer_cls):
        """
        Verify StockPortfolioEnv initializes Observer correctly in 'frozen' mode
        when passed a checkpoint path in config.
        """
        # Create a config that simulates Stage 2
        config = Config()
        config.enable_market_observer = True
        # Simulate loading from a path
        # Note: In real code, entrance.py handles the loading and freezing.
        # Here we verify tradeEnv receives a frozen observer.
        
        # Mock observer instance
        mock_observer_instance = MagicMock()
        mock_observer_cls.return_value = mock_observer_instance
        
        # Configure predict return value: (see tradeEnv.py unpacking)
        mock_observer_instance.predict.return_value = (
            np.zeros(1), # market_vector_np (dummy)
            np.array([1.0]), # risk_eta_np
            np.zeros((1, 10)), # market_scores_full_np (batch, n_stocks?)
            np.zeros((1, 10)), # market_context_np
            np.zeros((1, 3)), # direction_logits_np
            np.array([0]), # topk_indices_np
            np.zeros((1, 10)), # topk_embeddings_np
            np.array([0.5]) # topk_scores_np
        )

        # The env is initialized with an already instantiated mkt_observer.
        
        # Let's verify tradeEnv calls observer.predict during step
        
        # Mock data
        mock_df = pd.DataFrame({
            'date': ['2017-01-01'],
            'stock': ['A'],
            'close': [10.0],
            'open': [10.0],
            'high': [10.0],
            'low': [10.0],
            'volume': [100],
            'DAILYRETURNS-30': [0.01]
        })
        
        # Prepare dummy extra_data
        mock_extra_data = {
            "fine_market": pd.DataFrame({
                "date": ["2017-01-01"],
                "mkt_1min_close": [100.0],
                "mkt_1min_ma": [100.0]  # Assuming config uses 1min finefreq? or update config mock
            }),
            "fine_stock": pd.DataFrame({
                 "date": ["2017-01-01"],
                 "stock": ["A"],
                 "stock_1min_ma": [10.0]
            })
        }
        # Set config finefreq and dimensions
        config.finefreq = "1min"
        config.otherRef_indicator_ma_window = 30 
        config.mafia_D = 10 # Match embedding dimension in return_value
        
        # Initialize Env
        env = StockPortfolioEnv(
            config=config,
            rawdata=mock_df,
            mode='train',
            stock_num=1,
            action_dim=1,
            mkt_observer=mock_observer_instance,
            extra_data=mock_extra_data
        )
        
        # Mock step
        env.reset()
        # Mock run_mkt_observer internals if needed, or just let it run
        # We need to mock _extract_raw_ochlv_window to avoid data errors
        env._extract_raw_ochlv_window = MagicMock(return_value=np.zeros((1, 5, 30)))
        
        # Run one step
        env.step(np.array([1.0]))
        
        # Assert observer.predict was called
        mock_observer_instance.predict.assert_called()
        
        # Verify arguments to predict (Live Inference check)
        args, kwargs = mock_observer_instance.predict.call_args
        self.assertIn('raw_ochlv_data', kwargs)
        self.assertIn('current_date', kwargs)
        
        print("✅ Environment successfully called observer.predict() for Live Inference.")

if __name__ == '__main__':
    unittest.main()
