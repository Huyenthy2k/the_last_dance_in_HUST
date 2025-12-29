import unittest
import os
import shutil
import tempfile
import sys
from unittest.mock import patch, MagicMock

# Ensure agents/MAFIA is in path
MAFIA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MAFIA_ROOT not in sys.path:
    sys.path.insert(0, MAFIA_ROOT)

from scripts.train_separated import run_phase2_walkforward, create_phase2_iterative_config

class TestPhase2WalkForward(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.obs_dir = os.path.join(self.test_dir, "observer_ckpts")
        self.out_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.obs_dir)
        os.makedirs(self.out_dir)

        # Create dummy observer checkpoints
        self.start_year = 2018
        self.end_year = 2020
        # Checkpoints needed for (start-1) to (end-1)
        # If infer year is 2018, train end is 2017. checkpt: observer_best_2017.pth
        for y in range(self.start_year - 1, self.end_year): # 2017, 2018, 2019
            path = os.path.join(self.obs_dir, f"observer_best_{y}.pth")
            with open(path, "w") as f:
                f.write("dummy content")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dry_run_loop(self):
        """Verify the loop iterates correctly in dry run mode."""
        print("\n--- Testing Phase 2 Walk-Forward (Dry Run) ---")
        
        with patch("scripts.train_separated.create_phase2_iterative_config") as mock_create:
            mock_create.return_value = MagicMock() # Mock config
            
            run_phase2_walkforward(
                observer_checkpoint_dir=self.obs_dir,
                start_year=2015,
                first_infer_year=self.start_year, # 2018
                last_infer_year=self.end_year,    # 2020
                output_dir=self.out_dir,
                valid_years=1,
                test_years=1,
                dry_run=True,
                seed=42
            )
            
            # Expected iterations: 2018, 2019, 2020 -> 3 calls
            expected_calls = 3
            self.assertEqual(mock_create.call_count, expected_calls)
            
            # Check arguments for last call (2020)
            # Train years should be 2015-2019
            args, kwargs = mock_create.call_args
            self.assertEqual(kwargs['train_years'], [2015, 2016, 2017, 2018, 2019])
            self.assertEqual(kwargs['iteration_label'], "iter_2020")
            
    def test_resume_logic(self):
        """Verify checkpoint passing logic."""
        print("\n--- Testing Phase 2 Walk-Forward (Resume Logic) ---")
        
        def side_effect_rl_controller(config):
            # Simulate training artifact creation
            print(f"Mock training for config: {config.res_dir}")
            os.makedirs(config.res_model_dir, exist_ok=True)
            # Create dummy best model
            with open(os.path.join(config.res_model_dir, "best_model.zip"), "w") as f:
                f.write("dummy model content")
        
        # Patch entrance module
        mock_entrance = MagicMock()
        mock_entrance.RLcontroller.side_effect = side_effect_rl_controller
        
        with patch.dict(sys.modules, {"entrance": mock_entrance}):
             run_phase2_walkforward(
                observer_checkpoint_dir=self.obs_dir,
                start_year=2015,
                first_infer_year=self.start_year, # 2018
                last_infer_year=self.start_year + 1,    # 2019
                output_dir=self.out_dir,
                valid_years=1,
                test_years=1,
                dry_run=False,
                seed=42
            )
             
             # 1. Check if files created
             iter_2018_dir = os.path.join(self.out_dir, "iter_2018")
             iter_2019_dir = os.path.join(self.out_dir, "iter_2019")
             
             self.assertTrue(os.path.exists(os.path.join(iter_2018_dir, "model", "best_model.zip")))
             self.assertTrue(os.path.exists(os.path.join(iter_2019_dir, "model", "best_model.zip")))
             
             # 2. Check resume logic?
             # We can't easily check resume unless we mock create_phase2_iterative_config too and inspect calls
             # But the fact it ran without crashing suggests basic flow is OK.
             # Ideally we'd verify that Iter 2019 config received Iter 2018 checkpoint.
             pass

    def test_resume_parameter_passing(self):
        """Ensure resume_checkpoint is passed to subsequent iterations."""
        print("\n--- Testing Resume Parameter Passing ---")
        
        # We need to control output files and inspect config creation
        
        # 1. Mock RLcontroller to create file
        mock_entrance = MagicMock()
        def side_effect(config):
            os.makedirs(config.res_model_dir, exist_ok=True)
            with open(os.path.join(config.res_model_dir, "best_model.zip"), "w") as f:
                f.write("data")
        mock_entrance.RLcontroller.side_effect = side_effect
        
        # 2. Spy on create_phase2_iterative_config
        # We want to WRAP the real function so it does real logic but we can see args
        real_create_config = create_phase2_iterative_config
        
        with patch("scripts.train_separated.create_phase2_iterative_config", side_effect=real_create_config) as mock_create:
            with patch.dict(sys.modules, {"entrance": mock_entrance}):
                 run_phase2_walkforward(
                    observer_checkpoint_dir=self.obs_dir,
                    start_year=2015,
                    first_infer_year=2018,
                    last_infer_year=2019,
                    output_dir=self.out_dir,
                    valid_years=1,
                    test_years=1,
                    dry_run=False,
                    seed=42
                )
                
            # Check calls
            # First call (2018): resume_checkpoint=None
            args1, kwargs1 = mock_create.call_args_list[0]
            self.assertIsNone(kwargs1.get('resume_checkpoint'))
            
            # Second call (2019): resume_checkpoint should be path to 2018 best model
            args2, kwargs2 = mock_create.call_args_list[1]
            resume_ckpt = kwargs2.get('resume_checkpoint')
            self.assertIsNotNone(resume_ckpt)
            self.assertIn("iter_2018", resume_ckpt)
            self.assertIn("best_model.zip", resume_ckpt)
            
            print("Verified resume_checkpoint passed to 2nd iteration:", resume_ckpt)

    def test_walkforward_info_in_config(self):
        """Verify walk-forward metadata is correctly populated in config."""
        print("\n--- Testing Walk-Forward Info in Config ---")
        
        # Test creation of a single config
        config = create_phase2_iterative_config(
            observer_checkpoint=os.path.join(self.obs_dir, "observer_best_2017.pth"),
            train_years=[2015, 2016, 2017],
            valid_years=1,
            test_years=1,
            iteration_label="iter_2018",
            output_basedir=self.out_dir,
            resume_checkpoint="dummy_path.zip",
            iteration_index=0,
            total_iterations=5
        )
        
        # 1. Check if attribute exists
        self.assertTrue(hasattr(config, "walkforward_info"))
        
        # 2. Check content
        info = config.walkforward_info
        print(f"Info generated: {info}")
        
        self.assertEqual(info["iteration"], 0)
        self.assertEqual(info["total_iterations"], 5)
        self.assertEqual(info["train_years"], "2015→2017")
        self.assertEqual(info["valid_year"], 2018)
        self.assertEqual(info["infer_year"], 2019) # 2018 is valid, so test is 2019?
        # Note: create_phase2_iterative_config logic:
        # train_years gives train_end. 
        # valid starts at train_end + 1 day (so effectively next year unless dates align awkwardly)
        # Actually config logic:
        # train_end = max(train_years) -> 2017
        # valid_start = 2018
        # test_start = 2018 + 1 = 2019
        
        self.assertTrue(info["is_finetune"])
        self.assertIn("train_range", info)
        self.assertIn("valid_range", info)
        self.assertIn("infer_range", info)


if __name__ == "__main__":
    unittest.main()
