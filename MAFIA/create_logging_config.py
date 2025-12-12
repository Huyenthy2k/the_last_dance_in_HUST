#!/usr/bin/env python3
"""
Quick test script with trajectory logging enabled.
"""
import sys
sys.path.insert(0, '/Users/nguyensiry/Documents/the_last_dance/agents/MAFIA')

from config import Config

# Create config with trajectory logging enabled
config = Config()
config.log_trajectory_details = True
config.mafia_trajectory_length = 40  # Shorter for demo
config.mafia_batch_size = 4  # Fewer batches for demo
config.res_root = "./observer_logging_demo"

# Save to ensure it's pickled correctly
import pickle
with open('temp_config_logging.pkl', 'wb') as f:
    pickle.dump(config, f)

print("Config created with trajectory logging enabled")
print(f"  log_trajectory_details: {config.log_trajectory_details}")
print(f"  trajectory_length: {config.mafia_trajectory_length}")
print(f"  batch_size: {config.mafia_batch_size}")
print(f"  output_dir: {config.res_root}")
