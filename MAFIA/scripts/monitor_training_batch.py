#!/usr/bin/env python3
"""
Monitor Training Batch - Detect potential issues in gradient flow and model components.

Issues to monitor:
1. Gate weights collapse (all 0.25 uniform)
2. Expert NaN in CSA/TA modules
3. Gradient norm anomalies
"""

import os
import sys
import time
import numpy as np
import torch as th

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAFIA_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MAFIA_DIR)

from config import Config, MafiaTrainMode
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer


def setup_config():
    """Setup minimal config for testing."""
    config = Config()
    config.mafia_train_mode = MafiaTrainMode.SELECTION_ONLY
    config.batch_size = 4
    config.mafia_trajectory_length = 64
    config.mafia_topk_k = 10
    config.mafia_rebalance_interval = 8
    config.use_live_display = False
    config.simple_logging = True
    config.mafia_smart_rotation_enabled = True
    # [SYNC 2026-01-02] Use 10x scaled penalty values (scale mismatch fix)
    config.mafia_pg_alpha_turnover = 3.0
    config.mafia_pg_alpha_change = 3.0
    return config


def check_gate_weights(trainer, threshold=0.05):
    """Check if gate weights are collapsed to uniform."""
    issues = []

    if trainer._latest_gate_weights is not None:
        gw = trainer._latest_gate_weights  # (B, 4)

        # Check for NaN
        if th.isnan(gw).any():
            issues.append("CRITICAL: Gate weights contain NaN!")
            return issues

        # Check for uniform collapse
        uniform = 0.25
        deviation = (gw - uniform).abs().mean().item()

        if deviation < threshold:
            issues.append(f"WARNING: Gate weights near uniform (dev={deviation:.4f} < {threshold})")
        else:
            print(f"   [OK] Gate weights diverse (dev={deviation:.4f})")

        # Show distribution
        mean_gw = gw.mean(dim=0)
        print(f"   Gate means: Tech={mean_gw[0]:.3f}, DC1={mean_gw[1]:.3f}, DC2={mean_gw[2]:.3f}, DC3={mean_gw[3]:.3f}")

    else:
        issues.append("WARNING: No gate weights captured (hook may not be registered)")

    return issues


def check_expert_nan(model):
    """Check for NaN in expert module outputs."""
    issues = []

    # Check CSA modules
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            if th.isnan(module.weight).any():
                issues.append(f"CRITICAL: NaN in {name} weights!")
            if hasattr(module, 'bias') and module.bias is not None:
                if th.isnan(module.bias).any():
                    issues.append(f"CRITICAL: NaN in {name} bias!")

    if not issues:
        print("   [OK] No NaN in model parameters")

    return issues


def check_gradient_norm(model):
    """Check gradient norms for anomalies."""
    issues = []

    total_norm = 0.0
    max_norm = 0.0
    min_norm = float('inf')
    nan_count = 0
    zero_count = 0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_count += 1
            norm = param.grad.data.norm(2).item()

            if np.isnan(norm):
                nan_count += 1
                issues.append(f"CRITICAL: NaN gradient in {name}")
            elif np.isinf(norm):
                issues.append(f"CRITICAL: Inf gradient in {name}")
            else:
                total_norm += norm ** 2
                max_norm = max(max_norm, norm)
                if norm > 0:
                    min_norm = min(min_norm, norm)
                else:
                    zero_count += 1

    total_norm = total_norm ** 0.5

    if nan_count > 0:
        issues.append(f"CRITICAL: {nan_count} parameters have NaN gradients!")

    # Check if gradients were cleared (expected after optimizer.step())
    if param_count == 0:
        print("   [INFO] No gradients available (cleared after optimizer step)")
        return issues

    if zero_count > param_count * 0.5:
        issues.append(f"WARNING: {zero_count}/{param_count} parameters have zero gradients")

    if total_norm > 100:
        issues.append(f"WARNING: Large gradient norm: {total_norm:.2f}")
    elif total_norm < 1e-6 and param_count > 0:
        # Only warn if there ARE gradients but they're tiny
        issues.append(f"WARNING: Vanishing gradient norm: {total_norm:.8f}")
    else:
        print(f"   [OK] Gradient norm: {total_norm:.4f} (max={max_norm:.4f})")

    return issues


def run_monitoring():
    """Run single training batch with monitoring."""
    print("=" * 70)
    print("TRAINING BATCH MONITORING")
    print("=" * 70)

    all_issues = []

    # 1. Setup
    print("\n[1/5] Setting up config and loading data...")
    config = setup_config()

    from utils.mafia_data_loader import load_mafia_data
    data = load_mafia_data(config)

    if isinstance(data, dict):
        train_data = data.get("train")
    else:
        train_data = data

    if train_data is None:
        print("ERROR: Could not load training data!")
        return

    stock_num = train_data["stock"].nunique() if "stock" in train_data.columns else 100
    print(f"   Loaded data: {len(train_data)} rows, {stock_num} stocks")

    # 2. Create model and trainer
    print("\n[2/5] Creating model and trainer...")
    observer = MAFIAObserver(config=config, action_dim=stock_num)
    trainer = ObserverOfflineBatchTrainer(config, observer)
    trainer._epoch = 0

    print(f"   Device: {trainer.device}")
    print(f"   alpha_turnover: {trainer.alpha_turnover}")
    print(f"   alpha_change: {trainer.alpha_change}")

    # 3. Prepare data tensors
    print("\n[3/5] Preparing data tensors...")
    stock_list = train_data["stock"].unique().tolist()
    dates = sorted(train_data["date"].unique())

    split_idx = int(len(dates) * 0.8)
    train_start = dates[0]
    train_end = dates[split_idx - 1]

    import pandas as pd
    train_tensors = trainer.prepare_data_tensors(
        train_data, stock_list, pd.Timestamp(train_start), pd.Timestamp(train_end)
    )
    train_tensors["T_total"] = len(train_tensors["dates"])
    print(f"   Train tensors: T_total={train_tensors['T_total']}")

    # 4. Run single training batch
    print("\n[4/5] Running training batch...")

    # Check expert parameters before training
    print("\n   Checking model parameters before training:")
    issues = check_expert_nan(trainer.observer.mafia_model)
    all_issues.extend(issues)

    # Run training
    start_time = time.time()

    try:
        result = trainer.train_epoch(
            data_tensors=train_tensors,
            steps_per_epoch=1,  # Just 1 batch for monitoring
            epoch=0,
            verbose=True,
            limit_batches=1,
            training_mode="SELECTION_ONLY",
        )
        elapsed = time.time() - start_time
        print(f"   Training completed in {elapsed:.2f}s")
        # Handle both dict and ObserverValidationResult
        if hasattr(result, 'loss'):
            print(f"   Loss: {result.loss}")
            print(f"   Grad Norm: {getattr(result, 'grad_norm', 'N/A')}")
        elif isinstance(result, dict):
            print(f"   Loss: {result.get('loss', 'N/A')}")
            print(f"   Grad Norm: {result.get('grad_norm', 'N/A')}")

    except Exception as e:
        print(f"   ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        all_issues.append(f"CRITICAL: Training failed with {e}")

    # 5. Post-training checks
    print("\n[5/5] Post-training monitoring...")

    # Check gate weights
    print("\n   Checking gate weights:")
    issues = check_gate_weights(trainer)
    all_issues.extend(issues)

    # Check for NaN in model
    print("\n   Checking model parameters after training:")
    issues = check_expert_nan(trainer.observer.mafia_model)
    all_issues.extend(issues)

    # Check gradient norms
    print("\n   Checking gradient norms:")
    issues = check_gradient_norm(trainer.observer.mafia_model)
    all_issues.extend(issues)

    # Summary
    print("\n" + "=" * 70)
    print("MONITORING SUMMARY")
    print("=" * 70)

    if all_issues:
        print(f"\n{len(all_issues)} issues detected:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nAll checks passed!")

    return all_issues


if __name__ == "__main__":
    issues = run_monitoring()

    if issues:
        print(f"\nExit with {len(issues)} issues detected")
        sys.exit(1)
    else:
        print("\nExit successfully")
        sys.exit(0)
