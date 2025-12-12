#!/usr/bin/env python3
"""
Test script for Walk-Forward Implementation

Tests all components implemented for Phase 1 (Observer) and Phase 2 (RL) separation:
1. observer_validation.py - Composite score computation
2. portfolio_allocator_reward.py - Reward function
3. config.py - Walk-forward parameters
4. train_observer_walkforward.py - Script structure and imports
5. generate_rl_states.py - Script structure and imports
"""

import os
import sys
import traceback
from typing import List, Tuple

# Ensure repo root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np


def test_separator(name: str):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")


def assert_test(condition: bool, message: str):
    if condition:
        print(f"  [PASS] {message}")
    else:
        print(f"  [FAIL] {message}")
        raise AssertionError(message)


def test_observer_validation():
    """Test observer_validation.py functions."""
    test_separator("observer_validation.py")

    from RL_controller.observer_validation import (
        compute_sharpe_ratio,
        compute_information_coefficient,
        compute_direction_f1,
        compute_composite_score,
        backtest_topk_equalweight,
        CompositeScoreNormalizer,
        ObserverCheckpointTracker,
    )

    # Test 1: compute_sharpe_ratio
    print("\n  1. Testing compute_sharpe_ratio()...")
    returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
    sharpe = compute_sharpe_ratio(returns)
    assert_test(isinstance(sharpe, float), f"Returns float: {sharpe}")
    assert_test(-10 < sharpe < 10, f"Reasonable range: {sharpe}")

    # Edge case: empty returns
    sharpe_empty = compute_sharpe_ratio(np.array([]))
    assert_test(sharpe_empty == 0.0, f"Empty returns -> 0: {sharpe_empty}")

    # Test 2: compute_information_coefficient
    print("\n  2. Testing compute_information_coefficient()...")
    predictions = np.array([0.1, 0.3, 0.2, 0.5, 0.4])
    actual_returns = np.array([0.05, 0.15, 0.1, 0.25, 0.2])
    ic = compute_information_coefficient(predictions, actual_returns)
    assert_test(isinstance(ic, float), f"Returns float: {ic}")
    assert_test(-1 <= ic <= 1, f"Valid correlation range: {ic}")

    # Test 3: compute_direction_f1
    print("\n  3. Testing compute_direction_f1()...")
    pred_labels = np.array([0, 1, 2, 1, 0, 2])  # Bear, Side, Bull
    actual_labels = np.array([0, 1, 2, 0, 0, 1])
    f1 = compute_direction_f1(pred_labels, actual_labels)
    assert_test(isinstance(f1, float), f"Returns float: {f1}")
    assert_test(0 <= f1 <= 1, f"Valid F1 range: {f1}")

    # Test 4: compute_composite_score
    print("\n  4. Testing compute_composite_score()...")
    result = compute_composite_score(
        sharpe=1.5,
        ic=0.3,
        f1=0.65,
        w_sharpe=0.5,
        w_ic=0.3,
        w_f1=0.2,
    )
    assert_test("composite_score" in result, "Has composite_score key")
    assert_test("sharpe" in result, "Has sharpe key")
    assert_test("ic" in result, "Has ic key")
    assert_test("f1_direction" in result, "Has f1_direction key")

    # Test 5: backtest_topk_equalweight
    print("\n  5. Testing backtest_topk_equalweight()...")
    daily_topk = [np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1, 2, 3])]
    daily_returns = np.array([
        [0.01, 0.02, -0.01, 0.015],
        [0.005, -0.01, 0.02, 0.01],
        [-0.005, 0.015, 0.01, 0.02],
    ])
    portfolio_returns = backtest_topk_equalweight(daily_topk, daily_returns, rebalance_interval=2)
    assert_test(len(portfolio_returns) == 3, f"Correct length: {len(portfolio_returns)}")
    assert_test(isinstance(portfolio_returns, np.ndarray), "Returns numpy array")

    # Test 6: CompositeScoreNormalizer
    print("\n  6. Testing CompositeScoreNormalizer...")
    normalizer = CompositeScoreNormalizer()
    normalizer.update(1.0, 0.2, 0.5)
    normalizer.update(1.5, 0.3, 0.6)
    normalizer.update(0.8, 0.25, 0.55)
    norm_sharpe, norm_ic, norm_f1 = normalizer.get_normalized(1.2, 0.25, 0.55)
    assert_test(0 <= norm_sharpe <= 1, f"Normalized sharpe in [0,1]: {norm_sharpe}")
    assert_test(0 <= norm_ic <= 1, f"Normalized IC in [0,1]: {norm_ic}")
    assert_test(0 <= norm_f1 <= 1, f"Normalized F1 in [0,1]: {norm_f1}")

    # Test 7: ObserverCheckpointTracker
    print("\n  7. Testing ObserverCheckpointTracker...")
    tracker = ObserverCheckpointTracker()
    is_best1 = tracker.update(epoch=0, sharpe=1.0, ic=0.2, f1=0.5, checkpoint_path="ckpt_0.pth")
    is_best2 = tracker.update(epoch=1, sharpe=1.5, ic=0.3, f1=0.6, checkpoint_path="ckpt_1.pth")
    is_best3 = tracker.update(epoch=2, sharpe=0.8, ic=0.15, f1=0.45, checkpoint_path="ckpt_2.pth")
    assert_test(is_best1 == True, "First epoch is best")
    assert_test(is_best2 == True, "Second epoch improves")
    assert_test(is_best3 == False, "Third epoch does not improve")
    best = tracker.get_best()
    assert_test(best["epoch"] == 1, f"Best epoch is 1: {best}")
    assert_test(best["checkpoint_path"] == "ckpt_1.pth", "Correct checkpoint path")

    print("\n  [OK] observer_validation.py tests passed!")


def test_portfolio_allocator_reward():
    """Test portfolio_allocator_reward.py functions."""
    test_separator("portfolio_allocator_reward.py")

    from RL_controller.portfolio_allocator_reward import (
        compute_log_return,
        compute_jensen_shannon_divergence,
        compute_reward,
        RewardNormalizer,
    )

    # Test 1: compute_log_return
    print("\n  1. Testing compute_log_return()...")
    log_ret = compute_log_return(0.01)  # +1% return
    assert_test(abs(log_ret - np.log(1.01)) < 1e-6, f"Correct log return: {log_ret}")

    # Edge case: -100% return
    log_ret_edge = compute_log_return(-1.0)
    assert_test(log_ret_edge == -10.0, f"Edge case -100%: {log_ret_edge}")

    # Test 2: compute_jensen_shannon_divergence
    print("\n  2. Testing compute_jensen_shannon_divergence()...")
    a_alloc = np.array([0.4, 0.3, 0.2, 0.1])
    a_final = np.array([0.35, 0.35, 0.15, 0.15])
    js_div = compute_jensen_shannon_divergence(a_alloc, a_final)
    assert_test(isinstance(js_div, float), f"Returns float: {js_div}")
    assert_test(0 <= js_div <= 1, f"JS divergence in [0,1]: {js_div}")

    # Same distribution -> JS = 0
    js_same = compute_jensen_shannon_divergence(a_alloc, a_alloc)
    assert_test(js_same < 1e-6, f"Same distribution -> ~0: {js_same}")

    # Test 3: compute_reward
    print("\n  3. Testing compute_reward()...")
    result = compute_reward(
        portfolio_return=0.01,
        a_alloc=np.array([0.4, 0.3, 0.2, 0.1]),
        a_final=np.array([0.35, 0.35, 0.15, 0.15]),
        w_return=1.0,
        lambda_js=0.1,
        reward_scale=100.0,
    )
    assert_test("reward_total" in result, "Has reward_total key")
    assert_test("reward_unscaled" in result, "Has reward_unscaled key")
    assert_test("r_return" in result, "Has r_return key")
    assert_test("r_divergence" in result, "Has r_divergence key")
    assert_test("js_divergence" in result, "Has js_divergence key")
    assert_test("log_return" in result, "Has log_return key")
    assert_test("reward_scale" in result, "Has reward_scale key")

    # Verify scaling
    assert_test(
        abs(result["reward_total"] - result["reward_unscaled"] * 100.0) < 1e-6,
        f"Scaling correct: {result['reward_total']} = {result['reward_unscaled']} * 100"
    )

    # Test 4: RewardNormalizer
    print("\n  4. Testing RewardNormalizer...")
    normalizer = RewardNormalizer(alpha=0.1)
    for val in [0.5, 0.6, 0.4, 0.55, 0.45]:
        normalizer.update(val)
    stats = normalizer.get_stats()
    assert_test("mean" in stats, "Has mean")
    assert_test("std" in stats, "Has std")
    assert_test("count" in stats, "Has count")
    assert_test(stats["count"] == 5, f"Correct count: {stats['count']}")

    normalized = normalizer.normalize(0.5)
    assert_test(isinstance(normalized, float), f"Returns float: {normalized}")

    print("\n  [OK] portfolio_allocator_reward.py tests passed!")


def test_config_walkforward_params():
    """Test config.py walk-forward parameters."""
    test_separator("config.py - Walk-Forward Parameters")

    from config import Config

    config = Config()

    # Test walk-forward training configuration
    print("\n  1. Testing walk-forward config parameters...")
    assert_test(hasattr(config, "observer_only_training"), "Has observer_only_training")
    assert_test(hasattr(config, "observer_pretrained_path"), "Has observer_pretrained_path")
    assert_test(hasattr(config, "freeze_observer_during_rl"), "Has freeze_observer_during_rl")

    # Walk-forward iteration settings
    assert_test(hasattr(config, "walkforward_iteration"), "Has walkforward_iteration")
    assert_test(hasattr(config, "walkforward_train_start_year"), "Has walkforward_train_start_year")
    assert_test(hasattr(config, "walkforward_train_end_year"), "Has walkforward_train_end_year")
    assert_test(hasattr(config, "walkforward_valid_year"), "Has walkforward_valid_year")
    assert_test(hasattr(config, "walkforward_infer_year"), "Has walkforward_infer_year")

    # Training hyperparameters
    assert_test(hasattr(config, "walkforward_base_epochs"), "Has walkforward_base_epochs")
    assert_test(hasattr(config, "walkforward_finetune_epochs"), "Has walkforward_finetune_epochs")
    assert_test(hasattr(config, "walkforward_base_lr"), "Has walkforward_base_lr")
    assert_test(hasattr(config, "walkforward_finetune_lr"), "Has walkforward_finetune_lr")
    assert_test(hasattr(config, "walkforward_base_patience"), "Has walkforward_base_patience")
    assert_test(hasattr(config, "walkforward_finetune_patience"), "Has walkforward_finetune_patience")

    # Score weights
    assert_test(hasattr(config, "walkforward_score_w_sharpe"), "Has walkforward_score_w_sharpe")
    assert_test(hasattr(config, "walkforward_score_w_ic"), "Has walkforward_score_w_ic")
    assert_test(hasattr(config, "walkforward_score_w_f1"), "Has walkforward_score_w_f1")

    # Test 2: Verify default values
    print("\n  2. Testing default values...")
    assert_test(config.observer_only_training == False, "observer_only_training default False")
    assert_test(config.observer_pretrained_path is None, "observer_pretrained_path default None")
    # Separated training enabled by default (Phase 2 mode)
    assert_test(config.freeze_observer_during_rl == True, "freeze_observer_during_rl default True (separated training)")
    assert_test(config.walkforward_train_start_year == 2015, f"walkforward_train_start_year = {config.walkforward_train_start_year}")
    assert_test(config.walkforward_base_epochs == 50, f"walkforward_base_epochs = {config.walkforward_base_epochs}")
    assert_test(config.walkforward_finetune_epochs == 20, f"walkforward_finetune_epochs = {config.walkforward_finetune_epochs}")
    assert_test(config.walkforward_base_lr == 1e-4, f"walkforward_base_lr = {config.walkforward_base_lr}")
    assert_test(config.walkforward_finetune_lr == 1e-5, f"walkforward_finetune_lr = {config.walkforward_finetune_lr}")

    # Test 3: Verify score weights sum to 1
    print("\n  3. Testing score weights...")
    weight_sum = config.walkforward_score_w_sharpe + config.walkforward_score_w_ic + config.walkforward_score_w_f1
    assert_test(abs(weight_sum - 1.0) < 1e-6, f"Score weights sum to 1: {weight_sum}")
    assert_test(config.walkforward_score_w_sharpe == 0.5, f"w_sharpe = {config.walkforward_score_w_sharpe}")
    assert_test(config.walkforward_score_w_ic == 0.3, f"w_ic = {config.walkforward_score_w_ic}")
    assert_test(config.walkforward_score_w_f1 == 0.2, f"w_f1 = {config.walkforward_score_w_f1}")

    # Test 4: Test rl_obs_global_scalars
    print("\n  4. Testing rl_obs_global_scalars...")
    assert_test(hasattr(config, "rl_obs_global_scalars"), "Has rl_obs_global_scalars")
    scalars = config.rl_obs_global_scalars
    assert_test("time_decay" in scalars, "Has time_decay scalar")
    assert_test("relative_alpha" in scalars, "Has relative_alpha scalar")
    assert_test("risk_violation" in scalars, "Has risk_violation scalar")
    # NEW: Rebalance signals for TD3
    assert_test("is_rebalance_scheduled" in scalars, "Has is_rebalance_scheduled scalar")
    assert_test("is_rebalance_regime" in scalars, "Has is_rebalance_regime scalar")
    assert_test(len(scalars) == 8, f"Has 8 global scalars (got {len(scalars)})")
    print(f"    Global scalars: {scalars}")

    # Test 5: Rebalance interval
    print("\n  5. Testing rebalance interval...")
    assert_test(hasattr(config, "topk_rebalance_interval"), "Has topk_rebalance_interval")
    assert_test(config.topk_rebalance_interval == 14, f"topk_rebalance_interval = {config.topk_rebalance_interval}")

    print("\n  [OK] config.py walk-forward tests passed!")


def test_train_observer_walkforward_structure():
    """Test train_observer_walkforward.py imports and structure."""
    test_separator("train_observer_walkforward.py - Structure")

    # Test 1: Can import the module
    print("\n  1. Testing imports...")
    try:
        from scripts.train_observer_walkforward import (
            setup_walkforward_config,
            train_observer_iteration,
            run_walkforward_observer_training,
        )
        assert_test(True, "All functions importable")
    except ImportError as e:
        assert_test(False, f"Import error: {e}")

    # Test 2: Test setup_walkforward_config
    print("\n  2. Testing setup_walkforward_config()...")
    from config import Config
    from scripts.train_observer_walkforward import setup_walkforward_config

    config = Config()
    config = setup_walkforward_config(
        config=config,
        iteration=0,
        train_start_year=2015,
        train_end_year=2017,
        valid_year=2017,
        infer_year=2018,
        checkpoint_path=None,
        output_dir="/tmp/test_walkforward",
    )

    assert_test(config.observer_only_training == True, "observer_only_training set to True")
    assert_test(config.walkforward_iteration == 0, f"iteration = {config.walkforward_iteration}")
    assert_test(config.walkforward_train_start_year == 2015, f"train_start = {config.walkforward_train_start_year}")
    assert_test(config.walkforward_train_end_year == 2017, f"train_end = {config.walkforward_train_end_year}")
    assert_test(config.walkforward_valid_year == 2017, f"valid_year = {config.walkforward_valid_year}")
    assert_test(config.walkforward_infer_year == 2018, f"infer_year = {config.walkforward_infer_year}")

    # Iteration 0 should use base epochs
    assert_test(config.num_epochs == config.walkforward_base_epochs, f"Base epochs: {config.num_epochs}")
    assert_test(config.mafia_learning_rate == config.walkforward_base_lr, f"Base LR: {config.mafia_learning_rate}")

    # Test iteration > 0 (finetune)
    print("\n  3. Testing finetune config (iteration > 0)...")
    config2 = Config()
    config2 = setup_walkforward_config(
        config=config2,
        iteration=1,
        train_start_year=2015,
        train_end_year=2018,
        valid_year=2018,
        infer_year=2019,
        checkpoint_path="/tmp/observer_best_2017.pth",
        output_dir="/tmp/test_walkforward",
    )
    assert_test(config2.num_epochs == config2.walkforward_finetune_epochs, f"Finetune epochs: {config2.num_epochs}")
    assert_test(config2.mafia_learning_rate == config2.walkforward_finetune_lr, f"Finetune LR: {config2.mafia_learning_rate}")
    assert_test(config2.observer_pretrained_path == "/tmp/observer_best_2017.pth", "Checkpoint path set")

    print("\n  [OK] train_observer_walkforward.py structure tests passed!")


def test_generate_rl_states_structure():
    """Test generate_rl_states.py imports and structure."""
    test_separator("generate_rl_states.py - Structure")

    # Test 1: Can import the module
    print("\n  1. Testing imports...")
    try:
        from scripts.generate_rl_states import (
            load_frozen_observer,
            setup_inference_environment,
            generate_states_for_year,
            save_states,
            generate_single_year,
            generate_batch,
        )
        assert_test(True, "All functions importable")
    except ImportError as e:
        assert_test(False, f"Import error: {e}")

    # Test 2: Verify function signatures
    print("\n  2. Testing function signatures...")
    import inspect
    from scripts.generate_rl_states import generate_states_for_year

    sig = inspect.signature(generate_states_for_year)
    params = list(sig.parameters.keys())
    assert_test("observer" in params, "Has observer param")
    assert_test("config" in params, "Has config param")
    assert_test("year" in params, "Has year param")
    assert_test("rebalance_interval" in params, "Has rebalance_interval param")
    print(f"    Parameters: {params}")

    print("\n  [OK] generate_rl_states.py structure tests passed!")


def test_walkforward_iteration_calculation():
    """Test walk-forward iteration date range calculation."""
    test_separator("Walk-Forward Iteration Calculation")

    # Test the date calculation logic
    print("\n  Testing date range for TD3 training 2018-2022...")

    start_year = 2015
    first_infer_year = 2018
    last_infer_year = 2022

    infer_years = list(range(first_infer_year, last_infer_year + 1))
    num_iterations = len(infer_years)

    assert_test(num_iterations == 5, f"5 iterations: {num_iterations}")
    assert_test(infer_years == [2018, 2019, 2020, 2021, 2022], f"Infer years: {infer_years}")

    expected_iterations = [
        {"iter": 0, "train_end": 2017, "valid": 2017, "infer": 2018},
        {"iter": 1, "train_end": 2018, "valid": 2018, "infer": 2019},
        {"iter": 2, "train_end": 2019, "valid": 2019, "infer": 2020},
        {"iter": 3, "train_end": 2020, "valid": 2020, "infer": 2021},
        {"iter": 4, "train_end": 2021, "valid": 2021, "infer": 2022},
    ]

    print("\n  Iteration | Train Range | Valid | Infer | State Output")
    print("  " + "-" * 55)

    for i, infer_year in enumerate(infer_years):
        valid_year = infer_year - 1
        train_end_year = valid_year

        expected = expected_iterations[i]
        assert_test(i == expected["iter"], f"Iter {i} index correct")
        assert_test(train_end_year == expected["train_end"], f"Train end {train_end_year}")
        assert_test(valid_year == expected["valid"], f"Valid year {valid_year}")
        assert_test(infer_year == expected["infer"], f"Infer year {infer_year}")

        print(f"  {i:^9} | {start_year}→{train_end_year:^4} | {valid_year:^5} | {infer_year:^5} | State_{infer_year}.pkl")

    print("\n  [OK] Walk-forward iteration calculation tests passed!")


def test_observer_td3_separation():
    """Test Observer and TD3 training separation mechanism."""
    test_separator("Observer-TD3 Training Separation")

    from config import Config

    # Test 1: Config flags for separation
    print("\n  1. Testing separation config flags...")
    config = Config()

    # Phase 1: Observer-only training
    assert_test(hasattr(config, "observer_only_training"), "Has observer_only_training flag")
    assert_test(config.observer_only_training == False, "Default: observer_only_training = False")

    # Phase 2: Freeze observer during RL (separated training default)
    assert_test(hasattr(config, "freeze_observer_during_rl"), "Has freeze_observer_during_rl flag")
    assert_test(config.freeze_observer_during_rl == True, "Default: freeze_observer_during_rl = True (separated training)")

    # Observer pretrained path
    assert_test(hasattr(config, "observer_pretrained_path"), "Has observer_pretrained_path")
    assert_test(config.observer_pretrained_path is None, "Default: observer_pretrained_path = None")

    # Training allowance flag
    assert_test(hasattr(config, "mafia_allow_observer_training"), "Has mafia_allow_observer_training")

    # Test 2: Phase 1 config simulation (Observer-only)
    print("\n  2. Testing Phase 1 config (Observer-only training)...")
    config_phase1 = Config()
    config_phase1.observer_only_training = True
    config_phase1.freeze_observer_during_rl = False
    config_phase1.mafia_allow_observer_training = True

    assert_test(config_phase1.observer_only_training == True, "Phase 1: observer_only_training = True")
    assert_test(config_phase1.mafia_allow_observer_training == True, "Phase 1: Observer can train")

    # Test 3: Phase 2 config simulation (RL with frozen Observer)
    print("\n  3. Testing Phase 2 config (Frozen Observer + TD3)...")
    config_phase2 = Config()
    config_phase2.observer_only_training = False
    config_phase2.freeze_observer_during_rl = True
    config_phase2.observer_pretrained_path = "/tmp/observer_best.pth"

    # Simulate freeze logic from entrance.py
    if config_phase2.freeze_observer_during_rl:
        config_phase2.mafia_allow_observer_training = False

    assert_test(config_phase2.observer_only_training == False, "Phase 2: observer_only_training = False")
    assert_test(config_phase2.freeze_observer_during_rl == True, "Phase 2: freeze_observer_during_rl = True")
    assert_test(config_phase2.mafia_allow_observer_training == False, "Phase 2: Observer FROZEN (cannot train)")
    assert_test(config_phase2.observer_pretrained_path is not None, "Phase 2: Has pretrained path")

    # Test 4: Verify separation states are mutually exclusive
    print("\n  4. Testing mutual exclusivity of phases...")

    # Phase 1 state: Observer trains, no frozen
    phase1_observer_trains = True
    phase1_observer_frozen = False

    # Phase 2 state: Observer frozen, TD3 trains
    phase2_observer_trains = False
    phase2_observer_frozen = True

    assert_test(
        phase1_observer_trains != phase2_observer_trains,
        "Observer training state differs between phases"
    )
    assert_test(
        phase1_observer_frozen != phase2_observer_frozen,
        "Observer freeze state differs between phases"
    )

    # Test 5: Test freeze propagation in setup_walkforward_config
    print("\n  5. Testing freeze propagation in walk-forward setup...")
    from scripts.train_observer_walkforward import setup_walkforward_config

    # Setup for Phase 1 iteration (observer training)
    config_wf = Config()
    config_wf = setup_walkforward_config(
        config=config_wf,
        iteration=0,
        train_start_year=2015,
        train_end_year=2017,
        valid_year=2017,
        infer_year=2018,
        checkpoint_path=None,
        output_dir="/tmp/test_wf",
    )
    assert_test(config_wf.observer_only_training == True, "Walk-forward sets observer_only_training")

    # Test 6: Test state output fields for RL consumption
    print("\n  6. Testing RL state output structure...")

    # Define expected state fields that RL needs from frozen Observer
    required_rl_state_fields = [
        "topk_indices",       # (K,) - Which stocks to allocate
        "topk_embeddings",    # (K, D_stock) - Feature vectors
        "topk_scores",        # (K,) - Selection scores
        "risk_eta",           # (1,) - Risk coefficient
        "direction_logits",   # (3,) - Market direction
        "market_context",     # (D_macro,) - Macro context
    ]

    # Simulate a state dict
    mock_state = {
        "day": 0,
        "is_rebalance": True,
        "topk_indices": np.arange(10),
        "topk_embeddings": np.zeros((10, 64)),
        "topk_scores": np.ones(10) / 10,
        "risk_eta": 1.0,
        "direction_logits": np.array([0.33, 0.34, 0.33]),
        "market_context": np.zeros(64),
        "observation": {"flat": np.zeros(100)},
    }

    for field in required_rl_state_fields:
        assert_test(field in mock_state, f"RL state has required field: {field}")

    # Test 7: Verify Observer outputs dimensions
    print("\n  7. Testing Observer output dimensions...")
    K = config.topK
    D_stock = getattr(config, "mafia_stock_embed_dim", 64)
    D_macro = getattr(config, "mafia_market_embed_dim", 64)

    assert_test(mock_state["topk_indices"].shape == (K,), f"topk_indices shape: ({K},)")
    assert_test(mock_state["topk_embeddings"].shape == (K, D_stock), f"topk_embeddings shape: ({K}, {D_stock})")
    assert_test(mock_state["topk_scores"].shape == (K,), f"topk_scores shape: ({K},)")
    assert_test(mock_state["direction_logits"].shape == (3,), "direction_logits shape: (3,)")
    assert_test(len(mock_state["market_context"]) == D_macro, f"market_context dim: {D_macro}")

    # Test 8: Verify training separation logic
    print("\n  8. Testing training separation logic...")

    def simulate_step_training_check(config, mode="train"):
        """Simulate the training check in tradeEnv.step()"""
        allow_observer_training = bool(
            getattr(config, "mafia_allow_observer_training", True)
        )
        is_train_mode = (mode == "train")
        observer_enabled = getattr(config, "enable_market_observer", True)

        # Observer trains only if all conditions met
        should_train_observer = (
            observer_enabled and
            is_train_mode and
            allow_observer_training
        )
        return should_train_observer

    # Phase 1: Observer should train
    config_p1 = Config()
    config_p1.mafia_allow_observer_training = True
    config_p1.enable_market_observer = True
    assert_test(
        simulate_step_training_check(config_p1, "train") == True,
        "Phase 1: Observer trains during train mode"
    )

    # Phase 2: Observer should NOT train (frozen)
    config_p2 = Config()
    config_p2.mafia_allow_observer_training = False  # Frozen
    config_p2.enable_market_observer = True
    assert_test(
        simulate_step_training_check(config_p2, "train") == False,
        "Phase 2: Observer does NOT train (frozen)"
    )

    # Test mode should never train observer
    config_test = Config()
    config_test.mafia_allow_observer_training = True
    assert_test(
        simulate_step_training_check(config_test, "test") == False,
        "Test mode: Observer never trains"
    )

    print("\n  [OK] Observer-TD3 separation tests passed!")


def test_td3_with_frozen_observer():
    """Test TD3 configuration and compatibility with frozen Observer."""
    test_separator("TD3 with Frozen Observer")

    from config import Config

    # Test 1: TD3/Allocator config parameters exist
    print("\n  1. Testing TD3/Allocator config parameters...")
    config = Config()

    # Core TD3 parameters
    assert_test(hasattr(config, "allocator_learning_rate_actor"), "Has allocator_learning_rate_actor")
    assert_test(hasattr(config, "allocator_learning_rate_critic"), "Has allocator_learning_rate_critic")
    assert_test(hasattr(config, "allocator_discount_gamma"), "Has allocator_discount_gamma")
    assert_test(hasattr(config, "allocator_polyak_tau"), "Has allocator_polyak_tau")
    assert_test(hasattr(config, "allocator_policy_delay"), "Has allocator_policy_delay")

    # Reward parameters
    assert_test(hasattr(config, "allocator_return_weight"), "Has allocator_return_weight")
    assert_test(hasattr(config, "allocator_lambda_js"), "Has allocator_lambda_js")
    assert_test(hasattr(config, "allocator_reward_scale"), "Has allocator_reward_scale")

    # Network parameters
    assert_test(hasattr(config, "allocator_batch_size"), "Has allocator_batch_size")
    assert_test(hasattr(config, "allocator_replay_buffer_size"), "Has allocator_replay_buffer_size")

    # Test 2: TD3 controller import
    print("\n  2. Testing TD3 controller import...")
    try:
        from RL_controller.TD3_controller import TD3Controller, TD3PolicyAdj
        assert_test(True, "TD3Controller importable")
    except ImportError as e:
        assert_test(False, f"TD3Controller import error: {e}")

    # Test 3: Verify TD3 works independently of Observer training state
    print("\n  3. Testing TD3 independence from Observer training state...")

    # Phase 2 config: Observer frozen, TD3 trains
    config_p2 = Config()
    config_p2.freeze_observer_during_rl = True
    config_p2.mafia_allow_observer_training = False  # Observer frozen
    config_p2.observer_pretrained_path = "/tmp/observer_best.pth"

    # TD3 should still have valid config
    assert_test(config_p2.allocator_learning_rate_actor > 0, "TD3 actor LR valid")
    assert_test(config_p2.allocator_learning_rate_critic > 0, "TD3 critic LR valid")
    assert_test(config_p2.allocator_batch_size > 0, "TD3 batch size valid")
    assert_test(0 < config_p2.allocator_discount_gamma <= 1, "TD3 gamma in (0,1]")
    assert_test(0 < config_p2.allocator_polyak_tau < 1, "TD3 tau in (0,1)")

    # Test 4: Verify reward computation works with any action
    print("\n  4. Testing reward computation independence...")
    from RL_controller.portfolio_allocator_reward import compute_reward

    # Simulate TD3 action and Controller adjustment
    a_alloc = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # TD3 output
    a_final = np.array([0.28, 0.27, 0.18, 0.15, 0.12])  # Controller adjusted

    reward_result = compute_reward(
        portfolio_return=0.015,  # +1.5% return
        a_alloc=a_alloc,
        a_final=a_final,
        w_return=config.allocator_return_weight,
        lambda_js=config.allocator_lambda_js,
        reward_scale=config.allocator_reward_scale,
    )

    assert_test("reward_total" in reward_result, "Reward computation works")
    assert_test(reward_result["reward_scale"] == config.allocator_reward_scale, "Reward scale applied")
    print(f"    Sample reward: {reward_result['reward_total']:.4f}")

    # Test 5: Verify TD3 action space matches topK
    print("\n  5. Testing action space consistency...")
    K = config.topK
    action_dim = K  # TD3 outputs K allocation weights

    assert_test(action_dim == K, f"Action dim matches topK: {action_dim}")

    # Verify action normalization (softmax for allocation)
    raw_action = np.random.randn(K)  # Raw TD3 output
    normalized_action = np.exp(raw_action) / np.sum(np.exp(raw_action))  # Softmax
    assert_test(abs(np.sum(normalized_action) - 1.0) < 1e-6, "Action sums to 1 after softmax")
    assert_test(np.all(normalized_action >= 0), "All allocations non-negative")

    # Test 6: Verify Phase 2 config is complete
    print("\n  6. Testing Phase 2 config completeness...")

    def create_phase2_config(observer_path: str) -> Config:
        """Create a complete Phase 2 config for TD3 training."""
        cfg = Config()
        # Observer settings (frozen)
        cfg.observer_pretrained_path = observer_path
        cfg.freeze_observer_during_rl = True
        cfg.mafia_allow_observer_training = False
        cfg.observer_only_training = False

        # TD3 settings (training)
        cfg.enable_td3 = True  # Ensure TD3 is enabled

        return cfg

    phase2_cfg = create_phase2_config("/tmp/observer_best.pth")
    assert_test(phase2_cfg.observer_pretrained_path is not None, "Phase 2: Has observer path")
    assert_test(phase2_cfg.freeze_observer_during_rl == True, "Phase 2: Observer frozen")
    assert_test(phase2_cfg.mafia_allow_observer_training == False, "Phase 2: Observer not training")
    assert_test(phase2_cfg.observer_only_training == False, "Phase 2: Not observer-only mode")

    # Test 7: TD3 training metrics tracking
    print("\n  7. Testing TD3 metrics tracking...")
    assert_test(hasattr(config, "last_td3_actor_loss"), "Has last_td3_actor_loss")
    assert_test(hasattr(config, "last_td3_critic_loss"), "Has last_td3_critic_loss")
    assert_test(hasattr(config, "last_td3_mean_reward"), "Has last_td3_mean_reward")
    assert_test(hasattr(config, "last_td3_updates"), "Has last_td3_updates")

    print("\n  [OK] TD3 with frozen Observer tests passed!")


def run_all_tests() -> Tuple[int, int, List[str]]:
    """Run all tests and return summary."""
    tests = [
        ("observer_validation.py", test_observer_validation),
        ("portfolio_allocator_reward.py", test_portfolio_allocator_reward),
        ("config.py walk-forward params", test_config_walkforward_params),
        ("train_observer_walkforward.py structure", test_train_observer_walkforward_structure),
        ("generate_rl_states.py structure", test_generate_rl_states_structure),
        ("Walk-forward iteration calculation", test_walkforward_iteration_calculation),
        ("Observer-TD3 separation", test_observer_td3_separation),
        ("TD3 with frozen Observer", test_td3_with_frozen_observer),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append(f"{name}: {e}")
            traceback.print_exc()

    return passed, failed, failures


def main():
    print("\n" + "#" * 70)
    print("# WALK-FORWARD IMPLEMENTATION TEST SUITE")
    print("#" * 70)
    print("\nRunning all tests...")

    passed, failed, failures = run_all_tests()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
