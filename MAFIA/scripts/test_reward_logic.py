#!/usr/bin/env python3
"""
Test cases for Reward Logic (Optimized Option B)
Covers all buy/sell/hold scenarios to verify reward/advantage/cost are correct.

Run: python scripts/test_reward_logic.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
import sys

# =============================================================================
# CONFIGURATION (Match with config.py)
# =============================================================================
SCALE_FACTOR = 100.0
ALPHA_TURNOVER = 0.33  # From analysis
ALPHA_CHANGE = 0.33
ALPHA_EXIT = 0.55
K = 10  # Top-K stocks
REBALANCE_INTERVAL = 21

# Optimized Option B weights
PERIOD_WEIGHT = 0.7
CUMULATIVE_WEIGHT = 0.3


# =============================================================================
# REWARD CALCULATION FUNCTIONS
# =============================================================================
@dataclass
class Stock:
    """Track stock state"""
    symbol: str
    entry_price: float
    entry_step: int
    last_rebalance_price: float
    last_rebalance_step: int


@dataclass
class RewardComponents:
    """All reward components for a rebalance"""
    R_select: float
    r_hold: float
    turnover_penalty: float
    symdiff_penalty: float
    total_penalty: float
    R_net: float
    baseline: float
    advantage: float


def calculate_r_hold_option_a(
    held_stocks: List[Stock],
    current_prices: Dict[str, float],
    current_step: int,
) -> float:
    """Option A: Accumulated Profit (current implementation)"""
    if not held_stocks:
        return 0.0

    r_hold_sum = 0.0
    for stock in held_stocks:
        current_price = current_prices[stock.symbol]
        profit_pct = (current_price - stock.entry_price) / (stock.entry_price + 1e-8)
        duration = current_step - stock.entry_step
        duration_bonus = np.log1p(duration)
        r_hold_sum += ALPHA_EXIT * profit_pct * duration_bonus

    return (r_hold_sum / len(held_stocks)) * SCALE_FACTOR


def calculate_r_hold_option_b(
    held_stocks: List[Stock],
    current_prices: Dict[str, float],
    market_period_return: float,
    current_step: int,
    apply_scaling: bool = True,  # [FIX 2026-01-02] Apply n_held/K scaling
) -> float:
    """Optimized Option B: Period Alpha + Momentum Memory

    With scaling fix: r_hold scales by n_held / K
    """
    if not held_stocks:
        return 0.0

    valid_stocks = [s for s in held_stocks if s.last_rebalance_step >= 0]
    if not valid_stocks:
        return 0.0

    r_hold_sum = 0.0
    for stock in valid_stocks:
        current_price = current_prices[stock.symbol]

        # Period profit (since last rebalance)
        period_profit = (current_price - stock.last_rebalance_price) / (stock.last_rebalance_price + 1e-8)

        # Period alpha (excess return over market) - CLAMPED >= 0
        period_alpha = max(0.0, period_profit - market_period_return)

        # Cumulative profit (since entry) - CAPPED [-1, 1]
        cumulative_profit = (current_price - stock.entry_price) / (stock.entry_price + 1e-8)
        cumulative_profit = np.clip(cumulative_profit, -1.0, 1.0)

        # Hybrid (70% period alpha + 30% cumulative)
        weighted_profit = PERIOD_WEIGHT * period_alpha + CUMULATIVE_WEIGHT * cumulative_profit

        # Duration bonus (from last rebalance)
        period_duration = current_step - stock.last_rebalance_step
        duration_bonus = np.log1p(period_duration)

        r_hold_sum += ALPHA_EXIT * weighted_profit * duration_bonus

    r_hold = (r_hold_sum / len(valid_stocks)) * SCALE_FACTOR

    # [FIX 2026-01-02] Scale by n_held / K
    # If only 1/10 stocks are held, r_hold should be 1/10 of full reward
    if apply_scaling:
        n_held = len(valid_stocks)
        scaling_factor = n_held / K
        r_hold = r_hold * scaling_factor

    return r_hold


def calculate_penalty(
    prev_portfolio: Set[str],
    curr_portfolio: Set[str],
    prev_weights: Optional[Dict[str, float]] = None,
    curr_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float]:
    """Calculate turnover and symdiff penalties"""
    if prev_portfolio is None:
        return 0.0, 0.0, 0.0

    held = prev_portfolio & curr_portfolio
    n_held = len(held)

    # SymDiff (normalized 0-2)
    symdiff = 2 * (K - n_held) / K
    symdiff_penalty = ALPHA_CHANGE * symdiff

    # Turnover (weight change)
    if prev_weights and curr_weights:
        weight_diff = 0.0
        all_stocks = prev_portfolio | curr_portfolio
        for s in all_stocks:
            w_prev = prev_weights.get(s, 0.0)
            w_curr = curr_weights.get(s, 0.0)
            weight_diff += abs(w_curr - w_prev)
        turnover = min(0.5 * weight_diff, 1.0)
    else:
        # Simplified: proportional to membership change
        n_changed = K - n_held
        turnover = n_changed / K * 0.5

    turnover_penalty = ALPHA_TURNOVER * turnover

    total_penalty = turnover_penalty + symdiff_penalty
    return turnover_penalty, symdiff_penalty, total_penalty


def calculate_R_select(
    portfolio_return: float,
) -> float:
    """R_select = scale_factor * portfolio_return"""
    return SCALE_FACTOR * portfolio_return


def calculate_baseline(
    market_return: float,
) -> float:
    """Baseline = scale_factor * market_return"""
    return SCALE_FACTOR * market_return


# =============================================================================
# TEST CASES
# =============================================================================
class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages = []

    def check(self, condition: bool, msg: str):
        if not condition:
            self.passed = False
            self.messages.append(f"FAIL: {msg}")
        else:
            self.messages.append(f"OK: {msg}")

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        result = f"\n{'='*60}\n{status}: {self.name}\n{'='*60}\n"
        for msg in self.messages:
            result += f"  {msg}\n"
        return result


def test_case_1_new_entry():
    """Test Case 1: New Entry (Buy) - No r_hold"""
    test = TestResult("Case 1: New Entry (Buy)")

    # Setup: Buy 10 new stocks
    prev_portfolio = None
    curr_portfolio = {f"STOCK_{i}" for i in range(K)}
    current_prices = {f"STOCK_{i}": 100.0 for i in range(K)}

    # No held stocks (all new)
    held_stocks = []

    # Calculate
    r_hold = calculate_r_hold_option_b(held_stocks, current_prices, 0.02, current_step=21)
    turn_pen, sym_pen, total_pen = calculate_penalty(prev_portfolio, curr_portfolio)
    R_select = calculate_R_select(0.05)  # 5% portfolio return
    baseline = calculate_baseline(0.03)   # 3% market return

    R_net = R_select + r_hold - total_pen
    advantage = R_net - baseline

    # Assertions
    test.check(r_hold == 0.0, f"r_hold should be 0 for new entries, got {r_hold:.4f}")
    test.check(turn_pen == 0.0, f"No turnover penalty for first rebalance, got {turn_pen:.4f}")
    test.check(sym_pen == 0.0, f"No symdiff penalty for first rebalance, got {sym_pen:.4f}")
    test.check(R_select == 5.0, f"R_select = 100 * 0.05 = 5.0, got {R_select:.4f}")
    test.check(baseline == 3.0, f"Baseline = 100 * 0.03 = 3.0, got {baseline:.4f}")
    test.check(advantage == 2.0, f"Advantage = 5.0 - 3.0 = 2.0, got {advantage:.4f}")

    print(test)
    return test.passed


def test_case_2_hold_winner_outperform_market():
    """Test Case 2: Hold Winning Stock that Outperforms Market"""
    test = TestResult("Case 2: Hold Winner (Outperform Market)")

    # Setup: Stock bought at 100, now 120 (+20%), market +5%
    held_stocks = [
        Stock("WINNER", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0)
    ]
    current_prices = {"WINNER": 120.0}
    market_period_return = 0.05  # 5%
    current_step = 21

    # Calculate r_hold
    r_hold_a = calculate_r_hold_option_a(held_stocks, current_prices, current_step)
    r_hold_b = calculate_r_hold_option_b(held_stocks, current_prices, market_period_return, current_step)

    # Manual calculation for Option B:
    # period_profit = (120 - 100) / 100 = 0.20
    # period_alpha = max(0, 0.20 - 0.05) = 0.15
    # cumulative_profit = clip(0.20, -1, 1) = 0.20
    # weighted = 0.7 * 0.15 + 0.3 * 0.20 = 0.105 + 0.06 = 0.165
    # duration_bonus = log1p(21) = 3.09
    # r_hold = 0.55 * 0.165 * 3.09 * 100 = 28.06

    expected_period_alpha = max(0, 0.20 - 0.05)
    expected_weighted = 0.7 * expected_period_alpha + 0.3 * 0.20
    expected_r_hold_b = 0.55 * expected_weighted * np.log1p(21) * 100

    test.check(abs(r_hold_b - expected_r_hold_b) < 0.01,
               f"r_hold_B should be {expected_r_hold_b:.2f}, got {r_hold_b:.2f}")
    test.check(r_hold_b > 0, f"r_hold should be POSITIVE for winner, got {r_hold_b:.2f}")
    test.check(r_hold_b < r_hold_a,
               f"Option B ({r_hold_b:.2f}) should be < Option A ({r_hold_a:.2f}) due to market adjustment")

    print(test)
    return test.passed


def test_case_3_hold_winner_underperform_market():
    """Test Case 3: Hold Winning Stock that Underperforms Market"""
    test = TestResult("Case 3: Hold Winner (Underperform Market)")

    # Setup: Stock +10%, but market +15%
    held_stocks = [
        Stock("LAGGARD", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0)
    ]
    current_prices = {"LAGGARD": 110.0}
    market_period_return = 0.15  # 15%
    current_step = 21

    r_hold_b = calculate_r_hold_option_b(held_stocks, current_prices, market_period_return, current_step)

    # period_alpha = max(0, 0.10 - 0.15) = 0 (clamped)
    # cumulative_profit = 0.10
    # weighted = 0.7 * 0 + 0.3 * 0.10 = 0.03
    # r_hold = 0.55 * 0.03 * log1p(21) * 100 = 5.10

    expected_weighted = 0.7 * 0 + 0.3 * 0.10
    expected_r_hold = 0.55 * expected_weighted * np.log1p(21) * 100

    test.check(abs(r_hold_b - expected_r_hold) < 0.01,
               f"r_hold should be {expected_r_hold:.2f}, got {r_hold_b:.2f}")
    test.check(r_hold_b > 0,
               f"r_hold still POSITIVE due to cumulative memory (30%), got {r_hold_b:.2f}")
    test.check(r_hold_b < 10,
               f"r_hold should be SMALL (laggard), got {r_hold_b:.2f}")

    print(test)
    return test.passed


def test_case_4_hold_loser():
    """Test Case 4: Hold Losing Stock"""
    test = TestResult("Case 4: Hold Loser")

    # Setup: Stock -15%, market +3%
    held_stocks = [
        Stock("LOSER", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0)
    ]
    current_prices = {"LOSER": 85.0}
    market_period_return = 0.03
    current_step = 21

    r_hold_b = calculate_r_hold_option_b(held_stocks, current_prices, market_period_return, current_step)

    # period_alpha = max(0, -0.15 - 0.03) = 0 (clamped)
    # cumulative_profit = clip(-0.15, -1, 1) = -0.15
    # weighted = 0.7 * 0 + 0.3 * (-0.15) = -0.045
    # r_hold = 0.55 * (-0.045) * log1p(21) * 100 = -7.65

    expected_weighted = 0.7 * 0 + 0.3 * (-0.15)
    expected_r_hold = 0.55 * expected_weighted * np.log1p(21) * 100

    test.check(abs(r_hold_b - expected_r_hold) < 0.01,
               f"r_hold should be {expected_r_hold:.2f}, got {r_hold_b:.2f}")
    test.check(r_hold_b < 0,
               f"r_hold should be NEGATIVE for loser (due to cumulative memory), got {r_hold_b:.2f}")

    print(test)
    return test.passed


def test_case_5_exit_winner():
    """Test Case 5: Exit Winner - Should pay penalty"""
    test = TestResult("Case 5: Exit Winner")

    prev_portfolio = {"WINNER_1", "WINNER_2", "STOCK_3", "STOCK_4", "STOCK_5",
                      "STOCK_6", "STOCK_7", "STOCK_8", "STOCK_9", "STOCK_10"}
    curr_portfolio = {"NEW_1", "NEW_2", "STOCK_3", "STOCK_4", "STOCK_5",
                      "STOCK_6", "STOCK_7", "STOCK_8", "STOCK_9", "STOCK_10"}

    # 2 stocks changed: WINNER_1, WINNER_2 → NEW_1, NEW_2
    turn_pen, sym_pen, total_pen = calculate_penalty(prev_portfolio, curr_portfolio)

    # n_held = 8, n_changed = 2
    # symdiff = 2 * (10 - 8) / 10 = 0.4
    # turnover = 2/10 * 0.5 = 0.1
    # sym_pen = 0.33 * 0.4 = 0.132
    # turn_pen = 0.33 * 0.1 = 0.033
    # total = 0.165

    expected_symdiff = 2 * (10 - 8) / 10
    expected_turnover = 2 / 10 * 0.5
    expected_sym_pen = ALPHA_CHANGE * expected_symdiff
    expected_turn_pen = ALPHA_TURNOVER * expected_turnover

    test.check(abs(sym_pen - expected_sym_pen) < 0.001,
               f"symdiff_penalty = {expected_sym_pen:.4f}, got {sym_pen:.4f}")
    test.check(abs(turn_pen - expected_turn_pen) < 0.001,
               f"turnover_penalty = {expected_turn_pen:.4f}, got {turn_pen:.4f}")
    test.check(total_pen > 0,
               f"Total penalty should be > 0 for exiting stocks, got {total_pen:.4f}")

    print(test)
    return test.passed


def test_case_6_full_churn():
    """Test Case 6: Full Churn (100% portfolio change)"""
    test = TestResult("Case 6: Full Churn (100%)")

    prev_portfolio = {f"OLD_{i}" for i in range(K)}
    curr_portfolio = {f"NEW_{i}" for i in range(K)}

    turn_pen, sym_pen, total_pen = calculate_penalty(prev_portfolio, curr_portfolio)

    # n_held = 0, n_changed = 10
    # symdiff = 2 * (10 - 0) / 10 = 2.0 (max)
    # turnover = 10/10 * 0.5 = 0.5
    # sym_pen = 0.33 * 2.0 = 0.66
    # turn_pen = 0.33 * 0.5 = 0.165
    # total = 0.825

    expected_symdiff = 2.0
    expected_turnover = 0.5
    expected_total = ALPHA_CHANGE * expected_symdiff + ALPHA_TURNOVER * expected_turnover

    test.check(abs(total_pen - expected_total) < 0.001,
               f"Full churn penalty = {expected_total:.4f}, got {total_pen:.4f}")
    test.check(sym_pen == ALPHA_CHANGE * 2.0,
               f"Max symdiff penalty = {ALPHA_CHANGE * 2.0:.4f}, got {sym_pen:.4f}")

    # r_hold should be 0 (no held stocks)
    held_stocks = []
    r_hold = calculate_r_hold_option_b(held_stocks, {}, 0.0, 21)
    test.check(r_hold == 0.0, f"r_hold = 0 for full churn, got {r_hold:.4f}")

    print(test)
    return test.passed


def test_case_7_no_churn():
    """Test Case 7: No Churn (100% hold)"""
    test = TestResult("Case 7: No Churn (100% Hold)")

    portfolio = {f"STOCK_{i}" for i in range(K)}
    prev_portfolio = portfolio.copy()
    curr_portfolio = portfolio.copy()

    turn_pen, sym_pen, total_pen = calculate_penalty(prev_portfolio, curr_portfolio)

    # n_held = 10, symdiff = 0, turnover = 0
    test.check(total_pen == 0.0, f"No penalty for 100% hold, got {total_pen:.4f}")
    test.check(sym_pen == 0.0, f"symdiff = 0, got {sym_pen:.4f}")
    test.check(turn_pen == 0.0, f"turnover = 0, got {turn_pen:.4f}")

    print(test)
    return test.passed


def test_case_8_triggered_rebalance_short_duration():
    """Test Case 8: Triggered Rebalance (Short Duration)"""
    test = TestResult("Case 8: Triggered Rebalance (Short Duration)")

    # Stock held for only 7 days (triggered by vol shock)
    held_stocks = [
        Stock("QUICK", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0)
    ]
    current_prices = {"QUICK": 105.0}  # +5%
    market_period_return = 0.02  # +2%
    current_step = 7  # Short duration

    r_hold_short = calculate_r_hold_option_b(held_stocks, current_prices, market_period_return, current_step)

    # Compare with full duration (21 days)
    held_stocks_full = [
        Stock("QUICK", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0)
    ]
    r_hold_full = calculate_r_hold_option_b(held_stocks_full, current_prices, market_period_return, current_step=21)

    # duration_bonus(7) = log1p(7) = 2.08
    # duration_bonus(21) = log1p(21) = 3.09
    # Ratio = 2.08 / 3.09 = 0.67

    ratio = np.log1p(7) / np.log1p(21)
    test.check(abs(r_hold_short / r_hold_full - ratio) < 0.01,
               f"Short duration r_hold ratio = {ratio:.2f}, got {r_hold_short/r_hold_full:.2f}")
    test.check(r_hold_short < r_hold_full,
               f"Short duration r_hold ({r_hold_short:.2f}) < full ({r_hold_full:.2f})")

    print(test)
    return test.passed


def test_case_9_multiple_rebalances():
    """Test Case 9: Stock Held Through Multiple Rebalances"""
    test = TestResult("Case 9: Multiple Rebalances (Long Hold)")

    # Stock bought at rebalance 0, now at rebalance 3
    # Entry: 100, Rebalance 1: 110, Rebalance 2: 115, Current: 120

    # At Rebalance 3:
    held_stocks = [
        Stock("LONG_HOLD", entry_price=100.0, entry_step=0,
              last_rebalance_price=115.0, last_rebalance_step=42)  # Rebalance 2 was at step 42
    ]
    current_prices = {"LONG_HOLD": 120.0}
    market_period_return = 0.02  # 2% in this period
    current_step = 63  # Rebalance 3

    r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_period_return, current_step)

    # Period profit = (120 - 115) / 115 = 4.35%
    # Period alpha = max(0, 0.0435 - 0.02) = 0.0235
    # Cumulative profit = (120 - 100) / 100 = 0.20
    # Weighted = 0.7 * 0.0235 + 0.3 * 0.20 = 0.0765
    # Duration = 63 - 42 = 21
    # r_hold = 0.55 * 0.0765 * log1p(21) * 100 = 13.0

    period_profit = (120 - 115) / 115
    period_alpha = max(0, period_profit - 0.02)
    cumulative_profit = (120 - 100) / 100
    weighted = 0.7 * period_alpha + 0.3 * cumulative_profit
    expected_r_hold = 0.55 * weighted * np.log1p(21) * 100

    test.check(abs(r_hold - expected_r_hold) < 0.1,
               f"r_hold should be {expected_r_hold:.2f}, got {r_hold:.2f}")

    # Cumulative doesn't dominate
    cumulative_only = 0.55 * cumulative_profit * np.log1p(21) * 100
    test.check(r_hold < cumulative_only,
               f"r_hold ({r_hold:.2f}) < cumulative-only ({cumulative_only:.2f})")

    print(test)
    return test.passed


def test_case_10_first_rebalance_after_entry():
    """Test Case 10: First Rebalance After Entry (No Previous last_rebalance_price)"""
    test = TestResult("Case 10: First Rebalance After Entry")

    # Stock just bought at this rebalance - should NOT have r_hold
    # (it's in curr but not in prev, so not in held_stocks)

    # This is correct behavior - new entries don't get r_hold
    # But we need to ensure last_rebalance_price is set correctly for NEXT rebalance

    # Simulate: Stock A bought at step 0
    stock_a = Stock("NEW_A", entry_price=100.0, entry_step=0,
                    last_rebalance_price=100.0, last_rebalance_step=0)

    # At step 21, stock A is still held
    current_prices = {"NEW_A": 110.0}
    market_period_return = 0.03
    current_step = 21

    r_hold = calculate_r_hold_option_b([stock_a], current_prices, market_period_return, current_step)

    test.check(r_hold > 0, f"r_hold should be > 0 for first rebalance with gain, got {r_hold:.2f}")

    # Test edge case: last_rebalance_step = -1 (not set)
    stock_invalid = Stock("INVALID", entry_price=100.0, entry_step=0,
                          last_rebalance_price=100.0, last_rebalance_step=-1)
    r_hold_invalid = calculate_r_hold_option_b([stock_invalid], current_prices, market_period_return, current_step)

    test.check(r_hold_invalid == 0.0,
               f"r_hold = 0 when last_rebalance_step = -1, got {r_hold_invalid:.2f}")

    print(test)
    return test.passed


def test_case_11_cumulative_cap():
    """Test Case 11: Cumulative Profit Cap (±100%)"""
    test = TestResult("Case 11: Cumulative Profit Cap")

    # Stock with extreme gain: 100 → 350 (+250%)
    held_stocks = [
        Stock("MOON", entry_price=100.0, entry_step=0,
              last_rebalance_price=300.0, last_rebalance_step=42)
    ]
    current_prices = {"MOON": 350.0}
    market_period_return = 0.05
    current_step = 63

    r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_period_return, current_step)

    # Cumulative profit = 2.5 → capped to 1.0
    # Period profit = (350 - 300) / 300 = 16.67%
    # Period alpha = max(0, 0.1667 - 0.05) = 0.1167
    # Weighted = 0.7 * 0.1167 + 0.3 * 1.0 = 0.3817

    capped_cumulative = 1.0  # Max
    period_profit = (350 - 300) / 300
    period_alpha = max(0, period_profit - 0.05)
    weighted = 0.7 * period_alpha + 0.3 * capped_cumulative
    expected_r_hold = 0.55 * weighted * np.log1p(21) * 100

    test.check(abs(r_hold - expected_r_hold) < 0.1,
               f"r_hold with capped cumulative = {expected_r_hold:.2f}, got {r_hold:.2f}")

    # Without cap, r_hold would be higher
    weighted_uncapped = 0.7 * period_alpha + 0.3 * 2.5
    r_hold_uncapped = 0.55 * weighted_uncapped * np.log1p(21) * 100
    test.check(r_hold < r_hold_uncapped,
               f"Capped ({r_hold:.2f}) < Uncapped ({r_hold_uncapped:.2f})")

    print(test)
    return test.passed


def test_case_12_negative_cumulative_cap():
    """Test Case 12: Negative Cumulative Profit Cap (-100%)"""
    test = TestResult("Case 12: Negative Cumulative Cap")

    # Stock with extreme loss: 100 → 10 (-90%)
    held_stocks = [
        Stock("CRASH", entry_price=100.0, entry_step=0,
              last_rebalance_price=20.0, last_rebalance_step=42)
    ]
    current_prices = {"CRASH": 10.0}
    market_period_return = -0.10  # Market also down
    current_step = 63

    r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_period_return, current_step)

    # Cumulative profit = -0.90 → NOT capped (within [-1, 1])
    # Period profit = (10 - 20) / 20 = -50%
    # Period alpha = max(0, -0.50 - (-0.10)) = max(0, -0.40) = 0
    # Weighted = 0.7 * 0 + 0.3 * (-0.90) = -0.27

    expected_weighted = 0.7 * 0 + 0.3 * (-0.90)
    expected_r_hold = 0.55 * expected_weighted * np.log1p(21) * 100

    test.check(abs(r_hold - expected_r_hold) < 0.1,
               f"r_hold for crash = {expected_r_hold:.2f}, got {r_hold:.2f}")
    test.check(r_hold < 0, f"r_hold should be NEGATIVE for crash, got {r_hold:.2f}")

    print(test)
    return test.passed


def test_case_13_advantage_calculation():
    """Test Case 13: Full Advantage Calculation"""
    test = TestResult("Case 13: Full Advantage Calculation")

    # Setup scenario
    portfolio_return = 0.08  # 8%
    market_return = 0.03     # 3%

    held_stocks = [
        Stock("WINNER", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0)
    ]
    current_prices = {"WINNER": 108.0}

    prev_portfolio = {"WINNER", "STOCK_2", "STOCK_3", "STOCK_4", "STOCK_5",
                      "STOCK_6", "STOCK_7", "STOCK_8", "STOCK_9", "STOCK_10"}
    curr_portfolio = {"WINNER", "NEW_1", "STOCK_3", "STOCK_4", "STOCK_5",
                      "STOCK_6", "STOCK_7", "STOCK_8", "STOCK_9", "STOCK_10"}  # 1 change

    # Calculate all components
    R_select = calculate_R_select(portfolio_return)
    baseline = calculate_baseline(market_return)
    r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_return, current_step=21)
    _, _, total_pen = calculate_penalty(prev_portfolio, curr_portfolio)

    R_net = R_select + r_hold - total_pen
    advantage = R_net - baseline

    # R_select = 100 * 0.08 = 8.0
    # baseline = 100 * 0.03 = 3.0
    # r_hold ≈ 13.4 (calculated earlier)
    # penalty ≈ 0.099 (1 change)
    # R_net = 8.0 + 13.4 - 0.099 ≈ 21.3
    # advantage = 21.3 - 3.0 = 18.3

    test.check(R_select == 8.0, f"R_select = 8.0, got {R_select:.2f}")
    test.check(baseline == 3.0, f"baseline = 3.0, got {baseline:.2f}")
    test.check(r_hold > 0, f"r_hold > 0, got {r_hold:.2f}")
    test.check(total_pen < 1.0, f"penalty < 1.0 for 1 change, got {total_pen:.4f}")
    test.check(advantage > 0, f"advantage > 0 for outperforming portfolio, got {advantage:.2f}")

    # Check component ratios
    r_hold_ratio = r_hold / (abs(R_select) + 1e-8) * 100
    penalty_ratio = total_pen / (abs(R_select) + 1e-8) * 100

    print(f"  Component Ratios:")
    print(f"    r_hold / |R_select|: {r_hold_ratio:.1f}%")
    print(f"    penalty / |R_select|: {penalty_ratio:.1f}%")

    test.check(r_hold_ratio > 30 and r_hold_ratio < 200,
               f"r_hold ratio in reasonable range (30-200%), got {r_hold_ratio:.1f}%")

    print(test)
    return test.passed


def test_case_14_bear_market():
    """Test Case 14: Bear Market (Negative Returns)"""
    test = TestResult("Case 14: Bear Market")

    portfolio_return = -0.05  # -5%
    market_return = -0.10     # -10%

    held_stocks = [
        Stock("DEFENSIVE", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0)
    ]
    current_prices = {"DEFENSIVE": 95.0}  # -5%

    R_select = calculate_R_select(portfolio_return)
    baseline = calculate_baseline(market_return)
    r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_return, current_step=21)

    advantage = R_select + r_hold - baseline  # Simplified (no penalty)

    # R_select = -5.0
    # baseline = -10.0
    # period_alpha = max(0, -0.05 - (-0.10)) = 0.05 (outperform in bear!)
    # r_hold should be positive!

    test.check(R_select < 0, f"R_select negative in bear market, got {R_select:.2f}")
    test.check(baseline < 0, f"baseline negative in bear market, got {baseline:.2f}")
    test.check(r_hold > 0, f"r_hold POSITIVE for outperforming in bear, got {r_hold:.2f}")
    test.check(advantage > 0, f"advantage POSITIVE for beating market, got {advantage:.2f}")

    print(test)
    return test.passed


def test_case_15_duration_zero():
    """Test Case 15: Edge Case - Duration = 0"""
    test = TestResult("Case 15: Duration Zero (Edge Case)")

    # Immediate rebalance after previous
    held_stocks = [
        Stock("QUICK", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=21)  # Last rebalance at 21
    ]
    current_prices = {"QUICK": 101.0}
    current_step = 21  # Same step!

    r_hold = calculate_r_hold_option_b(held_stocks, current_prices, 0.01, current_step)

    # Duration = 21 - 21 = 0
    # log1p(0) = 0
    # r_hold = 0

    test.check(r_hold == 0.0, f"r_hold = 0 when duration = 0, got {r_hold:.4f}")

    print(test)
    return test.passed


def test_case_16_mixed_portfolio():
    """Test Case 16: Mixed Portfolio (Winners and Losers)"""
    test = TestResult("Case 16: Mixed Portfolio")

    held_stocks = [
        Stock("WINNER_1", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0),
        Stock("WINNER_2", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0),
        Stock("LOSER_1", entry_price=100.0, entry_step=0,
              last_rebalance_price=100.0, last_rebalance_step=0),
    ]
    current_prices = {
        "WINNER_1": 120.0,  # +20%
        "WINNER_2": 115.0,  # +15%
        "LOSER_1": 85.0,    # -15%
    }
    market_period_return = 0.05
    current_step = 21

    r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_period_return, current_step)

    # Calculate expected
    r_hold_parts = []
    for stock in held_stocks:
        price = current_prices[stock.symbol]
        period_profit = (price - stock.last_rebalance_price) / stock.last_rebalance_price
        period_alpha = max(0, period_profit - market_period_return)
        cumulative_profit = np.clip((price - stock.entry_price) / stock.entry_price, -1, 1)
        weighted = 0.7 * period_alpha + 0.3 * cumulative_profit
        r_hold_parts.append(ALPHA_EXIT * weighted * np.log1p(21))

    expected_r_hold = np.mean(r_hold_parts) * SCALE_FACTOR

    test.check(abs(r_hold - expected_r_hold) < 0.1,
               f"Mixed portfolio r_hold = {expected_r_hold:.2f}, got {r_hold:.2f}")

    # r_hold should be positive overall (winners > loser in this case)
    test.check(r_hold > 0, f"r_hold positive with more winners, got {r_hold:.2f}")

    print(test)
    return test.passed


def test_case_17_r_hold_contribution_analysis():
    """Test Case 17: Analyze r_hold / R_select Ratio Across Scenarios"""
    test = TestResult("Case 17: r_hold Contribution Analysis")

    scenarios = [
        # (name, n_held, avg_period_alpha, avg_cumulative, portfolio_return)
        ("Low Churn (6 held, moderate gain)", 6, 0.05, 0.10, 0.08),
        ("Medium Churn (4 held, good gain)", 4, 0.08, 0.15, 0.10),
        ("High Churn (2 held, strong gain)", 2, 0.12, 0.25, 0.12),
        ("Full Hold (10 held, small gain)", 10, 0.02, 0.05, 0.05),
        ("Typical (6 held, outperform)", 6, 0.05, 0.08, 0.08),
    ]

    print(f"\n  {'Scenario':<40} | {'n_held':>6} | {'r_hold':>8} | {'R_select':>8} | {'Ratio':>8}")
    print(f"  {'-'*40}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    ratios = []
    for name, n_held, avg_alpha, avg_cum, port_ret in scenarios:
        # Simulate held stocks
        held_stocks = []
        current_prices = {}
        for i in range(n_held):
            # Vary individual stocks around average
            noise = np.random.uniform(-0.02, 0.02)
            entry_p = 100.0
            # Set current price to achieve avg returns
            curr_p = entry_p * (1 + avg_cum + noise)
            held_stocks.append(Stock(
                f"STOCK_{i}", entry_price=entry_p, entry_step=0,
                last_rebalance_price=entry_p, last_rebalance_step=0
            ))
            current_prices[f"STOCK_{i}"] = curr_p

        market_return = port_ret - avg_alpha  # So that alpha = port_ret - market
        r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_return, current_step=21)
        R_select = calculate_R_select(port_ret)

        ratio = r_hold / (abs(R_select) + 1e-8) * 100
        ratios.append(ratio)

        print(f"  {name:<40} | {n_held:>6} | {r_hold:>8.2f} | {R_select:>8.2f} | {ratio:>7.1f}%")

    avg_ratio = np.mean(ratios)
    print(f"\n  Average Ratio: {avg_ratio:.1f}%")

    # Check if ratios are reasonable (target: 30-70%)
    test.check(avg_ratio > 20 and avg_ratio < 100,
               f"Average r_hold/R_select ratio in range (20-100%), got {avg_ratio:.1f}%")

    # Flag if any ratio is too extreme
    max_ratio = max(ratios)
    test.check(max_ratio < 150,
               f"Max ratio < 150%, got {max_ratio:.1f}%")

    print(test)
    return test.passed


def test_case_18_scaling_by_n_held():
    """Test Case 18: r_hold MUST Scale by n_held / K"""
    test = TestResult("Case 18: Scaling Analysis (n_held / K) - ISSUE FOUND")

    # Scenario: Same profit, different number of held stocks
    market_return = 0.03
    profit = 0.10  # 10% gain

    print(f"\n  🚨 ISSUE: r_hold should scale by n_held / K")
    print(f"     - Hold 1/10 stocks → should get 1/10 of r_hold reward")
    print(f"     - Hold 10/10 stocks → should get full r_hold reward")
    print(f"\n  {'n_held':<8} | {'r_hold (current)':>16} | {'r_hold (FIXED)':>16} | {'% of Full':>10}")
    print(f"  {'-'*8}-+-{'-'*16}-+-{'-'*16}-+-{'-'*10}")

    r_hold_full = None
    results = []

    for n_held in [1, 3, 5, 7, 10]:
        held_stocks = []
        current_prices = {}
        for i in range(n_held):
            held_stocks.append(Stock(
                f"S_{i}", entry_price=100.0, entry_step=0,
                last_rebalance_price=100.0, last_rebalance_step=0
            ))
            current_prices[f"S_{i}"] = 100.0 * (1 + profit)

        r_hold_current = calculate_r_hold_option_b(held_stocks, current_prices, market_return, current_step=21)

        # FIXED: Scale by n_held / K
        r_hold_fixed = r_hold_current * (n_held / K)

        if n_held == 10:
            r_hold_full = r_hold_current

        pct_of_full = (n_held / K) * 100
        results.append((n_held, r_hold_current, r_hold_fixed, pct_of_full))
        print(f"  {n_held:<8} | {r_hold_current:>16.2f} | {r_hold_fixed:>16.2f} | {pct_of_full:>9.0f}%")

    # Verify the issue
    print(f"\n  ❌ CURRENT BEHAVIOR:")
    print(f"     Hold 1/10: r_hold = {results[0][1]:.2f} (SAME as hold 10/10!)")
    print(f"     Hold 10/10: r_hold = {results[4][1]:.2f}")

    print(f"\n  ✅ CORRECT BEHAVIOR (with scaling):")
    print(f"     Hold 1/10: r_hold = {results[0][2]:.2f} (10% of full)")
    print(f"     Hold 10/10: r_hold = {results[4][2]:.2f} (100% of full)")

    # This test FAILS to highlight the issue
    r_hold_1 = results[0][1]
    r_hold_10 = results[4][1]
    test.check(r_hold_1 < r_hold_10 * 0.2,
               f"ISSUE: Hold 1/10 should have much smaller r_hold than 10/10. Got {r_hold_1:.2f} vs {r_hold_10:.2f}")

    print(f"\n  💡 FIX REQUIRED in observer_offline_trainer.py:")
    print(f"     r_hold = mean(profits) * alpha * duration * (n_held / K)")

    print(test)
    return test.passed


def calculate_r_hold_option_b_fixed(
    held_stocks: List[Stock],
    current_prices: Dict[str, float],
    market_period_return: float,
    current_step: int,
) -> float:
    """FIXED Option B: Scale by n_held / K"""
    if not held_stocks:
        return 0.0

    valid_stocks = [s for s in held_stocks if s.last_rebalance_step >= 0]
    if not valid_stocks:
        return 0.0

    r_hold_sum = 0.0
    for stock in valid_stocks:
        current_price = current_prices[stock.symbol]

        # Period profit
        period_profit = (current_price - stock.last_rebalance_price) / (stock.last_rebalance_price + 1e-8)
        period_alpha = max(0.0, period_profit - market_period_return)

        # Cumulative profit (capped)
        cumulative_profit = (current_price - stock.entry_price) / (stock.entry_price + 1e-8)
        cumulative_profit = np.clip(cumulative_profit, -1.0, 1.0)

        # Hybrid
        weighted_profit = PERIOD_WEIGHT * period_alpha + CUMULATIVE_WEIGHT * cumulative_profit

        # Duration bonus
        period_duration = current_step - stock.last_rebalance_step
        duration_bonus = np.log1p(period_duration)

        r_hold_sum += ALPHA_EXIT * weighted_profit * duration_bonus

    # FIXED: Scale by n_held / K
    n_held = len(valid_stocks)
    scaling_factor = n_held / K

    return (r_hold_sum / len(valid_stocks)) * SCALE_FACTOR * scaling_factor


def test_case_18b_verify_fix():
    """Test Case 18b: Verify the fix for n_held scaling"""
    test = TestResult("Case 18b: Verify n_held Scaling Fix")

    market_return = 0.03
    profit = 0.10

    print(f"\n  Testing FIXED r_hold calculation with n_held scaling")
    print(f"\n  {'n_held':<8} | {'r_hold (FIXED)':>16} | {'Expected %':>12} | {'Status':>10}")
    print(f"  {'-'*8}-+-{'-'*16}-+-{'-'*12}-+-{'-'*10}")

    r_hold_at_10 = None

    for n_held in [1, 2, 5, 10]:
        held_stocks = []
        current_prices = {}
        for i in range(n_held):
            held_stocks.append(Stock(
                f"S_{i}", entry_price=100.0, entry_step=0,
                last_rebalance_price=100.0, last_rebalance_step=0
            ))
            current_prices[f"S_{i}"] = 100.0 * (1 + profit)

        r_hold_fixed = calculate_r_hold_option_b_fixed(held_stocks, current_prices, market_return, 21)

        if n_held == 10:
            r_hold_at_10 = r_hold_fixed

        expected_pct = n_held / K * 100

        if r_hold_at_10:
            actual_pct = r_hold_fixed / r_hold_at_10 * 100
            status = "✅" if abs(actual_pct - expected_pct) < 1 else "❌"
        else:
            status = "-"

        print(f"  {n_held:<8} | {r_hold_fixed:>16.2f} | {expected_pct:>11.0f}% | {status:>10}")

    # Verify scaling
    held_1 = [Stock("S", 100.0, 0, 100.0, 0)]
    held_10 = [Stock(f"S_{i}", 100.0, 0, 100.0, 0) for i in range(10)]
    prices_1 = {"S": 110.0}
    prices_10 = {f"S_{i}": 110.0 for i in range(10)}

    r_1 = calculate_r_hold_option_b_fixed(held_1, prices_1, 0.03, 21)
    r_10 = calculate_r_hold_option_b_fixed(held_10, prices_10, 0.03, 21)

    ratio = r_1 / r_10
    expected_ratio = 1 / 10

    test.check(abs(ratio - expected_ratio) < 0.01,
               f"Hold 1/10 should give 10% of r_hold. Got ratio = {ratio:.2f}, expected = {expected_ratio:.2f}")

    print(f"\n  ✅ FIXED: Hold 1/10 gives r_hold = {r_1:.2f} ({ratio*100:.0f}% of full)")
    print(f"           Hold 10/10 gives r_hold = {r_10:.2f} (100%)")

    print(test)
    return test.passed


def test_case_19_penalty_vs_r_hold_balance():
    """Test Case 19: Penalty vs r_hold Balance at Different Churn Rates"""
    test = TestResult("Case 19: Penalty vs r_hold Balance")

    print(f"\n  {'Churn%':<8} | {'n_held':>6} | {'r_hold':>8} | {'Penalty':>8} | {'Net Bonus':>10} | {'Assessment':>12}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}")

    market_return = 0.03
    avg_profit = 0.08  # 8% average profit for held stocks

    results = []
    for churn_pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        n_held = int(K * (1 - churn_pct))
        n_changed = K - n_held

        # Create held stocks
        held_stocks = []
        current_prices = {}
        for i in range(n_held):
            held_stocks.append(Stock(
                f"S_{i}", entry_price=100.0, entry_step=0,
                last_rebalance_price=100.0, last_rebalance_step=0
            ))
            current_prices[f"S_{i}"] = 100.0 * (1 + avg_profit)

        r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_return, current_step=21) if n_held > 0 else 0.0

        # Calculate penalty
        prev_portfolio = {f"S_{i}" for i in range(K)}
        curr_portfolio = {f"S_{i}" for i in range(n_held)} | {f"NEW_{i}" for i in range(n_changed)}
        _, _, penalty = calculate_penalty(prev_portfolio, curr_portfolio)

        net_bonus = r_hold - penalty

        if net_bonus > 0:
            assessment = "HOLD wins"
        elif net_bonus < 0:
            assessment = "CHURN wins"
        else:
            assessment = "Neutral"

        results.append((churn_pct, n_held, r_hold, penalty, net_bonus, assessment))
        print(f"  {churn_pct*100:>6.0f}% | {n_held:>6} | {r_hold:>8.2f} | {penalty:>8.4f} | {net_bonus:>10.2f} | {assessment:>12}")

    # Check that holding is encouraged (net_bonus > 0) for low churn
    low_churn_net = [r[4] for r in results if r[0] <= 0.4]
    test.check(all(nb > 0 for nb in low_churn_net),
               "Low churn (<=40%) should have positive net bonus")

    # Check that high churn is penalized less than r_hold gain
    # (model can still churn if needed, just with reduced total reward)
    high_churn_penalty = [r[3] for r in results if r[0] >= 0.8]
    test.check(all(p < 1.0 for p in high_churn_penalty),
               f"High churn penalty reasonable (<1.0), got {high_churn_penalty}")

    print(test)
    return test.passed


def test_case_20_realistic_trading_scenario():
    """Test Case 20: Realistic Trading Scenario (Full Cycle)"""
    test = TestResult("Case 20: Realistic Trading Scenario")

    # Simulate 3 consecutive rebalances
    print(f"\n  SIMULATING 3 REBALANCE CYCLES")
    print(f"  " + "="*60)

    # Initial portfolio (10 stocks)
    portfolio = {f"STOCK_{i}": {"entry_price": 100.0, "entry_step": 0,
                                 "last_rebalance_price": 100.0, "last_rebalance_step": 0}
                 for i in range(K)}

    total_R_select = 0
    total_r_hold = 0
    total_penalty = 0

    for cycle in range(1, 4):
        print(f"\n  Cycle {cycle}:")

        # Simulate price changes
        current_step = cycle * 21
        market_return = np.random.uniform(0.01, 0.05)

        current_prices = {}
        for symbol, data in portfolio.items():
            # Simulate random return with slight upward bias
            stock_return = np.random.uniform(-0.05, 0.15)
            current_prices[symbol] = data["entry_price"] * (1 + stock_return)

        # Decide churn (target 40%)
        symbols = list(portfolio.keys())
        n_to_sell = 4
        to_sell = np.random.choice(symbols, n_to_sell, replace=False)
        held_symbols = [s for s in symbols if s not in to_sell]

        # Create held_stocks for r_hold calculation
        held_stocks = [
            Stock(s, portfolio[s]["entry_price"], portfolio[s]["entry_step"],
                  portfolio[s]["last_rebalance_price"], portfolio[s]["last_rebalance_step"])
            for s in held_symbols
        ]

        # Calculate r_hold
        held_prices = {s: current_prices[s] for s in held_symbols}
        r_hold = calculate_r_hold_option_b(held_stocks, held_prices, market_return, current_step)

        # Calculate R_select (using average of current prices vs entry)
        avg_return = np.mean([(current_prices[s] - portfolio[s]["entry_price"]) / portfolio[s]["entry_price"]
                              for s in symbols])
        R_select = calculate_R_select(avg_return)

        # Calculate penalty
        prev_set = set(symbols)
        new_symbols = [f"NEW_{cycle}_{i}" for i in range(n_to_sell)]
        curr_set = set(held_symbols) | set(new_symbols)
        _, _, penalty = calculate_penalty(prev_set, curr_set)

        R_net = R_select + r_hold - penalty
        baseline = calculate_baseline(market_return)
        advantage = R_net - baseline

        print(f"    Market return: {market_return*100:.1f}%")
        print(f"    Portfolio return: {avg_return*100:.1f}%")
        print(f"    Held: {len(held_symbols)}, Sold: {n_to_sell}")
        print(f"    R_select: {R_select:.2f}, r_hold: {r_hold:.2f}, Penalty: {penalty:.4f}")
        print(f"    R_net: {R_net:.2f}, Advantage: {advantage:.2f}")

        total_R_select += R_select
        total_r_hold += r_hold
        total_penalty += penalty

        # Update portfolio for next cycle
        new_portfolio = {}
        for s in held_symbols:
            new_portfolio[s] = portfolio[s].copy()
            new_portfolio[s]["last_rebalance_price"] = current_prices[s]
            new_portfolio[s]["last_rebalance_step"] = current_step
        for s in new_symbols:
            new_portfolio[s] = {
                "entry_price": 100.0,  # New entry
                "entry_step": current_step,
                "last_rebalance_price": 100.0,
                "last_rebalance_step": current_step
            }
        portfolio = new_portfolio

    print(f"\n  " + "-"*60)
    print(f"  TOTALS over 3 cycles:")
    print(f"    Total R_select: {total_R_select:.2f}")
    print(f"    Total r_hold: {total_r_hold:.2f}")
    print(f"    Total Penalty: {total_penalty:.4f}")
    print(f"    r_hold/R_select ratio: {total_r_hold / (abs(total_R_select) + 1e-8) * 100:.1f}%")

    # Assertions
    test.check(total_penalty < abs(total_R_select),
               f"Total penalty ({total_penalty:.2f}) < |Total R_select| ({abs(total_R_select):.2f})")

    r_hold_ratio = total_r_hold / (abs(total_R_select) + 1e-8) * 100
    test.check(r_hold_ratio < 150,
               f"r_hold/R_select ratio ({r_hold_ratio:.1f}%) < 150%")

    print(test)
    return test.passed


def test_case_21_alpha_exit_sensitivity():
    """Test Case 21: Alpha Exit Sensitivity Analysis"""
    test = TestResult("Case 21: Alpha Exit Sensitivity")

    print(f"\n  Testing different alpha_exit values")
    print(f"  " + "-"*50)

    # REALISTIC scenario based on actual simulation data:
    # - Random selection R_select mean ~1.5 (from analysis)
    # - Trained model R_select ~6-8
    # - Average stock profit ~5-8% (outperformers)
    # - Market return ~3%
    held_stocks = [
        Stock("S_0", 100.0, 0, 100.0, 0),
        Stock("S_1", 100.0, 0, 100.0, 0),
        Stock("S_2", 100.0, 0, 100.0, 0),
        Stock("S_3", 100.0, 0, 100.0, 0),
        Stock("S_4", 100.0, 0, 100.0, 0),
        Stock("S_5", 100.0, 0, 100.0, 0),
    ]
    # REALISTIC: 5% gain (not 10%)
    current_prices = {f"S_{i}": 105.0 for i in range(6)}
    market_return = 0.03
    # REALISTIC: Portfolio return = 5%, R_select = 5.0
    R_select = 5.0

    print(f"  Scenario: 6 held stocks, +5% profit, market +3%, R_select = {R_select}")
    print(f"  {'Alpha_Exit':>10} | {'r_hold':>10} | {'r_hold/R_sel':>12} | {'Assessment':>15}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*15}")

    alpha_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55]
    results = []

    for alpha_test in alpha_values:
        # Temporarily override ALPHA_EXIT
        global ALPHA_EXIT
        original_alpha = ALPHA_EXIT
        ALPHA_EXIT = alpha_test

        r_hold = calculate_r_hold_option_b(held_stocks, current_prices, market_return, current_step=21)
        ratio = r_hold / R_select * 100

        if ratio < 30:
            assessment = "Too Low"
        elif ratio <= 60:
            assessment = "OPTIMAL"
        elif ratio <= 100:
            assessment = "Acceptable"
        else:
            assessment = "Too High"

        results.append((alpha_test, r_hold, ratio, assessment))
        print(f"  {alpha_test:>10.2f} | {r_hold:>10.2f} | {ratio:>11.1f}% | {assessment:>15}")

        ALPHA_EXIT = original_alpha

    # Find optimal alpha range
    optimal = [r for r in results if r[3] == "OPTIMAL"]
    acceptable = [r for r in results if r[3] in ["OPTIMAL", "Acceptable"]]

    if optimal:
        test.check(True, f"Found OPTIMAL alpha values: {[r[0] for r in optimal]}")
    else:
        test.check(len(acceptable) > 0, f"Found acceptable alpha values: {[r[0] for r in acceptable]}")

    # Recommend optimal alpha
    if optimal:
        rec_alpha = optimal[len(optimal)//2][0]  # Middle value
        print(f"\n  💡 RECOMMENDATION: alpha_exit = {rec_alpha:.2f}")
    elif acceptable:
        rec_alpha = acceptable[0][0]  # First acceptable
        print(f"\n  💡 RECOMMENDATION: alpha_exit = {rec_alpha:.2f}")

    print(test)
    return test.passed


def test_case_22_issue_summary():
    """Test Case 22: Summary of Issues Found and Fixes"""
    test = TestResult("Case 22: Issue Summary & Fixes")

    print(f"\n  ===== IDENTIFIED ISSUES =====")

    # Issue 1: r_hold too high with current alpha_exit
    print(f"\n  Issue 1: r_hold / R_select ratio too high")
    print(f"    Current alpha_exit: 0.55")
    print(f"    Observed ratios: 125-360% (should be 30-60%)")
    print(f"    Root cause: Optimized Option B has different scale than Option A")

    # Issue 2: Need to recalibrate alpha_exit
    print(f"\n  Issue 2: alpha_exit needs recalibration for Option B")

    # Calculate recommended alpha
    # Target: r_hold/R_select = 50%
    # Current with alpha=0.55: ratio ~= 125-170%
    # Need to reduce by factor of 2.5-3.5x
    # Recommended alpha = 0.55 / 3 = 0.18

    global ALPHA_EXIT

    target_ratio = 0.50
    current_ratio = 1.50  # Average observed
    recommended_alpha = ALPHA_EXIT * (target_ratio / current_ratio)

    print(f"\n  RECOMMENDED FIX:")
    print(f"    Current alpha_exit: {ALPHA_EXIT}")
    print(f"    Target ratio: {target_ratio*100:.0f}%")
    print(f"    Observed ratio: {current_ratio*100:.0f}%")
    print(f"    Recommended alpha_exit: {recommended_alpha:.2f}")

    # Verify with calculation
    held_stocks = [Stock("S", 100.0, 0, 100.0, 0)]
    current_prices = {"S": 105.0}  # +5%
    market_return = 0.03
    R_select = 5.0

    original = ALPHA_EXIT
    ALPHA_EXIT = recommended_alpha
    r_hold_new = calculate_r_hold_option_b(held_stocks, current_prices, market_return, 21)
    new_ratio = r_hold_new / R_select * 100
    ALPHA_EXIT = original

    print(f"\n  VERIFICATION:")
    print(f"    With alpha_exit = {recommended_alpha:.2f}")
    print(f"    r_hold = {r_hold_new:.2f}")
    print(f"    r_hold/R_select = {new_ratio:.1f}%")

    test.check(20 < new_ratio < 80, f"New ratio in acceptable range, got {new_ratio:.1f}%")

    print(test)
    return test.passed


# =============================================================================
# RUN ALL TESTS
# =============================================================================
def run_all_tests():
    print("\n" + "=" * 70)
    print("REWARD LOGIC TEST SUITE (Optimized Option B)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  SCALE_FACTOR: {SCALE_FACTOR}")
    print(f"  ALPHA_TURNOVER: {ALPHA_TURNOVER}")
    print(f"  ALPHA_CHANGE: {ALPHA_CHANGE}")
    print(f"  ALPHA_EXIT: {ALPHA_EXIT}")
    print(f"  K (Top-K): {K}")
    print(f"  PERIOD_WEIGHT: {PERIOD_WEIGHT}")
    print(f"  CUMULATIVE_WEIGHT: {CUMULATIVE_WEIGHT}")

    tests = [
        test_case_1_new_entry,
        test_case_2_hold_winner_outperform_market,
        test_case_3_hold_winner_underperform_market,
        test_case_4_hold_loser,
        test_case_5_exit_winner,
        test_case_6_full_churn,
        test_case_7_no_churn,
        test_case_8_triggered_rebalance_short_duration,
        test_case_9_multiple_rebalances,
        test_case_10_first_rebalance_after_entry,
        test_case_11_cumulative_cap,
        test_case_12_negative_cumulative_cap,
        test_case_13_advantage_calculation,
        test_case_14_bear_market,
        test_case_15_duration_zero,
        test_case_16_mixed_portfolio,
        test_case_17_r_hold_contribution_analysis,
        test_case_18_scaling_by_n_held,
        test_case_18b_verify_fix,
        test_case_19_penalty_vs_r_hold_balance,
        test_case_20_realistic_trading_scenario,
        test_case_21_alpha_exit_sensitivity,
        test_case_22_issue_summary,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_func.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total Tests: {len(tests)}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    print("=" * 70)

    if failed > 0:
        print("\n⚠️  Some tests failed. Please review the logic.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
