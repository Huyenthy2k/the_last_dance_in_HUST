
import torch
import sys
import os

def test_adaptive_hold_reward():
    print("Testing Adaptive Duration-Based Hold Bonus Logic...")
    
    # Mock Config
    alpha_hold = 2.0
    K = 2
    B = 1
    N = 3
    h_len = 5 # 5 days lookahead
    
    # Mock Data
    # Shape: (B, h, N)
    future_returns = torch.tensor([
        [
            [0.02, 0.05, -0.01], # Day 1: S0>M, S1>M, S2<0
            [0.03, 0.05, 0.02],  # Day 2: S0>M, S1>M, S2>M
            [-0.01, 0.05, 0.02], # Day 3: S0<0, S1>M, S2>M
            [0.02, 0.05, 0.005], # Day 4: S0>M, S1>M, S2<M
            [0.02, 0.05, 0.02],  # Day 5: S0>M, S1>M, S2>M
        ]
    ])
    
    # Shape: (B, h)
    market_returns = torch.tensor([
        [0.01, 0.01, 0.01, 0.01, 0.01]
    ])
    
    # Prev Holdings (B, N) - Assume we hold S0 and S2
    prev_holdings = torch.tensor([[1.0, 0.0, 1.0]])
    
    # Curr Holdings (B, K) indices -> (B, N) mask
    # Assume we KEEP holding S0 and S2
    curr_holdings = torch.tensor([[1.0, 0.0, 1.0]])
    
    # Expected Logic:
    # Market return is always 0.01.
    
    # Stock 0:
    # D1: 0.02 > 0.01 & > 0 -> WIN
    # D2: 0.03 > 0.01 & > 0 -> WIN
    # D3: -0.01 < 0 -> LOSS
    # D4: 0.02 > 0.01 -> WIN
    # D5: 0.02 > 0.01 -> WIN
    # Wins: 4/5. Score = 0.8.
    
    # Stock 1 (Not Held):
    # Wins: 5/5. Score = 1.0.
    
    # Stock 2:
    # D1: -0.01 < 0 -> LOSS
    # D2: 0.02 > 0.01 -> WIN
    # D3: 0.02 > 0.01 -> WIN
    # D4: 0.005 < 0.01 -> LOSS (Positive but below market)
    # D5: 0.02 > 0.01 -> WIN
    # Wins: 3/5. Score = 0.6.
    
    # Portfolio Score:
    # Held Stocks: S0 and S2.
    # Scores: 0.8 + 0.6 = 1.4.
    
    # R_hold = Alpha * (Sum / K)
    # R_hold = 2.0 * (1.4 / 2) = 1.4.
    
    # Implementation Simulation
    mkt_ret_expanded = market_returns.unsqueeze(-1) # (1, 5, 1)
    win_day_mask = (future_returns > mkt_ret_expanded) & (future_returns > 0)
    stock_win_consistency = win_day_mask.float().sum(dim=1) / float(h_len)
    
    held_mask = prev_holdings * curr_holdings
    portfolio_win_score = (stock_win_consistency * held_mask).sum(dim=1)
    
    r_hold = alpha_hold * (portfolio_win_score / K)
    
    print(f"Stock 0 Score: {stock_win_consistency[0,0]:.2f} (Expected 0.8)")
    print(f"Stock 1 Score: {stock_win_consistency[0,1]:.2f} (Expected 1.0)")
    print(f"Stock 2 Score: {stock_win_consistency[0,2]:.2f} (Expected 0.6)")
    print(f"Calculated R_hold: {r_hold.item():.4f}")
    
    expected = 1.4
    if abs(r_hold.item() - expected) < 1e-4:
        print("✅ PASSED: Calculation matches expectation matches Spec.")
    else:
        print(f"❌ FAILED: Expected {expected}, got {r_hold.item()}")
        sys.exit(1)

if __name__ == "__main__":
    test_adaptive_hold_reward()
