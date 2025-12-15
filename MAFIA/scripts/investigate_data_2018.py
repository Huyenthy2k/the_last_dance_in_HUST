
import pandas as pd
import numpy as np
import os

STOCK_FILE = "agents/MAFIA/data/stock_data_dynamic143.csv"
MARKET_FILE = "agents/MAFIA/data/VNINDEX_1d_index.csv"

def compute_metrics(df, name="Asset"):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Filter 2018
    mask = (df['date'] >= '2018-01-01') & (df['date'] <= '2018-12-31')
    df_2018 = df.loc[mask].copy()
    
    if len(df_2018) < 2:
        # print(f"[{name}] Not enough data for 2018")
        return None

    # Calculate daily returns exactly as in code
    # returns[1:] = (close[1:] - close[:-1]) / clip(close[:-1], 1e-8)
    close = df_2018['close'].values
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-8)
    
    # Exclude first 0 return
    daily_returns = returns[1:]
    
    # Cumulative Return
    net_profit = np.prod(1 + daily_returns) - 1
    
    # Annualized Return (CAGR)
    days = len(daily_returns)
    tradeDays_per_year = 252
    annual_return = np.power((1 + net_profit), (tradeDays_per_year / days)) - 1
    
    # Volatility
    std_daily = np.std(daily_returns, ddof=1)
    vol_annual = std_daily * np.sqrt(tradeDays_per_year)
    
    # Sharpe (Rf=0 for simplicity here, code uses 3% or similar)
    sharpe = annual_return / vol_annual if vol_annual > 0 else 0
    
    # Only print for Index or specific checks
    if name == "VNINDEX":
        print(f"--- {name} (2018) ---")
        print(f"Days: {days}")
        print(f"Start Close: {close[0]:.2f}")
        print(f"End Close:   {close[-1]:.2f}")
        print(f"Net Profit:  {net_profit*100:.2f}%")
        print(f"Ann Return:  {annual_return*100:.2f}%")
        print(f"Ann Vol:     {vol_annual*100:.2f}%")
        print(f"Sharpe:      {sharpe:.4f}")
        print("----------------")
    return annual_return

def main():
    if os.path.exists(MARKET_FILE):
        mkt = pd.read_csv(MARKET_FILE)
        compute_metrics(mkt, "VNINDEX")
    else:
        print(f"Market file not found at {MARKET_FILE}")

    if os.path.exists(STOCK_FILE):
        stocks = pd.read_csv(STOCK_FILE)
        stocks['date'] = pd.to_datetime(stocks['date'])
        
        # Calculate for a few stocks
        stock_list = stocks['stock'].unique()
        print(f"Total stocks: {len(stock_list)}")
        
        results = []
        for s in stock_list:
            sdf = stocks[stocks['stock'] == s].copy()
            ann_ret = compute_metrics(sdf, f"Stock_{s}")
            if ann_ret is not None:
                results.append((s, ann_ret))
        
        # Top 10
        results.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 10 Stocks 2018 by Annual Return (CAGR):")
        for s, ret in results[:10]:
            print(f"{s}: {ret*100:.2f}%")
    else:
        print(f"Stock file not found at {STOCK_FILE}")

if __name__ == "__main__":
    main()
