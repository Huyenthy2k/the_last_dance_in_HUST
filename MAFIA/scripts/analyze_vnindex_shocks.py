
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data_path = 'agents/MAFIA/data/VNINDEX_1d_index.csv'
try:
    df = pd.read_csv(data_path)
    # Check column names
    print("Columns:", df.columns)
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
        
    df = df.sort_values('date')
    df = df.set_index('date')
    
    # Filter for relevant years
    df_train = df['2015-01-01':'2019-12-31']
    df_2020 = df['2020-01-01':'2020-12-31']
    
    # Calculate daily returns
    # Assuming 'close' or 'Close' column exists
    col = 'close' if 'close' in df.columns else 'Close'
    
    df['returns'] = df[col].pct_change()
    
    # Calculate Rolling Volatility (e.g., 21 days)
    df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252) # Annualized
    
    # Calculate Max Drawdown per year
    def get_max_drawdown(series):
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        return drawdown.min(), drawdown.idxmin()

    print("\n--- Market Analysis 2015-2020 ---")
    
    # Analyze each year
    for year in range(2015, 2021):
        # boolean mask for safe filtering
        mask = (df.index >= f'{year}-01-01') & (df.index <= f'{year}-12-31')
        sub_df = df[mask]
        
        if len(sub_df) == 0:
            print(f"Year {year}: No data")
            continue
            
        min_dd, dd_date = get_max_drawdown(sub_df[col])
        max_vol = sub_df['volatility'].max()
        
        print(f"Year {year}:")
        print(f"  Max Drawdown: {min_dd:.2%}")
        print(f"  Peak Volatility (Ann.): {max_vol:.2%}")
        if len(sub_df) > 0:
            print(f"  Return: {(sub_df[col].iloc[-1] / sub_df[col].iloc[0] - 1):.2%}")
        print("-" * 30)

    # Specific look at 2018 crash vs 2020 crash
    print("\n--- Comparing Major Drops ---")
    # 2018 Drop
    mask_2018 = (df.index >= '2018-01-01') & (df.index <= '2018-12-31')
    df_2018 = df[mask_2018]
    if len(df_2018) > 0:
        dd_2018, _ = get_max_drawdown(df_2018[col])
        print(f"2018 Max Drawdown: {dd_2018:.2%}")
    
    # 2020 Drop (first half)
    mask_2020 = (df.index >= '2020-01-01') & (df.index <= '2020-06-30')
    df_2020_h1 = df[mask_2020]
    if len(df_2020_h1) > 0:
        dd_2020, _ = get_max_drawdown(df_2020_h1[col])
        print(f"2020 H1 Max Drawdown: {dd_2020:.2%}")

except Exception as e:
    print(f"Error: {e}")
