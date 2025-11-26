from vnstock_data import Trading
trading = Trading(symbol='MSN', source='VCI')
START_DATE = '2024-01-02'
END_DATE = '2024-11-08'

df = trading.insider_deal(start='2024-01-02', end=END_DATE)

print(df)