from vnstock import Vnstock
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import time
from database.connect_db import engine
from vnstock import Quote

def fetch_and_append(symbol):
    with engine.connect() as conn:
        # Lấy ngày cuối cùng có dữ liệu trong DB cho mã này
        result = conn.execute(text(f"""
            SELECT MAX(time) FROM stock_prices WHERE symbol = :symbol
        """), {"symbol": symbol}).fetchone()
        last_date = result[0] if result and result[0] else None

    # Nếu chưa có dữ liệu => lấy từ 2017
    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d') if last_date else '2017-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"🔄 Cập nhật {symbol} từ {start_date} → {end_date}")
    if symbol.strip() == "VNINDEX":
        quote = Quote(symbol='VNINDEX', source='VCI')
        df = quote.history(start=start_date, end=end_date, interval='1D')
    else:
        quote = Quote(symbol=symbol, source='VCI')
        df = quote.history(start=start_date, end=end_date, interval='1D')

    if df.empty:
        print(f"⚠️ Không có dữ liệu mới cho {symbol}")
        return

    df['symbol'] = symbol
    df = df.rename(columns={
        'time': 'time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

    # Ghi thêm dữ liệu mới
    df.to_sql('stock_prices', con=engine, if_exists='append', index=False)
    print(f"✅ Đã thêm {len(df)} dòng mới cho {symbol}")

if __name__ == "__main__":

    symbols_df = pd.read_csv("D:/project/thesis/crawl_data/price_stock_market/symbols.csv")
    for symbol in symbols_df['symbol']:
       try:
           fetch_and_append(symbol)
       except Exception as e:
           print(f"❌ Lỗi khi cập nhật {symbol}: {e}")
