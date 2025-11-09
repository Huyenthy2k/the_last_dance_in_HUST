from vnstock import stock_historical_data
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text

# Add parent directory to path to import connect_db
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from database.connect_db import engine

def get_symbols_from_db():
    """Get all symbols from symbol_reference table"""
    query = """
        SELECT symbol_id, symbol, exchange 
        FROM raw.symbol_reference 
        ORDER BY exchange, symbol
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data for a symbol"""
    try:
        # vnstock3 API: stock_historical_data(symbol, start_date, end_date, resolution, type)
        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            resolution='1D',
            type='stock'
        )
        
        if df is not None and not df.empty:
            return df
        return None
    except Exception as e:
        print(f"  Error fetching data for {symbol}: {e}")
        return None

def process_and_insert_data(symbol_info, df_prices):
    """Process price data and insert into database"""
    try:
        # Prepare data for insertion
        # vnstock columns: time, open, high, low, close, volume
        # DB columns: symbol_id, symbol, exchange, trade_date, open, high, low, close, volume
        
        insert_data = pd.DataFrame({
            'symbol_id': symbol_info['symbol_id'],
            'symbol': symbol_info['symbol'],
            'exchange': symbol_info['exchange'],
            'trade_date': pd.to_datetime(df_prices['time']).dt.date,
            'open': df_prices['open'],
            'high': df_prices['high'],
            'low': df_prices['low'],
            'close': df_prices['close'],
            'volume': df_prices['volume'].fillna(0).astype('int64')
        })
        
        # Insert into database (on conflict do nothing to avoid duplicates)
        with engine.connect() as conn:
            # Use raw SQL with ON CONFLICT to handle duplicates
            for _, row in insert_data.iterrows():
                sql = text("""
                    INSERT INTO raw.prices_daily 
                    (symbol_id, symbol, exchange, trade_date, open, high, low, close, volume)
                    VALUES 
                    (:symbol_id, :symbol, :exchange, :trade_date, :open, :high, :low, :close, :volume)
                    ON CONFLICT (symbol_id, trade_date) DO NOTHING
                """)
                conn.execute(sql, {
                    'symbol_id': int(row['symbol_id']),
                    'symbol': row['symbol'],
                    'exchange': row['exchange'],
                    'trade_date': row['trade_date'],
                    'open': float(row['open']) if pd.notna(row['open']) else None,
                    'high': float(row['high']) if pd.notna(row['high']) else None,
                    'low': float(row['low']) if pd.notna(row['low']) else None,
                    'close': float(row['close']) if pd.notna(row['close']) else None,
                    'volume': int(row['volume']) if pd.notna(row['volume']) else None
                })
            conn.commit()
        
        return len(insert_data)
    except Exception as e:
        print(f"  Error inserting data: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    # Configuration
    # Default: get last 1 year of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # You can customize the date range here
    # start_date = datetime(2024, 1, 1)
    # end_date = datetime(2024, 12, 31)
    
    print(f"Fetching stock prices from {start_date.date()} to {end_date.date()}")
    print("=" * 70)
    
    # Get all symbols from database
    print("\nReading symbols from database...")
    symbols_df = get_symbols_from_db()
    
    if symbols_df.empty:
        print("No symbols found in database. Please run crawl_symbol_code.py first.")
        sys.exit(1)
    
    print(f"Found {len(symbols_df)} symbols to process\n")
    
    # Process each symbol
    total_inserted = 0
    success_count = 0
    fail_count = 0
    
    for idx, symbol_info in symbols_df.iterrows():
        symbol = symbol_info['symbol']
        exchange = symbol_info['exchange']
        
        print(f"[{idx+1}/{len(symbols_df)}] Processing {symbol} ({exchange})...", end=' ')
        
        # Fetch data
        df_prices = fetch_stock_data(symbol, start_date, end_date)
        
        if df_prices is not None and not df_prices.empty:
            # Insert data
            inserted = process_and_insert_data(symbol_info, df_prices)
            total_inserted += inserted
            success_count += 1
            print(f"✓ Inserted {inserted} records")
        else:
            fail_count += 1
            print(f"✗ No data")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total symbols processed: {len(symbols_df)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"Total records inserted: {total_inserted}")
    print("=" * 70)

if __name__ == "__main__":
    main()
