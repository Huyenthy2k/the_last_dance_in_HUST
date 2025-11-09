from vnstock import Listing
import os
import sys
import pandas as pd

# Add parent directory to path to import connect_db
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from database.connect_db import engine

listing = Listing()

# Get all symbols from all exchanges at once
print("Fetching all symbols from exchanges...")
df = listing.symbols_by_exchange()

if df is None or df.empty:
    print("No data fetched. Exiting.")
    sys.exit(1)

# print(f"Total symbols fetched: {len(df)}")
# print(f"Columns: {list(df.columns)}")

# Filter out rows without exchange
# Check if 'exchange' column exists and filter
if 'exchange' in df.columns:
    before_filter = len(df)
    df = df[df['exchange'].notna() & (df['exchange'] != '')]
    after_filter = len(df)
    print(f"Filtered out {before_filter - after_filter} symbols without exchange")
    # print(f"Remaining symbols: {after_filter}")
else:
    print("Warning: 'exchange' column not found in data")
    # print("Available columns:", list(df.columns))

# Select and rename columns to match database schema
# vnstock columns: symbol, organ_name, exchange
# DB columns: symbol, exchange, name
try:
    # Create dataframe with correct columns
    insert_df = pd.DataFrame({
        'symbol': df['symbol'].astype(str),
        'exchange': df['exchange'].astype(str),
        'name': df['organ_name'].fillna('').astype(str) if 'organ_name' in df.columns else ''
    })
    
    # Remove duplicates based on symbol and exchange
    insert_df = insert_df.drop_duplicates(subset=['symbol', 'exchange'])
    
    print(f"Unique symbols to insert: {len(insert_df)}")
    
    # Insert into database
    print("\nInserting data into database...")
    insert_df.to_sql(
        name='symbol_reference',
        schema='raw',
        con=engine,
        if_exists='append',  # Change to 'replace' if you want to recreate the table
        index=False,
        method='multi'
    )
    
    print(f"✓ Successfully inserted {len(insert_df)} symbols into raw.symbol_reference")
    
    # Optional: Save to CSV for backup
    output_path = os.path.join(os.path.dirname(__file__), 'symbols.csv')
    insert_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Backup saved to {output_path}")
    
except Exception as e:
    print(f"✗ Error inserting data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)