import pandas as pd
from datetime import datetime
from vnstock import Quote


def get_stock_prices(symbol: str, start_date: str, end_date: str) -> str:
    """
    Lấy dữ liệu giá cổ phiếu từ vnstock cho mã cổ phiếu và khoảng thời gian chỉ định.
    
    Args:
        symbol (str): Mã cổ phiếu, ví dụ: VCI, FPT, ACB
        start_date (str): Ngày bắt đầu định dạng yyyy-mm-dd
        end_date (str): Ngày kết thúc định dạng yyyy-mm-dd
        
    Returns:
        str: Chuỗi định dạng chứa dữ liệu giá cổ phiếu cho mã và khoảng thời gian chỉ định.
    """
    try:
        # Validate date format
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get stock data from vnstock
        quote = Quote(symbol=symbol, source='VCI')
        df = quote.history(start=start_date, end=end_date, interval='1D')
        
        if df is None or df.empty:
            return f"Không có dữ liệu giá cổ phiếu cho {symbol} trong khoảng thời gian từ {start_date} đến {end_date}"
        
        # Rename columns for clarity
        df = df.rename(columns={
            'time': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # Select relevant columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Format the output
        result = f"## Dữ liệu giá cổ phiếu {symbol} từ {start_date} đến {end_date}:\n\n"
        result += df.to_string(index=False)
        result += f"\n\nTổng số ngày giao dịch: {len(df)}"
        
        return result
        
    except ValueError as e:
        return f"Lỗi định dạng ngày: {str(e)}. Vui lòng sử dụng định dạng yyyy-mm-dd"
    except Exception as e:
        return f"Lỗi khi lấy dữ liệu giá cổ phiếu cho {symbol}: {str(e)}"


# Test hàm
if __name__ == "__main__":
    # Test với mã VCI
    symbol = 'VCI'
    start_date = '2024-01-01'
    end_date = '2024-06-10'
    
    print("=" * 70)
    print("Test hàm get_stock_prices")
    print("=" * 70)
    
    result = get_stock_prices(symbol, start_date, end_date)
    print(result)