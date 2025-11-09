#!/usr/bin/env python3
"""
Script để chuyển đổi CSV từ định dạng:
symbol,time,open,high,low,close,volume
sang định dạng:
date,stock,open,high,low,close,volume

- Nếu một stock có ít nhất 1 record với volume = 0, xóa TẤT CẢ các record của stock đó
- Xóa các record có symbol = "VNINDEX"
- Chuyển đổi time thành date (YYYY-MM-DD)
- Giữ nguyên tên symbol làm stock (không chuyển thành số)
"""

import pandas as pd
import sys
from pathlib import Path


def transform_csv(input_file, output_file):
    """
    Chuyển đổi CSV từ định dạng cũ sang định dạng mới
    
    Args:
        input_file: Đường dẫn file CSV đầu vào
        output_file: Đường dẫn file CSV đầu ra
    """
    print(f"Đang đọc file: {input_file}")
    
    # Đọc file CSV với low_memory=False để tránh cảnh báo mixed types
    df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Số dòng ban đầu: {len(df)}")
    
    # Lọc bỏ các record có symbol = "VNINDEX" trước
    df = df[df['symbol'] != 'VNINDEX']
    print(f"Số dòng sau khi xóa VNINDEX: {len(df)}")
    
    # Chuyển đổi volume sang numeric để xử lý chính xác
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Tìm các symbol có ít nhất 1 record với volume = 0 hoặc NaN
    symbols_with_zero_volume = df[(df['volume'] == 0) | (df['volume'].isna())]['symbol'].unique()
    print(f"Số lượng symbol có ít nhất 1 record volume = 0: {len(symbols_with_zero_volume)}")
    
    # Xóa TẤT CẢ các record của những symbol đó
    df = df[~df['symbol'].isin(symbols_with_zero_volume)]
    print(f"Số dòng sau khi xóa các symbol có volume = 0: {len(df)}")
    
    # Chuyển đổi time thành date (YYYY-MM-DD)
    df['date'] = pd.to_datetime(df['time']).dt.date
    
    # Giữ nguyên symbol làm stock (đổi tên cột)
    df['stock'] = df['symbol']
    
    # Chọn và sắp xếp lại các cột theo thứ tự yêu cầu
    df_output = df[['date', 'stock', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Sắp xếp theo date và stock
    df_output = df_output.sort_values(['date', 'stock'])
    
    # Lưu file CSV mới
    print(f"Đang ghi file: {output_file}")
    df_output.to_csv(output_file, index=False)
    
    print(f"Hoàn thành! Đã tạo file với {len(df_output)} dòng dữ liệu.")
    print(f"Số lượng stock duy nhất: {df_output['stock'].nunique()}")


if __name__ == "__main__":
    # Nếu có tham số dòng lệnh, sử dụng chúng
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Mặc định sử dụng file stock_prices_all
        input_file = "/Users/nguyensiry/Documents/the_last_dance/agents/MAFIA/data/stock_prices_all_20251108_214916.csv"
        output_file = "/Users/nguyensiry/Documents/the_last_dance/agents/MAFIA/data/stock_prices_transformed.csv"
        print("Sử dụng file mặc định:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
        print()
    
    # Kiểm tra file đầu vào có tồn tại không
    if not Path(input_file).exists():
        print(f"Lỗi: File không tồn tại: {input_file}")
        sys.exit(1)
    
    try:
        transform_csv(input_file, output_file)
    except Exception as e:
        print(f"Lỗi khi xử lý: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

