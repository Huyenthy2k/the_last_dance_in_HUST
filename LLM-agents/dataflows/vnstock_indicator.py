import pandas as pd
from vnstock_ta import DataSource, Indicator
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import os


def _find_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Find a column in `df` containing all `keywords` (case-insensitive).

    Returns the first matching column, or the last column as a fallback.
    """
    if df is None or df.empty:
        return None

    keys = [k.lower() for k in keywords if k]
    # try to find a column that contains all keywords
    for col in df.columns:
        lname = str(col).lower()
        if all(k in lname for k in keys):
            return col

    # fallback: return last column
    return df.columns[-1]


def _get_column_name_from_result(res, ta_df: pd.DataFrame, prefer_keywords: list[str] | None = None) -> str | None:
    """Derive a useful column name from the returned result `res` or `ta_df`.

    prefer_keywords: list of fragments to search for (e.g. ['sma','50']).
    """
    # If result is a Series
    if isinstance(res, pd.Series):
        if getattr(res, 'name', None):
            return res.name
        # if no name, fallthrough to search ta_df

    # If result is a DataFrame
    if isinstance(res, pd.DataFrame):
        # try to find a column inside res
        if prefer_keywords:
            col = _find_column(res, prefer_keywords)
            if col is not None:
                return col
        # fallback to last column
        return res.columns[-1] if len(res.columns) else None

    # If res is None or doesn't help, inspect ta_df
    if ta_df is not None and not ta_df.empty:
        return _find_column(ta_df, prefer_keywords or [])

    return None


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    interval: str = "1D",
    time_period: int = 14,
    output_dir: str = "output"
) -> str:
    """
    Trả về giá trị các chỉ báo kỹ thuật sử dụng vnstock_ta.

    Args:
        symbol: Mã cổ phiếu (vd: 'VCI', 'ACB')
        indicator: Tên chỉ báo kỹ thuật cần lấy
        curr_date: Ngày giao dịch hiện tại, định dạng YYYY-MM-DD
        look_back_days: Số ngày nhìn lại
        interval: Khung thời gian (1D - ngày, 1W - tuần, 1M - tháng)
        time_period: Số chu kỳ để tính toán
        output_dir: Thư mục lưu file JSON

    Returns:
        String chứa giá trị chỉ báo và mô tả
    """
    
    supported_indicators = {
        "close_50_sma": ("50 SMA", 50),
        "close_200_sma": ("200 SMA", 200),
        "close_10_ema": ("10 EMA", 10),
        "sma": ("SMA", time_period),
        "ema": ("EMA", time_period),
        "macd": ("MACD", None),
        "macds": ("MACD Signal", None),
        "macdh": ("MACD Histogram", None),
        "rsi": ("RSI", time_period),
        "boll": ("Bollinger Middle", 20),
        "boll_ub": ("Bollinger Upper Band", 20),
        "boll_lb": ("Bollinger Lower Band", 20),
        "atr": ("ATR", time_period),
        "vwma": ("VWMA", 20)
    }

    indicator_descriptions = {
        "close_50_sma": "50 SMA: Chỉ báo xu hướng trung hạn. Sử dụng: Xác định hướng xu hướng và đóng vai trò hỗ trợ/kháng cự động. Lưu ý: Chậm hơn giá; kết hợp với các chỉ báo nhanh hơn để có tín hiệu kịp thời.",
        "close_200_sma": "200 SMA: Chuẩn mực xu hướng dài hạn. Sử dụng: Xác nhận xu hướng thị trường tổng thể và xác định golden/death cross. Lưu ý: Phản ứng chậm; phù hợp cho xác nhận xu hướng chiến lược hơn là giao dịch thường xuyên.",
        "close_10_ema": "10 EMA: Đường trung bình ngắn hạn nhạy bén. Sử dụng: Bắt các thay đổi nhanh trong động lượng và điểm vào tiềm năng. Lưu ý: Dễ bị nhiễu trong thị trường đi ngang; sử dụng cùng đường trung bình dài hơn để lọc tín hiệu giả.",
        "sma": "SMA: Đường trung bình di động đơn giản. Sử dụng: Xác định xu hướng và mức hỗ trợ/kháng cự. Lưu ý: Chậm phản ứng với thay đổi giá đột ngột.",
        "ema": "EMA: Đường trung bình di động hàm mũ. Sử dụng: Nhạy hơn với thay đổi giá gần đây, tốt cho xu hướng ngắn hạn. Lưu ý: Có thể tạo nhiều tín hiệu giả trong thị trường biến động.",
        "macd": "MACD: Tính động lượng qua chênh lệch các EMA. Sử dụng: Tìm kiếm crossover và phân kỳ như tín hiệu thay đổi xu hướng. Lưu ý: Xác nhận với các chỉ báo khác trong thị trường biến động thấp.",
        "macds": "MACD Signal: EMA làm mượt đường MACD. Sử dụng: Sử dụng crossover với đường MACD để kích hoạt giao dịch. Lưu ý: Nên là một phần của chiến lược rộng hơn để tránh tín hiệu giả.",
        "macdh": "MACD Histogram: Hiển thị khoảng cách giữa MACD và signal line. Sử dụng: Trực quan hóa sức mạnh động lượng và phát hiện phân kỳ sớm. Lưu ý: Có thể biến động; bổ sung với các bộ lọc trong thị trường nhanh.",
        "rsi": "RSI: Đo động lượng để đánh dấu điều kiện quá mua/quá bán. Sử dụng: Áp dụng ngưỡng 70/30 và theo dõi phân kỳ để báo hiệu đảo chiều. Lưu ý: Trong xu hướng mạnh, RSI có thể ở mức cực; luôn kiểm tra chéo với phân tích xu hướng.",
        "boll": "Bollinger Middle: Đường SMA 20 làm cơ sở cho Bollinger Bands. Sử dụng: Đóng vai trò chuẩn động cho chuyển động giá. Lưu ý: Kết hợp với dải trên và dưới để phát hiện breakout hoặc đảo chiều hiệu quả.",
        "boll_ub": "Bollinger Upper Band: Thường là 2 độ lệch chuẩn phía trên đường giữa. Sử dụng: Báo hiệu điều kiện quá mua tiềm năng và vùng breakout. Lưu ý: Xác nhận tín hiệu với công cụ khác; giá có thể đi dọc dải trong xu hướng mạnh.",
        "boll_lb": "Bollinger Lower Band: Thường là 2 độ lệch chuẩn phía dưới đường giữa. Sử dụng: Chỉ ra điều kiện quá bán tiềm năng. Lưu ý: Sử dụng phân tích bổ sung để tránh tín hiệu đảo chiều giả.",
        "atr": "ATR: Trung bình true range để đo biến động. Sử dụng: Đặt mức stop-loss và điều chỉnh kích thước vị thế dựa trên biến động thị trường hiện tại. Lưu ý: Là thước đo phản ứng, sử dụng như một phần của chiến lược quản lý rủi ro rộng hơn.",
        "vwma": "VWMA: Đường trung bình di động có trọng số khối lượng. Sử dụng: Xác nhận xu hướng bằng cách tích hợp hành động giá với dữ liệu khối lượng. Lưu ý: Theo dõi kết quả sai lệch từ đỉnh khối lượng; sử dụng kết hợp với các phân tích khối lượng khác."
    }

    if indicator not in supported_indicators:
        raise ValueError(
            f"Chỉ báo {indicator} không được hỗ trợ. Vui lòng chọn từ: {list(supported_indicators.keys())}"
        )

    try:
        # Tính toán ngày bắt đầu
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        before = curr_date_dt - relativedelta(days=look_back_days)
        
        # Thêm buffer để đảm bảo có đủ dữ liệu cho tính toán
        buffer_days = max(200, time_period * 3)  # Buffer lớn hơn cho các chỉ báo như 200 SMA
        start_date = before - relativedelta(days=buffer_days)
        
        # Lấy dữ liệu giá từ vnstock
        data = DataSource(
            symbol=symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=curr_date,
            interval=interval,
            source='VCI'
        ).get_data()
        
        if data.empty:
            return f"Lỗi: Không có dữ liệu cho {symbol}"
        
        # Khởi tạo Indicator
        ta = Indicator(data)
        
        # Tính toán chỉ báo tương ứng
        result_data = None
        column_name = None
        
        # Compute indicator and try to derive the created column name robustly.
        if indicator == "close_50_sma":
            res = ta.sma(length=50)
            column_name = _get_column_name_from_result(res, data, prefer_keywords=['sma', '50']) or f"SMA_50"
        elif indicator == "close_200_sma":
            res = ta.sma(length=200)
            column_name = _get_column_name_from_result(res, data, prefer_keywords=['sma', '200']) or f"SMA_200"
        elif indicator == "close_10_ema":
            res = ta.ema(length=10)
            column_name = _get_column_name_from_result(res, data, prefer_keywords=['ema', '10']) or f"EMA_10"
        elif indicator == "sma":
            res = ta.sma(length=time_period)
            column_name = _get_column_name_from_result(res, data, prefer_keywords=['sma', str(time_period)]) or f"SMA_{time_period}"
        elif indicator == "ema":
            res = ta.ema(length=time_period)
            column_name = _get_column_name_from_result(res, data, prefer_keywords=['ema', str(time_period)]) or f"EMA_{time_period}"
        elif indicator == "rsi":
            res = ta.rsi(length=time_period)
            column_name = _get_column_name_from_result(res, data, prefer_keywords=['rsi', str(time_period)]) or f"RSI_{time_period}"
        elif indicator in ["macd", "macds", "macdh"]:
            res = ta.macd(fast=12, slow=26, signal=9)
            # prefer any column containing 'macd'
            col = _get_column_name_from_result(res, data, prefer_keywords=['macd'])
            if indicator == "macd":
                column_name = col or "MACD_12_26_9"
            elif indicator == "macds":
                # pick the signal column if available
                column_name = _find_column(res if isinstance(res, pd.DataFrame) else data, ['signal', 'macd']) or col or "MACDs_12_26_9"
            else:  # macdh
                column_name = _find_column(res if isinstance(res, pd.DataFrame) else data, ['hist', 'macd']) or col or "MACDh_12_26_9"
        elif indicator in ["boll", "boll_ub", "boll_lb"]:
            res = ta.bbands(length=20, std=2)
            if indicator == "boll":
                column_name = _get_column_name_from_result(res, data, prefer_keywords=['bb', 'm', 'middle']) or _get_column_name_from_result(res, data, prefer_keywords=['bbm', '20']) or "BBM_20_2.0"
            elif indicator == "boll_ub":
                column_name = _get_column_name_from_result(res, data, prefer_keywords=['bb', 'upper', 'ub']) or "BBU_20_2.0"
            else:  # boll_lb
                column_name = _get_column_name_from_result(res, data, prefer_keywords=['bb', 'lower', 'lb']) or "BBL_20_2.0"
        elif indicator == "atr":
            res = ta.atr(length=time_period)
            column_name = _get_column_name_from_result(res, data, prefer_keywords=['atr', str(time_period)]) or f"ATR_{time_period}"
        elif indicator == "vwma":
            res = ta.vwma(length=20)
            column_name = _get_column_name_from_result(res, data, prefer_keywords=['vwma', '20']) or "VWMA_20"
        
        # Build a working DataFrame that contains the original price `data`
        # plus the computed indicator columns (from `res`). This avoids
        # relying on an internal `ta.df` attribute which may not exist.
        working_df = data.copy()

        # If `res` is a Series, insert it into working_df under the
        # discovered `column_name`. If DataFrame, concat columns.
        if isinstance(res, pd.Series):
            colname = column_name or getattr(res, 'name', None) or 'value'
            # align series to working_df index
            working_df[colname] = res.reindex(working_df.index)
            column_name = colname
        elif isinstance(res, pd.DataFrame):
            # join (prefer columns from res)
            working_df = pd.concat([working_df, res], axis=1)
            # if column_name still None, try to get one from res
            if column_name is None and len(res.columns):
                column_name = res.columns[-1]

        # Ensure index is datetime-like for filtering
        try:
            working_df.index = pd.to_datetime(working_df.index)
        except Exception:
            pass

        before_dt = pd.to_datetime(before)
        curr_dt = pd.to_datetime(curr_date)

        filtered_data = working_df.loc[(working_df.index >= before_dt) & (working_df.index <= curr_dt)]
        
        if column_name not in filtered_data.columns:
            return f"Lỗi: Cột {column_name} không tìm thấy trong dữ liệu. Các cột có sẵn: {list(filtered_data.columns)}"
        
        # Tạo chuỗi kết quả
        ind_string = ""
        result_dict = {}
        
        for idx, row in filtered_data.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            value = row[column_name]
            
            if pd.notna(value):
                ind_string += f"{date_str}: {value:.4f}\n"
                result_dict[date_str] = float(value)
        
        if not ind_string:
            ind_string = "Không có dữ liệu cho khoảng thời gian được chỉ định.\n"
        
        # Lưu kết quả ra file JSON
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{symbol}_{indicator}_{curr_date}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, indent=4, ensure_ascii=False, fp=f)
        
        print(f"Đã lưu {indicator} vào: {output_file}")
        
        # Tạo chuỗi kết quả
        result_str = (
            f"## {indicator.upper()} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + ind_string
            + "\n\n"
            + indicator_descriptions.get(indicator, "Không có mô tả.")
        )
        
        return result_str
        
    except Exception as e:
        error_msg = f"Lỗi khi lấy dữ liệu chỉ báo {indicator} cho {symbol}: {str(e)}"
        print(error_msg)
        return error_msg


# Test hàm
if __name__ == "__main__":
    symbol = 'VCI'
    curr_date = '2024-06-10'
    look_back_days = 30
    
    print("=" * 70)
    print("Test các chỉ báo kỹ thuật với vnstock_ta")
    print("=" * 70)
    
    # Test một số chỉ báo
    test_indicators = ['close_50_sma', 'ema', 'rsi', 'macd', 'boll', 'atr']
    
    for indicator in test_indicators:
        print(f"\n{'='*70}")
        print(f"Testing {indicator.upper()}")
        print('='*70)
        result = get_indicator(
            symbol=symbol,
            indicator=indicator,
            curr_date=curr_date,
            look_back_days=look_back_days,
            time_period=14
        )
        print(result[:500])  # In 500 ký tự đầu để xem trước
        print("...\n")