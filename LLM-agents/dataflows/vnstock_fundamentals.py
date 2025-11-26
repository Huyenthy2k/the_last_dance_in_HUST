from vnstock import Company, Finance
import json
import os


def get_company_overview(ticker: str, output_dir: str = "output") -> dict:
    """
    Hàm 1: Lấy thông tin tổng quan công ty
    
    Args:
        ticker: Mã cổ phiếu (vd: 'ACB', 'VNM')
        output_dir: Thư mục lưu file JSON
    
    Returns:
        dict: Dictionary chứa symbol và company_profile
    """
    try:
        company = Company(symbol=ticker, source='VCI')
        df2 = company.overview()
        
        result = {
            "ticker": ticker,
            "company_profile": None
        }
        
        if not df2.empty:
            result["company_profile"] = df2.get('company_profile', [None])[0] if 'company_profile' in df2.columns else None
        
        # Tạo thư mục output nếu chưa có
        # os.makedirs(output_dir, exist_ok=True)
        
        # # Xuất ra file JSON
        # output_file = os.path.join(output_dir, f"{ticker}_overview.json")
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(result, indent=4, ensure_ascii=False, fp=f)
        
        # print(f"Đã lưu thông tin overview vào: {output_file}")
        return result
        
    except Exception as e:
        print(f"Lỗi khi lấy overview cho {ticker}: {str(e)}")
        return None


def get_fundamentals(ticker: str, period: str = 'quarter', output_dir: str = "output") -> dict:
    """
    Hàm 2: Lấy các chỉ số tài chính cơ bản (ratio)
    
    Args:
        ticker: Mã cổ phiếu
        period: 'quarter' hoặc 'year'
        output_dir: Thư mục lưu file JSON
    
    Returns:
        dict: Dictionary chứa các chỉ số tài chính của phần tử đầu tiên
    """
    try:
        finance = Finance(symbol=ticker, source='VCI')
        df6 = finance.ratio(period=period, lang='en', dropna=True)
        
        result = {}
        
        if not df6.empty:
            # Lấy phần tử đầu tiên (dữ liệu mới nhất)
            latest_data = df6.iloc[0]
            
            # Chuyển đổi tất cả các cột thành dictionary
            for column in df6.columns:
                value = latest_data[column]
                
                # Xử lý tên cột (có thể là tuple trong MultiIndex)
                if isinstance(column, tuple):
                    # Nối các phần tử của tuple thành string
                    column_name = '_'.join(str(c) for c in column if str(c) != '')
                else:
                    column_name = str(column)
                
                # Xử lý các kiểu dữ liệu đặc biệt
                if hasattr(value, 'item'):  # numpy types
                    result[column_name] = value.item()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    result[column_name] = value
                else:
                    result[column_name] = str(value)
        
        # # Tạo thư mục output nếu chưa có
        # os.makedirs(output_dir, exist_ok=True)
        
        # # Xuất ra file JSON
        # output_file = os.path.join(output_dir, f"{ticker}_fundamentals.json")
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(result, indent=4, ensure_ascii=False, fp=f, default=str)
        
        # print(f"Đã lưu fundamentals vào: {output_file}")
        return result
        
    except Exception as e:
        print(f"Lỗi khi lấy fundamentals cho {ticker}: {str(e)}")
        return None


def get_balance_sheet(ticker: str, period: str = 'quarter', output_dir: str = "output") -> dict:
    """
    Hàm 3: Lấy bảng cân đối kế toán
    
    Args:
        ticker: Mã cổ phiếu
        period: 'quarter' hoặc 'year'
        output_dir: Thư mục lưu file JSON
    
    Returns:
        dict: Dictionary chứa dữ liệu balance sheet của phần tử đầu tiên
    """
    try:
        finance = Finance(symbol=ticker, source='VCI')
        df3 = finance.balance_sheet(period=period, lang='en', dropna=True)
        
        result = {}
        
        if not df3.empty:
            # Lấy phần tử đầu tiên (dữ liệu mới nhất)
            latest_data = df3.iloc[0]
            
            # Chuyển đổi tất cả các cột thành dictionary
            for column in df3.columns:
                value = latest_data[column]
                
                # Xử lý tên cột (có thể là tuple trong MultiIndex)
                if isinstance(column, tuple):
                    column_name = '_'.join(str(c) for c in column if str(c) != '')
                else:
                    column_name = str(column)
                
                # Xử lý các kiểu dữ liệu đặc biệt
                if hasattr(value, 'item'):  # numpy types
                    result[column_name] = value.item()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    result[column_name] = value
                else:
                    result[column_name] = str(value)
        
        # # Tạo thư mục output nếu chưa có
        # os.makedirs(output_dir, exist_ok=True)
        
        # # Xuất ra file JSON
        # output_file = os.path.join(output_dir, f"{ticker}_balance_sheet.json")
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(result, indent=4, ensure_ascii=False, fp=f, default=str)
        
        # print(f"Đã lưu balance sheet vào: {output_file}")
        return result
        
    except Exception as e:
        print(f"Lỗi khi lấy balance sheet cho {ticker}: {str(e)}")
        return None


def get_income_statement(ticker: str, period: str = 'quarter', output_dir: str = "output") -> dict:
    """
    Hàm 4: Lấy báo cáo kết quả kinh doanh
    
    Args:
        ticker: Mã cổ phiếu
        period: 'quarter' hoặc 'year'
        output_dir: Thư mục lưu file JSON
    
    Returns:
        dict: Dictionary chứa dữ liệu income statement của phần tử đầu tiên
    """
    try:
        finance = Finance(symbol=ticker, source='VCI')
        df4 = finance.income_statement(period=period, lang='en', dropna=True)
        
        result = {}
        
        if not df4.empty:
            # Lấy phần tử đầu tiên (dữ liệu mới nhất)
            latest_data = df4.iloc[0]
            
            # Chuyển đổi tất cả các cột thành dictionary
            for column in df4.columns:
                value = latest_data[column]
                
                # Xử lý tên cột (có thể là tuple trong MultiIndex)
                if isinstance(column, tuple):
                    column_name = '_'.join(str(c) for c in column if str(c) != '')
                else:
                    column_name = str(column)
                
                # Xử lý các kiểu dữ liệu đặc biệt
                if hasattr(value, 'item'):  # numpy types
                    result[column_name] = value.item()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    result[column_name] = value
                else:
                    result[column_name] = str(value)
        
        # # Tạo thư mục output nếu chưa có
        # os.makedirs(output_dir, exist_ok=True)
        
        # # Xuất ra file JSON
        # output_file = os.path.join(output_dir, f"{ticker}_income_statement.json")
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(result, indent=4, ensure_ascii=False, fp=f, default=str)
        
        # print(f"Đã lưu income statement vào: {output_file}")
        return result
        
    except Exception as e:
        print(f"Lỗi khi lấy income statement cho {ticker}: {str(e)}")
        return None


def get_cashflow(ticker: str, period: str = 'quarter', output_dir: str = "output") -> dict:
    """
    Hàm 5: Lấy báo cáo lưu chuyển tiền tệ
    
    Args:
        ticker: Mã cổ phiếu
        period: 'quarter' hoặc 'year'
        output_dir: Thư mục lưu file JSON
    
    Returns:
        dict: Dictionary chứa dữ liệu cashflow của phần tử đầu tiên
    """
    try:
        finance = Finance(symbol=ticker, source='VCI')
        df5 = finance.cash_flow(period=period, lang='en', dropna=True)
        
        result = {}
        
        if not df5.empty:
            # Lấy phần tử đầu tiên (dữ liệu mới nhất)
            latest_data = df5.iloc[0]
            
            # Chuyển đổi tất cả các cột thành dictionary
            for column in df5.columns:
                value = latest_data[column]
                
                # Xử lý tên cột (có thể là tuple trong MultiIndex)
                if isinstance(column, tuple):
                    column_name = '_'.join(str(c) for c in column if str(c) != '')
                else:
                    column_name = str(column)
                
                # Xử lý các kiểu dữ liệu đặc biệt
                if hasattr(value, 'item'):  # numpy types
                    result[column_name] = value.item()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    result[column_name] = value
                else:
                    result[column_name] = str(value)
        
        # # Tạo thư mục output nếu chưa có
        # os.makedirs(output_dir, exist_ok=True)
        
        # # Xuất ra file JSON
        # output_file = os.path.join(output_dir, f"{ticker}_cashflow.json")
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(result, indent=4, ensure_ascii=False, fp=f, default=str)
        
        # print(f"Đã lưu cashflow vào: {output_file}")
        return result
        
    except Exception as e:
        print(f"Lỗi khi lấy cashflow cho {ticker}: {str(e)}")
        return None


# Test các hàm
if __name__ == "__main__":
    ticker = 'ACB'
    
    print("=" * 50)
    print("Test các hàm lấy dữ liệu tài chính")
    print("=" * 50)
    
    # Test hàm 1: Overview
    print("\n1. Lấy thông tin tổng quan:")
    overview = get_company_overview(ticker)
    
    # Test hàm 2: Fundamentals (Ratio)
    print("\n2. Lấy chỉ số tài chính:")
    fundamentals = get_fundamentals(ticker)
    
    # Test hàm 3: Balance Sheet
    print("\n3. Lấy bảng cân đối kế toán:")
    balance_sheet = get_balance_sheet(ticker)
    
    # Test hàm 4: Income Statement
    print("\n4. Lấy báo cáo kết quả kinh doanh:")
    income_statement = get_income_statement(ticker)
    
    # Test hàm 5: Cashflow
    print("\n5. Lấy báo cáo lưu chuyển tiền tệ:")
    cashflow = get_cashflow(ticker)
    
    print("\n" + "=" * 50)
    print("Hoàn thành! Kiểm tra thư mục 'output' để xem các file JSON.")
    print("=" * 50)