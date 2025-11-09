# Quick Start Guide - MASA Framework

## Cài Đặt Nhanh (3 bước)

### 1. Setup Environment

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Setup Data

```bash
# Kiểm tra và validate data files trong thư mục data/
python setup_data.py
```

**Lưu ý về Data Files:**
- Bạn có thể sử dụng bất kỳ file CSV nào trong thư mục `data/` 
- File phải có định dạng: `date,stock,open,high,low,close,volume`
- Để chỉ định file cụ thể, chỉnh sửa `config.py`:
  ```python
  self.stock_data_file = 'stock_prices_all_20251108_234851.csv'  # Tên file của bạn
  self.index_data_file = 'DJIA_1d_index.csv'  # File index (hoặc None để auto-detect)
  ```
- Để tự động tìm file, đặt `self.stock_data_file = None`

### 3. Chạy Training

```bash
# Chạy với cấu hình mặc định (MASA-MAFIA)
python entrance.py
```

## Các Câu Lệnh Thường Dùng

### Training

```bash
# MASA-MAFIA (mặc định)
python entrance.py

# MASA với DC observer
# (Chỉnh config.py: benchmark_algo = 'MASA-dc')
python entrance.py

# TD3-only
# (Chỉnh config.py: benchmark_algo = 'TD3-Profit')
python entrance.py
```

### Testing

```bash
# Test weight mapping
python test_weight_symbol_mapping.py

# Test components
python test_mafia_components.py

# Quick test
python test_quick_run.py
```

### Utilities

```bash
# Xem ví dụ weight mapping
python example_weight_mapping_usage.py
```

## Cấu Hình Nhanh

Chỉnh sửa `config.py`:

```python
# Dòng 31: Chọn algorithm
self.benchmark_algo = 'MASA-mafia'  # hoặc 'MASA-dc', 'TD3-Profit', etc.

# Dòng 32-34: Market và số stocks
self.market_name = 'DJIA'
self.topK = 10

# Dòng 34: Số epochs
self.num_epochs = 50

# Dòng 90-91: Data files (mới)
self.stock_data_file = 'stock_prices_all_20251108_234851.csv'  # File stock data của bạn
self.index_data_file = None  # File index (None để auto-detect)
```

## Kết Quả

Results được lưu tại:
```
res/RLcontroller/TD3/DJIA-10/YYYY-MM-DD-HH-MM-SS/
```

Các file quan trọng:
- `*_profile.csv` - Performance metrics
- `*_actions.csv` - Portfolio weights với symbols
- `*_stepdata.csv` - Step-by-step data

## Troubleshooting Nhanh

| Lỗi | Giải pháp |
|-----|-----------|
| "Cannot load data" | Chạy `python setup_data.py` |
| "CUDA out of memory" | Giảm `batch_size` hoặc dùng CPU |
| Import errors | `pip install -r requirements.txt` |

## Xem Chi Tiết

Xem `README.md` để có hướng dẫn đầy đủ.

