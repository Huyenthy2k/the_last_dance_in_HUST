# Hướng Dẫn Tách Data Thành Train/Validation/Test Sets

## Tổng Quan

MAFIA framework tự động tách data thành training, validation, và test sets **trong quá trình training** dựa trên date ranges được định nghĩa trong `config.py`. Tuy nhiên, bạn cũng có thể tách data trước và lưu thành các file CSV riêng để dễ quản lý.

## Cách Data Được Tách Hiện Tại

### 1. Date Ranges trong Config

Trong `config.py`, date ranges được định nghĩa như sau:

```python
date_split_dict = {            
    1: {
        'train_date_start': '2017-01-03 00:00:00',
        'train_date_end': '2021-12-31 23:59:59',
        'valid_date_start': '2022-01-01 00:00:00',
        'valid_date_end': '2023-12-31 23:59:59',
        'test_date_start': '2024-01-01 00:00:00',
        'test_date_end': '2025-11-06 23:59:59'
    },
}
```

### 2. Tách Data Trong Training

Khi chạy `entrance.py`, data được tách tự động trong `utils/featGen.py`:

- **Training set**: Data từ `train_date_start` đến `train_date_end`
- **Validation set**: Data từ `valid_date_start` đến `valid_date_end`
- **Test set**: Data từ `test_date_start` đến `test_date_end`

### 3. Data Format

Data file CSV cần có các columns:
- `date`: Ngày giao dịch
- `stock`: Mã cổ phiếu
- `open`, `high`, `low`, `close`: Giá mở/đóng/cao/thấp
- `volume`: Khối lượng giao dịch

## Sử Dụng Script Tách Data

### Cách 1: Sử Dụng Date Ranges Từ Config (Khuyến nghị)

```bash
# Tách data sử dụng date ranges từ config.py
python split_data.py --use-config
```

Script sẽ:
1. Đọc date ranges từ `config.py`
2. Tìm data file từ `config.dataDir` và `config.stock_data_file`
3. Tách data theo các date ranges
4. Lưu thành các file:
   - `{original_filename}_train.csv`
   - `{original_filename}_valid.csv`
   - `{original_filename}_test.csv`

### Cách 2: Chỉ Định Date Ranges Thủ Công

```bash
# Tách data với date ranges tùy chỉnh
python split_data.py \
    --data-file ./data/stock_prices_all_20251108_234851.csv \
    --output-dir ./data/split \
    --train-start 2017-01-03 \
    --train-end 2021-12-31 \
    --valid-start 2022-01-01 \
    --valid-end 2023-12-31 \
    --test-start 2024-01-01 \
    --test-end 2025-11-06
```

### Cách 3: Chỉ Tách Training Set

```bash
# Chỉ tách training set (không có validation và test)
python split_data.py \
    --data-file ./data/stock_prices_all_20251108_234851.csv \
    --output-dir ./data/split \
    --train-start 2017-01-03 \
    --train-end 2021-12-31
```

## Ví Dụ Sử Dụng

### Example 1: Tách Data Từ Config

```bash
cd agents/MAFIA
python split_data.py --use-config
```

**Output:**
```
============================================================
Splitting Data from Config
============================================================
Data file: ./data/stock_prices_all_20251108_234851.csv
Output directory: ./data

Date ranges:
  Training: 2017-01-03 to 2021-12-31
  Validation: 2022-01-01 to 2023-12-31
  Test: 2024-01-01 to 2025-11-06
============================================================

Reading data from: ./data/stock_prices_all_20251108_234851.csv

✓ Training set:
  File: ./data/stock_prices_all_20251108_234851_train.csv
  Rows: 12,500
  Date range: 2017-01-03 to 2021-12-31
  Stocks: 10

✓ Validation set:
  File: ./data/stock_prices_all_20251108_234851_valid.csv
  Rows: 5,000
  Date range: 2022-01-01 to 2023-12-31
  Stocks: 10

✓ Test set:
  File: ./data/stock_prices_all_20251108_234851_test.csv
  Rows: 4,500
  Date range: 2024-01-01 to 2025-11-06
  Stocks: 10

============================================================
Summary:
  Total original rows: 22,000
  Training rows: 12,500 (56.8%)
  Validation rows: 5,000 (22.7%)
  Test rows: 4,500 (20.5%)
============================================================

✓ Data splitting completed successfully!
```

### Example 2: Tách Data Với Date Ranges Tùy Chỉnh

```python
# Sử dụng trong Python script
from split_data import split_data_by_dates

results = split_data_by_dates(
    data_file='./data/stock_prices_all_20251108_234851.csv',
    output_dir='./data/split',
    train_start='2017-01-03',
    train_end='2021-12-31',
    valid_start='2022-01-01',
    valid_end='2023-12-31',
    test_start='2024-01-01',
    test_end='2025-11-06',
    verbose=True
)

print(f"Training file: {results['train']['file']}")
print(f"Validation file: {results['valid']['file']}")
print(f"Test file: {results['test']['file']}")
```

## Sử Dụng Data Đã Tách

### Option 1: Sử Dụng File Gốc (Mặc định)

Framework sẽ tự động tách data trong quá trình training. Bạn không cần làm gì thêm.

### Option 2: Sử Dụng File Đã Tách

Nếu bạn muốn sử dụng các file đã tách riêng, bạn có thể:

1. **Chỉnh sửa config để trỏ đến file đã tách:**
```python
# Trong config.py, thay đổi stock_data_file
self.stock_data_file = 'stock_prices_all_20251108_234851_train.csv'  # Chỉ training
```

2. **Hoặc tạo script riêng để load từng set:**
```python
import pandas as pd

# Load từng set riêng
train_data = pd.read_csv('./data/split/stock_prices_all_20251108_234851_train.csv')
valid_data = pd.read_csv('./data/split/stock_prices_all_20251108_234851_valid.csv')
test_data = pd.read_csv('./data/split/stock_prices_all_20251108_234851_test.csv')
```

## Thay Đổi Date Ranges

### Trong Config.py

```python
# Trong config.py, chỉnh sửa date_split_dict
date_split_dict = {            
    1: {
        'train_date_start': '2018-01-01 00:00:00',  # Thay đổi ngày
        'train_date_end': '2022-12-31 23:59:59',
        'valid_date_start': '2023-01-01 00:00:00',
        'valid_date_end': '2024-12-31 23:59:59',
        'test_date_start': '2025-01-01 00:00:00',
        'test_date_end': '2025-12-31 23:59:59'
    },
}
```

### Tạo Period Mode Mới

```python
# Thêm period mode mới
date_split_dict = {            
    1: {...},  # Period mode 1
    2: {  # Period mode 2 - mới
        'train_date_start': '2015-01-01 00:00:00',
        'train_date_end': '2019-12-31 23:59:59',
        'valid_date_start': '2020-01-01 00:00:00',
        'valid_date_end': '2021-12-31 23:59:59',
        'test_date_start': '2022-01-01 00:00:00',
        'test_date_end': '2023-12-31 23:59:59'
    },
}

# Sử dụng period mode 2
self.period_mode = 2
```

## Kiểm Tra Data Đã Tách

### Xem Thống Kê

```python
import pandas as pd

# Load và kiểm tra
train = pd.read_csv('./data/stock_prices_all_20251108_234851_train.csv')
valid = pd.read_csv('./data/stock_prices_all_20251108_234851_valid.csv')
test = pd.read_csv('./data/stock_prices_all_20251108_234851_test.csv')

print("Training set:")
print(f"  Rows: {len(train):,}")
print(f"  Date range: {train['date'].min()} to {train['date'].max()}")
print(f"  Stocks: {train['stock'].nunique()}")

print("\nValidation set:")
print(f"  Rows: {len(valid):,}")
print(f"  Date range: {valid['date'].min()} to {valid['date'].max()}")
print(f"  Stocks: {valid['stock'].nunique()}")

print("\nTest set:")
print(f"  Rows: {len(test):,}")
print(f"  Date range: {test['date'].min()} to {test['date'].max()}")
print(f"  Stocks: {test['stock'].nunique()}")
```

### Kiểm Tra Overlap

```python
# Kiểm tra xem có overlap giữa các sets không
train_dates = set(pd.to_datetime(train['date']).dt.date)
valid_dates = set(pd.to_datetime(valid['date']).dt.date)
test_dates = set(pd.to_datetime(test['date']).dt.date)

overlap_train_valid = train_dates & valid_dates
overlap_valid_test = valid_dates & test_dates
overlap_train_test = train_dates & test_dates

if overlap_train_valid:
    print(f"⚠️ Warning: {len(overlap_train_valid)} dates overlap between train and valid")
if overlap_valid_test:
    print(f"⚠️ Warning: {len(overlap_valid_test)} dates overlap between valid and test")
if overlap_train_test:
    print(f"⚠️ Warning: {len(overlap_train_test)} dates overlap between train and test")
```

## Troubleshooting

### Lỗi: "Cannot find data file"

**Giải pháp:**
```bash
# Kiểm tra file tồn tại
ls -lh ./data/*.csv

# Hoặc chỉ định đường dẫn đầy đủ
python split_data.py --data-file /full/path/to/data.csv --use-config
```

### Lỗi: "Date range is empty"

**Nguyên nhân:** Date range không có data trong file

**Giải pháp:**
```python
# Kiểm tra date range trong data
import pandas as pd
data = pd.read_csv('./data/stock_prices_all_20251108_234851.csv')
data['date'] = pd.to_datetime(data['date'])
print(f"Data date range: {data['date'].min()} to {data['date'].max()}")

# Đảm bảo date ranges trong config nằm trong range này
```

### Lỗi: "Validation failed"

**Nguyên nhân:** File data không đúng format

**Giải pháp:**
```python
# Kiểm tra format
from utils.data_validator import validate_stock_data_file
is_valid, error_msg = validate_stock_data_file('./data/your_file.csv')
print(f"Valid: {is_valid}, Error: {error_msg}")
```

## Best Practices

### 1. Tỷ Lệ Train/Valid/Test

Khuyến nghị:
- **Training**: 60-70% data
- **Validation**: 15-20% data
- **Test**: 15-20% data

### 2. Không Overlap

Đảm bảo các date ranges không overlap:
- Training end < Validation start
- Validation end < Test start

### 3. Chronological Order

Giữ thứ tự thời gian:
- Training (cũ nhất) → Validation → Test (mới nhất)

### 4. Lưu Backup

Trước khi tách, backup file gốc:
```bash
cp ./data/stock_prices_all_20251108_234851.csv ./data/stock_prices_all_20251108_234851_backup.csv
```

## Kết Luận

- ✅ Framework tự động tách data trong quá trình training
- ✅ Script `split_data.py` cho phép tách trước và lưu thành file riêng
- ✅ Có thể tùy chỉnh date ranges trong `config.py`
- ✅ Dễ dàng kiểm tra và validate data đã tách

**Lưu ý:** Bạn không cần tách data trước khi training. Framework sẽ tự động làm điều này. Script `split_data.py` chỉ hữu ích khi bạn muốn:
- Kiểm tra data trước khi training
- Sử dụng các file đã tách riêng
- Chia sẻ data với người khác

