# MASA Framework - Hướng Dẫn Chạy Pipeline Hoàn Chỉnh

## Tổng Quan

MASA (Multi-Agent Solver-based Agent) là framework cho portfolio optimization sử dụng Reinforcement Learning kết hợp với solver-based controller và market observer. Framework hỗ trợ nhiều algorithms và market observers khác nhau.

## Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên
- **OS**: Linux, macOS, hoặc Windows (với WSL)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **GPU**: Tùy chọn (CUDA-compatible GPU cho training nhanh hơn)
- **Disk**: Tối thiểu 5GB cho data và results

## Cài Đặt

### 1. Clone Repository

```bash
cd /path/to/your/workspace
git clone <repository-url>
cd "MASA copy"
```

### 2. Tạo Virtual Environment (Khuyến nghị)

#### Option A: Sử dụng venv

```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt virtual environment
# Trên Linux/macOS:
source venv/bin/activate
# Trên Windows:
venv\Scripts\activate
```

#### Option B: Sử dụng UV (Nhanh hơn)

Xem hướng dẫn chi tiết trong `UV_GUIDE.md`

```bash
# Cài đặt UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Tạo và kích hoạt environment
uv venv
source .venv/bin/activate  # Linux/macOS
# hoặc
.venv\Scripts\activate  # Windows
```

### 3. Cài Đặt Dependencies

#### Option A: Sử dụng pip

```bash
# Cài đặt từ requirements.txt
pip install -r requirements.txt
```

#### Option B: Sử dụng script tự động

```bash
# Chạy script cài đặt
bash install_dependencies.sh
# hoặc
chmod +x install_dependencies.sh
./install_dependencies.sh
```

#### Option C: Sử dụng Python script

```bash
python install_dependencies.py
```

### 4. Kiểm Tra Cài Đặt

```bash
python -c "import torch, gym, stable_baselines3, cvxopt, cvxpy; print('✓ All packages installed successfully')"
```

## Setup Data

### 1. Chuẩn Bị Data Files

Đảm bảo bạn có các file data trong thư mục `data/`:

- `DJIA_10_1d.csv` - Daily stock data cho 10 stocks
- `DJIA_1d_index.csv` - Daily index data
- `DJIA_10_60m.csv` - 60-minute data (optional, cho fine-grained features)
- `DJIA_60m_index.csv` - 60-minute index data (optional)

### 2. Setup Data Tự Động

Nếu bạn có sample data files, chạy script setup:

```bash
python setup_data.py
```

Script này sẽ tự động copy sample files sang tên file mong đợi nếu chưa tồn tại.

### 3. Kiểm Tra Data

```bash
# Kiểm tra data files
ls -lh data/
# Hoặc trên Windows
dir data
```

## Chạy Pipeline

### Cấu Trúc Cơ Bản

Pipeline được chạy thông qua file `entrance.py`:

```bash
python entrance.py
```

### Các Mode Chạy

Framework hỗ trợ 3 modes chính:

1. **RLonly**: Chỉ sử dụng RL agent (TD3-Profit, TD3-PR, TD3-SR)
2. **RLcontroller**: MASA framework với RL + Controller + Market Observer
3. **Benchmark**: Baseline algorithms (chưa implement đầy đủ)

### 1. Chạy MASA Framework với MAFIA Observer

Đây là mode mặc định và được khuyến nghị:

```bash
python entrance.py
```

**Cấu hình mặc định:**
- Algorithm: `MASA-mafia`
- Market: `DJIA`
- TopK: `10` stocks
- Epochs: `50`
- Market Observer: `mafia_1` (MAFIA)

### 2. Chạy MASA với Market Observer Khác

Để sử dụng market observer khác, chỉnh sửa `config.py`:

```python
# Trong config.py, dòng 31
self.benchmark_algo = 'MASA-dc'  # hoặc 'MASA-mlp', 'MASA-lstm'
```

Sau đó chạy:

```bash
python entrance.py
```

**Các options:**
- `MASA-dc`: Directional Change observer
- `MASA-mlp`: MLP-based observer
- `MASA-lstm`: LSTM-based observer
- `MASA-mafia`: MAFIA observer (mặc định)

### 3. Chạy TD3-only (Không có Controller)

Chỉnh sửa `config.py`:

```python
# Trong config.py, dòng 31
self.benchmark_algo = 'TD3-Profit'  # hoặc 'TD3-PR', 'TD3-SR'
```

Sau đó chạy:

```bash
python entrance.py
```

**Các options:**
- `TD3-Profit`: Maximize profit
- `TD3-PR`: Profit-Risk optimization
- `TD3-SR`: Sharpe Ratio optimization

### 4. Thay Đổi Cấu Hình Khác

Chỉnh sửa các tham số trong `config.py`:

```python
# Market và số lượng stocks
self.market_name = 'DJIA'  # 'DJIA', 'SP500', 'CSI300'
self.topK = 10  # 10, 20, 30

# Số epochs training
self.num_epochs = 50  # Số episodes để train

# Risk parameters
self.risk_default = 0.017
self.risk_up_bound = 0.012
self.risk_hold_bound = 0.014
self.risk_down_bound = 0.017

# Training parameters
self.learning_rate = 0.0001
self.batch_size = 50
self.num_epochs = 50
```

## Các Câu Lệnh Chạy Pipeline Chi Tiết

### 1. Training MASA-MAFIA (Mặc định)

```bash
# Chạy với cấu hình mặc định
python entrance.py
```

### 2. Training với Custom Seed

```bash
# Chỉnh sửa config.py để set seed cố định
# Hoặc chạy với environment variable
PYTHONHASHSEED=42 python entrance.py
```

### 3. Training với GPU

```bash
# Đảm bảo CUDA_VISIBLE_DEVICES được set trong entrance.py
# Hoặc set manually:
CUDA_VISIBLE_DEVICES=0 python entrance.py
```

### 4. Training với Logging Chi Tiết

```bash
# Enable verbose logging
python entrance.py 2>&1 | tee training.log
```

### 5. Chạy Test Suite

```bash
# Test weight-symbol mapping
python test_weight_symbol_mapping.py

# Test MAFIA components
python test_mafia_components.py

# Test end-to-end
python test_mafia_end_to_end.py

# Quick run test
python test_quick_run.py
```

## Cấu Hình Nâng Cao

### Thay Đổi Date Range

Trong `config.py`, chỉnh sửa `date_split_dict`:

```python
date_split_dict = {            
    1: {
        'train_date_start': '2013-09-01 00:00:00',
        'train_date_end': '2018-08-31 23:59:59',
        'valid_date_start': '2018-09-01 00:00:00',
        'valid_date_end': '2020-08-31 23:59:59',
        'test_date_start': '2020-09-01 00:00:00',
        'test_date_end': '2023-08-31 23:59:59'
    },
}
```

### Thay Đổi Market Observer Parameters

Cho MAFIA observer, trong `config.py`:

```python
# MAFIA Hyperparameters
self.mafia_T_w = 30  # Observation window size
self.mafia_DC_thresholds = [0.005, 0.01, 0.02]  # DC thresholds
self.mafia_D = 64  # Embedding dimension
self.mafia_D_h = 128  # Hidden layer dimension
self.mafia_encoder_layers = 2  # Transformer encoder layers
self.mafia_encoder_heads = 4  # Attention heads
```

### Thay Đổi RL Model Parameters

```python
# TD3 config
self.learning_rate = 0.0001
self.batch_size = 50
self.gradient_steps = 1
self.ars_trial = 10  # Iterative risk relaxation trials
```

## Kết Quả Output

### 1. Thư Mục Results

Kết quả được lưu trong `res/` với cấu trúc:

```
res/
└── RLcontroller/
    └── TD3/
        └── DJIA-10/
            └── YYYY-MM-DD-HH-MM-SS/
                ├── model/
                │   └── *.zip (saved models)
                ├── graph/
                │   └── *.png (visualizations)
                ├── train_profile.csv
                ├── train_bestmodel.csv
                ├── train_stepdata.csv
                ├── train_actions.csv  # Portfolio weights với symbols
                ├── valid_profile.csv
                ├── valid_bestmodel.csv
                ├── valid_stepdata.csv
                ├── valid_actions.csv
                ├── test_profile.csv
                ├── test_bestmodel.csv
                ├── test_stepdata.csv
                └── test_actions.csv
```

### 2. Các File Quan Trọng

- **`*_profile.csv`**: Performance metrics (Sharpe ratio, MDD, returns, etc.)
- **`*_bestmodel.csv`**: Best model information
- **`*_stepdata.csv`**: Step-by-step data (capital, returns, risk, etc.)
- **`*_actions.csv`**: Portfolio weights với stock symbols mapping

### 3. Xem Portfolio Weights với Symbols

Sử dụng utility function để map weights với symbols:

```python
from utils.weight_symbol_mapper import print_weight_mapping
import pandas as pd

# Đọc actions file
actions_df = pd.read_csv('res/.../train_actions.csv')

# Lấy weights cho một ngày
day_idx = 0
weights = actions_df.iloc[day_idx].drop('date').values
symbols = actions_df.columns.drop('date').values

# In mapping
print_weight_mapping(weights, symbols, title="Day 0 Portfolio")
```

Hoặc chạy example:

```bash
python example_weight_mapping_usage.py
```

## Monitoring Training

### 1. Xem Logs Trong Terminal

Training sẽ in ra:
- Current epoch capital
- Historical best capital
- Solver statistics (solvable/insolvable)
- Sample portfolio weights
- Time usage

### 2. TensorBoard (Nếu có)

```bash
# Khởi động TensorBoard
tensorboard --logdir=./res/

# Mở browser tại http://localhost:6006
```

## Troubleshooting

### 1. Lỗi: "Cannot load the data from ..."

**Nguyên nhân**: Thiếu data files

**Giải pháp**:
```bash
# Kiểm tra data files
ls data/

# Chạy setup script
python setup_data.py

# Đảm bảo file có tên đúng: DJIA_10_1d.csv
```

### 2. Lỗi: "CUDA out of memory"

**Nguyên nhân**: GPU memory không đủ

**Giải pháp**:
- Giảm `batch_size` trong `config.py`
- Sử dụng CPU: Set `CUDA_VISIBLE_DEVICES=""` hoặc comment out GPU code
- Giảm `topK` (số stocks)

### 3. Lỗi: "Length mismatch" trong weight mapping

**Nguyên nhân**: Weights và symbols không match

**Giải pháp**:
```python
# Validate mapping
from utils.weight_symbol_mapper import validate_weight_symbol_mapping
result = validate_weight_symbol_mapping(weights, symbols)
if not result['valid']:
    print(f"Error: {result['error']}")
```

### 4. Lỗi: Solver không giải được

**Nguyên nhân**: Risk constraints quá strict

**Giải pháp**:
- Tăng `ars_trial` trong `config.py` (iterative risk relaxation)
- Điều chỉnh `risk_up_bound`, `risk_hold_bound`, `risk_down_bound`
- Kiểm tra `is_enable_dynamic_risk_bound` setting

### 5. Lỗi Import

**Nguyên nhân**: Thiếu dependencies

**Giải pháp**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Hoặc
bash install_dependencies.sh
```

### 6. Performance Chậm

**Giải pháp**:
- Sử dụng GPU nếu có
- Giảm `num_epochs` cho testing
- Giảm `topK` (số stocks)
- Tối ưu `batch_size` và `gradient_steps`

## Ví Dụ Chạy Pipeline Hoàn Chỉnh

### Example 1: Training MASA-MAFIA từ đầu

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup data
python setup_data.py

# 4. Run training
python entrance.py

# 5. Check results
ls -lh res/RLcontroller/TD3/DJIA-10/*/train_profile.csv
```

### Example 2: Training với Custom Config

```bash
# 1. Edit config.py
#    - Set benchmark_algo = 'MASA-mafia'
#    - Set topK = 20
#    - Set num_epochs = 100

# 2. Run
python entrance.py

# 3. Results sẽ ở res/RLcontroller/TD3/DJIA-20/...
```

### Example 3: Quick Test Run

```bash
# 1. Edit config.py
#    - Set num_epochs = 5  # Quick test

# 2. Run
python entrance.py

# 3. Verify output
python test_weight_symbol_mapping.py
```

## Scripts Hữu Ích

### 1. Test Environment

```bash
python test_environment.py
```

### 2. Test MAFIA Components

```bash
python test_mafia_components.py
```

### 3. Test End-to-End

```bash
python test_mafia_end_to_end.py
```

### 4. Quick Run Test

```bash
python test_quick_run.py
```

## Tài Liệu Tham Khảo

- **MAFIA Implementation**: Xem `MAFIA_README.md`
- **Technical Specification**: Xem `Đặc Tả Kỹ Thuật (Technical Specification) - MAFIA.md`
- **Weight-Symbol Mapping**: Xem `example_weight_mapping_usage.py`
- **UV Guide**: Xem `UV_GUIDE.md` (nếu dùng UV)

## Support

Nếu gặp vấn đề:

1. Kiểm tra logs trong terminal
2. Xem các file test để hiểu cách sử dụng
3. Kiểm tra data files và cấu hình
4. Xem troubleshooting section ở trên

## License

Xem file `LICENSE` để biết thêm chi tiết.

---

**Lưu ý**: Đảm bảo bạn đã setup data files trước khi chạy pipeline. Framework sẽ tự động tạo thư mục results và lưu tất cả outputs.

