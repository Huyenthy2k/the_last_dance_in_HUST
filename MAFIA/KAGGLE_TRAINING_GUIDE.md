# Hướng Dẫn Train MAFIA trên Kaggle

## Tổng Quan

Kaggle cung cấp GPU miễn phí (30 giờ/tuần) và môi trường Python đã được cấu hình sẵn. Hướng dẫn này sẽ giúp bạn train MAFIA trên Kaggle Notebooks.

## Bước 1: Chuẩn Bị Dataset trên Kaggle

### 1.1. Upload Data Files lên Kaggle Dataset

1. **Tạo Kaggle Dataset mới:**
   - Vào [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload các file data từ thư mục `data/`:
     - `stock_prices_all_20251108_234851.csv` (hoặc file data tương ứng)
     - `DJIA_1d_index.csv` (nếu có)
   - Đặt tên dataset: `mafia-training-data`
   - Click "Create"

2. **Hoặc sử dụng Kaggle API:**
```bash
# Cài đặt Kaggle API (trên máy local)
pip install kaggle

# Upload dataset
kaggle datasets create -p ./data -r zip
```

### 1.2. Tạo Kaggle Notebook

1. Vào [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Chọn:
   - **Settings** → **Accelerator**: GPU T4 x2 (hoặc P100)
   - **Internet**: ON (để pip install)
   - **File persistence**: ON (để lưu models)

## Bước 2: Setup Notebook

### 2.1. Add Dataset vào Notebook

1. Trong notebook, click **"+ Add data"**
2. Tìm và add dataset `mafia-training-data` (hoặc dataset bạn đã tạo)
3. Dataset sẽ được mount tại `/kaggle/input/mafia-training-data/`

### 2.2. Code Setup Cell

Tạo cell đầu tiên để setup môi trường:

```python
# Cell 1: Setup Environment
import os
import sys
import shutil

# Tạo cấu trúc thư mục giống local
os.makedirs('/kaggle/working/MAFIA', exist_ok=True)
os.makedirs('/kaggle/working/MAFIA/data', exist_ok=True)
os.makedirs('/kaggle/working/MAFIA/res', exist_ok=True)

# Copy data files từ input dataset
data_source = '/kaggle/input/mafia-training-data'
data_dest = '/kaggle/working/MAFIA/data'

if os.path.exists(data_source):
    for file in os.listdir(data_source):
        if file.endswith('.csv'):
            shutil.copy(
                os.path.join(data_source, file),
                os.path.join(data_dest, file)
            )
            print(f"Copied {file}")

# Thay đổi working directory
os.chdir('/kaggle/working/MAFIA')
print(f"Working directory: {os.getcwd()}")
print(f"Data files: {os.listdir(data_dest)}")
```

### 2.3. Upload Code Files

Có 2 cách:

#### Cách 1: Upload từng file (Khuyến nghị cho lần đầu)

1. Click **"+ Add file"** trong notebook
2. Upload các file cần thiết:
   - `config.py`
   - `entrance.py`
   - `RL_controller/` (toàn bộ thư mục)
   - `utils/` (toàn bộ thư mục)

#### Cách 2: Clone từ GitHub (Nếu có repo)

```python
# Cell 2: Clone Repository (nếu code trên GitHub)
!git clone https://github.com/your-username/mafia-repo.git
!cp -r mafia-repo/* /kaggle/working/MAFIA/
```

#### Cách 3: Upload ZIP và extract

```python
# Cell 2: Extract từ ZIP file
import zipfile

# Upload MAFIA.zip qua "Add file"
with zipfile.ZipFile('/kaggle/input/mafia-code/MAFIA.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/MAFIA/')
```

## Bước 3: Cài Đặt Dependencies

### 3.1. Install Packages

```python
# Cell 3: Install Dependencies
!pip install -q cvxopt>=1.3.2
!pip install -q "cvxpy>=1.3.1,<2.0.0"
!pip install -q empyrical>=0.5.5
!pip install -q "gym>=0.21.0,<0.27.0"
!pip install -q h5py>=3.9.0
!pip install -q matplotlib>=3.8.0
!pip install -q numba>=0.59.0
!pip install -q "numpy>=1.24.0,<2.0.0"
!pip install -q "pandas>=2.0.0,<3.0.0"
!pip install -q pandas-datareader>=0.10.0
!pip install -q scikit-learn>=1.3.0
!pip install -q "scipy>=1.10.0,<2.0.0"
!pip install -q stable-baselines>=2.10.2
!pip install -q stable-baselines3>=2.0.0
!pip install -q tensorboard>=2.15.0
!pip install -q "torch>=2.0.0"

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 3.2. Fix Import Paths

```python
# Cell 4: Setup Python Path
import sys
sys.path.insert(0, '/kaggle/working/MAFIA')
sys.path.insert(0, '/kaggle/working/MAFIA/RL_controller')
sys.path.insert(0, '/kaggle/working/MAFIA/utils')

# Verify imports
try:
    from config import Config
    from entrance import entrance
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
```

## Bước 4: Cấu Hình cho Kaggle

### 4.1. Modify config.py cho Kaggle

Tạo cell để override config:

```python
# Cell 5: Configure for Kaggle
import os
import sys

# Modify config paths
# Tạo file config_kaggle.py
config_kaggle_code = '''
import sys
sys.path.insert(0, "/kaggle/working/MAFIA")
from config import Config as BaseConfig

class KaggleConfig(BaseConfig):
    def __init__(self, seed_num=2022, current_date=None):
        super().__init__(seed_num=seed_num, current_date=current_date)
        
        # Override paths for Kaggle
        self.dataDir = '/kaggle/working/MAFIA/data'
        self.res_dir = os.path.join('/kaggle/working/MAFIA/res', 
                                    self.mode, self.rl_model_name, 
                                    '{}-{}'.format(self.market_name, self.topK), 
                                    self.cur_datetime)
        os.makedirs(self.res_dir, exist_ok=True)
        self.res_model_dir = os.path.join(self.res_dir, 'model')
        os.makedirs(self.res_model_dir, exist_ok=True)
        self.res_img_dir = os.path.join(self.res_dir, 'graph')
        os.makedirs(self.res_img_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.res_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Kaggle-specific settings
        # Giảm epochs nếu muốn test nhanh
        # self.num_epochs = 10  # Uncomment để test nhanh
        
        # Đảm bảo GPU được sử dụng
        import torch as th
        if th.cuda.is_available():
            print(f"Using GPU: {th.cuda.get_device_name(0)}")
        else:
            print("Warning: CUDA not available, using CPU")
'''

with open('/kaggle/working/MAFIA/config_kaggle.py', 'w') as f:
    f.write(config_kaggle_code)

# Hoặc modify entrance.py để sử dụng KaggleConfig
print("✓ Config file created")
```

### 4.2. Modify entrance.py (nếu cần)

```python
# Cell 6: Patch entrance.py for Kaggle
# Đọc entrance.py
with open('/kaggle/working/MAFIA/entrance.py', 'r') as f:
    entrance_code = f.read()

# Thay đổi import config nếu cần
# entrance_code = entrance_code.replace(
#     'from config import Config',
#     'from config_kaggle import KaggleConfig as Config'
# )

# Ghi lại
# with open('/kaggle/working/MAFIA/entrance.py', 'w') as f:
#     f.write(entrance_code)

print("✓ Entrance file ready")
```

## Bước 5: Chạy Training

### 5.1. Training Cell

```python
# Cell 7: Run Training
import os
import sys
import datetime
import random
import time
import numpy as np
import torch as th

# Set working directory
os.chdir('/kaggle/working/MAFIA')
sys.path.insert(0, '/kaggle/working/MAFIA')

# Set random seeds
current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))

random.seed(rand_seed)
os.environ['PYTHONHASHSEED'] = str(rand_seed)
np.random.seed(rand_seed)
th.manual_seed(rand_seed)
if th.cuda.is_available():
    th.cuda.manual_seed(rand_seed)
    th.cuda.manual_seed_all(rand_seed)

# Import và chạy
from config import Config
from entrance import entrance

print("=" * 50)
print("Starting MAFIA Training on Kaggle")
print("=" * 50)
print(f"Working directory: {os.getcwd()}")
print(f"Data directory: {os.listdir('/kaggle/working/MAFIA/data')}")
print("=" * 50)

# Chạy training
entrance()

print("=" * 50)
print("Training Completed!")
print("=" * 50)
```

### 5.2. Monitor Training

Kaggle sẽ tự động hiển thị output trong notebook. Bạn có thể:

1. **Xem logs real-time** trong cell output
2. **Check GPU usage**: Click "Settings" → "Accelerator" → xem GPU usage
3. **Monitor memory**: Kaggle hiển thị RAM/GPU memory usage

## Bước 6: Lưu Kết Quả

### 6.1. Save Models và Results

Kaggle tự động lưu mọi thứ trong `/kaggle/working/`. Để đảm bảo files được lưu:

```python
# Cell 8: Save Important Files
import shutil
import os

# Tạo output directory
output_dir = '/kaggle/working/output'
os.makedirs(output_dir, exist_ok=True)

# Copy results
res_dir = '/kaggle/working/MAFIA/res'
if os.path.exists(res_dir):
    # Copy toàn bộ results
    shutil.copytree(res_dir, os.path.join(output_dir, 'results'), dirs_exist_ok=True)
    print("✓ Results copied to output/")

# List important files
print("\nImportant files in /kaggle/working/MAFIA/res/:")
for root, dirs, files in os.walk('/kaggle/working/MAFIA/res'):
    for file in files:
        if file.endswith(('.csv', '.zip', '.pth', '.png')):
            print(f"  {os.path.join(root, file)}")
```

### 6.2. Download Results

1. **Tự động download**: Files trong `/kaggle/working/` sẽ được lưu khi notebook kết thúc
2. **Manual download**: 
   - Click "Save Version" → "Save & Run All"
   - Sau khi chạy xong, click "Output" tab
   - Download files từ output

### 6.3. Tạo Output Dataset

```python
# Cell 9: Create Output Dataset (Optional)
# Tạo dataset mới chứa trained models
import zipfile
import os

output_zip = '/kaggle/working/trained_models.zip'
with zipfile.ZipFile(output_zip, 'w') as zipf:
    for root, dirs, files in os.walk('/kaggle/working/MAFIA/res'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, '/kaggle/working/MAFIA')
            zipf.write(file_path, arcname)

print(f"✓ Created {output_zip}")
print("You can download this file from the Output tab")
```

## Tips và Best Practices

### 1. Tối Ưu GPU Usage

```python
# Sử dụng mixed precision training (nếu model hỗ trợ)
import torch
torch.backends.cudnn.benchmark = True
```

### 2. Giảm Memory Usage

```python
# Trong config.py, giảm batch size nếu cần
# self.batch_size = 32  # Thay vì 50
```

### 3. Resume từ Checkpoint

```python
# Trong config.py, set checkpoint path
# config.resume_from_checkpoint = '/kaggle/working/MAFIA/res/.../checkpoint_info.json'
```

### 4. Test Nhanh Trước

```python
# Giảm epochs để test
# config.num_epochs = 5  # Test với 5 epochs trước
```

### 5. Sử dụng Kaggle Secrets (cho API keys nếu cần)

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
# secret_value = user_secrets.get_secret("your_secret_name")
```

## Troubleshooting

### Lỗi: "Cannot find data file"

**Giải pháp:**
```python
# Kiểm tra data files
import os
data_dir = '/kaggle/working/MAFIA/data'
print(f"Data files: {os.listdir(data_dir)}")

# Đảm bảo file names đúng trong config.py
# self.stock_data_file = 'stock_prices_all_20251108_234851.csv'
```

### Lỗi: "CUDA out of memory"

**Giải pháp:**
- Giảm `batch_size` trong config
- Giảm `topK` (số stocks)
- Sử dụng CPU: Set `CUDA_VISIBLE_DEVICES=""`

### Lỗi: "Module not found"

**Giải pháp:**
```python
# Thêm paths vào sys.path
import sys
sys.path.insert(0, '/kaggle/working/MAFIA')
sys.path.insert(0, '/kaggle/working/MAFIA/RL_controller')
sys.path.insert(0, '/kaggle/working/MAFIA/utils')
```

### Notebook Timeout

Kaggle có giới hạn thời gian chạy. Để tránh timeout:
- Lưu checkpoint thường xuyên
- Resume từ checkpoint nếu bị timeout
- Chia training thành nhiều sessions

## Template Notebook Hoàn Chỉnh

Tạo notebook mới với các cells sau:

1. **Setup Environment** (Cell 1)
2. **Copy Data Files** (Cell 2)
3. **Upload/Extract Code** (Cell 3)
4. **Install Dependencies** (Cell 4)
5. **Setup Paths** (Cell 5)
6. **Configure for Kaggle** (Cell 6)
7. **Run Training** (Cell 7)
8. **Save Results** (Cell 8)

## Kết Luận

Kaggle là nền tảng tuyệt vời để train MAFIA với:
- ✅ GPU miễn phí (30h/tuần)
- ✅ Môi trường đã cấu hình sẵn
- ✅ Dễ dàng chia sẻ và reproduce
- ✅ Tự động lưu outputs

Chúc bạn training thành công! 🚀

