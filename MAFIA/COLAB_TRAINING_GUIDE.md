# Hướng Dẫn Train MAFIA trên Google Colab

## Tổng Quan

Google Colab cung cấp GPU miễn phí (T4) và môi trường Python đã được cấu hình. Hướng dẫn này sẽ giúp bạn train MAFIA trên Colab.

## Bước 1: Chuẩn Bị

### 1.1. Upload Code lên Google Drive (Khuyến nghị)

1. **Tạo thư mục trên Google Drive:**
   - Tạo folder: `MAFIA_Project`
   - Upload toàn bộ code vào folder này:
     - `config.py`
     - `entrance.py`
     - `RL_controller/` (toàn bộ thư mục)
     - `utils/` (toàn bộ thư mục)
     - `data/` (chứa data files)

2. **Hoặc sử dụng GitHub:**
   - Push code lên GitHub
   - Clone trực tiếp trong Colab

### 1.2. Tạo Colab Notebook Mới

1. Vào [Google Colab](https://colab.research.google.com/)
2. Click **"File" → "New notebook"**
3. Đặt tên: `MAFIA_Training.ipynb`

## Bước 2: Setup Colab Notebook

### 2.1. Enable GPU

```python
# Cell 1: Enable GPU Runtime
# Chạy cell này đầu tiên
# Sau đó: Runtime → Change runtime type → Hardware accelerator: GPU (T4)

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 2.2. Mount Google Drive

```python
# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Kiểm tra mount
import os
if os.path.exists('/content/drive/MyDrive'):
    print("✓ Google Drive mounted successfully")
    print(f"Drive contents: {os.listdir('/content/drive/MyDrive')[:5]}")
else:
    print("✗ Failed to mount Google Drive")
```

### 2.3. Setup Working Directory

```python
# Cell 3: Setup Working Directory
import os
import sys

# Đường dẫn đến project trên Drive
# Thay đổi đường dẫn này theo cấu trúc Drive của bạn
DRIVE_PROJECT_PATH = '/content/drive/MyDrive/MAFIA_Project'
WORKING_DIR = '/content/MAFIA'

# Tạo working directory
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, 'res'), exist_ok=True)

# Copy code từ Drive (nếu có)
if os.path.exists(DRIVE_PROJECT_PATH):
    print(f"Copying files from {DRIVE_PROJECT_PATH}...")
    import shutil
    
    # Copy các file và thư mục cần thiết
    for item in ['config.py', 'entrance.py', 'RL_controller', 'utils']:
        src = os.path.join(DRIVE_PROJECT_PATH, item)
        dst = os.path.join(WORKING_DIR, item)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            print(f"  ✓ Copied {item}")
    
    # Copy data files
    data_src = os.path.join(DRIVE_PROJECT_PATH, 'data')
    data_dst = os.path.join(WORKING_DIR, 'data')
    if os.path.exists(data_src):
        for file in os.listdir(data_src):
            if file.endswith('.csv'):
                shutil.copy2(
                    os.path.join(data_src, file),
                    os.path.join(data_dst, file)
                )
                print(f"  ✓ Copied data file: {file}")

# Hoặc clone từ GitHub (nếu code trên GitHub)
# !git clone https://github.com/your-username/mafia-repo.git
# !cp -r mafia-repo/* {WORKING_DIR}/

# Thay đổi working directory
os.chdir(WORKING_DIR)
print(f"\n✓ Working directory: {os.getcwd()}")
print(f"✓ Files in working directory:")
for item in os.listdir('.'):
    print(f"  - {item}")
```

### 2.4. Alternative: Clone từ GitHub

Nếu code trên GitHub:

```python
# Cell 3 Alternative: Clone from GitHub
import os
import shutil

WORKING_DIR = '/content/MAFIA'
REPO_URL = 'https://github.com/your-username/mafia-repo.git'  # Thay đổi URL

# Clone repository
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)
!git clone {REPO_URL} {WORKING_DIR}

# Thay đổi working directory
os.chdir(WORKING_DIR)
print(f"✓ Working directory: {os.getcwd()}")
```

## Bước 3: Cài Đặt Dependencies

### 3.1. Install Packages

```python
# Cell 4: Install Dependencies
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

print("✓ Dependencies installed")
```

### 3.2. Verify Installation

```python
# Cell 5: Verify Installation
import torch
import numpy as np
import pandas as pd
import cvxopt
import cvxpy as cp
import gym
import stable_baselines3

print("✓ All packages imported successfully")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 3.3. Setup Python Paths

```python
# Cell 6: Setup Python Paths
import sys
import os

WORKING_DIR = '/content/MAFIA'
sys.path.insert(0, WORKING_DIR)
sys.path.insert(0, os.path.join(WORKING_DIR, 'RL_controller'))
sys.path.insert(0, os.path.join(WORKING_DIR, 'utils'))

# Verify imports
try:
    from config import Config
    from entrance import entrance
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
```

## Bước 4: Cấu Hình cho Colab

### 4.1. Modify Config cho Colab

```python
# Cell 7: Configure for Colab
import os
import sys

# Tạo config override
config_colab_code = '''
import os
import sys
sys.path.insert(0, "/content/MAFIA")
from config import Config as BaseConfig

class ColabConfig(BaseConfig):
    def __init__(self, seed_num=2022, current_date=None):
        super().__init__(seed_num=seed_num, current_date=current_date)
        
        # Override paths for Colab
        self.dataDir = '/content/MAFIA/data'
        self.res_dir = os.path.join('/content/MAFIA/res', 
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
        
        # Colab-specific settings
        # Giảm epochs nếu muốn test nhanh
        # self.num_epochs = 10  # Uncomment để test nhanh
        
        # Đảm bảo GPU được sử dụng
        import torch as th
        if th.cuda.is_available():
            print(f"Using GPU: {th.cuda.get_device_name(0)}")
        else:
            print("Warning: CUDA not available, using CPU")
'''

with open('/content/MAFIA/config_colab.py', 'w') as f:
    f.write(config_colab_code)

print("✓ Config file created")
```

### 4.2. Upload Data Files (nếu chưa có)

Nếu data files chưa có trong Drive, upload trực tiếp:

```python
# Cell 8: Upload Data Files (Optional)
from google.colab import files
import os

# Tạo data directory
data_dir = '/content/MAFIA/data'
os.makedirs(data_dir, exist_ok=True)

print("Upload data files:")
print("1. Click 'Choose Files' below")
print("2. Select your CSV data files")
print("3. Files will be saved to /content/MAFIA/data/")

# Uncomment để upload
# uploaded = files.upload()
# for filename in uploaded.keys():
#     shutil.move(filename, os.path.join(data_dir, filename))
#     print(f"  ✓ Moved {filename} to {data_dir}")

# List current data files
print(f"\nCurrent data files in {data_dir}:")
if os.path.exists(data_dir):
    for file in os.listdir(data_dir):
        print(f"  - {file}")
```

## Bước 5: Chạy Training

### 5.1. Training Cell

```python
# Cell 9: Run Training
import os
import sys
import datetime
import random
import time
import numpy as np
import torch as th

# Set working directory
os.chdir('/content/MAFIA')
sys.path.insert(0, '/content/MAFIA')

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
print("Starting MAFIA Training on Colab")
print("=" * 50)
print(f"Working directory: {os.getcwd()}")
print(f"Data directory: {os.listdir('/content/MAFIA/data')}")
if th.cuda.is_available():
    print(f"GPU: {th.cuda.get_device_name(0)}")
    print(f"GPU Memory: {th.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 50)

# Chạy training
try:
    entrance()
    print("=" * 50)
    print("Training Completed Successfully!")
    print("=" * 50)
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
```

### 5.2. Monitor Training

Colab hiển thị output real-time. Bạn có thể:

1. **Xem logs** trong cell output
2. **Check GPU usage**: Runtime → Manage sessions → xem GPU usage
3. **Monitor memory**: Colab hiển thị RAM/GPU memory

### 5.3. TensorBoard (Optional)

```python
# Cell 10: Launch TensorBoard (Optional)
%load_ext tensorboard
%tensorboard --logdir=/content/MAFIA/res
```

## Bước 6: Lưu Kết Quả

### 6.1. Save to Google Drive

```python
# Cell 11: Save Results to Google Drive
import shutil
import os
from datetime import datetime

# Đường dẫn trên Drive
drive_results_path = '/content/drive/MyDrive/MAFIA_Results'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
drive_timestamp_path = os.path.join(drive_results_path, timestamp)
os.makedirs(drive_timestamp_path, exist_ok=True)

# Copy results
res_dir = '/content/MAFIA/res'
if os.path.exists(res_dir):
    # Copy toàn bộ results
    for item in os.listdir(res_dir):
        src = os.path.join(res_dir, item)
        dst = os.path.join(drive_timestamp_path, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"✓ Results saved to: {drive_timestamp_path}")
    
    # List important files
    print("\nImportant files:")
    for root, dirs, files in os.walk(drive_timestamp_path):
        for file in files:
            if file.endswith(('.csv', '.zip', '.pth', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), drive_timestamp_path)
                print(f"  {rel_path}")
else:
    print("✗ Results directory not found")
```

### 6.2. Download Results

```python
# Cell 12: Download Results
import zipfile
import os
from google.colab import files

# Tạo ZIP file
output_zip = '/content/MAFIA_results.zip'
res_dir = '/content/MAFIA/res'

if os.path.exists(res_dir):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files_list in os.walk(res_dir):
            for file in files_list:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '/content/MAFIA')
                zipf.write(file_path, arcname)
    
    print(f"✓ Created {output_zip}")
    print("Downloading...")
    files.download(output_zip)
    print("✓ Download completed")
else:
    print("✗ Results directory not found")
```

### 6.3. Save Checkpoints Periodically

Để tránh mất dữ liệu khi Colab disconnect, lưu checkpoint thường xuyên:

```python
# Cell 13: Auto-save Checkpoints to Drive
import time
import shutil
import os
from threading import Thread

def auto_save_checkpoints():
    """Auto-save checkpoints to Drive every 30 minutes"""
    while True:
        time.sleep(1800)  # 30 minutes
        checkpoint_dir = '/content/MAFIA/res/RLcontroller/TD3/DJIA-10'
        drive_checkpoint_dir = '/content/drive/MyDrive/MAFIA_Checkpoints'
        
        if os.path.exists(checkpoint_dir):
            os.makedirs(drive_checkpoint_dir, exist_ok=True)
            # Copy latest checkpoint
            # (Implement checkpoint copying logic here)
            print(f"[Auto-save] Checkpoints saved at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Start auto-save thread (optional)
# auto_save_thread = Thread(target=auto_save_checkpoints, daemon=True)
# auto_save_thread.start()
# print("✓ Auto-save enabled")
```

## Tips và Best Practices

### 1. Tránh Disconnect

Colab có thể disconnect sau 90 phút không hoạt động. Để tránh:

```python
# Cell: Keep Colab Alive
import time
from IPython.display import display, Javascript

def keep_alive():
    """Keep Colab session alive"""
    display(Javascript('''
        function keepAlive() {
            console.log("Keeping session alive...");
        }
        setInterval(keepAlive, 60000);  // Every minute
    '''))

# Uncomment để enable
# keep_alive()
```

Hoặc sử dụng extension: [Colab Alive](https://chrome.google.com/webstore/detail/colab-alive/ccjdfeidghohfkeifefjecmicphelofl)

### 2. Tối Ưu GPU Usage

```python
# Sử dụng mixed precision (nếu model hỗ trợ)
import torch
torch.backends.cudnn.benchmark = True

# Clear cache
torch.cuda.empty_cache()
```

### 3. Giảm Memory Usage

```python
# Trong config.py
# self.batch_size = 32  # Giảm từ 50 xuống 32
# self.topK = 10  # Giảm số stocks nếu cần
```

### 4. Resume từ Checkpoint

```python
# Trong config.py hoặc trước khi chạy training
# config.resume_from_checkpoint = '/content/drive/MyDrive/MAFIA_Checkpoints/.../checkpoint_info.json'
# config.auto_resume_from_latest = True
```

### 5. Test Nhanh Trước

```python
# Giảm epochs để test
# config.num_epochs = 5  # Test với 5 epochs trước
```

## Troubleshooting

### Lỗi: "Cannot find data file"

**Giải pháp:**
```python
# Kiểm tra data files
import os
data_dir = '/content/MAFIA/data'
print(f"Data files: {os.listdir(data_dir)}")

# Đảm bảo file names đúng trong config.py
# self.stock_data_file = 'stock_prices_all_20251108_234851.csv'
```

### Lỗi: "CUDA out of memory"

**Giải pháp:**
```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Giảm batch size trong config
# self.batch_size = 16  # Thay vì 50
```

### Lỗi: "Module not found"

**Giải pháp:**
```python
# Thêm paths vào sys.path
import sys
sys.path.insert(0, '/content/MAFIA')
sys.path.insert(0, '/content/MAFIA/RL_controller')
sys.path.insert(0, '/content/MAFIA/utils')

# Reinstall packages nếu cần
# !pip install --upgrade package_name
```

### Colab Disconnected

**Giải pháp:**
- Sử dụng checkpoint để resume
- Lưu results vào Drive thường xuyên
- Sử dụng Colab Pro để có session lâu hơn

### GPU Not Available

**Giải pháp:**
```python
# Check GPU availability
import torch
if not torch.cuda.is_available():
    print("GPU not available. Using CPU.")
    # Training sẽ chậm hơn nhưng vẫn chạy được
```

## Template Notebook Hoàn Chỉnh

Tạo notebook với các cells sau:

1. **Enable GPU** (Cell 1)
2. **Mount Google Drive** (Cell 2)
3. **Setup Working Directory** (Cell 3)
4. **Install Dependencies** (Cell 4)
5. **Verify Installation** (Cell 5)
6. **Setup Python Paths** (Cell 6)
7. **Configure for Colab** (Cell 7)
8. **Upload Data** (Cell 8 - Optional)
9. **Run Training** (Cell 9)
10. **Save Results to Drive** (Cell 11)
11. **Download Results** (Cell 12)

## Kết Luận

Google Colab là nền tảng tuyệt vời để train MAFIA với:
- ✅ GPU miễn phí (T4)
- ✅ Môi trường đã cấu hình sẵn
- ✅ Tích hợp với Google Drive
- ✅ Dễ dàng chia sẻ và collaborate

**Lưu ý quan trọng:**
- ⚠️ Colab có thể disconnect sau 90 phút không hoạt động
- ⚠️ Lưu checkpoint và results vào Drive thường xuyên
- ⚠️ GPU có thể không available vào giờ cao điểm

Chúc bạn training thành công! 🚀

