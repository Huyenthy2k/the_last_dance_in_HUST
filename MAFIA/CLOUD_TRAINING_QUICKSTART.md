# Quick Start: Train MAFIA trên Kaggle/Colab

## So Sánh Nhanh

| Tính năng | Kaggle | Google Colab |
|-----------|--------|--------------|
| **GPU** | T4 x2 hoặc P100 | T4 |
| **Thời gian GPU** | 30 giờ/tuần | Không giới hạn (có thể disconnect) |
| **Lưu trữ** | `/kaggle/working/` (tự động lưu) | `/content/` (cần mount Drive) |
| **Internet** | Cần bật để pip install | Luôn bật |
| **Timeout** | Có giới hạn | 90 phút không hoạt động |
| **Dễ sử dụng** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Tốt nhất cho** | Training dài, cần GPU ổn định | Experiment nhanh, testing |

## Quick Start: Kaggle

### 1. Tạo Notebook
- Vào [Kaggle Notebooks](https://www.kaggle.com/code)
- New Notebook → Settings → GPU T4 x2

### 2. Upload Data
- Tạo Dataset: Upload CSV files vào Kaggle Datasets
- Add dataset vào notebook

### 3. Code Setup (Copy-paste vào notebook)

```python
# Cell 1: Setup
import os
os.makedirs('/kaggle/working/MAFIA/data', exist_ok=True)
import shutil
for f in os.listdir('/kaggle/input/mafia-training-data'):
    if f.endswith('.csv'):
        shutil.copy(f'/kaggle/input/mafia-training-data/{f}', 
                   f'/kaggle/working/MAFIA/data/{f}')

# Cell 2: Upload code files (qua "Add file" button)
# Hoặc clone từ GitHub:
# !git clone https://github.com/your-repo.git /kaggle/working/MAFIA

# Cell 3: Install
!pip install -q torch stable-baselines3 cvxopt cvxpy pandas numpy scipy gym

# Cell 4: Run
import sys
sys.path.insert(0, '/kaggle/working/MAFIA')
os.chdir('/kaggle/working/MAFIA')
from entrance import entrance
entrance()
```

### 4. Download Results
- Files trong `/kaggle/working/` tự động được lưu
- Click "Output" tab để download

## Quick Start: Colab

### 1. Tạo Notebook
- Vào [Google Colab](https://colab.research.google.com/)
- New notebook → Runtime → Change runtime type → GPU

### 2. Setup Code (Copy-paste)

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Setup
import os
os.makedirs('/content/MAFIA', exist_ok=True)
# Copy code từ Drive hoặc clone từ GitHub
# !git clone https://github.com/your-repo.git /content/MAFIA

# Cell 3: Install
!pip install -q torch stable-baselines3 cvxopt cvxpy pandas numpy scipy gym

# Cell 4: Run
import sys
sys.path.insert(0, '/content/MAFIA')
os.chdir('/content/MAFIA')
from entrance import entrance
entrance()

# Cell 5: Save to Drive
import shutil
shutil.copytree('/content/MAFIA/res', 
                '/content/drive/MyDrive/MAFIA_Results', 
                dirs_exist_ok=True)
```

## Checklist Trước Khi Train

### ✅ Chuẩn Bị
- [ ] Code files đã sẵn sàng (config.py, entrance.py, RL_controller/, utils/)
- [ ] Data files đã sẵn sàng (CSV files)
- [ ] Đã chọn platform (Kaggle hoặc Colab)

### ✅ Setup
- [ ] GPU đã được enable
- [ ] Dependencies đã install
- [ ] Data files đã được copy vào đúng thư mục
- [ ] Python paths đã được setup
- [ ] Config paths đã được chỉnh cho platform

### ✅ Trước Khi Chạy
- [ ] Đã kiểm tra data files tồn tại
- [ ] Đã set num_epochs phù hợp (test với số nhỏ trước)
- [ ] Đã setup checkpoint saving
- [ ] Đã biết cách lưu/download results

## Common Issues & Quick Fixes

### ❌ "Cannot find data file"
```python
# Check files
import os
print(os.listdir('/kaggle/working/MAFIA/data'))  # Kaggle
print(os.listdir('/content/MAFIA/data'))  # Colab
```

### ❌ "Module not found"
```python
import sys
sys.path.insert(0, '/kaggle/working/MAFIA')  # Kaggle
sys.path.insert(0, '/content/MAFIA')  # Colab
```

### ❌ "CUDA out of memory"
```python
# Giảm batch size trong config.py
# self.batch_size = 32  # Thay vì 50
```

### ❌ "Colab disconnected"
- Lưu checkpoint vào Drive thường xuyên
- Sử dụng Colab Pro
- Resume từ checkpoint

## Tips

### 🚀 Tăng Tốc
- Sử dụng GPU (luôn enable)
- Giảm `topK` nếu test nhanh
- Giảm `num_epochs` cho lần đầu test
- Sử dụng checkpoint để resume

### 💾 Lưu Trữ
- **Kaggle**: Tự động lưu trong `/kaggle/working/`
- **Colab**: Lưu vào Drive hoặc download ZIP

### 🔄 Resume Training
```python
# Trong config.py hoặc trước khi chạy
config.resume_from_checkpoint = 'path/to/checkpoint_info.json'
config.auto_resume_from_latest = True
```

## Next Steps

1. **Đọc hướng dẫn chi tiết:**
   - `KAGGLE_TRAINING_GUIDE.md` - Hướng dẫn đầy đủ cho Kaggle
   - `COLAB_TRAINING_GUIDE.md` - Hướng dẫn đầy đủ cho Colab

2. **Test nhanh:**
   - Set `num_epochs = 5` trong config.py
   - Chạy training để verify setup

3. **Training đầy đủ:**
   - Set `num_epochs = 50` (hoặc số bạn muốn)
   - Đảm bảo có đủ thời gian GPU
   - Monitor training progress

4. **Lưu results:**
   - Download models và CSV files
   - Lưu vào Drive hoặc local machine

## Resources

- **Kaggle**: [kaggle.com](https://www.kaggle.com)
- **Colab**: [colab.research.google.com](https://colab.research.google.com)
- **MAFIA Docs**: Xem `README.md` và `MAFIA_README.md`

---

**Chúc bạn training thành công!** 🎉

