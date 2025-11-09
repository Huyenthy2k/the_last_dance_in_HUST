# Hướng dẫn sử dụng UV với môi trường ảo
# Bước 1: Tạo và kích hoạt virtual environment
chmod +x activate_venv.sh
source activate_venv.sh

# Bước 2: Cài đặt dependencies (sau khi đã activate venv)
chmod +x install_dependencies.sh
./install_dependencies.sh 

Script activate_venv.sh sẽ:
Tự tạo .venv nếu chưa có
Kích hoạt virtual environment
Kiểm tra dependencies

## Các bước cơ bản

### 1. Tạo môi trường ảo với uv
```bash
# Tạo môi trường ảo với Python 3.11 (tương thích với các packages cũ)
uv venv --python 3.11

# Hoặc chỉ định Python cụ thể
uv venv --python python3.11
```

### 2. Kích hoạt môi trường ảo
```bash
# Trên macOS/Linux
source .venv/bin/activate

# Sau khi kích hoạt, terminal sẽ hiển thị (.venv)
```

### 3. Cài đặt packages từ requirements.txt
```bash
# Sử dụng uv pip để cài đặt (nhanh hơn pip thông thường)
uv pip install -r requirements.txt

# Hoặc chỉ định Python từ môi trường ảo
uv pip install -r requirements.txt --python .venv/bin/python
```

### 4. Chạy script Python
```bash
# Sau khi kích hoạt môi trường ảo, chạy như bình thường
python entrance.py

# Hoặc sử dụng Python từ venv trực tiếp (không cần activate)
.venv/bin/python entrance.py
```

### 5. Deactivate môi trường ảo
```bash
deactivate
```

## Lưu ý quan trọng

1. **Python version**: Dự án này yêu cầu Python 3.10 hoặc 3.11 (không dùng 3.12 vì một số packages không tương thích)

2. **Nếu gặp lỗi khi cài đặt**: Một số packages cũ có thể cần cài đặt thủ công hoặc nâng cấp phiên bản

3. **Ưu điểm của uv**:
   - Cài đặt packages nhanh hơn pip thông thường
   - Quản lý dependencies tốt hơn
   - Tự động xử lý conflicts

## Các lệnh hữu ích khác

```bash
# Xem packages đã cài đặt
uv pip list

# Xóa môi trường ảo
rm -rf .venv

# Tạo môi trường ảo mới
uv venv --python 3.11

# Cài đặt một package cụ thể
uv pip install package_name

# Cài đặt package với phiên bản cụ thể
uv pip install package_name==1.2.3
```

