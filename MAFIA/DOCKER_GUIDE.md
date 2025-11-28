# Hướng Dẫn Chạy MAFIA Training Với Docker

## 📋 Yêu Cầu

- Docker Desktop (Windows/Mac) hoặc Docker Engine (Linux)
- 8GB RAM trở lên (khuyến nghị 16GB)
- 20GB dung lượng trống
- (Tùy chọn) NVIDIA GPU + nvidia-docker cho training nhanh hơn

## 🚀 Cài Đặt Docker

### Windows
1. Download Docker Desktop: https://www.docker.com/products/docker-desktop
2. Cài đặt và khởi động Docker Desktop
3. Kiểm tra: `docker --version`

### Linux
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

## 📦 Cấu Trúc Files

```
MAFIA/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore          # Files to exclude from image
├── .env.example           # Environment variables template
├── run_docker.ps1         # PowerShell script (Windows)
├── run_docker.sh          # Bash script (Linux/Mac)
└── DOCKER_GUIDE.md        # This file
```

## ⚙️ Cấu Hình

### 1. Tạo file `.env`

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

### 2. Sửa file `.env`

```env
# Hugging Face Configuration
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HF_REPO_ID=yourusername/mafia-model

# GPU Configuration (0 for first GPU, empty for CPU)
CUDA_VISIBLE_DEVICES=0

# Training Configuration
HPARAM_TRIALS=10
HPARAM_EPOCHS=10
TRAIN_EPOCHS=100
NUM_SEEDS=5
```

## 🏗️ Build Docker Image

### Cách 1: Sử dụng Docker Compose

```bash
docker-compose build
```

### Cách 2: Sử dụng Docker CLI

```bash
docker build -t mafia-rl-training:latest .
```

### Cách 3: Sử dụng Script

```powershell
# Windows
.\run_docker.ps1 -Build

# Linux/Mac
./run_docker.sh --build
```

## 🎯 Chạy Training

### Phương Án 1: Docker Compose (Đơn Giản Nhất)

```bash
# Chạy với config trong docker-compose.yml
docker-compose up

# Chạy trong background
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng
docker-compose down
```

### Phương Án 2: Sử Dụng Script PowerShell (Windows)

#### A. Training cơ bản (không upload)
```powershell
.\run_docker.ps1
```

#### B. Training với Hugging Face upload (chỉ official run)
```powershell
.\run_docker.ps1 -HfRepoId "yourusername/mafia-model"
```

#### C. Training với tham số tùy chỉnh
```powershell
.\run_docker.ps1 `
  -HparamTrials 5 `
  -HparamEpochs 5 `
  -TrainEpochs 50 `
  -NumSeeds 3 `
  -HfRepoId "yourusername/mafia-model"
```

#### D. Upload tất cả seed runs
```powershell
.\run_docker.ps1 `
  -HfRepoId "yourusername/mafia-model" `
  -UploadAll
```

#### E. Rebuild image và chạy
```powershell
.\run_docker.ps1 -Build -HfRepoId "yourusername/mafia-model"
```

#### F. Xem hướng dẫn
```powershell
.\run_docker.ps1 -Help
```

### Phương Án 3: Sử Dụng Script Bash (Linux/Mac)

```bash
# Make script executable
chmod +x run_docker.sh

# Training cơ bản
./run_docker.sh

# Training với upload
./run_docker.sh --hf-repo-id "yourusername/mafia-model"

# Training với tham số tùy chỉnh
./run_docker.sh \
  --hparam-trials 5 \
  --hparam-epochs 5 \
  --train-epochs 50 \
  --num-seeds 3 \
  --hf-repo-id "yourusername/mafia-model"

# Upload tất cả seed runs
./run_docker.sh \
  --hf-repo-id "yourusername/mafia-model" \
  --upload-all

# Rebuild và chạy
./run_docker.sh --build --hf-repo-id "yourusername/mafia-model"

# Xem hướng dẫn
./run_docker.sh --help
```

### Phương Án 4: Docker CLI Trực Tiếp

```bash
# Run với mount volumes
docker run --rm -it \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/res:/app/res \
  -v ${PWD}/log:/app/log \
  -e HF_TOKEN="your-token" \
  mafia-rl-training:latest \
  python scripts/auto_pipeline.py \
    --hparam-trials 10 \
    --hparam-epochs 10 \
    --train-epochs 100 \
    --num-seeds 5 \
    --hf-repo-id "yourusername/mafia-model" \
    --hf-upload-official-only
```

## 📊 Kiểm Tra Kết Quả

### 1. Trong khi training đang chạy

```bash
# Xem logs real-time
docker logs -f mafia-training

# Hoặc với docker-compose
docker-compose logs -f
```

### 2. Sau khi training xong

Kết quả sẽ được lưu trong các thư mục được mount:

```
./res/     # Checkpoints và model files
./log/     # Training logs
./data/    # Data files
```

### 3. Kiểm tra trên Hugging Face

Truy cập: `https://huggingface.co/yourusername/mafia-model`

## 🔧 Các Tùy Chọn Nâng Cao

### 1. Sử Dụng GPU (NVIDIA)

Uncomment phần GPU trong `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Hoặc với Docker CLI:

```bash
docker run --rm -it --gpus all \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/res:/app/res \
  -v ${PWD}/log:/app/log \
  mafia-rl-training:latest \
  python scripts/auto_pipeline.py
```

### 2. Mount Source Code (Development Mode)

Uncomment phần volumes trong `docker-compose.yml`:

```yaml
volumes:
  - ./scripts:/app/scripts
  - ./RL_controller:/app/RL_controller
  - ./utils:/app/utils
```

Điều này cho phép sửa code mà không cần rebuild image.

### 3. Chạy Interactive Shell

```bash
docker run --rm -it \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/res:/app/res \
  mafia-rl-training:latest \
  bash
```

Sau đó có thể chạy lệnh thủ công:

```bash
python scripts/auto_pipeline.py --help
python train.py
```

### 4. Giới Hạn Resource

```yaml
# Thêm vào docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

## 🐛 Troubleshooting

### Lỗi: "Cannot connect to Docker daemon"

```bash
# Windows: Khởi động Docker Desktop
# Linux: Start Docker service
sudo systemctl start docker
```

### Lỗi: "Permission denied"

```bash
# Linux: Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

### Lỗi: "Out of memory"

```bash
# Tăng memory cho Docker Desktop
# Settings → Resources → Memory → 8GB+

# Hoặc giảm số seeds
.\run_docker.ps1 -NumSeeds 3
```

### Lỗi: "No space left on device"

```bash
# Clean up Docker
docker system prune -a

# Check disk usage
docker system df
```

### Container bị dừng đột ngột

```bash
# Xem logs để debug
docker logs mafia-training

# Hoặc
docker-compose logs
```

### Build image quá lâu

```bash
# Sử dụng cache
docker build --cache-from mafia-rl-training:latest -t mafia-rl-training:latest .

# Hoặc build với multi-stage nếu cần optimize
```

## 📝 Tips & Best Practices

### 1. Development Workflow

```bash
# 1. Build image lần đầu
docker-compose build

# 2. Chạy với mounted source (sửa docker-compose.yml trước)
docker-compose up

# 3. Sửa code trong ./scripts/, ./utils/, etc.
# Code thay đổi ngay lập tức, không cần rebuild

# 4. Rebuild khi thay đổi dependencies
docker-compose build
```

### 2. Production Workflow

```bash
# 1. Build optimized image (không mount source)
docker build -t mafia-rl-training:prod .

# 2. Run với auto-restart
docker run -d --restart=unless-stopped \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/res:/app/res \
  -v ${PWD}/log:/app/log \
  mafia-rl-training:prod \
  python scripts/auto_pipeline.py ...
```

### 3. Backup Checkpoints

```bash
# Tự động backup sang HF
.\run_docker.ps1 -HfRepoId "yourusername/mafia-model"

# Hoặc manual copy
docker cp mafia-training:/app/res ./backup/
```

### 4. Monitor Training

```bash
# Terminal 1: Run training
docker-compose up

# Terminal 2: Monitor logs
docker-compose logs -f

# Terminal 3: Check resources
docker stats mafia-training
```

## 📚 Tham Khảo Thêm

- Docker Documentation: https://docs.docker.com/
- Docker Compose: https://docs.docker.com/compose/
- Hugging Face Hub: https://huggingface.co/docs/hub/
- MAFIA Training Guide: `HUGGINGFACE_UPLOAD_GUIDE.md`

## 🎓 Ví Dụ Hoàn Chỉnh

### Scenario 1: Quick Test

```powershell
# Windows - Quick test với 2 seeds, ít epochs
.\run_docker.ps1 -Build -HparamTrials 2 -HparamEpochs 2 -TrainEpochs 10 -NumSeeds 2
```

### Scenario 2: Full Training với Upload

```powershell
# Windows - Full training và upload lên HF
$env:HF_TOKEN = "hf_xxxxx"
.\run_docker.ps1 `
  -HfRepoId "yourusername/mafia-vnindex" `
  -HparamTrials 10 `
  -HparamEpochs 10 `
  -TrainEpochs 100 `
  -NumSeeds 10 `
  -Build
```

### Scenario 3: Production Long-Running

```bash
# Linux - Production run with docker-compose
# Edit docker-compose.yml first with your parameters
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop when done
docker-compose down
```

Chúc bạn training thành công! 🚀
