# Hướng Dẫn Upload Checkpoint Lên Hugging Face

## Cài Đặt

1. **Cài đặt dependencies mới:**
```bash
pip install -r requirements.txt
```

## Cấu Hình Hugging Face

### Cách 1: Sử dụng biến môi trường (Khuyến nghị)

```bash
# Windows PowerShell
$env:HF_TOKEN = "your-huggingface-token-here"

# Hoặc thêm vào system environment variables để sử dụng lâu dài
```

### Cách 2: Truyền token qua argument

```bash
python scripts/auto_pipeline.py --hf-token "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" --hf-repo-id "Siry2k/mafia"
```

### Lấy Hugging Face Token

1. Đăng nhập vào https://huggingface.co
2. Vào Settings → Access Tokens
3. Tạo token mới với quyền "Write"
4. Copy token và lưu lại

## Cách Sử Dụng

### 1. Upload Tất Cả Checkpoints (Official Run + Seed Runs + Logs)

```bash
python scripts/auto_pipeline.py --hf-repo-id "username/mafia-model"
```

Lệnh này sẽ upload:
- Official run checkpoint → `username/mafia-model`
- Seed run checkpoints → `username/mafia-model/seed-{seed_number}`
- Aggregate logs → `username/mafia-model/logs`

### 2. Chỉ Upload Official Run Checkpoint

```bash
python scripts/auto_pipeline.py --hf-repo-id "username/mafia-model" --hf-upload-official-only
```

### 3. Ví Dụ Hoàn Chỉnh

```bash
python scripts/auto_pipeline.py \
  --hparam-trials 5 \
  --hparam-epochs 5 \
  --train-epochs 100 \
  --num-seeds 10 \
  --base-seed 2025 \
  --hf-repo-id "yourusername/mafia-vnindex-model" \
  --hf-upload-official-only
```

### 4. Với Token Trực Tiếp

```bash
python scripts/auto_pipeline.py \
  --hparam-trials 5 \
  --hparam-epochs 5 \
  --train-epochs 100 \
  --hf-repo-id "yourusername/mafia-model" \
  --hf-token "hf_xxxxxxxxxxxxxxxxxxxxx"
```

## Tham Số Mới

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--hf-repo-id` | Repository ID trên Hugging Face (vd: `username/model-name`) | "" (không upload) |
| `--hf-token` | Hugging Face API token | None (dùng env variable) |
| `--hf-upload-official-only` | Chỉ upload official run, không upload seed runs | False |

## Cấu Trúc Upload

```
username/mafia-model/              # Official run
├── checkpoints/
│   └── checkpoint_best_valid/
├── graph/
├── model/
└── test_profile.csv

username/mafia-model/seed-2025/    # Seed run 1
├── checkpoints/
├── graph/
└── ...

username/mafia-model/seed-2026/    # Seed run 2
└── ...

username/mafia-model/logs/         # Aggregate results
├── optuna_trials_*.csv
├── optuna_best_*.json
├── seed_results.csv
├── seed_summary.csv
└── run_info.json
```

## Kiểm Tra Upload

1. Truy cập https://huggingface.co/username/model-name
2. Xem các files đã được upload
3. Download checkpoint để test:

```python
from huggingface_hub import hf_hub_download

# Download một file cụ thể
file_path = hf_hub_download(
    repo_id="username/mafia-model",
    filename="checkpoints/checkpoint_best_valid/rl_model.zip",
    token="your-token"
)
```

## Lưu Ý

1. **Kích thước file**: Hugging Face hỗ trợ files lớn, nhưng upload có thể mất thời gian
2. **Token bảo mật**: Không commit token vào git, sử dụng biến môi trường
3. **Repository visibility**: Mặc định repos là public, có thể đặt private trong settings
4. **Network**: Đảm bảo kết nối internet ổn định khi upload

## Troubleshooting

### Lỗi: "Repository not found"
- Kiểm tra tên repo đúng format: `username/repo-name`
- Đảm bảo token có quyền write

### Lỗi: "Invalid token"
- Tạo token mới với quyền "Write"
- Kiểm tra token không bị space hoặc ký tự thừa

### Upload chậm
- Upload chỉ official run với `--hf-upload-official-only`
- Kiểm tra kết nối internet

## Ví Dụ Script Tự Động

Tạo file `run_and_upload.ps1`:

```powershell
# Set HF token
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxx"

# Run training and auto-upload
python scripts/auto_pipeline.py `
  --hparam-trials 10 `
  --hparam-epochs 10 `
  --train-epochs 100 `
  --num-seeds 5 `
  --hf-repo-id "yourusername/mafia-vnindex" `
  --hf-upload-official-only
```

Chạy script:
```bash
.\run_and_upload.ps1
```
