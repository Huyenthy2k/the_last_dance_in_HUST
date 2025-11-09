# Checkpoint Training Guide

Hệ thống checkpoint cho phép bạn lưu và resume training, đặc biệt hữu ích khi train trên Kaggle với time limit 9 giờ.

## Cấu hình Checkpoint

Trong `config.py`, có các tham số checkpoint:

```python
self.checkpoint_freq = 1  # Save checkpoint mỗi N epochs (0 để disable)
self.checkpoint_dir = os.path.join(self.res_dir, 'checkpoints')
self.resume_from_checkpoint = None  # Path đến checkpoint để resume
```

### Thay đổi tần suất checkpoint

```python
# Trong config.py
self.checkpoint_freq = 5  # Lưu mỗi 5 epochs
# hoặc
self.checkpoint_freq = 10  # Lưu mỗi 10 epochs
# hoặc
self.checkpoint_freq = 0  # Tắt checkpoint
```

## Cấu trúc Checkpoint

Mỗi checkpoint được lưu trong thư mục:
```
res/{mode}/{rl_model_name}/{market_name}-{topK}/{datetime}/checkpoints/checkpoint_epoch_{N}/
├── rl_model.zip              # RL model (TD3)
├── mafia_observer.pth        # MAFIA observer model (nếu có)
└── checkpoint_info.json      # Thông tin checkpoint
```

## Sử dụng Checkpoint

### 1. Lưu Checkpoint Tự Động

Checkpoint sẽ tự động được lưu mỗi `checkpoint_freq` epochs trong quá trình training.

### 2. Resume Training

Để resume từ checkpoint, set `resume_from_checkpoint` trong config:

```python
# Trong config.py hoặc khi tạo Config object
config = Config(seed_num=2022)
config.resume_from_checkpoint = 'res/RLcontroller/TD3/DJIA-10/2024-01-01-12-00-00/checkpoints/checkpoint_epoch_10/checkpoint_info.json'
```

Hoặc chỉ định path đến checkpoint directory:

```python
config.resume_from_checkpoint = 'res/.../checkpoints/checkpoint_epoch_10/checkpoint_info.json'
```

### 3. Ví dụ: Training trên Kaggle

#### Session 1: Train 30 epochs đầu
```python
config = Config()
config.num_epochs = 30
config.checkpoint_freq = 5  # Lưu mỗi 5 epochs
# Training sẽ lưu checkpoint ở epoch 5, 10, 15, 20, 25, 30
```

#### Session 2: Resume từ epoch 20
```python
config = Config()
config.num_epochs = 50  # Tổng 50 epochs
config.checkpoint_freq = 5
config.resume_from_checkpoint = '/kaggle/working/res/.../checkpoints/checkpoint_epoch_20/checkpoint_info.json'
# Training sẽ tiếp tục từ epoch 20 đến 50
```

## Lưu ý

1. **Checkpoint bao gồm:**
   - RL model (TD3) state
   - MAFIA observer model state (nếu có)
   - Optimizer state
   - Learning rate scheduler state
   - Epoch number

2. **Khi resume:**
   - Training sẽ tiếp tục từ epoch đã lưu
   - Số epochs còn lại = `num_epochs - start_epoch`
   - Tất cả states (optimizer, scheduler) được restore

3. **Best model:**
   - Best model vẫn được lưu riêng trong `res_model_dir`
   - Checkpoint chỉ lưu model tại thời điểm đó

## Troubleshooting

### Checkpoint không được lưu
- Kiểm tra `checkpoint_freq > 0`
- Kiểm tra `checkpoint_dir` có được tạo không
- Kiểm tra quyền ghi file

### Resume không hoạt động
- Kiểm tra path đến `checkpoint_info.json` đúng không
- Kiểm tra `rl_model.zip` và `mafia_observer.pth` có tồn tại không
- Đảm bảo config giống với lúc training (số stocks, market_name, etc.)

### MAFIA observer không load
- Kiểm tra `enable_market_observer = True`
- Kiểm tra `mktobs_algo = 'mafia_1'`
- Kiểm tra file `mafia_observer.pth` có tồn tại không

