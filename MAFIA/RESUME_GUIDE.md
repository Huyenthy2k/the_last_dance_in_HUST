# Hướng Dẫn Resume Training từ Checkpoint

## 🚀 Cách 1: Auto Resume (Đơn giản nhất - Khuyến nghị)

Hệ thống sẽ **tự động tìm và resume từ checkpoint mới nhất**:

```bash
cd /Users/nguyensiry/Documents/the_last_dance/agents/MAFIA
./train.sh
```

**Hoặc:**

```bash
cd /Users/nguyensiry/Documents/the_last_dance/agents/MAFIA
./resume_training.sh
```

Hệ thống sẽ:
- ✅ Tự động tìm checkpoint mới nhất (theo timesteps)
- ✅ Load RL model và MAFIA observer
- ✅ Tính remaining timesteps chính xác
- ✅ Tiếp tục training từ điểm đã lưu

---

## 📂 Cách 2: Manual Resume (Chỉ định checkpoint cụ thể)

Nếu muốn resume từ một checkpoint cụ thể:

### Option A: Sửa trong `config.py`

```python
# Trong config.py, tìm dòng:
self.resume_from_checkpoint = None

# Thay đổi thành:
self.resume_from_checkpoint = './res/RLcontroller/TD3/VNINDEX-10/2025-11-14-00-26-31/checkpoints/checkpoint_step_1000/checkpoint_info.json'
self.auto_resume_from_latest = False  # Tắt auto-resume
```

Sau đó chạy:
```bash
./train.sh
```

### Option B: Dùng script với argument

```bash
./resume_training.sh ./res/RLcontroller/TD3/VNINDEX-10/2025-11-14-00-26-31/checkpoints/checkpoint_step_1000/checkpoint_info.json
```

---

## 🔍 Kiểm tra Checkpoints có sẵn

### Liệt kê tất cả checkpoints:

```bash
cd /Users/nguyensiry/Documents/the_last_dance/agents/MAFIA
find res -name "checkpoint_info.json" -type f | sort -V
```

### Xem checkpoint mới nhất:

```bash
cd /Users/nguyensiry/Documents/the_last_dance/agents/MAFIA
find res -name "checkpoint_info.json" -type f | sort -V | tail -1
```

### Xem thông tin checkpoint:

```bash
# Thay path bằng checkpoint của bạn
cat res/RLcontroller/TD3/VNINDEX-10/2025-11-14-00-26-31/checkpoints/checkpoint_step_1000/checkpoint_info.json
```

Output sẽ có dạng:
```json
{
  "type": "step",
  "epoch": 1,
  "day_in_epoch": 1000,
  "timesteps": 1000,
  "timestamp": "2025-11-14 01:10:33",
  "rl_model_path": "...",
  "mafia_observer_path": "..."
}
```

---

## 📊 Checkpoint hiện tại của bạn

**Checkpoint mới nhất**: `checkpoint_step_1000`
- **Epoch**: 1
- **Day in epoch**: 1000/1002
- **Timesteps**: 1000
- **Type**: step (partial checkpoint)

**Path đầy đủ**:
```
res/RLcontroller/TD3/VNINDEX-10/2025-11-14-00-26-31/checkpoints/checkpoint_step_1000/checkpoint_info.json
```

---

## ⚙️ Cấu hình Resume

Trong `config.py`:

```python
# Auto-resume (mặc định)
self.auto_resume_from_latest = True  # Tự động tìm checkpoint mới nhất
self.resume_from_checkpoint = None   # Không chỉ định cụ thể

# Manual resume
self.auto_resume_from_latest = False  # Tắt auto-resume
self.resume_from_checkpoint = 'path/to/checkpoint_info.json'  # Chỉ định cụ thể
```

---

## 🎯 Resume từ checkpoint hiện tại

Với checkpoint `checkpoint_step_1000` của bạn:

### Cách nhanh nhất (Auto):
```bash
cd /Users/nguyensiry/Documents/the_last_dance/agents/MAFIA
./train.sh
```

### Hoặc manual:
```bash
cd /Users/nguyensiry/Documents/the_last_dance/agents/MAFIA

# Sửa config.py:
# self.resume_from_checkpoint = './res/RLcontroller/TD3/VNINDEX-10/2025-11-14-00-26-31/checkpoints/checkpoint_step_1000/checkpoint_info.json'
# self.auto_resume_from_latest = False

./train.sh
```

---

## ✅ Verify Resume

Khi resume thành công, bạn sẽ thấy output như:

```
Auto-detected latest checkpoint: .../checkpoint_step_1000/checkpoint_info.json
Resuming training from checkpoint: ...
Checkpoint type: step
Resuming from epoch 1, day 1000, timestep 1000
RL model loaded from ...
MAFIA observer loaded from ... (epoch 1)
Resuming from timestep 1000/100200
```

---

## 🐛 Troubleshooting

### Nếu không tìm thấy checkpoint:
- Kiểm tra path có đúng không
- Kiểm tra `checkpoint_info.json` có tồn tại không
- Kiểm tra `rl_model.zip` và `mafia_observer.pth` có trong cùng folder không

### Nếu resume sai checkpoint:
- Set `auto_resume_from_latest = False`
- Chỉ định `resume_from_checkpoint` cụ thể

### Nếu muốn start fresh:
```python
# Trong config.py
self.auto_resume_from_latest = False
self.resume_from_checkpoint = None
```

---

## 📝 Notes

- **Partial checkpoints** (step-based) được save mỗi 50 timesteps (theo config)
- **Epoch checkpoints** được save sau mỗi epoch hoàn thành
- Auto-resume sẽ chọn checkpoint có **timesteps cao nhất**
- Resume sẽ tính remaining timesteps chính xác từ checkpoint timesteps

---

**Last Updated**: 2025-11-14

