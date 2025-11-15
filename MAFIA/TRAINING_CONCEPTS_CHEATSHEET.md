# MAFIA Training Concepts - Quick Reference Cheatsheet

## 📋 Core Concepts at a Glance

### 🔢 Numeric Values

| Concept | Value | Description |
|---------|-------|-------------|
| **num_epochs** | 100 | Số lần chạy qua toàn bộ training set |
| **totalTradeDay** | ~1,260 | Số ngày giao dịch trong train period (2017-2021) |
| **total_timesteps** | ~126,000 | num_epochs × totalTradeDay |
| **T_w (window_size)** | 30 days | Kích thước cửa sổ quan sát MAFIA |
| **batch_size** | 32 | Số samples mỗi gradient update |
| **buffer_size** | 1,000,000 | Dung lượng max Replay Buffer |
| **learning_starts** | 100 | Số timesteps trước khi bắt đầu train |
| **train_freq** | [1, 'episode'] | Train sau mỗi episode |
| **gradient_steps** | -1 (auto) | Số gradient updates = totalTradeDay |
| **policy_delay** | 2 | Actor update sau mỗi 2 Critic updates |
| **tau (τ)** | 0.005 | Soft update coefficient cho target networks |
| **gamma (γ)** | 0.99 | Discount factor cho future rewards |
| **learning_rate (TD3)** | 0.0001 | Learning rate cho Actor và Critic |
| **learning_rate (MAFIA)** | 0.0001 | Learning rate cho MAFIA Observer |

---

## 🔄 Hierarchy of Loops

```
TRAINING
├── EPOCH (100 iterations)
│   ├── EPISODE (1 per epoch)
│   │   ├── TIMESTEP (1,260 per episode)
│   │   │   ├── OBSERVATION: Extract window, MAFIA predict
│   │   │   ├── ACTION: RL predict, CBF adjust
│   │   │   ├── EXECUTION: env.step()
│   │   │   └── STORAGE: Store to buffer
│   │   └── TRAINING (after episode)
│   │       ├── TD3: 1,260 gradient steps
│   │       └── MAFIA: 1 policy gradient update
│   └── EVALUATION & CHECKPOINT
└── DONE
```

---

## 🎯 Key Relationships

### Timestep ↔ Trading Day
```
1 Timestep = 1 Trading Day
```
- Mỗi timestep, agent thực hiện 1 trading decision
- Environment tiến 1 ngày, nhận price data của ngày tiếp theo

### Episode ↔ Epoch
```
1 Episode = 1 Epoch (trong setup hiện tại)
```
- 1 Episode = chạy từ đầu đến cuối train period
- 1 Episode = totalTradeDay timesteps

### Window ↔ Timestep
```
At timestep t: Window = [t-29, t-28, ..., t-1, t]
```
- Window **trượt** theo mỗi timestep
- Window size **cố định** = 30 days
- MAFIA nhìn 30 ngày quá khứ để dự đoán

### Batch ↔ Buffer
```
Batch = Random sample 32 transitions from Buffer
```
- Batch size **≠** episode length
- Mỗi gradient step sample 1 batch mới
- Sampling **with replacement** (có thể lặp lại)

### Gradient Steps ↔ Episode Length
```
gradient_steps = -1 → gradient_steps = totalTradeDay
```
- Auto-adjust để train tương xứng với data collected
- 1 Episode có 1,260 transitions → train với 1,260 gradient updates

---

## 📊 Update Frequencies (Per Epoch)

| Component | Frequency | Total per Epoch | Note |
|-----------|-----------|-----------------|------|
| **MAFIA Predict** | Every timestep | 1,260 | Forward pass, store with gradient |
| **TD3 Predict** | Every timestep | 1,260 | Forward pass, store to buffer |
| **Replay Buffer Store** | Every timestep | 1,260 | Add new transition |
| **TD3 Critic Update** | Every gradient step | 1,260 | After episode completes |
| **TD3 Actor Update** | Every 2 gradient steps | 630 | Delayed policy update |
| **Target Networks Soft Update** | With Actor update | 630 | τ=0.005 |
| **MAFIA Observer Update** | Once per episode | 1 | Policy Gradient on full trajectory |

**Total after 100 Epochs:**
- TD3 Critic updates: **126,000**
- TD3 Actor updates: **63,000**
- MAFIA updates: **100**

---

## 🧠 Memory & Storage

### Replay Buffer Contents

Each transition stores:
```python
Transition = {
    'state': (state_dim,),          # Portfolio state + market info
    'action': (action_dim,),         # Normalized [-1,1] for TD3
    'reward': float,                 # Scalar reward
    'next_state': (state_dim,),      # Next state
    'done': bool                     # Episode termination flag
}
```

### Buffer Growth Pattern

```
Day 1:     [T₁]                                Size: 1
Day 2:     [T₁, T₂]                            Size: 2
Day 100:   [T₁, ..., T₁₀₀]                     Size: 100 ← Learning starts!
Day 1,260: [T₁, ..., T₁₂₆₀]                    Size: 1,260 ← Episode end

After Epoch 1 training: Buffer still has all 1,260 transitions

Epoch 2 starts:
Day 1,261: [T₁, ..., T₁₂₆₀, T₁₂₆₁]             Size: 1,261
Day 2,520: [T₁, ..., T₂₅₂₀]                    Size: 2,520

...continues until buffer_size = 1,000,000 (FIFO after that)
```

---

## 🎓 Training Phases Detail

### Phase 1: Collection (During Episode)

```
for day in range(1, totalTradeDay+1):
    # 1. Observation
    window = extract_window(day-30, day)  # [day-29, ..., day]
    market_vector, scores, risk = mafia.predict(window)
    state = construct_state(portfolio, market_vector)
    
    # 2. Action
    a_rl = actor.predict(state)
    a_cbf = cbf_solver.solve(a_rl, risk)
    a_final = a_rl + a_cbf
    
    # 3. Execution
    next_state, reward, done = env.step(a_final)
    
    # 4. Storage
    buffer.add(state, a_rl, reward, next_state, done)
    mafia.store_for_pg(market_vector, rate_of_price_change)
    
    if done:
        break
```

### Phase 2: Training (After Episode)

```
# TD3 Training
for grad_step in range(gradient_steps):  # gradient_steps = 1,260
    # Sample batch
    batch = buffer.sample(batch_size=32)
    
    # Update Critic (every step)
    target_q = reward + gamma * min(Q1_target, Q2_target)(s', a')
    critic_loss = MSE(Q1(s,a), target_q) + MSE(Q2(s,a), target_q)
    critic_optimizer.step()
    
    # Update Actor (every 2 steps)
    if grad_step % policy_delay == 0:
        actor_loss = -mean(Q1(s, actor(s)))
        actor_optimizer.step()
        
        # Soft update targets
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
        # Same for critics

# MAFIA Training (once)
rewards = compute_rewards(market_vectors, rate_of_price_changes)
mafia_loss = -mean(rewards)  # Policy Gradient
mafia_optimizer.step()
mafia_lr_scheduler.step()
```

---

## 🔍 State & Action Dimensions

### MAFIA Input
```
OCHLV Window: (N, M, T_w) = (10, 5, 30)
  ├─ N = 10: Number of stocks
  ├─ M = 5: Open, Close, High, Low, Volume
  └─ T_w = 30: Window size (days)
```

### MAFIA Output
```
market_vector: (N,) = (10,)
  - Sparse Top-K weights (K=10 in this case, so all non-zero)
  - Sum = 1, values ∈ [0, 1]

market_scores_full: (N,) = (10,)
  - Dense scores for all stocks
  - Sum = 1, values ∈ [0, 1]
  - Used in full-score mode or for CBF prior

boundary_risk: scalar
  - Continuous risk value
  - Used by CBF controller
```

### RL State (Compact Mode)
```
state = [portfolio_features, market_vector]
  ├─ portfolio_features: Variable dim (depends on env)
  │   ├─ Current positions
  │   ├─ Available cash
  │   ├─ Portfolio value
  │   └─ Other env-specific features
  └─ market_vector: (K,) = (10,) - Top-K from MAFIA
```

### RL Action
```
a_rl: (N,) = (10,)
  - Raw RL output (before CBF)
  - Range: [0, 1], sum = 1
  - Portfolio weights

a_cbf: (N,) = (10,)
  - CBF adjustment
  - sum(a_cbf) = 0 (constraint)
  - Applied only if CBF solver successful

a_final: (N,) = (10,)
  - Final action = a_rl + a_cbf (if CBF succeeds)
  - Final action = a_rl (if CBF fails)
  - Normalized to sum = 1, all ≥ 0
```

---

## 🎯 Two Training Modes Compared

### TD3 (Off-Policy, Step-Level)

| Aspect | Detail |
|--------|--------|
| **Learning Signal** | TD Error (Temporal Difference) |
| **Update Frequency** | Every gradient step (after episode) |
| **Data Source** | Replay Buffer (random samples) |
| **Batch Size** | 32 |
| **Gradient Steps** | 1,260 per epoch |
| **Advantage** | Sample efficient, stable |
| **Key Trick** | Clipped Double Q-learning, Target smoothing, Delayed updates |

### MAFIA (On-Policy, Episode-Level)

| Aspect | Detail |
|--------|--------|
| **Learning Signal** | Policy Gradient (REINFORCE-style) |
| **Update Frequency** | Once per episode |
| **Data Source** | Episode trajectory (sequential) |
| **Batch Size** | Full episode (1,260 timesteps) |
| **Gradient Steps** | 1 per epoch |
| **Advantage** | End-to-end differentiable, learns episode-level strategy |
| **Key Trick** | Gumbel-Softmax for Top-K, Dense MoE gating |

---

## 💡 Common Pitfalls & Tips

### ❌ Pitfall 1: Confusing Batch Size with Episode Length
- **Wrong**: "Batch size = 32 means we only train on 32 timesteps"
- **Right**: "Batch size = 32 means each gradient update uses 32 samples, but we do 1,260 gradient updates per episode"

### ❌ Pitfall 2: Thinking Window Size Limits Training Start
- **Wrong**: "Need to wait 30 days before starting training"
- **Right**: "Environment handles initial days with padding/defaults. Learning starts after 100 timesteps (not days)"

### ❌ Pitfall 3: Expecting Same Update Frequency
- **Wrong**: "MAFIA and TD3 update same number of times"
- **Right**: "MAFIA updates 1×/epoch, TD3 updates ~1,890×/epoch (1,260 Critic + 630 Actor)"

### ✅ Tip 1: Monitor _n_updates Counter
```python
# In callback or after training
print(f"Total network updates: {model._n_updates}")
# Should be ~126,000 after 100 epochs (Critic updates)
```

### ✅ Tip 2: Check Buffer Size
```python
print(f"Buffer size: {model.replay_buffer.size()}")
# Should grow from 0 to min(total_timesteps, buffer_size)
```

### ✅ Tip 3: Validate Action Distributions
```python
# Actions should be valid portfolio weights
assert np.allclose(np.sum(action), 1.0), "Action weights don't sum to 1"
assert np.all(action >= 0), "Action has negative weights"
assert np.all(action <= 1), "Action has weights > 1"
```

---

## 🔗 File References

| File | Purpose |
|------|---------|
| `entrance.py` | Main training loop, epoch management |
| `config.py` | Hyperparameters, data paths |
| `TD3_controller.py` | TD3 implementation, collect & train |
| `mafia_observer.py` | MAFIA Observer, predict & train |
| `tradeEnv.py` | Trading environment, state/reward |
| `controllers.py` | CBF Controller, SOCP solver |
| `callback_func.py` | Evaluation, checkpointing |

---

## 📚 Math Formulas Quick Ref

### TD3 Losses

**Critic Loss:**
```
Q_target = r + γ × min(Q₁'(s', a'), Q₂'(s', a'))
where a' = actor_target(s') + noise

L_critic = MSE(Q₁(s,a), Q_target) + MSE(Q₂(s,a), Q_target)
```

**Actor Loss:**
```
L_actor = -𝔼[Q₁(s, actor(s))]
```

**Target Update:**
```
θ_target ← τ×θ + (1-τ)×θ_target
where τ = 0.005
```

### MAFIA Loss

**Policy Gradient:**
```
reward_t = log(1 + Σᵢ (rate_i,t - 1) × market_vector_i,t)

L_mafia = -𝔼[reward_1 + reward_2 + ... + reward_T]
where T = totalTradeDay
```

### CBF Constraint

**SOCP Formulation:**
```
minimize   ||a_cbf||²
subject to:
  • √(w'Σw) ≤ risk_safe - risk_market  (Risk constraint)
  • sum(a_cbf) = 0                      (Adjustment sum)
  • 0 ≤ a_rl + a_cbf ≤ 1                (Bounds)
  • w = a_rl + a_cbf                     (Combined weights)
```

---

## 🎓 Training Progress Indicators

### What to Monitor

| Metric | Expected Behavior | Red Flag |
|--------|-------------------|----------|
| **Portfolio Value** | Increasing over epochs | Constant or decreasing |
| **Actor Loss** | Decreasing, then stable | Exploding or oscillating wildly |
| **Critic Loss** | Decreasing steadily | Not decreasing or NaN |
| **MAFIA Loss** | Decreasing (negative, so increasing reward) | Constant or not learning |
| **_n_updates** | Steadily increasing | Not increasing |
| **Buffer Size** | Growing to capacity | Not growing |
| **Solver Success Rate** | > 70% | < 30% |
| **Action Entropy** | High early, lower later | Constant high (not learning) |

### Sample Training Output

```
Epoch 1/100:
  Rollout: 1,260 steps collected
  TD3: 1,260 gradient steps
    - Critic loss: 0.523
    - Actor loss: -45.2
  MAFIA: 1 update
    - Loss: -8.3 (reward: 8.3)
  Portfolio value: $1,050,000 (+5.0%)

Epoch 50/100:
  Rollout: 1,260 steps collected
  TD3: 1,260 gradient steps
    - Critic loss: 0.089
    - Actor loss: -102.5
  MAFIA: 1 update
    - Loss: -15.7 (reward: 15.7)
  Portfolio value: $1,320,000 (+32.0%)

Epoch 100/100:
  Rollout: 1,260 steps collected
  TD3: 1,260 gradient steps
    - Critic loss: 0.042
    - Actor loss: -125.8
  MAFIA: 1 update
    - Loss: -18.4 (reward: 18.4)
  Portfolio value: $1,480,000 (+48.0%)
```

---

## 🚀 Quick Start Commands

```bash
# Start training
cd agents/MAFIA
python entrance.py

# Monitor training (if using nohup)
tail -f nohup.out

# Watch GPU usage
watch -n 1 nvidia-smi

# Check checkpoints
ls -lh res/RLcontroller/TD3/VNINDEX-10/*/checkpoints/

# Resume from checkpoint
# Edit config.py:
#   self.resume_from_checkpoint = 'path/to/checkpoint_info.json'
#   self.auto_resume_from_latest = False
python entrance.py
```

---

## 📖 Further Reading

1. **TD3 Paper**: "Addressing Function Approximation Error in Actor-Critic Methods"
   - https://arxiv.org/abs/1802.09477

2. **Policy Gradient**: Sutton & Barto, "Reinforcement Learning: An Introduction"
   - Chapter 13: Policy Gradient Methods

3. **Replay Buffer**: "Human-level control through deep reinforcement learning" (DQN paper)
   - https://www.nature.com/articles/nature14236

4. **CBF Controller**: "Control Barrier Functions: Theory and Applications"
   - https://arxiv.org/abs/1903.11199

---

**Last Updated**: 2025-11-13  
**Version**: 1.0  
**Maintainer**: AI Assistant

