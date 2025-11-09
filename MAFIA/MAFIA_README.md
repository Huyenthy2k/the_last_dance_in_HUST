# MAFIA (Multi-Agent Fusion and Intelligent Adaptation) Implementation

## Overview

MAFIA is a Multi-Agent Market Observer designed to replace the original Market Observer in the MASA framework. It consists of **1 Technical Agent + 3 DC Agents**, each analyzing market data from different perspectives and fusing their insights to generate `market_vector` and `boundary_risk` outputs.


### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAFIA Framework Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

Timestep t:
    │
    ├─► 1. Trading Environment
    │      Input:  a_t^Final (final portfolio weights)
    │      Output: o_t (market states), J_r (return reward)
    │      └─► Extract raw OCHLV data (T_w=30 window)
    │
    ├─► 2. MAFIA Observer (Market Observer Agent)
    │      Input:  o_t (market states)
    │      Process:
    │        ├─► Extract raw OCHLV window (N × 5 × T_w)
    │        ├─► Technical Agent: Process OCHLV + indicators
    │        ├─► DC Agents (×3): Process DC features at thresholds [0.005, 0.01, 0.02]
    │        ├─► Each agent: CSA → TA → ST-Fusion
    │        └─► Signal Generator: Aggregate all agents
    │      Output:
    │        ├─► v_m,t (market_vector) → RL-based Agent
    │        └─► σ_s,t (boundary_risk) → Solver-based Agent
    │
    ├─► 3. RL-based Agent (TD3)
    │      Input:
    │        ├─► o_t (market states from Environment)
    │        ├─► v_m,t (market_vector from MAFIA Observer)
    │        └─► J_r (return reward from Environment)
    │      Process:
    │        ├─► Concatenate o_t and v_m,t into state vector
    │        ├─► TD3 Actor: π(s_t) → a_t^RL
    │        └─► Add exploration noise (if training)
    │      Output: a_t^RL (portfolio weights proposal) → Solver-based Agent
    │
    ├─► 4. Solver-based Agent (CBF Controller)
    │      Input:
    │        ├─► a_t^RL (from RL-based Agent)
    │        ├─► σ_s,t (boundary_risk from MAFIA Observer)
    │        └─► Price predictions (MA model)
    │      Process:
    │        ├─► Compute covariance matrix from daily returns
    │        ├─► Build CBF constraint: Risk(a_t^RL + a_t^Ctrl) ≤ σ_s,t
    │        ├─► Solve SOCP optimization problem:
    │        │     min ||a_t^Ctrl||²
    │        │     s.t. Risk(a_t^RL + a_t^Ctrl) ≤ σ_s,t
    │        │          Σ(a_t^RL + a_t^Ctrl) = 1
    │        │          0 ≤ (a_t^RL + a_t^Ctrl) ≤ 1
    │        └─► If solver fails: a_t^Ctrl = 0 (fallback to RL action)
    │      Output: a_t^Ctrl (adjustment weights)
    │
    ├─► 5. Combine Actions
    │      a_t^Final = a_t^RL + a_t^Ctrl
    │      └─► Normalize to ensure constraints (sum = 1, all ≥ 0)
    │
    ├─► 6. Execute & Update
    │      Environment:
    │        ├─► Execute a_t^Final
    │        ├─► Calculate portfolio value C_t
    │        ├─► Compute return reward J_r
    │        ├─► Transition to o_{t+1}
    │        └─► Update MAFIA Observer reward (if training)
    │
    └─► 7. Training (End of Episode)
         ├─► RL-based Agent (TD3):
         │     ├─► Sample mini-batch from replay buffer
         │     ├─► Compute J_JS (Jensen-Shannon divergence reward)
         │     ├─► Compute J_Total = λ₁·J_r + λ₂·J_JS
         │     └─► Update Actor and Critic networks
         │
         └─► MAFIA Observer:
               ├─► Collect rewards from episode
               ├─► Compute Policy Gradient loss: -mean(reward)
               └─► Update MAFIA model parameters
```
## Market Observer (MAFIA_Observer))

### Agents

1. **Technical Agent (i=0)**: Analyzes raw prices (OCHLV) + Technical Indicators
   - Features: 8 (5 OCHLV + 3 indicators: SMA(20), RSI(14), ATR(14))
   
2. **DC Agents (i=1,2,3)**: Analyze Directional Change events at different thresholds
   - Thresholds: [0.005, 0.01, 0.02] (0.5%, 1.0%, 2.0%)
   - Features: 5 (State, Magnitude, Duration, Volume_Ratio, Event_Flag)

### Modules (per agent)

Each agent has three modules:

1. **CSA (Cross-Sectional Analysis)**: Learns spatial correlations between N assets
   - Input: P_i ∈ ℝ^(N × T_w × M)
   - Output: O_i^CSA ∈ ℝ^(N × D)

2. **TA (Temporal Analysis)**: Learns temporal correlations between T_w time points
   - Input: P_i ∈ ℝ^(N × T_w × M)
   - Output: O_i^TA ∈ ℝ^(T_w × D)
   - Includes Positional Encoding and Event Detector (DC agents only)

3. **ST-Fusion (Spatial-Temporal Fusion)**: Fuses CSA and TA outputs using attention
   - Input: O_i^CSA, O_i^TA
   - Output: O_i ∈ ℝ^(N × 1) (logits)

### Signal Generator

Aggregates outputs from all agents:
- **market_vector**: v_m,t = Σ O_i ∈ ℝ^(N × 1)
- **boundary_risk**: σ_s,t = RiskHead(Average(O_i^TA)) ∈ ℝ^+

## Usage

### Configuration

Set in `config.py`:
```python
config.benchmark_algo = 'MASA-mafia'  # Enable MAFIA
config.enable_market_observer = True
```

### Hyperparameters

MAFIA hyperparameters (set in `config.py`):
- `mafia_T_w = 30`: Observation window size
- `mafia_DC_thresholds = [0.005, 0.01, 0.02]`: DC thresholds
- `mafia_D = 64`: Embedding dimension
- `mafia_D_h = 128`: Hidden layer dimension
- `mafia_encoder_layers = 2`: Transformer encoder layers
- `mafia_encoder_heads = 4`: Attention heads
- `mafia_M_tech = 8`: Technical agent features
- `mafia_M_dc = 5`: DC agent features
- `mafia_learning_rate = 1e-4`: Learning rate
- `mafia_weight_decay = 0.001`: Weight decay

## RL-based Agent (TD3)

### Overview

The RL-based Agent uses the **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** algorithm to optimize portfolio allocation for maximum returns. It receives market states and market_vector from MAFIA Observer to make informed portfolio decisions.

### Architecture

**Algorithm**: TD3 (Twin Delayed DDPG)
- **Actor Network**: Generates deterministic actions (portfolio weights)
- **Critic Networks**: Two Q-networks (twin) to reduce overestimation bias
- **Target Networks**: Delayed updates for stable training

### Input Processing

**State Composition**:
- **Market States** (`o_t`): Flattened market features
  - Shape: (N × L × F) → flattened to 1D vector
  - Includes: price features, technical indicators, portfolio value ratio
- **Market Vector** (`v_m,t`): From MAFIA Observer
  - Shape: (N,) - Market trend vector for N assets
  - Provides market sentiment and trend information

**State Concatenation**:
```python
state_full = np.concatenate([o_t, v_m,t])  # Combined state vector
```

### Actor Network (Policy)

**Architecture**: `ActorAdj` (Adjusted Actor for MASA)

**Forward Pass**:
1. **Feature Extraction**: Extract features from full state
2. **Split State**: 
   - `features[:, :-N]` → TD3 decision path
   - `features[:, -N:]` → Market vector (from MAFIA)
3. **TD3 Decision**: MLP → Softmax → `td3_decision` (range [0, 1], sum=1)
4. **Market Decision**: Direct use of `v_m,t` (range [0, 1], sum=1)
5. **Final Output**: `(td3_decision + mkt_decision) - 1` (range [-1, 1])

**Output**: `a_t^RL` - Portfolio weights proposal (N assets)

### Reward Function

**Combined Reward**: `J_Total = λ₁·J_r + λ₂·J_JS`

**Components**:

1. **Return Reward** (`J_r`):
   - **Formula**: `J_r = log(1 + r_t)` where `r_t = C_t / C_{t-1}`
   - **Description**: Instantaneous log return of portfolio
   - **Weight**: `λ₁ = 1000.0` (default)
   - **Purpose**: Maximize portfolio returns

2. **Jensen-Shannon Divergence Reward** (`J_JS`):
   - **Formula**: `J_JS = -D_JS(a_t^RL || a_t^Final)`
   - **Description**: Measures divergence between RL proposal and final action
   - **Weight**: `λ₂ = 10.0` (default)
   - **Purpose**: Guide RL agent to learn from risk-adjusted final actions

### Training Process

**Algorithm**: TD3 (Off-policy, Actor-Critic)

**Steps**:
1. **Experience Collection**: Store transitions `(s_t, a_t, r_t, s_{t+1})` in replay buffer
2. **Mini-batch Sampling**: Sample random batches from replay buffer
3. **Critic Update**: 
   - Compute target Q-values using target networks
   - Update both Q-networks with TD error
4. **Actor Update** (delayed):
   - Update every `policy_delay` steps (default: 2)
   - Use deterministic policy gradient
5. **Target Network Update**: Soft update with `τ = 0.005`

**Hyperparameters** (in `config.py`):
- `learning_rate = 0.0001`: Learning rate for all networks
- `batch_size = 50`: Mini-batch size
- `buffer_size = 1000000`: Replay buffer size
- `tau = 0.005`: Target network update coefficient
- `gamma = 0.99`: Discount factor
- `policy_delay = 2`: Actor update frequency
- `target_policy_noise = 0.2`: Noise for target policy smoothing
- `target_noise_clip = 0.5`: Clip range for target noise

### Output

**Action**: `a_t^RL ∈ ℝ^N`
- Portfolio weights proposal for N assets
- Range: [-1, 1] (before normalization)
- Normalized by environment to ensure constraints

## Solver-based Agent (CBF Controller)

### Overview

The Solver-based Agent uses **Controller Barrier Function (CBF)** method to adjust RL actions to meet dynamic risk constraints. It solves a Second-Order Cone Programming (SOCP) optimization problem to ensure portfolio risk stays within the boundary provided by MAFIA Observer.

### Architecture

**Method**: CBF (Controller Barrier Function) with SOCP Optimization

**Solver**: 
- **Small portfolios** (N ≤ 10): `cvxopt.solvers.coneqp`
- **Large portfolios** (N > 10): `cvxpy` with ECOS solver

### Input Processing

**Inputs**:
1. **RL Action** (`a_t^RL`): Portfolio proposal from RL agent
   - Shape: (N,) - Portfolio weights
   - Constraint: `Σ|a_t^RL| = 1` (normalized)

2. **Risk Boundary** (`σ_s,t`): From MAFIA Observer
   - Type: Continuous scalar (ℝ^+)
   - Dynamic: Changes based on market conditions
   - Usage: Upper bound for portfolio risk

3. **Price Predictions**: Short-term price change predictions
   - Model: Moving Average (MA)
   - Window: 5 days (configurable)
   - Purpose: Predict future returns for risk calculation

4. **Historical Returns**: Daily returns for covariance estimation
   - Lookback: 5 days (configurable via `dailyRetun_lookback`)
   - Purpose: Estimate portfolio risk via covariance matrix

### Risk Calculation

**Portfolio Risk** (Markowitz Variance Model):
```
Risk(w) = √(w^T · Σ · w)
```
where:
- `w`: Portfolio weights vector
- `Σ`: Covariance matrix of daily returns

**Covariance Matrix**:
- Computed from historical daily returns
- Updated with predicted returns for future risk estimation

### CBF Optimization Problem

**Objective Function**:
```
min ||a_t^Ctrl||²
```

**Constraints**:

1. **Risk Constraint** (CBF):
   ```
   Risk(a_t^RL + a_t^Ctrl) ≤ σ_s,t
   ```
   - Expressed as Second-Order Cone constraint
   - Ensures portfolio risk stays within MAFIA boundary

2. **Sum Constraint**:
   ```
   Σ(a_t^RL + a_t^Ctrl) = 0
   ```
   - Ensures adjustment doesn't change total weight

3. **Bounds**:
   ```
   0 ≤ (a_t^RL + a_t^Ctrl) ≤ 1, ∀i
   ```
   - No short selling
   - No leverage

4. **CBF Safety Condition**:
   ```
   ḣ(x) ≥ -γ·h(x)
   ```
   where `h(x) = σ_s,t - Risk(w)` and `γ = 0.7` (default)

### Optimization Process

**Steps**:

1. **Compute Current Risk** (`risk_stg_t0`):
   - Calculate risk of current portfolio
   - Use covariance from historical returns

2. **Predict Future Risk** (`risk_stg_t1`):
   - Append predicted returns to historical returns
   - Compute future covariance matrix
   - Calculate risk of proposed portfolio

3. **Build SOCP Problem**:
   - Formulate as Second-Order Cone Program
   - Include all constraints

4. **Solve Optimization**:
   - Attempt to solve with initial risk boundary
   - If fails: Relax risk boundary iteratively (up to `ars_trial` times)
   - Step size: [0.002, 0.002, ..., 0.005, 0.005, ...]

5. **Handle Solution**:
   - **If solvable**: Return `a_t^Ctrl` adjustment
   - **If unsolvable**: Return zero adjustment (fallback to RL action)

### Output

**Action Adjustment**: `a_t^Ctrl ∈ ℝ^N`
- Adjustment weights to add to RL action
- Range: Unbounded (but constrained by optimization)
- Final action: `a_t^Final = a_t^RL + a_t^Ctrl`

**Status**: `is_solvable` (boolean)
- Indicates whether optimization succeeded
- Used to decide whether to use adjustment or fallback

### Hyperparameters

**CBF Parameters** (in `config.py`):
- `cbf_gamma = 0.7`: CBF safety parameter
- `ars_trial = 10`: Maximum risk relaxation attempts
- `risk_market = 0.001`: Market risk component (Σ_β)
- `risk_up_bound = 0.012`: Risk bound for up market
- `risk_hold_bound = 0.014`: Risk bound for hold market
- `risk_down_bound = 0.017`: Risk bound for down market
- `risk_default = 0.017`: Default risk bound
- `dailyRetun_lookback = 5`: Days for covariance calculation
- `otherRef_indicator_ma_window = 5`: MA window for price prediction

### Integration with MAFIA

**Dynamic Risk Boundary**:
- MAFIA provides continuous `σ_s,t` based on market analysis
- Solver uses this as constraint in optimization
- If MAFIA indicates high risk, solver tightens constraints
- If MAFIA indicates low risk, solver allows more aggressive allocation

**Fallback Mechanism**:
- If solver fails to find solution within risk boundary
- System falls back to RL action: `a_t^Final = a_t^RL`
- Risk constraint is logged but not enforced

## Files

### Core Implementation

- `RL_controller/mafia_observer.py`: Main MAFIAObserver class (MarketObserver interface)
- `RL_controller/mafia_modules.py`: MAFIA model modules (CSA, TA, ST-Fusion, Signal Generator)
- `RL_controller/mafia_feature_processor.py`: Feature preprocessing (Technical indicators, DC features)

### Integration

- `config.py`: MAFIA hyperparameters and configuration
- `entrance.py`: MAFIAObserver instantiation
- `utils/tradeEnv.py`: Environment integration

### Testing

- `test_mafia_components.py`: Unit tests for individual components
- `test_mafia_end_to_end.py`: End-to-end integration tests

## Running Tests

### Unit Tests
```bash
python test_mafia_components.py
```

### End-to-End Tests
```bash
python test_mafia_end_to_end.py
```

## Interface Compatibility

MAFIAObserver implements the same interface as MarketObserver:

```python
# Initialization
observer = MAFIAObserver(config=config, action_dim=stock_num)

# Prediction
market_vector, lambda_val, boundary_risk = observer.predict(
    raw_ochlv_data=ochlv_data,  # (N, 5, T_w) or use finemkt_feat, finestock_feat
    mode='train'  # or 'valid', 'test'
)

# Training
observer.update_hidden_vec_reward(mode, rate_of_price_change, mkt_direction)
observer.train(**label_kwargs)  # Called at end of episode

# Reset
observer.reset()  # Called at start of new episode
```

## Output Format

- **market_vector**: (batch, N) numpy array - Market trend vector for N assets
- **lambda_val**: (batch,) numpy array - Dummy zeros (not used by MAFIA)
- **boundary_risk**: (batch,) numpy array - Continuous risk boundary value (ℝ^+)

## Training

MAFIA uses Policy Gradient training:
- Rewards are collected during episode via `update_hidden_vec_reward()`
- Training occurs at end of episode via `train()`
- Loss: `-mean(reward)` (maximize expected reward)

## Notes

- MAFIA requires raw OCHLV data with T_w=30 window
- The environment automatically extracts this data via `_extract_raw_ochlv_window()`
- `boundary_risk` is continuous (not discrete like MarketObserver)
- All modules support both CPU and GPU (CUDA)

## Complete Pipeline Flow

The MAFIA framework operates as part of the MASA (Multi-Agent System Architecture) pipeline, which consists of three main agents working in a loosely-coupled, pipelined manner at each timestep t.

### Detailed Step-by-Step Flow

#### Step 1: Environment Provides State
```python
# In StockPortfolioEnv.step() or reset()
o_t = [
    flattened_market_features,  # (N × L × F) → flattened
    log(C_t / C_0),             # Portfolio value ratio
    v_{m,t-1}                   # Previous market_vector (if available)
]
```

#### Step 2: MAFIA Observer Processing
```python
# In StockPortfolioEnv.run_mkt_observer()
raw_ochlv = env._extract_raw_ochlv_window(cur_date, T_w=30)  # (N, 5, 30)

# MAFIA Observer forward pass
v_m,t, _, σ_s,t = mafia_observer.predict(
    raw_ochlv_data=raw_ochlv,
    mode='train'  # or 'valid', 'test'
)

# Update state with market_vector
env.state = np.append(env.state, v_m,t[-1])
env.risk_adj_lst.append(σ_s,t)  # For solver
```

#### Step 3: RL-based Agent (TD3)
```python
# In TD3 Policy
state_with_market = np.concatenate([o_t, v_m,t])  # Combine states
a_t^RL = td3_actor.predict(state_with_market)     # (N,) portfolio weights
```

#### Step 4: Solver-based Agent (CBF)
```python
# In RL_withController()
# Get risk boundary from MAFIA
σ_s,t = env.risk_adj_lst[-1]

# Solve CBF optimization
a_t^Ctrl, is_solvable = cbf_opt(
    env=env,
    a_rl=a_t^RL,
    pred_dict={'shortterm': predicted_price_changes}
)

# Constraint: Risk(a_t^RL + a_t^Ctrl) ≤ σ_s,t
```

#### Step 5: Combine & Normalize
```python
# In RL_withController()
if is_solvable:
    a_t^Final = a_t^RL + a_t^Ctrl
else:
    a_t^Final = a_t^RL  # Fallback if solver fails

# Normalize to ensure sum = 1, all ≥ 0
a_t^Final = normalize_weights(a_t^Final)
```

#### Step 6: Environment Execution
```python
# In StockPortfolioEnv.step()
# Execute portfolio rebalancing
C_t = execute_portfolio(a_t^Final, prices_t)

# Compute rewards
J_r = (C_t - C_{t-1}) / C_{t-1}  # Return reward

# Update MAFIA Observer reward (for training)
if mode == 'train':
    rate_of_price_change = prices_t / prices_{t-1}
    mkt_observer.update_hidden_vec_reward(
        mode, rate_of_price_change, mkt_direction
    )
```

#### Step 7: Training (End of Episode)
```python
# RL-based Agent Training
# In TD3.learn() callback
for batch in replay_buffer:
    J_r = compute_return_reward(batch)
    J_JS = compute_js_divergence(
        a_t^RL=batch['action_rl'],
        a_t^Final=batch['action_final']
    )
    J_Total = λ₁·J_r + λ₂·J_JS
    update_actor_critic(J_Total)

# MAFIA Observer Training
# In StockPortfolioEnv.step() at terminal
if terminal and mode == 'train':
    mkt_observer.train(
        mode='train',
        ori_profit=original_profit_rates,
        adj_profit=adjusted_profit_rates,
        ori_risk=original_risks,
        adj_risk=adjusted_risks
    )
```

### Data Flow Diagram

```
┌──────────────┐
│ Environment │
│    o_t      │──┐
└──────────────┘  │
                  │
                  ▼
         ┌─────────────────┐
         │  MAFIA Observer │
         │                 │
         │  Input: o_t     │
         │  Output:        │
         │    • v_m,t ─────┼──► ┌──────────────┐
         │    • σ_s,t ─────┼──► │ RL Agent     │
         └─────────────────┘    │ (TD3)        │
                                 │              │
                                 │ Input:       │
                                 │   • o_t      │
                                 │   • v_m,t    │
                                 │              │
                                 │ Output:      │
                                 │   a_t^RL ────┼──► ┌──────────────┐
                                 └──────────────┘    │ Solver Agent │
                                                      │ (CBF)        │
                                                      │              │
                                                      │ Input:       │
                                                      │   • a_t^RL   │
                                                      │   • σ_s,t    │
                                                      │              │
                                                      │ Output:      │
                                                      │   a_t^Ctrl ──┼──► ┌──────────────┐
                                                      └──────────────┘    │   Combine    │
                                                                          │              │
                                                                          │ a_t^Final =  │
                                                                          │ a_t^RL +     │
                                                                          │ a_t^Ctrl     │
                                                                          │              │
                                                                          │      ────────┼──► ┌──────────────┐
                                                                          └──────────────┘    │ Environment  │
                                                                                              │ Execute &    │
                                                                                              │ Update       │
                                                                                              └──────────────┘
```

### Key Interactions

1. **MAFIA → RL Agent**: `v_m,t` provides market trend information to guide portfolio allocation
2. **MAFIA → Solver**: `σ_s,t` provides dynamic risk boundary for constraint optimization
3. **RL → Solver**: `a_t^RL` is the base portfolio that solver adjusts to meet risk constraints
4. **Solver → Environment**: `a_t^Ctrl` adjusts RL action to ensure risk compliance
5. **Environment → MAFIA**: Provides rewards for Policy Gradient training

### Training Schedule

- **RL Agent**: Updates every episode (or configured frequency)
  - Uses replay buffer with TD3 algorithm
  - Loss: `J_Total = λ₁·J_r + λ₂·J_JS`
  
- **MAFIA Observer**: Updates at end of each training episode
  - Uses Policy Gradient method
  - Loss: `-mean(reward)` (maximize expected reward)

## References

See `Đặc Tả Kỹ Thuật (Technical Specification) - MAFIA.md` for detailed technical specification.
See `Specification.md` for complete MASA framework specification.

