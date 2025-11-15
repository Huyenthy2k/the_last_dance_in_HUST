# MAFIA Training Flow - Visual Diagrams

## 📊 Overview: High-Level Architecture

```mermaid
graph TB
    A[Start Training] --> B[Epoch Loop<br/>100 iterations]
    B --> C[Episode Loop<br/>~1,260 trading days]
    C --> D[Timestep<br/>Collect Phase]
    D --> E[Training Phase]
    E --> F{Epoch<br/>Complete?}
    F -->|No| C
    F -->|Yes| G{All Epochs<br/>Done?}
    G -->|No| B
    G -->|Yes| H[End Training]
    
    style A fill:#e1f5e1
    style H fill:#ffe1e1
    style D fill:#e1e5ff
    style E fill:#ffe5e1
```

---

## 🔄 Single Timestep Detailed Flow

```mermaid
sequenceDiagram
    participant Env as Environment<br/>(Trading Env)
    participant MAFIA as MAFIA Observer
    participant RL as TD3 Actor
    participant CBF as CBF Controller
    participant Buffer as Replay Buffer
    
    Note over Env,Buffer: Timestep t (e.g., Day 100)
    
    Env->>MAFIA: OCHLV window<br/>(N, 5, 30)
    Note right of MAFIA: 30-day history<br/>[t-29...t]
    
    MAFIA->>MAFIA: Technical Agent<br/>Features (N,30,8)
    MAFIA->>MAFIA: 3× DC Agents<br/>Features (N,30,5)
    MAFIA->>MAFIA: Market-Index Agent<br/>Features (1,30,19)
    MAFIA->>MAFIA: Dense MoE Gating<br/>Combine agents
    MAFIA->>MAFIA: Scoring Head<br/>Top-K selection
    
    MAFIA-->>Env: market_vector (N,)<br/>market_scores_full (N,)<br/>boundary_risk
    
    Env->>RL: State<br/>[portfolio, market_vector]
    RL->>RL: Actor Network<br/>[1024,512,128]→Softmax
    RL-->>CBF: a_rl (N,)
    
    CBF->>CBF: Compute Covariance<br/>Solve SOCP
    CBF-->>Env: a_final = a_rl + a_cbf
    
    Env->>Env: Execute Action<br/>Rebalance Portfolio
    Env-->>Buffer: Store Transition<br/>(s, a, r, s', done)
    
    Note over MAFIA,Buffer: Transition stored with gradient!
```

---

## 🎓 Training Phase - After Episode

```mermaid
flowchart TD
    A[Episode Complete<br/>1,260 transitions collected] --> B{Buffer Size<br/> ≥ 100?}
    B -->|No| Z[Skip Training<br/>Continue Collecting]
    B -->|Yes| C[Start Training Phase]
    
    C --> D[TD3 Training Loop<br/>gradient_steps = 1,260]
    C --> E[MAFIA Training<br/>1 Policy Gradient Update]
    
    D --> D1[Sample Batch 32<br/>from Buffer]
    D1 --> D2[Update Critic<br/>MSE Loss]
    D2 --> D3{Step % 2 == 0?}
    D3 -->|Yes| D4[Update Actor<br/>Policy Gradient<br/>Soft Update Targets]
    D3 -->|No| D5[Skip Actor Update]
    D4 --> D6{More<br/>Steps?}
    D5 --> D6
    D6 -->|Yes| D1
    D6 -->|No| F[TD3 Complete<br/>1,260 Critic updates<br/>630 Actor updates]
    
    E --> E1[Compute Rewards<br/>from market_vectors]
    E1 --> E2[Policy Gradient Loss<br/>L = -mean(rewards)]
    E2 --> E3[Backprop through<br/>MAFIA model]
    E3 --> E4[Update All<br/>MAFIA Weights]
    E4 --> G[MAFIA Complete<br/>1 update]
    
    F --> H[Evaluation & Checkpoint]
    G --> H
    H --> I[Next Episode]
    
    style A fill:#e1f5e1
    style F fill:#ffe5e1
    style G fill:#ffe5e1
    style H fill:#e1e5ff
```

---

## 📊 Weight Update Timeline (1 Epoch)

```mermaid
gantt
    title Weight Update Schedule (1 Epoch = 1,260 Trading Days)
    dateFormat X
    axisFormat %s
    
    section Data Collection
    Collect 1,260 transitions :active, 0, 1260
    
    section TD3 Critic
    Update 1,260 times :crit, 1260, 1261
    
    section TD3 Actor
    Update 630 times :crit, 1260, 1261
    
    section MAFIA
    Update 1 time :milestone, 1260, 1261
    
    section Target Networks
    Soft update 630 times :1260, 1261
```

---

## 🔢 Replay Buffer Evolution

```mermaid
graph LR
    A[Day 1<br/>Size: 1] --> B[Day 2<br/>Size: 2]
    B --> C[Day 3<br/>Size: 3]
    C --> D[.....]
    D --> E[Day 100<br/>Size: 100<br/>🎓 Learning Starts!]
    E --> F[.....]
    F --> G[Day 1,260<br/>Size: 1,260<br/>📚 Episode End]
    G --> H[Train Phase<br/>Sample 32×1,260<br/>= 40,320 samples]
    H --> I[Next Episode<br/>Size: still 1,260<br/>Add new transitions]
    
    style E fill:#e1f5e1
    style G fill:#ffe5e1
    style H fill:#e1e5ff
```

---

## 🎯 MAFIA Observer Architecture

```mermaid
graph TB
    Input[OCHLV Data<br/>N, 5, 30] --> Tech[Technical Agent<br/>ΔO,ΔC,ΔH,ΔL,ΔV<br/>SMA, RSI, ATR]
    Input --> DC1[DC Agent θ=0.5%<br/>State, Magnitude<br/>Duration, Volume, Flag]
    Input --> DC2[DC Agent θ=1%<br/>State, Magnitude<br/>Duration, Volume, Flag]
    Input --> DC3[DC Agent θ=2%<br/>State, Magnitude<br/>Duration, Volume, Flag]
    
    MktInput[Market Index OCHLV<br/>1, 5, 30] --> Mkt[Market-Index Agent<br/>ΔOCHLV + 3 basic<br/>+ 11 extended indicators]
    
    Tech --> CSA1[CSA Module]
    Tech --> TA1[TA Module]
    CSA1 --> Fusion1[ST-Fusion]
    TA1 --> Fusion1
    Fusion1 --> Embed1[Embedding<br/>N, D=64]
    
    DC1 --> CSA2[CSA Module]
    DC1 --> TA2[TA Module]
    CSA2 --> Fusion2[ST-Fusion]
    TA2 --> Fusion2
    Fusion2 --> Embed2[Embedding<br/>N, D=64]
    
    DC2 --> CSA3[CSA Module]
    DC2 --> TA3[TA Module]
    CSA3 --> Fusion3[ST-Fusion]
    TA3 --> Fusion3
    Fusion3 --> Embed3[Embedding<br/>N, D=64]
    
    DC3 --> CSA4[CSA Module]
    DC3 --> TA4[TA Module]
    CSA4 --> Fusion4[ST-Fusion]
    TA4 --> Fusion4
    Fusion4 --> Embed4[Embedding<br/>N, D=64]
    
    Mkt --> CSAM[CSA Module]
    Mkt --> TAM[TA Module]
    CSAM --> FusionM[ST-Fusion]
    TAM --> FusionM
    FusionM --> EmbedM[Embedding<br/>1, D=64]
    
    Embed1 --> Gate[Dense MoE Gating<br/>Learn weights per stock]
    Embed2 --> Gate
    Embed3 --> Gate
    Embed4 --> Gate
    EmbedM -.Optional.-> Gate
    
    Gate --> Combined[Combined Embedding<br/>N, D=64]
    
    Combined --> Score[Scoring Head<br/>MLP → Softmax]
    Combined --> Risk[Risk Head<br/>MLP → Scalar]
    
    Score --> TopK[Gumbel Top-K]
    TopK --> MV[market_vector<br/>N, sparse K]
    TopK --> MS[market_scores_full<br/>N, dense]
    Risk --> BR[boundary_risk<br/>scalar]
    
    style Input fill:#e1f5e1
    style MktInput fill:#e1f5e1
    style Gate fill:#ffe5e1
    style TopK fill:#e1e5ff
```

---

## 🧠 TD3 Network Update Flow

```mermaid
graph TB
    RB[Replay Buffer<br/>~1,260 transitions] --> Sample[Random Sample<br/>Batch Size = 32]
    
    Sample --> S[States s]
    Sample --> A[Actions a]
    Sample --> R[Rewards r]
    Sample --> SN[Next States s']
    Sample --> D[Dones]
    
    SN --> ActorT[Actor Target<br/>+ noise]
    ActorT --> Q1T[Critic1 Target]
    ActorT --> Q2T[Critic2 Target]
    Q1T --> MinQ[Min Q-value]
    Q2T --> MinQ
    MinQ --> Target[Q_target = r + γ×min Q']
    
    S --> Q1[Critic1]
    A --> Q1
    S --> Q2[Critic2]
    A --> Q2
    
    Q1 --> Loss1[MSE Loss1]
    Q2 --> Loss2[MSE Loss2]
    Target --> Loss1
    Target --> Loss2
    
    Loss1 --> UpdateC[Update Critics<br/>Backprop]
    Loss2 --> UpdateC
    
    UpdateC --> Check{Step % 2<br/>== 0?}
    
    Check -->|Yes| S2[States s]
    S2 --> Actor[Actor Network]
    Actor --> Q1F[Critic1 Forward]
    Q1F --> LossA[Actor Loss<br/>-mean Q1]
    LossA --> UpdateA[Update Actor<br/>Backprop]
    UpdateA --> Soft[Soft Update<br/>Targets<br/>τ=0.005]
    
    Check -->|No| Skip[Skip Actor Update]
    
    style RB fill:#e1f5e1
    style UpdateC fill:#ffe5e1
    style UpdateA fill:#ffe5e1
    style Soft fill:#e1e5ff
```

---

## 📈 Learning Curve (Conceptual)

```mermaid
graph LR
    A[Epoch 0<br/>Random Policy<br/>Low Return] --> B[Epoch 10<br/>Exploration<br/>Improving]
    B --> C[Epoch 30<br/>Exploitation<br/>Better Returns]
    C --> D[Epoch 50<br/>Refinement<br/>Stable Policy]
    D --> E[Epoch 100<br/>Converged<br/>Optimal Policy]
    
    style A fill:#ffe1e1
    style B fill:#ffe5e1
    style C fill:#fff5e1
    style D fill:#e5ffe1
    style E fill:#e1f5e1
```

---

## 🎯 Key Concepts Summary

```mermaid
mindmap
  root((MAFIA<br/>Training))
    Timestep
      1 Trading Day = 1 Timestep
      Window T_w = 30 days history
      Sliding window
    Episode
      1 Episode = ~1,260 timesteps
      1 Epoch = 1 Episode
      Collect all transitions
    Batch
      Batch Size = 32
      Random sample from buffer
      Break correlation
    Buffer
      Max Size = 1,000,000
      FIFO when full
      Off-policy learning
    Updates
      TD3 Critic: 1,260/epoch
      TD3 Actor: 630/epoch
      MAFIA: 1/epoch
      Different time scales
    Networks
      MAFIA: 4 agents + gating
      TD3: Actor + 2 Critics
      Target networks
```

---

## 🔍 Detailed Dimensions Flow

```mermaid
flowchart LR
    A[OCHLV<br/>N=10, M=5, T=30] --> B[Tech Agent<br/>10, 30, 8]
    A --> C[DC Agent 1<br/>10, 30, 5]
    A --> D[DC Agent 2<br/>10, 30, 5]
    A --> E[DC Agent 3<br/>10, 30, 5]
    
    B --> F[CSA+TA+Fusion<br/>10, 64]
    C --> G[CSA+TA+Fusion<br/>10, 64]
    D --> H[CSA+TA+Fusion<br/>10, 64]
    E --> I[CSA+TA+Fusion<br/>10, 64]
    
    F --> J[MoE Gating<br/>10, 64]
    G --> J
    H --> J
    I --> J
    
    J --> K[Scoring Head<br/>10, 1]
    K --> L[Softmax<br/>10]
    L --> M[Gumbel Top-K]
    M --> N[market_vector<br/>10, sparse]
    M --> O[market_scores_full<br/>10, dense]
    
    J --> P[Risk Head<br/>1]
    P --> Q[boundary_risk<br/>scalar]
    
    N --> R[RL State<br/>state_dim]
    R --> S[Actor<br/>1024→512→128→10]
    S --> T[a_rl<br/>10]
    T --> U[CBF<br/>SOCP]
    U --> V[a_final<br/>10]
```

---

## 📊 Training Stats Example

After 1 Epoch (hypothetical):

| Component | Updates | Time | Memory |
|-----------|---------|------|--------|
| **Data Collection** | 1,260 timesteps | ~5 min | Buffer: 10 MB |
| **TD3 Critic** | 1,260 updates | ~3 min | Batch: 32 samples |
| **TD3 Actor** | 630 updates | ~1.5 min | Batch: 32 samples |
| **MAFIA Observer** | 1 update | ~0.5 min | Episode: 1,260 samples |
| **Evaluation** | Valid + Test | ~2 min | Full episodes |
| **Total** | - | ~12 min | ~20 MB |

**Scalability for 100 Epochs:**
- Total time: ~20 hours
- Total TD3 Critic updates: 126,000
- Total TD3 Actor updates: 63,000
- Total MAFIA updates: 100
- Buffer max size: ~100 MB (at full capacity)

---

## 🎓 Learning Dynamics

```mermaid
graph TB
    subgraph "Early Training (Epoch 1-20)"
        A1[High Exploration] --> A2[Random Actions]
        A2 --> A3[Low Returns]
        A3 --> A4[Fill Buffer]
        A4 --> A5[Learn Basic Patterns]
    end
    
    subgraph "Mid Training (Epoch 21-60)"
        B1[Balanced Exploration/Exploitation] --> B2[Better Action Selection]
        B2 --> B3[Improving Returns]
        B3 --> B4[Rich Buffer]
        B4 --> B5[Learn Market Dynamics]
    end
    
    subgraph "Late Training (Epoch 61-100)"
        C1[Exploitation Dominant] --> C2[Refined Actions]
        C2 --> C3[Stable Returns]
        C3 --> C4[Mature Policy]
        C4 --> C5[Convergence]
    end
    
    A5 --> B1
    B5 --> C1
    
    style A3 fill:#ffe1e1
    style B3 fill:#fff5e1
    style C3 fill:#e1f5e1
```

---

## 💡 Quick Reference

### Time Concepts
- **1 Epoch** = 1 Episode = ~1,260 Trading Days
- **1 Timestep** = 1 Trading Day
- **100 Epochs** = 100 Episodes = ~126,000 Timesteps

### Space Concepts
- **Window Size (T_w)** = 30 days of history
- **Batch Size** = 32 samples per gradient update
- **Buffer Size** = 1,000,000 max transitions

### Update Frequencies (per Epoch)
- **TD3 Critic**: 1,260 updates
- **TD3 Actor**: 630 updates (every 2 steps)
- **MAFIA**: 1 update (end of episode)

### Key Dimensions
- **N** = 10 (number of stocks)
- **M** = 5 (OCHLV features)
- **T_w** = 30 (window size)
- **D** = 64 (embedding dimension)
- **K** = 10 (Top-K selection)

---

**Note**: All diagrams can be rendered in any Markdown viewer that supports Mermaid (e.g., GitHub, VS Code with extensions, Obsidian, etc.)

---

**Để view diagrams này:**
1. Open trong GitHub (auto-render Mermaid)
2. Dùng VS Code với extension "Markdown Preview Mermaid Support"
3. Copy vào Obsidian
4. Dùng online tool: https://mermaid.live/

