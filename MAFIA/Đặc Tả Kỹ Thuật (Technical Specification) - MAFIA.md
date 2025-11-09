# **Đặc Tả Kỹ Thuật (Technical Specification) \- MAFIA (Multi-Agent Fusion and Intelligent Adaptation)**

## **I. Tổng quan**

Tài liệu này mô tả đặc tả kỹ thuật để triển khai Market Observer tên là MAFIA để thay thế Market Observer của framework MASA. 

Kiến trúc cốt lõi bao gồm $1 \+ M\_a$ tác nhân:

1. **01 Tác nhân kỹ thuật (Technical agent):** Phân tích dựa trên giá thô (CHLV) và các chỉ báo kỹ thuật cổ điển.  
2. $M\_a$ **Tác nhân Directional Change (DC Agents):** Mỗi tác nhân phân tích dữ liệu "Directional Change" (DC) ở một mức độ chi tiết (ngưỡng $\\Delta x\_{dc}$) khác nhau.

Mỗi tác nhân (agent) bao gồm hai module chính:

* **Module Phân tích Cắt ngang (Cross-sectional Analysis \- CSA):** Để nắm bắt tương quan *không gian* (spatial) giữa các tài sản.  
* **Module Phân tích Thời gian (Temporal Analysis \- TA):** Để nắm bắt sự phụ thuộc *thời gian* (temporal) giữa các điểm dữ liệu.

Các đề xuất danh mục từ tất cả các tác nhân được tổng hợp (ensemble) trong **Bộ Tạo Danh mục (Portfolio Generator)** để đưa ra quyết định cuối cùng. Toàn bộ mô hình được huấn luyện bằng phương pháp Policy Gradient (Reinforcement Learning).

## **II. Dữ liệu và Tiền xử lý**

### **1\. Dữ liệu Đầu vào (Input Data)**

Dữ liệu đầu vào cho mô hình là một tensor chuỗi thời gian $P$.

* **Ký hiệu:** $P \\in \\mathbb{R}^{N \\times M \\times T\_w}$  
* $N$**:** Số lượng tài sản trong danh mục.  
  * **Cấu hình:** $N=10$ .  
* $M$**:** Số lượng đặc trưng cho mỗi tài sản  
* $T\_w$**:** Kích thước cửa sổ quan sát (observation period).  
  * **Đề xuất (`HYPERPARAM_T_w`):** $T\_w \= 30$ (ngày). Các paper về tối ưu danh mục thích ứng (adaptive portfolio) thường đề xuất 30-60 ngày. Việc chọn 30 ngày cho phép mô hình tập trung vào các xu hướng gần đây, phù hợp với tính chất "thích ứng" (self-adaptive) của MAFIA.

### **2\. Tiền xử lý:Phân tách Input cho các Agents**

Input $o\_t$ (dữ liệu thô OCHLV) sẽ được xử lý song song để tạo ra tensor input cho từng agent:

**A. Tác nhân Phân tích Kỹ thuật (Technical Agent,** $i=0$**):**

* **Mục tiêu:** Phân tích dựa trên giá thô và các chỉ báo kỹ thuật cổ điển.  
* **Tính toán:** Dựa trên $o\_t$, tính toán 3 chỉ báo kỹ thuật (Technical Indicators \- TIs):  
  * **Trend (Xu hướng):** `SMA(20)` (Simple Moving Average 20 ngày, tính trên `Close`).  
  * **Momentum (Động lượng):** `RSI(14)` (Relative Strength Index 14 ngày, tính trên `Close`).  
  * **Volatility (Biến động):** `ATR(14)` (Average True Range 14 ngày, tính trên `High, Low, Close`).  
* **Input Tensor:** $P\_{Tech} \\in \\mathbb{R}^{N \\times T\_w \\times M\_{Tech}}$  
  * $M\_{Tech} \= 8$  
  * **8 đặc trưng:** 5 (Open, Close, High, Low, Volume) \+ 3 (SMA, RSI, ATR).

**B. Tác nhân Directional Change (DC Agents,** $i=1..M\_a$**):**

* **Mục tiêu:** Phân tích dựa trên các "sự kiện" thay đổi xu hướng, lọc bỏ nhiễu.  
* **Số lượng (**$M\_a$**):** **3**. Mỗi tác nhân $i$ ($1 \\le i \\le M\_a$) sử dụng một ngưỡng $\\Delta x\_{dc,i}$ khác nhau.  
* **Ngưỡng DC (`HYPERPARAM_DC_THRESHOLDS`):** **`[0.005, 0.01, 0.02]`**. Bộ 3 ngưỡng này cho phép mô hình:  
  * `0.5%`: Nắm bắt các biến động nhỏ (noise, volatile stocks).  
  * `1.0%`: Nắm bắt các chuyển động thị trường tiêu chuẩn, cân bằng tín hiệu/nhiễu.  
  * `2.0%`: Chỉ tập trung vào các xu hướng lớn, quan trọng.  
* **Input Tensor:** $P\_{DC,i} \\in \\mathbb{R}^{N \\times T\_w \\times M\_{DC}}$  
  * $M\_{DC} \= 5$

**Quy trình tạo đặc trưng DC (cho mỗi tác nhân DC** $i$**):**

* **Input:** Chuỗi giá $p\_t$ (giá đóng cửa) cho mỗi tài sản.  
* **Logic:**  
  * **Sự kiện Upward DC:** Được xác nhận khi $p\_t \\ge p\_{t-1}^l \\times (1 \+ \\Delta x\_{dc,i})$, trong đó $p\_{t-1}^l$ là giá thấp nhất (Low) trong xu hướng giảm (downward trend) trước đó.  
  * **Sự kiện Downward DC:** Được xác nhận khi $p\_t \\le p\_{t-1}^h \\times (1 \- \\Delta x\_{dc,i})$, trong đó $p\_{t-1}^h$ là giá cao nhất (High) trong xu hướng tăng (upward trend) trước đó.  
* **Output:** Tạo ra một bản đồ đặc trưng (feature map) $P\_{DC,i} \\in \\mathbb{R}^{N \\times M \\times T\_w}$.  
  * **State (Trạng thái):** $+1$ (Up-trend), $-1$ (Down-trend).  
    * **Công thức:** `State[t] = current_trend_t`  
    * **Mô tả:** Ghi lại xu hướng hiện tại mà logic DC đang theo dõi. Trả về `+1` cho Xu hướng Tăng (Up-trend, bao gồm cả Upward DC và Upward OS) và `-1` cho Xu hướng Giảm (Down-trend).  
  * **Magnitude (Biên độ):** $\\%$ thay đổi so với điểm DC event gần nhất.  
    * **Công thức:** `Magnitude[t] = (Close[t] / event_price_t) - 1`  
    * **Mô tả:** Đo lường phần trăm thay đổi của giá đóng cửa hiện tại (`Close[t]`) so với giá tại *thời điểm DC event* gần nhất (`event_price_t`). Nó cho biết giá đã đi được bao xa (tăng hoặc giảm) trong xu hướng hiện tại.  
  * **Duration (Thời lượng):** Số ngày kể từ DC event gần nhất.  
    * **Công thức:** `Duration[t] = t - event_time_t`  
    * **Mô tả:** Đếm số bước thời gian (ngày) đã trôi qua kể từ DC event gần nhất. Nó cho biết xu hướng hiện tại đã kéo dài bao lâu.  
  * **Volume\_Ratio:**   
    * **Công thức:** `Volume_Ratio[t] = Volume[t] / Mean(Volume[0...L-1])`  
    * **Mô tả:** Tỷ lệ khối lượng giao dịch của ngày $t$ so với khối lượng giao dịch trung bình của *toàn bộ cửa sổ L ngày* đang quan sát. (Giá trị `Mean(Volume[0...L-1])` là một hằng số chuẩn hóa (normalizer) cho cả $L$ ngày).  
  * **Event\_Flag:**   
    * **Công thức:** `Event_Flag[t] = 1.0 if is_event_t else 0.5`  
    * **Mô tả:** Một cờ (flag) để nhấn mạnh tầm quan trọng của thời điểm $t$. Nó được gán giá trị `1.0` nếu $t$ là một *DC event* (sự kiện đảo chiều xu hướng) và `0.5` nếu $t$ là một *OS event* (Overshoot \- tiếp tục đi trong xu hướng).

## **III. Kiến trúc Mô hình**

Mỗi tác tử $i$ (cho dù là Technical agent hay DC agents) bao gồm 3 module:

1. Module Cross-sectional Analysis (CSA).  
2. Module Temporal Analysis (TA).  
3. Module Spatial-Temporal Fusion (ST-Fusion).

### **1\. Module Phân tích Cắt ngang (CSA)** 

**Mục tiêu:** Học tương quan giữa $N$ tài sản. 

1. **Token Generation (Reshape):**  
   * **Agent** $i=0$ **(Tech):** $\\varphi\_0^{CSA}(\\cdot): \\mathbb{R}^{N \\times T\_w \\times 8} \\rightarrow \\mathbb{R}^{N \\times (T\_w \\cdot 8)}$  
     * $D\_{token\\\_csa\\\_0} \= 30 \\times 8 \= 240$.  
   * **Agents** $i\>0$ **(DC):** $\\varphi\_i^{CSA}(\\cdot): \\mathbb{R}^{N \\times T\_w \\times 5} \\rightarrow \\mathbb{R}^{N \\times (T\_w \\cdot 5)}$  
     * $D\_{token\\\_csa\\\_i} \= 30 \\times 5 \= 150$.  
2. **Embedding (MLP):** *MLP không được chia sẻ (share) giữa các loại agent.*  
   * **Agent** $i=0$**:** $f\_0^{CSA}(\\cdot): \\mathbb{R}^{N \\times 240} \\rightarrow \\mathbb{R}^{N \\times D}$.  
     * Kiến trúc: `Linear(240, D_h)`, `GELU()`, `Linear(D_h, D)`.  
   * **Agents** $i\>0$**:** $f\_i^{CSA}(\\cdot): \\mathbb{R}^{N \\times 150} \\rightarrow \\mathbb{R}^{N \\times D}$. (MLP này được *chia sẻ* giữa 3 agent DC).  
     * Kiến trúc: `Linear(150, D_h)`, `GELU()`, `Linear(D_h, D)`.  
3. **Stacked Encoders:**  
   * Input: $\\mathbb{R}^{N \\times D}$.  
   * Cấu trúc: $L$ lớp Transformer Encoder tiêu chuẩn (Self-Attention \+ Feed-Forward Network).  
   * **Output CSA:** $O\_i^{CSA} \\in \\mathbb{R}^{N \\times D}$.

### **2\. Module Phân tích Thời gian (TA)**

**Mục tiêu:** Học tương quan giữa $T\_w$ điểm thời gian. **Input:** $P\_i \\in \\mathbb{R}^{N \\times M \\times T\_w}$.

1. **Token Generation (Reshape):**  
   * Hàm $\\rho\_i^{TA}(\\cdot)$:  
     * Bước 1: Hoán vị (permute) $P\_i$ từ $(N, M, T\_w)$ thành $(T\_w, N, M)$.  
     * Bước 2: Reshape $\\rightarrow \\mathbb{R}^{T\_w \\times (NM)}$.  
   * Mỗi điểm thời gian $t \\in T\_w$ trở thành một "token".  
   * Mỗi token có kích thước $D\_{token\\\_ta} \= N \\times M \= 10 \\times 3 \= 30$.  
   * Kết quả: $T\_w=30$ tokens.  
2. **Embedding (Shared MLP):**  
   * Hàm $f\_i^{TA}(\\cdot): \\mathbb{R}^{T\_w \\times 30} \\rightarrow \\mathbb{R}^{T\_w \\times D}$.  
   * Áp dụng một MLP chung (shared) cho $T\_w$ tokens.  
   * **Kiến trúc MLP:** Tương tự CSA.  
     * `Linear_1(input_dim=30, output_dim=HYPERPARAM_D_h)`  
     * `GELU()`  
     * `Linear_2(input_dim=HYPERPARAM_D_h, output_dim=D)`  
   * Kết quả: $P\_{i}^{TA'} \\in \\mathbb{R}^{T\_w \\times D}$.  
   * *MLP không được chia sẻ (share) giữa các loại agent.*  
     * **Agent** $i=0$**:** $f\_0^{TA}(\\cdot): \\mathbb{R}^{T\_w \\times 80} \\rightarrow \\mathbb{R}^{T\_w \\times D}$.  
       1. Kiến trúc: `Linear(80, D_h)`, `GELU()`, `Linear(D_h, D)`.  
     * **Agents** $i\>0$**:** $f\_i^{TA}(\\cdot): \\mathbb{R}^{T\_w \\times 50} \\rightarrow \\mathbb{R}^{T\_w \\times D}$. (MLP này được *chia sẻ* giữa 3 agent DC).  
       1. Kiến trúc: `Linear(50, D_h)`, `GELU()`, `Linear(D_h, D)`.  
3. **Tín hiệu Đầu vào cho Encoders (Signal Merging):** Input cho Encoders là tổng của 3 tín hiệu:  
   * **Tín hiệu 1 (Chính):** $P\_{i}^{TA'} \\in \\mathbb{R}^{T\_w \\times D}$ (từ bước 2).  
   * **Tín hiệu 2 (Time Seq Mask):**  
     * Tạo ra một tín hiệu positional encoding $PE\_{time} \\in \\mathbb{R}^{T\_w \\times D}$.  
     * "sử dụng hàm $sin(t')$, trong đó $t$ được ánh xạ tuyến tính tới $t' \\in \[0, \\pi/2\]$".  
     * Triển khai: `t_prime = linear_map(t, 0, T_w-1, 0, pi/2)`. $PE\_{time}$ là một embedding sinusoidal tiêu chuẩn (tương tự Transformer gốc) nhưng dựa trên `t_prime` (hoặc chỉ cần dùng `sin` và `cos` của `t_prime` nhân với một ma trận trọng số learnable).  
   * **Tín hiệu 3 (High-order DC Signals):**  
     * **Chỉ áp dụng cho các tác nhân DC (** $i \> 0$ **).**  
     * Yêu cầu một module "Event Detector" (Hình 1\) để "trích xuất các thay đổi của sự kiện DC".  
     * Triển khai **(Event Detector)**: Tính $\\Delta(\\text{State}\_t)$ (từ đặc trưng DC 1 \- State) và $\\Delta(\\text{Duration}\_t)$ (từ đặc trưng DC 3 \- Duration). Embed các tín hiệu thay đổi này về kích thước $\\mathbb{R}^{T\_w \\times D}$ và cộng vào.  
       1. Tính $Change\_{State} \= \\Delta(\\text{State}\_t) \= \\text{State}\_t \- \\text{State}\_{t-1}$. (Kết quả sẽ là 0, \+2, hoặc \-2).  
       2. Tính $Reset\_{Duration} \= \\Delta(\\text{Duration}\_t) \= \\text{Duration}\_t \- \\text{Duration}\_{t-1}$. (Kết quả sẽ là \+1, hoặc một số âm nếu có reset sự kiện).  
       3. Combine (ví dụ: concatenate) $Change\_{State}$ và $Reset\_{Duration}$ thành một tín hiệu sự kiện (event signal).  
       4. Embed tín hiệu sự kiện này (ví dụ: qua một lớp Linear) về kích thước $\\mathbb{R}^{T\_w \\times D}$ để tạo ra $H\_{DC,i}$.  
   * **Input cuối cùng cho Encoder:**  
     * Với tác nhân Technical ($i=0$): $Input\_{TA} \= P\_{0}^{TA'} \+ PE\_{time}$  
     * Với tác nhân DC ($i\>0$): $Input\_{TA} \= P\_{i}^{TA'} \+ PE\_{time} \+ H\_{DC,i}$ (với $H\_{DC,i}$ là embedding của high-order signals).  
4. **Stacked Encoders:**  
   * Input: $Input\_{TA} \\in \\mathbb{R}^{T\_w \\times D}$.  
   * Cấu trúc: $L$ lớp Transformer Encoder tiêu chuẩn (tương tự CSA).  
   * **Output TA:** $O\_i^{TA} \\in \\mathbb{R}^{T\_w \\times D}$.

Chú ý: Kiến trúc lõi (CSA/TA) được giữ nguyên, nhưng kích thước MLP Embedding sẽ khác nhau cho từng loại agent do $M\_{Tech} \\neq M\_{DC}$.

### **3\. Signal Generator**

**A. Spatial-Temporal Fusion (Cho mỗi tác nhân** $i$**)**

**Input:** $O\_i^{CSA} \\in \\mathbb{R}^{N \\times D}$ và $O\_i^{TA} \\in \\mathbb{R}^{T\_w \\times D}$. **Logic:** Sử dụng cơ chế attention, trong đó $O\_i^{CSA}$ là Query, và $O\_i^{TA}$ là Key và Value.

**Phương trình:** $O\_i \= V\_i(\\text{Softmax}(\\lambda O\_i^{CSA}(O\_i^{TA})^T) O\_i^{TA}) \+ b\_i$

1. **Tính Attention Scores:**  
   * $A \= O\_i^{CSA} @ (O\_i^{TA})^T$. Kích thước: $(N \\times D) @ (D \\times T\_w) \= (N \\times T\_w)$.  
2. **Scale & Softmax:**  
   * $\\lambda \= 1 / \\sqrt{D}$ (theo chuẩn Transformer, với $D$ là `HYPERPARAM_D`).  
   * $A' \= \\text{Softmax}(\\lambda \\cdot A, \\text{dim}=-1)$. Kích thước: $(N \\times T\_w)$.  
3. **Weighted Sum (Value):**  
   * $Fused \= A' @ O\_i^{TA}$. Kích thước: $(N \\times T\_w) @ (T\_w \\times D) \= (N \\times D)$.  
4. **MLP cuối:**  
   * $V\_i$ là một lớp Linear $(D \\rightarrow 1)$. $b\_i$ là bias.  
   * $O\_i \= \\text{Linear}(Fused, D, 1\) \+ b\_i$.  
   * **Output (Logits của tác nhân** $i$**):** $O\_i \\in \\mathbb{R}^{N \\times 1}$.

**B. Tạo Output 1: `market_vector` (**$v\_{m,t}$**)**

1. **Ensemble Logits:** Tổng hợp logits từ tất cả các agent. $O\_{final} \= \\sum\_{i=0}^{M\_a} O\_i$  
2. **Output:** $v\_{m,t} \= O\_{final} \\in \\mathbb{R}^{N \\times 1}$  
   * *Lý do:* Vector logits hợp nhất này đại diện cho đánh giá của MAFIA về xu hướng và độ hấp dẫn của $N$ tài sản, chính xác là thứ mà RL-based Agent cần (theo MASA Spec 4.3).

**C. Tạo Output 2: `boundary_risk` (**$\\sigma\_{s,t}$**)**

1. **Aggregate Temporal Embeddings:** Thu thập tất cả các embedding thời gian (chứa thông tin về biến động và thay đổi xu hướng) từ tất cả agent. $O\_{agg}^{TA} \= \\text{Average}(O\_0^{TA}, O\_1^{TA}, \\dots, O\_{M\_a}^{TA}) \\in \\mathbb{R}^{L \\times D}$  
2. **Flatten:** $O\_{flat}^{TA} \= \\text{Flatten}(O\_{agg}^{TA}) \\in \\mathbb{R}^{L \\cdot D}$  
3. **Risk Head (MLP dự đoán Rủi ro):**  
   * `Linear_1(input_dim = L \cdot D, output_dim = D)`  
   * `ReLU()`  
   * `Linear_2(input_dim = D, output_dim = 1)`  
   * `Softplus()` (Đảm bảo $\\sigma\_{s,t} \> 0$, vì rủi ro không thể âm).  
4. **Output:** $\\sigma\_{s,t} \= \\text{RiskHead}(O\_{flat}^{TA}) \\in \\mathbb{R}^+$

**IV. Huấn luyện (Training)**  
→ Giống MASA framework

## **V. Cấu hình Thử nghiệm (Theo Mục IV-A)**

* **Datasets:** VNSTOCK  
* **Số tài sản (**$N$**):** 10 (top 10 vốn hóa của mỗi chỉ số).  
* **Train:** 2015-01-01 đến 2021-12-31 (7 năm).  
* **Validate:** 2022-01-01 đến 2023-12-31 (2 năm).  
* **Test:** 2024-01-01 đến 2025-11-7 (2 năm).  
* **Số tác nhân DC (**$M\_a$**):** 3\.

## **VI. Các Siêu Tham Số (Hyperparameters) \- Đề Xuất Tối Ưu**

Bảng tóm tắt các đề xuất giá trị siêu tham số để triển khai:

* **`HYPERPARAM_M` (Số đặc trưng):**  
  * **`M_Tech` (Agent** $i=0$**): 8** (OCHLV \+ SMA(20), RSI(14), ATR(14)).  
  * **`M_DC` (Agents** $i\>0$**): 5** (State, Magnitude, Duration, Volume\_Ratio, Event\_Flag).  
* **`HYPERPARAM_T_w` (Cửa sổ quan sát):** **30** (ngày).  
* **`HYPERPARAM_DC_THRESHOLDS` (Ngưỡng DC):** **`[0.005, 0.01, 0.02]`** (0.5%, 1.0%, 2.0%).  
* **`HYPERPARAM_D` (Kích thước embedding):** **64**. Đây là giá trị hiệu quả (efficient) cho Transformer, đặc biệt khi $N=10$ và $T\_w=30$.  
* **`HYPERPARAM_D_h` (Lớp ẩn MLP):** **128**. (Gấp 2 lần $D$, một tỷ lệ phổ biến để học biểu diễn trung gian).  
* **`HYPERPARAM_ENCODER_LAYERS` (Số lớp Encoder):** **2**. (Với chuỗi ngắn $N=10$ và $T\_w=30$, $L=2$ là đủ để nắm bắt quan hệ, tránh overfitting).  
* **`HYPERPARAM_ENCODER_HEADS` (Số Attention Heads):** **4**. ($D=64$ chia hết cho $H=4$, mỗi head có $dim=16$, là một cấu hình chuẩn).  
* **`LEARNING_RATE` (Tốc độ học):** **1e-4**. (Một learning rate an toàn, tiêu chuẩn cho các mô hình RL/Transformer, sử dụng với Adam Optimizer).

