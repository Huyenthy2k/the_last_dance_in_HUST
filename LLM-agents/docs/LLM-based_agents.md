# Multi-Agent Adversarial Debate System for Qualitative Stock Analysis

## Abstract

Tài liệu này trình bày kiến trúc của hệ thống phân tích định tính cổ phiếu dựa trên mô hình Multi-Agent Adversarial Debate. Hệ thống được thiết kế để bổ sung khả năng đánh giá định tính cho mô hình định lượng MAFIA, nhằm cung cấp cái nhìn toàn diện về tiềm năng đầu tư của doanh nghiệp.

---

## Mục Lục

**Phần I: Tổng Quan Hệ Thống**
1. [Giới Thiệu và Bối Cảnh](#1-giới-thiệu-và-bối-cảnh)
2. [Kiến Trúc Multi-Agent System](#2-kiến-trúc-multi-agent-system)
3. [Luồng Hoạt Động của Hệ Thống](#3-luồng-hoạt-động-của-hệ-thống)

**Phần II: Chi Tiết Các Thành Phần**
4. [Data Layer - Tầng Thu Thập Dữ Liệu](#4-data-layer---tầng-thu-thập-dữ-liệu)
5. [Tool Layer - Tầng Trừu Tượng Hóa Công Cụ](#5-tool-layer---tầng-trừu-tượng-hóa-công-cụ)
6. [Analysis Layer - Tầng Phân Tích](#6-analysis-layer---tầng-phân-tích)
7. [Debate Layer - Tầng Tranh Luận Đối Kháng](#7-debate-layer---tầng-tranh-luận-đối-kháng)
8. [Decision Layer - Tầng Ra Quyết Định](#8-decision-layer---tầng-ra-quyết-định)
9. [Memory System - Hệ Thống Bộ Nhớ](#9-memory-system---hệ-thống-bộ-nhớ)
10. [Orchestration Layer - Tầng Điều Phối](#10-orchestration-layer---tầng-điều-phối)

**Phần III: Tích Hợp và Mở Rộng**
11. [Tích Hợp với MAFIA Model](#11-tích-hợp-với-mafia-model)
12. [Kết Luận](#12-kết-luận)

---

# PHẦN I: TỔNG QUAN HỆ THỐNG

---

## 1. Giới Thiệu và Bối Cảnh

### 1.1 Vấn Đề Cần Giải Quyết

Trong phân tích đầu tư, tồn tại hai cách tiếp cận bổ sung cho nhau: **phân tích định lượng** xử lý dữ liệu số học để xếp hạng cổ phiếu, và **phân tích định tính** đánh giá các yếu tố không thể số hóa như tin tức, tâm lý thị trường, và bối cảnh kinh doanh.

Mô hình MAFIA đã giải quyết phần định lượng - trả ra top K cổ phiếu tiềm năng. Tuy nhiên, ranking thuần số học bỏ qua nhiều yếu tố quan trọng: một công ty có thể đang đối mặt với scandal, một ngành có thể sắp bị regulated, một CEO có thể vừa từ chức. Những thông tin này ảnh hưởng lớn đến giá cổ phiếu nhưng khó capture bằng công thức.

**Hệ thống LLM-Agents được thiết kế để lấp đầy khoảng trống này** - thực hiện phân tích định tính tự động cho từng cổ phiếu trong top K, cung cấp đánh giá toàn diện trước khi đưa ra quyết định đầu tư cuối cùng.

### 1.2 Tại Sao Cần Multi-Agent thay vì Single LLM?

Cách tiếp cận đơn giản nhất là dùng một LLM duy nhất: đưa vào dữ liệu cổ phiếu, yêu cầu phân tích và đưa ra quyết định. Tuy nhiên, cách này có ba hạn chế nghiêm trọng:

**Thiên kiến xác nhận (Confirmation Bias):** Khi LLM hình thành ấn tượng đầu tiên (ví dụ: "công ty này có vẻ tốt"), nó có xu hướng tìm kiếm bằng chứng ủng hộ ấn tượng đó và bỏ qua tín hiệu ngược chiều. Đây là hành vi tự nhiên của LLM do cách training - model được tối ưu để đưa ra câu trả lời consistent, không phải để tự thách thức bản thân.

**Phân tích một chiều:** Không có cơ chế buộc LLM xem xét góc nhìn đối lập. Nếu ban đầu model nghiêng về "mua", nó sẽ viết báo cáo thiên về bullish mà không đào sâu vào rủi ro. Ngược lại nếu nghiêng về "bán", các cơ hội tiềm năng bị bỏ qua.

**Tự tin quá mức:** LLM thường đưa ra kết luận với độ chắc chắn cao mà không nhận ra điểm yếu trong lập luận của mình. Trong đầu tư, sự tự tin thiếu căn cứ này có thể dẫn đến quyết định thiếu cân nhắc rủi ro.

**Giải pháp Multi-Agent:** Thay vì dựa vào một LLM tự kiểm soát bias, chúng tôi thiết kế hệ thống với nhiều agent có vai trò đối kháng có chủ đích. Một agent được giao nhiệm vụ tìm mọi lý do để mua, agent khác tìm mọi lý do để không mua. Sự đối kháng cấu trúc này buộc hệ thống phải xem xét cả hai mặt của vấn đề.

### 1.3 Nguyên Lý Adversarial Debate

Mô hình tranh luận đối kháng lấy cảm hứng từ hệ thống tư pháp: trong phiên tòa, công tố viên tìm mọi bằng chứng buộc tội, luật sư bào chữa tìm mọi bằng chứng gỡ tội, thẩm phán đánh giá lập luận hai bên và đưa ra phán quyết. Không ai được yêu cầu trung lập - chính sự đối kháng có chủ đích tạo ra cái nhìn cân bằng.

Trong hệ thống của chúng tôi:
- **Bull Researcher** = Công tố viên của "mua cổ phiếu" - tìm mọi lý do ủng hộ đầu tư
- **Bear Researcher** = Luật sư bào chữa - tìm mọi rủi ro và điểm yếu
- **Research Manager** = Thẩm phán - đánh giá chất lượng lập luận và đưa ra quyết định

Điểm quan trọng: các researcher không được yêu cầu objective. Họ được khuyến khích **advocate mạnh mẽ** cho lập trường của mình. Tính objectivity nổi lên từ sự đối kháng, không phải từ việc yêu cầu mỗi agent tự objective.

---

## 2. Kiến Trúc Multi-Agent System

### 2.1 Tổng Quan Kiến Trúc

Hệ thống được tổ chức thành **sáu tầng logic**, mỗi tầng có trách nhiệm rõ ràng và tương tác với các tầng khác theo nguyên tắc một chiều (tầng trên gọi tầng dưới, không ngược lại):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         ORCHESTRATION LAYER                                 │
│                                                                             │
│    Vai trò: Điều phối toàn bộ workflow, quản lý vòng đời của phiên          │
│    phân tích, quyết định agent nào chạy tiếp theo                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          DECISION LAYER                                     │
│                                                                             │
│    Vai trò: Tổng hợp kết quả tranh luận, cân nhắc các lập luận,             │
│    đưa ra quyết định đầu tư cuối cùng (BUY/HOLD/SELL)                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           DEBATE LAYER                                      │
│                                                                             │
│    Vai trò: Thực hiện tranh luận đối kháng giữa Bull và Bear,               │
│    mỗi bên đưa ra lập luận và phản biện lập luận đối phương                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          ANALYSIS LAYER                                     │
│                                                                             │
│    Vai trò: Xử lý dữ liệu thô thành báo cáo phân tích có cấu trúc,          │
│    mỗi analyst chuyên về một loại dữ liệu (kỹ thuật, cơ bản, tin tức)       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            TOOL LAYER                                       │
│                                                                             │
│    Vai trò: Cung cấp interface chuẩn hóa để LLM có thể "gọi" các            │
│    công cụ thu thập dữ liệu, LLM tự quyết định cần công cụ nào              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            DATA LAYER                                       │
│                                                                             │
│    Vai trò: Kết nối với nguồn dữ liệu bên ngoài (vnstock API),              │
│    thu thập và chuẩn hóa dữ liệu thô                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Các Agent trong Hệ Thống

Hệ thống bao gồm **6 agent chính**, mỗi agent có vai trò và trách nhiệm riêng biệt:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                              AGENTS ECOSYSTEM                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ANALYSIS AGENTS                               │   │
│  │                                                                      │   │
│  │   ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │   │
│  │   │   MARKET     │   │    NEWS      │   │    FUNDAMENTALS      │   │   │
│  │   │   ANALYST    │   │   ANALYST    │   │      ANALYST         │   │   │
│  │   │              │   │              │   │                      │   │   │
│  │   │ Phân tích    │   │ Phân tích    │   │ Phân tích báo cáo    │   │   │
│  │   │ kỹ thuật:    │   │ tin tức:     │   │ tài chính:           │   │   │
│  │   │ giá, volume, │   │ sentiment,   │   │ P/E, ROE, nợ,        │   │   │
│  │   │ indicators   │   │ events       │   │ doanh thu, lợi nhuận │   │   │
│  │   └──────────────┘   └──────────────┘   └──────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      │ Báo cáo phân tích                    │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DEBATE AGENTS                                 │   │
│  │                                                                      │   │
│  │   ┌──────────────────────┐       ┌──────────────────────┐          │   │
│  │   │    BULL RESEARCHER   │◄─────►│   BEAR RESEARCHER    │          │   │
│  │   │                      │       │                      │          │   │
│  │   │ Advocate cho MUA:    │ Tranh │ Advocate cho KHÔNG   │          │   │
│  │   │ tìm điểm mạnh,       │ luận  │ MUA: tìm rủi ro,     │          │   │
│  │   │ cơ hội tăng trưởng,  │ đối   │ điểm yếu, threats    │          │   │
│  │   │ lợi thế cạnh tranh   │ kháng │ từ thị trường        │          │   │
│  │   └──────────────────────┘       └──────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      │ Lịch sử tranh luận                   │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       DECISION AGENT                                 │   │
│  │                                                                      │   │
│  │   ┌────────────────────────────────────────────────────────────┐   │   │
│  │   │                   RESEARCH MANAGER                          │   │   │
│  │   │                                                             │   │   │
│  │   │ Trọng tài của cuộc tranh luận:                              │   │   │
│  │   │ - Đánh giá chất lượng lập luận mỗi bên                      │   │   │
│  │   │ - Cân nhắc evidence và counter-arguments                    │   │   │
│  │   │ - Đưa ra quyết định BUY/HOLD/SELL                           │   │   │
│  │   │ - Tạo kế hoạch đầu tư chi tiết                              │   │   │
│  │   └────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Cơ Chế Giao Tiếp: Shared State Pattern

Các agent trong hệ thống **không gọi trực tiếp lẫn nhau**. Thay vào đó, chúng giao tiếp thông qua một **Shared State** - một object trung tâm chứa toàn bộ thông tin của phiên phân tích.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                              SHARED STATE                                   │
│                    (Trung tâm giao tiếp của hệ thống)                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CONTEXT                                                            │   │
│  │  ├── company_of_interest: "FPT"     ← Mã cổ phiếu cần phân tích    │   │
│  │  └── trade_date: "2025-01-15"       ← Ngày giao dịch               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ANALYSIS REPORTS (được điền bởi Analysts)                          │   │
│  │  ├── market_report: "..."           ← Báo cáo phân tích kỹ thuật   │   │
│  │  ├── fundamentals_report: "..."     ← Báo cáo phân tích cơ bản     │   │
│  │  └── news_report: "..."             ← Báo cáo phân tích tin tức    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DEBATE STATE (được cập nhật bởi Researchers)                       │   │
│  │  ├── bull_history: "..."            ← Lịch sử lập luận Bull        │   │
│  │  ├── bear_history: "..."            ← Lịch sử lập luận Bear        │   │
│  │  ├── current_response: "..."        ← Lập luận mới nhất            │   │
│  │  └── count: 2                       ← Số lượt tranh luận           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT (được điền bởi Research Manager)                            │   │
│  │  ├── judge_decision: "BUY"          ← Quyết định cuối cùng         │   │
│  │  └── investment_plan: "..."         ← Kế hoạch đầu tư chi tiết     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Tại sao chọn Shared State thay vì Direct Communication?**

Trong mô hình direct communication, agent A gọi trực tiếp agent B, tạo ra sự phụ thuộc chặt chẽ. Nếu muốn thêm agent C, cần sửa đổi A và B.

Với Shared State, mỗi agent chỉ cần biết:
- Đọc thông tin gì từ state
- Ghi thông tin gì vào state

Agent không cần biết agent nào khác tồn tại. Điều này cho phép thêm/bớt agent dễ dàng mà không ảnh hưởng đến agents hiện có.

---

## 3. Luồng Hoạt Động của Hệ Thống

### 3.1 Tổng Quan Workflow

Một phiên phân tích đi qua **ba pha chính**: Thu Thập Dữ Liệu → Tranh Luận Đối Kháng → Ra Quyết Định.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    INPUT: Mã cổ phiếu (ví dụ: "FPT") + Ngày giao dịch (2025-01-15)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    PHA 1: THU THẬP VÀ PHÂN TÍCH DỮ LIỆU                     │
│                                                                             │
│    Ba Analyst agents hoạt động, mỗi agent:                                  │
│    1. Nhận context từ State (ticker, date)                                  │
│    2. Quyết định cần thu thập dữ liệu gì (tool calls)                       │
│    3. Xử lý dữ liệu thô thành báo cáo có cấu trúc                           │
│    4. Ghi báo cáo vào State                                                 │
│                                                                             │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐           │
│    │   Market    │      │    News     │      │  Fundamentals   │           │
│    │   Analyst   │      │   Analyst   │      │    Analyst      │           │
│    └──────┬──────┘      └──────┬──────┘      └────────┬────────┘           │
│           │                    │                      │                     │
│           ▼                    ▼                      ▼                     │
│    market_report         news_report          fundamentals_report          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                      PHA 2: TRANH LUẬN ĐỐI KHÁNG                            │
│                                                                             │
│    Bull và Bear Researchers thay phiên tranh luận:                          │
│                                                                             │
│    Vòng 1:                                                                  │
│    ┌──────────────────┐                                                     │
│    │ BULL: "Tôi cho   │ ─────► Đọc reports, query memory                   │
│    │ rằng FPT là cơ   │        Đưa ra lập luận ủng hộ đầu tư               │
│    │ hội tốt vì..."   │                                                     │
│    └──────────────────┘                                                     │
│             │                                                               │
│             ▼                                                               │
│    ┌──────────────────┐                                                     │
│    │ BEAR: "Tôi phản  │ ─────► Đọc reports, query memory                   │
│    │ đối vì những     │        Phản biện Bull + đưa ra rủi ro              │
│    │ rủi ro sau..."   │                                                     │
│    └──────────────────┘                                                     │
│             │                                                               │
│             ▼                                                               │
│    [Lặp lại nếu chưa đủ số vòng tranh luận]                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                        PHA 3: RA QUYẾT ĐỊNH                                 │
│                                                                             │
│    Research Manager đọc toàn bộ lịch sử tranh luận:                         │
│                                                                             │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │                      RESEARCH MANAGER                               │  │
│    │                                                                     │  │
│    │  1. Đọc: market_report, fundamentals_report, news_report           │  │
│    │  2. Đọc: debate_history (toàn bộ tranh luận Bull-Bear)             │  │
│    │  3. Query: invest_judge_memory (bài học từ quyết định quá khứ)     │  │
│    │  4. Đánh giá: lập luận nào có data support tốt hơn?                │  │
│    │  5. Quyết định: BUY / HOLD / SELL                                  │  │
│    │  6. Output: investment_plan với rationale chi tiết                 │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    OUTPUT:                                                                  │
│    ├── Quyết định: BUY / HOLD / SELL                                       │
│    ├── Investment Plan: lý do, actions đề xuất, risk management            │
│    └── Full State: toàn bộ báo cáo và lịch sử tranh luận (để debug/audit)  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Chi Tiết Luồng Tool-Calling trong Pha 1

Analyst agents không thu thập dữ liệu theo script cứng. Thay vào đó, chúng được trang bị một "catalog" các công cụ và **tự quyết định** công cụ nào cần thiết cho từng tình huống.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    TOOL-CALLING LOOP (ví dụ: Market Analyst)                │
│                                                                             │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  Bước 1: ANALYST NHẬN CONTEXT                                      │  │
│    │                                                                     │  │
│    │  "Tôi cần phân tích kỹ thuật cho FPT ngày 2025-01-15.              │  │
│    │   Tôi có các tools: get_stock_data, get_indicators.                │  │
│    │   Tôi sẽ cần dữ liệu giá và một số chỉ báo..."                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  Bước 2: LLM QUYẾT ĐỊNH GỌI TOOL                                   │  │
│    │                                                                     │  │
│    │  LLM output: "Tôi cần gọi get_stock_data(FPT, 2024-07-15,          │  │
│    │              2025-01-15) để lấy 6 tháng dữ liệu giá"               │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  Bước 3: TOOL ĐƯỢC THỰC THI                                        │  │
│    │                                                                     │  │
│    │  System thực thi get_stock_data(), trả về dữ liệu OHLCV            │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  Bước 4: LLM NHẬN KẾT QUẢ, QUYẾT ĐỊNH TIẾP                         │  │
│    │                                                                     │  │
│    │  "Đã có dữ liệu giá. Bây giờ tôi cần RSI và MACD để đánh giá       │  │
│    │   momentum. Gọi get_indicators(FPT, rsi) và get_indicators(FPT,    │  │
│    │   macd)..."                                                        │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│                   [Loop tiếp tục cho đến khi LLM có đủ thông tin]           │
│                         │                                                   │
│                         ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  Bước 5: LLM TẠO BÁO CÁO                                           │  │
│    │                                                                     │  │
│    │  "Đã đủ dữ liệu. Tôi sẽ tổng hợp thành market_report:              │  │
│    │   - Xu hướng giá: tăng 15% trong 3 tháng                           │  │
│    │   - RSI: 65 (chưa overbought)                                      │  │
│    │   - MACD: bullish crossover tuần trước                             │  │
│    │   - Nhận định: momentum tích cực..."                               │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Tại sao để LLM tự quyết định tool?**

Cách tiếp cận truyền thống: hard-code "luôn lấy RSI, MACD, Bollinger Bands". Vấn đề là không phải lúc nào cũng cần tất cả - phân tích cổ phiếu ngân hàng có thể cần chỉ báo khác với cổ phiếu công nghệ.

Với tool-calling, LLM có khả năng **context-sensitive data gathering** - chọn công cụ phù hợp với từng tình huống cụ thể. Điều này linh hoạt hơn và tránh thu thập dữ liệu thừa.

### 3.3 Chi Tiết Luồng Tranh Luận trong Pha 2

Tranh luận diễn ra theo vòng (rounds), mỗi vòng gồm một lượt Bull và một lượt Bear.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         DEBATE FLOW CONTROL                                 │
│                                                                             │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  KHỞI ĐẦU: Bull Researcher luôn bắt đầu trước                      │  │
│    │                                                                     │  │
│    │  Lý do: Trong tranh luận đầu tư, bên ủng hộ (Bull) thường          │  │
│    │  trình bày case trước để Bear có cái gì để phản biện.              │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  BULL RESEARCHER TURN                                              │  │
│    │                                                                     │  │
│    │  Input đọc từ State:                                               │  │
│    │  ├── market_report (phân tích kỹ thuật)                            │  │
│    │  ├── fundamentals_report (phân tích cơ bản)                        │  │
│    │  ├── news_report (phân tích tin tức)                               │  │
│    │  ├── bear_history (lập luận Bear trước đó, nếu có)                 │  │
│    │  └── bull_memory (bài học từ quá khứ)                              │  │
│    │                                                                     │  │
│    │  Nhiệm vụ:                                                         │  │
│    │  ├── Tìm mọi điểm mạnh từ data                                     │  │
│    │  ├── Phản biện lập luận Bear (nếu đã có)                           │  │
│    │  └── Đưa ra lập luận bullish mạnh nhất có thể                      │  │
│    │                                                                     │  │
│    │  Output ghi vào State:                                             │  │
│    │  ├── bull_history += new_argument                                  │  │
│    │  ├── current_response = "Bull: ..."                                │  │
│    │  └── count += 1                                                    │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  ROUTING DECISION: Kiểm tra current_response                       │  │
│    │                                                                     │  │
│    │  Nếu bắt đầu bằng "Bull" → chuyển đến Bear Researcher              │  │
│    │  Nếu bắt đầu bằng "Bear" → kiểm tra count                          │  │
│    │  Nếu count >= 2 * max_rounds → chuyển đến Research Manager         │  │
│    │  Ngược lại → quay lại Bull Researcher                              │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  BEAR RESEARCHER TURN (tương tự Bull nhưng ngược lập trường)       │  │
│    │                                                                     │  │
│    │  Nhiệm vụ:                                                         │  │
│    │  ├── Tìm mọi rủi ro và điểm yếu từ data                            │  │
│    │  ├── Phản biện lập luận Bull vừa đưa ra                            │  │
│    │  └── Đưa ra cảnh báo bearish mạnh nhất có thể                      │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│                   [Loop cho đến khi đủ số vòng]                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Memory Integration trong Tranh Luận

Một đặc điểm quan trọng là researchers có khả năng **học từ quá khứ** thông qua Semantic Memory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    MEMORY QUERY TRONG DEBATE                                │
│                                                                             │
│    Trước khi xây dựng lập luận, researcher query memory:                    │
│                                                                             │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │  BULL RESEARCHER                                                   │  │
│    │                                                                     │  │
│    │  current_situation = market_report + fundamentals_report           │  │
│    │                                                                     │  │
│    │  similar_past = bull_memory.query(current_situation, n=2)          │  │
│    │                                                                     │  │
│    │  Kết quả có thể là:                                                │  │
│    │  "6 tháng trước với cổ phiếu XYZ có tình huống tương tự:           │  │
│    │   P/E thấp, growth cao. Lập luận về P/E thấp đã thuyết phục        │  │
│    │   nhưng bỏ qua yếu tố debt ratio cao → quyết định sai.             │  │
│    │   Bài học: Luôn address debt khi argue về P/E."                    │  │
│    │                                                                     │  │
│    │  → Bull incorporate bài học: vẫn argue về P/E nhưng                │  │
│    │    proactively address câu hỏi về debt                             │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│    Cơ chế này tạo ra LEARNING LOOP:                                         │
│    Sai lầm quá khứ → Memory → Lập luận tương lai tốt hơn                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# PHẦN II: CHI TIẾT CÁC THÀNH PHẦN

---

## 4. Data Layer - Tầng Thu Thập Dữ Liệu

### 4.1 Bản Chất và Vai Trò

Data Layer là **điểm tiếp xúc duy nhất** của hệ thống với thế giới bên ngoài. Mọi dữ liệu từ nguồn external (vnstock API) đều đi qua tầng này trước khi được sử dụng bởi các tầng khác.

**Tại sao cần tách riêng Data Layer?**

Khi các agent gọi trực tiếp API, logic thu thập dữ liệu bị phân tán khắp codebase. Nếu API thay đổi hoặc cần chuyển sang nguồn dữ liệu khác, phải sửa nhiều nơi. Với Data Layer tập trung, thay đổi chỉ ở một nơi duy nhất.

Ngoài ra, Data Layer là nơi tự nhiên để thêm cross-cutting concerns như caching (tránh gọi API lặp), rate limiting (không vượt quota), và error handling (retry khi API fail).

### 4.2 Stock Price Data

**Bản chất:** Thu thập dữ liệu giá OHLCV (Open, High, Low, Close, Volume) - nền tảng của mọi phân tích kỹ thuật.

**Thiết kế đặc biệt:** Output là **formatted text** thay vì structured data (DataFrame, JSON). Lý do:
- LLM xử lý text tự nhiên tốt hơn structured data
- Giảm token consumption do không cần LLM tự format
- Đảm bảo consistency trong cách data được present

### 4.3 Technical Indicators

**Bản chất:** Tính toán các chỉ báo kỹ thuật từ dữ liệu giá - biến đổi raw price thành signals có ý nghĩa.

**Cơ chế Buffer:** Đây là thiết kế quan trọng giải quyết một vấn đề technical. Để tính SMA-200 (trung bình 200 ngày), cần ít nhất 200 ngày dữ liệu. Nhưng user có thể chỉ yêu cầu 30 ngày gần nhất.

Giải pháp: Module tự động lấy thêm dữ liệu buffer (200+ ngày), tính indicator trên toàn bộ, rồi chỉ trả về phần user yêu cầu. User không cần hiểu technical này - interface đơn giản trong khi implementation phức tạp được ẩn đi.

### 4.4 Fundamental Data

**Bản chất:** Thu thập báo cáo tài chính - Balance Sheet, Income Statement, Cash Flow, và các Financial Ratios. Đây là data cho fundamental analysis.

**Latest-First Principle:** Khi có nhiều kỳ báo cáo, ưu tiên trả về dữ liệu mới nhất. Lý do: trong đầu tư, data mới nhất thường quan trọng nhất. Agent có thể yêu cầu historical data nếu cần, nhưng default case là snapshot hiện tại.

### 4.5 News Data

**Bản chất:** Aggregates tin tức từ nhiều nguồn (CafeF, Vietstock) - input cho sentiment analysis và event detection.

**Time-Based Filtering:** Tự động filter tin cũ hơn 3 tháng. Tin cũ thường không relevant cho quyết định đầu tư hiện tại, giữ lại chỉ tạo noise và tốn token.

---

## 5. Tool Layer - Tầng Trừu Tượng Hóa Công Cụ

### 5.1 Bản Chất và Vai Trò

Tool Layer tạo ra **semantic interface** giữa LLM và Data Layer. Thay vì expose raw Python functions, mỗi data function được "wrap" thành một Tool với rich metadata.

**Vấn đề cần giải quyết:** LLM không thể trực tiếp gọi Python function. LLM chỉ có thể generate text - cần một cơ chế để biến đổi text output thành function execution.

**Giải pháp:** Mỗi tool có schema mô tả:
- Tên tool và mục đích
- Các parameters cần thiết
- Kiểu dữ liệu của từng parameter

Khi LLM được "bind" với tools, nó nhận được catalog các tool schemas. Dựa trên context, LLM generate một "tool call request" (dưới dạng JSON) chứa tên tool và arguments. System parse request này và thực thi function tương ứng.

### 5.2 Lợi Ích của Tool Abstraction

**Context-Sensitive Data Gathering:** Cùng một Market Analyst, khi phân tích cổ phiếu khác nhau có thể chọn indicators khác nhau:
- Cổ phiếu growth: RSI, MACD để đánh giá momentum
- Cổ phiếu value: Bollinger Bands, ATR để đánh giá volatility

Điều này không thể đạt được với hard-coded logic.

**Graceful Degradation:** Nếu một tool fail (API down), LLM có thể quyết định dùng tool khác hoặc tiến hành với data có sẵn, thay vì crash toàn bộ workflow.

---

## 6. Analysis Layer - Tầng Phân Tích

### 6.1 Bản Chất và Vai Trò

Analysis Layer biến đổi **raw data thành insights có cấu trúc**. Mỗi Analyst chuyên về một loại dữ liệu, tạo ra báo cáo mà các agent downstream có thể tiêu thụ.

**Tại sao cần Analysts thay vì để Researchers tự thu thập data?**

Separation of concerns: Researchers nên tập trung vào việc xây dựng lập luận, không phải lo về data gathering và processing. Analysts làm công việc "nặng" về data, researchers làm công việc "nặng" về reasoning.

### 6.2 Market Analyst

**Bản chất:** Thực hiện **Technical Analysis** - phân tích xu hướng giá và momentum dựa trên dữ liệu lịch sử.

**Triết lý Selective Indicator Analysis:** Thay vì tính tất cả indicators có sẵn, Market Analyst được hướng dẫn chọn tối đa 8 indicators cung cấp **complementary insights** (không trùng lặp).

Ví dụ về redundancy: RSI và Stochastic RSI đều đo overbought/oversold - dùng cả hai là lãng phí.

Ví dụ về complementary: RSI (momentum) + MACD (trend) + Bollinger Bands (volatility) - mỗi cái đo một khía cạnh khác nhau.

**Output:** Báo cáo phân tích kỹ thuật với xu hướng giá, tín hiệu từ indicators, và nhận định tổng hợp.

### 6.3 News Analyst

**Bản chất:** Thực hiện **Sentiment Analysis** và **Event Detection** - phân tích ảnh hưởng của tin tức đến cổ phiếu.

**Yêu cầu Impact Analysis:** Agent không chỉ summarize tin tức mà phải phân tích **impact tiềm tàng**. Tin về kết quả kinh doanh tốt có impact khác với tin về thay đổi CEO - cả hai đều positive nhưng theo cách khác nhau.

**Tránh "Mixed Signals":** Prompt hướng dẫn agent tránh kết luận chung chung như "signals are mixed". Thay vào đó, đưa ra đánh giá cụ thể về cán cân giữa positive và negative news.

### 6.4 Fundamentals Analyst

**Bản chất:** Thực hiện **Fundamental Analysis** - đánh giá sức khỏe tài chính và giá trị nội tại của doanh nghiệp.

**Multi-Tool Decision Making:** Agent có quyền truy cập 4 tools (fundamentals, balance_sheet, income_statement, cashflow) và **tự quyết định** cần gọi tool nào:
- Đánh giá nhanh: chỉ cần Financial Ratios
- Phân tích cấu trúc nợ: cần Balance Sheet
- Phân tích khả năng sinh lời: cần Income Statement

Thiết kế này phản ánh thực tế: analyst chuyên nghiệp không phải lúc nào cũng cần toàn bộ báo cáo tài chính.

### 6.5 Factory Pattern cho Agent Creation

**Vấn đề:** Agents cần LLM instance, nhưng LLM được configure ở level cao hơn (TradingAgentsGraph).

**Giải pháp:** Factory function nhận LLM và trả về configured agent. Pattern này cho phép:
- Dependency injection: LLM được inject từ ngoài
- Testability: dễ mock LLM trong unit test
- Flexibility: có thể dùng LLM khác nhau cho agents khác nhau

---

## 7. Debate Layer - Tầng Tranh Luận Đối Kháng

### 7.1 Bản Chất và Vai Trò

Debate Layer là **trái tim của hệ thống** - nơi cơ chế adversarial debate được thực thi. Đây là điểm khác biệt chính so với single-agent systems.

**Nguyên lý cốt lõi:** Bắt buộc tồn tại hai quan điểm đối lập, mỗi quan điểm được đại diện bởi một agent **advocate mạnh mẽ** cho lập trường của mình. Agents không được yêu cầu neutral - tính objectivity nổi lên từ sự đối kháng.

### 7.2 Bull Researcher

**Bản chất:** Advocate cho quyết định **MUA** cổ phiếu. Nhiệm vụ là tìm và trình bày mọi lý do ủng hộ đầu tư.

**Focus Areas:**

**Growth Potential:** Các cơ hội thị trường, tiềm năng mở rộng, dự báo tăng trưởng. Bull tìm kiếm và nhấn mạnh tín hiệu tích cực về tương lai.

**Competitive Advantages:** Lợi thế cạnh tranh bền vững - moat, brand value, network effects. Những yếu tố giúp doanh nghiệp duy trì vị thế.

**Positive Indicators:** Data points hỗ trợ bull case - P/E hợp lý, growth rate cao, technical signals tích cực.

**Bear Counterpoints:** Phản biện trực tiếp lập luận Bear. Đây là yếu tố tạo nên tính "debate" - không chỉ trình bày quan điểm riêng mà phải engage với đối phương.

### 7.3 Bear Researcher

**Bản chất:** Advocate cho quyết định **KHÔNG MUA** (hoặc BÁN). Nhiệm vụ là tìm và trình bày mọi rủi ro và điểm yếu.

**Focus Areas (đối xứng với Bull):**

**Market Saturation:** Dấu hiệu thị trường bão hòa, tăng trưởng chậm lại, competitive pressure.

**Competitive Weaknesses:** Điểm yếu trong mô hình kinh doanh, nguy cơ bị disrupt, thiếu hụt công nghệ/nhân sự.

**Negative Indicators:** Tín hiệu cảnh báo - debt ratio cao, margin thu hẹp, technical warnings.

**Bull Counterpoints:** Chỉ ra assumptions quá lạc quan, data interpretation có bias trong lập luận Bull.

### 7.4 Cơ Chế Tranh Luận

**Turn-Taking:** Bull luôn bắt đầu trước (first mover), sau đó alternates với Bear cho đến khi đạt số vòng cấu hình.

**Tại sao Bull bắt đầu?** Trong context đầu tư, "default action" là không làm gì (không mua). Bull phải make the case cho action (mua), Bear defends status quo (không mua). Đây là structure tự nhiên hơn.

**Response Prefix:** Mỗi response được prefix với "Bull Analyst:" hoặc "Bear Analyst:" để routing logic biết agent nào vừa phát biểu và quyết định chuyển đến ai tiếp theo.

---

## 8. Decision Layer - Tầng Ra Quyết Định

### 8.1 Bản Chất và Vai Trò

Decision Layer tổng hợp toàn bộ kết quả tranh luận và đưa ra **quyết định cuối cùng**. Research Manager đóng vai trò trọng tài - không tham gia tranh luận nhưng đánh giá chất lượng lập luận.

### 8.2 Research Manager Philosophy

**Decisive, Not Neutral:** Một anti-pattern phổ biến khi dùng LLM cho decision making là xu hướng hedging: "Both sides have valid points, it depends on..." Loại output này vô dụng về mặt operational.

Research Manager được explicitly instruct: **"Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments."**

Triết lý này phản ánh thực tế trading: không quyết định cũng là một quyết định (opportunity cost). Manager phải commit với BUY, HOLD, hoặc SELL.

### 8.3 Decision Framework

Research Manager đánh giá tranh luận theo framework có cấu trúc:

**Argument Quality Assessment:** Lập luận nào được support bởi data cụ thể? Lập luận nào dựa trên assumptions không kiểm chứng? Evidence quality là yếu tố quan trọng.

**Counter-argument Effectiveness:** Mỗi bên handle phản biện như thế nào? Address trực tiếp hay né tránh? Khả năng defend position dưới challenge cho biết strength của lập luận.

**Risk-Reward Balance:** Ngay cả khi bull case mạnh hơn, nếu downside risk quá lớn, HOLD có thể hợp lý. Cân nhắc magnitude của outcomes, không chỉ probability.

**Historical Lessons:** Query invest_judge_memory để tìm quyết định tương tự trong quá khứ. Những lessons này influence cách cân nhắc các factors.

### 8.4 LLM Selection Strategy

**Research Manager dùng deep_thinking_llm** (model mạnh hơn như o4-mini), trong khi các agents khác dùng **quick_thinking_llm** (model nhanh hơn như gpt-4o-mini).

**Rationale:** Cost-benefit analysis. Analysts và Researchers được gọi nhiều lần (tool loops, debate rounds) - cần response nhanh và chi phí hợp lý. Research Manager chỉ được gọi một lần ở cuối - là điểm quyết định quan trọng nhất, xứng đáng investment vào model có reasoning capability cao hơn.

---

## 9. Memory System - Hệ Thống Bộ Nhớ

### 9.1 Bản Chất và Vai Trò

Memory System cho phép hệ thống **học từ kinh nghiệm**. Thay vì mỗi phiên phân tích bắt đầu từ zero, agents có thể reference các tình huống tương tự trong quá khứ và áp dụng bài học.

### 9.2 Tại Sao Semantic Memory thay vì Rule-Based?

**Rule-Based Approach:** "if P/E > 25 then bearish" - đơn giản nhưng quá rigid. P/E cao có ý nghĩa khác nhau cho tech startup vs mature utility company.

**Semantic Memory Approach:** Lưu trữ tình huống dưới dạng vector trong không gian ngữ nghĩa. Khi gặp tình huống mới, tìm các tình huống **tương tự** (không cần identical) và retrieve bài học.

Ưu điểm:
- Capture nuances và context mà rules không thể
- Không cần định nghĩa rules explicitly
- Có thể generalize từ examples

### 9.3 Vector-Based Similarity

**Cơ chế hoạt động:**
1. Situation (text mô tả tình huống) được chuyển thành vector 1536-chiều qua OpenAI Embeddings
2. Vector được lưu trong ChromaDB cùng với recommendation (bài học)
3. Khi query, tình huống mới được embed và tìm vectors gần nhất (cosine similarity)

**Tại sao cosine similarity work?** Embedding models được train để đặt texts có meaning tương tự gần nhau trong vector space. "Công ty tech với P/E cao nhưng growth mạnh" sẽ gần với "Startup công nghệ định giá cao, doanh thu tăng nhanh" - dù words khác nhau.

### 9.4 Three Memory Instances

Hệ thống duy trì **ba memory riêng biệt**, mỗi cái phục vụ một agent:

**bull_memory:** Lưu kinh nghiệm về bullish arguments - arguments nào hiệu quả, assumptions nào dễ sai. Bull Researcher query để học từ quá khứ.

**bear_memory:** Tương tự cho bearish arguments - cảnh báo nào chính xác, lo ngại nào hóa ra không đáng kể.

**invest_judge_memory:** Lưu kinh nghiệm ra quyết định của Manager - factors nào thực sự predictive, cách weigh bull vs bear trong các scenarios khác nhau.

**Tại sao tách riêng?** Mỗi agent có vai trò khác nhau, cần học từ experiences khác nhau. Bull không cần biết Bear đã sai ở đâu, và ngược lại.

### 9.5 Learning Loop

Memory system đóng vai trò quan trọng trong việc tạo **closed learning loop**:

```
Analysis → Decision → Execution → Outcome → Reflection → Memory Update
    ↑                                                           │
    └───────────────────────────────────────────────────────────┘
                    (Next analysis informed by memory)
```

Ban đầu, system hoạt động với empty memories. Sau mỗi reflection (phân tích kết quả thực tế), memories trở nên richer, providing more relevant context cho analyses tương lai.

---

## 10. Orchestration Layer - Tầng Điều Phối

### 10.1 Bản Chất và Vai Trò

Orchestration Layer là **conductor** của hệ thống - quyết định agent nào chạy, khi nào, và với input gì. Tầng này implement State Machine architecture sử dụng LangGraph.

### 10.2 State Machine Model

Hệ thống được model như một **state machine** với:
- **Nodes:** Mỗi agent là một node
- **Edges:** Luồng chuyển đổi giữa agents
- **State:** Shared state object được pass qua các nodes

**Tại sao State Machine?**

Workflow của hệ thống có structure rõ ràng: analysts → researchers → manager. State machine cho phép define workflow này declaratively (khai báo nodes và edges) thay vì imperatively (hard-code sequence).

Ngoài ra, state machine tự nhiên handle conditional flows (tool loops, debate rounds) và provide debugging/tracing capabilities.

### 10.3 TradingAgentsGraph

**Vai trò:** Entry point và coordinator của toàn bộ hệ thống.

**Responsibilities:**
- **Initialization:** Tạo LLMs, memories, tool nodes, graph setup
- **Execution:** Chạy workflow với input (ticker, date)
- **Logging:** Lưu state cuối cùng để debug/audit

### 10.4 GraphSetup

**Vai trò:** Xây dựng LangGraph workflow từ configuration.

**Key Feature - Selective Analyst Inclusion:** Không phải lúc nào cũng cần cả ba analysts. Parameter `selected_analysts` cho phép chọn subset. GraphSetup dynamically build graph với chỉ những components được chọn.

### 10.5 ConditionalLogic

**Vai trò:** Encapsulate routing decisions tại các branching points.

**Tool Loop Routing:** Kiểm tra message cuối có tool_calls không. Nếu có → route đến ToolNode để execute. Nếu không → analyst đã xong, route đến next stage.

**Debate Loop Routing:** Kiểm tra count và current_response prefix. Route đến Bull/Bear/Manager dựa trên logic đã define.

### 10.6 Propagator

**Vai trò:** Tạo initial state và configure graph execution.

**Initial State:** Populate state với input (ticker, date) và initialize empty fields cho reports, debate history, etc.

---

# PHẦN III: TÍCH HỢP VÀ MỞ RỘNG

---

## 11. Tích Hợp với MAFIA Model

### 11.1 Complementary Analysis Framework

MAFIA và LLM-Agents được thiết kế như **hai components bổ sung** trong investment analysis:

**MAFIA (Quantitative):** Xử lý dữ liệu số học, apply quantitative criteria, output ranked list của top K stocks.

**LLM-Agents (Qualitative):** Với mỗi stock trong top K, thực hiện deep qualitative analysis - narrative xung quanh company, qualitative aspects của financials, synthesize perspectives qua debate.

### 11.2 Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    STAGE 1: QUANTITATIVE SCREENING (MAFIA)                                  │
│                                                                             │
│    Input: Universe of stocks (ví dụ: 500 cổ phiếu VN30, HNX30...)          │
│    Process: Apply quantitative factors, calculate scores                    │
│    Output: Top K stocks ranked by quantitative score                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    STAGE 2: QUALITATIVE DEEP DIVE (LLM-Agents)                              │
│                                                                             │
│    For each ticker in Top K:                                                │
│    ├── Technical Analysis (Market Analyst)                                  │
│    ├── Fundamental Analysis (Fundamentals Analyst)                          │
│    ├── Sentiment Analysis (News Analyst)                                    │
│    ├── Adversarial Debate (Bull vs Bear)                                    │
│    └── Final Decision (Research Manager)                                    │
│                                                                             │
│    Output per ticker:                                                       │
│    ├── Recommendation: BUY / HOLD / SELL                                    │
│    ├── Confidence level                                                     │
│    ├── Key supporting factors                                               │
│    └── Identified risks                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    STAGE 3: COMBINED RANKING                                                │
│                                                                             │
│    Combined Score = f(MAFIA_score, LLM_recommendation, confidence)          │
│                                                                             │
│    Possible strategies:                                                     │
│    ├── Weighted combination: 0.6 * quant + 0.4 * qual                       │
│    ├── Filter-based: Only include stocks with BUY recommendation           │
│    └── Veto-based: Exclude stocks with SELL recommendation                 │
│                                                                             │
│    Output: Final portfolio recommendation                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Value Addition của LLM-Agents

**Explainability:** MAFIA score là một con số - không giải thích tại sao. LLM-Agents cung cấp human-readable justification, giúp investor hiểu và tin tưởng recommendation.

**Context Sensitivity:** Quantitative models miss context. LLM-Agents có thể factor in recent news (M&A announcement, regulatory change) chưa fully reflected in price.

**Risk Articulation:** Bear Researcher explicitly identify và articulate risks. Investor có clear view về potential downsides, không chỉ upside potential.

---

## 12. Kết Luận

### 12.1 Tóm Tắt Contributions

Hệ thống Multi-Agent Adversarial Debate giải quyết một gap quan trọng trong investment analysis:

**Adversarial Debate Mechanism:** Vượt qua limitations của single-agent LLM (confirmation bias, one-sided analysis) thông qua structured debate giữa Bull và Bear.

**Tool-Augmented Analysis:** Kết hợp reasoning capability của LLMs với real-time data access, enabling context-sensitive data gathering.

**Semantic Memory:** Experience-based learning qua vector similarity, cho phép hệ thống improve over time.

**Layered Architecture:** Modular design cho phép independent evolution của components và easy extension.

### 12.2 Integration với MAFIA

Khi tích hợp với MAFIA quantitative model, LLM-Agents completes investment analysis framework:
- Qualitative assessment bổ sung quantitative ranking
- Human-readable justifications cho recommendations
- Explicit risk articulation
- Continuous improvement through learning

### 12.3 Hướng Phát Triển

**Expanded Coverage:** Thêm domain-specific analysts (Macro Analyst, Sector Analyst).

**Enhanced Memory:** Hierarchical memory với different abstraction levels (trade-specific, sector-level, macro-level).

**Real-time Adaptation:** Streaming data feeds cho intra-day re-evaluation.

---

## Appendix A: File Reference

| Component | File Path | Layer |
|-----------|-----------|-------|
| Stock Data | dataflows/vnstock_stock.py | Data |
| Indicators | dataflows/vnstock_indicator.py | Data |
| Fundamentals | dataflows/vnstock_fundamentals.py | Data |
| News | dataflows/vnstock_newsStock.py | Data |
| Stock Tool | agents/utils/core_stock_tool.py | Tool |
| Indicator Tools | agents/utils/technical_indicators_tools.py | Tool |
| Fundamental Tools | agents/utils/fundamental_data_tools.py | Tool |
| News Tools | agents/utils/news_data_tools.py | Tool |
| Market Analyst | agents/analysts/market_analyst.py | Analysis |
| News Analyst | agents/analysts/news_analyst.py | Analysis |
| Fundamentals Analyst | agents/analysts/fundamentals_analyst.py | Analysis |
| Bull Researcher | agents/researchers/bull_researcher.py | Debate |
| Bear Researcher | agents/researchers/bear_researcher.py | Debate |
| Research Manager | agents/managers/research_manager.py | Decision |
| Memory System | agents/utils/memory.py | Memory |
| State Definitions | agents/utils/agent_states.py | State |
| Main Orchestrator | graph/trading_graph.py | Orchestration |
| Graph Setup | graph/setup.py | Orchestration |
| Routing Logic | graph/conditional_logic.py | Orchestration |
| Propagation | graph/propagation.py | Orchestration |
| Reflection | graph/reflection.py | Learning |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| Adversarial Debate | Cơ chế tranh luận có cấu trúc giữa hai bên có lập trường đối lập |
| Agent | Đơn vị xử lý tự trị, có khả năng nhận input, thực hiện reasoning, và tạo output |
| Bull Researcher | Agent advocate cho quyết định đầu tư, tập trung tìm điểm mạnh |
| Bear Researcher | Agent advocate chống đầu tư, tập trung tìm rủi ro |
| Semantic Memory | Lưu trữ kinh nghiệm dựa trên similarity trong vector space |
| Tool-Augmented LLM | LLM có khả năng gọi công cụ bên ngoài để thu thập thông tin |
| State Machine | Mô hình trong đó hệ thống chuyển đổi giữa các trạng thái |
| Shared State | Pattern giao tiếp qua state object chung |
| Reflection | Phân tích quyết định quá khứ để rút ra bài học |
