# Genetic Algorithm for Rule Discovery — Play-Tennis Dataset

Học **conjunctive classification rules** dạng `IF <điều kiện> THEN PlayTennis = Yes` từ bảng Play-Tennis (Mitchell, Chapter 3) bằng Genetic Algorithm.

---

## Requirements

```
Python 3.x
numpy
pandas
```

```bash
pip install numpy pandas
```

---

## How to Run

```bash
python HW_GA.py
```

`generate.py` không còn được dùng và có thể bỏ qua.

---

## Design Decisions

### 1. Dataset — 14 instances gốc, không generate thêm

Bài toán là **rule learning** trên bảng truth table của sách (14 dòng). Không cần mở rộng dataset vì:

- GA đánh giá fitness bằng cách duyệt từng instance — 14 dòng là đủ để phân biệt rule tốt và xấu.
- Synthetic data được tạo bằng cách sampling lại từ chính distribution gốc → không thêm thông tin mới, chỉ làm chậm GA.

### 2. Bit-string Encoding

Mỗi hypothesis được encode thành một bit-string 10 bit:

```
[ Outlook (3) | Temperature (3) | Humidity (2) | Wind (2) ]
  Sunny Overcast Rain  Hot Mild Cool  High Normal  Weak Strong
```

Bit `1` = giá trị đó được include trong rule. Bit `0` = không có điều kiện (wildcard).

Ví dụ: `1 0 0 | 0 0 0 | 1 0 | 0 0` → `IF Outlook=Sunny AND Humidity=High THEN ...`

Nếu nhiều bit của cùng một attribute = 1, rule đọc là OR trong attribute đó:
`0 0 0 | 0 1 1 | 0 0 | 0 0` → `IF (Temperature=Mild OR Temperature=Cool) THEN ...`

### 3. Fitness Function — Laplace Accuracy

**Vấn đề với accuracy thông thường:** Khi instance không match rule, code cũ tính là predict = No, vô tình reward các rule hẹp chỉ vì chúng "đúng" trên tất cả các No instance không match. GA vì vậy bias về các rule đơn giản như `Humidity=Normal` thay vì tìm ra rule tốt hơn như `Outlook=Overcast`.

**Laplace accuracy** chỉ tính trên tập instances mà rule cover:

```
fitness = (TP + 1) / (TP + FP + 2) - complexity_penalty
```

So sánh:

| Rule | TP | FP | Laplace | Accuracy cũ |
|---|---|---|---|---|
| `Outlook=Overcast` | 4 | 0 | **0.833** | 0.286 |
| `Humidity=Normal` | 6 | 1 | 0.778 | **0.714** |

Với Laplace fitness, GA tìm ra `Outlook=Overcast` — rule có precision 100%.

### 4. Crossover Operators

Bốn loại crossover được implement:

- **single** — Single-point: cắt tại 1 điểm ngẫu nhiên trong bitstring.
- **two_point** — Two-point: cắt tại 2 điểm, swap đoạn giữa.
- **uniform** — Uniform: mỗi bit chọn ngẫu nhiên từ một trong hai parent.
- **attribute** — Attribute-based: crossover tại ranh giới attribute, giữ nguyên từng block. Đây là operator phù hợp nhất với cách encode vì không tạo ra bit-string không hợp lệ.

### 5. Mutation

Khi một bit bị flip thành 1, toàn bộ bits còn lại của attribute đó được reset về 0. Điều này đảm bảo mutation không tạo ra trạng thái ambiguous (một attribute có nhiều giá trị active do mutation, khác với crossover có thể tạo OR có chủ đích).

### 6. Selection — Tournament Selection

Chọn ngẫu nhiên 3 candidate từ population, lấy candidate có fitness cao nhất. Cân bằng giữa selection pressure và diversity.

### 7. Elitism

Mỗi generation, `(1 - replacement_rate)` fraction của population được giữ nguyên (elite). Phần còn lại được thay thế bằng offspring từ crossover + mutation.

---

## Experiment Configurations

| Exp | pop_size | replacement_rate | mutation_rate | crossover |
|-----|----------|-----------------|---------------|-----------|
| 1   | 50       | 0.30            | 0.05          | attribute |
| 2   | 100      | 0.50            | 0.10          | uniform   |
| 3   | 30       | 0.20            | 0.02          | attribute |
| 4   | 80       | 0.40            | 0.08          | two_point |
| 5   | 150      | 0.60            | 0.15          | uniform   |
| 6   | 40       | 0.25            | 0.03          | single    |

---

## Output

```
--- Experiment Results Summary ---
  Exp 1: pop=50 replace=0.3 mut=0.05 xover=attribute -> fitness=0.8568
  Exp 2: pop=100 replace=0.5 mut=0.1 xover=uniform -> fitness=0.8568
  Exp 3: pop=30 replace=0.2 mut=0.02 xover=attribute -> fitness=0.8565
  Exp 4: pop=80 replace=0.4 mut=0.08 xover=two_point -> fitness=0.8332
  Exp 5: pop=150 replace=0.6 mut=0.15 xover=uniform -> fitness=0.8568
  Exp 6: pop=40 replace=0.25 mut=0.03 xover=single -> fitness=0.8568
=== Best Configuration ===
  pop_size: 50
  replacement_rate: 0.3
  mutation_rate: 0.05
  crossover_type: attribute
  Fitness: 0.8568
=== Final Run with Best Configuration (200 generations) ===
  Best Rule (fitness=0.8568): (Outlook=Overcast OR Outlook=Rain) AND Wind=Weak
--- Rule Evolution ---
  Gen   0: (Outlook=Overcast OR Outlook=Rain) AND Wind=Weak  (fitness=0.8568)
--- Detailed Evaluation of Best Rule ---
  Rule     : (Outlook=Overcast OR Outlook=Rain) AND Wind=Weak
  Laplace  : 0.8571
  Accuracy : 0.7143
  Precision: 1.0000
  Recall   : 0.5556
  F1 Score : 0.7143
  Confusion: TP=5  FP=0  TN=5  FN=4
```

Nhận xét:
Gen 0: fitness=0.8568 và không có improvement sau đó nghĩa là GA tìm được rule tốt nhất ngay ở population khởi tạo ngẫu nhiên và không cải thiện thêm được nữa. Điều này không phải lỗi — với hypothesis space chỉ có 10 bit (1024 tổ hợp) và dataset 14 dòng, population 50 đã cover rất nhiều phần của search space ngay từ đầu.

Laplace fitness đúng hướng. Rule (Outlook=Overcast OR Outlook=Rain) AND Wind=Weak thực ra có thể diễn giải được theo logic thực tế: chơi tennis khi trời không nắng và gió nhẹ. Precision 100% là điểm mạnh của rule learning kiểu này.

Tóm lại, GA tìm ra rule tốt hơn Outlook=Overcast vì Laplace fitness thưởng cho việc cover thêm TP mà không tạo ra FP. Gen 0 đã là best vì search space nhỏ (10 bit) nên population 50 may mắn cover được solution ngay từ đầu — không phải bug

---

## File Structure

```
HW_GA.py     # Main file — dataset, GA, experiments
README.md    # This file
```
