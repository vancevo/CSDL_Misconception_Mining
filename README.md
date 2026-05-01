# 🔬 Misconception Mining — ASAG Research Framework

> **Đề tài:** Khai phá Lỗi sai và Mẫu hình Sai lầm trong Câu trả lời Sinh viên  
> **Phương pháp:** Sentence Embedding (SBERT) × UMAP × HDBSCAN/BERTopic  
> **Dữ liệu:** 10,000+ mẫu câu trả lời ngắn — 2 nguồn chính

---

## 📋 Mục lục

1. [Tổng quan đề tài](#1-tổng-quan-đề-tài)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Sơ đồ Pipeline](#3-sơ-đồ-pipeline)
4. [Cơ sở dữ liệu & Schema](#4-cơ-sở-dữ-liệu--schema)
5. [Dữ liệu mẫu](#5-dữ-liệu-mẫu)
6. [Cấu hình thực nghiệm](#6-cấu-hình-thực-nghiệm)
7. [Kết quả thực nghiệm](#7-kết-quả-thực-nghiệm)
8. [Cách chạy](#8-cách-chạy)

---

## 1. Tổng quan đề tài

Hệ thống **Automatic Short Answer Grading (ASAG)** thông thường chỉ phân loại câu trả lời thành *đúng/sai*, mà **không phân tích tại sao sinh viên sai**. Đề tài này xây dựng pipeline **Misconception Mining** để:

- Tự động **phát hiện nhóm lỗi sai** (misconception) từ câu trả lời sinh viên
- **So sánh 9 cấu hình** (3 embedding strategies × 3 clustering methods)
- Đánh giá bằng cả **metric nội tại** (Silhouette, CH, DB) và **ngoại tại** (NMI, ARI, Purity)
- Hiển thị kết quả qua **giao diện web tương tác**

---

## 2. Cấu trúc thư mục

```
DBMS_Misconception_Mining/
├── configs/
│   ├── misconception.yaml      # Siêu tham số: UMAP, HDBSCAN, embedding
│   └── data.yaml               # Đường dẫn dataset
│
├── data/
│   ├── raw/                    # Dữ liệu thô (SciEntsBank XML, v.v.)
│   └── unified/
│       ├── data_generate.jsonl # 10,000 mẫu tổng hợp (có annotation đầy đủ)
│       └── data_scraping.jsonl # Câu hỏi/đáp án tham chiếu từ OpenStax
│
├── demos/
│   ├── project2-misconception/ # Next.js dashboard (npm run dev)
│   └── demo.html               # 🌟 Demo standalone — mở thẳng bằng trình duyệt
│
├── essays/
│   └── Essay3_Misconception_Mining.md  # Tiểu luận đầy đủ
│
├── experiments/
│   └── phase3_misconception.py # Script chạy toàn bộ 9 thực nghiệm
│
├── src/
│   ├── data/
│   │   ├── schema.py           # UnifiedRecord dataclass (cấu trúc dữ liệu chính)
│   │   ├── loaders.py          # Loader cho từng nguồn dữ liệu
│   │   ├── dataset.py          # Dataset wrapper
│   │   ├── harmonizer.py       # Chuẩn hóa nhãn giữa các nguồn
│   │   ├── splitter.py         # Train/test/val split
│   │   └── audit.py            # Kiểm tra chất lượng dữ liệu
│   │
│   ├── misconception/
│   │   ├── embedder.py         # 3 chiến lược embedding (A, B, C)
│   │   ├── clustering.py       # KMeans, HDBSCAN, BERTopic, c-TF-IDF
│   │   └── evaluator.py        # Intrinsic & extrinsic metrics
│   │
│   ├── evaluation/
│   │   └── reporting.py        # Lưu kết quả JSON
│   └── utils.py                # set_seed, helpers
│
├── tests/
│   ├── test_phase3_misconception.py
│   ├── test_clustering.py
│   ├── test_embedder.py
│   └── test_evaluator.py
│
├── results/phase3/             # Kết quả thực nghiệm (JSON, tự động tạo)
├── data-generate.csv           # Dataset gốc (CSV, 25MB)
└── data-scraping.json          # Câu hỏi scraping (JSON, 52KB)
```

---

## 3. Sơ đồ Pipeline

### 3.1 Sơ đồ đầu vào — đầu ra tổng quát

```
┌─────────────────────────────────────────────────────────────────┐
│                         ĐẦU VÀO                                 │
│                                                                  │
│  data_generate.jsonl          data_scraping.jsonl               │
│  (10,000 mẫu có label)        (câu hỏi/tham chiếu)             │
│       │                              │                           │
│       └──────────────┬───────────────┘                          │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   Lọc dữ liệu (Filter)   │
        │ label_5way ∈ {           │
        │   partially_correct,     │
        │   contradictory,         │
        │   irrelevant             │
        │ }                        │
        └──────────────┬───────────┘
                       │ ~6,000–7,000 mẫu sai
                       ▼
        ┌──────────────────────────┐
        │   SBERT Embedding        │  all-MiniLM-L6-v2
        │                          │  → vector 384 chiều
        │  Strategy A: embed(s)    │
        │  Strategy B: embed(q+s)  │
        │  Strategy C: embed(q+r+s)│
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   UMAP Reduction         │  384D → 5D (clustering)
        │   n_neighbors=15         │  384D → 2D (visualization)
        │   metric=cosine          │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   Clustering             │
        │                          │
        │  ① KMeans (K=10)        │
        │  ② HDBSCAN              │
        │  ③ BERTopic             │
        │    (HDBSCAN + c-TF-IDF) │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   Đánh giá               │
        │                          │
        │  Intrinsic:              │
        │    Silhouette, CH, DB    │
        │  Extrinsic:              │
        │    NMI, ARI, Purity,     │
        │    V-measure             │
        └──────────────┬───────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ĐẦU RA                                  │
│                                                                  │
│  results/phase3/                                                 │
│  ├── global_grid_results.json    (9 cấu hình × global)         │
│  ├── granularity_results.json    (per-question, per-domain)     │
│  ├── cluster_keyword_summaries.json  (top-5 từ khóa/cluster)   │
│  └── phase3_all_results.json     (tổng hợp)                    │
│                                                                  │
│  demos/demo.html  →  Giao diện web tương tác                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Sơ đồ luồng dữ liệu chi tiết (Data Flow)

```
data-generate.csv  ──►  load_data_generate()  ──►  [UnifiedRecord × 10,000]
                                                          │
data_generate.jsonl ◄── (convert & save) ◄───────────────┘
                                                          │
                              filter_misconception_records()
                                                          │
                              [UnifiedRecord × ~6,500]  (sai)
                                                          │
                    ┌─────────────────────────────────────┤
                    │                                     │
              Granularity A                         Granularity B
              (global)                              (per_question)
                    │                                     │
             MisconceptionEmbedder.embed()          MisconceptionEmbedder.embed()
                    │                                     │
             EmbeddingResult[]                     EmbeddingResult[] (per Q)
                    │
             run_clustering() × 3 methods
                    │
             ClusterResult { labels, n_clusters, embeddings, keywords }
                    │
             compute_intrinsic_metrics()  +  compute_extrinsic_metrics()
                    │
             save_results() → results/phase3/*.json
```

---

## 4. Cơ sở dữ liệu & Schema

### 4.1 Bảng `UnifiedRecord` — Cấu trúc dữ liệu chính

Tất cả dữ liệu được chuẩn hóa về 1 cấu trúc duy nhất `UnifiedRecord` (Python dataclass, lưu dạng JSONL):

| Nhóm | Trường | Kiểu | Mô tả |
|------|--------|------|-------|
| **🔑 Identity** | `sample_id` | `str` | ID duy nhất, ví dụ `GEN_00001` |
| | `source_dataset` | `str` | Nguồn: `data_generate`, `scientsbank`, `mohler`, `data_scraping` |
| | `original_id` | `str` | ID gốc trong dataset nguồn |
| | `question_id` | `str` | ID câu hỏi, dùng để nhóm per-question |
| **🌐 Domain** | `domain` | `str` | Lĩnh vực: `physics`, `biology`, `chemistry`, ... |
| | `subdomain` | `str` | Chuyên ngành hẹp hơn |
| | `difficulty` | `str` | `easy` / `medium` / `hard` / `unknown` |
| **📝 Bộ ba chính** | `question` | `str` | Câu hỏi |
| | `reference_answer` | `str` | Câu trả lời tham chiếu (đúng) |
| | `student_answer` | `str` | Câu trả lời của sinh viên |
| | `alternative_reference_answers` | `list[str]` | Các đáp án tham chiếu thay thế |
| **🏷️ Nhãn chấm điểm** | `score_raw` | `float?` | Điểm gốc (0–5) |
| | `score_normalized` | `float?` | Điểm chuẩn hóa [0.0–1.0] |
| | `label_2way` | `str?` | `correct` / `incorrect` |
| | `label_3way` | `str?` | `correct` / `partially_correct` / `incorrect` |
| | `label_5way` | `str?` | `correct` / `partially_correct_incomplete` / `contradictory` / `irrelevant` / `non_domain` |
| **🔍 Annotation lỗi sai** | `misconception_tags` | `list[str]` | Nhãn lỗi sai (gold label cho extrinsic eval) |
| | `misconception_inventory` | `list[dict]` | Chi tiết từng lỗi: `{name, description, severity}` |
| | `missing_concepts` | `list[str]` | Khái niệm bị thiếu trong câu trả lời |
| | `extra_incorrect_claims` | `list[str]` | Khẳng định sai được thêm vào |
| | `key_concepts` | `list[str]` | Các khái niệm chính câu hỏi kiểm tra |
| **💬 Phản hồi** | `feedback_short` | `str?` | Phản hồi ngắn cho sinh viên |
| | `feedback_detailed` | `str?` | Phản hồi chi tiết |
| | `feedback_type` | `str?` | Loại phản hồi |
| | `feedback_tone` | `str?` | Giọng điệu phản hồi |
| **⚙️ Metadata** | `split` | `str` | `train` / `test` / `val` |
| | `is_human_annotated` | `bool` | Người gán nhãn hay tự động |
| | `is_synthetic` | `bool` | Dữ liệu tổng hợp bởi LLM |
| | `is_adversarial` | `bool` | Có biến thể nhiễu loạn |
| | `perturbation_type` | `str?` | Kiểu nhiễu loạn |
| | `annotation_confidence` | `float?` | Độ tin cậy annotation [0–1] |
| **🚦 Usability Flags** | `usable_for_grading` | `bool` | Dùng được cho chấm điểm |
| | `usable_for_feedback` | `bool` | Dùng được tạo phản hồi |
| | `usable_for_misconception_mining` | `bool` | Dùng được cho khai phá lỗi sai |
| | `usable_for_robustness_eval` | `bool` | Dùng được đánh giá độ bền |

### 4.2 Bảng nguồn dữ liệu

| Dataset | File | Số mẫu | Định dạng | Có annotation lỗi sai? | Ghi chú |
|---------|------|--------|-----------|----------------------|---------|
| **Data_Generate** | `data-generate.csv` / `data_generate.jsonl` | ~10,000 | CSV → JSONL | ✅ Đầy đủ | Sinh bởi LLM, có `misconception_tags` (gold label) |
| **Data_Scraping** | `data-scraping.json` / `data_scraping.jsonl` | ~700 | JSON → JSONL | ❌ | Câu hỏi/đáp án từ OpenStax, không có student answer |
| **SciEntsBank** | `data/raw/scientsbank/` | ~10,000 | XML → JSONL | ⚠️ Chỉ label_5way | Dataset benchmark SemEval-2013 |
| **MohlerASAG** | `data/raw/mohler/` | ~2,442 | CSV → JSONL | ❌ | Khoa học máy tính, có điểm số 0-5 |

### 4.3 Phân phối nhãn `label_5way`

| Nhãn | Ý nghĩa | Dùng cho Mining? |
|------|---------|----------------|
| `correct` | Câu trả lời đúng hoàn toàn | ❌ Loại bỏ |
| `partially_correct_incomplete` | Đúng một phần, thiếu thông tin | ✅ |
| `contradictory` | Mâu thuẫn với đáp án đúng | ✅ |
| `irrelevant` | Không liên quan đến câu hỏi | ✅ |
| `non_domain` | Ngoài phạm vi môn học | ❌ Loại bỏ |

---

## 5. Dữ liệu mẫu

### 5.1 Mẫu bản ghi `data_generate.jsonl`

```json
{
  "sample_id": "GEN_00042",
  "source_dataset": "data_generate",
  "original_id": "gen_q015_s042",
  "question_id": "Q015",

  "domain": "physics",
  "subdomain": "mechanics",
  "difficulty": "medium",

  "question": "What is the relationship between force and acceleration?",
  "reference_answer": "Force equals mass times acceleration (F = ma). A larger force produces greater acceleration for the same mass.",
  "student_answer": "Force and acceleration are the same thing, just measured in different units.",

  "score_raw": 0.5,
  "score_normalized": 0.1,
  "label_5way": "contradictory",
  "label_3way": "incorrect",
  "label_2way": "incorrect",

  "key_concepts": ["Newton's second law", "force", "acceleration", "mass"],
  "misconception_tags": ["confuses_force_with_acceleration"],
  "misconception_inventory": [
    {
      "name": "confuses_force_with_acceleration",
      "description": "Student treats force and acceleration as equivalent quantities",
      "severity": "high"
    }
  ],
  "missing_concepts": ["F = ma relationship", "role of mass"],
  "extra_incorrect_claims": ["force and acceleration have same units"],

  "feedback_short": "Force and acceleration are different quantities related by F=ma.",
  "feedback_detailed": "Newton's second law states F = ma...",

  "split": "train",
  "is_synthetic": true,
  "is_human_annotated": false,
  "annotation_confidence": 0.92
}
```

### 5.2 Mẫu kết quả phân cụm (`global_grid_results.json`)

```json
{
  "experiment": "global_grid",
  "strategy": "question_answer",
  "method": "bertopic",
  "granularity": "global",
  "group_key": "global",
  "n_samples": 6482,
  "n_clusters": 9,
  "keywords": {
    "0": ["energy", "force", "work", "power", "heat"],
    "1": ["force", "direction", "push", "pull", "gravity"],
    "2": ["unit", "measure", "kilogram", "newton", "joule"]
  },
  "intrinsic": {
    "silhouette": 0.42,
    "calinski_harabasz": 91.2,
    "davies_bouldin": 1.41
  },
  "extrinsic": {
    "nmi": 0.63,
    "ari": 0.49,
    "purity": 0.72,
    "v_measure": 0.61
  }
}
```

### 5.3 Ví dụ các nhóm lỗi sai phát hiện được

| Cluster | Từ khóa đặc trưng (c-TF-IDF) | Ví dụ câu trả lời sai |
|---------|------------------------------|----------------------|
| **#0 Energy Confusion** | energy, force, work, power, heat | *"Energy and force are the same because both make things move"* |
| **#1 Force Direction** | force, direction, push, pull, gravity | *"Force always goes in the direction of motion"* |
| **#2 Unit Confusion** | unit, kilogram, newton, joule | *"Kilograms and Newtons measure the same thing"* |
| **#3 Process Reversal** | reverse, opposite, backward, order | *"Heat flows from cold to hot objects"* |
| **#4 Scope Error** | scope, general, specific, broad | *"All chemical reactions produce heat"* |
| **#5 Terminology Mix-up** | term, definition, vocabulary | *"Speed and velocity are exactly the same"* |

---

## 6. Cấu hình thực nghiệm

### 6.1 Ma trận 9 cấu hình (3 Embedding × 3 Clustering)

| Config | Embedding Strategy | Clustering | UMAP Pre-reduction |
|--------|--------------------|------------|--------------------|
| **C1** | A — `answer_only`: `embed(student_answer)` | KMeans (K=10) | Không |
| **C2** | A — `answer_only` | UMAP+HDBSCAN | 384D → 5D |
| **C3** | A — `answer_only` | BERTopic | 384D → 5D + c-TF-IDF |
| **C4** | B — `question_answer`: `embed(q + s)` | KMeans (K=10) | Không |
| **C5** | B — `question_answer` | UMAP+HDBSCAN | 384D → 5D |
| **C6** | B — `question_answer` | BERTopic | 384D → 5D + c-TF-IDF |
| **C7** | C — `full_triplet`: `embed(q + ref + s)` | KMeans (K=10) | Không |
| **C8** | C — `full_triplet` | UMAP+HDBSCAN | 384D → 5D |
| **C9** | C — `full_triplet` | BERTopic | 384D → 5D + c-TF-IDF |

### 6.2 Siêu tham số cố định (`configs/misconception.yaml`)

| Thành phần | Tham số | Giá trị |
|------------|---------|---------|
| **SBERT** | Model | `all-MiniLM-L6-v2` (22.7M params) |
| **UMAP** | `n_components` | 5 (clustering) / 2 (visualization) |
| | `n_neighbors` | 15 |
| | `min_dist` | 0.1 |
| | `metric` | cosine |
| **HDBSCAN** | `min_cluster_size` | 5 |
| | `min_samples` | 3 |
| | `cluster_selection_method` | `eom` |
| **KMeans** | `n_clusters` | 10 |
| | `n_init` | 10 |
| **c-TF-IDF** | `top_n_keywords` | 5 |
| **Global** | `seed` | 42 |

---

## 7. Kết quả thực nghiệm

### 7.1 Intrinsic Metrics (Global Granularity)

| Config | Strategy | Method | Silhouette ↑ | CH Index ↑ | DB Index ↓ | K tìm được |
|--------|----------|--------|:------------:|:----------:|:----------:|:---------:|
| C1 | answer_only | KMeans | 0.15 | 45.2 | 2.31 | 10 |
| C2 | answer_only | HDBSCAN | 0.28 | 62.8 | 1.87 | 7 |
| C3 | answer_only | BERTopic | 0.29 | 64.1 | 1.82 | 7 |
| C4 | question_answer | KMeans | 0.22 | 58.3 | 2.05 | 10 |
| C5 | question_answer | HDBSCAN | 0.41 | 89.5 | 1.45 | 9 |
| **C6** | **question_answer** | **BERTopic** | **0.42** | **91.2** | **1.41** | **9** |
| C7 | full_triplet | KMeans | 0.19 | 52.7 | 2.18 | 10 |
| C8 | full_triplet | HDBSCAN | 0.38 | 82.1 | 1.56 | 8 |
| C9 | full_triplet | BERTopic | 0.39 | 84.3 | 1.52 | 8 |

### 7.2 Extrinsic Metrics — so sánh với gold `misconception_tags`

| Config | NMI ↑ | ARI ↑ | Purity ↑ | V-measure ↑ |
|--------|:-----:|:-----:|:--------:|:-----------:|
| C1 | 0.32 | 0.18 | 0.45 | 0.30 |
| C2 | 0.48 | 0.31 | 0.58 | 0.46 |
| C3 | 0.49 | 0.32 | 0.59 | 0.47 |
| C4 | 0.45 | 0.28 | 0.55 | 0.43 |
| C5 | 0.62 | 0.48 | 0.71 | 0.60 |
| **C6** | **0.63** | **0.49** | **0.72** | **0.61** |
| C7 | 0.41 | 0.25 | 0.52 | 0.39 |
| C8 | 0.57 | 0.42 | 0.66 | 0.55 |
| C9 | 0.58 | 0.43 | 0.67 | 0.56 |

> 🏆 **C6 (Strategy B + BERTopic) là cấu hình tốt nhất** trên tất cả metric

---

## 8. Cách chạy

### 8.1 Demo trực quan (không cần cài đặt)

```bash
open /Users/Vinh/Documents/STUDY/DBMS_Misconception_Mining/demos/demo.html
```

### 8.2 Chạy toàn bộ thực nghiệm Python

```bash
# Bước 1: Di chuyển vào thư mục
cd /Users/Vinh/Documents/STUDY/DBMS_Misconception_Mining

# Bước 2: Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Bước 3: Cài thư viện
pip install numpy pyyaml scikit-learn hdbscan umap-learn sentence-transformers

# Bước 4: Chạy thực nghiệm
python3 experiments/phase3_misconception.py

# Kết quả lưu tại:  results/phase3/
```

### 8.3 Chạy Next.js Dashboard

```bash
cd demos/project2-misconception
npm install
npm run dev
# Mở: http://localhost:3002
```

### 8.4 Chạy Unit Tests

```bash
source venv/bin/activate
pytest tests/test_phase3_misconception.py tests/test_clustering.py tests/test_embedder.py -v
```

---

*Tài liệu đầy đủ: xem [`essays/Essay3_Misconception_Mining.md`](essays/Essay3_Misconception_Mining.md)*
