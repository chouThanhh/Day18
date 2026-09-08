# Group Report — Lab 18: Production RAG

**Người thực hiện:** Nguyễn Châu Thanh (bài cá nhân — toàn bộ 5 module)
**Ngày:** 2026-09-08

## Phân công & Hoàn thành

| Module | Hàm chính | Hoàn thành | Tests |
|--------|-----------|-----------|-------|
| M1: Chunking | `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware` | ✅ | 13/13 |
| M2: Hybrid Search | `segment_vietnamese`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion` | ✅ | 5/5 |
| M3: Reranking | `CrossEncoderReranker._load_model` / `.rerank` | ✅ | 5/5 |
| M4: Evaluation | `evaluate_ragas`, `failure_analysis` | ✅ | 4/4 |
| M5: Enrichment | 4 techniques + `_enrich_single_call` (combined 1-call) + fallback | ✅ | 10/10 |
| Pipeline | `build_pipeline`, `run_query`, `_expand_to_parents` (small-to-big) | ✅ | — |

**Tổng: 37/37 test pass · 0 TODO.**

## Kết quả RAGAS

| Metric | Naive | Production v1 (child) | **Production v2 (child→parent)** | Δ (v2 − naive) |
|--------|-------|----------------------|----------------------------------|----------------|
| Faithfulness | 0.8667 | 0.6250 | **0.8938** | +0.0271 |
| Answer Relevancy | 0.7130 | 0.5983 | **0.7838** | +0.0708 |
| Context Precision | 0.9250 | 0.9458 | **0.9750** | +0.0500 |
| Context Recall | 0.9250 | 0.8250 | **0.9500** | +0.0250 |

→ **v2: cả 4 metric ≥ 0.78; faithfulness ≥ 0.85** (rubric #7: 10/10; bonus faithfulness ≥ 0.85 và all ≥ 0.75).

## Key Findings

1. **Biggest improvement:** Answer Relevancy +0.071 và Context Precision +0.050 — nhờ hybrid (BM25+dense+RRF) + cross-encoder rerank giữ top-3 gần như luôn liên quan (0.975), cộng với việc trả parent đủ ngữ cảnh cho LLM.
2. **Biggest challenge:** Hierarchical chunking **làm giảm** faithfulness ở v1 (0.87 → 0.63) vì child 256 ký tự cắt giữa câu → 5/20 câu LLM trả "Không tìm thấy" dù context đúng. Phải thêm bước **retrieve child → return parent** (v2) mới lấy lại grounding (→ 0.89).
3. **Surprise finding:** `context_recall`/`precision` cao **không** đảm bảo answer tốt — ở v1 có 5 câu P=R≈1.0 vẫn fail. Chất lượng *lắp ráp context* (chunk boundary, đủ câu) quan trọng ngang chất lượng *retrieval*. Sau khi sửa, các failure còn lại chuyển sang loại **numeric reasoning** và **multi-hop** (lỗi generation/retrieval-strategy, không phải chunking).

## Presentation Notes (5 phút)

1. RAGAS naive vs production v1 vs v2 (bảng trên) — nhấn hành trình sửa faithfulness 0.63 → 0.89.
2. Biggest win: M2 + M3 (hybrid + rerank) cho Context Precision; small-to-big cho Faithfulness.
3. Case study: "Thâm niên mấy năm được cộng phép?" — Error Tree: context precision 0.5 do lẫn policy v2023/v2024 → fix = metadata versioning.
4. Next 1h: lọc version bằng `auto_metadata`, cấm LLM tự tính số (numeric), query decomposition cho multi-hop.

## Latency breakdown (Production v2, CPU — tổng 720.9s cho build + 20 query + eval)

| Bước | Thời gian |
|------|-----------|
| Chunking (M1) | ~1 s |
| Enrichment 105 chunks (M5, 1 call/chunk, combined mode) | ~287 s |
| Index BM25 + Dense bge-m3 (M2) | ~82 s |
| 20 queries: hybrid search + cross-encoder rerank + LLM answer | ~315 s (~16 s/query) |
| RAGAS 4 metrics × 20 câu (M4) | ~35 s |

> Nút cổ chai: enrichment (LLM/chunk) và cross-encoder rerank trên CPU. Production nên chạy GPU cho reranker và batch/async enrichment.
