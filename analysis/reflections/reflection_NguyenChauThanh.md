# Individual Reflection — Lab 18: Production RAG Pipeline

**Tên:** Nguyễn Châu Thanh
**Module phụ trách:** Toàn bộ M1 → M5 + pipeline (làm cá nhân)

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking (nhóm câu theo similarity) | M1 | `chunk_semantic()` | Embed câu bằng `all-MiniLM-L6-v2`, cắt chunk mới khi cosine(sent[i-1], sent[i]) < threshold. Threshold 0.85 (config) tạo rất nhiều chunk nhỏ; hạ xuống ~0.5 mới gom câu cùng chủ đề. Ưu điểm: không cắt giữa ý. |
| Hierarchical / small-to-big retrieval | M1 | `chunk_hierarchical()` + `pipeline._expand_to_parents()` | Parent 2048 char, child 256 char, 26 doc → 105 child / 26 parent. **Quan sát quan trọng:** pipeline ban đầu index+trả child → faithfulness tụt 0.87→0.63 (child cắt giữa câu, 5/20 câu "Không tìm thấy" dù context recall/precision ≈1.0). Sau khi thêm bước retrieve child → **return parent**, faithfulness lên 0.89. "Return parent" là nửa quan trọng nhất của hierarchical, không được bỏ. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | `re.split` theo markdown header `#{1,3}`, giữ header trong text + `metadata["section"]`. Hợp với corpus HR toàn file .md có heading rõ ràng. |
| Vietnamese tokenization cho BM25 | M2 | `segment_vietnamese()` | `underthesea.word_tokenize(format="text")` rồi `replace("_", " ")`. Nếu không replace `_`, "nghỉ_phép" là 1 token còn query "nghỉ phép" là 2 token → BM25 không khớp. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | score(d) = Σ 1/(k + rank + 1), k=60. RRF chỉ dùng thứ hạng nên không cần chuẩn hoá score giữa BM25 (không giới hạn) và cosine (0-1). Giải quyết việc lexical match (BM25) và semantic match (dense) bổ sung cho nhau. |
| Dense retrieval với bge-m3 + Qdrant | M2 | `DenseSearch.index/search` | qdrant-client mới dùng `query_points()` (không phải `search()`). Payload nhét cả `text` để lấy lại nội dung. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")`, `model.predict([(query, doc)])`. Lấy top-20 hybrid → rerank → top-3. Cross-encoder đọc query+doc cùng lúc nên chính xác hơn bi-encoder, đổi lại chậm hơn (chạy CPU ~vài trăm ms/batch). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Bọc try/except vì RAGAS cần OPENAI_API_KEY + gọi LLM/metric. Kết quả cuối: faithfulness 0.894 / answer_relevancy 0.784 / context_precision 0.975 / context_recall 0.950 (baseline 0.867/0.713/0.925/0.925). Metric thấp nhất = answer_relevancy vì các câu multi-hop ("phép năm VÀ lương") chỉ trả lời được 1 vế. |
| Diagnostic / Error Tree | M4 | `failure_analysis()` | Với mỗi câu: tính avg 4 metric, lấy `worst_metric` → tra bảng chẩn đoán (vd context_recall thấp → thiếu chunk → sửa chunking/BM25). Sort tăng dần lấy bottom-N. |
| Contextual embeddings (Anthropic) | M5 | `contextual_prepend()` / `_enrich_single_call()` | Prepend 1 câu mô tả chunk nằm ở đâu trong tài liệu trước khi embed → giảm retrieval failure khi chunk bị mất ngữ cảnh (đại từ, "điều này"...). |
| Enrichment combined (cost) | M5 | `_enrich_single_call()` | 1 API call/chunk trả JSON gồm summary + questions + context + metadata thay vì 4 call riêng → giảm 75% chi phí. |

## Phần 2: Khó khăn & cách giải quyết

- **Production pipeline regression (lỗi lớn nhất):** sau khi ghép đủ M1–M5, RAGAS faithfulness *giảm* so với baseline (0.6250 vs 0.8667). Debug: đọc `ragas_report.json` → thấy 5 câu answer = `"Không tìm thấy."` nhưng `context_precision`/`context_recall` của chính các câu đó ≈ 1.0 → retrieval đúng, generation từ chối. Nguyên nhân: `chunk_hierarchical` trả child 256 ký tự bị cắt mid-sentence; pipeline index và *trả về luôn child*. Sửa: thêm `parent_map` + `_expand_to_parents()` để retrieve child → generate trên parent. Re-run: faithfulness 0.8938, cả 4 metric ≥ 0.78.
- **Console Windows `UnicodeEncodeError: 'charmap' codec can't encode`** khi pipeline in emoji (⚠️, ✓) qua stdout redirect. Fix: `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`.
- **`OPENAI_API_KEY` trong `.env` là placeholder `sk-...`** → RAGAS trả `AuthenticationError 401` và toàn bộ metric = 0. Fix: đặt key thật vào `.env` (không commit).
- **`pytest` không có trong `.venv`** (`No module named pytest`) dù các package nặng đã cài. Fix: `.venv/Scripts/python.exe -m pip install pytest`.
- **`SEMANTIC_THRESHOLD = 0.85` quá cao** với `all-MiniLM-L6-v2` trên tiếng Việt → gần như mỗi câu 1 chunk. Test dùng threshold 0.5. Bài học: threshold semantic phụ thuộc model embedding, phải tune bằng eval chứ không có hằng số vạn năng.
- **2 PDF (BCTC, Nghị định 13) là scan ảnh, không có text layer** → `load_documents()` bỏ qua kèm cảnh báo. RAG text-based không đọc được nếu chưa OCR.
- **qdrant-client API**: `recreate_collection` cần keyword `collection_name=`, search phải dùng `query_points()`.

## Phần 3: Action Plan cho project cá nhân

### Hiện tại
- RAG pipeline: chunking cố định + dense-only search + không rerank.
- Known issues: trả lời sai khi câu hỏi có version ("chính sách mới nhất"), recall thấp với câu hỏi đa nguồn.

### Plan áp dụng
1. [ ] **Chunking**: hierarchical (parent 2048 / child 256) làm mặc định, structure-aware cho tài liệu có heading. Lý do: cân bằng precision khi retrieve và context khi generate.
2. [ ] **Search**: Hybrid BM25 (underthesea) + dense (bge-m3) + RRF. Lý do: tiếng Việt nhiều thuật ngữ/mã số → cần lexical; RRF khỏi phải chuẩn hoá score.
3. [ ] **Reranking**: có — `bge-reranker-v2-m3`, top-20 → top-5. Chấp nhận thêm latency để tăng context precision.
4. [ ] **Evaluation**: RAGAS 4 metrics làm CI gate + failure_analysis Error Tree cho bottom-10 mỗi lần thay đổi.
5. [ ] **Enrichment**: `contextual_prepend` (combined single-call) — rẻ, tác động lớn nhất tới recall theo benchmark Anthropic; thêm auto-metadata để lọc theo version/phòng ban.

### Timeline
- Tuần 1: thay chunking + dựng test_set 30 câu + đo baseline RAGAS.
- Tuần 2: hybrid search + rerank, so sánh Δ.
- Tuần 3: enrichment + metadata filter cho câu hỏi version, chốt cấu hình theo RAGAS.
