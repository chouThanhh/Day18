# Failure Analysis — Lab 18: Production RAG

**Người thực hiện:** Nguyễn Châu Thanh (làm cá nhân — M1→M5 + pipeline)
**Ngày:** 2026-09-08 · **Test set:** 20 câu (`test_set.json`)

---

## RAGAS Scores

| Metric | Naive Baseline | Production v1 (return child) | Production v2 (child→parent) | Δ (v2 − baseline) |
|--------|---------------|------------------------------|-------------------------------|-------------------|
| Faithfulness | 0.8667 | 0.6250 | **0.8938** | **+0.0271** |
| Answer Relevancy | 0.7130 | 0.5983 | **0.7838** | **+0.0708** |
| Context Precision | 0.9250 | 0.9458 | **0.9750** | **+0.0500** |
| Context Recall | 0.9250 | 0.8250 | **0.9500** | **+0.0250** |

> **Baseline** = paragraph chunking (~500 ký tự) + dense-only, top-3, không rerank/enrichment.
> **Production v1** = hierarchical child 256 + enrichment + hybrid + rerank, **trả về đúng child đã retrieve**.
> **Production v2** = như v1 nhưng **retrieve child → trả về parent (2048)** khi lắp context cho LLM.
> → v2: **cả 4 metric ≥ 0.78**, faithfulness ≥ 0.85.

---

## Bài học lớn nhất: v1 regression và cách sửa

`context_precision`/`context_recall` của v1 vẫn cao (0.95 / 0.83) → **retrieval tốt**, nhưng `faithfulness` tụt còn 0.63: **5/20 câu trả về "Không tìm thấy."** dù RAGAS xác nhận context liên quan.

**Root cause:** child chunk 256 ký tự **cắt giữa câu**. Câu chứa con số/điều kiện cần trả lời bị tách sang chunk kế và không lọt top-3 sau rerank → LLM (prompt siết "chỉ dùng context") từ chối trả lời.
Pipeline v1 làm "retrieve child" nhưng **quên "return parent"** — mất toàn bộ lợi ích hierarchical.

**Fix (v2):** giữ `parent_map`; sau rerank, `_expand_to_parents()` map mỗi child → text parent rồi mới đưa cho LLM. Kết quả: 5 câu "Không tìm thấy" biến mất, faithfulness 0.63 → 0.89.

---

## Bottom-5 Failures (Production v2) + Error Tree

### #1 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm **và** lương trong khoảng nào?"
- **Expected:** 18 ngày phép (15 + 3) **và** dải lương bậc Senior theo bảng lương 2024.
- **Got:** "…được nghỉ 18 ngày phép năm. Về lương, không có thông tin cụ thể."
- **Worst metric:** answer_relevancy 0.0 · faithfulness 0.75 · context_precision 1.0 · context_recall 0.5
- **Error Tree:**
  1. Output sai? → **Một nửa** — phần phép đúng, phần lương thiếu.
  2. Context đúng? → Chỉ lấy được chunk "nghỉ phép", **không** lấy chunk "bảng lương Senior" (recall 0.5).
  3. Query OK? → Câu hỏi **multi-hop 2 chủ đề** — single-vector retrieval lấy trúng 1 chủ đề trội.
  4. Fix ở bước: **retrieval strategy** cho multi-hop.
- **Suggested fix:** query decomposition (tách "phép" / "lương" thành 2 sub-query, gộp context); tăng `HYBRID_TOP_K`; multi-query retrieval.

### #2 — "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **Expected:** Phạt 2%/tháng trên số chưa hoàn ứng, quá hạn 5 ngày → tiền phạt tính theo tỷ lệ.
- **Got:** LLM **tự tính ra một con số** (15tr × 2% …) — đúng công thức nhưng con số không có literal trong context.
- **Worst metric:** faithfulness 0.125 · answer_relevancy 0.79 · context_precision 1.0 · context_recall 1.0
- **Error Tree:** Output "hợp lý" → **Context hoàn hảo (P=R=1.0)** → Query OK → Fix ở **generation**: RAGAS phạt faithfulness vì answer chứa **phép tính suy diễn** (con số mới) không xuất hiện nguyên văn trong context.
- **Suggested fix:** prompt "chỉ trích dẫn công thức/tỷ lệ có trong tài liệu, KHÔNG tự tính toán số cụ thể trừ khi tài liệu đã tính sẵn"; hoặc tách bước tính toán ra khỏi bước trích dẫn.

### #3 — "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"
- **Expected:** +1 ngày cho mỗi 5 năm thâm niên (chính sách nghỉ phép năm hiện hành).
- **Got:** "…từ 3 năm trở lên, +1 ngày cho mỗi 3 năm… theo chính sách 2024."
- **Worst metric:** context_precision 0.5 · faithfulness 1.0 · answer_relevancy 0.78 · context_recall 1.0
- **Error Tree:** Output sai số → **Context lẫn 2 phiên bản** (`nghi_phep_nam_v2023` "5 năm" và `nghi_phep_nam_v2024`) → parent expansion kéo cả 2 vào → LLM chọn nhầm bản.
- **Suggested fix:** dùng `auto_metadata` (M5 `extract_metadata`) gắn `version`/ngày hiệu lực → lọc/ưu tiên bản mới nhất trước khi generate; hoặc xoá tài liệu superseded khỏi index.

### #4 — "Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?"
- **Expected:** Hoàn trả theo tỷ lệ cam kết còn lại (thường 24 tháng) → không phải 100%.
- **Got:** "Phải hoàn trả 100% chi phí = 25 triệu."
- **Worst metric:** faithfulness 0.5 · answer_relevancy 0.80 · context_precision 1.0 · context_recall 1.0
- **Error Tree:** Output sai → Context đúng (P=R=1.0) → Query OK → Fix ở **generation**: câu hỏi cần **suy luận điều kiện theo mốc thời gian** (8/24 tháng); LLM bỏ qua điều khoản giảm trừ theo thời gian phục vụ.
- **Suggested fix:** few-shot ví dụ tính hoàn trả theo tỷ lệ; prompt yêu cầu liệt kê điều kiện áp dụng trước khi kết luận.

### #5 — "Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"
- **Expected:** Ngưỡng 30tr → cấp phê duyệt tương ứng (Trưởng phòng/Giám đốc) + xác nhận cấu hình từ CNTT.
- **Got:** "Giám đốc phòng ban phê duyệt; cần CNTT xác nhận cấu hình." (thiếu/nhầm ngưỡng cụ thể)
- **Worst metric:** faithfulness 0.5 · answer_relevancy 0.82 · context_precision 1.0 · context_recall 1.0
- **Error Tree:** Output sai một phần → Context đúng (P=R=1.0) → Query OK → Fix ở **generation**: câu hỏi **multi-part** (ai duyệt + cần gì) + cần map "30tr" vào đúng bậc trong bảng ngưỡng.
- **Suggested fix:** prompt yêu cầu trả lời từng ý; structure-aware chunking giữ nguyên bảng ngưỡng phê duyệt (`mua_sam.md`).

---

## Diagnostic (Error) Tree — tổng hợp

```
Answer sai / thiếu?
├─ Context precision & recall cao (≈1.0)?  → lỗi ở GENERATION
│   ├─ Answer chứa phép tính / con số suy diễn   → #2 #4  → prompt cấm tự tính, few-shot
│   └─ Câu hỏi multi-part, answer bỏ sót ý        → #5     → prompt trả lời từng ý
├─ Context recall thấp (0.5)?              → lỗi ở RETRIEVAL
│   └─ Câu hỏi multi-hop (2 chủ đề)               → #1     → query decomposition / multi-query
└─ Context precision thấp (0.5)?           → lỗi ở DATA/INDEX
    └─ Lẫn 2 phiên bản chính sách                 → #3     → metadata versioning, bỏ doc superseded
```

**Đã sửa được ở lab này:** boundary của child chunk (v1→v2, child→parent).
**Còn lại (ngoài phạm vi 2h):** numeric reasoning trong generation, multi-hop retrieval, version filtering.

---

## Case Study cho presentation

**Question:** "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?" (#3)

**Error Tree walkthrough:**
1. Output đúng? → Không — trả "3 năm" thay vì "5 năm".
2. Context đúng? → Precision 0.5: parent expansion kéo cả `nghi_phep_nam_v2023` (5 năm) lẫn `v2024` vào context.
3. Query rewrite OK? → Có.
4. Fix ở bước: **DATA/INDEX** — lọc version bằng `auto_metadata`, hoặc loại tài liệu superseded.

**Nếu có thêm 1 giờ:**
- Bật lọc `auto_metadata["version"]` / ngày hiệu lực trong M5 để loại chính sách cũ (#3).
- Prompt generation: cấm LLM tự tính con số, chỉ trích công thức (#2 #4).
- Query decomposition cho câu multi-hop (#1) + structure-aware cho file có bảng (#5).
