# Failure Cluster Analysis — Phase A

**Sinh viên:** Do Huyen
**Ngày:** 2026-06-30

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.908 | 0.386 | 0.700 |
| answer_relevancy | 0.710 | 0.633 | 0.605 |
| context_precision | 0.933 | 0.933 | 0.858 |
| context_recall | 0.850 | 0.746 | 0.650 |
| **avg_score** | **0.850** | **0.674** | **0.703** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0... | 0.125 | faithfulness |
| 2 | multi_hop | Nhân viên Junior P1 có lương cơ bản 12 triệu vừa vào thử việc... | 0.313 | faithfulness |
| 3 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp... | 0.333 | faithfulness |
| 4 | multi_hop | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và chính thức | 0.375 | faithfulness |
| 5 | multi_hop | Nhân viên cấp Lead nhận tổng phụ cấp hàng tháng... | 0.396 | answer_relevancy |
| 6 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.396 | faithfulness |
| 7 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân...? | 0.417 | faithfulness |
| 8 | adversarial | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép? | 0.417 | faithfulness |
| 9 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.500 | answer_relevancy |
| 10 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con? | 0.500 | faithfulness |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 3 | 15 | 3 | 21 |
| answer_relevancy | 12 | 3 | 0 | 15 |
| context_precision | 2 | 0 | 1 | 3 |
| context_recall | 3 | 2 | 6 | 11 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** multi_hop (avg_score thấp nhất: 0.674)
**Dominant metric:** faithfulness (21/50 câu — 42%)

**Lý do phân tích:**

> Multi-hop là distribution có avg_score thấp nhất (0.674) dù factual có cùng số failure count (20). Nguyên nhân chính là faithfulness trên multi_hop cực kỳ thấp (0.386) — 15/20 câu multi_hop có worst_metric là faithfulness. Điều này cho thấy LLM hallucinate nghiêm trọng khi phải tổng hợp thông tin từ nhiều tài liệu. Corpus HR tiếng Việt có nhiều tài liệu overlap về chủ đề (ví dụ: nghỉ phép có cả v2023 và v2024), khiến LLM dễ trộn lẫn quy định giữa các phiên bản. Context_recall trên adversarial cũng thấp (0.650, 6/10 câu worst=context_recall), cho thấy retrieval không lấy được đúng tài liệu cho các câu bẫy version conflict.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating — trộn quy định giữa các tài liệu, bịa số | Thêm explicit version check trong prompt: "Ưu tiên policy mới nhất (v2024/v2.0). Nếu có conflict, dùng phiên bản hiện hành." |
| context_recall | Thiếu chunk liên quan — retrieval không lấy đúng tài liệu cho version-specific queries | Thêm metadata filter theo version + tăng top_k cho multi_hop queries |
| context_precision | Quá nhiều chunk nhiễu — các tài liệu overlap về chủ đề | Thêm reranking chuyên sâu + deduplicate chunks từ nhiều phiên bản policy |
| answer_relevancy | Câu trả lời lệch khỏi câu hỏi | Siết prompt: "Trả lời đúng trọng tâm câu hỏi, không thêm thông tin ngoài lề" |

---

## 6. Nhận xét về Adversarial Distribution

> Adversarial có avg_score (0.703) thấp hơn factual (0.850) đúng như kỳ vọng — pipeline có bị ảnh hưởng bởi version conflicts. Các câu adversarial trong bottom 10 (#6, #7, #8) đều liên quan đến version conflicts (v2023 vs v2024, v1.0 vs v2.0). Pipeline thường retrieve cả 2 tài liệu conflicting và LLM không xác định được đâu là phiên bản hiện hành, dẫn đến hallucination. Cần cải thiện version awareness trong retrieval và prompt. Bonus Phase A đạt được vì adversarial avg_score < factual avg_score (0.703 < 0.850).
