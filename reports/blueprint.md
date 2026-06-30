# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Do Huyen
**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~20ms P95, có cache)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~1760ms P95 với fallback LLM)
[NeMo Input Rail + Fallback]
    │ block if: off-topic / jailbreak / prompt injection
    │ Nếu NeMo empty → fallback direct GPT-4o-mini call
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Từ kết quả measure_p95_latency() — có caching Presidio + NeMo fallback active)*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 11.3 | 19.5 | ~20 | <10ms |
| NeMo Input (+ fallback LLM) | 985.6 | 1757.2 | ~2000 | <300ms |
| RAG Pipeline | ~1500 | ~2000 | ~2500 | <2000ms |
| NeMo Output Rail | ~250 | ~300 | ~350 | <300ms |
| **Total Guard** | 996.3 | **1773.0** | ~2000 | **<500ms** |

**Budget OK?** [ ] Yes / [x] No
**Comment:** NeMo fallback dùng direct GPT-4o-mini call (~1-2s/gọi) — chậm hơn NeMo native (~300ms) nhưng đảm bảo hoạt động ổn định. Presidio rất nhanh với cache (<20ms). Để đạt budget 500ms: (1) fix NeMo native API để bỏ fallback, (2) chạy Presidio + NeMo song song, (3) cache kết quả guard cho input tương tự, (4) dùng model nhỏ hơn cho guard (GPT-4o-mini → distilled model).

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 2000ms | Investigate NeMo/fallback |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | factual=0.853, multi_hop=0.676, adversarial=0.695 |
| Worst metric | faithfulness (overall ~0.666, multi_hop chỉ 0.390 với 16/20 câu) |
| Dominant failure distribution | multi_hop (avg_score thấp nhất 0.676) |
| Dominant failure metric | faithfulness (22/50 câu, 44%) |
| Cohen's κ (LLM judge vs human) | **0.583 (moderate)** — gần đạt substantial (0.6). Dao động 0.58–0.80 tùy lần chạy |
| Position bias rate | 50% — swap-and-average phát hiện hiệu quả |
| Verbosity bias | 100% — winner luôn dài hơn loser (là tín hiệu chất lượng) |
| Adversarial pass rate | **19/20 (95.0%)** — vượt bonus threshold 90% |
| Guard P95 latency (có cache) | Presidio 19.5ms + NeMo/fallback 1757ms = **1773ms** |
| Guard P95 latency (NeMo native, lần đầu) | Presidio 6144ms (cold start) + NeMo 300ms = ~6444ms |

---

## Nhận xét & Cải tiến

### Điểm mạnh
- **Presidio PII detection hoạt động xuất sắc:** Phát hiện chính xác VN_CCCD (12 số, score 0.9), VN_PHONE (0[3-9]xxxxxxxx, score 0.9), EMAIL. Block 4/5 pii_injection inputs. Module-level caching giảm latency từ 6144ms → 19ms.
- **Adversarial defense 95%:** 19/20 inputs bị chặn — chỉ 1 input (id=5: yêu cầu PII người khác) lọt qua do không chứa PII trực tiếp và LLM fallback không phát hiện được.
- **Swap-and-average hiệu quả:** Phát hiện 50% position inconsistency, chứng minh swap là cần thiết.
- **Score-based judge > winner-based:** Cải thiện κ từ -0.207 → 0.583 bằng cách dùng quality score thay vì binary winner.

### Điểm cần cải thiện
- **Multi-hop faithfulness (0.390):** LLM hallucinate khi tổng hợp nhiều tài liệu — cần metadata filter chọn đúng policy version + prompt siết chặt hơn.
- **NeMo Guardrails không ổn định:** Native API thỉnh thoảng trả empty → cần fallback. Fallback chậm (1.7s P95). Nên upgrade NeMo hoặc thay bằng lightweight classifier.
- **Cohen's κ dao động (0.58–0.80):** gpt-4o-mini non-deterministic → nên dùng GPT-4o hoặc ensemble 3 judges lấy majority vote.
- **Latency vượt budget:** 1773ms P95 (chủ yếu do fallback LLM). Nếu NeMo native hoạt động, ước tính ~320ms.

### Nếu deploy production thực sự
1. Thay NeMo Colang bằng custom Python guard (regex + finetuned classifier) — nhanh hơn, ổn định hơn
2. Semantic cache cho guard decisions — input tương tự → reuse kết quả
3. Vietnamese embedding model thay BGE-M3 — cải thiện context_recall
4. Metadata filter để chọn đúng policy version — giảm version conflict hallucination
5. Circuit breaker pattern: nếu NeMo fail 3 lần liên tiếp → tự động chuyển sang fallback + alert
