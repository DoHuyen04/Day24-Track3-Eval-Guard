from __future__ import annotations

"""Production RAG Pipeline — ghép M1+M2+M3+M4+M5 + latency breakdown."""

import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline() -> tuple[HybridSearch, CrossEncoderReranker, dict]:
    """Build production RAG pipeline. Trả về (search, reranker, build_timings_ms)."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)
    timings = {}

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    timings["1_chunking"] = (time.time() - t0) * 1000
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({timings['1_chunking']/1000:.1f}s)", flush=True)

    # Step 2: Enrichment (M5) — có cache để tránh gọi lại API khi retry pipeline.
    t0 = time.time()
    cache_path = "reports/enriched_chunks.json"
    if os.path.exists(cache_path):
        print(f"\n[2/4] Loading enriched chunks from cache ({cache_path})...", flush=True)
        with open(cache_path, encoding="utf-8") as f:
            all_chunks = json.load(f)
        print(f"  ✓ Loaded {len(all_chunks)} enriched chunks from cache", flush=True)
    else:
        print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
        enriched = enrich_chunks(all_chunks)
        if enriched:
            all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)
            print(f"  ✓ Enriched {len(enriched)} chunks, cached ({(time.time()-t0):.1f}s)", flush=True)
        else:
            print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    timings["2_enrichment"] = (time.time() - t0) * 1000

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    timings["3_indexing"] = (time.time() - t0) * 1000
    print(f"  ✓ Indexed ({timings['3_indexing']/1000:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    timings["4_reranker_load"] = (time.time() - t0) * 1000
    print(f"  ✓ Reranker ready ({timings['4_reranker_load']/1000:.1f}s)", flush=True)

    timings["_num_chunks"] = len(all_chunks)
    return search, reranker, timings


# Prompt siết chặt: chỉ dùng context, không suy diễn/tính toán ngoài context,
# không trộn quy định giữa các tài liệu → nâng faithfulness.
# Prompt gốc (đơn giản) cho faithfulness/relevancy tốt nhất; giữ temperature=0 để tái lập.
SYSTEM_PROMPT = "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"


def _llm_answer(query: str, contexts: list[str]) -> str:
    """Gọi LLM trả lời dựa trên context (prompt siết, temperature=0)."""
    from config import OPENAI_API_KEY
    if not (OPENAI_API_KEY and contexts):
        return contexts[0] if contexts else "Không tìm thấy thông tin."
    try:
        from openai import OpenAI
        client = OpenAI()
        context_str = "\n\n".join(contexts)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  ⚠️  LLM generation failed: {e}", flush=True)
        return contexts[0]


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              query_timings: list | None = None) -> tuple[str, list[str]]:
    """Run single query end-to-end (retrieval → rerank → LLM)."""
    t = time.perf_counter()
    results = search.search(query)
    t_search = (time.perf_counter() - t) * 1000

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    t = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    t_rerank = (time.perf_counter() - t) * 1000
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    t = time.perf_counter()
    answer = _llm_answer(query, contexts)
    t_llm = (time.perf_counter() - t) * 1000

    if query_timings is not None:
        query_timings.append({"search": t_search, "rerank": t_rerank, "llm": t_llm})
    return answer, contexts


def _free_dense_encoder(search: HybridSearch) -> None:
    """Giải phóng SentenceTransformer encoder khỏi RAM (giữa 2 pass eval)."""
    import gc
    search.dense._encoder = None
    gc.collect()
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def save_latency_report(build_timings: dict, query_timings: list[dict], ragas_ms: float,
                        path: str = "reports/latency_report.md") -> None:
    """Xuất bảng latency breakdown ra Markdown + JSON."""
    n = len(query_timings) or 1
    avg_search = sum(q["search"] for q in query_timings) / n
    avg_rerank = sum(q["rerank"] for q in query_timings) / n
    avg_llm = sum(q["llm"] for q in query_timings) / n
    avg_query = avg_search + avg_rerank + avg_llm

    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Latency Breakdown — Production RAG Pipeline",
        "",
        f"- Số chunks index: **{build_timings.get('_num_chunks', '?')}**",
        f"- Số query đo: **{len(query_timings)}**",
        "",
        "## Build (one-time)",
        "",
        "| Bước | Thời gian |",
        "|------|-----------|",
        f"| 1. Chunking (M1) | {build_timings.get('1_chunking', 0)/1000:.2f} s |",
        f"| 2. Enrichment (M5) | {build_timings.get('2_enrichment', 0)/1000:.2f} s |",
        f"| 3. Indexing BM25+Dense (M2) | {build_timings.get('3_indexing', 0)/1000:.2f} s |",
        f"| 4. Load reranker (M3) | {build_timings.get('4_reranker_load', 0)/1000:.2f} s |",
        "",
        "## Per-query (trung bình)",
        "",
        "| Bước | Latency (ms) | % |",
        "|------|-------------:|---:|",
        f"| Retrieval (M2 BM25+Dense+RRF) | {avg_search:.1f} | {avg_search/avg_query*100:.1f}% |",
        f"| Rerank (M3 cross-encoder) | {avg_rerank:.1f} | {avg_rerank/avg_query*100:.1f}% |",
        f"| LLM answer (gpt-4o-mini) | {avg_llm:.1f} | {avg_llm/avg_query*100:.1f}% |",
        f"| **Tổng / query** | **{avg_query:.1f}** | **100%** |",
        "",
        f"## Evaluation\n\n- RAGAS (4 metrics × {len(query_timings)} câu): **{ragas_ms/1000:.1f} s**",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(path.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump({
            "build_ms": build_timings,
            "per_query_avg_ms": {"search": avg_search, "rerank": avg_rerank,
                                 "llm": avg_llm, "total": avg_query},
            "ragas_ms": ragas_ms,
        }, f, ensure_ascii=False, indent=2)
    print(f"Latency report saved to {path}", flush=True)


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, build_timings: dict):
    """Run evaluation on test set theo 2-pass để giảm peak RAM:
    Pass 1 chỉ retrieval (encoder), giải phóng encoder; Pass 2 rerank + LLM."""
    test_set = load_test_set()
    questions = [item["question"] for item in test_set]
    ground_truths = [item["ground_truth"] for item in test_set]
    query_timings: list[dict] = []

    # --- Pass 1: Retrieval (chỉ dense encoder + BM25 trong RAM) ---
    print(f"\n[Eval/pass1] Retrieval cho {len(test_set)} queries...", flush=True)
    retrieved_docs: list[list[dict]] = []
    search_times: list[float] = []
    for i, q in enumerate(questions):
        t = time.perf_counter()
        results = search.search(q)
        search_times.append((time.perf_counter() - t) * 1000)
        retrieved_docs.append([{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results])
        print(f"  [{i+1}/{len(test_set)}] retrieved", flush=True)

    # Giải phóng encoder trước khi load reranker → tránh giữ 2 model cùng lúc.
    _free_dense_encoder(search)
    print("  ✓ Đã giải phóng dense encoder khỏi RAM", flush=True)

    # --- Pass 2: Rerank (M3) + LLM answer ---
    print(f"\n[Eval/pass2] Rerank + LLM cho {len(test_set)} queries...", flush=True)
    answers, all_contexts = [], []
    for i, (q, docs) in enumerate(zip(questions, retrieved_docs)):
        t = time.perf_counter()
        reranked = reranker.rerank(q, docs, top_k=RERANK_TOP_K)
        t_rerank = (time.perf_counter() - t) * 1000
        contexts = [r.text for r in reranked] if reranked else [d["text"] for d in docs[:3]]
        t = time.perf_counter()
        answer = _llm_answer(q, contexts)
        t_llm = (time.perf_counter() - t) * 1000
        answers.append(answer)
        all_contexts.append(contexts)
        query_timings.append({"search": search_times[i], "rerank": t_rerank, "llm": t_llm})
        print(f"  [{i+1}/{len(test_set)}] {q[:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    ragas_ms = (time.time() - t0) * 1000
    print(f"  ✓ RAGAS done ({ragas_ms/1000:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    save_latency_report(build_timings, query_timings, ragas_ms)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, build_timings = build_pipeline()
    evaluate_pipeline(search, reranker, build_timings)
    print(f"\nTotal: {time.time() - start:.1f}s")
