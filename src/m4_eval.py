from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation. Cần OPENAI_API_KEY và Python 3.11+."""
    import math

    def _f(v):
        try:
            v = float(v)
            return 0.0 if math.isnan(v) else v
        except (TypeError, ValueError):
            return 0.0

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                faithfulness=_f(row.get("faithfulness", 0.0)),
                answer_relevancy=_f(row.get("answer_relevancy", 0.0)),
                context_precision=_f(row.get("context_precision", 0.0)),
                context_recall=_f(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]

        def _avg(attr):
            vals = [getattr(p, attr) for p in per_question]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "faithfulness": _avg("faithfulness"),
            "answer_relevancy": _avg("answer_relevancy"),
            "context_precision": _avg("context_precision"),
            "context_recall": _avg("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — câu trả lời không bám context",
                         "Siết prompt 'chỉ dùng context', giảm temperature"),
        "context_recall": ("Thiếu chunk liên quan trong context",
                           "Cải thiện chunking hoặc tăng trọng số BM25/top_k"),
        "context_precision": ("Quá nhiều chunk nhiễu trong context",
                             "Thêm reranking hoặc metadata filter"),
        "answer_relevancy": ("Câu trả lời lệch khỏi câu hỏi",
                            "Cải thiện prompt template / câu hỏi"),
    }
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    scored = []
    for r in eval_results:
        metrics = {m: getattr(r, m) for m in metric_names}
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "worst_score": round(metrics[worst_metric], 4),
            "avg_score": round(avg, 4),
            "metrics": {m: round(v, 4) for m, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
