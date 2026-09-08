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
    """Run RAGAS evaluation."""
    def _mean(values):
        vals = [v for v in values if v == v]  # drop NaN
        return round(sum(vals) / len(vals), 4) if vals else 0.0

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

        def col(row, name):
            try:
                return float(row[name])
            except (KeyError, TypeError, ValueError):
                return 0.0

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                faithfulness=col(row, "faithfulness"),
                answer_relevancy=col(row, "answer_relevancy"),
                context_precision=col(row, "context_precision"),
                context_recall=col(row, "context_recall"),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": _mean([r.faithfulness for r in per_question]),
            "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
            "context_precision": _mean([r.context_precision for r in per_question]),
            "context_recall": _mean([r.context_recall for r in per_question]),
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
                         "Siết prompt ('chỉ dùng context'), giảm temperature"),
        "context_recall": ("Thiếu chunk liên quan trong context",
                           "Cải thiện chunking hoặc tăng trọng số BM25 / top_k"),
        "context_precision": ("Quá nhiều chunk nhiễu trong context",
                              "Thêm reranking hoặc lọc theo metadata"),
        "answer_relevancy": ("Câu trả lời lạc đề so với câu hỏi",
                             "Cải thiện prompt template, thêm few-shot"),
    }

    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / 4
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
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
