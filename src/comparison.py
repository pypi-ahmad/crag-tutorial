"""Paired naive-RAG versus CRAG evaluation over identical retrieval results."""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from src.agnes_client import count_requests
from src.cache import fingerprint, read_json, write_json
from src.config import CACHE_DIR, MODEL_NAME
from src.crag import ABSTENTION, generate_answer
from src.pipeline import answer_contains_gold, run_crag
from src.qdrant_store import DEFAULT_COLLECTION, INDEX_MANIFEST
from src.retrieve import search

COMPARISON_VERSION = "naive-vs-crag-v1"
REVIEW_CORRECTNESS = {"correct", "partially_correct", "incorrect", "abstain"}
REVIEW_SUPPORT = {"fully_supported", "partially_supported", "unsupported", "no_answer"}
REVIEW_PREFERENCES = {"naive_better", "crag_better", "tie_both_good", "tie_both_inadequate"}
REVIEW_ERRORS = {"retrieval_gap", "distractor_use", "evaluator_error", "refinement_loss",
                 "generation_error", "string_metric_mismatch", "appropriate_abstention",
                 "unsupported_generation", "reference_conflict", "none"}


def run_naive_rag(question: str, docs: list[dict[str, Any]], model: str = MODEL_NAME) -> dict[str, Any]:
    """Generate directly from complete retrieved passages, without correction."""
    evidence = [doc["text"] for doc in docs if doc.get("text", "").strip()]
    with count_requests() as counter:
        answer = generate_answer(question, evidence, model=model)
    return {
        "answer": answer,
        "evidence": evidence,
        "evidence_titles": [doc.get("title", "") for doc in docs if doc.get("text", "").strip()],
        "n_llm_calls": counter["attempts"],
    }


def comparison_fingerprint(row: dict[str, Any]) -> str:
    """Bind a manual review to the exact question, evidence, and two answers."""
    return fingerprint({
        "id": row["id"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "hits": [(hit["title"], hit["text"]) for hit in row["hits"]],
        "naive_answer": row["naive"]["answer"],
        "naive_evidence": row["naive"]["evidence"],
        "crag_answer": row["crag"]["answer"],
        "crag_strips": row["crag"]["strips"],
    })


def _comparison_row(record: dict[str, Any], hits: list[dict[str, Any]], naive: dict[str, Any], crag: dict[str, Any]) -> dict[str, Any]:
    gold_titles = set(record["gold_titles"])
    hit_titles = {hit["title"] for hit in hits}
    row = {
        "id": record["id"],
        "type": record.get("question_type", "unknown"),
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "gold_titles": record["gold_titles"],
        "hits": [{
            "title": hit["title"],
            "text": hit["text"],
            "score": hit["score"],
            "is_gold": hit.get("payload", {}).get("is_gold"),
        } for hit in hits],
        "gold_title_recall_at_k": len(gold_titles & hit_titles) / len(gold_titles) if gold_titles else None,
        "naive": {
            **naive,
            "answer_contains_gold": answer_contains_gold(naive["answer"], record["gold_answer"]),
            "abstained": naive["answer"].strip() == ABSTENTION,
        },
        "crag": {
            "answer": crag.answer,
            "action": crag.action,
            "evaluations": [{key: item[key] for key in ("title", "score", "label", "why")} for item in crag.evaluations],
            "kept_titles": crag.kept_titles,
            "strips": crag.strips,
            "answer_contains_gold": answer_contains_gold(crag.answer, record["gold_answer"]),
            "abstained": crag.answer.strip() == ABSTENTION,
            "n_llm_calls": crag.n_llm_calls,
        },
    }
    row["comparison_fingerprint"] = comparison_fingerprint(row)
    return row


def run_comparison_50(
    records: list[dict[str, Any]],
    k: int = 3,
    allow_web: bool = False,
    max_retries: int = 3,
    backoff_delay: float = 2,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Run a resumable paired comparison on the first 50 records."""
    if len(records) < 50:
        raise ValueError("The comparison requires at least 50 records.")
    if type(k) is not int or k < 1 or max_retries < 1:
        raise ValueError("k and max_retries must be positive integers.")
    settings = {
        "version": COMPARISON_VERSION,
        "model": MODEL_NAME,
        "k": k,
        "allow_web": allow_web,
        "collection": collection_name,
        "index": read_json(INDEX_MANIFEST),
    }
    selected = records[:50]
    root = CACHE_DIR / "naive_vs_crag" / fingerprint([settings, selected])[:16]
    rows = []
    for index, record in enumerate(selected):
        path = root / f"{record['id']}.json"
        cached = read_json(path)
        if (isinstance(cached, dict) and cached.get("settings") == settings
                and isinstance(cached.get("row"), dict)
                and cached["row"].get("id") == record["id"]
                and cached["row"].get("comparison_fingerprint") == comparison_fingerprint(cached["row"])):
            row = cached["row"]
            row["naive"]["n_llm_calls"] = 0
            row["crag"]["n_llm_calls"] = 0
            row["cache_hit"] = True
        else:
            for attempt in range(max_retries):
                try:
                    hits = search(record["question"], k=k, question_id=record["id"], collection_name=collection_name)
                    naive = run_naive_rag(record["question"], hits)
                    crag = run_crag(record["question"], record["id"], k=k, allow_web=allow_web,
                                    collection_name=collection_name, docs=hits)
                    row = _comparison_row(record, hits, naive, crag)
                    row["cache_hit"] = False
                    write_json(path, {"settings": settings, "row": row})
                    break
                except Exception as error:
                    write_json(root / f"{record['id']}.failure.json", {
                        "id": record["id"],
                        "attempt": attempt + 1,
                        "exception_type": type(error).__name__,
                        "traceback": traceback.format_tb(error.__traceback__),
                    })
                    print(f"Comparison failed for {record['id']}: {type(error).__name__}; checkpoint saved.", flush=True)
                    if attempt + 1 == max_retries:
                        raise RuntimeError(f"Comparison stopped at {record['id']}; completed rows are checkpointed.") from None
                    time.sleep(min(30, backoff_delay * 2 ** attempt))
        failure_path = root / f"{record['id']}.failure.json"
        failure = read_json(failure_path)
        if isinstance(failure, dict) and not failure.get("resolved"):
            write_json(failure_path, {**failure, "resolved": True})
        rows.append(row)
        calls = row["naive"]["n_llm_calls"] + row["crag"]["n_llm_calls"]
        print(f"comparison: {index + 1}/50; cached={row['cache_hit']}; requests={calls}", flush=True)
    return rows


def automated_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive, non-human metrics for the paired results."""
    if len(rows) != 50:
        raise ValueError("Expected exactly 50 paired rows.")
    return {
        "n_questions": 50,
        "gold_title_recall@3": sum(row["gold_title_recall_at_k"] for row in rows) / 50,
        "naive_substring_match": sum(row["naive"]["answer_contains_gold"] for row in rows) / 50,
        "crag_substring_match": sum(row["crag"]["answer_contains_gold"] for row in rows) / 50,
        "naive_abstention_rate": sum(row["naive"]["abstained"] for row in rows) / 50,
        "crag_abstention_rate": sum(row["crag"]["abstained"] for row in rows) / 50,
        "naive_llm_calls": sum(row["naive"]["n_llm_calls"] for row in rows),
        "crag_llm_calls": sum(row["crag"]["n_llm_calls"] for row in rows),
        "cached_rows": sum(row["cache_hit"] for row in rows),
    }


def load_manual_reviews(rows: list[dict[str, Any]], path: Path | str) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate completed human judgements and identify changed or missing rows."""
    payload = read_json(path)
    reviews = payload.get("reviews", []) if isinstance(payload, dict) else []
    by_id = {review.get("id"): review for review in reviews if isinstance(review, dict)}
    valid, stale = [], []
    required = {"id", "comparison_fingerprint", "naive_correctness", "naive_support",
                "crag_correctness", "crag_support", "preference", "error_types", "rationale"}
    for row in rows:
        review = by_id.get(row["id"])
        okay = (isinstance(review, dict) and required <= review.keys()
                and review["comparison_fingerprint"] == comparison_fingerprint(row)
                and review["naive_correctness"] in REVIEW_CORRECTNESS
                and review["crag_correctness"] in REVIEW_CORRECTNESS
                and review["naive_support"] in REVIEW_SUPPORT
                and review["crag_support"] in REVIEW_SUPPORT
                and review["preference"] in REVIEW_PREFERENCES
                and isinstance(review["error_types"], list) and bool(review["error_types"])
                and all(error in REVIEW_ERRORS for error in review["error_types"])
                and isinstance(review["rationale"], str) and bool(review["rationale"].strip()))
        if okay:
            valid.append(review)
        else:
            stale.append(row["id"])
    return valid, stale
