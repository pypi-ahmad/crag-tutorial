"""CRAG orchestration and resumable descriptive evaluation."""
import time
import traceback
import pandas as pd
from src.agnes_client import count_requests
from src.cache import fingerprint, read_json, write_json
from src.config import CACHE_DIR, CACHE_VERSION, MODEL_NAME
from src.crag import evaluate_documents, decide_action, refine_strips, generate_answer
from src.retrieve import search
from src.qdrant_store import DEFAULT_COLLECTION, INDEX_MANIFEST

class CRAGResult(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

def run_crag(question, question_id=None, k=3, upper=0.7, lower=0.3, allow_web=False,
             collection_name=DEFAULT_COLLECTION, model=MODEL_NAME, docs=None):
    with count_requests() as counter:
        hits = docs if docs is not None else search(question, k, question_id, collection_name)
        evaluations = evaluate_documents(question, hits, model=model)
        action = decide_action(evaluations, upper, lower)
        selected = [] if action == "Incorrect" else [e["doc"] for e in evaluations if e["score"] > lower]
        strips, kept_titles = [], []
        for doc in selected:
            kept = refine_strips(question, doc, model=model)
            if kept:
                kept_titles.append(doc["title"])
                strips.extend(kept)
        web_sources = []
        if allow_web and action != "Correct":
            from src.search import ExternalWebSearcher
            report = ExternalWebSearcher(model_name=model).search_and_extract(question)
            for result in report.results:
                kept = refine_strips(question, {"text": result.snippet}, model=model)
                if kept:
                    strips.extend(kept)
                    kept_titles.append(result.title)
                    web_sources.append(result.url)
        strips = list(dict.fromkeys(strips))
        answer = generate_answer(question, strips, model=model)
    return CRAGResult(hits=hits, evaluations=evaluations, action=action, strips=strips,
                      answer=answer, n_llm_calls=counter["attempts"], kept_titles=list(dict.fromkeys(kept_titles)), web_sources=web_sources)

def answer_contains_gold(answer, gold):
    """Literal nonempty, one-way, case-insensitive substring proxy; not accuracy."""
    return bool(gold.strip()) and gold.strip().casefold() in answer.casefold()

def _row(record, result):
    gold = set(record["gold_titles"])
    hit_titles = {hit["title"] for hit in result.hits}
    return {"id": record["id"], "type": record.get("question_type", "unknown"),
            "question": record["question"], "gold_answer": record["gold_answer"],
            "generated_answer": result.answer, "action": result.action,
            "gold_in_hits": bool(gold & hit_titles),
            "gold_title_recall_at_k": len(gold & hit_titles) / len(gold) if gold else None,
            "gold_in_kept_titles": bool(gold & set(result.kept_titles)),
            "answer_contains_gold": answer_contains_gold(result.answer, record["gold_answer"]),
            "n_llm_calls": result.n_llm_calls, "kept_titles": result.kept_titles,
            "kept_strips": result.strips, "scores": [e["score"] for e in result.evaluations],
            "evaluations": [{key: e[key] for key in ("title", "score", "label", "why")} for e in result.evaluations]}

def _benchmark(records, name, k, allow_web, max_retries, backoff_delay):
    if not records or max_retries < 1:
        raise ValueError("Benchmark requires records and at least one attempt.")
    settings = {"version": CACHE_VERSION, "model": MODEL_NAME, "k": k, "allow_web": allow_web, "upper": .7, "lower": .3,
                "index": read_json(INDEX_MANIFEST)}
    root = CACHE_DIR / name / fingerprint([settings, records])[:16]
    rows = []
    for index, record in enumerate(records):
        path = root / (record["id"] + ".json")
        cached = read_json(path)
        required = {"id", "generated_answer", "gold_title_recall_at_k", "gold_in_hits", "answer_contains_gold", "action", "evaluations", "kept_strips", "n_llm_calls"}
        if (isinstance(cached, dict) and cached.get("settings") == settings
                and isinstance(cached.get("row"), dict) and required <= cached["row"].keys()
                and cached["row"]["id"] == record["id"]):
            row = dict(cached["row"])
            row["n_llm_calls"] = 0
            row["cache_hit"] = True
        else:
            with count_requests() as counter:
                for attempt in range(max_retries):
                    try:
                        result = run_crag(record["question"], record["id"], k=k, allow_web=allow_web)
                        row = _row(record, result)
                        row["n_llm_calls"] = counter["attempts"]
                        row["cache_hit"] = False
                        write_json(path, {"settings": settings, "row": row})
                        break
                    except Exception as error:
                        # Stack locations are useful; exception bodies may contain provider data.
                        failure = {"id": record["id"], "exception_type": type(error).__name__,
                                   "traceback": traceback.format_tb(error.__traceback__), "attempt": attempt + 1,
                                   "n_llm_calls": counter["attempts"]}
                        write_json(root / (record["id"] + ".failure.json"), failure)
                        print(f"Failed {record['id']}: {type(error).__name__}; sanitized traceback checkpoint saved.", flush=True)
                        if attempt + 1 == max_retries:
                            raise RuntimeError(f"Benchmark stopped at {record['id']}; completed IDs are checkpointed.") from None
                        time.sleep(min(30, backoff_delay * 2 ** attempt))
        failure_path = root / (record["id"] + ".failure.json")
        failure = read_json(failure_path)
        if isinstance(failure, dict) and not failure.get("resolved"):
            write_json(failure_path, {**failure, "resolved": True})
        rows.append(row)
        print(f"{name}: {index + 1}/{len(records)}; cached={row['cache_hit']}; requests={row['n_llm_calls']}", flush=True)
    return rows

def run_smoke_benchmark(smoke_ids, records_by_id, k=3, allow_web=False,
                        out_cache_path=CACHE_DIR / "smoke_crag.json", max_retries=3, backoff_delay=2):
    missing = set(smoke_ids) - records_by_id.keys()
    if missing:
        raise ValueError("Smoke IDs must all belong to the indexed slice.")
    rows = _benchmark([records_by_id[qid] for qid in smoke_ids], "smoke_crag_v2", k, allow_web, max_retries, backoff_delay)
    write_json(out_cache_path, rows)
    return rows

def run_slice_50_benchmark(records, k=3, allow_web=False, max_retries=3, backoff_delay=2):
    rows = _benchmark(records[:50], "slice_50_crag_v2", k, allow_web, max_retries, backoff_delay)
    frame = pd.DataFrame(rows)
    metrics = {"n_questions": len(rows), "gold_title_recall@k": float(frame.gold_title_recall_at_k.mean()),
               "answer_substring_match": float(frame.answer_contains_gold.mean()),
               "mean_llm_calls": float(frame.n_llm_calls.mean()), "total_llm_calls": int(frame.n_llm_calls.sum()),
               "cached_rows": int(frame.cache_hit.sum()),
               "action_when_gold_present_vs_absent": pd.crosstab(frame.gold_in_hits, frame.action).to_dict()}
    failures = [row for row in rows if not row["answer_contains_gold"]][:5]
    return rows, metrics, failures
