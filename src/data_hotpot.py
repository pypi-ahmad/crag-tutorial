"""Deterministic HotpotQA distractor validation slice, with provenance."""
import json
import os
from pathlib import Path
import datasets
from src.cache import fingerprint, read_json, write_json, write_text
from src.config import CACHE_DIR

def download_or_load(cache_dir=None, token=None):
    token = token if token is not None else os.environ["HF_TOKEN"]
    try:
        return datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                                     cache_dir=str(cache_dir or CACHE_DIR / "hotpot_distractor_val"), token=token)
    except Exception as error:
        raise RuntimeError(f"HotpotQA load failed ({type(error).__name__}); check HF_TOKEN, network and the dataset cache.") from None

def build_slice(n=200, seed=42, output_path=None, smoke_path=None, force_rebuild=False):
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer.")
    out = Path(output_path or CACHE_DIR / f"hotpot_slice_{n}.jsonl")
    smoke = Path(smoke_path or CACHE_DIR / "hotpot_smoke_8.json")
    manifest = out.with_suffix(".manifest.json")
    settings = {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation", "n": n, "seed": seed, "schema": 2}
    meta = read_json(manifest)
    if not force_rebuild and isinstance(meta, dict) and meta.get("settings") == settings and out.exists():
        try:
            records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(records) == n and fingerprint(records) == meta.get("fingerprint"):
                ids = _smoke(records, smoke)
                return records, ids
        except (ValueError, KeyError):
            pass
    ds = download_or_load()
    if n > len(ds):
        raise ValueError("Requested slice exceeds validation split.")
    records = []
    for item in ds.shuffle(seed=seed).select(range(n)):
        facts = item.get("supporting_facts")
        gold = list(dict.fromkeys(facts["title"])) if facts else []
        paragraphs = [{"title": title, "text": title + ". " + " ".join(sentences).strip(),
                       "sentences": sentences, "is_gold": title in gold if facts else None}
                      for title, sentences in zip(item["context"]["title"], item["context"]["sentences"])]
        records.append({"id": item["id"], "question": item["question"], "gold_answer": item["answer"],
                        "gold_titles": gold, "question_type": item.get("type", "unknown"),
                        "level": item.get("level", "unknown"), "context_paragraphs": paragraphs,
                        "is_gold": [p["is_gold"] for p in paragraphs]})
    write_text(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write_json(manifest, {"settings": settings, "fingerprint": fingerprint(records)})
    return records, _smoke(records, smoke)

def _smoke(records, path):
    bridge = [r["id"] for r in records if r["question_type"] == "bridge"][:4]
    comparison = [r["id"] for r in records if r["question_type"] == "comparison"][:4]
    ids = bridge + comparison
    ids += [r["id"] for r in records if r["id"] not in ids][:max(0, 8 - len(ids))]
    write_json(path, {"count": len(ids), "bridge_ids": bridge, "comparison_ids": comparison, "smoke_ids": ids})
    return ids
