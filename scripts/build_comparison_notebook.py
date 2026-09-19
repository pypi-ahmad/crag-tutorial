"""Generate the paired naive-RAG versus CRAG tutorial notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []

def prose(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def lesson(title, motivation, paper, action, failure, interpretation, code):
    prose(f"### {title}\n\n**Motivation.** {motivation}\n\n**Paper mapping.** {paper}\n\n**Next cell.** {action}\n\n**Failure signals.** {failure}\n\n**Read the output.** {interpretation}")
    cells.append(nbf.v4.new_code_cell(code.strip()))

prose("""# Naive RAG vs CRAG on the same 50 HotpotQA questions

This tutorial compares a controlled naive-RAG baseline with the corrective pipeline described by Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling in [*Corrective Retrieval Augmented Generation*, arXiv:2401.15884](https://arxiv.org/abs/2401.15884).

## Learning goals

By the end, you can identify what the comparison holds constant, separate string metrics from human answer quality, inspect paired failures, and judge whether correction helped on this 50-question slice.

## Fairness contract

Both methods receive the same question and identical ranked top-3 passages from the question's HotpotQA distractor pool. Both use `agnes-3.0-flash` and the same evidence-only answer generator. Naive RAG sends all complete passages directly to generation. CRAG evaluates, routes, selects exact sentence strips, and then generates. Web search is disabled. Gold answers and gold labels never enter model prompts.

The setup isolates the corrective stages. It does not reproduce the paper's trained T5-large evaluator or establish general accuracy. Human judgements cover all 50 pairs and are tied to exact answer/evidence fingerprints.
""")

lesson("Set up the reproducible environment", "Use the same repo paths, credentials, and fixed model as the main course.",
       "Infrastructure needed for the comparison.",
       "Resolve the repo root, import the comparison APIs, and verify credential presence without printing values.",
       "Missing variables, a wrong kernel, or imports from another checkout stop execution before model calls.",
       "Both flags should be True and the model must be agnes-3.0-flash.", '''
import json
import sys
from collections import Counter
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import check_environment, MODEL_NAME
from src.data_hotpot import build_slice
from src.qdrant_store import index_slice, close_qdrant_client
from src.retrieve import search
from src.pipeline import run_crag
from src.comparison import run_naive_rag, run_comparison_50, automated_metrics, load_manual_reviews

env = check_environment()
print(env)
assert all(env.values()) and MODEL_NAME == "agnes-3.0-flash"
''')

lesson("Load the exact course slice and index", "A paired comparison needs the same questions and corpus for both methods.",
       "The paper motivates correction after retrieval. This adaptation uses HotpotQA.",
       "Load the seed-42 slice, index it idempotently, and lock the first 50 IDs.",
       "A dataset provenance mismatch rebuilds the slice; a Qdrant lock means another kernel owns the store.",
       "Expect 200 records, 50 unique comparison IDs, and 1,992 indexed paragraphs.", '''
records, _ = build_slice(n=200, seed=42)
indexed = index_slice(records)
comparison_ids = [row["id"] for row in records[:50]]
assert len(records) == 200 and len(set(comparison_ids)) == 50
print("Indexed paragraphs:", indexed)
print("Comparison IDs:", len(comparison_ids), comparison_ids[:3], "...", comparison_ids[-1])
''')

prose("""## What “naive” means here

Naive RAG is retrieval → generation. It does not inspect relevance scores, choose Correct/Incorrect/Ambiguous, filter sentences, or search the web. It still uses a grounded answer prompt so both methods follow the same evidence rule. The baseline context is the full text of all three retrieved passages in rank order.
""")

lesson("Verify one shared retrieval set", "Check the experimental control before comparing answers.",
       "Both branches begin from the same retriever output; only the corrective stages differ.",
       "Retrieve the first question once and display its ranked titles, similarities, and diagnostic gold labels.",
       "Different question IDs in payloads indicate a broken pool filter; similarity is not relevance confidence.",
       "The same three objects shown here are passed to both methods.", '''
worked = records[0]
worked_hits = search(worked["question"], k=3, question_id=worked["id"])
assert all(hit["payload"]["question_id"] == worked["id"] for hit in worked_hits)
display(pd.DataFrame([{"rank": i + 1, "title": hit["title"], "similarity": hit["score"], "is_gold": hit["payload"]["is_gold"]} for i, hit in enumerate(worked_hits)]))
''')

lesson("Run naive RAG on complete passages", "Establish the baseline answer without evaluator or refinement calls.",
       "This is the uncorrected retrieval-to-generation path that CRAG is intended to improve.",
       "Pass the three complete retrieved texts directly to the grounded generator.",
       "An abstention means the generator found the full retrieved context insufficient; it is not a transport failure.",
       "Inspect the answer and the exact titles supplied to it; gold labels were not included in the prompt.", '''
worked_naive = run_naive_rag(worked["question"], worked_hits)
print("Question:", worked["question"])
print("Evidence titles:", worked_naive["evidence_titles"])
print("Naive answer:", worked_naive["answer"])
print("Actual requests:", worked_naive["n_llm_calls"])
''')

lesson("Run CRAG on those identical passages", "Observe the effect of evaluation, routing, and strip refinement without changing retrieval.",
       "Maps directly to the paper's corrective control flow, implemented here with a prompted Agnes evaluator.",
       "Pass the already-retrieved hit objects into CRAG and display its decision and retained strips.",
       "A Correct action does not prove both hops are present; an empty strip set leads to a safe abstention.",
       "Compare the retained evidence with the full naive context before comparing answer text.", '''
worked_crag = run_crag(worked["question"], worked["id"], k=3, allow_web=False, docs=worked_hits)
print("Action:", worked_crag.action)
display(pd.DataFrame([{k: item[k] for k in ("title", "score", "label", "why")} for item in worked_crag.evaluations]))
print("Kept strips:", json.dumps(worked_crag.strips, indent=2, ensure_ascii=False))
print("CRAG answer:", worked_crag.answer)
print("Gold answer:", worked["gold_answer"])
''')

prose("""## Paired 50-question run

Each ID is atomically checkpointed, so a crash resumes at the failed ID. A first run can make one naive generation request plus CRAG evaluator, refinement, and generation requests per row; primitive CRAG caches can reduce that count. A warm rerun reports zero new calls. It does not erase the requests that created the cache.
""")

lesson("Execute all 50 paired questions", "Collect enough paired cases to see systematic wins, losses, abstentions, and metric disagreements.",
       "A tutorial comparison of the paper-inspired correction mechanism, not the paper benchmark.",
       "Run both methods on the same 50 questions with k=3 and web disabled.",
       "Provider/schema failures are checkpointed and retried; exhausted retries stop without dropping the row.",
       "Every progress line identifies cache state and actual requests for that comparison invocation.", '''
rows = run_comparison_50(records, k=3, allow_web=False)
assert len(rows) == 50 and [row["id"] for row in rows] == comparison_ids
print("Completed paired rows:", len(rows))
''')

lesson("Compare automated proxies", "Quantify retrieval coverage, string overlap, abstention, and current-run cost before human interpretation.",
       "These diagnostics are tutorial additions; they are not the paper's reported results.",
       "Compute paired metrics and a compact per-method table.",
       "Substring match can reward negation or miss aliases; it must not be called accuracy.",
       "Look for different answer behavior despite identical title recall, then check whether this invocation was cold or cached.", '''
auto = automated_metrics(rows)
print(json.dumps(auto, indent=2))
display(pd.DataFrame({
    "method": ["Naive RAG", "CRAG"],
    "substring_match": [auto["naive_substring_match"], auto["crag_substring_match"]],
    "abstention_rate": [auto["naive_abstention_rate"], auto["crag_abstention_rate"]],
    "actual_requests_this_run": [auto["naive_llm_calls"], auto["crag_llm_calls"]],
}))
''')

prose("""## Human audit of all 50 pairs

All 50 pairs were reviewed manually against the question, reference answer, and evidence given to each method. The audit is method-aware: the reviewer could see which answer came from naive RAG and which came from CRAG. That supports pipeline-specific error attribution and can introduce reviewer bias, which remains a limitation.

Correctness labels are `correct`, `partially_correct`, `incorrect`, and `abstain`. Evidence labels are `fully_supported`, `partially_supported`, `unsupported`, and `no_answer`. Pairwise preference weighs semantic correctness first, then evidence support, appropriate abstention, and concision or contradiction. A safe abstention is preferred to an unsupported wrong answer. Reviews are human judgements, not additional Agnes calls.
""")

lesson("Validate reviews against exact outputs", "Prevent a judgement from silently attaching to a regenerated answer or changed evidence set.",
       "Manual evaluation supplements automated metrics; it is not part of the CRAG algorithm.",
       "Load the tracked 50-row audit and compare its fingerprints with this run.",
       "Missing or changed rows are listed for re-review while the notebook remains runnable.",
       "A fully reviewed run shows 50 valid and zero stale IDs.", '''
review_path = ROOT / "reviews/naive_vs_crag_50_manual.json"
reviews, stale_ids = load_manual_reviews(rows, review_path)
print("Valid manual reviews:", len(reviews))
print("Stale or missing IDs:", stale_ids)
''')

lesson("Summarize human answer quality", "Use semantic and evidence judgements alongside the string heuristic.",
       "This is an external tutorial audit of the two pipelines. It is not a trained evaluator or paper metric.",
       "Count correctness, grounding, and pairwise preferences for all fingerprint-valid reviews.",
       "If fewer than 50 reviews validate, totals are incomplete and no overall winner should be claimed.",
       "Compare method correctness with preference counts and note ties; the sample remains descriptive.", '''
if len(reviews) == 50:
    manual_summary = {
        "naive_correctness": dict(Counter(r["naive_correctness"] for r in reviews)),
        "crag_correctness": dict(Counter(r["crag_correctness"] for r in reviews)),
        "naive_support": dict(Counter(r["naive_support"] for r in reviews)),
        "crag_support": dict(Counter(r["crag_support"] for r in reviews)),
        "preference": dict(Counter(r["preference"] for r in reviews)),
    }
    print(json.dumps(manual_summary, indent=2))
else:
    manual_summary = {}
    print("Manual summary withheld until all 50 exact outputs are reviewed.")
''')

lesson("Find string-metric disagreements", "Show why literal answer overlap cannot replace human inspection.",
       "The paper motivates stronger correction; this cell critiques our tutorial evaluation proxy.",
       "Join reviews to outputs and display cases where substring match disagrees with manual correctness.",
       "An empty table means agreement on this run, not proof that substring matching is generally valid.",
       "Alias misses and incidental matches should be visible with their human rationale.", '''
review_by_id = {review["id"]: review for review in reviews}
disagreements = []
for row in rows:
    review = review_by_id.get(row["id"])
    if not review:
        continue
    for method in ("naive", "crag"):
        manually_correct = review[f"{method}_correctness"] == "correct"
        substring = row[method]["answer_contains_gold"]
        if manually_correct != substring:
            disagreements.append({"id": row["id"], "method": method, "question": row["question"],
                                  "gold": row["gold_answer"], "answer": row[method]["answer"],
                                  "substring": substring, "manual": review[f"{method}_correctness"],
                                  "why": review["rationale"]})
display(pd.DataFrame(disagreements))
print("Metric disagreements:", len(disagreements))
''')

prose("""## Exercise

Pick one `naive_better`, one `crag_better`, and one inadequate tie. For each, trace retrieval → supplied evidence → answer. Decide whether retrieval, CRAG evaluation, refinement, or generation caused the result. The next cell provides a compact scaffold from the completed human audit.
""")

lesson("Build a three-case error-analysis scaffold", "Practice causal diagnosis alongside win counts.",
       "CRAG's components create identifiable failure boundaries after shared retrieval.",
       "Select one reviewed example from each requested preference category and expose its evidence path.",
       "A missing category is reported honestly; the notebook does not substitute an unrelated example.",
       "Use the shown action, titles, strips, and rationale to explain where the methods diverged.", '''
for preference in ("naive_better", "crag_better", "tie_both_inadequate"):
    chosen = next((review for review in reviews if review["preference"] == preference), None)
    print("\\n", preference.upper())
    if not chosen:
        print("No reviewed row in this category.")
        continue
    row = next(item for item in rows if item["id"] == chosen["id"])
    print("Q:", row["question"])
    print("Retrieved:", [hit["title"] for hit in row["hits"]])
    print("CRAG action/kept:", row["crag"]["action"], row["crag"]["kept_titles"])
    print("Diagnosis:", chosen["error_types"], "—", chosen["rationale"])
''')

lesson("Release Qdrant before the final report", "Avoid a Windows file lock after all retrieval-dependent work is complete.",
       "Operational cleanup is outside the comparison algorithm.",
       "Close the shared embedded client; the final reporting cell uses in-memory rows only.",
       "If another kernel owns the store, shut it down. Do not delete lock files.",
       "The confirmation means this kernel released its client.", '''
close_qdrant_client()
print("Qdrant client closed.")
''')

lesson("Final answer-by-answer comparison", "End with the complete human-readable evidence needed to inspect every verdict.",
       "This reporting layer compares the naive baseline with the paper-inspired corrective path.",
       "Print all 50 questions with gold, naive, CRAG, manual verdict, and rationale in slice order.",
       "A NOT REVIEWED verdict means the answer/evidence fingerprint changed and requires a new human judgement.",
       "Read each block as a paired case; aggregate claims must agree with these 50 underlying decisions.", '''
for number, row in enumerate(rows, 1):
    review = review_by_id.get(row["id"])
    verdict = review["preference"] if review else "NOT REVIEWED"
    reason = review["rationale"] if review else "The saved manual review is missing or stale for this exact output."
    print("=" * 100)
    print(f"Q{number:02d} [{row['id']}]: {row['question']}")
    print("Gold answer:", row["gold_answer"])
    print("Naive RAG:", row["naive"]["answer"])
    print("CRAG:", row["crag"]["answer"])
    print("Manual verdict:", verdict)
    print("Why:", reason)
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python (CRAG Tutorial)", "language": "python", "name": "crag-tutorial"}})
nbf.write(nb, ROOT / "notebooks/02_naive_vs_crag_comparison.ipynb")
print(f"Generated comparison notebook: {len(cells)} cells.")
