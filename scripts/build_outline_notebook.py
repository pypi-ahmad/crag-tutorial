"""Generate the canonical teaching notebook; executed outputs are saved separately."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []

def prose(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def lesson(title, motivation, paper, action, failure, interpretation, code):
    prose(f"### {title}\n\n**Motivation.** {motivation}\n\n**Paper mapping.** {paper}\n\n**Next cell.** {action}\n\n**Failure signals.** {failure}\n\n**Read the output.** {interpretation}")
    cells.append(nbf.v4.new_code_cell(code.strip()))

prose("""# Corrective Retrieval Augmented Generation: a Windows-native course

Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling, [*Corrective Retrieval Augmented Generation*, arXiv:2401.15884](https://arxiv.org/abs/2401.15884).

This notebook follows the paper's evaluate → route → refine → generate structure. It adapts that flow for a small Windows course. Run cells in order with **Restart Kernel and Run All**. The first run may download HotpotQA and make provider-metered Agnes calls; later runs reuse versioned caches. The course uses no search key, Docker, WSL, paid embedding API, or LangChain.

## 1. Why naive RAG fails

A retriever ranks similarity. It does not decide whether the evidence is sufficient. A passage can share every query word while describing the wrong person, and a multi-hop question can retrieve one useful bridge while missing the other. Generating directly from those passages can produce a plausible, unsupported answer. Correction adds a decision between retrieval and generation, though the evaluator can still admit distractors or discard useful facts.

Ask three separate questions: did retrieval find the annotated evidence? Did refinement keep the necessary facts? Does the answer follow from those facts? High relevance does not establish a complete multi-hop chain, and a literal substring match does not establish factual correctness.

## 2. CRAG paper map

| Paper component | This course | Important difference |
|---|---|---|
| Retrieval evaluator (§4.2) | Agnes JSON score for each passage | Paper uses a trained T5-large (0.77B) evaluator; we do not train it |
| Correct / Incorrect / Ambiguous (§4.3) | Maximum score with upper .7 and lower .3 | Heuristic 0–1 thresholds, not calibrated paper defaults |
| Knowledge refinement (§4.4) | Select exact sentence IDs | A prompted approximation of fine-grained filtering |
| Knowledge searching (§4.5) | Optional real web snippets | Disabled in reported runs; Incorrect therefore abstains |
| Generator | Agnes Chat Completions | Fixed agnes-3.0-flash, not the paper's generator setup |

The paper evaluates PopQA, Biography, PubHealth and ARC. This course uses HotpotQA for inspectable multi-hop evidence. Neither its data nor its scores reproduce paper results. The paper's relevance scale is not our 0–1 prompt scale.

## 3. Environment check

Store credentials in Windows **user environment variables**. Restart the terminal or Jupyter launcher after changing them so the process inherits them. `.env.example` lists variable names only and is not loaded. `AGNES_BASE_URL` is informational; the implementation uses the required fixed endpoint.
""")
lesson("Check imports and credential presence", "Fail before downloading data or making API requests if setup is incomplete.",
       "Infrastructure used by every stage.", "Locate the repo from either its root or notebooks directory and print booleans only.",
       "A missing variable, Python below 3.11, or ModuleNotFoundError means setup must be corrected first.",
       "Both credential flags should be True. No credential value is displayed.", '''
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
assert sys.version_info >= (3, 11)
from src.config import check_environment, MODEL_NAME, BASE_URL
env = check_environment()
print(env)
assert all(env.values()), "Set AGNESAI_API_KEY and HF_TOKEN in the Windows user environment; restart Jupyter."
print("Model:", MODEL_NAME, "Endpoint:", BASE_URL)
''')
prose("""## 4. Dataset

[HotpotQA on Hugging Face](https://huggingface.co/datasets/hotpotqa/hotpot_qa), configuration `distractor`, split `validation`, supplies annotated supporting facts. The usual candidate pool has two gold paragraphs plus distractors; inspect actual counts instead of assuming ten for every row. A paragraph is gold when its title occurs in `supporting_facts.title`. These labels are for evaluation only and never enter model prompts.

Shuffle validation with seed 42 and select 200 rows. The smoke set takes up to four bridge and four comparison questions **from this slice**, filling from remaining slice rows if necessary. A bridge question follows an entity relation; a comparison question needs facts about both subjects. Gold titles annotate evidence, not a relevance probability.

This is not Meta Comprehensive RAG Benchmark, which shares the CRAG acronym. The dataset slice, raw download, provenance manifest, model judgements, and resumable results live under ignored `data/cache/`.
""")
lesson("Load the reproducible slice", "Make every later comparison use the same questions and labels.", "A tutorial dataset substitution, not a paper benchmark.",
       "Load or build the 200-row slice and show type and pool-size counts.", "HF authentication/network errors or a wrong config prevent loading; do not fabricate rows.",
       "Expect 200 unique IDs and eight smoke IDs inside the slice; pool sizes may vary.", '''
import pandas as pd
from src.data_hotpot import build_slice
records, smoke_ids = build_slice(n=200, seed=42)
records_by_id = {r["id"]: r for r in records}
assert len(records_by_id) == 200 and len(smoke_ids) == 8
assert set(smoke_ids) <= records_by_id.keys()
display(pd.Series([r["question_type"] for r in records]).value_counts().rename("questions"))
display(pd.Series([len(r["context_paragraphs"]) for r in records]).value_counts().rename("pool sizes"))
print("Smoke IDs:", smoke_ids)
''')
lesson("Read one complete example", "See the evidence before interpreting retrieval scores.", "Connects retrieval inputs to supporting-fact supervision used only for diagnostics here.",
       "Print the full first record, including every paragraph and gold title.", "Missing sentences or gold labels would limit evaluation and must not be silently invented.",
       "Follow the supporting titles and ask whether both hops needed for the answer are explicit.", '''
import json
print(json.dumps(records[0], indent=2, ensure_ascii=False))
''')
prose("""## 5. Retriever: embedded Qdrant

TF-IDF weights lexical features. TruncatedSVD projects the sparse matrix to at most 256 latent dimensions, pads when needed, and normalizes vectors for cosine search. This free CPU baseline differs from a modern semantic embedding model. It fits candidate paragraph text only, never answers or gold labels. Unknown query vocabulary can produce weak or zero vectors.

Main retrieval filters `question_id`: each question competes only with its own supplied distractor pool. An unfiltered search over the 200-question corpus is a separate demonstration, not the reported evaluation protocol. We store labels in payloads for diagnostics but send only title and text to the evaluator. Count **and corpus/encoder fingerprints** must match before index reuse; count alone misses changed documents.
""")
lesson("Index and compare search scopes", "Keep evaluation boundaries explicit and make reruns idempotent.", "Provides the retrieved document set evaluated by CRAG.",
       "Index the slice, run pool-filtered retrieval and then an unfiltered demo.", "A Qdrant lock means another kernel owns the same path; stale encoder metadata triggers a rebuild.",
       "Filtered payload IDs must all match the requested question; similarity scores are not evaluator confidence.", '''
from src.qdrant_store import index_slice, close_qdrant_client
from src.retrieve import search
count = index_slice(records)
sample = records[0]
pool_hits = search(sample["question"], k=3, question_id=sample["id"])
assert all(h["payload"]["question_id"] == sample["id"] for h in pool_hits)
print("Indexed paragraphs:", count)
display(pd.DataFrame([{ "title": h["title"], "similarity": h["score"], "is_gold": h["payload"]["is_gold"]} for h in pool_hits]))
open_hits = search(sample["question"], k=5)
display(pd.DataFrame([{ "title": h["title"], "question_id": h["payload"]["question_id"]} for h in open_hits]))
''')
prose("""## 6. Evaluator

For each passage, Agnes returns `score`, `label`, and `why`. An intermediate hop can be relevant; a shared name alone is not enough. A 0.9 score is a prompted judgement, not a calibrated 90% probability. JSON extraction accepts fences, preamble, and braces inside strings, then validates finite scores in [0,1], allowed labels, and a nonempty rationale. Transport and schema failures raise instead of becoming cached zero scores.
""")
lesson("Inspect document judgements", "A routing decision is only as good as the evidence assessment behind it.", "Approximates the retrieval evaluator in §4.2 with an LLM rather than trained T5-large.",
       "Evaluate three retrieved passages, showing each score, label and explanation.", "Malformed JSON or invalid scores stop the cell; rerun after correction, reusing successful cached judgements.",
       "Compare rationales with actual text and labels; do not assume a confident explanation is correct.", '''
from src.crag import evaluate_documents, decide_action, refine_strips, generate_answer
evaluations = evaluate_documents(sample["question"], pool_hits)
display(pd.DataFrame([{k: e[k] for k in ("title", "score", "label", "why")} for e in evaluations]))
''')
prose("""## 7. Actions

Let m be the maximum passage score. `m >= .7` selects Correct; `m <= .3` selects Incorrect; otherwise the route is Ambiguous. No passages also gives Incorrect. These inclusive boundaries are deliberate. Correct says that at least one passage looks strong. It does not establish that every required hop is present. Incorrect discards local evidence and abstains when web recovery is disabled. Ambiguous retains passages above the lower threshold; optional web search can supplement them. Labels explain the decision, while numeric scores control routing and filtering.
""")
lesson("Check routing boundaries", "Make the decision rule testable independently of stochastic model outputs.", "Implements the three-way control decision from §4.3 with tutorial thresholds.",
       "Show the current route and deterministic boundary examples.", "Out-of-range or NaN scores and reversed thresholds must raise rather than silently route.",
       "Synthetic boundary scores are unit examples, not measured model results.", '''
print("Observed action:", decide_action(evaluations))
for scores in ([], [0.3], [0.5], [0.7], [0.1, 0.8]):
    print(scores, "->", decide_action(scores))
''')
prose("""## 8. Strip refinement

Long paragraphs can contain distractors even when their title is useful. Split them into sentences, number those sentences, and ask the model for IDs. The selected text comes from the original sentences, so the model cannot insert a new fact through an invented strip. Exact copying preserves provenance, though it cannot guarantee relevance or enough context. The punctuation-based splitter is deliberately simple and can mishandle abbreviations.
""")
lesson("Keep source-backed sentences", "Reduce irrelevant context while retaining intermediate evidence.", "A sentence-selection approximation of knowledge refinement in §4.4.",
       "Refine one passage and print the exact retained strings.", "Invalid sentence IDs fail validation; an empty result is legitimate and must not be replaced with heuristic evidence.",
       "Check whether each kept sentence answers a needed subquestion; missing a bridge can break generation.", '''
kept = refine_strips(sample["question"], pool_hits[0])
print("Kept strips:", json.dumps(kept, indent=2, ensure_ascii=False))
''')
prose("""## 9. Generation and three diagnostic scenarios

The generator sees only the question and selected strips. Empty evidence returns a deterministic abstention without an API request. With nonempty evidence, the model is told to abstain unless the strips establish the complete answer. That instruction does not formally guarantee grounding. Gold answers stay outside prompts.

Annotations select examples for diagnostic cases only. The ordinary pipeline retrieves and evaluates without gold labels. The hidden-gold case removes annotated paragraphs to test the evaluator. Removing gold does not mathematically force an LLM to choose Incorrect. A Correct decision in that case is an evaluator failure to inspect, not a reason to overwrite its scores.
""")
lesson("Generate from the inspected evidence", "Separate source selection from answer formulation.", "The final answer-generation stage consumes refined knowledge.",
       "Generate from the selected strips and compare with the gold answer, then demonstrate empty-evidence abstention.",
       "A plausible answer without supporting facts is still a failure; provider failures are raised and not cached as answers.",
       "Treat answer-versus-gold as a diagnostic, not evidence of paper-level accuracy.", '''
print("Answer:", generate_answer(sample["question"], kept))
print("Gold:", sample["gold_answer"])
print("Empty evidence:", generate_answer(sample["question"], []))
''')
lesson("Run three evidence stress tests", "Expose easy retrieval, multi-hop completeness, and evaluator false positives.",
       "Exercises evaluation, action selection, refinement and generation together.",
       "Find two different questions with both gold titles retrievable, then run a third question with gold paragraphs hidden.",
       "If no suitable question exists the assertions fail transparently; a hidden-gold Correct action is a finding, not an exception.",
       "Read action, scores, retained strips and answer side by side. Diagnostic gold labels never choose strips inside run_crag.", '''
from src.pipeline import run_crag
easy = next(r for r in records if set(r["gold_titles"]) <= {h["title"] for h in search(r["question"], 3, r["id"])})
multi = next(r for r in records if r["id"] != easy["id"] and r["question_type"] == "bridge" and set(r["gold_titles"]) <= {h["title"] for h in search(r["question"], 4, r["id"])})
hidden = next(r for r in records if r["id"] not in (easy["id"], multi["id"]) and len(r["context_paragraphs"]) > 2)
distractors = [{"title": p["title"], "text": p["text"], "payload": {"question_id": hidden["id"]}} for p in hidden["context_paragraphs"] if not p["is_gold"]][:3]
scenarios = [("Easy retrieval", easy, 3, None), ("Both hops retrievable", multi, 4, None), ("Gold hidden", hidden, 3, distractors)]
for name, row, k, docs in scenarios:
    result = run_crag(row["question"], row["id"], k=k, docs=docs)
    print("\\n", name, row["id"], row["question"])
    print("Action:", result.action, "Actual requests:", result.n_llm_calls)
    display(pd.DataFrame([{key: e[key] for key in ("title", "score", "label", "why")} for e in result.evaluations]))
    print("Kept strips:", json.dumps(result.strips, ensure_ascii=False, indent=2))
    print("Answer:", result.answer, "\\nGold:", row["gold_answer"])
''')
prose("""## 10. Full pipeline on the eight-ID smoke set

Use k=3 and `allow_web=False`. Each successful ID is atomically checkpointed; rerunning skips completed IDs and retries only failures. Provider 429 responses receive bounded exponential backoff; benchmark-level retries reuse already-cached successful primitive calls. Exhausted retries raise with a sanitized stack checkpoint instead of silently dropping a question.

`gold_in_kept_titles` means at least one annotated title contributed an actual retained strip. It does not mean both gold titles survived. `n_llm_calls` counts actual HTTP attempts during this invocation, including retries; cache hits and empty-evidence abstentions count zero. Literal `answer_contains_gold` tests a nonempty case-insensitive gold substring in the answer in one direction only.
""")
lesson("Run and interpret smoke results", "Check all stages before spending requests on 50 questions.", "An engineering smoke test, not a paper evaluation protocol.",
       "Execute eight IDs, save smoke_crag.json, and derive two prose interpretations from the actual rows.",
       "A crash leaves completed IDs checkpointed; fix the cause and rerun this cell. Never replace failed IDs with invented outputs.",
       "A substring hit can be accidental; no-hit can be a valid paraphrase. Prose below reports observations without claiming correctness.", '''
from src.pipeline import run_smoke_benchmark
from IPython.display import Markdown, display
smoke_rows = run_smoke_benchmark(smoke_ids, records_by_id, k=3, allow_web=False)
smoke_df = pd.DataFrame(smoke_rows)
display(smoke_df[["id", "type", "action", "gold_in_kept_titles", "answer_contains_gold", "n_llm_calls", "cache_hit"]])
for row in smoke_rows[:2]:
    display(Markdown(f"For `{row['id']}`, the route was **{row['action']}**. At least one annotated title contributed a kept strip: **{row['gold_in_kept_titles']}**. The literal gold substring test returned **{row['answer_contains_gold']}**. The answer was `{row['generated_answer']}` versus reference `{row['gold_answer']}`. Inspect the saved strips before interpreting this as success; this run made {row['n_llm_calls']} model requests for the row."))
''')
prose("""## 11. Evaluation on the first 50 slice questions

This small descriptive sample is neither a held-out tuning protocol nor a statistically robust benchmark. Do not calibrate thresholds on these rows and then present them as unbiased evaluation.

For each question, **gold_title_recall@k = number of distinct annotated titles retrieved / number of annotated titles**; report the mean across questions. Retrieving one of two titles gives 0.5, not 1.0. The action table groups questions by whether *any* gold title was present. This binary grouping must not be confused with title recall or complete evidence.

**Answer substring match** is a deliberately weak string proxy, not exact-match accuracy or factual correctness. `no` can match `not`; a sentence may mention the reference while denying it. Conversely `Kurt Weill` need not contain `Kurt Julian Weill` despite referring to the same person. We preserve the requested literal definition and label its limitations.

Mean model calls measures this invocation's HTTP attempts. A warm checkpoint run should report zero; it does not mean the original experiment was free. Cache keys include corpus, model, prompt-version and evaluation settings. Legacy caches are retained but not trusted by the repaired implementation.
""")
lesson("Run the 50-question slice", "Measure retrieval coverage, routing and answer proxies separately.", "A tutorial evaluation rather than the paper's datasets, models or reported metrics.",
       "Resume per-ID checkpoints, display aggregate metrics and the action contingency table.",
       "A failed ID stops the cell after bounded retries; sanitized traceback files identify it and completed IDs remain reusable.",
       "Compare mean title recall with the action table; inspect cache counts before interpreting mean calls.", '''
from src.pipeline import run_slice_50_benchmark
eval_rows, metrics, failure_gallery = run_slice_50_benchmark(records, k=3, allow_web=False)
eval_df = pd.DataFrame(eval_rows)
print(json.dumps(metrics, indent=2))
display(pd.crosstab(eval_df["gold_in_hits"], eval_df["action"], rownames=["Any gold title retrieved"]))
assert len(eval_rows) == 50
''')
prose("""## 12. Failure analysis

The gallery selects up to five substring nonmatches. They are candidate errors that require review. Aliases, abstentions, and incomplete references need human interpretation. Compare question → retrieved titles/scores → retained strips → generated answer. A relevant distractor may be a false positive, while a non-gold passage can still be useful evidence.

When all supplied passages fail to support either hop, Incorrect should have fired. A high-score distractor can instead force Correct because routing uses the maximum. Conversely a correct bridge paragraph can deserve relevance even before the second hop is present. Diagnose evaluator relevance separately from answer sufficiency; do not overwrite scores to force the expected story.

With web disabled, Incorrect cannot recover missing knowledge. This is an explicit limitation, not a reproduction of the paper's corrective web branch. Optional DDGS search uses real returned snippets and URLs only; network errors propagate, and snippets are not verified full-page evidence.
""")
lesson("Inspect candidate failures and distractors", "Turn aggregate numbers into specific repair hypotheses without inventing outcomes.", "Examines failure modes of the evaluator and correction policy.",
       "Show up to five nonmatches with all scores, then list high-scoring non-gold titles from evaluated rows.",
       "Fewer than five nonmatches is reported honestly. Gold labels are annotations, not proof that every other paragraph is irrelevant.",
       "Decide whether each row reflects retrieval loss, false-positive relevance, refinement loss, unsupported generation, or metric mismatch.", '''
print("Candidate nonmatches shown:", len(failure_gallery))
for row in failure_gallery:
    print("\\nID:", row["id"], "Question:", row["question"])
    print("Action:", row["action"], "Answer:", row["generated_answer"], "Gold:", row["gold_answer"])
    display(pd.DataFrame(row["evaluations"]))
    print("Kept strips:", json.dumps(row["kept_strips"], ensure_ascii=False, indent=2))
suspects = []
for row in eval_rows:
    gold = set(records_by_id[row["id"]]["gold_titles"])
    for e in row["evaluations"]:
        if e["title"] not in gold and e["score"] >= .7:
            suspects.append({"id": row["id"], "action": row["action"], **e})
display(pd.DataFrame(suspects))
print("High-scoring non-gold passages:", len(suspects), "(inspect text before calling them false positives)")
''')
prose("""## Common errors and recovery

- **AGNESAI_API_KEY vs AGNES_API_KEY:** only the former is read. Confirm presence in Windows user environment settings, then restart the launcher. Never paste keys into cells, files, screenshots or logs.
- **HF download:** `HF_TOKEN` must reach the process. Use `hotpotqa/hotpot_qa`, `distractor`, `validation`; check network access and the `data/cache/hotpot_distractor_val` cache. Do not substitute invented records.
- **429 / transient server errors:** bounded exponential backoff retries the request. If exhausted, rerun the failed benchmark cell; successful per-ID and primitive caches remain. Authentication failures require correcting credentials, not endless retries.
- **JSON fences or extra prose:** the parser extracts a valid object and validates its schema. Invalid scores/IDs raise and are not cached. A retry may fix a transient formatting failure; persistent failures require inspecting the prompt/schema without exposing secrets.
- **Qdrant lock:** only one process/kernel may own `data/qdrant`. Run the closing cell, shut down the other kernel, or restart it. Do not delete the database or lock file while another kernel is active. Run quick-check and tutorial sequentially.
- **Imports:** launch from the repo or its notebooks folder using the registered `Python (CRAG Tutorial)` kernel. `run.cmd` installs dependencies into `.venv`; a different notebook kernel may not have them.

## Limits and next questions

This course uses lexical SVD vectors, an uncalibrated LLM evaluator, simple sentence boundaries, no default web recovery, and only 50 evaluation questions. It cannot establish paper-level accuracy, production readiness, or universal improvement over naive RAG.

Continue with [`02_naive_vs_crag_comparison.ipynb`](02_naive_vs_crag_comparison.ipynb), which runs the fixed naive baseline and CRAG over the identical 50 questions and top-3 passages, then manually reviews every answer pair. A separate tuning/evaluation split remains future work.
""")
lesson("Release the local database lock", "Allow the next notebook or fresh kernel to reopen the embedded store safely.", "Operational cleanup, independent of the CRAG algorithm.",
       "Close the shared Qdrant client after all inspection is complete.", "If a previous cell crashed, run this cell manually or shut down its kernel before opening another notebook.",
       "The confirmation means this kernel released its client; another process can still own a separate lock.", '''
close_qdrant_client()
print("Qdrant client closed.")
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python (CRAG Tutorial)", "language": "python", "name": "crag-tutorial"}})
nbf.write(nb, ROOT / "notebooks/01_crag_tutorial.ipynb")
print(f"Generated main notebook: {len(cells)} cells.")
