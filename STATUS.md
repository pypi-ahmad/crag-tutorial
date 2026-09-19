# Implementation and verification status

Scope: `D:\AI\Github\crag-tutorial` only. Native Windows 11; no WSL, Docker, LangChain, or Agentic RAG. The project was later published at `https://github.com/pypi-ahmad/crag-tutorial` on `main`; the execution evidence below predates publication.

Reference: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu and Zhen-Hua Ling, [Corrective Retrieval Augmented Generation, arXiv:2401.15884](https://arxiv.org/abs/2401.15884).

## Files created or repaired

- `notebooks/01_crag_tutorial.ipynb`: canonical 12-section course, 13 code cells, full dataset example, three diagnostic scenarios, eight-ID smoke run, first-50 evaluation and candidate-error gallery.
- `notebooks/02_naive_vs_crag_comparison.ipynb`: executed 13-cell paired tutorial over the same 50 questions and retrieval hits, ending with all 50 question/answer/verdict/rationale blocks.
- `notebooks/00_quick_check.ipynb`: four code cells for safe environment checks, live Agnes ping, real Hotpot row and embedded Qdrant count.
- `scripts/build_outline_notebook.py`, `scripts/build_quick_check_notebook.py`: reproducible teaching sources. Every code cell has immediately preceding Markdown covering motivation, paper mapping, next action, failure signals and output interpretation.
- `src/agnes_client.py`, `src/config.py`: fixed Agnes endpoint/model, environment-only credentials, bounded retries, sanitized exceptions and actual request-attempt accounting.
- `src/cache.py`: content fingerprints and atomic JSON/text writes.
- `src/data_hotpot.py`: deterministic slice provenance, seed-aware cache reuse, smoke IDs restricted to the slice and unknown labels left unknown.
- `src/qdrant_store.py`, `src/retrieve.py`: CPU TF-IDF/SVD vectors, filtered/unfiltered search, corpus/encoder/scikit-learn provenance, missing-encoder repair and safe rebuilds without stale tail points.
- `src/crag.py`, `src/pipeline.py`: schema validation, exact source-sentence selection, abstention, actual kept-title provenance, resumable per-ID checkpoints and corrected title-recall/substring definitions.
- `src/search.py`: isolated optional real web snippets; provider failures never create canned evidence.
- `src/comparison.py` and `reviews/naive_vs_crag_50_manual.json`: resumable paired runner, automated summaries, fingerprint validation, and the completed method-aware human audit.
- `tests/test_core.py`: 20 offline regression tests.
- `scripts/verify_notebook.py`, `scripts/verify_launcher.py`, `scripts/audit_repository.py`: teaching/execution checks, isolated Windows launcher integration test, and read-only dependency/secret/ignore checks.
- `README.md`, `docs/tutorial-guide.md`, `docs/technical-reference.md`, `docs/evaluation-and-limitations.md`: zero-to-mastery learning path, source-grounded technical reference, reproducible evaluation guidance, and a concise Windows entry point. Existing MIT `LICENSE` preserved.
- `requirements.txt`, `run.cmd`, `.env.example`, `.gitignore`, `src/__init__.py`: dependency coverage, safe Windows setup and package exports.

Removed duplicate notebook `01_corrective_rag_tutorial.ipynb`, obsolete generator `scripts/build_notebook.py`, and unused parallel implementations `src/retriever.py`, `src/evaluator.py`, `src/refinement.py`, `src/generator.py`. They are recoverable from `data/cache/legacy_backup/20260919-225655/`, along with pre-edit source/docs and the old smoke aggregate. Legacy primitive/50-ID caches remain on disk but are not trusted by `crag-v2`.

## What actually executed

Verification date: 2026-09-19 (local Windows session).

- **Offline tests:** all 17 passed in the project Python 3.12.14 venv and in the isolated regular Python 3.14.7 venv with freshly installed dependencies. Tests cover malformed JSON, NaN/out-of-range scores, invalid strip IDs, uncached failures, empty-evidence abstention, retry counts including failed attempts, checkpoint resume/settings invalidation, exact substring direction, fractional title recall, label leakage, dataset seed changes, unknown labels, filtered search, changed/shrunken indexes, missing encoders and optional-search failure.
- **Static checks:** Ruff `E4,E7,E9,F` passed. `uv pip check` reported compatible installed packages. Direct imports are explicitly covered by `requirements.txt`.
- **Documentation:** README and the three linked Markdown guides were checked for local-link targets, source-grounded interface references, and configured-secret exposure. This documentation update did not rerun notebooks or alter prior execution evidence.
- **Launcher:** the first clean test exposed `py -3` selecting free-threaded Python 3.14 and gRPC build failures. The repaired launcher scopes `PY_PYTHON3` to the detected minor version, clears inherited active-venv selection only during creation, and still uses `py -3 -m venv .venv`. No global Python configuration changes. Clean regular-Python 3.14.7 setup and an existing-venv rerun both reached the local notebook route with HTTP 200 and registered the isolated kernel. A separate Python 3.12.14 fixture also passed. Test servers were terminated by their exact test process IDs. Browser rendering was not visually inspected.
- **Dataset:** real HotpotQA cached source was loaded through `datasets`, not replaced by synthetic records. The 200-row seed-42 slice has 158 bridge and 42 comparison questions; smoke IDs are four of each, all in-slice. There are 1,992 paragraphs: 199 pools of ten and one pool of two. Every row has two annotated gold titles. Existing raw download data was reused; a fresh full network download was not required.
- **Quick check:** fresh kernel completed all four cells, including a live `agnes-3.0-flash` PONG, one real cached Hotpot row and Qdrant collection count.
- **Full notebook:** all 13 code cells completed in a fresh kernel, including three real diagnostic scenarios, eight smoke IDs and all 50 evaluation IDs. Two subsequent fresh-kernel runs completed: the first reused primitive model caches after index/cache provenance hardening; the final run reused all eight smoke and all 50 evaluation checkpoints. Both reported zero additional benchmark model requests. The quick-check notebook was also rerun sequentially each time and made its intentional live ping. Both active notebooks retain their final successful executed outputs, with no error outputs.
- **Naive-vs-CRAG notebook:** all 13 code cells completed over the same first 50 IDs and identical top-3 hits. The cold paired invocation made 61 naive requests and zero new CRAG requests because verified primitive CRAG outputs were already cached. One naive provider failure recovered on the next benchmark attempt. All 50 comparison rows and all 50 manual-review fingerprints validated. A warm rerun completed with 50 cache hits and zero comparison requests; the saved notebook has no error outputs.

## Observed results (not paper accuracy)

Protocol: first 50 rows of the seed-42 slice, question-pool retrieval, k=3, upper=.7, lower=.3, `allow_web=False`.

| Measure | Observed value |
|---|---:|
| Mean gold-title recall@3 | 0.42 |
| Any annotated title retrieved | 34/50 |
| Literal answer substring match | 13/50 (0.26) |
| Initial repaired 50-ID invocation: actual request attempts | 252 |
| Mean attempts in that invocation | 5.04 |
| Initial eight-ID smoke invocation: actual request attempts | 33 |
| Smoke substring matches | 1/8 |
| Final warm 50-ID invocation: checkpoint hits / requests | 50 / 0 |
| Final warm smoke invocation: checkpoint hits / requests | 8 / 0 |

The initial 50-ID invocation already reused primitive caches from smoke/diagnostic cells; 252 is its additional request count, not the entire tutorial's lifetime cost. Request counts include transport retries and attempts that returned invalid content. Two benchmark-level failures were recorded: invalid strip IDs at `5ab874ba5542990e739ec904`, and a provider request failure at `5abe6f8455429965af743f03`. Both IDs succeeded on retry without discarding completed IDs. The recorded failure type alone does not establish a specific HTTP status.

| Any gold title in retrieved set? | Correct | Incorrect | Ambiguous |
|---|---:|---:|---:|
| Yes | 30 | 4 | 0 |
| No | 3 | 12 | 1 |

Correct without an annotated title and Incorrect despite a gold title both warrant inspection. These counts are observations, not proof that every non-gold paragraph is useless or every gold-containing pool is sufficient.

First-run executed notebooks and smoke results are preserved under `data/cache/verification/01_first_live_run.ipynb`, `00_first_live_run.ipynb` and `smoke_first_live_run.json`. The intermediate final-code/primitive-cache rerun is preserved as `01_final_code_primitive_cache_run.ipynb`. The main notebook's saved outputs reflect the last verified warm run, not historical request counts. Metrics and action counts remained identical across reruns.

## Security verification

The read-only audit scans Git candidate files (including notebook outputs) for the configured `AGNESAI_API_KEY` and `HF_TOKEN` values plus common HF/OpenAI-key/private-key patterns, reporting filenames only on a match. No matches were found. `.env.example` contains exactly three variable names, with no values. Ignore checks passed for `.env`, `.venv`, `data/cache`, `data/qdrant`, `__pycache__` and notebook checkpoints.

This repository currently has no committed/tracked history to audit; the check covers local candidate files, not a remote repository or a dedicated external secret scanner. It is not a proof that every possible credential format is absent.

## Limits and interpretation

- TF-IDF plus 256-dimensional SVD is a lexical CPU baseline, not paid/learned semantic embeddings.
- Agnes is a prompted LLM evaluator, not the paper's trained T5-large (0.77B). The .7/.3 thresholds are tutorial heuristics, not calibrated paper defaults.
- A Correct action proves neither both hops were found nor the final answer was grounded. Sentence copying prevents invented strip text but does not prove evidence sufficiency; the generator's grounding instruction is not a formal guarantee.
- The first 50 questions are a small descriptive slice. No paper-level accuracy, statistical generalization, naive-RAG improvement or production readiness is claimed.
- Literal answer substring match can reward incidental/negated mentions and miss valid aliases. The gallery therefore shows candidate nonmatches for inspection, not five automatically verified wrong answers.
- `allow_web=False` for reported runs. Incorrect abstains without web recovery. The optional DDGS branch was tested with mocks for failure safety, not as a live web-quality benchmark.
- Actual model-call counts depend on cache state. Zero on a warm run does not erase the requests that created the caches. HTTP 429 behavior was regression-tested with mocks; no live rate-limit incident is claimed without a recorded failure.
- Windows Jupyter emitted a nonfatal selector-thread fallback warning and a local TCP transport warning during execution; these did not themselves fail cells. Only local test servers were used.

## Observed naive-vs-CRAG comparison

| Measure | Naive RAG | CRAG |
|---|---:|---:|
| Literal substring match | 0.30 | 0.26 |
| Abstention rate | 0.66 | 0.72 |
| Manually correct | 15 | 13 |
| Manually partially correct | 1 | 0 |
| Manually incorrect | 1 | 1 |
| Manual abstentions | 33 | 36 |

Pairwise manual preferences: naive better 4, CRAG better 2, both adequate 11, both inadequate 33. Both methods shared mean gold-title recall@3 of 0.42 because retrieval was held constant. Review was method-aware, not blinded; it used the question, reference answer, and exact method evidence. Important findings include CRAG refinement loss on the Plymouth Barracuda and A.P. Møller rows, CRAG generation abstentions despite decisive retained strips, one evidence/reference date conflict, and string-metric errors for aliases and short `no` references.

## Reproduce checks

```powershell
uv run --no-project --python .venv\Scripts\python.exe python -m unittest discover -s tests -v
uvx ruff check src scripts tests --select E4,E7,E9,F
uv run --no-project --python .venv\Scripts\python.exe python scripts/audit_repository.py
uv run --no-project --python .venv\Scripts\python.exe python scripts/verify_notebook.py --execute
uv run --no-project --python .venv\Scripts\python.exe python scripts/verify_launcher.py
```

The last two commands perform real execution: notebook checks can make provider-metered requests; launcher checks create an ignored fixture venv and install dependencies. Do not run notebook verification while another kernel owns `data/qdrant`.
