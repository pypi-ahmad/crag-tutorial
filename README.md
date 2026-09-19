# Corrective Retrieval Augmented Generation: Windows tutorial

This repository is a Windows-native, notebook-led course on **Corrective Retrieval Augmented Generation (CRAG)**. It takes a reader from the basics of retrieval-augmented generation (RAG) through a real, inspectable implementation and a controlled naive-RAG-versus-CRAG comparison.

It is based on Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling, [*Corrective Retrieval Augmented Generation* (arXiv:2401.15884)](https://arxiv.org/abs/2401.15884). It is a teaching adaptation, not a reproduction of the paper or a production-ready RAG system.

## Start here

You need Windows 11, a regular (non-free-threaded) Python 3.11+ installation with the Windows `py` launcher, and user environment variables named `AGNESAI_API_KEY` and `HF_TOKEN`. Restart the terminal or launcher after setting either variable. Never put their values in this repository.

1. Double-click `run.cmd`. It creates `.venv`, installs `requirements.txt`, registers the **Python (CRAG Tutorial)** kernel, and opens the main notebook.
2. In Jupyter, use that kernel and run the notebooks in order:
   1. [00_quick_check.ipynb](notebooks/00_quick_check.ipynb) — checks credential presence without printing values, pings the model, loads one HotpotQA row, and opens embedded Qdrant.
   2. [01_crag_tutorial.ipynb](notebooks/01_crag_tutorial.ipynb) — the main CRAG course.
   3. [02_naive_vs_crag_comparison.ipynb](notebooks/02_naive_vs_crag_comparison.ipynb) — compares both methods on the same 50 questions and ends by printing every answer pair for inspection.
3. Choose **Restart Kernel and Run All** for each notebook. The first live run needs network access and makes provider-metered model requests.

Only `agnes-3.0-flash` is used, through the official `openai` Python client and Chat Completions at `https://apihub.agnes-ai.com/v1` (`POST /v1/chat/completions`). The endpoint is fixed; `AGNES_BASE_URL` is listed in `.env.example` for visibility but does not override it. The tutorial uses no WSL, Docker, LangChain, Qdrant Cloud, Qdrant API key, or Agentic RAG.

> Run notebooks sequentially. Embedded Qdrant uses `data/qdrant`, so two kernels cannot own that path at the same time.

## Learn from zero to mastery

- [Tutorial guide](docs/tutorial-guide.md) starts with the difference between an LLM, retrieval, passages, vectors, grounding, and answer quality. It then maps each idea to a notebook checkpoint, failure mode, and mastery exercise.
- [Technical reference](docs/technical-reference.md) documents the architecture, public Python interfaces, data contracts, cache/index provenance, recovery behavior, and Windows launcher.
- [Evaluation and limitations](docs/evaluation-and-limitations.md) explains the 8-question smoke run, first-50 evaluation, paired comparison, manual audit, metrics, results, and what they do **not** prove.
- [STATUS.md](STATUS.md) is the dated evidence record: files, real executions, observed results, security checks, and remaining limits.

## What the tutorial uses

The dataset is [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), configuration `distractor`, split `validation`. The deterministic course slice contains 200 rows selected with seed 42. Each question supplies a candidate pool containing annotated supporting paragraphs and distractors. This is **not Meta Comprehensive RAG Benchmark**.

Retrieval is free and local: TF-IDF plus 256-dimensional TruncatedSVD vectors in embedded Qdrant at `data/qdrant`. Main retrieval filters on the current question ID; the notebook also demonstrates unfiltered slice-wide search. The CRAG path evaluates retrieved documents, chooses Correct/Incorrect/Ambiguous, selects exact source sentence strips, and generates only from those strips or abstains.

The paper uses a trained T5-large evaluator. This tutorial instead uses prompted Agnes judgements and fixed `.7` / `.3` routing thresholds. SVD retrieval, the small evaluation slice, cache state, and disabled-by-default web recovery all limit conclusions.

## Safe recovery

Versioned primitive caches and atomic per-question checkpoints live under ignored `data/cache/`. A stopped benchmark resumes completed rows. Warm caches can show zero **new** model calls; that does not mean the original run was free.

Common recovery paths:

- Use `AGNESAI_API_KEY`, not `AGNES_API_KEY`.
- For HotpotQA errors, check `HF_TOKEN`, network availability, and the exact dataset/config/split above.
- The client retries transient failures, including HTTP 429, with bounded backoff. Exhausted retries stop with a sanitized error rather than caching a fake success.
- JSON fences are tolerated; invalid evaluator or strip schemas raise and are not stored as successful output.
- For a Qdrant lock, shut down the other kernel that owns `data/qdrant`. Do not delete a live database to clear a lock.

## Evaluation in brief

The first 50 slice rows are evaluated with `k=3` and `allow_web=False`. The paired notebook holds the question, retriever, top-three passages, answer model, and web setting constant: naive RAG sends full passages to generation, while CRAG evaluates and refines them first.

The saved, method-aware manual audit found 4 rows where naive RAG was better, 2 where CRAG was better, 11 adequate ties, and 33 inadequate ties. This is descriptive review on one slice, not paper-level accuracy or a claim that either approach generally wins. Read the [full protocol and limitations](docs/evaluation-and-limitations.md) before interpreting the counts.

## Development checks

```powershell
uv run --no-project --python .venv\Scripts\python.exe python -m unittest discover -s tests -v
uvx ruff check src scripts tests --select E4,E7,E9,F
uv run --no-project --python .venv\Scripts\python.exe python scripts/audit_repository.py
uv run --no-project --python .venv\Scripts\python.exe python scripts/verify_notebook.py
```

`python scripts/verify_notebook.py --execute` runs every notebook and saves outputs; it can make live model requests. `python scripts/verify_launcher.py` creates an ignored fixture venv and installs dependencies. Do not run notebook execution while another kernel owns `data/qdrant`.

MIT license: [LICENSE](LICENSE).
