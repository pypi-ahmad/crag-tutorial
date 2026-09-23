---
type: Quickstart
title: Quickstart
description: Windows setup and notebook run order for the CRAG tutorial, including credential boundaries, first-run network/model usage, and local Qdrant ownership.
tags: [quickstart, windows, notebooks, setup]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:07:27.938Z
sources:
  - id: openwiki-source-5e1999b7ac163b8349c0803a
    resource: repo://notebooks/00_quick_check.ipynb
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-0b361e5538aca06f3052fb20
    resource: repo://run.cmd
  - id: openwiki-source-d502c275990c6476221bf080
    resource: repo://src/config.py
generated: { by: "codex", at: "2026-09-23T14:07:27.938Z" }
---

# Quickstart

This repository is a Windows-native, notebook-led teaching implementation of Corrective Retrieval Augmented Generation. It adapts the paper for learning and inspection; it is not a paper reproduction or a production RAG service.

## Prerequisites and credentials

- Windows 11 with a regular (non-free-threaded) Python 3.11+ installation and the Windows `py` launcher.
- Set `AGNESAI_API_KEY` and `HF_TOKEN` in the Windows user environment. Restart the terminal or Jupyter process after changing them. Do not put credential values in repository files, notebook cells, screenshots, or logs.
- The tutorial fixes the model to `agnes-3.0-flash` and the Agnes API endpoint in source configuration. `AGNES_BASE_URL` appears in `.env.example` but is not used as a runtime override.

## Start the course

1. Double-click [`run.cmd`](../run.cmd). It creates `.venv` if needed, installs [`requirements.txt`](../requirements.txt), registers the `Python (CRAG Tutorial)` kernel, and starts Jupyter on the main notebook.
2. In Jupyter, select that kernel and run the notebooks sequentially:
   - [`00_quick_check.ipynb`](../notebooks/00_quick_check.ipynb) checks credential presence without printing values, pings the model, loads one HotpotQA row, and opens embedded Qdrant.
   - [`01_crag_tutorial.ipynb`](../notebooks/01_crag_tutorial.ipynb) builds and inspects the local CRAG path and descriptive runs.
   - [`02_naive_vs_crag_comparison.ipynb`](../notebooks/02_naive_vs_crag_comparison.ipynb) compares both evidence-treatment paths on the same first 50 questions and shows the reviewed rows.
3. Use **Restart Kernel and Run All** for each notebook when executing the course. A cold run needs network access and may make provider-metered model requests; inspect the notebook checkpoints before rerunning expensive cells.

The main data/index artifacts and caches live under `data/`. Embedded Qdrant uses `data/qdrant`; only one notebook kernel should own that path at a time. Shut down the current owner cleanly before running another notebook against it—do not delete the live store to clear a lock.

For an offline regression check without running the tutorial notebooks, use the command in [Testing and validation](testing-and-validation.md). Notebook execution and the paired experiment answer different questions from that mocked unit suite.

## Related pages

- [Architecture](architecture.md) maps the modules behind the notebook workflow.
- [Testing and validation](testing-and-validation.md) documents offline tests and the scope of optional live checks.
