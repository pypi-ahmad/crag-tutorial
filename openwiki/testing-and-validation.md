---
type: Testing Guide
title: Testing and validation
description: The repository's mocked offline unit suite and structural notebook/repository checks, with the supported test command and the side effects of optional live verification.
tags: [testing, validation, unittest, notebooks]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:07:27.938Z
sources:
  - id: openwiki-source-2638013df91e0b992b7cf331
    resource: repo://scripts/audit_repository.py
  - id: openwiki-source-d0c02c9f901f4f8aa2a405ff
    resource: repo://scripts/verify_launcher.py
  - id: openwiki-source-3f8b63fc6b38a6e032c072df
    resource: repo://scripts/verify_notebook.py
  - id: openwiki-source-63fdccb791696f475b33ce12
    resource: repo://tests/test_core.py
generated: { by: "codex", at: "2026-09-23T14:07:27.938Z" }
---

# Testing and validation

The fast regression suite is [`tests/test_core.py`](../tests/test_core.py), written with Python's `unittest`. Its setup redirects selected module caches to temporary directories, and provider/model boundaries are patched for focused tests. It is designed to check local invariants without making live Agnes requests or writing to the production Qdrant path.

## Run the offline suite

From the repository root, after `run.cmd` has installed dependencies into `.venv`, use:

```powershell
uv run --no-project --python .venv\Scripts\python.exe python -m unittest discover -s tests -v
```

Coverage includes JSON response parsing and schema validation, score boundaries, exact-strip behavior, abstention, shared-hit comparison, review fingerprints, retry accounting, benchmark resume and settings invalidation, seed-sensitive slice reuse, in-slice smoke IDs, model restrictions, optional-web failure handling, and Qdrant index recovery. The checks do not validate live model quality, external dataset availability, or full notebook execution.

## Other repository checks

| Check | What it verifies | Execution boundary |
|---|---|---|
| `scripts/verify_notebook.py` | Notebook structure, markdown-before-code teaching checks, and code-cell syntax. | Default mode is structural. `--execute` runs cells and saves notebook outputs; live provider/data requests may occur. |
| `scripts/audit_repository.py` | Candidate files for configured-key/pattern matches, imported dependency coverage, notebook error outputs, and ignored-path rules. | Read-only local audit; it reports matching filenames/counts, not credential values, and does not audit remote history. |
| `scripts/verify_launcher.py` | The Windows launcher in an isolated fixture, including dependency installation and a local Jupyter route. | Creates an ignored fixture under `data/cache`, installs dependencies, starts a server, and cleans up its exact process. |
| Ruff command in the technical reference | Selected static lint rules over source, scripts, and tests. | Local lint; it does not execute the tutorial. |

The current implementation and test suite are the authority for behavior. `STATUS.md` and the evaluation guide preserve results from earlier notebook runs; they are not a substitute for re-running a live experiment. Avoid notebook execution while another kernel owns `data/qdrant`.

## Related pages

- [Quickstart](quickstart.md) gives the supported notebook startup sequence and runtime prerequisites.
- [Data and retrieval](data-and-retrieval.md) describes the invariants covered by the slice and index tests.
- [Evaluation and comparison](evaluation-and-comparison.md) separates descriptive experiment signals from regression evidence.
