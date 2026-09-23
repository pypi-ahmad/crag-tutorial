---
type: Architecture
title: Architecture
description: The repository is a local, notebook-led CRAG tutorial whose data, index, retrieval, answer pipeline, and paired comparison are implemented as separate Python modules.
tags: [architecture, data-flow, components]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:07:27.938Z
sources:
  - id: openwiki-source-01da97ee47cdf6f529c26ced
    resource: repo://src/comparison.py
  - id: openwiki-source-d502c275990c6476221bf080
    resource: repo://src/config.py
  - id: openwiki-source-9378b56cb922743eb245e061
    resource: repo://src/data_hotpot.py
  - id: openwiki-source-b5af19f3aed01f87f5138d42
    resource: repo://src/pipeline.py
  - id: openwiki-source-e2014ebb95fa196dc061f9c7
    resource: repo://src/qdrant_store.py
  - id: openwiki-source-91a9f6fc5fe099224c443ff8
    resource: repo://src/retrieve.py
  - id: openwiki-source-63fdccb791696f475b33ce12
    resource: repo://tests/test_core.py
generated: { by: "codex", at: "2026-09-23T14:07:27.938Z" }
---

# Architecture

The tutorial is organized as a data path rather than a service deployment. Notebook entry points prepare a deterministic HotpotQA slice, build a local vector index, retrieve evidence, and pass it through either the corrective pipeline or the paired naive-RAG comparison. The relevant implementation boundaries live in `src/` and are exercised by the offline regression suite in [`tests/test_core.py`](../tests/test_core.py).

## Component flow

| Stage | Owner | Responsibility |
|---|---|---|
| Configuration and local state | [`src/config.py`](../src/config.py), [`src/cache.py`](../src/cache.py) | Resolve repository-local data, cache, and Qdrant paths; provide content fingerprints and atomic JSON/text checkpoints. |
| Dataset preparation | [`src/data_hotpot.py`](../src/data_hotpot.py) | Load HotpotQA and materialize a seeded slice with a manifest and an in-slice smoke set. |
| Index and retrieval | [`src/qdrant_store.py`](../src/qdrant_store.py), [`src/retrieve.py`](../src/retrieve.py) | Fit TF-IDF/SVD vectors, persist paragraph points in embedded Qdrant, and retrieve either from one question's candidate pool or across the indexed collection. |
| Corrective answer path | [`src/crag.py`](../src/crag.py), [`src/pipeline.py`](../src/pipeline.py), [`src/agnes_client.py`](../src/agnes_client.py) | Validate model evaluations, route by score, select exact evidence sentences, generate or abstain, and checkpoint descriptive runs. |
| Paired baseline | [`src/comparison.py`](../src/comparison.py) | Run naive generation and CRAG over the same retrieved hits and bind manual review to the compared outputs. |

The configured runtime artifacts are rooted under `data/`: cached records and model-derived checkpoints go under `data/cache`, while embedded Qdrant uses `data/qdrant`. The API key and dataset token are read from the process environment; they are not part of the repository-local artifact flow.

## Data and control flow

1. `build_slice` loads the HotpotQA `distractor` validation split, shuffles with the requested seed, selects the requested number of rows, and writes a manifest tied to a content fingerprint. Its smoke IDs are selected from that slice.
2. `index_slice` turns context paragraphs into unit-normalized 256-dimensional TF-IDF/SVD vectors and stores them in an embedded Qdrant collection. `retrieve.search` transforms the question and can constrain results by `question_id`; omitting that ID searches the collection-wide demonstration corpus.
3. `run_crag` accepts supplied hits or calls `search`, evaluates each hit, routes the evidence, refines retained documents into exact source sentences, optionally tries web snippets when explicitly enabled for a non-`Correct` route, and asks the grounded generator to answer or abstain.
4. `run_comparison_50` retrieves once for each selected record, then gives those same hits to the naive full-passage path and the CRAG path. That shared retrieval makes the intended comparison about evidence treatment after retrieval, not two independent searches.

## Boundaries and failure handling

- The vector representation is a local TF-IDF plus TruncatedSVD baseline, not a learned semantic embedding service.
- Embedded Qdrant is a local, disk-backed store. Its client is a singleton per path and has an explicit close function for releasing Windows file locks.
- Model requests are owned by the Agnes client wrapper, which restricts the configured model and bounds retries. Cache helpers write checkpoints atomically; the benchmark runners use settings and data fingerprints to decide whether prior rows can be reused.
- Offline tests patch external/model boundaries and use temporary paths. They verify selected invariants, but do not replace notebook execution or a live provider/data check.

## Related pages

- [Data and retrieval](data-and-retrieval.md) follows slice provenance through indexing and search.
- [CRAG pipeline](crag-pipeline.md) traces scoring, routing, refinement, generation, and recovery.
- [Evaluation and comparison](evaluation-and-comparison.md) describes the shared-hit baseline and its interpretation.
- [Testing and validation](testing-and-validation.md) maps the regression suite and repository checks.
