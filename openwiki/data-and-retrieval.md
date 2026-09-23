---
type: Data Pipeline
title: Data and retrieval
description: How HotpotQA records become a fingerprinted tutorial slice and then a local TF-IDF/SVD Qdrant index searched either within a question pool or across the full slice.
tags: [data, retrieval, qdrant, provenance]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:07:27.938Z
sources:
  - id: openwiki-source-9378b56cb922743eb245e061
    resource: repo://src/data_hotpot.py
  - id: openwiki-source-e2014ebb95fa196dc061f9c7
    resource: repo://src/qdrant_store.py
  - id: openwiki-source-91a9f6fc5fe099224c443ff8
    resource: repo://src/retrieve.py
  - id: openwiki-source-63fdccb791696f475b33ce12
    resource: repo://tests/test_core.py
generated: { by: "codex", at: "2026-09-23T14:07:27.938Z" }
---

# Data and retrieval

This path supplies the CRAG pipeline with ranked titled paragraphs. Dataset preparation lives in [`src/data_hotpot.py`](../src/data_hotpot.py); indexing and vector encoding live in [`src/qdrant_store.py`](../src/qdrant_store.py); query-time search lives in [`src/retrieve.py`](../src/retrieve.py).

## Build and reuse the dataset slice

The loader targets the HotpotQA `distractor` validation split and accepts an explicit token or reads `HF_TOKEN`. `build_slice(n=200, seed=42)` shuffles with that seed and selects `n` records. Each record retains the question, reference answer, question metadata, supporting titles when available, and its context paragraphs with their original sentence lists.

The JSONL slice has a companion manifest containing the dataset/config/split, requested size, seed, schema version, and record fingerprint. If the settings match and the stored rows reproduce the fingerprint, the slice is reused; otherwise it is rebuilt. A separate smoke file contains up to four `bridge` and four `comparison` IDs when present, then fills any remaining places from other IDs in the same slice. Missing gold annotations remain unknown rather than being treated as negative labels.

## Build the local index

`TFIDFSVDEncoder` fits a scikit-learn `TfidfVectorizer` followed by deterministic `TruncatedSVD`; it pads small fitted representations to 256 dimensions and unit-normalizes vectors. `index_slice` creates one Qdrant point per context paragraph, with the question ID, title, text, and gold flag in the payload. The collection is embedded on disk under `data/qdrant`; the encoder and index manifest are cached under `data/cache`.

Index reuse is conditional: point count, vector size, saved encoder provenance, and a signature covering the corpus and implementation must agree. If they do not, the encoder and vectors are rebuilt before the old collection is replaced. Qdrant's disk client is a process singleton for its path; close it before switching paths or before another notebook kernel opens the same directory.

## Search scope

`search(query, k=5, question_id=None)` transforms the query with the saved encoder and verifies that its corpus signature matches the indexed collection. `k` must be a positive integer. Passing `question_id` adds a Qdrant payload filter and limits candidates to that question's paragraph pool—the normal benchmark path. Omitting it searches across the entire indexed slice, which the tutorial uses as a separate demonstration.

Search results carry the paragraph title and text, vector score, and stored payload. A matching gold title is a retrieval diagnostic; it does not establish that all facts needed for a multi-hop answer were retrieved.

## Regression coverage

The offline suite verifies that slice caching depends on the seed, unknown labels stay unknown, and smoke IDs remain within the slice. Its Qdrant test uses an in-memory client to check question filtering, whole-collection search, idempotent reuse, missing-encoder recovery, changed corpus contents, and shrinking a collection when records are removed. See [`tests/test_core.py`](../tests/test_core.py).

## Related pages

- [Architecture](architecture.md) places these components in the full request flow.
- [CRAG pipeline](crag-pipeline.md) describes how ranked passages become grounded answers.
- [Testing and validation](testing-and-validation.md) explains the scope of these offline checks.
