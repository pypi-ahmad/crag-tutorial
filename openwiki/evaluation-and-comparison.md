---
type: Evaluation Protocol
title: Evaluation and comparison
description: The tutorial's first-50 paired comparison shares retrieval results between naive RAG and CRAG, stores fingerprinted rows and reviews, and reports descriptive proxies rather than general accuracy.
tags: [evaluation, comparison, metrics, limitations]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:07:27.938Z
sources:
  - id: openwiki-source-c5c14bcd9f1bd1802ff48f88
    resource: repo://docs/evaluation-and-limitations.md
  - id: openwiki-source-01da97ee47cdf6f529c26ced
    resource: repo://src/comparison.py
  - id: openwiki-source-b5af19f3aed01f87f5138d42
    resource: repo://src/pipeline.py
  - id: openwiki-source-63fdccb791696f475b33ce12
    resource: repo://tests/test_core.py
generated: { by: "codex", at: "2026-09-23T14:07:27.938Z" }
---

# Evaluation and comparison

The repository has two related evaluation paths: a resumable CRAG run and a paired naive-RAG-versus-CRAG comparison. Their outputs are intended for inspection and diagnosis. The implementation and repository evaluation guide explicitly distinguish these small tutorial measurements from paper-level accuracy or general performance claims.

## Paired protocol

`run_comparison_50` takes the first 50 records, searches once for each question using the question-pool filter, and passes the exact same hit list to both methods. Naive RAG sends the complete retrieved passage text to the existing grounded generator. The CRAG branch evaluates and refines those same hits before generation. Web recovery is disabled by default.

| Held constant | Method difference |
|---|---|
| Same first 50 records and one top-`k` question-pool search per record | Naive uses the full text of ranked hits; CRAG scores, routes, and selects exact evidence sentences. |
| Same generator and configured model | Only the evidence-treatment path differs by design. |
| Same retrieved hit objects | The comparison does not run separate retrievers for the two methods. |

## Checkpoints and manual review

Each paired row is checkpointed under a cache root derived from the run settings and selected records. Reuse additionally checks the stored row ID and its comparison fingerprint. The fingerprint binds the question and reference, retrieved title/text pairs, naive answer/evidence, and CRAG answer/strips. A manual review is valid only when its fingerprint still matches and its categorical fields satisfy the supported values; changed outputs therefore make the old review stale.

The review artifact records qualitative judgements such as correctness, evidence support, preference, error types, and rationale. It is linked to exact outputs, but the fingerprint verifies linkage rather than reviewer independence or blinding.

## Read metrics as diagnostics

- **Gold-title recall@k** measures annotated supporting-title coverage among retrieved titles; it does not establish that every required fact was retrieved.
- **Answer substring match** checks whether the non-empty reference answer appears case-insensitively in the generated text. It can miss aliases and can count incidental or negated mentions.
- **Abstention and action counts** summarize routing and answer behavior, not calibrated correctness.
- **LLM request counts** describe new attempts for that invocation. Cached results can reduce a warm run's count without erasing the requests that produced those results.
- **Manual review** gives a human judgement of the saved examples. This small, method-aware audit is not a blinded or statistically generalizable estimate.

The first-50 slice is a descriptive tutorial sample. Treat a disagreement between an automatic signal and a manual verdict as a reason to inspect the evidence, not as an automatic correction to either result. `STATUS.md` preserves prior local run observations; those saved observations are not evidence that a live run was repeated now.

## Related pages

- [CRAG pipeline](crag-pipeline.md) explains the evidence-processing path used by the CRAG arm.
- [Testing and validation](testing-and-validation.md) separates mocked regression evidence from notebook execution and recorded experiment results.
