---
type: Workflow
title: CRAG pipeline
description: How retrieved passages are evaluated, routed, reduced to exact source sentences, and passed to grounded generation, including cache, retry, and optional web behavior.
tags: [crag, workflow, grounding, failure-handling]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:07:27.938Z
sources:
  - id: openwiki-source-5197f4d221291125d753ceb2
    resource: repo://src/agnes_client.py
  - id: openwiki-source-d502c275990c6476221bf080
    resource: repo://src/config.py
  - id: openwiki-source-436c706a9ac345bef68fd07c
    resource: repo://src/crag.py
  - id: openwiki-source-b5af19f3aed01f87f5138d42
    resource: repo://src/pipeline.py
  - id: openwiki-source-3fb602cbf4598db27263bd8c
    resource: repo://src/search.py
  - id: openwiki-source-63fdccb791696f475b33ce12
    resource: repo://tests/test_core.py
generated: { by: "codex", at: "2026-09-23T14:07:27.938Z" }
---

# CRAG pipeline

The tutorial's corrective path is implemented by [`src/pipeline.py`](../src/pipeline.py), which composes the smaller evaluation, routing, refinement, and generation primitives in [`src/crag.py`](../src/crag.py). It is an inspectable teaching implementation; its module describes the evaluator as a prompt-based substitute, not the trained evaluator from the CRAG paper.

## Request flow

1. **Choose the hits.** `run_crag` accepts caller-supplied documents or invokes `retrieve.search`. The paired comparison supplies its already-retrieved hits so both comparison branches use the same passages.
2. **Evaluate each document.** The Agnes model returns a JSON object with a finite score in `[0, 1]`, a `relevant`/`irrelevant` label, and a non-empty explanation. The code validates this shape before it caches the result. Intermediate multi-hop facts count as relevant in the evaluator instructions; keyword overlap alone is not enough.
3. **Route on scores.** The maximum score determines the action: `Correct` at or above the upper threshold, `Incorrect` at or below the lower threshold, and `Ambiguous` between them. Defaults are `0.7` and `0.3`. An `Incorrect` route keeps no local documents; otherwise, documents scoring above the lower threshold are candidates for refinement.
4. **Keep exact evidence.** Each candidate is represented as sentences. The model selects sentence IDs, and Python returns only those exact source sentences. Invalid IDs fail rather than creating rewritten evidence. Repeated strips are de-duplicated before generation.
5. **Generate or abstain.** The generator receives the question and retained strips only. With no strips, it returns the fixed abstention without making a model call; otherwise, its prompt restricts the answer to supplied evidence and specifies the same abstention when that evidence is insufficient.

## Optional web recovery

`allow_web` defaults to `False`. When explicitly enabled, the runner tries web search only when the local route is not `Correct`; returned snippets are passed through the same exact-sentence refinement before they can reach generation. The search wrapper accepts only HTTP(S) result URLs and raises a sanitized failure rather than substituting fabricated evidence. The repository's reported notebooks keep this branch disabled.

## Requests, caches, and failures

- The Agnes wrapper reads `AGNESAI_API_KEY` from the process environment, restricts the model to the configured tutorial model, and retries transient API failures with bounded backoff. It reports request-attempt counts and avoids exposing provider response bodies or credentials in its error message.
- Evaluations, strips, and generations have separate content-addressed cache keys. A malformed evaluation or invalid strip selection is rejected before it is cached. JSON/text checkpoint writes use a temporary sibling file and atomic replacement.
- Benchmark runners checkpoint rows per question with their run settings. Failed attempts save a sanitized failure checkpoint; completed work can be reused on a compatible rerun.

## Tests that pin down behavior

[`tests/test_core.py`](../tests/test_core.py) checks score boundaries and invalid values, validation-before-cache behavior, rejection of invented/out-of-range strip IDs, empty-evidence abstention, model restrictions, counted retries, benchmark resume/invalidation, and the optional-search failure path. These are offline regression checks with mocked provider boundaries, not a live assessment of model answer quality.

## Related pages

- [Architecture](architecture.md) maps the wider data and control flow.
- [Data and retrieval](data-and-retrieval.md) explains how candidate passages are built and searched.
- [Evaluation and comparison](evaluation-and-comparison.md) covers benchmark checkpoints and the paired baseline.
- [Testing and validation](testing-and-validation.md) gives the supported local verification commands.
