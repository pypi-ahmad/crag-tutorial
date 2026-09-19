# Evaluation and limitations

This document explains what the repository measures, what the verified local run observed, and how to report those results accurately. It describes the tutorial implementation, not the full CRAG paper protocol.

Read [CRAG from zero to mastery](tutorial-guide.md) for concepts and [technical reference](technical-reference.md) for interfaces and cache mechanics.

## Evaluation protocol

### Dataset and retrieval

All reported tutorial runs use the deterministic 200-row HotpotQA `distractor` validation slice selected with seed 42. Main search is limited to the current question’s candidate pool. The index uses local TF-IDF plus 256-dimensional TruncatedSVD vectors in embedded Qdrant.

The eight-ID smoke set contains four bridge questions and four comparison questions when those labels are available. The larger descriptive evaluation uses the first 50 rows of the same slice.

| Setting | Reported value |
|---|---|
| Retrieval depth | `k=3` |
| Model | `agnes-3.0-flash` |
| Route thresholds | upper `.7`, lower `.3` |
| Web recovery | `allow_web=False` |
| Evaluator | Prompted Agnes relevance evaluator, not paper T5-large |

### What each evaluation does

| Run | Purpose | Cache/recovery behavior |
|---|---|---|
| Three diagnostic scenarios | Make retrieval/evaluator/refinement behavior inspectable on selected examples. | Primitive outputs are content-addressed. |
| Eight-ID smoke set | Quick integrated check across bridge and comparison questions. | Per-ID benchmark checkpoints resume. |
| First-50 CRAG run | Describe retrieval, actions, answer proxy, and call behavior on a small slice. | Per-ID checkpoints preserve completed rows after a failure. |
| Paired first-50 comparison | Compare naive RAG and CRAG using identical retrieved hits. | One paired checkpoint per question; cached rows report zero new calls. |

## Metrics: useful signals with strict limits

| Metric | Definition here | Correct interpretation | Not a valid interpretation |
|---|---|---|---|
| Gold-title recall@k | Fraction of annotated gold titles among retrieved top-`k` titles. | A retrieval coverage diagnostic. | End-to-end answer accuracy. |
| Any annotated title retrieved | At least one gold title is among hits. | A coarse signal that some supporting evidence appeared. | Proof that both hops or a complete answer are present. |
| Action by gold presence | Count of Correct/Incorrect/Ambiguous grouped by any gold hit. | A way to inspect evaluator routing against annotations. | Calibrated evaluator accuracy. |
| Answer substring match | Nonempty reference answer appears case-insensitively in answer text. | A simple automated triage signal. | Semantic correctness, support, or factuality. |
| Mean LLM calls | Actual new request attempts in that invocation. | Incremental provider activity after accounting for retries. | Lifetime cost when caches already exist. |
| Manual review | Human verdict attached to exact fingerprinted evidence and answers. | Qualitative assessment of these 50 rows. | Blinded evaluation or general performance estimate. |

The substring rule can miss correct aliases and can count an incidental or negated reference mention. It must be read with the failure gallery and manual review, not alone.

## Controlled naive-RAG versus CRAG comparison

The comparison notebook is designed to isolate the effect of correction after retrieval.

| Held constant | Both methods receive |
|---|---|
| Questions | Same first 50 deterministic-slice rows |
| Retriever and scope | Same TF-IDF/SVD Qdrant question-pool search |
| Retrieved evidence | Same top-three hit objects, including title/text/score |
| Generator and model | Same existing grounded answer function with `agnes-3.0-flash` |
| External recovery | Disabled |

| Different step | Naive RAG | CRAG |
|---|---|---|
| Evidence passed to generation | Complete text of all ranked hits | Exact strips selected after evaluation and routing |
| Evaluator/action | None | Per-document score plus Correct/Incorrect/Ambiguous action |
| Refinement | None | Sentence-ID selection from source text |

The protocol does **not** compare two independent retrieval systems or prove that CRAG improves RAG generally. It compares two evidence-treatment paths on one small, fixed retrieval setting.

## Observed local results

The following values are preserved in [STATUS.md](../STATUS.md) and the executed notebooks. They are observations from the stated protocol, not paper-level accuracy.

### First-50 CRAG evaluation

| Measure | Observed value |
|---|---:|
| Mean gold-title recall@3 | 0.42 |
| Questions with any annotated title retrieved | 34 / 50 |
| Literal answer substring match | 13 / 50 (0.26) |
| Initial repaired invocation: new request attempts | 252 |
| Initial repaired invocation: mean attempts | 5.04 |
| Final warm invocation: cached rows / new requests | 50 / 0 |

Evaluator actions grouped by whether any annotated title appeared in hits:

| Any gold title in hits? | Correct | Incorrect | Ambiguous |
|---|---:|---:|---:|
| Yes | 30 | 4 | 0 |
| No | 3 | 12 | 1 |

Use the surprising cells as teaching material. `Correct` without an annotated title can indicate a useful non-annotated passage, a label limitation, or an evaluator error. `Incorrect` despite an annotated title can reflect a weak or partial passage, a false-negative evaluator judgement, or a threshold limitation. The counts do not distinguish among these causes.

### Paired answer-quality review

| Measure | Naive RAG | CRAG |
|---|---:|---:|
| Literal substring match | 0.30 | 0.26 |
| Abstention rate | 0.66 | 0.72 |
| Manually correct | 15 | 13 |
| Manually partially correct | 1 | 0 |
| Manually incorrect | 1 | 1 |
| Manual abstentions | 33 | 36 |

Pairwise review marked naive RAG better on 4 rows, CRAG better on 2, both adequate on 11, and both inadequate on 33. Both methods shared mean gold-title recall@3 of 0.42 because retrieval was intentionally held constant.

The audit is **method-aware, not blinded**: the reviewer saw the question, reference answer, exact method evidence, answers, and method identity. The review artifact validates each row’s fingerprint before it is summarized, but fingerprint validation protects linkage—not reviewer independence or generalizability.

## Failure patterns worth studying

The completed review and failure gallery identify patterns that readers should inspect directly in notebook outputs:

- Retrieval gaps dominated inadequate ties. When the shared top-three hits omitted the second multi-hop fact, neither evidence-treatment path could establish a complete grounded answer.
- The Plymouth Barracuda and A.P. Møller rows show that sentence splitting or strip selection can discard a bridge that full-passage naive RAG retained.
- The Bill Dudman and NBA rows show that a cautious generation prompt can abstain despite a decisive-looking strip.
- Aliases and short reference answers such as `no` make literal substring matching brittle.
- One reviewed date conflict shows why a dataset reference and retrieved evidence require human inspection before they become a single automatic label.

These examples motivate diagnosis, not a change to the reported aggregate results. A reader should open the final comparison cell, locate the named rows, and inspect exact evidence before drawing a conclusion.

## Limits of this tutorial

- The vector retriever is a lexical TF-IDF/SVD baseline, not a learned semantic retriever.
- The evaluator is prompted Agnes, not the CRAG paper’s trained T5-large evaluator. The thresholds are heuristics.
- The sample is only the first 50 rows of a deterministic 200-row slice. It is not a representative benchmark estimate.
- Web recovery is disabled for reported runs. This tutorial does not evaluate web search quality.
- Cache state changes apparent new-request counts. A warm run’s zero calls do not erase the calls required to create its cache.
- Grounded prompting and exact-strip copying improve traceability but do not formally guarantee factual correctness, completeness, or safety.
- The manual audit is useful qualitative evidence but is method-aware, unblinded, and limited to these outputs.
- The launcher and notebooks were verified in a local Windows session. Browser rendering was not visually inspected as part of that verification.

## How to report these results responsibly

Use language such as:

> In this Windows tutorial’s first-50 HotpotQA slice, using fixed question-pool retrieval and disabled web recovery, the method-aware manual audit found more naive-better rows than CRAG-better rows. The result is descriptive and does not estimate paper-level or production performance.

Avoid claims such as “CRAG is more accurate,” “the paper was reproduced,” “the evaluator is calibrated,” or “zero warm-cache calls mean zero cost.”

## Reproduce and inspect

1. Run notebook 01 to prepare the deterministic slice, index, and CRAG checkpoints.
2. Run notebook 02 to create or validate paired checkpoints and the fingerprinted review artifact.
3. Read every block in notebook 02’s final output: question, reference answer, naive answer, CRAG answer, verdict, and rationale.
4. Treat a mismatch between an automatic proxy and manual verdict as an investigation prompt, not an automatic score correction.

For exact commands and live-execution cautions, see the [README](../README.md) and [technical reference](technical-reference.md).
