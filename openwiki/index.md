---
okf_version: "0.2"
---

# Files

- [Architecture](architecture.md) - The repository is a local, notebook-led CRAG tutorial whose data, index, retrieval, answer pipeline, and paired comparison are implemented as separate Python modules.
- [CRAG pipeline](crag-pipeline.md) - How retrieved passages are evaluated, routed, reduced to exact source sentences, and passed to grounded generation, including cache, retry, and optional web behavior.
- [Data and retrieval](data-and-retrieval.md) - How HotpotQA records become a fingerprinted tutorial slice and then a local TF-IDF/SVD Qdrant index searched either within a question pool or across the full slice.
- [Evaluation and comparison](evaluation-and-comparison.md) - The tutorial's first-50 paired comparison shares retrieval results between naive RAG and CRAG, stores fingerprinted rows and reviews, and reports descriptive proxies rather than general accuracy.
- [Quickstart](quickstart.md) - Windows setup and notebook run order for the CRAG tutorial, including credential boundaries, first-run network/model usage, and local Qdrant ownership.
- [Testing and validation](testing-and-validation.md) - The repository's mocked offline unit suite and structural notebook/repository checks, with the supported test command and the side effects of optional live verification.
