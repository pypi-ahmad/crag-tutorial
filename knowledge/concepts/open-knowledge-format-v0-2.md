---
type: Reference
title: Open Knowledge Format v0.2
description: OKF defines a portable knowledge bundle using Markdown concepts and YAML frontmatter, with optional provenance, trust, lifecycle, and attested-computation fields.
tags: [okf, knowledge-format, markdown, provenance]
status: draft
generated:
  by: okf-skill/0.2
  at: 2026-09-23T19:44:40+05:30
sources:
  - id: okf-spec-v02
    resource: "https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md"
    title: Open Knowledge Format v0.2 specification
  - id: okf-google-introduction
    resource: "https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing/"
    title: Google Cloud introduction to Open Knowledge Format
  - id: okf-canonical-move
    resource: "https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/README.md"
    title: Knowledge Catalog OKF directory README
  - id: okf-frozen-spec
    resource: "https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md"
    title: Frozen Knowledge Catalog copy of the OKF specification
  - id: okf-homepage
    resource: "https://okf.md/"
    title: OKF homepage
  - id: openwiki-okf-announcement
    resource: "https://www.langchain.com/blog/openwiki-0-2-adds-okf-support"
    title: OpenWiki 0.2 OKF support announcement
---

# Open Knowledge Format v0.2

Open Knowledge Format (OKF) is a vendor-neutral convention for storing portable knowledge as UTF-8 Markdown files with YAML frontmatter. A bundle can live in a Git repository or a subdirectory; each concept is one Markdown document, and its path identifies it within the bundle.[^okf-spec][^okf-google-introduction]

## Core shape

- `type` is the only required concept frontmatter field. `title`, `description`, `resource`, and `tags` are recommended or optional; producers choose the concept taxonomy and body structure.[^okf-spec]
- Standard Markdown links express relationships between concepts. `index.md` and `log.md` are reserved for directory navigation and update history; indexes are optional.[^okf-spec]
- Version 0.2 makes source provenance (`sources`), producer and edit time (`generated`), verification (`verified`), lifecycle (`status`, `stale_after`), and attested computations part of the format. These fields are optional; their absence does not make an ordinary concept non-conformant.[^okf-spec]
- Verification metadata is an advisory trust signal, not cryptographic proof. OKF defines an attested-computation contract but does not itself execute the computation or standardize every runtime detail.[^okf-spec]

The format standardizes a small interoperability surface rather than a fixed ontology, runtime, storage system, or domain schema. Its purpose is to let independent producers and consumers exchange structured knowledge without requiring a shared SDK or service.[^okf-spec][^okf-google-introduction]

## Reading the supplied references

Use the [standalone GoogleCloudPlatform specification](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md) for normative v0.2 rules. The supplied `knowledge-catalog/okf` directory is explicitly marked as a frozen snapshot and directs readers to the standalone repository; its copied `SPEC.md` identifies itself as v0.2, but should not be treated as the maintained source.[^okf-canonical-move][^okf-frozen-spec]

The Google Cloud article is the format's introductory announcement and describes the initial v0.1 release. The supplied `okf.md` homepage also labels itself v0.1. They are useful for motivation and discovery, but their version-specific examples should not override the current specification.[^okf-google-introduction][^okf-homepage][^okf-spec]

LangChain's July 2026 article describes OpenWiki 0.2 adopting OKF-style metadata and generated indexes/logs. It is product-integration context, not the normative OKF specification; consult the standalone spec for current field names and reserved filenames.[^openwiki-okf-announcement][^okf-spec]

[^okf-spec]: [Open Knowledge Format v0.2 specification](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)
[^okf-google-introduction]: [Introducing the Open Knowledge Format](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing/)
[^okf-canonical-move]: [Knowledge Catalog OKF directory README](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/README.md)
[^okf-frozen-spec]: [Frozen Knowledge Catalog OKF specification copy](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
[^okf-homepage]: [OKF homepage](https://okf.md/)
[^openwiki-okf-announcement]: [OpenWiki 0.2 is adopting the OKF support](https://www.langchain.com/blog/openwiki-0-2-adds-okf-support)
