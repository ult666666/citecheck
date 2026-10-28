# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Preserve Markdown code-fence boundaries when assigning section titles and citation IDs:
  shorter markers, incorrect closing suffixes and invalid backtick info strings no longer
  expose code headings or hide genuine document sections. Regression tests cover real SQLite
  indexing and retrieval, source line spans, and both backtick and tilde fences.
- `citecheck index` prints "1 document" and "1 chunk" instead of "1 documents" and "1 chunks".

## [0.1.0] - 2026-10-23

### Added

- Ingestion of Markdown and text files into chunks with stable IDs
  (`path#heading-slug`, `:2` for extra parts) and 1-based source line spans; YAML front matter
  is skipped.
- A SQLite FTS5 index (Porter stemming, external-content table, atomic rebuilds) ranked with
  BM25, safe query building from free-text questions, and a score threshold below which
  citecheck abstains without a model call.
- Cited answering through a strict `submit_answer` tool (sentences, citation IDs and an
  `insufficient_evidence` flag) behind a small `ModelClient` interface, with an Anthropic SDK
  adapter (explicit effort, prompt-cache breakpoint, server-side refusal fallback, typed errors)
  and a fixture client that replays hand-written synthetic responses offline.
- A sentence-level checker: citations to passages that were not retrieved are rejected, support
  is the share of content words found in the cited passages, a missing number makes a sentence
  unsupported, and labels use configurable thresholds. An optional LLM-judge pass is reported
  next to the lexical label.
- Token, cost (user-supplied price table, `Decimal` arithmetic) and per-stage latency accounting.
- A CLI (`index`, `search`, `ask`, `eval`, `serve`), a FastAPI JSON endpoint and an accessible
  HTML page (optional `[api]` extra).
- A fictional 25-document handbook, 36 labeled questions and synthetic fixtures, and
  `citecheck eval` with recall@k, citation precision and recall, the unsupported-sentence rate,
  abstention accuracy and checker-vs-human agreement.
- CI on Python 3.12 and 3.13: ruff, mypy (strict), pytest with a 90% coverage floor and an
  offline CLI smoke test.

[Unreleased]: https://github.com/ult666666/citecheck/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ult666666/citecheck/releases/tag/v0.1.0
