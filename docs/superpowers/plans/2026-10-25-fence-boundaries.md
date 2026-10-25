# Markdown Fence Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent code-fence contents from creating false source sections and citation IDs.

**Architecture:** Retain the existing scanner and remember the complete opening marker. Validate captured suffixes separately for opening and closing lines; exercise actual SQLite retrieval alongside chunking boundaries.

**Tech Stack:** Python 3.12+, standard-library regex/SQLite, existing pytest/Hypothesis/Ruff/mypy toolchain.

**Spec:** `docs/superpowers/specs/2026-10-25-fence-boundaries-design.md`

## Global Constraints

- Work only in the task-owned Citecheck clone and publish a scoped PR for parent review; do not merge.
- Code contents remain searchable under their real enclosing section.
- Preserve existing handbook IDs and offline evaluation output.
- No runtime dependency, index format or provider behavior change.

## Review Focus

- Shorter/wrong-character fence lines cannot expose headings inside code.
- A closer with text or non-ASCII whitespace after its marker remains code.
- Equal/longer closers with zero to three spaces and trailing ASCII spaces/tabs restore real headings.
- Invalid backtick info strings cannot hide real document headings; tilde info strings may contain backticks.
- Actual indexed results retain correct IDs, enclosing titles and source line spans.

### Task 1: Fence boundaries and citation regressions

**Files:** Modify `src/citecheck/chunking.py`, `tests/test_chunking.py`, `tests/test_index.py`, `docs/DESIGN.md`, `CHANGELOG.md`.

**Interfaces:** Existing `chunk_text(text, doc_path, *, markdown, max_chunk_chars)`, `build_index(corpus_dir, index_path)` and `retrieve(index, question, *, k, min_score)` remain unchanged.

- [ ] Add parameterized chunking regressions for opener/closer marker, length, indentation and suffix rules, asserting section IDs, source line spans and retained literal code.
- [ ] Add real SQLite indexing/retrieval cases showing code terms cite the real enclosing section and later policy headings remain searchable.
- [ ] Run targeted tests; expect failures exposing early closure and invalid opener recognition.
- [ ] Capture trailing text and remember the full opening marker; accept only matching, long-enough, whitespace-only closers and valid opening info strings.
- [ ] Run targeted tests, then all existing lint/format/type/coverage checks; expect a green suite and coverage >=90%.
- [ ] Run actual offline handbook index → expense ask → eval; preserve 25 documents/103 chunks and published evaluation output.
- [ ] Commit scoped source/tests/docs, refresh public main, push feature branch, create/attach PR and send exact head to parent for review.
