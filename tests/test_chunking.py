import doctest
import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from citecheck import chunking
from citecheck.chunking import chunk_text, load_corpus, slugify
from citecheck.errors import ConfigError, CorpusError

CorpusFactory = Callable[[dict[str, str]], Path]


def test_docstring_examples_hold() -> None:
    result = doctest.testmod(chunking)
    assert result.attempted > 0
    assert result.failed == 0


def test_sections_get_stable_ids_titles_and_line_spans() -> None:
    text = (
        "---\nfictional: true\n---\n"  # lines 1-3: front matter, skipped
        "# Handbook\n"  # line 4
        "\n"
        "Intro paragraph.\n"  # line 6
        "\n"
        "## Leave\n"  # line 8
        "\n"
        "Leave text line one\n"  # line 10
        "line two.\n"  # line 11
        "\n"
        "### Sick days\n"  # line 13
        "\n"
        "Sick text.\n"  # line 15
    )
    title, chunks = chunk_text(text, "a/b.md", markdown=True)
    assert title == "Handbook"
    assert [c.chunk_id for c in chunks] == ["a/b.md#handbook", "a/b.md#leave", "a/b.md#sick-days"]
    assert [c.title for c in chunks] == [
        "Handbook",
        "Handbook > Leave",
        "Handbook > Leave > Sick days",
    ]
    assert [(c.start_line, c.end_line) for c in chunks] == [(4, 6), (8, 11), (13, 15)]
    assert chunks[1].text == "Leave text line one\nline two."
    assert "fictional" not in " ".join(c.text for c in chunks)


def test_line_spans_point_at_the_source_lines() -> None:
    text = "# T\n\nA one.\nA two.\n\n## S\n\nB.\n\nC one.\n"
    _, chunks = chunk_text(text, "t.md", markdown=True)
    source = text.split("\n")
    for chunk in chunks:
        span = "\n".join(source[chunk.start_line - 1 : chunk.end_line])
        for line in chunk.text.split("\n"):
            assert line in span


def test_heading_without_text_is_skipped_but_stays_in_the_path() -> None:
    title, chunks = chunk_text("# Benefits\n## Health\n\nText.\n", "x.md", markdown=True)
    assert title == "Benefits"
    assert [(c.chunk_id, c.title) for c in chunks] == [("x.md#health", "Benefits > Health")]


def test_duplicate_headings_get_numbered_slugs() -> None:
    text = "## Step\n\na\n\n## Step\n\nb\n\n## Step 2\n\nc\n"
    _, chunks = chunk_text(text, "x.md", markdown=True)
    assert [c.chunk_id for c in chunks] == ["x.md#step", "x.md#step-2", "x.md#step-2-2"]


def test_code_fences_hide_headings_and_keep_blank_lines() -> None:
    text = "## Setup\n\n```\n# not a heading\n\nstill code\n```\n\nAfter.\n"
    _, chunks = chunk_text(text, "x.md", markdown=True)
    assert [c.chunk_id for c in chunks] == ["x.md#setup"]
    assert chunks[0].text == "```\n# not a heading\n\nstill code\n```\n\nAfter."


def test_a_fence_closes_only_with_its_own_marker() -> None:
    text = "## A\n\n~~~\n```\n# still code\n~~~\n\n## B\n\nb\n"
    _, chunks = chunk_text(text, "x.md", markdown=True)
    assert [c.chunk_id for c in chunks] == ["x.md#a", "x.md#b"]


def test_long_sections_split_into_numbered_parts() -> None:
    paragraph = ("word " * 30).strip()  # 149 characters
    text = "## Section\n\n" + "\n\n".join([paragraph] * 5) + "\n"
    _, chunks = chunk_text(text, "x.md", markdown=True, max_chunk_chars=400)
    assert [c.chunk_id for c in chunks] == ["x.md#section", "x.md#section:2", "x.md#section:3"]
    assert all(len(c.text) <= 400 for c in chunks)
    assert chunks[0].start_line == 1  # the first part starts at the heading
    assert chunks[1].start_line == 7  # later parts start at their first paragraph


def test_an_oversized_paragraph_becomes_its_own_chunk() -> None:
    text = "## S\n\n" + "x" * 500 + "\n"
    _, chunks = chunk_text(text, "x.md", markdown=True, max_chunk_chars=200)
    assert len(chunks) == 1
    assert len(chunks[0].text) == 500


def test_plain_text_files_have_no_headings() -> None:
    title, chunks = chunk_text(
        "# not a heading\n\nsecond paragraph\n", "notes.txt", markdown=False
    )
    assert title == "notes"
    assert [c.chunk_id for c in chunks] == ["notes.txt#top"]
    assert chunks[0].title == "notes"
    assert chunks[0].text == "# not a heading\n\nsecond paragraph"


def test_closing_hashes_and_empty_headings() -> None:
    _, chunks = chunk_text("## Title ##\n\nText\n\n##\n\nMore\n", "x.md", markdown=True)
    assert [c.chunk_id for c in chunks] == ["x.md#title", "x.md#section"]
    assert chunks[0].title == "Title"


def test_text_before_the_first_heading_uses_the_top_slug() -> None:
    title, chunks = chunk_text("Preamble.\n\n# Title\n\nBody.\n", "x.md", markdown=True)
    assert title == "Title"
    assert [(c.chunk_id, c.title) for c in chunks] == [
        ("x.md#top", "Title"),
        ("x.md#title", "Title"),
    ]


def test_windows_newlines() -> None:
    _, chunks = chunk_text("# T\r\n\r\nA\r\n", "x.md", markdown=True)
    assert [(c.start_line, c.end_line, c.text) for c in chunks] == [(1, 3, "A")]


def test_unterminated_front_matter_is_an_error() -> None:
    with pytest.raises(CorpusError, match="never closed"):
        chunk_text("---\nkey: value\n# Title\n", "x.md", markdown=True)


@pytest.mark.parametrize("limit", [10, 50_000])
def test_chunk_size_limit_is_validated(limit: int) -> None:
    with pytest.raises(ConfigError):
        chunk_text("text", "x.md", markdown=True, max_chunk_chars=limit)


def test_slugify_keeps_unicode_letters() -> None:
    assert slugify("Café Hours") == "café-hours"


def test_load_corpus_walks_folders_in_sorted_order(make_corpus: CorpusFactory) -> None:
    root = make_corpus(
        {
            "b.md": "# B\n\nbee\n",
            "sub/a.txt": "ay\n",
            "a.markdown": "# A\n\nay\n",
            ".hidden.md": "# H\n\nhidden\n",
            ".git/config.md": "# G\n\ngit\n",
            "notes.pdf": "not indexed",
        }
    )
    documents = load_corpus(root)
    assert [d.path for d in documents] == ["a.markdown", "b.md", "sub/a.txt"]
    raw = (root / "b.md").read_bytes()
    assert documents[1].sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlink support")
def test_symbolic_links_are_skipped(make_corpus: CorpusFactory, tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("# Secret\n\nsecret\n", encoding="utf-8")
    root = make_corpus({"real.md": "# Real\n\nreal\n"})
    (root / "link.md").symlink_to(outside)
    assert [d.path for d in load_corpus(root)] == ["real.md"]


def test_byte_order_mark_is_ignored(make_corpus: CorpusFactory) -> None:
    root = make_corpus({"x.md": "﻿# Title\n\nBody.\n"})
    assert load_corpus(root)[0].title == "Title"


def test_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="not found"):
        load_corpus(tmp_path / "nope")


def test_a_file_is_not_a_folder(make_corpus: CorpusFactory) -> None:
    root = make_corpus({"x.md": "x"})
    with pytest.raises(CorpusError, match="not a folder"):
        load_corpus(root / "x.md")


def test_folder_without_supported_files(make_corpus: CorpusFactory) -> None:
    root = make_corpus({"x.pdf": "x"})
    with pytest.raises(CorpusError, match=r"no \.md"):
        load_corpus(root)


def test_folder_with_only_empty_files(make_corpus: CorpusFactory) -> None:
    root = make_corpus({"a.md": "", "b.txt": "\n\n"})
    with pytest.raises(CorpusError, match="no text"):
        load_corpus(root)


def test_invalid_utf8_names_the_file(make_corpus: CorpusFactory) -> None:
    root = make_corpus({"ok.md": "# Ok\n\nfine\n"})
    (root / "bad.md").write_bytes(b"# Bad\n\n\xff\xfe broken\n")
    with pytest.raises(CorpusError, match=r"bad\.md: not valid UTF-8"):
        load_corpus(root)


def test_oversized_files_are_rejected(
    make_corpus: CorpusFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_corpus({"big.md": "# Big\n\n" + "x" * 200})
    monkeypatch.setattr(chunking, "MAX_FILE_BYTES", 100)
    with pytest.raises(CorpusError, match="larger than"):
        load_corpus(root)


@pytest.mark.parametrize(
    ("opener", "invalid_closer", "closer"),
    [
        ("````markdown", "```", "````"),
        ("~~~~markdown", "~~~", "~~~~"),
        ("````markdown", "~~~~", "````"),
        ("~~~~markdown", "````", "~~~~"),
        ("```python", "``` trailing text", "```"),
        ("~~~python", "~~~ trailing text", "~~~"),
        ("```", "```\u00a0", "```"),
        ("~~~", "~~~\v", "~~~"),
        ("```", "    ```", "```"),
        ("~~~", "    ~~~", "~~~"),
    ],
)
def test_invalid_fence_closers_keep_code_under_the_real_section(
    opener: str, invalid_closer: str, closer: str
) -> None:
    text = (
        "# Guide\n\n"
        f"{opener}\n"
        "Example commands.\n"
        f"{invalid_closer}\n"
        "## Fake policy\n"
        "quartzoverride stays code.\n"
        f"{closer}\n\n"
        "## Actual policy\n"
        "Rollback to the previous release.\n"
    )
    _, chunks = chunk_text(text, "guide.md", markdown=True)
    assert [chunk.chunk_id for chunk in chunks] == ["guide.md#guide", "guide.md#actual-policy"]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 8), (10, 11)]
    assert "## Fake policy\nquartzoverride stays code." in chunks[0].text
    assert chunks[1].title == "Guide > Actual policy"


@pytest.mark.parametrize(
    ("opener", "closer"),
    [
        ("```python", "```"),
        ("````markdown", "`````"),
        ("~~~python", "~~~ \t"),
        ("   ~~~~markdown", " ~~~~~\t "),
        (" ```python", "   ```\t"),
    ],
)
def test_equal_or_longer_fence_closers_restore_real_headings(opener: str, closer: str) -> None:
    text = (
        f"# Guide\n\n{opener}\n## Example header\ncodewords.\n{closer}\n\n"
        "## Actual policy\nPolicy text.\n"
    )
    _, chunks = chunk_text(text, "guide.md", markdown=True)
    assert [chunk.chunk_id for chunk in chunks] == ["guide.md#guide", "guide.md#actual-policy"]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 6), (8, 9)]
    assert "## Example header\ncodewords." in chunks[0].text
    assert chunks[1].text == "Policy text."


@pytest.mark.parametrize(
    ("candidate", "expected_slugs"),
    [
        ("```py`thon", ["guide", "actual-policy"]),
        ("````lang``code", ["guide", "actual-policy"]),
        ("``` lang ```", ["guide", "actual-policy"]),
        ("~~~py`thon", ["guide"]),
        ("~~~lang~~~code", ["guide"]),
        ("```lang~~~code", ["guide"]),
        ("    ```python", ["guide", "actual-policy"]),
        ("\t~~~python", ["guide", "actual-policy"]),
        ("``python", ["guide", "actual-policy"]),
        ("~~python", ["guide", "actual-policy"]),
    ],
)
def test_opening_fence_rules_do_not_hide_genuine_sections(
    candidate: str, expected_slugs: list[str]
) -> None:
    text = f"# Guide\n\n{candidate}\n## Actual policy\nPolicy text.\n"
    _, chunks = chunk_text(text, "guide.md", markdown=True)
    assert [chunk.chunk_id for chunk in chunks] == [f"guide.md#{slug}" for slug in expected_slugs]
    assert "Policy text." in chunks[-1].text
