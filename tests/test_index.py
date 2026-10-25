import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

from citecheck.errors import IndexFormatError, IndexNotFoundError, QueryError
from citecheck.index import SCHEMA_VERSION, Index, build_index

CorpusFactory = Callable[[dict[str, str]], Path]

FILES = {
    "travel.md": (
        "# Travel\n\n## Receipts\n\nKeep every hotel invoice for your claim.\n\n"
        "## Flights\n\nBook economy for short flights and keep the receipts for later.\n"
    ),
    "pay.md": "# Pay\n\n## Reimbursement\n\nExpenses are reimbursed in the next payroll.\n",
    "notes.txt": "Plain text about office plants and watering schedules.\n",
}


@pytest.fixture
def built(make_corpus: CorpusFactory, tmp_path: Path) -> Index:
    path = tmp_path / "idx" / "index.sqlite"
    stats = build_index(make_corpus(FILES), path)
    assert (stats.documents, stats.chunks) == (3, 4)
    return Index(path)


def test_build_writes_a_complete_index_and_no_temporary_file(built: Index) -> None:
    assert built.stats.documents == 3
    assert built.stats.chunks == 4
    assert not built.path.with_name(built.path.name + ".tmp").exists()
    assert built.chunk_ids() == {
        "notes.txt#top",
        "pay.md#reimbursement",
        "travel.md#flights",
        "travel.md#receipts",
    }


def test_search_ranks_best_first_with_positive_scores(built: Index) -> None:
    results = built.search('"receipts"', limit=5)
    assert [r.chunk_id for r in results] == ["travel.md#receipts", "travel.md#flights"]
    assert results[0].score > results[1].score > 0
    assert results[0].title == "Travel > Receipts"
    assert (results[0].start_line, results[0].end_line) == (3, 5)


def test_title_matches_outweigh_body_matches(built: Index) -> None:
    # "receipts" is the heading of one chunk and a body word of the other.
    top = built.search('"receipts"', limit=1)[0]
    assert top.chunk_id == "travel.md#receipts"


def test_porter_stemming_matches_other_word_forms(built: Index) -> None:
    results = built.search('"reimbursements"', limit=5)
    assert [r.chunk_id for r in results] == ["pay.md#reimbursement"]


def test_limit_and_no_match(built: Index) -> None:
    assert len(built.search('"receipts" OR "plants"', limit=1)) == 1
    assert built.search('"zeppelin"', limit=5) == []


def test_malformed_match_query_is_a_query_error(built: Index) -> None:
    with pytest.raises(QueryError):
        built.search('"unbalanced', limit=5)


def test_to_dict_rounds_the_score(built: Index) -> None:
    data = built.search('"receipts"', limit=1)[0].to_dict()
    assert set(data) == {
        "chunk_id",
        "doc_path",
        "title",
        "start_line",
        "end_line",
        "text",
        "score",
    }
    assert data["score"] == round(data["score"], 4)


def test_rebuild_replaces_the_previous_index(make_corpus: CorpusFactory, tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    build_index(make_corpus(FILES), path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "only.md").write_text("# Only\n\nzeppelin\n", encoding="utf-8")
    build_index(other, path)
    assert Index(path).chunk_ids() == {"only.md#only"}


def test_failed_build_keeps_the_old_index_and_removes_the_temporary_file(
    make_corpus: CorpusFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.sqlite"
    build_index(make_corpus(FILES), path)

    def fail(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        build_index(make_corpus(FILES), path)
    assert not path.with_name(path.name + ".tmp").exists()
    assert Index(path).stats.chunks == 4


def test_missing_index(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError, match="citecheck index"):
        Index(tmp_path / "missing.sqlite")


def test_a_file_that_is_not_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "junk.sqlite"
    path.write_bytes(b"this is not a database at all" * 10)
    with pytest.raises(IndexFormatError, match="not a citecheck index"):
        Index(path)


def test_a_sqlite_file_without_citecheck_tables(tmp_path: Path) -> None:
    path = tmp_path / "other.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE something (x INTEGER)")
        conn.commit()
    with pytest.raises(IndexFormatError):
        Index(path)


def test_an_index_from_another_format_version(built: Index) -> None:
    with closing(sqlite3.connect(built.path)) as conn:
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
        conn.commit()
    with pytest.raises(IndexFormatError, match=f"reads format '{SCHEMA_VERSION}'"):
        Index(built.path)


@pytest.mark.parametrize(
    ("opener", "invalid_closer", "closer"),
    [
        ("````markdown", "```", "````"),
        ("~~~~markdown", "~~~", "~~~~"),
        ("```markdown", "``` not a closer", "```"),
        ("~~~markdown", "~~~ not a closer", "~~~"),
    ],
)
def test_indexed_code_is_retrieved_with_its_real_section_citation(
    make_corpus: CorpusFactory,
    tmp_path: Path,
    opener: str,
    invalid_closer: str,
    closer: str,
) -> None:
    from citecheck.retrieval import retrieve

    source = (
        "# Deployment Guide\n\n"
        f"{opener}\n"
        "Example commands.\n"
        f"{invalid_closer}\n"
        "## Override Checklist\n"
        "quartzoverride command example.\n"
        f"{closer}\n\n"
        "## Rollback Policy\n"
        "Rollback to the previous release.\n"
    )
    corpus = make_corpus({"deploy.md": source})
    path = tmp_path / "index.sqlite"
    stats = build_index(corpus, path)
    index = Index(path)
    assert (stats.documents, stats.chunks) == (1, 2)
    assert index.chunk_ids() == {"deploy.md#deployment-guide", "deploy.md#rollback-policy"}
    offered = retrieve(index, "quartzoverride", k=5, min_score=0.0).passages
    assert [passage.chunk_id for passage in offered] == ["deploy.md#deployment-guide"]
    assert offered[0].title == "Deployment Guide"
    assert (offered[0].start_line, offered[0].end_line) == (1, 8)
    assert "## Override Checklist\nquartzoverride command example." in offered[0].text
    policy = retrieve(index, "rollback", k=5, min_score=0.0).passages
    assert [passage.chunk_id for passage in policy] == ["deploy.md#rollback-policy"]
    assert (policy[0].start_line, policy[0].end_line) == (10, 11)


def test_invalid_backtick_info_does_not_swallow_indexed_policy_heading(
    make_corpus: CorpusFactory, tmp_path: Path
) -> None:
    from citecheck.retrieval import retrieve

    corpus = make_corpus(
        {"policy.md": "# Guide\n\n```python`example\n## Retention\nKeep audit records.\n"}
    )
    path = tmp_path / "index.sqlite"
    build_index(corpus, path)
    offered = retrieve(Index(path), "records", k=5, min_score=0.0).passages
    assert [passage.chunk_id for passage in offered] == ["policy.md#retention"]
    assert offered[0].title == "Guide > Retention"
    assert (offered[0].start_line, offered[0].end_line) == (4, 5)
