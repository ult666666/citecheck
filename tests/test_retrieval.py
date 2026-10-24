import doctest
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from citecheck import retrieval
from citecheck.errors import QueryError
from citecheck.index import Index, build_index
from citecheck.retrieval import (
    MAX_QUERY_TERMS,
    MAX_QUESTION_CHARS,
    build_match_query,
    retrieve,
    validate_question,
)


@pytest.fixture(scope="module")
def small_index(tmp_path_factory: pytest.TempPathFactory) -> Index:
    root = tmp_path_factory.mktemp("corpus")
    (root / "a.md").write_text(
        "# Expenses\n\n## Receipts\n\nAttach a receipt for expenses of $25 or more.\n\n"
        "## Deadline\n\nSubmit expenses within 30 days.\n",
        encoding="utf-8",
    )
    (root / "b.md").write_text(
        "# Travel\n\n## Hotels\n\nThe hotel limit is $250 per night.\n", encoding="utf-8"
    )
    path = tmp_path_factory.mktemp("index") / "index.sqlite"
    build_index(root, path)
    return Index(path)


def test_docstring_examples_hold() -> None:
    result = doctest.testmod(retrieval)
    assert result.attempted > 0
    assert result.failed == 0


def test_validate_question_strips_whitespace() -> None:
    assert validate_question("  hello?  ") == "hello?"


@pytest.mark.parametrize("question", ["", "   \n\t"])
def test_empty_questions_are_rejected(question: str) -> None:
    with pytest.raises(QueryError, match="empty"):
        validate_question(question)


def test_overlong_questions_are_rejected() -> None:
    with pytest.raises(QueryError, match="limit"):
        validate_question("a" * (MAX_QUESTION_CHARS + 1))


def test_fts_keywords_and_syntax_characters_become_plain_terms() -> None:
    query = build_match_query('receipts AND hotels NEAR(x) "quoted" -minus col:val *')
    assert query == ('"receipts" OR "hotels" OR "near" OR "quoted" OR "minus" OR "col" OR "val"')


def test_query_is_capped_at_max_terms() -> None:
    question = " ".join(f"word{n}" for n in range(MAX_QUERY_TERMS + 10))
    query = build_match_query(question)
    assert query is not None
    assert query.count(" OR ") == MAX_QUERY_TERMS - 1


def test_retrieve_splits_candidates_at_the_threshold(small_index: Index) -> None:
    everything = retrieve(small_index, "receipt for expenses", k=5, min_score=0.0)
    assert [c.chunk_id for c in everything.passages] == ["a.md#receipts", "a.md#deadline"]
    best = everything.passages[0].score
    strict = retrieve(small_index, "receipt for expenses", k=5, min_score=best)
    assert [c.chunk_id for c in strict.passages] == ["a.md#receipts"]
    assert [c.chunk_id for c in strict.below_threshold] == ["a.md#deadline"]


def test_retrieve_without_searchable_words(small_index: Index) -> None:
    result = retrieve(small_index, "What is it?", k=5, min_score=0.0)
    assert result.query is None
    assert result.candidates == ()
    assert result.passages == ()


def test_retrieve_without_matches(small_index: Index) -> None:
    result = retrieve(small_index, "zeppelin flights", k=5, min_score=0.0)
    assert result.query is not None
    assert result.candidates == ()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(st.text(min_size=1, max_size=400).filter(lambda text: text.strip()))
def test_any_question_produces_a_query_sqlite_accepts(small_index: Index, question: str) -> None:
    result = retrieve(small_index, question, k=3, min_score=0.0)
    assert len(result.candidates) <= 3


def test_index_path_is_not_needed_for_query_building(tmp_path: Path) -> None:
    # Building a query is pure: no index or file access involved.
    assert build_match_query("hotel limit") == '"hotel" OR "limit"'
    assert not any(tmp_path.iterdir())
