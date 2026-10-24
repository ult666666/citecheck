"""Integrity checks for the bundled demo data, so its claims stay checkable."""

import json
import tomllib
from pathlib import Path

import pytest

from citecheck.config import Settings
from citecheck.costs import load_price_table
from citecheck.evaluation import load_questions
from citecheck.fixtures import load_fixtures, replay_key
from citecheck.index import Index, build_index
from citecheck.retrieval import retrieve
from tests.helpers import FIXTURES, HANDBOOK, QUESTIONS, REPO_ROOT

# Citations in the fixtures that are wrong on purpose (see eval/README.md).
DELIBERATELY_INVALID_CITATIONS = {
    "security/device.md#lost-or-stolen-devices",  # q19: typo in the file name
    "engineering/code-review.md#billing-changes",  # q22: no such section
}


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> Index:
    path = tmp_path_factory.mktemp("index") / "index.sqlite"
    build_index(HANDBOOK, path)
    return Index(path)


def test_every_document_is_labeled_fictional() -> None:
    documents = sorted(HANDBOOK.rglob("*.md"))
    assert len(documents) == 25
    for path in documents:
        lines = path.read_text(encoding="utf-8").split("\n")
        assert lines[0] == "---", path
        front_matter = lines[1 : lines.index("---", 1)]
        assert "fictional: true" in front_matter, path


def test_the_fictional_label_is_not_indexed(index: Index) -> None:
    result = retrieve(index, "fictional notice demo corpus", k=5, min_score=0.0)
    assert result.candidates == ()


def test_fixtures_match_the_question_set() -> None:
    questions = {q.id: q for q in load_questions(QUESTIONS)}
    fixtures = load_fixtures(FIXTURES)
    for fixture in fixtures.values():
        assert fixture.question_id in questions, fixture.path
        assert replay_key(questions[fixture.question_id].question) == replay_key(fixture.question)
    covered = {fixture.question_id for fixture in fixtures.values()}
    # Only questions that abstain at retrieval (no model call) lack a fixture.
    assert set(questions) - covered == {"u03", "u07"}


def test_questions_without_fixtures_really_abstain_at_retrieval(index: Index) -> None:
    questions = {q.id: q for q in load_questions(QUESTIONS)}
    settings = Settings()
    for qid in ("u03", "u07"):
        result = retrieve(
            index, questions[qid].question, k=settings.k, min_score=settings.min_score
        )
        assert result.passages == (), qid


def test_every_fixture_citation_exists_unless_deliberately_invalid(index: Index) -> None:
    known = index.chunk_ids()
    cited = {
        citation
        for fixture in load_fixtures(FIXTURES).values()
        for sentence in fixture.answer.get("sentences", [])
        for citation in sentence["citations"]
    }
    assert cited - known == DELIBERATELY_INVALID_CITATIONS


def test_every_gold_chunk_exists(index: Index) -> None:
    known = index.chunk_ids()
    for question in load_questions(QUESTIONS):
        assert set(question.gold_chunks) <= known, question.id


def test_fixture_files_are_pretty_printed_json() -> None:
    for path in sorted(FIXTURES.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n", path


def test_the_example_price_table_parses_and_says_it_is_a_placeholder() -> None:
    path = REPO_ROOT / "examples" / "prices.example.toml"
    table = load_price_table(path)
    assert "claude-opus-5-5" in table.models
    assert "placeholders, NOT Anthropic's prices" in path.read_text(encoding="utf-8")
    assert tomllib.loads(path.read_text(encoding="utf-8"))["currency"] == "USD"


def test_no_absolute_home_paths_in_tracked_text() -> None:
    home_prefix = "/" + "Users" + "/"
    for folder in ("src", "tests", "examples", "eval", "docs"):
        for path in (REPO_ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".toml", ".html"}:
                assert home_prefix not in path.read_text(encoding="utf-8"), path
    for name in ("README.md", "pyproject.toml", "CHANGELOG.md"):
        path = Path(REPO_ROOT / name)
        if path.exists():
            assert home_prefix not in path.read_text(encoding="utf-8"), name
