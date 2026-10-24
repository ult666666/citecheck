import json
from pathlib import Path
from typing import Any

import pytest

from citecheck.config import Settings
from citecheck.errors import EvalDataError
from citecheck.evaluation import (
    EvalQuestion,
    evaluate,
    format_details,
    format_report,
    load_questions,
    run_eval,
)
from citecheck.fixtures import FixtureClient
from citecheck.index import Index, build_index
from tests.helpers import FIXTURES, HANDBOOK, QUESTIONS, ScriptedClient, tool_response


def write_questions(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def question(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": "q1",
        "question": "How many days carry over?",
        "answerable": True,
        "gold_chunks": ["a.md#x"],
    }
    item.update(overrides)
    return item


def test_load_valid_questions(tmp_path: Path) -> None:
    path = write_questions(
        tmp_path,
        {
            "questions": [
                question(notes="why"),
                question(id="u1", question="Who?", answerable=False, gold_chunks=[]),
            ]
        },
    )
    loaded = load_questions(path)
    assert loaded == [
        EvalQuestion("q1", "How many days carry over?", True, ("a.md#x",), "why"),
        EvalQuestion("u1", "Who?", False, (), ""),
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "expected an object"),
        ({"questions": []}, "is empty"),
        ({"questions": ["x"]}, "must be a JSON object"),
        ({"questions": [{"id": "q1"}]}, "is missing"),
        ({"questions": [question(extra=1)]}, "unknown keys: extra"),
        ({"questions": [question(id="")]}, "id must be"),
        ({"questions": [question(question="")]}, "question must be"),
        ({"questions": [question(answerable="yes")]}, "true or false"),
        ({"questions": [question(gold_chunks="a.md#x")]}, "list of chunk IDs"),
        ({"questions": [question(gold_chunks=[""])]}, "list of chunk IDs"),
        ({"questions": [question(gold_chunks=[])]}, "need gold_chunks"),
        ({"questions": [question(answerable=False)]}, "must have none"),
        ({"questions": [question(notes=1)]}, "notes must be"),
        ({"questions": [question(), question(question="Other?")]}, "duplicate question id"),
        (
            {"questions": [question(), question(id="q2", question="how many days CARRY over")]},
            "duplicate",
        ),
    ],
)
def test_invalid_question_files_are_rejected(
    tmp_path: Path, payload: object, message: str
) -> None:
    with pytest.raises(EvalDataError, match=message):
        load_questions(write_questions(tmp_path, payload))


def test_missing_and_malformed_question_files(tmp_path: Path) -> None:
    with pytest.raises(EvalDataError, match="not found"):
        load_questions(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(EvalDataError, match="invalid JSON"):
        load_questions(bad)


@pytest.fixture(scope="module")
def tiny_index(tmp_path_factory: pytest.TempPathFactory) -> Index:
    root = tmp_path_factory.mktemp("corpus")
    (root / "a.md").write_text("# A\n\n## X\n\nFive unused days carry over.\n", encoding="utf-8")
    path = tmp_path_factory.mktemp("index") / "index.sqlite"
    build_index(root, path)
    return Index(path)


def test_unknown_gold_chunks_fail_loudly(tiny_index: Index) -> None:
    items = [EvalQuestion("q1", "days carry over", True, ("a.md#typo",))]
    with pytest.raises(EvalDataError, match=r"unknown gold chunks: a\.md#typo"):
        run_eval(items, tiny_index, ScriptedClient([]), Settings())


def test_live_mode_report_has_no_human_label_section(tiny_index: Index) -> None:
    answer = {
        "sentences": [{"text": "Five unused days carry over.", "citations": ["a.md#x"]}],
        "insufficient_evidence": False,
    }
    items = [
        EvalQuestion("q1", "How many unused days carry over?", True, ("a.md#x",)),
        EvalQuestion("u1", "What is it?", False, ()),
    ]
    client = ScriptedClient([tool_response("submit_answer", answer)], source="live")
    report = run_eval(items, tiny_index, client, Settings(min_score=0.0))
    summary = report.summary
    assert (summary.recall_at_k, summary.citation_precision, summary.abstention_accuracy) == (
        1.0,
        1.0,
        1.0,
    )
    text = format_report(report)
    assert "live calls to claude-opus-5-5" in text
    assert "human labels" not in text
    assert "from the API" in text


def test_ratios_are_not_available_without_answerable_questions(tiny_index: Index) -> None:
    report = run_eval(
        [EvalQuestion("u1", "What is it?", False, ())], tiny_index, ScriptedClient([]), Settings()
    )
    assert report.summary.recall_at_k is None
    assert report.summary.citation_precision is None
    assert "n/a" in format_report(report)


@pytest.fixture(scope="module")
def bundled_report_settings() -> Settings:
    return Settings()


def test_bundled_evaluation_numbers(bundled_report_settings: Settings) -> None:
    # These are the numbers quoted in README.md. They come from hand-written
    # synthetic fixtures, so they describe the harness, not a model.
    report = evaluate(
        HANDBOOK, QUESTIONS, FixtureClient.from_directory(FIXTURES), bundled_report_settings
    )
    s = report.summary
    assert (s.questions, s.answerable, s.unanswerable) == (36, 28, 8)
    assert (s.recall_at_k, s.hit_at_k) == (1.0, 1.0)
    assert (s.citation_true_positives, s.citation_false_positives) == (29, 4)
    assert s.citation_false_negatives == 2
    assert s.rejected_citations == 2
    assert s.sentences == 49
    assert s.label_counts == {"supported": 42, "weakly_supported": 1, "unsupported": 6}
    assert s.abstention_correct == 34
    assert (s.unanswerable_abstained, s.unanswerable_abstained_without_model) == (7, 2)
    assert (s.unanswerable_answered, s.answerable_abstained) == (1, 1)
    assert (s.human_unsupported, s.checker_caught) == (7, 5)
    assert (s.human_supported, s.checker_false_flags) == (42, 2)
    assert s.usage.calls == 34
    assert s.usage.estimated is True


def test_report_text_json_and_details_are_deterministic(bundled_report_settings: Settings) -> None:
    def run() -> tuple[str, str, str]:
        report = evaluate(
            HANDBOOK, QUESTIONS, FixtureClient.from_directory(FIXTURES), bundled_report_settings
        )
        return format_report(report), json.dumps(report.to_dict()), format_details(report)

    first, second = run(), run()
    assert first == second
    text, data, details = first
    assert "hand-written synthetic response, not a real model output" in text
    assert "they do not measure any model" in text
    assert len(details.splitlines()) == 2 + 36
    assert "WRONG" in details
    assert json.loads(data)["summary"]["questions"] == 36
