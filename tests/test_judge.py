from typing import Any

import pytest

from citecheck.answer import DraftSentence
from citecheck.config import Settings
from citecheck.errors import ModelResponseError
from citecheck.index import RetrievedChunk
from citecheck.judge import JudgeVerdict, build_judge_request, parse_judge_input, run_judge
from citecheck.verify import verify_sentences
from tests.helpers import ScriptedClient, refusal_response, tool_response

A = RetrievedChunk("a.md#x", "a.md", "A", 1, 2, "Five unused days carry over.", 9.0)
B = RetrievedChunk("b.md#y", "b.md", "B", 1, 2, "Sick days do not carry over.", 7.0)
PASSAGES = {A.chunk_id: A, B.chunk_id: B}
CHECKS = verify_sentences(
    [
        DraftSentence("Five unused days carry over.", ("a.md#x", "b.md#y")),
        DraftSentence("An uncited remark.", ()),
        DraftSentence("Sick days do not carry over.", ("b.md#y",)),
    ],
    PASSAGES,
    Settings(),
)
GOOD = {
    "verdicts": [
        {"sentence": 1, "verdict": "supported", "reason": " Stated in a. "},
        {"sentence": 3, "verdict": "not_supported", "reason": "Contradicted."},
    ]
}


def test_only_validly_cited_sentences_are_sent_to_the_judge() -> None:
    request = build_judge_request("Q?", CHECKS, PASSAGES, Settings(judge=True))
    assert request is not None
    assert request.purpose == "judge"
    assert '<sentence number="1"' in request.user_message
    assert '<sentence number="2"' not in request.user_message
    assert request.user_message.count('<passage id="b.md#y"') == 1


def test_nothing_to_judge_without_valid_citations() -> None:
    checks = verify_sentences([DraftSentence("Uncited.", ())], PASSAGES, Settings())
    assert build_judge_request("Q?", checks, PASSAGES, Settings()) is None


def test_valid_verdicts_are_parsed() -> None:
    assert parse_judge_input(GOOD, [1, 3]) == {
        1: JudgeVerdict("supported", "Stated in a."),
        3: JudgeVerdict("not_supported", "Contradicted."),
    }


def _verdict(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"sentence": 1, "verdict": "supported", "reason": "ok"}
    item.update(overrides)
    return item


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "only 'verdicts'"),
        ({"verdicts": [], "extra": 1}, "only 'verdicts'"),
        ({"verdicts": "x"}, "must be a list"),
        ({"verdicts": [{"sentence": 1}]}, "exactly 'sentence'"),
        ({"verdicts": [_verdict(sentence=True)]}, "unknown sentence"),
        ({"verdicts": [_verdict(sentence=7)]}, "unknown sentence"),
        ({"verdicts": [_verdict(), _verdict()]}, "more than one verdict"),
        ({"verdicts": [_verdict(verdict="maybe")]}, "unknown verdict"),
        ({"verdicts": [_verdict(reason=3)]}, "must be a string"),
        ({"verdicts": []}, r"no verdict for sentence\(s\) \[1\]"),
    ],
)
def test_malformed_verdicts_are_rejected(raw: Any, message: str) -> None:
    with pytest.raises(ModelResponseError, match=message):
        parse_judge_input(raw, [1])


def test_run_judge_returns_verdicts_and_responses() -> None:
    client = ScriptedClient([tool_response("submit_verdicts", GOOD)])
    result = run_judge(client, "Q?", CHECKS, PASSAGES, Settings(judge=True))
    assert sorted(result.verdicts) == [1, 3]
    assert len(result.responses) == 1
    assert result.skipped_reason is None


def test_run_judge_skips_without_a_model_call_when_nothing_is_cited() -> None:
    checks = verify_sentences([DraftSentence("Uncited.", ())], PASSAGES, Settings())
    result = run_judge(ScriptedClient([]), "Q?", checks, PASSAGES, Settings(judge=True))
    assert result.verdicts == {}
    assert result.skipped_reason == "no sentence has a valid citation to judge"


def test_run_judge_reports_a_refusal() -> None:
    client = ScriptedClient([refusal_response()])
    result = run_judge(client, "Q?", CHECKS, PASSAGES, Settings(judge=True))
    assert result.verdicts == {}
    assert result.skipped_reason == "the judge model declined to respond"
    assert len(result.responses) == 1
