from typing import Any

import pytest

from citecheck.answer import (
    DraftSentence,
    build_answer_request,
    parse_answer_input,
    request_answer,
)
from citecheck.config import Settings
from citecheck.errors import ModelResponseError
from citecheck.index import RetrievedChunk
from citecheck.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_TOOL
from tests.helpers import ScriptedClient, refusal_response, tool_response

PASSAGE = RetrievedChunk("a.md#x", "a.md", "A > X", 1, 3, "Five days carry over.", 9.0)


def test_request_carries_the_settings_and_the_passages() -> None:
    settings = Settings(model="claude-sonnet-5-5", max_tokens=2000, effort="high", fallback=False)
    req = build_answer_request("How many?", [PASSAGE], settings)
    assert (req.purpose, req.replay_key, req.model, req.max_tokens) == (
        "answer",
        "How many?",
        "claude-sonnet-5-5",
        2000,
    )
    assert req.system == ANSWER_SYSTEM_PROMPT
    assert dict(req.tool) == ANSWER_TOOL
    assert (req.effort, req.fallback, req.cache_system_prompt) == ("high", False, True)
    assert 'id="a.md#x"' in req.user_message
    assert req.user_message.endswith("<question>How many?</question>")


def test_valid_input_is_parsed_with_whitespace_and_blank_citations_removed() -> None:
    sentences, insufficient = parse_answer_input(
        {
            "sentences": [{"text": "  Five days.  ", "citations": [" a.md#x ", "", "  "]}],
            "insufficient_evidence": False,
        }
    )
    assert sentences == (DraftSentence(text="Five days.", citations=("a.md#x",)),)
    assert insufficient is False


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("text", "not a JSON object"),
        ({"sentences": []}, "exactly the keys"),
        ({"sentences": [], "insufficient_evidence": False, "x": 1}, "exactly the keys"),
        ({"sentences": [], "insufficient_evidence": "no"}, "true or false"),
        ({"sentences": "a", "insufficient_evidence": False}, "must be a list"),
        ({"sentences": ["a"], "insufficient_evidence": False}, "sentence 1 must be"),
        (
            {
                "sentences": [{"text": "a", "citations": [], "x": 1}],
                "insufficient_evidence": False,
            },
            "sentence 1 must be",
        ),
        (
            {"sentences": [{"text": " ", "citations": []}], "insufficient_evidence": False},
            "no text",
        ),
        (
            {"sentences": [{"text": "a", "citations": "x"}], "insufficient_evidence": False},
            "list of",
        ),
        (
            {"sentences": [{"text": "a", "citations": [1]}], "insufficient_evidence": False},
            "list of",
        ),
    ],
)
def test_malformed_input_is_rejected(raw: Any, message: str) -> None:
    with pytest.raises(ModelResponseError, match=message):
        parse_answer_input(raw)


def test_request_answer_returns_a_draft() -> None:
    tool_input = {
        "sentences": [{"text": "Five days carry over.", "citations": ["a.md#x"]}],
        "insufficient_evidence": False,
    }
    client = ScriptedClient([tool_response("submit_answer", tool_input)])
    draft = request_answer(client, "How many?", [PASSAGE], Settings())
    assert draft.sentences == (DraftSentence("Five days carry over.", ("a.md#x",)),)
    assert not draft.refused
    assert len(draft.responses) == 1


def test_request_answer_reports_a_refusal() -> None:
    client = ScriptedClient([refusal_response("cyber")])
    draft = request_answer(client, "How many?", [PASSAGE], Settings())
    assert draft.refused
    assert draft.refusal_category == "cyber"
    assert draft.sentences == ()
