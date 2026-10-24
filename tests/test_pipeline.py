import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from citecheck.config import Settings
from citecheck.costs import parse_price_table
from citecheck.errors import QueryError
from citecheck.index import Index, build_index
from citecheck.llm import ModelResponse, TokenUsage, ToolCall
from citecheck.pipeline import ask
from tests.helpers import MODEL, ScriptedClient, refusal_response, tool_response

CARRYOVER = "time-off.md#carryover"


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> Index:
    root = tmp_path_factory.mktemp("corpus")
    (root / "time-off.md").write_text(
        "# Paid Time Off\n\n## Carryover\n\n"
        "Up to 5 unused PTO days carry over into the next calendar year.\n\n"
        "## Sick leave\n\nEmployees receive 8 paid sick days per calendar year.\n",
        encoding="utf-8",
    )
    (root / "travel.md").write_text(
        "# Travel\n\n## Hotels\n\nThe nightly hotel limit is $250.\n", encoding="utf-8"
    )
    path = tmp_path_factory.mktemp("index") / "index.sqlite"
    build_index(root, path)
    return Index(path)


def answer_input(*sentences: tuple[str, list[str]], insufficient: bool = False) -> dict[str, Any]:
    return {
        "sentences": [{"text": text, "citations": cites} for text, cites in sentences],
        "insufficient_evidence": insufficient,
    }


SETTINGS = Settings(min_score=0.5)


def test_no_searchable_words_abstains_without_a_model_call(index: Index) -> None:
    result = ask("What is it?", index=index, client=ScriptedClient([]), settings=SETTINGS)
    assert result.status == "abstained"
    assert result.abstain_reason == "the question has no searchable words"
    assert result.model_calls == 0
    assert result.cost is None


def test_no_matching_passage_abstains(index: Index) -> None:
    result = ask("zeppelin repairs", index=index, client=ScriptedClient([]), settings=SETTINGS)
    assert result.abstain_reason == "no passage matched the question's words"


def test_scores_below_the_threshold_abstain(index: Index) -> None:
    settings = Settings(min_score=1000.0)
    result = ask("PTO carry over", index=index, client=ScriptedClient([]), settings=settings)
    assert result.status == "abstained"
    assert result.abstain_reason is not None
    assert "no passage scored at least 1000; the best score was" in result.abstain_reason
    assert result.passages == ()
    assert result.below_threshold


def test_a_supported_answer(index: Index) -> None:
    tool_input = answer_input(("Up to 5 unused PTO days carry over.", [CARRYOVER]))
    client = ScriptedClient([tool_response("submit_answer", tool_input)])
    result = ask("How many PTO days carry over?", index=index, client=client, settings=SETTINGS)
    assert result.status == "answered"
    assert result.label_counts == {"supported": 1, "weakly_supported": 0, "unsupported": 0}
    assert result.passages[0].chunk_id == CARRYOVER
    assert client.requests[0].replay_key == "How many PTO days carry over?"
    data = result.to_dict()
    json.dumps(data)  # serializable
    assert data["served_by"] == [MODEL]
    assert data["usage"]["input_tokens"] == 100
    assert data["judge"] is None
    assert result.timings.total_ms >= result.timings.retrieval_ms >= 0


def test_fixture_answers_do_not_claim_a_serving_model(index: Index) -> None:
    tool_input = answer_input(("Up to 5 unused PTO days carry over.", [CARRYOVER]))
    client = ScriptedClient([tool_response("submit_answer", tool_input)], source="fixture")
    result = ask("PTO carry over", index=index, client=client, settings=SETTINGS)
    assert result.to_dict()["served_by"] == []


def test_partial_answers_keep_their_sentences(index: Index) -> None:
    tool_input = answer_input(
        ("Up to 5 unused PTO days carry over.", [CARRYOVER]), insufficient=True
    )
    client = ScriptedClient([tool_response("submit_answer", tool_input)])
    result = ask("PTO carry over and payout?", index=index, client=client, settings=SETTINGS)
    assert result.status == "partial"
    assert len(result.sentences) == 1


@pytest.mark.parametrize(
    ("insufficient", "reason"),
    [
        (True, "the model reported that the passages do not contain the answer"),
        (False, "the model returned no sentences"),
    ],
)
def test_empty_answers_abstain(index: Index, insufficient: bool, reason: str) -> None:
    client = ScriptedClient(
        [tool_response("submit_answer", answer_input(insufficient=insufficient))]
    )
    result = ask("PTO carry over", index=index, client=client, settings=SETTINGS)
    assert result.status == "abstained"
    assert result.abstain_reason == reason
    assert result.model_calls == 1


def test_a_refusal_becomes_an_abstention(index: Index) -> None:
    result = ask(
        "PTO carry over",
        index=index,
        client=ScriptedClient([refusal_response("cyber")]),
        settings=SETTINGS,
    )
    assert result.status == "abstained"
    assert result.abstain_reason == "the model declined to answer (category: cyber)"


def test_a_refusal_without_a_category(index: Index) -> None:
    result = ask(
        "PTO carry over",
        index=index,
        client=ScriptedClient([refusal_response(None)]),
        settings=SETTINGS,
    )
    assert result.abstain_reason == "the model declined to answer"


def test_judge_pass_adds_verdicts_and_a_second_call(index: Index) -> None:
    tool_input = answer_input(
        ("Up to 5 unused PTO days carry over.", [CARRYOVER]), ("Uncited remark.", [])
    )
    verdicts = {"verdicts": [{"sentence": 1, "verdict": "supported", "reason": "Stated."}]}
    client = ScriptedClient(
        [tool_response("submit_answer", tool_input), tool_response("submit_verdicts", verdicts)]
    )
    settings = Settings(min_score=0.5, judge=True)
    result = ask("PTO carry over", index=index, client=client, settings=settings)
    assert result.judge is not None
    assert list(result.judge.verdicts) == [1]
    assert result.model_calls == 2
    assert result.to_dict()["judge"]["verdicts"]["1"]["verdict"] == "supported"
    assert client.requests[1].purpose == "judge"


def test_cost_uses_the_price_table(index: Index) -> None:
    tool_input = answer_input(("Up to 5 unused PTO days carry over.", [CARRYOVER]))
    client = ScriptedClient([tool_response("submit_answer", tool_input)])
    prices = parse_price_table(
        {
            "models": {
                MODEL: {
                    "input_per_mtok": 2,
                    "output_per_mtok": 10,
                    "cache_write_per_mtok": 2.5,
                    "cache_read_per_mtok": 0.2,
                }
            }
        },
        source="p",
    )
    result = ask("PTO carry over", index=index, client=client, settings=SETTINGS, prices=prices)
    assert result.cost is not None
    assert result.cost.total == Decimal("0.0004")  # (100 * 2 + 20 * 10) / 1e6


def test_fallback_use_is_reported(index: Index) -> None:
    tool_input = answer_input(("Up to 5 unused PTO days carry over.", [CARRYOVER]))
    served = ModelResponse(
        model="claude-opus-4-8",
        stop_reason="tool_use",
        tool_calls=(ToolCall("submit_answer", tool_input),),
        text="",
        usage=(TokenUsage(MODEL, 10, 0), TokenUsage("claude-opus-4-8", 100, 20)),
        fallback_used=True,
    )
    result = ask("PTO carry over", index=index, client=ScriptedClient([served]), settings=SETTINGS)
    assert result.fallback_used
    assert result.to_dict()["served_by"] == ["claude-opus-4-8"]


def test_empty_questions_are_rejected(index: Index, tmp_path: Path) -> None:
    with pytest.raises(QueryError):
        ask("   ", index=index, client=ScriptedClient([]))
