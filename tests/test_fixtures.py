import json
from pathlib import Path
from typing import Any

import pytest

from citecheck.errors import FixtureError, FixtureNotFoundError
from citecheck.fixtures import (
    FixtureClient,
    estimate_tokens,
    load_fixture,
    load_fixtures,
    replay_key,
)
from citecheck.llm import ModelRequest

ANSWER = {
    "sentences": [{"text": "Five days carry over.", "citations": ["a.md#carryover"]}],
    "insufficient_evidence": False,
}
JUDGE = {"verdicts": [{"sentence": 1, "verdict": "supported", "reason": "Stated."}]}


def fixture_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "synthetic": True,
        "question_id": "q1",
        "question": "How many days carry over?",
        "note": "test",
        "answer": ANSWER,
        "expected_labels": ["supported"],
    }
    data.update(overrides)
    return data


def write(directory: Path, name: str, data: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


def request(purpose: str = "answer", question: str = "How many days carry over?") -> ModelRequest:
    return ModelRequest(
        purpose="judge" if purpose == "judge" else "answer",
        replay_key=question,
        model="claude-opus-5-5",
        max_tokens=1000,
        system="S" * 400,
        tool={"name": "submit_answer"},
        user_message="U" * 800,
    )


def test_load_a_valid_fixture(tmp_path: Path) -> None:
    fixture = load_fixture(write(tmp_path, "q1.json", fixture_data(judge=JUDGE)))
    assert fixture.question_id == "q1"
    assert fixture.expected_labels == ("supported",)
    assert fixture.judge == JUDGE
    assert fixture.path == "q1.json"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("{not json", "invalid JSON"),
        ([], "must be a JSON object"),
        ({"synthetic": True}, "missing keys"),
        (fixture_data(extra=1), "unknown keys: extra"),
        (fixture_data(synthetic=False), '"synthetic": true'),
        (fixture_data(question_id=""), "question_id"),
        (fixture_data(question=" "), "question must"),
        (fixture_data(answer="text"), "answer must"),
        (fixture_data(expected_labels=["maybe"]), "expected_labels must be a list"),
        (fixture_data(expected_labels=["supported", "supported"]), "one label per"),
        (fixture_data(judge=[1]), "judge must"),
        (fixture_data(note=3), "note must"),
    ],
)
def test_invalid_fixtures_are_rejected(tmp_path: Path, data: object, message: str) -> None:
    with pytest.raises(FixtureError, match=message):
        load_fixture(write(tmp_path, "bad.json", data))


def test_fixture_labels_are_optional(tmp_path: Path) -> None:
    data = fixture_data()
    del data["expected_labels"]
    assert load_fixture(write(tmp_path, "q.json", data)).expected_labels == ()


def test_load_fixtures_requires_an_existing_non_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="not found"):
        load_fixtures(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FixtureError, match="no fixture files"):
        load_fixtures(tmp_path / "empty")


def test_two_fixtures_for_one_question_are_rejected(tmp_path: Path) -> None:
    write(tmp_path, "a.json", fixture_data())
    write(tmp_path, "b.json", fixture_data(question="how many days carry over"))
    with pytest.raises(FixtureError, match=r"b\.json: answers the same question as a\.json"):
        load_fixtures(tmp_path)


def test_replay_key_ignores_case_spacing_and_trailing_punctuation() -> None:
    assert replay_key("  How many  DAYS carry over?! ") == "how many days carry over"


def test_estimate_tokens_is_at_least_one() -> None:
    assert estimate_tokens(0) == 1
    assert estimate_tokens(8) == 2
    assert estimate_tokens(9) == 3


def test_client_replays_the_answer_with_estimated_usage(tmp_path: Path) -> None:
    write(tmp_path, "q1.json", fixture_data())
    client = FixtureClient.from_directory(tmp_path)
    response = client.complete(request(question="how many days CARRY over"))
    assert client.source == "fixture"
    assert response.stop_reason == "tool_use"
    assert response.usage_estimated is True
    assert response.model == "claude-opus-5-5"
    call = response.tool_calls[0]
    assert (call.name, dict(call.input)) == ("submit_answer", ANSWER)
    assert response.usage[0].input_tokens > 300  # ~1,200 prompt characters / 4
    assert list(client.fixtures) == ["how many days carry over"]


def test_replayed_input_is_a_copy(tmp_path: Path) -> None:
    write(tmp_path, "q1.json", fixture_data())
    client = FixtureClient.from_directory(tmp_path)
    first = client.complete(request())
    first.tool_calls[0].input["sentences"].clear()
    second = client.complete(request())
    assert second.tool_calls[0].input["sentences"]


def test_client_replays_judge_responses(tmp_path: Path) -> None:
    write(tmp_path, "q1.json", fixture_data(judge=JUDGE))
    response = FixtureClient.from_directory(tmp_path).complete(request("judge"))
    assert response.tool_calls[0].name == "submit_verdicts"


def test_missing_fixture_and_missing_judge(tmp_path: Path) -> None:
    write(tmp_path, "q1.json", fixture_data())
    client = FixtureClient.from_directory(tmp_path)
    with pytest.raises(FixtureNotFoundError, match="--live"):
        client.complete(request(question="Something else?"))
    with pytest.raises(FixtureNotFoundError, match="no judge response"):
        client.complete(request("judge"))
    assert client.fixture_for("Something else?") is None
