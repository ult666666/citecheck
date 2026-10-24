"""Replay hand-written synthetic model responses: no network, no API key.

Fixture files are JSON documents written by hand for citecheck's offline demo,
its tests and its evaluation. **They are not real model outputs.** Some are
deliberately wrong (an invented number, a citation to a passage that was not
retrieved, a negation flip) so that the verifier has something to catch. Every
file must declare ``"synthetic": true``; the loader refuses files that do not.

File format::

    {
      "synthetic": true,
      "question_id": "q01",
      "question": "How many unused PTO days carry over?",
      "note": "What this fixture demonstrates.",
      "answer": {"sentences": [...], "insufficient_evidence": false},
      "expected_labels": ["supported"],
      "judge": {"verdicts": [...]}
    }

``answer`` is exactly what the model would pass to the ``submit_answer`` tool.
``expected_labels`` is a human judgment ("supported" or "unsupported") for each
answer sentence, used by the evaluation to measure how often the automatic
checker agrees with a person. ``judge`` (optional) is a ``submit_verdicts`` input
for the ``--judge`` pass.

Token counts in fixture mode are estimated at about four characters per token
and flagged as estimates everywhere they are shown; they are not measurements.
"""

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from citecheck.errors import FixtureError, FixtureNotFoundError
from citecheck.llm import ClientSource, ModelRequest, ModelResponse, TokenUsage, ToolCall
from citecheck.prompts import ANSWER_TOOL_NAME, JUDGE_TOOL_NAME

#: Shown wherever fixture output is displayed, so it is never mistaken for a
#: real model's answer.
FIXTURE_NOTICE: Final = (
    "fixture replay: a hand-written synthetic response, not a real model output"
)
CHARS_PER_TOKEN_ESTIMATE: Final = 4
HUMAN_LABELS: Final = frozenset({"supported", "unsupported"})
_REQUIRED_KEYS: Final = frozenset({"synthetic", "question_id", "question", "answer"})
_OPTIONAL_KEYS: Final = frozenset({"note", "expected_labels", "judge"})


@dataclass(frozen=True)
class Fixture:
    """One prepared response, loaded from a JSON file."""

    path: str
    question_id: str
    question: str
    answer: Mapping[str, Any]
    expected_labels: tuple[str, ...]
    judge: Mapping[str, Any] | None
    note: str


def replay_key(question: str) -> str:
    """Normalize a question for fixture lookup.

    Matching ignores case, repeated whitespace and trailing punctuation, so
    "how many PTO days carry over" finds the fixture for "How many PTO days
    carry over?".
    """
    return " ".join(question.casefold().split()).rstrip("?.! ")


def estimate_tokens(chars: int) -> int:
    """Rough token estimate (about four characters per token). Not a tokenizer."""
    return max(1, math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE))


def _fail(path: Path, message: str) -> FixtureError:
    return FixtureError(f"{path.name}: {message}")


def load_fixture(path: Path) -> Fixture:
    """Load and validate one fixture file.

    Only the file's structure is validated here. The ``answer`` itself is
    checked later by the same parser that checks live model output, so a
    fixture can deliberately contain a malformed answer for testing.

    Raises:
        FixtureError: If the file is not valid JSON or not a valid fixture.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(path, f"invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise _fail(path, "a fixture must be a JSON object")
    keys = set(data)
    if missing := sorted(_REQUIRED_KEYS - keys):
        raise _fail(path, f"missing keys: {', '.join(missing)}")
    if unknown := sorted(keys - _REQUIRED_KEYS - _OPTIONAL_KEYS):
        raise _fail(path, f"unknown keys: {', '.join(unknown)}")
    if data["synthetic"] is not True:
        raise _fail(path, 'fixtures must declare "synthetic": true')
    question_id, question, answer = data["question_id"], data["question"], data["answer"]
    if not isinstance(question_id, str) or not question_id.strip():
        raise _fail(path, "question_id must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise _fail(path, "question must be a non-empty string")
    if not isinstance(answer, dict):
        raise _fail(path, "answer must be a JSON object (the submit_answer tool input)")
    labels = data.get("expected_labels", [])
    if not isinstance(labels, list) or not all(label in HUMAN_LABELS for label in labels):
        raise _fail(path, "expected_labels must be a list of 'supported' or 'unsupported'")
    sentences = answer.get("sentences")
    if labels and (not isinstance(sentences, list) or len(labels) != len(sentences)):
        raise _fail(path, "expected_labels must have one label per answer sentence")
    judge = data.get("judge")
    if judge is not None and not isinstance(judge, dict):
        raise _fail(path, "judge must be a JSON object (the submit_verdicts tool input)")
    note = data.get("note", "")
    if not isinstance(note, str):
        raise _fail(path, "note must be a string")
    return Fixture(
        path=path.name,
        question_id=question_id,
        question=question,
        answer=answer,
        expected_labels=tuple(labels),
        judge=judge,
        note=note,
    )


def load_fixtures(directory: Path) -> dict[str, Fixture]:
    """Load every ``*.json`` fixture in ``directory``, keyed by :func:`replay_key`.

    Raises:
        FixtureError: If the folder is missing or empty, a file is invalid, or
            two files answer the same question.
    """
    if not directory.is_dir():
        raise FixtureError(
            f"fixture folder not found: {directory} (run from the repository root, "
            "pass --fixtures DIR, or use --live)"
        )
    fixtures: dict[str, Fixture] = {}
    for path in sorted(directory.glob("*.json")):
        fixture = load_fixture(path)
        key = replay_key(fixture.question)
        if key in fixtures:
            raise _fail(path, f"answers the same question as {fixtures[key].path}")
        fixtures[key] = fixture
    if not fixtures:
        raise FixtureError(f"no fixture files (*.json) found in {directory}")
    return fixtures


class FixtureClient:
    """A :class:`~citecheck.llm.ModelClient` that replays fixture files.

    It returns the prepared tool input for the request's question, reports the
    requested model name, and estimates token usage from character counts so
    that cost accounting can be demonstrated offline (flagged as estimated).
    """

    def __init__(self, fixtures: Mapping[str, Fixture]) -> None:
        """Use already-loaded fixtures, keyed by :func:`replay_key`."""
        self._fixtures = dict(fixtures)

    @classmethod
    def from_directory(cls, directory: Path) -> "FixtureClient":
        """Load every fixture file in ``directory``."""
        return cls(load_fixtures(directory))

    @property
    def source(self) -> ClientSource:
        """Always "fixture"."""
        return "fixture"

    @property
    def fixtures(self) -> Mapping[str, Fixture]:
        """The loaded fixtures, keyed by :func:`replay_key`."""
        return self._fixtures

    def fixture_for(self, question: str) -> Fixture | None:
        """Return the fixture prepared for ``question``, if any."""
        return self._fixtures.get(replay_key(question))

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the prepared response for the request's question.

        Raises:
            FixtureNotFoundError: If no fixture matches the question, or the
                request is a judge request and the fixture has no judge part.
        """
        fixture = self.fixture_for(request.replay_key)
        if fixture is None:
            raise FixtureNotFoundError(
                f"no fixture for the question {request.replay_key!r}. Fixture mode only "
                "replays the prepared questions (see eval/questions.json); use --live "
                "to ask a real model."
            )
        if request.purpose == "answer":
            name, tool_input = ANSWER_TOOL_NAME, fixture.answer
        elif fixture.judge is not None:
            name, tool_input = JUDGE_TOOL_NAME, fixture.judge
        else:
            raise FixtureNotFoundError(f"fixture {fixture.path} has no judge response")
        prompt_chars = (
            len(request.system) + len(json.dumps(request.tool)) + len(request.user_message)
        )
        usage = TokenUsage(
            model=request.model,
            input_tokens=estimate_tokens(prompt_chars),
            output_tokens=estimate_tokens(len(json.dumps(tool_input))),
        )
        return ModelResponse(
            model=request.model,
            stop_reason="tool_use",
            tool_calls=(ToolCall(name=name, input=copy.deepcopy(dict(tool_input))),),
            text="",
            usage=(usage,),
            usage_estimated=True,
        )
