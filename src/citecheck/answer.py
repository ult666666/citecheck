"""Ask the model for a cited answer and validate the tool input it returns.

The model is offered only the retrieved passages and must answer through the
``submit_answer`` tool: a list of sentences, each with the IDs of the passages
it cites, plus an explicit ``insufficient_evidence`` flag. The schema is strict
on the API side, but this module still validates the input itself: replayed
fixtures, older models and future schema changes should all fail loudly here
rather than produce a half-parsed answer downstream.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from citecheck.config import Settings
from citecheck.errors import ModelResponseError
from citecheck.index import RetrievedChunk
from citecheck.llm import ModelClient, ModelRequest, ModelResponse, call_tool
from citecheck.prompts import (
    ANSWER_RETRY_REMINDER,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_TOOL,
    ANSWER_TOOL_NAME,
    answer_user_message,
    tool_definition,
)

_ANSWER_KEYS: Final = frozenset({"sentences", "insufficient_evidence"})
_SENTENCE_KEYS: Final = frozenset({"text", "citations"})


@dataclass(frozen=True)
class DraftSentence:
    """One sentence of the model's answer, before verification."""

    text: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class DraftAnswer:
    """The model's answer, before verification.

    Attributes:
        sentences: The answer sentences in reading order.
        insufficient_evidence: The model's own flag that the passages do not
            fully answer the question.
        responses: Every model response received (one, or two after a retry).
        refused: True if the model declined to respond (``stop_reason`` was
            "refusal"); then ``sentences`` is empty.
        refusal_category: The category the API gave for a refusal, if any.
    """

    sentences: tuple[DraftSentence, ...]
    insufficient_evidence: bool
    responses: tuple[ModelResponse, ...]
    refused: bool = False
    refusal_category: str | None = None


def build_answer_request(
    question: str, passages: Sequence[RetrievedChunk], settings: Settings
) -> ModelRequest:
    """Build the Messages API request that asks for a cited answer."""
    return ModelRequest(
        purpose="answer",
        replay_key=question,
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=ANSWER_SYSTEM_PROMPT,
        tool=tool_definition(ANSWER_TOOL),
        user_message=answer_user_message(question, passages),
        effort=settings.effort,
        cache_system_prompt=settings.cache_system_prompt,
        fallback=settings.fallback,
    )


def parse_answer_input(raw: object) -> tuple[tuple[DraftSentence, ...], bool]:
    """Validate ``submit_answer`` arguments and convert them to draft sentences.

    Mirrors the strict JSON schema: exactly the keys ``sentences`` and
    ``insufficient_evidence``; each sentence exactly ``text`` (non-empty
    string) and ``citations`` (list of strings). Blank citation strings are
    dropped; a sentence left with no citations is later labelled unsupported.

    Raises:
        ModelResponseError: If the input does not match the schema.
    """
    if not isinstance(raw, Mapping):
        raise ModelResponseError("submit_answer input is not a JSON object")
    keys = set(raw)
    if keys != _ANSWER_KEYS:
        raise ModelResponseError(
            "submit_answer input must have exactly the keys 'sentences' and "
            f"'insufficient_evidence', got {sorted(map(str, keys))}"
        )
    insufficient = raw["insufficient_evidence"]
    if not isinstance(insufficient, bool):
        raise ModelResponseError("'insufficient_evidence' must be true or false")
    items = raw["sentences"]
    if not isinstance(items, list):
        raise ModelResponseError("'sentences' must be a list")
    sentences: list[DraftSentence] = []
    for number, item in enumerate(items, start=1):
        if not isinstance(item, Mapping) or set(item) != _SENTENCE_KEYS:
            raise ModelResponseError(
                f"sentence {number} must be an object with exactly 'text' and 'citations'"
            )
        text = item["text"]
        citations = item["citations"]
        if not isinstance(text, str) or not text.strip():
            raise ModelResponseError(f"sentence {number} has no text")
        if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
            raise ModelResponseError(f"sentence {number}: 'citations' must be a list of strings")
        sentences.append(
            DraftSentence(
                text=text.strip(),
                citations=tuple(c.strip() for c in citations if c.strip()),
            )
        )
    return tuple(sentences), insufficient


def request_answer(
    client: ModelClient,
    question: str,
    passages: Sequence[RetrievedChunk],
    settings: Settings,
) -> DraftAnswer:
    """Ask the model to answer ``question`` from ``passages`` only.

    Raises:
        ModelResponseError: If no valid ``submit_answer`` call comes back.
        ModelAPIError: If the API call itself fails (live mode).
        FixtureNotFoundError: If no fixture matches (fixture mode).
    """
    outcome = call_tool(
        client,
        build_answer_request(question, passages, settings),
        tool_name=ANSWER_TOOL_NAME,
        retry_reminder=ANSWER_RETRY_REMINDER,
    )
    if outcome.tool_input is None:
        return DraftAnswer(
            sentences=(),
            insufficient_evidence=True,
            responses=outcome.responses,
            refused=True,
            refusal_category=outcome.refusal_category,
        )
    sentences, insufficient = parse_answer_input(outcome.tool_input)
    return DraftAnswer(
        sentences=sentences, insufficient_evidence=insufficient, responses=outcome.responses
    )
