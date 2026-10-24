"""Optional second opinion: ask a model whether each sentence is entailed.

The lexical checker in :mod:`citecheck.verify` is transparent but shallow. The
judge pass (``--judge``) sends each sentence with its cited passages to a model
and asks for "supported", "partially_supported" or "not_supported". Its verdict
is reported *next to* the lexical label, never instead of it: a model grading a
model is useful evidence, not ground truth, and it costs one more API call per
answer. Nothing in the test suite or the evaluation requires it.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, get_args

from citecheck.config import Settings
from citecheck.errors import ModelResponseError
from citecheck.index import RetrievedChunk
from citecheck.llm import ModelClient, ModelRequest, ModelResponse, call_tool
from citecheck.prompts import (
    JUDGE_RETRY_REMINDER,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TOOL,
    JUDGE_TOOL_NAME,
    judge_user_message,
    tool_definition,
)
from citecheck.verify import SentenceCheck

JudgeLabel = Literal["supported", "partially_supported", "not_supported"]
JUDGE_LABELS: Final[tuple[str, ...]] = get_args(JudgeLabel)
_VERDICT_KEYS: Final = frozenset({"sentence", "verdict", "reason"})


@dataclass(frozen=True)
class JudgeVerdict:
    """The judge's verdict on one sentence."""

    verdict: JudgeLabel
    reason: str


@dataclass(frozen=True)
class JudgeResult:
    """The outcome of a judge pass.

    Attributes:
        verdicts: Verdicts by sentence number. Sentences without a valid
            citation are not sent to the judge (the lexical check already
            marks them unsupported).
        responses: Every model response received, for usage accounting.
        skipped_reason: Why no verdicts were produced, if none were.
    """

    verdicts: Mapping[int, JudgeVerdict] = field(default_factory=dict)
    responses: tuple[ModelResponse, ...] = ()
    skipped_reason: str | None = None


def build_judge_request(
    question: str,
    checks: Sequence[SentenceCheck],
    passages: Mapping[str, RetrievedChunk],
    settings: Settings,
) -> ModelRequest | None:
    """Build the judge request, or return None if no sentence can be judged."""
    judged = [check for check in checks if check.valid_citations]
    if not judged:
        return None
    cited = dict.fromkeys(c for check in judged for c in check.valid_citations)
    return ModelRequest(
        purpose="judge",
        replay_key=question,
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=JUDGE_SYSTEM_PROMPT,
        tool=tool_definition(JUDGE_TOOL),
        user_message=judge_user_message(
            [(check.number, check.text, check.valid_citations) for check in judged],
            [passages[chunk_id] for chunk_id in cited],
        ),
        effort=settings.effort,
        cache_system_prompt=settings.cache_system_prompt,
        fallback=settings.fallback,
    )


def parse_judge_input(raw: object, expected: Collection[int]) -> dict[int, JudgeVerdict]:
    """Validate ``submit_verdicts`` arguments: one verdict per expected sentence.

    Raises:
        ModelResponseError: If the input does not match the schema, repeats or
            invents a sentence number, or leaves a sentence without a verdict.
    """
    if not isinstance(raw, Mapping) or set(raw) != {"verdicts"}:
        raise ModelResponseError("submit_verdicts input must be an object with only 'verdicts'")
    items = raw["verdicts"]
    if not isinstance(items, list):
        raise ModelResponseError("'verdicts' must be a list")
    verdicts: dict[int, JudgeVerdict] = {}
    for item in items:
        if not isinstance(item, Mapping) or set(item) != _VERDICT_KEYS:
            raise ModelResponseError(
                "each verdict must have exactly 'sentence', 'verdict' and 'reason'"
            )
        number, label, reason = item["sentence"], item["verdict"], item["reason"]
        if isinstance(number, bool) or not isinstance(number, int) or number not in expected:
            raise ModelResponseError(f"verdict for unknown sentence number {number!r}")
        if number in verdicts:
            raise ModelResponseError(f"sentence {number} has more than one verdict")
        if label not in JUDGE_LABELS:
            raise ModelResponseError(f"unknown verdict {label!r} for sentence {number}")
        if not isinstance(reason, str):
            raise ModelResponseError(f"the reason for sentence {number} must be a string")
        verdicts[number] = JudgeVerdict(verdict=label, reason=reason.strip())
    missing = sorted(set(expected) - set(verdicts))
    if missing:
        raise ModelResponseError(f"no verdict for sentence(s) {missing}")
    return verdicts


def run_judge(
    client: ModelClient,
    question: str,
    checks: Sequence[SentenceCheck],
    passages: Mapping[str, RetrievedChunk],
    settings: Settings,
) -> JudgeResult:
    """Ask the model for an entailment verdict on every validly cited sentence."""
    request = build_judge_request(question, checks, passages, settings)
    if request is None:
        return JudgeResult(skipped_reason="no sentence has a valid citation to judge")
    outcome = call_tool(
        client, request, tool_name=JUDGE_TOOL_NAME, retry_reminder=JUDGE_RETRY_REMINDER
    )
    if outcome.tool_input is None:
        return JudgeResult(
            responses=outcome.responses, skipped_reason="the judge model declined to respond"
        )
    expected = [check.number for check in checks if check.valid_citations]
    return JudgeResult(
        verdicts=parse_judge_input(outcome.tool_input, expected), responses=outcome.responses
    )
