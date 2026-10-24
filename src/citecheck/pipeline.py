"""The end-to-end flow: retrieve, (maybe) answer, verify, (optionally) judge.

::

    question
      -> retrieve top-k chunks with BM25          (citecheck.retrieval)
      -> none above min_score? abstain, no model call
      -> ask the model for a cited answer         (citecheck.answer)
      -> refusal / insufficient evidence? abstain
      -> check every sentence's citations/support (citecheck.verify)
      -> optional LLM judge                       (citecheck.judge)
      -> tokens, cost, timings                    (citecheck.costs)

The result is a :class:`CheckedAnswer`: the answer together with all of the
evidence needed to audit it (offered passages and scores, per-sentence verdicts
and reasons, model calls, token usage, cost, timings). It serializes to JSON for
the CLI's ``--json`` flag and the HTTP API.
"""

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal

from citecheck.answer import request_answer
from citecheck.config import Settings
from citecheck.costs import CostEstimate, PriceTable, UsageTotals, estimate_cost
from citecheck.index import Index, RetrievedChunk
from citecheck.judge import JudgeResult, run_judge
from citecheck.llm import ClientSource, ModelClient, ModelResponse
from citecheck.retrieval import RetrievalResult, retrieve, validate_question
from citecheck.verify import Label, SentenceCheck, count_labels, verify_sentences

Status = Literal["answered", "partial", "abstained"]


@dataclass(frozen=True)
class Timings:
    """Wall-clock time per stage, in milliseconds."""

    retrieval_ms: float
    model_ms: float
    verification_ms: float
    total_ms: float


@dataclass(frozen=True)
class CheckedAnswer:
    """An answer plus everything needed to audit it.

    Attributes:
        question: The question, stripped of surrounding whitespace.
        status: "answered"; "partial" (the model answered part and flagged
            insufficient evidence); or "abstained" (no answer).
        abstain_reason: Why there is no answer, when ``status`` is "abstained".
        passages: Chunks offered to the model, best first.
        below_threshold: Chunks retrieved but not offered (score < min_score).
        sentences: Per-sentence verdicts, in answer order.
        judge: The optional judge pass, or None if it did not run.
        responses: Every model response (answer and judge), for accounting.
        source: "fixture" or "live".
        settings: The settings used.
        cost: Cost under the supplied price table, or None if none was given.
        timings: Wall-clock time per stage.
    """

    question: str
    status: Status
    abstain_reason: str | None
    passages: tuple[RetrievedChunk, ...]
    below_threshold: tuple[RetrievedChunk, ...]
    sentences: tuple[SentenceCheck, ...]
    judge: JudgeResult | None
    responses: tuple[ModelResponse, ...]
    source: ClientSource
    settings: Settings
    cost: CostEstimate | None
    timings: Timings

    @property
    def model_calls(self) -> int:
        """Number of Messages API calls made (0 when retrieval abstained)."""
        return len(self.responses)

    @property
    def usage(self) -> UsageTotals:
        """Token totals over every model call."""
        return UsageTotals.from_responses(self.responses)

    @property
    def label_counts(self) -> dict[Label, int]:
        """Sentences per verifier label."""
        return count_labels(self.sentences)

    @property
    def rejected_citations(self) -> tuple[str, ...]:
        """Citations (across all sentences) to passages that were not offered."""
        return tuple(c for check in self.sentences for c in check.rejected_citations)

    @property
    def fallback_used(self) -> bool:
        """Whether a refusal fallback model served any call."""
        return any(response.fallback_used for response in self.responses)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        judge = None
        if self.judge is not None:
            judge = {
                "skipped_reason": self.judge.skipped_reason,
                "verdicts": {
                    str(number): asdict(verdict) for number, verdict in self.judge.verdicts.items()
                },
            }
        return {
            "question": self.question,
            "status": self.status,
            "abstain_reason": self.abstain_reason,
            "source": self.source,
            "model": self.settings.model if self.responses else None,
            # Fixture responses echo the requested model name, so "served_by"
            # is only reported for live calls.
            "served_by": sorted({response.model for response in self.responses})
            if self.source == "live"
            else [],
            "fallback_used": self.fallback_used,
            "sentences": [check.to_dict() for check in self.sentences],
            "label_counts": self.label_counts,
            "rejected_citations": list(self.rejected_citations),
            "passages": [passage.to_dict() for passage in self.passages],
            "below_threshold": [chunk.to_dict() for chunk in self.below_threshold],
            "judge": judge,
            "model_calls": self.model_calls,
            "usage": asdict(self.usage),
            "cost": None if self.cost is None else self.cost.to_dict(),
            "timings_ms": {key: round(value, 3) for key, value in asdict(self.timings).items()},
            "settings": asdict(self.settings),
        }


def _retrieval_abstain_reason(retrieval: RetrievalResult, settings: Settings) -> str:
    if retrieval.query is None:
        return "the question has no searchable words"
    if not retrieval.candidates:
        return "no passage matched the question's words"
    best = retrieval.candidates[0].score
    return f"no passage scored at least {settings.min_score:g}; the best score was {best:.2f}"


def ask(
    question: str,
    *,
    index: Index,
    client: ModelClient,
    settings: Settings | None = None,
    prices: PriceTable | None = None,
) -> CheckedAnswer:
    """Answer ``question`` from ``index`` and check every sentence.

    Args:
        question: The user's question.
        index: The search index to retrieve passages from.
        client: Where model responses come from (live API or fixtures).
        settings: Retrieval, model and verification settings.
        prices: Optional price table for a cost estimate.

    Raises:
        QueryError: If the question is empty or too long.
        ModelResponseError: If the model's reply cannot be used.
        ModelAPIError: If a live API call fails.
        FixtureNotFoundError: If fixture mode has no response for the question.
    """
    settings = settings or Settings()
    started = perf_counter()
    question = validate_question(question)
    retrieval = retrieve(index, question, k=settings.k, min_score=settings.min_score)
    retrieved_at = perf_counter()

    def result(
        status: Status,
        *,
        reason: str | None = None,
        sentences: tuple[SentenceCheck, ...] = (),
        judge: JudgeResult | None = None,
        responses: tuple[ModelResponse, ...] = (),
        model_ms: float = 0.0,
        verification_ms: float = 0.0,
    ) -> CheckedAnswer:
        return CheckedAnswer(
            question=question,
            status=status,
            abstain_reason=reason,
            passages=retrieval.passages,
            below_threshold=retrieval.below_threshold,
            sentences=sentences,
            judge=judge,
            responses=responses,
            source=client.source,
            settings=settings,
            cost=estimate_cost(responses, prices) if prices is not None and responses else None,
            timings=Timings(
                retrieval_ms=(retrieved_at - started) * 1000,
                model_ms=model_ms,
                verification_ms=verification_ms,
                total_ms=(perf_counter() - started) * 1000,
            ),
        )

    if not retrieval.passages:
        return result("abstained", reason=_retrieval_abstain_reason(retrieval, settings))

    draft = request_answer(client, question, retrieval.passages, settings)
    answered_at = perf_counter()
    model_ms = (answered_at - retrieved_at) * 1000
    if draft.refused:
        category = f" (category: {draft.refusal_category})" if draft.refusal_category else ""
        return result(
            "abstained",
            reason=f"the model declined to answer{category}",
            responses=draft.responses,
            model_ms=model_ms,
        )
    if not draft.sentences:
        reason = (
            "the model reported that the passages do not contain the answer"
            if draft.insufficient_evidence
            else "the model returned no sentences"
        )
        return result("abstained", reason=reason, responses=draft.responses, model_ms=model_ms)

    passages_by_id = {passage.chunk_id: passage for passage in retrieval.passages}
    checks = verify_sentences(draft.sentences, passages_by_id, settings)
    verified_at = perf_counter()
    responses = draft.responses
    judge = None
    if settings.judge:
        judge = run_judge(client, question, checks, passages_by_id, settings)
        responses += judge.responses
        model_ms += (perf_counter() - verified_at) * 1000
    return result(
        "partial" if draft.insufficient_evidence else "answered",
        sentences=checks,
        judge=judge,
        responses=responses,
        model_ms=model_ms,
        verification_ms=(verified_at - answered_at) * 1000,
    )
