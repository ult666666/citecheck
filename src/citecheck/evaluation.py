"""Score the pipeline against a labeled question set.

Each question in the set is labeled with whether the corpus answers it and, if
so, which chunks contain the answer ("gold chunks"). The evaluation runs every
question through :func:`citecheck.pipeline.ask` and reports:

* **recall@k**: per answerable question, the fraction of its gold chunks that
  were offered to the model, averaged over answerable questions (macro).
* **hit@k**: the fraction of answerable questions with at least one gold chunk
  offered.
* **citation precision**: of all chunk IDs cited (valid or rejected), pooled
  over every question, the fraction that are gold for their question.
* **citation recall**: of all gold chunk IDs of answerable questions, pooled,
  the fraction that were cited. Abstaining on an answerable question loses
  recall.
* **unsupported-sentence rate**: answer sentences the checker labels
  unsupported, over all answer sentences.
* **abstention accuracy**: the fraction of questions where the pipeline
  abstained exactly when the question is unanswerable.
* **checker vs. human** (fixture mode only): each fixture sentence carries a
  human label; this counts how many human-"unsupported" sentences the checker
  flagged (any label but "supported") and how many human-"supported" sentences
  it flagged anyway.

In fixture mode the "model" is a set of hand-written synthetic responses, so the
numbers demonstrate the harness and the checker, not the quality of any model.
"""

import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from citecheck.chunking import DEFAULT_MAX_CHUNK_CHARS
from citecheck.config import Settings
from citecheck.costs import UsageTotals
from citecheck.errors import EvalDataError
from citecheck.fixtures import FIXTURE_NOTICE, FixtureClient, replay_key
from citecheck.index import Index, build_index
from citecheck.llm import ClientSource, ModelClient
from citecheck.pipeline import CheckedAnswer, ask
from citecheck.verify import LABELS, Label, count_labels

_QUESTION_KEYS: Final = frozenset({"id", "question", "answerable", "gold_chunks", "notes"})
_REQUIRED_QUESTION_KEYS: Final = frozenset({"id", "question", "answerable", "gold_chunks"})


@dataclass(frozen=True)
class EvalQuestion:
    """One labeled question.

    Attributes:
        id: Short identifier, for example "q07".
        question: The question text.
        answerable: Whether the corpus contains the answer.
        gold_chunks: IDs of the chunks that contain the answer (empty when
            unanswerable).
        notes: Why the question is in the set (what it tests).
    """

    id: str
    question: str
    answerable: bool
    gold_chunks: tuple[str, ...]
    notes: str = ""


def _question_from(data: object, position: int) -> EvalQuestion:
    where = f"question #{position}"
    if not isinstance(data, dict):
        raise EvalDataError(f"{where} must be a JSON object")
    keys = set(data)
    if missing := sorted(_REQUIRED_QUESTION_KEYS - keys):
        raise EvalDataError(f"{where} is missing {', '.join(missing)}")
    if unknown := sorted(keys - _QUESTION_KEYS):
        raise EvalDataError(f"{where} has unknown keys: {', '.join(unknown)}")
    qid, text, answerable, gold = (
        data["id"],
        data["question"],
        data["answerable"],
        data["gold_chunks"],
    )
    notes = data.get("notes", "")
    if not isinstance(qid, str) or not qid.strip():
        raise EvalDataError(f"{where}: id must be a non-empty string")
    where = f"question {qid!r}"
    if not isinstance(text, str) or not text.strip():
        raise EvalDataError(f"{where}: question must be a non-empty string")
    if not isinstance(answerable, bool):
        raise EvalDataError(f"{where}: answerable must be true or false")
    if not isinstance(gold, list) or not all(isinstance(g, str) and g for g in gold):
        raise EvalDataError(f"{where}: gold_chunks must be a list of chunk IDs")
    if answerable != bool(gold):
        raise EvalDataError(
            f"{where}: answerable questions need gold_chunks and unanswerable ones must have none"
        )
    if not isinstance(notes, str):
        raise EvalDataError(f"{where}: notes must be a string")
    return EvalQuestion(
        id=qid, question=text, answerable=answerable, gold_chunks=tuple(gold), notes=notes
    )


def load_questions(path: Path) -> list[EvalQuestion]:
    """Load and validate a question set (``{"questions": [...]}`` JSON).

    Raises:
        EvalDataError: If the file is missing or malformed, or IDs or questions
            repeat.
    """
    if not path.is_file():
        raise EvalDataError(f"question file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalDataError(f"{path.name}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise EvalDataError(f'{path.name}: expected an object with a "questions" list')
    questions = [_question_from(item, n) for n, item in enumerate(data["questions"], start=1)]
    if not questions:
        raise EvalDataError(f"{path.name}: the question list is empty")
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for question in questions:
        if question.id in seen_ids:
            raise EvalDataError(f"{path.name}: duplicate question id {question.id!r}")
        key = replay_key(question.question)
        if key in seen_text:
            raise EvalDataError(f"{path.name}: question {question.id!r} is a duplicate")
        seen_ids.add(question.id)
        seen_text.add(key)
    return questions


@dataclass(frozen=True)
class QuestionResult:
    """The pipeline's result for one labeled question."""

    item: EvalQuestion
    answer: CheckedAnswer
    human_labels: tuple[str, ...] = ()

    @property
    def gold(self) -> frozenset[str]:
        """The question's gold chunk IDs."""
        return frozenset(self.item.gold_chunks)

    @property
    def offered(self) -> frozenset[str]:
        """Chunk IDs offered to the model."""
        return frozenset(passage.chunk_id for passage in self.answer.passages)

    @property
    def cited(self) -> frozenset[str]:
        """Every chunk ID cited in the answer, valid or rejected."""
        return frozenset(c for check in self.answer.sentences for c in check.citations)

    @property
    def abstained(self) -> bool:
        """Whether the pipeline gave no answer."""
        return self.answer.status == "abstained"

    @property
    def abstention_correct(self) -> bool:
        """Whether the pipeline abstained exactly when it should have."""
        return self.abstained == (not self.item.answerable)


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate metrics; see the module docstring for definitions."""

    questions: int
    answerable: int
    unanswerable: int
    recall_at_k: float | None
    hit_at_k: float | None
    citation_true_positives: int
    citation_false_positives: int
    citation_false_negatives: int
    citation_precision: float | None
    citation_recall: float | None
    rejected_citations: int
    sentences: int
    label_counts: dict[Label, int]
    unsupported_rate: float | None
    abstention_correct: int
    abstention_accuracy: float
    unanswerable_abstained: int
    unanswerable_abstained_without_model: int
    unanswerable_answered: int
    answerable_abstained: int
    human_labelled: int
    human_unsupported: int
    checker_caught: int
    human_supported: int
    checker_false_flags: int
    usage: UsageTotals


def summarize(results: Sequence[QuestionResult]) -> EvalSummary:
    """Compute every aggregate metric from per-question results."""
    answerable = [r for r in results if r.item.answerable]
    unanswerable = [r for r in results if not r.item.answerable]
    recalls = [len(r.gold & r.offered) / len(r.gold) for r in answerable]
    true_pos = sum(len(r.cited & r.gold) for r in results)
    false_pos = sum(len(r.cited - r.gold) for r in results)
    false_neg = sum(len(r.gold - r.cited) for r in answerable)
    checks = [check for r in results for check in r.answer.sentences]
    labels = count_labels(checks)
    caught = false_flags = human_unsupported = human_supported = 0
    for result in results:
        for check, human in zip(result.answer.sentences, result.human_labels, strict=False):
            flagged = check.label != "supported"
            if human == "unsupported":
                human_unsupported += 1
                caught += flagged
            else:
                human_supported += 1
                false_flags += flagged
    correct = sum(r.abstention_correct for r in results)
    usage = UsageTotals()
    for result in results:
        usage += result.answer.usage
    return EvalSummary(
        questions=len(results),
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        recall_at_k=_ratio(sum(recalls), len(recalls)),
        hit_at_k=_ratio(sum(1 for r in answerable if r.gold & r.offered), len(answerable)),
        citation_true_positives=true_pos,
        citation_false_positives=false_pos,
        citation_false_negatives=false_neg,
        citation_precision=_ratio(true_pos, true_pos + false_pos),
        citation_recall=_ratio(true_pos, true_pos + false_neg),
        rejected_citations=sum(len(r.answer.rejected_citations) for r in results),
        sentences=len(checks),
        label_counts=labels,
        unsupported_rate=_ratio(labels["unsupported"], len(checks)),
        abstention_correct=correct,
        abstention_accuracy=correct / len(results),
        unanswerable_abstained=sum(r.abstained for r in unanswerable),
        unanswerable_abstained_without_model=sum(
            r.abstained and r.answer.model_calls == 0 for r in unanswerable
        ),
        unanswerable_answered=sum(not r.abstained for r in unanswerable),
        answerable_abstained=sum(r.abstained for r in answerable),
        human_labelled=human_supported + human_unsupported,
        human_unsupported=human_unsupported,
        checker_caught=caught,
        human_supported=human_supported,
        checker_false_flags=false_flags,
        usage=usage,
    )


@dataclass(frozen=True)
class EvalReport:
    """Everything an evaluation run produced."""

    settings: Settings
    source: ClientSource
    questions_file: str
    results: tuple[QuestionResult, ...]
    summary: EvalSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary (deterministic: no timings)."""
        return {
            "source": self.source,
            "questions_file": self.questions_file,
            "settings": asdict(self.settings),
            "summary": asdict(self.summary),
            "questions": [
                {
                    "id": r.item.id,
                    "answerable": r.item.answerable,
                    "status": r.answer.status,
                    "abstain_reason": r.answer.abstain_reason,
                    "model_calls": r.answer.model_calls,
                    "gold_chunks": list(r.item.gold_chunks),
                    "offered": sorted(r.offered),
                    "cited": sorted(r.cited),
                    "labels": [check.label for check in r.answer.sentences],
                    "human_labels": list(r.human_labels),
                    "abstention_correct": r.abstention_correct,
                }
                for r in self.results
            ],
        }


def run_eval(
    questions: Sequence[EvalQuestion],
    index: Index,
    client: ModelClient,
    settings: Settings,
    *,
    questions_file: str = "questions.json",
) -> EvalReport:
    """Run every question through the pipeline and summarize.

    Raises:
        EvalDataError: If a gold chunk ID does not exist in the index (a typo
            in the labels would otherwise silently lower recall).
    """
    known = index.chunk_ids()
    for question in questions:
        if unknown := sorted(set(question.gold_chunks) - known):
            raise EvalDataError(
                f"question {question.id!r} names unknown gold chunks: {', '.join(unknown)}"
            )
    results: list[QuestionResult] = []
    for question in questions:
        answer = ask(question.question, index=index, client=client, settings=settings)
        labels: tuple[str, ...] = ()
        if isinstance(client, FixtureClient):
            fixture = client.fixture_for(question.question)
            labels = fixture.expected_labels if fixture is not None else ()
        results.append(QuestionResult(item=question, answer=answer, human_labels=labels))
    return EvalReport(
        settings=settings,
        source=client.source,
        questions_file=questions_file,
        results=tuple(results),
        summary=summarize(results),
    )


def evaluate(
    corpus_dir: Path,
    questions_path: Path,
    client: ModelClient,
    settings: Settings,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> EvalReport:
    """Build a fresh, temporary index of ``corpus_dir`` and evaluate on it.

    Building the index from scratch guarantees that the scores match the
    corpus in the repository, not a stale index left over from earlier.
    """
    questions = load_questions(questions_path)
    with tempfile.TemporaryDirectory(prefix="citecheck-eval-") as tmp:
        index_path = Path(tmp) / "index.sqlite"
        build_index(corpus_dir, index_path, max_chunk_chars=max_chunk_chars)
        return run_eval(
            questions, Index(index_path), client, settings, questions_file=questions_path.name
        )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _row(label: str, value: str, explanation: str = "") -> str:
    return f"  {label:<30}{value:<9}{explanation}".rstrip()


def format_report(report: EvalReport) -> str:
    """Render the summary as plain text (deterministic, so it can be diffed)."""
    s, cfg = report.summary, report.settings
    k = cfg.k
    lines = [
        f"citecheck eval: {s.questions} questions ({s.answerable} answerable, "
        f"{s.unanswerable} unanswerable) from {report.questions_file}",
        f"settings: k={k}, min_score={cfg.min_score:g}, supported>={cfg.supported_threshold:.2f}, "
        f"weak>={cfg.weak_threshold:.2f}",
    ]
    if report.source == "fixture":
        lines += [
            f"responses: {FIXTURE_NOTICE}.",
            "These numbers exercise the harness and the checker; they do not measure any model.",
        ]
    else:
        lines.append(f"responses: live calls to {cfg.model}.")
    lines += [
        "",
        f"Retrieval ({s.answerable} answerable questions)",
        _row(f"recall@{k}", _fmt(s.recall_at_k), "share of gold chunks offered, per question"),
        _row(f"hit@{k}", _fmt(s.hit_at_k), "questions with at least one gold chunk offered"),
        "Citations",
        _row(
            "precision",
            _fmt(s.citation_precision),
            f"{s.citation_true_positives} of "
            f"{s.citation_true_positives + s.citation_false_positives} cited chunk IDs are gold",
        ),
        _row(
            "recall",
            _fmt(s.citation_recall),
            f"{s.citation_true_positives} of "
            f"{s.citation_true_positives + s.citation_false_negatives} gold chunk IDs were cited",
        ),
        _row(
            "rejected citations", str(s.rejected_citations), "cited chunks that were not offered"
        ),
        f"Verification ({s.sentences} answer sentences)",
        *(_row(label.replace("_", " "), str(s.label_counts[label])) for label in LABELS),
        _row("unsupported-sentence rate", _fmt(s.unsupported_rate)),
        "Abstention",
        _row(
            "accuracy",
            _fmt(s.abstention_accuracy),
            f"{s.abstention_correct} of {s.questions} questions",
        ),
        _row(
            "unanswerable, abstained",
            f"{s.unanswerable_abstained} of {s.unanswerable}",
            f"{s.unanswerable_abstained_without_model} without a model call",
        ),
        _row(
            "unanswerable, answered", str(s.unanswerable_answered), "answers that should not exist"
        ),
        _row("answerable, abstained", f"{s.answerable_abstained} of {s.answerable}"),
    ]
    if s.human_labelled:
        lines += [
            "Checker vs. human labels in the fixtures",
            f"  flagged {s.checker_caught} of {s.human_unsupported} sentences labelled "
            f"unsupported (missed {s.human_unsupported - s.checker_caught})",
            f"  flagged {s.checker_false_flags} of {s.human_supported} sentences labelled "
            "supported (false flags)",
        ]
    usage = s.usage
    note = (
        "estimated at ~4 characters per token, not measured" if usage.estimated else "from the API"
    )
    lines += [
        f"Tokens ({note}): {usage.calls} model calls, {usage.input_tokens:,} input, "
        f"{usage.output_tokens:,} output, {usage.cache_read_input_tokens:,} cache-read, "
        f"{usage.cache_creation_input_tokens:,} cache-write",
    ]
    return "\n".join(lines)


def format_details(report: EvalReport) -> str:
    """Render a per-question table (status, gold hits, labels)."""
    header = (
        f"{'id':<5} {'answerable':<10} {'status':<9} {'calls':>5} {'gold offered':>12} "
        f"{'gold cited':>10}  labels (S/W/U)  abstention"
    )
    lines = [header, "-" * len(header)]
    for r in report.results:
        counts = count_labels(r.answer.sentences)
        gold = f"{len(r.gold & r.offered)}/{len(r.gold)}" if r.item.answerable else "-"
        cited = f"{len(r.gold & r.cited)}/{len(r.gold)}" if r.item.answerable else "-"
        labels = "/".join(str(counts[label]) for label in LABELS)
        lines.append(
            f"{r.item.id:<5} {('yes' if r.item.answerable else 'no'):<10} {r.answer.status:<9} "
            f"{r.answer.model_calls:>5} {gold:>12} {cited:>10}  {labels:<14}  "
            f"{'ok' if r.abstention_correct else 'WRONG'}"
        )
    return "\n".join(lines)
