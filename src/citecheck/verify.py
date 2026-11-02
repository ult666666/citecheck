"""The support checker: does each sentence's own evidence back it up?

For every answer sentence, two checks run, in order:

1. **Citation validity.** A citation counts only if it names a passage that was
   actually retrieved and offered to the model. Anything else (a made-up ID, a
   typo, a real chunk that was not in this question's top-k) is *rejected*, and
   a sentence with no valid citation is unsupported.
2. **Lexical support.** The sentence's content terms (see
   :func:`citecheck.textnorm.content_terms`) are compared with the terms of the
   passages it validly cites (heading path plus text)::

       support = |sentence terms found in cited passages| / |sentence terms|

   Then:

   * any number in the sentence that appears in none of its cited passages
     makes it **unsupported** (wrong figures are the most damaging error in
     policy answers, and one wrong digit barely moves an overlap score);
   * otherwise ``support >= supported_threshold`` is **supported**,
     ``support >= weak_threshold`` is **weakly_supported**, and anything lower
     is **unsupported**.

This is word overlap, not entailment. It cannot see negation scope, word order,
or paraphrase ("staff" vs "employees"), and citing more passages makes a
sentence easier to pass. Those limits are spelled out in docs/DESIGN.md, and the
evaluation reports where the checker and a human disagree.

The default ``claims`` checker keeps steps 1 and 2 but splits the sentence into
claims and replaces the number rule with :mod:`citecheck.claims`, which compares
each claim with the cited clause it matches best: negation, modality, opposite
words, and every quantity with its unit. A claim with any problem is
unsupported; otherwise its own word overlap decides its label. The sentence gets
its worst claim's label. ``checker="lexical"`` keeps the original behavior.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

from citecheck.answer import DraftSentence
from citecheck.claims import analyze
from citecheck.config import Checker, Settings
from citecheck.index import RetrievedChunk
from citecheck.textnorm import content_terms, numbers_in

Label = Literal["supported", "weakly_supported", "unsupported"]
LABELS: Final[tuple[Label, ...]] = ("supported", "weakly_supported", "unsupported")
_SEVERITY: Final[dict[Label, int]] = {label: rank for rank, label in enumerate(LABELS)}


@dataclass(frozen=True)
class ClaimCheck:
    """The verdict for one claim of a sentence (claims checker only).

    Attributes:
        text: The claim, a sentence or a part of one.
        support: Fraction of the claim's content terms found in the cited passages.
        aligned: The cited clause the claim was compared with ("" if none).
        problems: Failed checks, one sentence each; any problem makes the claim
            unsupported.
        label: "supported", "weakly_supported" or "unsupported".
    """

    text: str
    support: float
    aligned: str
    problems: tuple[str, ...]
    label: Label


@dataclass(frozen=True)
class SentenceCheck:
    """The verdict for one sentence, with the evidence behind it.

    Attributes:
        number: 1-based position of the sentence in the answer.
        text: The sentence.
        citations: Citations as written by the model (duplicates removed).
        valid_citations: Citations that name a passage offered to the model.
        rejected_citations: Citations that do not.
        support: Fraction of the sentence's content terms found in its validly
            cited passages (0.0 when there is nothing to compare).
        matched_terms: Sentence terms found in the cited passages.
        missing_terms: Sentence terms not found in the cited passages.
        unsupported_numbers: Numbers in the sentence that no cited passage has.
        label: "supported", "weakly_supported" or "unsupported".
        reason: A one-line, human-readable explanation of the label.
        claims: Per-claim verdicts from the claims checker (empty for the
            lexical checker and for sentences rejected before comparison).
    """

    number: int
    text: str
    citations: tuple[str, ...]
    valid_citations: tuple[str, ...]
    rejected_citations: tuple[str, ...]
    support: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    unsupported_numbers: tuple[str, ...]
    label: Label
    reason: str
    claims: tuple[ClaimCheck, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        data = asdict(self)
        data["support"] = round(self.support, 4)
        for claim in data["claims"]:
            claim["support"] = round(claim["support"], 4)
        return data


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def check_sentence(
    number: int,
    sentence: DraftSentence,
    passages: Mapping[str, RetrievedChunk],
    *,
    supported_threshold: float,
    weak_threshold: float,
    checker: Checker = "claims",
    question: str = "",
) -> SentenceCheck:
    """Check one sentence against the passages it cites.

    Args:
        number: 1-based sentence position, for reporting.
        sentence: The sentence and its citations.
        passages: The passages that were offered to the model, by chunk ID.
        supported_threshold: Minimum support for "supported".
        weak_threshold: Minimum support for "weakly_supported".
        checker: "claims" (default) or "lexical"; see the module docstring.
        question: The question, so the claims checker can accept its numbers
            inside a comparison the passages support.
    """
    citations = _dedupe(sentence.citations)
    valid = tuple(c for c in citations if c in passages)
    rejected = tuple(c for c in citations if c not in passages)
    terms = content_terms(sentence.text)

    def verdict(
        label: Label,
        reason: str,
        *,
        support: float = 0.0,
        matched: frozenset[str] = frozenset(),
        numbers: frozenset[str] = frozenset(),
        claims: tuple[ClaimCheck, ...] = (),
    ) -> SentenceCheck:
        return SentenceCheck(
            number=number,
            text=sentence.text,
            citations=citations,
            valid_citations=valid,
            rejected_citations=rejected,
            support=support,
            matched_terms=tuple(sorted(matched)),
            missing_terms=tuple(sorted(terms - matched)),
            unsupported_numbers=tuple(sorted(numbers)),
            label=label,
            reason=reason,
            claims=claims,
        )

    if not valid:
        if not citations:
            return verdict("unsupported", "the sentence has no citations")
        return verdict(
            "unsupported",
            "every citation names a passage that was not retrieved: " + ", ".join(rejected),
        )
    source_text = "\n".join(f"{passages[c].title}\n{passages[c].text}" for c in valid)
    if not terms:
        return verdict("unsupported", "the sentence has no content words to check")

    source_terms = content_terms(source_text)
    matched = terms & source_terms
    support = len(matched) / len(terms)
    overlap = f"{len(matched)} of {len(terms)} content words found in the cited passages"
    if checker == "claims":
        return _check_claims(
            sentence.text,
            [passages[c].text for c in valid],
            source_terms,
            question,
            verdict,
            support=support,
            matched=matched,
            supported_threshold=supported_threshold,
            weak_threshold=weak_threshold,
        )
    numbers = numbers_in(sentence.text) - numbers_in(source_text)
    if numbers:
        return verdict(
            "unsupported",
            "numbers not in the cited passages: " + ", ".join(sorted(numbers)),
            support=support,
            matched=matched,
            numbers=numbers,
        )
    if support >= supported_threshold:
        return verdict("supported", overlap, support=support, matched=matched)
    if support >= weak_threshold:
        return verdict(
            "weakly_supported",
            f"{overlap} (below {supported_threshold:.2f})",
            support=support,
            matched=matched,
        )
    return verdict(
        "unsupported", f"{overlap} (below {weak_threshold:.2f})", support=support, matched=matched
    )


def _overlap_label(support: float, supported_threshold: float, weak_threshold: float) -> Label:
    if support >= supported_threshold:
        return "supported"
    return "weakly_supported" if support >= weak_threshold else "unsupported"


def _check_claims(
    text: str,
    passage_texts: Sequence[str],
    source_terms: frozenset[str],
    question: str,
    verdict: Callable[..., SentenceCheck],
    *,
    support: float,
    matched: frozenset[str],
    supported_threshold: float,
    weak_threshold: float,
) -> SentenceCheck:
    """Label each claim, then give the sentence its worst claim's label."""
    findings = analyze(text, passage_texts, question)
    claims: list[ClaimCheck] = []
    for finding in findings:
        # A question number accepted inside a comparison is not a claim about the passages.
        terms = content_terms(finding.text) - frozenset(finding.question_numbers)
        claim_support = len(terms & source_terms) / len(terms) if terms else support
        label = (
            "unsupported"
            if finding.problems
            else _overlap_label(claim_support, supported_threshold, weak_threshold)
        )
        claims.append(
            ClaimCheck(finding.text, claim_support, finding.aligned, finding.problems, label)
        )
    worst_index = max(range(len(claims)), key=lambda i: (_SEVERITY[claims[i].label], -i))
    worst = claims[worst_index]
    prefix = f"claim {worst_index + 1} of {len(claims)}: " if len(claims) > 1 else ""
    if worst.problems:
        reason = prefix + "; ".join(worst.problems)
    else:
        worst_terms = content_terms(worst.text) - frozenset(findings[worst_index].question_numbers)
        found = len(worst_terms & source_terms)
        reason = f"{prefix}{found} of {len(worst_terms)} content words found in the cited passages"
        if worst.label == "weakly_supported":
            reason += f" (below {supported_threshold:.2f})"
        elif worst.label == "unsupported":
            reason += f" (below {weak_threshold:.2f})"
    numbers = frozenset(number for finding in findings for number in finding.missing_numbers)
    return verdict(
        worst.label,
        reason,
        support=support,
        matched=matched,
        numbers=numbers,
        claims=tuple(claims),
    )


def verify_sentences(
    sentences: Sequence[DraftSentence],
    passages: Mapping[str, RetrievedChunk],
    settings: Settings,
    question: str = "",
) -> tuple[SentenceCheck, ...]:
    """Check every sentence of an answer; see :func:`check_sentence`."""
    return tuple(
        check_sentence(
            number,
            sentence,
            passages,
            supported_threshold=settings.supported_threshold,
            weak_threshold=settings.weak_threshold,
            checker=settings.checker,
            question=question,
        )
        for number, sentence in enumerate(sentences, start=1)
    )


def count_labels(checks: Iterable[SentenceCheck]) -> dict[Label, int]:
    """Count sentences per label (every label present, possibly with 0)."""
    counts: dict[Label, int] = dict.fromkeys(LABELS, 0)
    for check in checks:
        counts[check.label] += 1
    return counts
