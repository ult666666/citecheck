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
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

from citecheck.answer import DraftSentence
from citecheck.config import Settings
from citecheck.index import RetrievedChunk
from citecheck.textnorm import content_terms, numbers_in

Label = Literal["supported", "weakly_supported", "unsupported"]
LABELS: Final[tuple[Label, ...]] = ("supported", "weakly_supported", "unsupported")


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        data = asdict(self)
        data["support"] = round(self.support, 4)
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
) -> SentenceCheck:
    """Check one sentence against the passages it cites.

    Args:
        number: 1-based sentence position, for reporting.
        sentence: The sentence and its citations.
        passages: The passages that were offered to the model, by chunk ID.
        supported_threshold: Minimum support for "supported".
        weak_threshold: Minimum support for "weakly_supported".
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

    matched = terms & content_terms(source_text)
    support = len(matched) / len(terms)
    overlap = f"{len(matched)} of {len(terms)} content words found in the cited passages"
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


def verify_sentences(
    sentences: Sequence[DraftSentence],
    passages: Mapping[str, RetrievedChunk],
    settings: Settings,
) -> tuple[SentenceCheck, ...]:
    """Check every sentence of an answer; see :func:`check_sentence`."""
    return tuple(
        check_sentence(
            number,
            sentence,
            passages,
            supported_threshold=settings.supported_threshold,
            weak_threshold=settings.weak_threshold,
        )
        for number, sentence in enumerate(sentences, start=1)
    )


def count_labels(checks: Iterable[SentenceCheck]) -> dict[Label, int]:
    """Count sentences per label (every label present, possibly with 0)."""
    counts: dict[Label, int] = dict.fromkeys(LABELS, 0)
    for check in checks:
        counts[check.label] += 1
    return counts
