import pytest

from citecheck.answer import DraftSentence
from citecheck.config import Checker, Settings
from citecheck.index import RetrievedChunk
from citecheck.verify import SentenceCheck, check_sentence, count_labels, verify_sentences

CARRYOVER = RetrievedChunk(
    chunk_id="time-off.md#carryover",
    doc_path="time-off.md",
    title="Paid Time Off > Carryover",
    start_line=17,
    end_line=19,
    text="Up to 5 unused PTO days carry over into the next calendar year.",
    score=18.6,
)
SICK = RetrievedChunk(
    chunk_id="time-off.md#sick-leave",
    doc_path="time-off.md",
    title="Paid Time Off > Sick leave",
    start_line=21,
    end_line=23,
    text="Employees receive 8 paid sick days per calendar year.",
    score=10.9,
)
PASSAGES = {chunk.chunk_id: chunk for chunk in (CARRYOVER, SICK)}


def check(
    text: str,
    *citations: str,
    supported: float = 0.8,
    weak: float = 0.5,
    checker: Checker = "claims",
) -> SentenceCheck:
    return check_sentence(
        1,
        DraftSentence(text, citations),
        PASSAGES,
        supported_threshold=supported,
        weak_threshold=weak,
        checker=checker,
    )


def test_fully_overlapping_sentence_is_supported() -> None:
    result = check("Up to 5 unused PTO days carry over.", CARRYOVER.chunk_id)
    assert result.label == "supported"
    assert result.support == 1.0
    assert result.reason == "7 of 7 content words found in the cited passages"
    assert result.missing_terms == ()


def test_partially_overlapping_sentence_is_weakly_supported() -> None:
    # 4 of 6 terms: unused, pto, day, carry found; manager, approval missing.
    result = check("Unused PTO days carry with manager approval.", CARRYOVER.chunk_id)
    assert result.label == "weakly_supported"
    assert result.support == pytest.approx(4 / 6)
    assert result.missing_terms == ("approval", "manager")
    assert "below 0.80" in result.reason


def test_low_overlap_is_unsupported() -> None:
    result = check("Managers approve remote work requests quickly.", CARRYOVER.chunk_id)
    assert result.label == "unsupported"
    assert "below 0.50" in result.reason


def test_a_number_missing_from_the_sources_overrides_high_overlap() -> None:
    result = check("Up to 10 unused PTO days carry over.", CARRYOVER.chunk_id, checker="lexical")
    assert result.label == "unsupported"
    assert result.unsupported_numbers == ("10",)
    assert result.support == pytest.approx(6 / 7)
    assert result.reason == "numbers not in the cited passages: 10"


def test_the_claims_checker_names_the_conflicting_value() -> None:
    result = check("Up to 10 unused PTO days carry over.", CARRYOVER.chunk_id)
    assert result.label == "unsupported"
    assert result.reason == "the claim says 10 days where the cited text says 5 days"
    assert [claim.label for claim in result.claims] == ["unsupported"]


@pytest.mark.parametrize(
    ("checker", "sentence"),
    [("lexical", "The limit is 1500 per year."), ("claims", "The limit is $1500 per year.")],
)
def test_numbers_are_compared_in_normalized_form(checker: Checker, sentence: str) -> None:
    passage = RetrievedChunk("p.md#x", "p.md", "P", 1, 1, "The limit is $1,500.00 per year.", 5)
    result = check_sentence(
        1,
        DraftSentence(sentence, ("p.md#x",)),
        {"p.md#x": passage},
        supported_threshold=0.8,
        weak_threshold=0.5,
        checker=checker,
    )
    assert result.label == "supported"


@pytest.mark.parametrize("checker", ["lexical", "claims"])
def test_a_wrong_number_written_with_a_suffix_is_unsupported(checker: Checker) -> None:
    # "15th" is not read as a quantity; the claims checker falls back to the number rule.
    text = "Timesheets are due on the 5th business day of each month."
    passage = RetrievedChunk("t.md#x", "t.md", "Timesheets", 1, 1, text, 5)
    result = check_sentence(
        1,
        DraftSentence(text.replace("5th", "15th"), ("t.md#x",)),
        {"t.md#x": passage},
        supported_threshold=0.8,
        weak_threshold=0.5,
        checker=checker,
    )
    assert result.label == "unsupported"
    assert result.unsupported_numbers == ("15",)


def test_sentence_without_citations_is_unsupported() -> None:
    result = check("Up to 5 unused PTO days carry over.")
    assert result.label == "unsupported"
    assert result.reason == "the sentence has no citations"
    assert result.support == 0.0


def test_citations_to_passages_that_were_not_retrieved_are_rejected() -> None:
    result = check("Up to 5 unused PTO days carry over.", "time-off.md#made-up")
    assert result.label == "unsupported"
    assert result.rejected_citations == ("time-off.md#made-up",)
    assert result.valid_citations == ()
    assert "not retrieved: time-off.md#made-up" in result.reason


def test_a_valid_citation_survives_alongside_a_rejected_one() -> None:
    result = check("Up to 5 unused PTO days carry over.", "nope.md#x", CARRYOVER.chunk_id)
    assert result.label == "supported"
    assert result.valid_citations == (CARRYOVER.chunk_id,)
    assert result.rejected_citations == ("nope.md#x",)


def test_duplicate_citations_are_collapsed() -> None:
    result = check("Up to 5 unused PTO days carry over.", CARRYOVER.chunk_id, CARRYOVER.chunk_id)
    assert result.citations == (CARRYOVER.chunk_id,)


def test_a_sentence_of_only_function_words_has_nothing_to_check() -> None:
    result = check("It is what it is.", CARRYOVER.chunk_id)
    assert result.label == "unsupported"
    assert result.reason == "the sentence has no content words to check"


def test_support_exactly_at_the_threshold_counts_as_supported() -> None:
    # "no" is not in the passage: 4 of 5 terms = 0.8.
    result = check("No, PTO days carry over.", CARRYOVER.chunk_id)
    assert result.support == pytest.approx(0.8)
    assert result.label == "supported"


def test_heading_words_count_as_evidence() -> None:
    result = check("The carryover rule covers 5 unused days.", CARRYOVER.chunk_id)
    assert "carryover" in result.matched_terms


def test_a_sentence_can_be_supported_by_two_passages_together() -> None:
    sentence = "Employees receive 8 sick days, and 5 unused PTO days carry over."
    assert check(sentence, SICK.chunk_id).label == "unsupported"  # 5 is missing
    assert check(sentence, SICK.chunk_id, CARRYOVER.chunk_id).label == "supported"


@pytest.mark.parametrize(
    ("checker", "label"), [("lexical", "supported"), ("claims", "unsupported")]
)
def test_negation_flips_fool_only_the_lexical_checker(checker: Checker, label: str) -> None:
    # Word overlap cannot see that this contradicts the source; the claims
    # checker compares the sentence with the clause it restates.
    passage = RetrievedChunk(
        "s.md#x", "s.md", "Sick", 1, 1, "A note is required. Sick days do not carry over.", 5
    )
    result = check_sentence(
        1,
        DraftSentence("A note is not required.", ("s.md#x",)),
        {"s.md#x": passage},
        supported_threshold=0.8,
        weak_threshold=0.5,
        checker=checker,
    )
    assert result.label == label


def test_thresholds_are_configurable() -> None:
    sentence = "Unused PTO days carry with manager approval."
    assert check(sentence, CARRYOVER.chunk_id, supported=0.6).label == "supported"
    assert check(sentence, CARRYOVER.chunk_id, weak=0.7, supported=0.9).label == "unsupported"


def test_verify_sentences_numbers_from_one_and_counts_labels() -> None:
    checks = verify_sentences(
        [
            DraftSentence("Up to 5 unused PTO days carry over.", (CARRYOVER.chunk_id,)),
            DraftSentence("Unsupported claim.", ()),
        ],
        PASSAGES,
        Settings(),
    )
    assert [c.number for c in checks] == [1, 2]
    assert count_labels(checks) == {"supported": 1, "weakly_supported": 0, "unsupported": 1}


def test_to_dict_rounds_support() -> None:
    data = check("Unused PTO days carry with manager approval.", CARRYOVER.chunk_id).to_dict()
    assert data["support"] == 0.6667
    assert data["label"] == "weakly_supported"
