import doctest

import pytest
from hypothesis import given
from hypothesis import strategies as st

from citecheck import textnorm
from citecheck.textnorm import (
    STOPWORDS,
    content_terms,
    normalize_number,
    numbers_in,
    query_terms,
    stem_lite,
    tokenize,
)


def test_docstring_examples_hold() -> None:
    result = doctest.testmod(textnorm)
    assert result.attempted > 0
    assert result.failed == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,500", "1500"),
        ("1,500.00", "1500"),
        ("0.50", "0.5"),
        ("007", "7"),
        ("00", "0"),
        ("10.0", "10"),
        ("2.4.1", "2.4.1"),
    ],
)
def test_normalize_number(raw: str, expected: str) -> None:
    assert normalize_number(raw) == expected


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("policies", "policy"),
        ("applied", "apply"),
        ("addresses", "address"),
        ("taxes", "tax"),
        ("matches", "match"),
        ("wishes", "wish"),
        ("booking", "book"),
        ("approved", "approv"),
        ("receipts", "receipt"),
        ("status", "status"),
        ("bonus", "bonus"),
        ("analysis", "analysis"),
        ("speed", "speed"),
        ("submitted", "submit"),
        ("called", "call"),
        ("days", "day"),
        ("fixes", "fix"),
        ("lies", "lies"),
        ("used", "used"),
        ("day", "day"),
        ("2026", "2026"),
    ],
)
def test_stem_lite_rules(word: str, stem: str) -> None:
    assert stem_lite(word) == stem


@pytest.mark.parametrize(
    "forms",
    [
        ("approve", "approved", "approves"),
        ("expense", "expenses"),
        ("require", "required", "requires"),
        ("submit", "submitted", "submits", "submitting"),
        ("policy", "policies"),
    ],
)
def test_inflected_forms_share_a_stem(forms: tuple[str, ...]) -> None:
    assert len({stem_lite(form) for form in forms}) == 1


def test_derivations_are_not_conflated() -> None:
    # Documented limit: only inflections are stripped.
    assert stem_lite("approval") != stem_lite("approve")


def test_tokenize_handles_contractions_possessives_and_curly_apostrophes() -> None:
    text = "The employee\u2019s laptop isn\u2019t encrypted; it's Bob's."
    assert tokenize(text) == ["the", "employee", "laptop", "not", "encrypted", "it", "bob"]


def test_tokenize_normalizes_numbers() -> None:
    assert tokenize("Costs $1,500.00 or 2.50 at 9:00") == [
        "costs",
        "1500",
        "or",
        "2.5",
        "at",
        "9",
        "0",
    ]


def test_content_terms_keep_negations_and_comparisons_but_drop_function_words() -> None:
    assert content_terms("It is not a 401(k) plan, and it must be over 4 percent") == {
        "not",
        "401",
        "plan",
        "over",
        "4",
        "percent",
    }


def test_modal_verbs_are_stopwords() -> None:
    # Documented limit: "must" and "may" are indistinguishable to the checker.
    assert content_terms("Employees must submit") == content_terms("Employees may submit")


def test_numbers_in() -> None:
    assert numbers_in("SEV1 within 30 minutes, $1,000 to $10,000") == {"1", "30", "1000", "10000"}


def test_query_terms_split_like_unicode61_and_deduplicate() -> None:
    assert query_terms("Receipts, receipts & RECEIPTS for $1,500?") == ["receipts", "1", "500"]


def test_query_terms_drop_stopwords_and_single_letters() -> None:
    assert query_terms("What is a 401(k)?") == ["401"]
    assert query_terms("What is it?") == []


@given(st.text(max_size=300))
def test_query_terms_are_plain_alphanumeric_non_stopwords(text: str) -> None:
    terms = query_terms(text)
    assert len(terms) == len(set(terms))
    for term in terms:
        assert term
        assert all(char.isalnum() for char in term)
        assert term not in STOPWORDS
        assert len(term) > 1 or term.isdigit()


@given(st.text(max_size=200), st.text(max_size=200))
def test_content_terms_of_joined_passages_are_the_union(first: str, second: str) -> None:
    # The checker joins cited passages with newlines; this property is why the
    # join cannot create or destroy a term.
    assert content_terms(first + "\n" + second) == content_terms(first) | content_terms(second)
