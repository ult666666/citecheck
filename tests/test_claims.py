import hashlib
import json
import re
import time
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from citecheck.claims import (
    OPPOSITES,
    align,
    analyze,
    modality,
    negated,
    opposite_conflicts,
    passage_clauses,
    quantities,
    split_claims,
)
from citecheck.cli import main
from citecheck.config import Settings
from citecheck.errors import EvalDataError
from citecheck.evaluation import (
    claim_split,
    evaluate_claims,
    format_claim_report,
    load_claim_cases,
    mcnemar_p,
)
from citecheck.index import Index, RetrievedChunk, build_index
from tests.helpers import HANDBOOK, REPO_ROOT

CASES = REPO_ROOT / "eval" / "claims.json"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, RetrievedChunk]:
    path = tmp_path_factory.mktemp("claims") / "index.sqlite"
    build_index(HANDBOOK, path)
    index = Index(path)
    return index.chunks(index.chunk_ids())


def sentences(corpus: dict[str, RetrievedChunk]) -> list[tuple[str, str]]:
    """Every (sentence, chunk text) pair of the bundled handbook."""
    return [
        (sentence, chunk.text)
        for chunk in corpus.values()
        for sentence in {pair[1] for pair in passage_clauses([chunk.text])}
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$60 per month", [("money", "60", "per month")]),
        ("$400 for each week on call", [("money", "400", "per week")]),
        (
            "30, 60 and 90 days",
            [("count", "30", "day"), ("count", "60", "day"), ("count", "90", "day")],
        ),
        ("3 to 5 peers", [("count", "3", "peer"), ("count", "5", "peer")]),
        ("at least 10 business days", [("count", "10", "day")]),
        ("Up to 5 unused PTO days", [("count", "5", "day")]),
        ("90 percent and 4.5:1", [("percent", "90", ""), ("ratio", "4.5:1", "")]),
        ("at 9:30 a.m. or 22:00", [("time", "9:30", ""), ("time", "22:00", "")]),
        ("by March 31 in April", [("date", "march 31", ""), ("month", "april", "")]),
        ("You may book on Fridays", [("weekday", "friday", "")]),
        ("every quarter", [("frequency", "quarter", "")]),
        ("a SEV1 and SEV2", [("id", "sev1", "sev"), ("id", "sev2", "sev")]),
        ("two retakes and one week", [("count", "2", "retak"), ("count", "1", "week")]),
        ("the 20-day limit", [("count", "20", "day")]),
        ("Ask one of them.", []),
    ],
)
def test_quantities_keep_values_with_their_units(
    text: str, expected: list[tuple[str, str, str]]
) -> None:
    assert [(q.kind, q.value, q.unit) for q in quantities(text)] == expected


def test_sentences_split_into_claims_only_between_substantial_parts() -> None:
    assert split_claims("Submit receipts within 30 days, and late receipts are rejected.") == [
        "Submit receipts within 30 days",
        "and late receipts are rejected.",
    ]
    assert split_claims("Courses, books, and workshops are covered.") == [
        "Courses, books, and workshops are covered."
    ]
    assert split_claims("No, deployments stop on Fridays.") == ["deployments stop on Fridays."]


def test_negation_modality_and_opposites() -> None:
    assert negated("Sick days do not carry over.")
    assert not negated("Access lasts no more than 72 hours.")
    assert modality("Laptops must lock.") == {"obligation"}
    assert modality("You may work remotely; it is optional.") == {"permission"}
    assert opposite_conflicts(
        "Request approval after it starts.", "Request approval before it starts."
    )
    assert not opposite_conflicts("Before and after", "before or after")
    assert len(set(OPPOSITES)) == len(OPPOSITES)


def test_alignment_must_be_unambiguous() -> None:
    clauses = [
        ("Deployments happen Monday to Thursday.", "s1"),
        ("No deployments ship on Fridays.", "s2"),
    ]
    assert (
        align("Deployments happen weekly.", clauses)[0] == "Deployments happen Monday to Thursday."
    )
    assert align("deployments ship and happen", clauses) == ("", "")  # two shared words each
    assert align("unrelated deployments", clauses) == ("", "")  # only one shared word


@pytest.mark.parametrize(
    ("sentence", "passage", "question", "fragment"),
    [
        (
            "Sick days carry over.",
            "Sick days do not carry over.",
            "",
            "only the cited text is negated",
        ),
        (
            "Employees must use a phased return.",
            "Employees may use a phased return.",
            "",
            "obligation",
        ),
        (
            "Book after you travel.",
            "Book before you travel.",
            "",
            "'after' where the cited text says 'before'",
        ),
        (
            "Backups are purged within 30 days.",
            "Data is deleted within 30 days. Backups are purged within 90 days.",
            "",
            "30 days where the cited text says 90 days",
        ),
        (
            "Access lasts 72 days.",
            "Access lasts 72 hours.",
            "",
            "72 days where the cited text says 72 hours",
        ),
        ("The stipend is $60 per year.", "The stipend is $60 per month.", "", "$60 per year"),
        (
            "A $12,000 purchase needs the department head.",
            "Purchases from $1,000 to $10,000 need the department head.",
            "Who approves a $12,000 purchase?",
            "$12,000",
        ),
        (
            "A note is required for absences of more than 3 sick days.",
            "A note is required for absences of no more than 3 sick days.",
            "",
            "'more than' where the cited text says 'no more than'",
        ),
        (
            "Employees who joined not less than 90 days before a cycle skip it.",
            "Employees who joined less than 90 days before a cycle skip it.",
            "",
            "'no less than' where the cited text says 'less than'",
        ),
        (
            "The front desk opens at 9am on weekdays.",
            "The front desk opens at 8:00 on weekdays.",
            "",
            "9 is not in the cited passages",
        ),
        (
            "Timesheets are due on the 15th business day of the month.",
            "Timesheets are due on the 5th business day of the month.",
            "",
            "15 is not in the cited passages",
        ),
        (
            "Report an incident within 48h of finding it.",
            "Report an incident within 24 hours of finding it.",
            "",
            "48 is not in the cited passages",
        ),
    ],
)
def test_each_check_flags_its_contradiction(
    sentence: str, passage: str, question: str, fragment: str
) -> None:
    problems = [p for finding in analyze(sentence, [passage], question) for p in finding.problems]
    assert any(fragment in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("sentence", "passage", "question"),
    [
        (
            "A $5,000 purchase falls in the $1,000 to $10,000 range.",
            "Purchases from $1,000 to $10,000 need approval.",
            "Who approves a $5,000 purchase?",
        ),
        (
            "A $700 purchase is under $1,000.",
            "Purchases under $1,000 need manager approval.",
            "Who approves a $700 purchase?",
        ),
        ("Access lasts no more than 72 hours.", "Access is granted for at most 72 hours.", ""),
        ("Access lasts not more than 72 hours.", "Access lasts no more than 72 hours.", ""),
        (
            "Timesheets are due on the 5th business day.",
            "Timesheets are due on the 5th business day of the month.",
            "",
        ),
        (
            "Gifts above $50 must be reported.",
            "Gifts worth up to $50 per person per year are fine. "
            "Anything above that value must be reported.",
            "",
        ),
        (
            "Managers meet new hires at 30, 60 and 90 days.",
            "Managers hold check-ins at 30, 60 and 90 days.",
            "",
        ),
    ],
)
def test_faithful_sentences_raise_no_problem(sentence: str, passage: str, question: str) -> None:
    assert [f.problems for f in analyze(sentence, [passage], question)] == [()] * len(
        split_claims(sentence)
    )


def test_every_handbook_sentence_copied_verbatim_passes(corpus: dict[str, RetrievedChunk]) -> None:
    pairs = sentences(corpus)
    assert len(pairs) > 150
    failures = [s for s, text in pairs if any(f.problems for f in analyze(s, [text]))]
    assert failures == []


def test_inserting_a_negation_into_any_aligned_sentence_is_caught(
    corpus: dict[str, RetrievedChunk],
) -> None:
    tried = 0
    for sentence, text in sentences(corpus):
        match = re.search(r"\b(is|are)\b", sentence)
        if negated(sentence) or not match or not all(f.aligned for f in analyze(sentence, [text])):
            continue
        flipped = sentence[: match.end()] + " not" + sentence[match.end() :]
        tried += 1
        assert any(f.problems for f in analyze(flipped, [text])), flipped
    assert tried > 40


def test_negating_any_comparison_in_an_aligned_sentence_is_caught(
    corpus: dict[str, RetrievedChunk],
) -> None:
    # "No more than" is not a negation for the polarity check, so adding the "no" to
    # "more than" needs its own comparison.
    tried = 0
    for sentence, text in sentences(corpus):
        match = re.search(r"\b(?:more|less|fewer|later|earlier) than\b", sentence)
        if not match or not all(f.aligned for f in analyze(sentence, [text])):
            continue
        flipped = sentence[: match.start()] + "no " + sentence[match.start() :]
        tried += 1
        assert any(f.problems for f in analyze(flipped, [text])), flipped
    assert tried >= 3


def test_quantities_stay_linear_on_a_long_run_of_digits() -> None:
    # A run of digits followed by a letter used to cost quadratic time: about 20 seconds
    # for 32,000 digits. Every pattern must stay linear on untrusted passages and answers.
    run = "1" * 30_000 + "a"
    started = time.perf_counter()
    assert quantities(run) == []
    analyze("Submit expenses within 30 days.", [f"Submit expenses within 30 days. {run}"])
    assert time.perf_counter() - started < 2


def test_changing_any_counted_number_is_caught(corpus: dict[str, RetrievedChunk]) -> None:
    tried = 0
    for sentence, text in sentences(corpus):
        counts = [
            q for q in quantities(sentence) if q.kind == "count" and q.unit and q.value.isdigit()
        ]
        if not counts:
            continue
        number = counts[0].text.split(" ")[0]
        taken = {q.value for q in quantities(text)}
        value = next(str(v) for v in range(int(counts[0].value) + 1, 10**6) if str(v) not in taken)
        changed = re.sub(rf"\b{re.escape(number)}\b", value, sentence, count=1)
        tried += 1
        assert any(f.problems for f in analyze(changed, [text])), changed
    assert tried > 30


def test_swapping_a_word_for_its_opposite_is_caught(corpus: dict[str, RetrievedChunk]) -> None:
    tried = 0
    for sentence, text in sentences(corpus):
        words = set(re.findall(r"[a-z]+", sentence.casefold()))
        for first, second in OPPOSITES:
            if (first in words) == (second in words):
                continue
            mine, other = (first, second) if first in words else (second, first)
            swapped = re.sub(rf"\b{mine}\b", other, sentence, count=1, flags=re.I)
            if not all(f.aligned for f in analyze(sentence, [text])):
                continue
            tried += 1
            assert any(f.problems for f in analyze(swapped, [text])), swapped
            break
    assert tried > 15


@given(st.text(max_size=300))
def test_any_text_is_analyzed_without_errors(text: str) -> None:
    findings = analyze(text, [text, "The limit is $250 per night, 30 days, SEV1."], text)
    assert all(isinstance(problem, str) for finding in findings for problem in finding.problems)


def test_claim_cases_keep_their_frozen_split_and_bytes() -> None:
    assert hashlib.sha256(CASES.read_bytes()).hexdigest() == (
        "96fd9a852982fb978cc77c4a607079a56a86f543d30543e75b81f03230d72fff"
    )
    cases = load_claim_cases(CASES)
    assert len(cases) == 140
    assert all(case.split == claim_split(case.id) for case in cases)
    assert sum(case.split == "test" for case in cases) == 77


def test_claim_cases_cite_existing_chunks(corpus: dict[str, RetrievedChunk]) -> None:
    assert {c for case in load_claim_cases(CASES) for c in case.chunks} <= set(corpus)


@pytest.mark.parametrize(
    "change",
    [
        {"split": "dev" if claim_split("c001") == "test" else "test"},
        {"label": "maybe"},
        {"chunks": []},
        {"extra": 1},
        {"sentence": " "},
    ],
)
def test_claim_case_loader_rejects_bad_cases(tmp_path: Path, change: dict[str, object]) -> None:
    case = {
        "id": "c001",
        "split": claim_split("c001"),
        "chunks": ["a.md#b"],
        "sentence": "Text.",
        "label": "supported",
        "category": "restatement",
        **change,
    }
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    with pytest.raises(EvalDataError):
        load_claim_cases(path)


def test_exact_mcnemar_values() -> None:
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(0, 6) == pytest.approx(1 / 32)
    assert mcnemar_p(1, 7) == pytest.approx(18 / 256)
    assert mcnemar_p(3, 3) == 1.0
    assert mcnemar_p(1, 23) == pytest.approx(50 / 2**24)
    # Cross-checked against scipy.stats.binomtest (SciPy is not a dependency).
    assert mcnemar_p(430, 500) == pytest.approx(0.0236088392925, rel=1e-9)
    assert mcnemar_p(600, 600) == 1.0


def test_held_out_numbers() -> None:
    report = evaluate_claims(HANDBOOK, CASES, Settings(), split="test")
    lexical, claims = report.scores["lexical"], report.scores["claims"]
    assert (lexical.caught, lexical.unsupported, lexical.false_flags, lexical.supported) == (
        20,
        48,
        1,
        29,
    )
    assert (claims.caught, claims.unsupported, claims.false_flags, claims.supported) == (
        44,
        48,
        2,
        29,
    )
    assert report.discordant() == (1, 24)
    text = format_claim_report(report)
    assert "exact McNemar p = 1.5e-06" in text
    assert "accuracy                      0.623       0.922" in text


def test_dev_numbers() -> None:
    report = evaluate_claims(HANDBOOK, CASES, Settings(), split="dev")
    lexical, claims = report.scores["lexical"], report.scores["claims"]
    assert (lexical.caught, lexical.unsupported, lexical.false_flags) == (6, 34, 8)
    assert (claims.caught, claims.unsupported, claims.false_flags) == (28, 34, 7)


def test_eval_claims_command(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["eval-claims", "--corpus", str(HANDBOOK), "--cases", str(CASES)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "77 test cases (48 unsupported, 29 supported)" in out
    assert main([*args, "--split", "all", "--json", "--details"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["cases"]) == 140
    assert data["scores"]["claims"]["caught"] == 72
    assert main([*args, "--details"]) == 0
    assert "Cases where a checker disagrees with the human label" in capsys.readouterr().out
