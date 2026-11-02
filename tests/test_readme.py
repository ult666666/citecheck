"""The README and the docs must quote what the code actually prints."""

import json

import pytest

from citecheck.cli import format_answer
from citecheck.config import Settings
from citecheck.evaluation import evaluate, evaluate_claims, format_claim_report, format_report
from citecheck.fixtures import FixtureClient
from citecheck.index import Index, build_index
from citecheck.pipeline import ask
from citecheck.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_TOOL
from tests.helpers import FIXTURES, HANDBOOK, QUESTIONS, REPO_ROOT

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
DESIGN = (REPO_ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")
EVALUATION = (REPO_ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def handbook_index(tmp_path_factory: pytest.TempPathFactory) -> Index:
    path = tmp_path_factory.mktemp("readme") / "index.sqlite"
    stats = build_index(HANDBOOK, path)
    assert f"Indexed {stats.documents} documents as {stats.chunks} chunks" in README
    return Index(path)


def test_evaluation_doc_shows_the_current_eval_output() -> None:
    report = evaluate(HANDBOOK, QUESTIONS, FixtureClient.from_directory(FIXTURES), Settings())
    assert format_report(report) in EVALUATION


def test_readme_shows_the_current_quick_start_output(handbook_index: Index) -> None:
    answer = ask(
        "What is the deadline for submitting an expense?",
        index=handbook_index,
        client=FixtureClient.from_directory(FIXTURES),
    )
    # Every line except the wall-clock timings must appear verbatim.
    for line in format_answer(answer).splitlines():
        if not line.startswith("Time:"):
            assert line in README, line


def test_readme_shows_the_claim_checker_flagging_the_negation(handbook_index: Index) -> None:
    answer = ask(
        "Do I need a doctor's note when I take a sick day?",
        index=handbook_index,
        client=FixtureClient.from_directory(FIXTURES),
    )
    lines = format_answer(answer).splitlines()
    first = lines.index("Answer:") + 1
    # The README quotes the first answer sentence, its citation and its verdict.
    quoted = lines[first : first + 3]
    assert quoted[0].startswith("  1. A doctor's note is not required")
    assert "unsupported: only the claim is negated" in quoted[2]
    assert "\n".join(quoted) in README


def test_documented_prefix_size_matches_the_prompt() -> None:
    size = len(ANSWER_SYSTEM_PROMPT) + len(json.dumps(ANSWER_TOOL))
    assert size == 2785
    assert "2,785 characters" in DESIGN


def test_documented_metric_values_match_the_pinned_ones() -> None:
    for fragment in ("29 / 33 = 0.879", "29 / 31 = 0.935", "6 / 49 = 0.122", "34 / 36 = 0.944"):
        assert fragment in DESIGN


def test_evaluation_doc_shows_the_current_claim_case_output() -> None:
    report = evaluate_claims(
        HANDBOOK, REPO_ROOT / "eval" / "claims.json", Settings(), split="test"
    )
    assert format_claim_report(report) in EVALUATION


@pytest.mark.parametrize("split", ["test", "dev"])
def test_claim_case_tables_match_the_reports(split: str) -> None:
    report = evaluate_claims(HANDBOOK, REPO_ROOT / "eval" / "claims.json", Settings(), split=split)
    lexical, claims = report.scores["lexical"], report.scores["claims"]
    name = "test (held out)" if split == "test" else "dev (tuning)"
    for checker, score in (("lexical", lexical), ("claims", claims)):
        row = (
            f"| {name} | {checker} | {score.caught} of {score.unsupported} | "
            f"{score.false_flags} of {score.supported} | {score.accuracy:.3f} |"
        )
        assert row in EVALUATION, row
    if split == "test":
        assert f"The two checkers were scored on {len(report.cases)} held-out sentences" in README
        rows = (
            f"| Unsupported sentences caught | {lexical.caught} of {lexical.unsupported} | "
            f"{claims.caught} of {claims.unsupported} |",
            f"| Supported sentences wrongly flagged | {lexical.false_flags} of "
            f"{lexical.supported} | {claims.false_flags} of {claims.supported} |",
            f"| Accuracy | {lexical.accuracy:.3f} | {claims.accuracy:.3f} |",
        )
        for row in rows:
            assert row in README, row
