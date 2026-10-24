"""The README and design notes must quote what the code actually prints."""

import json

import pytest

from citecheck.cli import format_answer
from citecheck.config import Settings
from citecheck.evaluation import evaluate, format_report
from citecheck.fixtures import FixtureClient
from citecheck.index import Index, build_index
from citecheck.pipeline import ask
from citecheck.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_TOOL
from tests.helpers import FIXTURES, HANDBOOK, QUESTIONS, REPO_ROOT

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
DESIGN = (REPO_ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")


def test_readme_shows_the_current_eval_output() -> None:
    report = evaluate(HANDBOOK, QUESTIONS, FixtureClient.from_directory(FIXTURES), Settings())
    assert format_report(report) in README


def test_readme_shows_the_current_quick_start_output(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    path = tmp_path_factory.mktemp("readme") / "index.sqlite"
    stats = build_index(HANDBOOK, path)
    assert f"Indexed {stats.documents} documents as {stats.chunks} chunks" in README
    answer = ask(
        "What is the deadline for submitting an expense?",
        index=Index(path),
        client=FixtureClient.from_directory(FIXTURES),
    )
    # Every line except the wall-clock timings must appear verbatim.
    for line in format_answer(answer).splitlines():
        if not line.startswith("Time:"):
            assert line in README, line


def test_documented_prefix_size_matches_the_prompt() -> None:
    size = len(ANSWER_SYSTEM_PROMPT) + len(json.dumps(ANSWER_TOOL))
    assert size == 2785
    assert "2,785 characters" in README
    assert "2,785 characters" in DESIGN


def test_documented_metric_values_match_the_pinned_ones() -> None:
    for fragment in ("29 / 33 = 0.879", "29 / 31 = 0.935", "6 / 49 = 0.122", "34 / 36 = 0.944"):
        assert fragment in DESIGN
