import json
import runpy
import sys
from pathlib import Path
from typing import Any, cast

import anthropic
import pytest

from citecheck import cli
from citecheck.cli import main
from citecheck.config import Settings
from citecheck.index import Index
from citecheck.llm import AnthropicClient
from citecheck.pipeline import ask as run_pipeline
from tests.helpers import (
    FIXTURES,
    HANDBOOK,
    QUESTIONS,
    ScriptedClient,
    beta_message,
    tool_response,
)

CARRYOVER_Q = "How many unused PTO days carry over into the next year?"


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "index.sqlite"
    assert main(["index", str(HANDBOOK), "--db", str(path)]) == 0
    return path


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def ask(capsys: pytest.CaptureFixture[str], db: Path, *args: str) -> tuple[int, str, str]:
    return run(capsys, "ask", *args, "--db", str(db), "--fixtures", str(FIXTURES))


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "citecheck 0.1.0"


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_index_reports_what_it_built(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = tmp_path / "x" / "index.sqlite"
    code, out, _ = run(capsys, "index", str(HANDBOOK), "--db", str(path))
    assert code == 0
    assert out.startswith("Indexed 25 documents as 103 chunks into ")


def test_search_marks_offered_and_below_threshold_chunks(
    capsys: pytest.CaptureFixture[str], db: Path
) -> None:
    code, out, _ = run(capsys, "search", "Who founded Larkmoor Analytics?", "--db", str(db))
    assert code == 0
    assert out.splitlines()[0] == 'FTS5 query: "founded" OR "larkmoor" OR "analytics"'
    assert "people/time-off.md#paid-time-off" in out
    assert "[offered]" in out
    assert "[below threshold]" in out


def test_search_json_and_edge_cases(capsys: pytest.CaptureFixture[str], db: Path) -> None:
    code, out, _ = run(capsys, "search", "hotel limit", "--db", str(db), "--json")
    assert code == 0
    assert json.loads(out)["offered"][0]["chunk_id"] == "workplace/travel.md#hotels"
    assert run(capsys, "search", "what is it", "--db", str(db))[1].strip() == (
        "No searchable words in the query."
    )
    assert "No chunks matched." in run(capsys, "search", "zeppelin", "--db", str(db))[1]


def test_invalid_settings_exit_with_an_error(capsys: pytest.CaptureFixture[str], db: Path) -> None:
    code, _, err = run(capsys, "search", "hotel", "--db", str(db), "-k", "0")
    assert code == 1
    assert err.startswith("citecheck: error: k must be between 1 and 50")


def test_ask_prints_verdicts_and_the_fixture_notice(
    capsys: pytest.CaptureFixture[str], db: Path
) -> None:
    code, out, _ = ask(capsys, db, "What is the deadline for submitting an expense?")
    assert code == 0
    assert (
        "Mode: fixture replay: a hand-written synthetic response, not a real model output" in out
    )
    assert "-> supported: " in out
    assert "-> unsupported: numbers not in the cited passages: 60" in out
    assert "Summary: 1 supported, 0 weakly supported, 1 unsupported; 0 rejected citations" in out
    assert "Model calls: 1 replayed from fixtures (no API call was made)" in out
    assert "(estimated at ~4 characters per token, not measured)" in out
    assert "Cost: not computed (pass --prices FILE)" in out


def test_ask_shows_missing_terms_for_flagged_sentences(
    capsys: pytest.CaptureFixture[str], db: Path
) -> None:
    out = ask(capsys, db, "What should I do if I clicked a link in a phishing email?")[1]
    assert "-> weakly_supported: 4 of 6 content words" in out
    assert "not found in cited passages: manager, notify" in out


def test_ask_json(capsys: pytest.CaptureFixture[str], db: Path) -> None:
    code, out, _ = ask(capsys, db, CARRYOVER_Q, "--json")
    data = json.loads(out)
    assert code == 0
    assert data["status"] == "answered"
    assert data["source"] == "fixture"
    assert data["label_counts"]["supported"] == 2


def test_ask_with_prices_and_judge(capsys: pytest.CaptureFixture[str], db: Path) -> None:
    prices = str(HANDBOOK.parent / "prices.example.toml")
    out = ask(
        capsys,
        db,
        "Do I need a doctor's note when I take a sick day?",
        "--judge",
        "--prices",
        prices,
    )[1]
    assert "judge: not_supported" in out
    assert "Cost: 0." in out
    assert "demonstrates the arithmetic only" in out


def test_ask_with_a_price_table_missing_the_model(
    capsys: pytest.CaptureFixture[str], db: Path, tmp_path: Path
) -> None:
    prices = tmp_path / "prices.toml"
    prices.write_text(
        '[models."other"]\ninput_per_mtok = 1\noutput_per_mtok = 1\n'
        "cache_write_per_mtok = 1\ncache_read_per_mtok = 1\n",
        encoding="utf-8",
    )
    out = ask(capsys, db, CARRYOVER_Q, "--prices", str(prices))[1]
    assert "Cost: unknown (no price for: claude-opus-5-5)" in out


def test_partial_answers_are_labelled(capsys: pytest.CaptureFixture[str], db: Path) -> None:
    question = (
        "How quickly must an on-call engineer acknowledge a page, and what is the on-call stipend?"
    )
    partial = ask(capsys, db, question)[1]
    assert "Answer (partial: the model flagged" in partial


def test_ask_abstention(capsys: pytest.CaptureFixture[str], db: Path) -> None:
    out = ask(capsys, db, "What is Larkmoor's stock ticker symbol?")[1]
    assert "Answer: none (abstained): no passage scored at least 3; the best score was 2.07" in out
    assert "Model calls: 0" in out
    assert "Retrieved but below the score threshold (not offered):" in out


def test_ask_without_a_fixture_explains_fixture_mode(
    capsys: pytest.CaptureFixture[str], db: Path
) -> None:
    code, _, err = ask(capsys, db, "How do I expense a conference hotel?")
    assert code == 1
    assert "no fixture for the question" in err
    assert "--live" in err


def test_ask_needs_an_index(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, _, err = ask(capsys, tmp_path / "missing.sqlite", CARRYOVER_Q)
    assert code == 1
    assert "no index found" in err


def test_dry_run_prints_the_request_without_loading_any_client(
    capsys: pytest.CaptureFixture[str], db: Path, tmp_path: Path
) -> None:
    code, out, _ = run(
        capsys,
        "ask",
        CARRYOVER_Q,
        "--db",
        str(db),
        "--dry-run",
        "--fixtures",
        str(tmp_path / "does-not-exist"),
    )
    params = json.loads(out)
    assert code == 0
    assert params["tools"][0]["name"] == "submit_answer"
    assert params["output_config"] == {"effort": "medium"}
    assert params["betas"] == ["server-side-fallback-2026-07-01"]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_dry_run_with_options_off(capsys: pytest.CaptureFixture[str], db: Path) -> None:
    out = run(
        capsys,
        "ask",
        CARRYOVER_Q,
        "--db",
        str(db),
        "--dry-run",
        "--effort",
        "none",
        "--no-fallback",
        "--no-cache",
        "--model",
        "claude-sonnet-5-5",
    )[1]
    params = json.loads(out)
    assert params["model"] == "claude-sonnet-5-5"
    assert "output_config" not in params
    assert "betas" not in params
    assert "cache_control" not in params["system"][0]


def test_dry_run_when_retrieval_abstains(capsys: pytest.CaptureFixture[str], db: Path) -> None:
    out = run(capsys, "ask", "What is it?", "--db", str(db), "--dry-run")[1]
    assert "no model call would be made" in out


class _StubMessages:
    def __init__(self, message: object) -> None:
        self.message = message

    def create(self, **kwargs: Any) -> object:
        return self.message


class _StubSDK:
    def __init__(self, message: object) -> None:
        self.messages = _StubMessages(message)
        self.beta = self


def test_live_mode_uses_the_sdk_client(
    capsys: pytest.CaptureFixture[str], db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_input = {
        "sentences": [
            {
                "text": "Up to 5 unused PTO days carry over into the next calendar year.",
                "citations": ["people/time-off.md#carryover"],
            }
        ],
        "insufficient_evidence": False,
    }
    message = beta_message(
        model="claude-opus-4-8",
        content=[{"type": "tool_use", "id": "t1", "name": "submit_answer", "input": tool_input}],
        usage={
            "input_tokens": 70,
            "output_tokens": 25,
            "iterations": [
                {
                    "type": "message",
                    "model": None,
                    "input_tokens": 60,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                {
                    "type": "fallback_message",
                    "model": "claude-opus-4-8",
                    "input_tokens": 70,
                    "output_tokens": 25,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            ],
        },
    )

    def from_environment(**kwargs: Any) -> AnthropicClient:
        return AnthropicClient(cast(anthropic.Anthropic, _StubSDK(message)))

    monkeypatch.setattr(AnthropicClient, "from_environment", staticmethod(from_environment))
    prices = Path(HANDBOOK.parent / "prices.example.toml")
    code, out, _ = run(
        capsys, "ask", CARRYOVER_Q, "--db", str(db), "--live", "--prices", str(prices)
    )
    assert code == 0
    assert "Mode: live (claude-opus-5-5)" in out
    assert "Model calls: 1 (claude-opus-4-8)" in out
    assert "Note: a refusal fallback model served part of this answer." in out
    assert "Cost: unknown (no price for: claude-opus-4-8)" in out


def test_eval_text_details_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    common = [
        "--corpus",
        str(HANDBOOK),
        "--questions",
        str(QUESTIONS),
        "--fixtures",
        str(FIXTURES),
    ]
    code, out, _ = run(capsys, "eval", *common, "--details")
    assert code == 0
    assert out.startswith("citecheck eval: 36 questions (28 answerable, 8 unanswerable)")
    assert "q28   yes        abstained" in out
    data = json.loads(run(capsys, "eval", *common, "--json")[1])
    assert data["summary"]["abstention_correct"] == 34


def test_serve_starts_uvicorn_with_the_app(
    capsys: pytest.CaptureFixture[str], db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    calls: list[dict[str, Any]] = []

    def fake_run(app: object, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    code, _, err = run(
        capsys, "serve", "--db", str(db), "--fixtures", str(FIXTURES), "--port", "8123"
    )
    assert code == 0
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 8123
    assert "fixture replay" in err


def test_serve_without_the_api_extra(
    capsys: pytest.CaptureFixture[str], db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    code, _, err = run(capsys, "serve", "--db", str(db))
    assert code == 1
    assert "pip install 'citecheck[api]'" in err


def test_python_dash_m_entry_point(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["citecheck", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("citecheck", run_name="__main__")
    assert exit_info.value.code == 0
    assert "citecheck 0.1.0" in capsys.readouterr().out


def test_default_paths_are_relative_to_the_repository_root() -> None:
    assert Path("examples/handbook") == cli.DEFAULT_CORPUS
    assert Path("eval/fixtures") == cli.DEFAULT_FIXTURES


def test_format_answer_shows_uncited_sentences_and_a_skipped_judge(db: Path) -> None:
    tool_input = {
        "sentences": [{"text": "Carryover is generous.", "citations": []}],
        "insufficient_evidence": False,
    }
    answer = run_pipeline(
        CARRYOVER_Q,
        index=Index(db),
        client=ScriptedClient([tool_response("submit_answer", tool_input)]),
        settings=Settings(judge=True),
    )
    text = cli.format_answer(answer)
    assert (
        "  1. Carryover is generous.\n     -> unsupported: the sentence has no citations" in text
    )
    assert "judge: skipped (no sentence has a valid citation to judge)" in text
    assert "Mode: live (claude-opus-5-5)" in text


def test_eval_summary_only(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--corpus", str(HANDBOOK), "--questions", str(QUESTIONS), "--fixtures", str(FIXTURES)]
    code, out, _ = run(capsys, "eval", *args)
    assert code == 0
    assert "labels (S/W/U)" not in out
    assert "  flagged 5 of 7 sentences labelled unsupported (missed 2)" in out
