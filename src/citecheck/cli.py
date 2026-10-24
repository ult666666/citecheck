"""Command-line interface: ``citecheck index | search | ask | eval | serve``.

Model responses come from the bundled fixture files unless ``--live`` is
given, so nothing here calls the Anthropic API (or needs a key) by accident.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from citecheck import __version__
from citecheck.answer import build_answer_request
from citecheck.chunking import DEFAULT_MAX_CHUNK_CHARS
from citecheck.config import (
    DEFAULT_EFFORT,
    DEFAULT_K,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_SCORE,
    DEFAULT_MODEL,
    DEFAULT_SUPPORTED_THRESHOLD,
    DEFAULT_WEAK_THRESHOLD,
    EFFORT_LEVELS,
    Effort,
    Settings,
)
from citecheck.costs import PriceTable, load_price_table
from citecheck.errors import CiteCheckError
from citecheck.evaluation import evaluate, format_details, format_report
from citecheck.fixtures import FIXTURE_NOTICE, FixtureClient
from citecheck.index import DEFAULT_INDEX_PATH, Index, RetrievedChunk, build_index
from citecheck.llm import AnthropicClient, ModelClient
from citecheck.pipeline import CheckedAnswer, ask
from citecheck.retrieval import retrieve, validate_question

DEFAULT_CORPUS: Final = Path("examples") / "handbook"
DEFAULT_QUESTIONS: Final = Path("eval") / "questions.json"
DEFAULT_FIXTURES: Final = Path("eval") / "fixtures"


def _add_index_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"index file (default: {DEFAULT_INDEX_PATH})",
    )


def _add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("retrieval")
    group.add_argument(
        "-k", type=int, default=DEFAULT_K, help=f"chunks to retrieve (default: {DEFAULT_K})"
    )
    group.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="minimum BM25 score for a chunk to be offered to the model "
        f"(default: {DEFAULT_MIN_SCORE:g})",
    )


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument(
        "--live",
        action="store_true",
        help="call the Anthropic API (needs ANTHROPIC_API_KEY or `ant auth login`); "
        "without it, responses are replayed from fixture files",
    )
    group.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES,
        help=f"fixture folder for offline replay (default: {DEFAULT_FIXTURES})",
    )
    group.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"model ID for --live (default: {DEFAULT_MODEL})"
    )
    group.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"output token cap per call (default: {DEFAULT_MAX_TOKENS})",
    )
    group.add_argument(
        "--effort",
        choices=[*EFFORT_LEVELS, "none"],
        default=DEFAULT_EFFORT,
        help=f"output_config.effort, or 'none' to omit it (default: {DEFAULT_EFFORT})",
    )
    group.add_argument(
        "--no-fallback",
        action="store_true",
        help='do not request server-side refusal fallback (fallbacks: "default")',
    )
    group.add_argument(
        "--no-cache",
        action="store_true",
        help="do not mark the system prompt for prompt caching",
    )
    group.add_argument(
        "--prices",
        type=Path,
        help="TOML price table for a cost estimate (see examples/prices.example.toml)",
    )


def _add_verification_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("verification")
    group.add_argument(
        "--supported-threshold",
        type=float,
        default=DEFAULT_SUPPORTED_THRESHOLD,
        help=f"minimum support for 'supported' (default: {DEFAULT_SUPPORTED_THRESHOLD})",
    )
    group.add_argument(
        "--weak-threshold",
        type=float,
        default=DEFAULT_WEAK_THRESHOLD,
        help=f"minimum support for 'weakly_supported' (default: {DEFAULT_WEAK_THRESHOLD})",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="citecheck",
        description="Cited question answering over your documents, with a checker that flags "
        "sentences their sources do not support.",
    )
    parser.add_argument("--version", action="version", version=f"citecheck {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    index = commands.add_parser("index", help="chunk a folder of .md/.txt files and index it")
    index.add_argument("directory", type=Path, help="folder of Markdown or text files")
    _add_index_option(index)
    index.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help=f"maximum chunk length in characters (default: {DEFAULT_MAX_CHUNK_CHARS})",
    )

    search = commands.add_parser("search", help="show the top-k chunks and scores for a query")
    search.add_argument("query", help="search text")
    _add_index_option(search)
    _add_retrieval_options(search)
    search.add_argument("--json", action="store_true", help="print JSON")

    ask_cmd = commands.add_parser("ask", help="answer a question with checked citations")
    ask_cmd.add_argument("question", help="the question to answer")
    _add_index_option(ask_cmd)
    _add_retrieval_options(ask_cmd)
    _add_verification_options(ask_cmd)
    _add_model_options(ask_cmd)
    ask_cmd.add_argument(
        "--judge", action="store_true", help="also ask a model to judge each sentence"
    )
    ask_cmd.add_argument("--json", action="store_true", help="print the full result as JSON")
    ask_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="print the Messages API request that would be sent, then exit without calling "
        "any model",
    )

    eval_cmd = commands.add_parser("eval", help="score the pipeline on a labeled question set")
    eval_cmd.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"document folder (default: {DEFAULT_CORPUS})",
    )
    eval_cmd.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=f"labeled questions (default: {DEFAULT_QUESTIONS})",
    )
    _add_retrieval_options(eval_cmd)
    _add_verification_options(eval_cmd)
    _add_model_options(eval_cmd)
    eval_cmd.add_argument("--details", action="store_true", help="add a per-question table")
    eval_cmd.add_argument("--json", action="store_true", help="print the full report as JSON")

    serve = commands.add_parser("serve", help="run the JSON API and HTML page (needs [api])")
    _add_index_option(serve)
    _add_retrieval_options(serve)
    _add_verification_options(serve)
    _add_model_options(serve)
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    effort: Effort | None = None if args.effort == "none" else args.effort
    return Settings(
        k=args.k,
        min_score=args.min_score,
        supported_threshold=args.supported_threshold,
        weak_threshold=args.weak_threshold,
        model=args.model,
        max_tokens=args.max_tokens,
        effort=effort,
        fallback=not args.no_fallback,
        cache_system_prompt=not args.no_cache,
        judge=getattr(args, "judge", False),
    )


def _client(args: argparse.Namespace) -> ModelClient:
    if args.live:
        return AnthropicClient.from_environment()
    return FixtureClient.from_directory(args.fixtures)


def _prices(args: argparse.Namespace) -> PriceTable | None:
    return load_price_table(args.prices) if args.prices else None


def _chunk_lines(chunks: Sequence[RetrievedChunk], suffixes: Sequence[str] = ()) -> list[str]:
    """One aligned line per chunk: ID, score and source line span."""
    width = max((len(chunk.chunk_id) for chunk in chunks), default=0)
    lines = [
        f"  {chunk.chunk_id:<{width}}  score {chunk.score:6.2f}  "
        f"lines {chunk.start_line}-{chunk.end_line}"
        for chunk in chunks
    ]
    return (
        [line + suffix for line, suffix in zip(lines, suffixes, strict=True)]
        if suffixes
        else lines
    )


def format_answer(answer: CheckedAnswer) -> str:
    """Render a checked answer for the terminal (plain ASCII)."""
    settings = answer.settings
    mode = FIXTURE_NOTICE if answer.source == "fixture" else f"live ({settings.model})"
    retrieved = len(answer.passages) + len(answer.below_threshold)
    lines = [f"Question: {answer.question}", f"Mode: {mode}", ""]
    lines.append(
        f"Passages offered to the model ({len(answer.passages)} of {retrieved} retrieved, "
        f"k={settings.k}, min score {settings.min_score:g}):"
    )
    lines += _chunk_lines(answer.passages) or ["  (none)"]
    if answer.below_threshold:
        lines.append("Retrieved but below the score threshold (not offered):")
        lines += _chunk_lines(answer.below_threshold)
    lines.append("")
    if answer.status == "abstained":
        lines.append(f"Answer: none (abstained): {answer.abstain_reason}")
    else:
        heading = (
            "Answer:"
            if answer.status == "answered"
            else (
                "Answer (partial: the model flagged that the passages do not answer everything):"
            )
        )
        lines.append(heading)
        verdicts = answer.judge.verdicts if answer.judge else {}
        for check in answer.sentences:
            lines.append(f"  {check.number}. {check.text}")
            if check.citations:
                lines.append(f"     cites: {', '.join(check.citations)}")
            lines.append(f"     -> {check.label}: {check.reason}")
            if check.label != "supported" and check.valid_citations and check.missing_terms:
                lines.append(
                    f"        not found in cited passages: {', '.join(check.missing_terms)}"
                )
            if check.number in verdicts:
                verdict = verdicts[check.number]
                lines.append(f"     judge: {verdict.verdict} ({verdict.reason})")
        if answer.judge and answer.judge.skipped_reason:
            lines.append(f"  judge: skipped ({answer.judge.skipped_reason})")
        counts = answer.label_counts
        lines.append(
            f"Summary: {counts['supported']} supported, {counts['weakly_supported']} weakly "
            f"supported, {counts['unsupported']} unsupported; "
            f"{len(answer.rejected_citations)} rejected citations"
        )
    usage = answer.usage
    if answer.model_calls:
        if answer.source == "fixture":
            calls = f"{answer.model_calls} replayed from fixtures (no API call was made)"
        else:
            calls = (
                f"{answer.model_calls} ({', '.join(sorted({r.model for r in answer.responses}))})"
            )
        note = " (estimated at ~4 characters per token, not measured)" if usage.estimated else ""
        lines.append(f"Model calls: {calls}")
        lines.append(
            f"Tokens: {usage.input_tokens:,} input, {usage.output_tokens:,} output, "
            f"{usage.cache_read_input_tokens:,} cache-read, "
            f"{usage.cache_creation_input_tokens:,} cache-write{note}"
        )
        if answer.fallback_used:
            lines.append("Note: a refusal fallback model served part of this answer.")
    else:
        lines.append("Model calls: 0")
    if answer.cost is not None:
        cost = answer.cost
        if cost.total is None:
            lines.append(f"Cost: unknown (no price for: {', '.join(cost.missing_models)})")
        else:
            caveat = (
                " (from estimated tokens: demonstrates the arithmetic only)"
                if cost.estimated_usage
                else ""
            )
            lines.append(f"Cost: {cost.total:.6f} {cost.currency}{caveat}")
    elif answer.model_calls:
        lines.append("Cost: not computed (pass --prices FILE)")
    t = answer.timings
    lines.append(
        f"Time: retrieval {t.retrieval_ms:.1f} ms, model {t.model_ms:.1f} ms, "
        f"verification {t.verification_ms:.1f} ms, total {t.total_ms:.1f} ms"
    )
    return "\n".join(lines)


def _cmd_index(args: argparse.Namespace) -> int:
    stats = build_index(args.directory, args.db, max_chunk_chars=args.max_chunk_chars)
    print(f"Indexed {stats.documents} documents as {stats.chunks} chunks into {args.db}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    Settings(k=args.k, min_score=args.min_score)  # validate the values
    retrieval = retrieve(Index(args.db), args.query, k=args.k, min_score=args.min_score)
    if args.json:
        data = {
            "query": retrieval.query,
            "offered": [chunk.to_dict() for chunk in retrieval.passages],
            "below_threshold": [chunk.to_dict() for chunk in retrieval.below_threshold],
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    if retrieval.query is None:
        print("No searchable words in the query.")
        return 0
    print(f"FTS5 query: {retrieval.query}")
    if not retrieval.candidates:
        print("No chunks matched.")
    markers = [
        "  [offered]" if chunk.score >= args.min_score else "  [below threshold]"
        for chunk in retrieval.candidates
    ]
    for line in _chunk_lines(retrieval.candidates, markers):
        print(line)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    index = Index(args.db)
    if args.dry_run:
        question = validate_question(args.question)
        retrieval = retrieve(index, question, k=settings.k, min_score=settings.min_score)
        if not retrieval.passages:
            print("No passage reached the score threshold, so no model call would be made.")
            return 0
        request = build_answer_request(question, retrieval.passages, settings)
        print(json.dumps(request.to_api_params(), indent=2, ensure_ascii=False))
        return 0
    answer = ask(
        args.question, index=index, client=_client(args), settings=settings, prices=_prices(args)
    )
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_answer(answer))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    report = evaluate(args.corpus, args.questions, _client(args), _settings(args))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(format_report(report))
    if args.details:
        print()
        print(format_details(report))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from citecheck.api import create_app
    except ImportError as exc:
        raise CiteCheckError(
            "the HTTP API needs optional dependencies: pip install 'citecheck[api]'"
        ) from exc
    app = create_app(
        index_path=args.db, client=_client(args), settings=_settings(args), prices=_prices(args)
    )
    mode = "live" if args.live else "fixture replay"
    print(f"citecheck API ({mode}) on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


_COMMANDS: Final = {
    "index": _cmd_index,
    "search": _cmd_search,
    "ask": _cmd_ask,
    "eval": _cmd_eval,
    "serve": _cmd_serve,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code (0 ok, 1 error, 2 usage error)."""
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except CiteCheckError as exc:
        print(f"citecheck: error: {exc}", file=sys.stderr)
        return 1
