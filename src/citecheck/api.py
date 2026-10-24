"""A small FastAPI app: a JSON endpoint and a minimal, accessible HTML page.

Endpoints:

* ``GET /``: the HTML page (a form that calls ``POST /api/ask``).
* ``GET /healthz``: index size and response mode.
* ``POST /api/search``: retrieval only, with scores.
* ``POST /api/ask``: the full pipeline; the body is ``{"question": ..., "k": ..., "judge": ...}``
  and the response is :meth:`citecheck.pipeline.CheckedAnswer.to_dict`.

Typed citecheck errors map to HTTP status codes in one place
(:func:`status_for`), so a fixture miss is a 404, a bad question a 422, and an
upstream API failure a 502 or 503, never an anonymous 500.
"""

from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from citecheck import __version__
from citecheck.config import MAX_K, Settings
from citecheck.costs import PriceTable
from citecheck.errors import (
    CiteCheckError,
    ConfigError,
    FixtureNotFoundError,
    IndexFormatError,
    IndexNotFoundError,
    ModelAPIError,
    ModelResponseError,
    QueryError,
)
from citecheck.index import Index
from citecheck.llm import ModelClient
from citecheck.pipeline import ask
from citecheck.retrieval import MAX_QUESTION_CHARS, retrieve


class SearchBody(BaseModel):
    """Request body for ``POST /api/search``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int | None = Field(default=None, ge=1, le=MAX_K)


class AskBody(SearchBody):
    """Request body for ``POST /api/ask``."""

    judge: bool = False


def status_for(error: CiteCheckError) -> int:
    """Map a citecheck error to an HTTP status code."""
    if isinstance(error, QueryError | ConfigError):
        return 422
    if isinstance(error, FixtureNotFoundError):
        return 404
    if isinstance(error, IndexNotFoundError | IndexFormatError):
        return 503
    if isinstance(error, ModelAPIError):
        return 503 if error.retryable else 502
    if isinstance(error, ModelResponseError):
        return 502
    return 500


def _page() -> str:
    return files("citecheck").joinpath("static/index.html").read_text(encoding="utf-8")


def create_app(
    *,
    index_path: Path,
    client: ModelClient,
    settings: Settings | None = None,
    prices: PriceTable | None = None,
) -> FastAPI:
    """Build the app around an existing index and a model client.

    The index is opened here, so a missing index fails at startup rather than
    on the first request.
    """
    index = Index(index_path)
    base_settings = settings or Settings()
    page = _page()
    app = FastAPI(
        title="citecheck",
        version=__version__,
        summary="Cited question answering with a sentence-level support checker.",
    )

    @app.exception_handler(CiteCheckError)
    async def handle_citecheck_error(request: Request, exc: CiteCheckError) -> JSONResponse:
        return JSONResponse(
            status_code=status_for(exc),
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return page

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        stats = index.stats
        return {
            "status": "ok",
            "documents": stats.documents,
            "chunks": stats.chunks,
            "mode": client.source,
        }

    def settings_for(body: SearchBody) -> Settings:
        overrides: dict[str, Any] = {}
        if body.k is not None:
            overrides["k"] = body.k
        if isinstance(body, AskBody) and body.judge:
            overrides["judge"] = True
        return replace(base_settings, **overrides) if overrides else base_settings

    @app.post("/api/search")
    def search(body: SearchBody) -> dict[str, Any]:
        settings_used = settings_for(body)
        retrieval = retrieve(
            index, body.question, k=settings_used.k, min_score=settings_used.min_score
        )
        return {
            "query": retrieval.query,
            "offered": [chunk.to_dict() for chunk in retrieval.passages],
            "below_threshold": [chunk.to_dict() for chunk in retrieval.below_threshold],
        }

    @app.post("/api/ask")
    def ask_endpoint(body: AskBody) -> dict[str, Any]:
        answer = ask(
            body.question,
            index=index,
            client=client,
            settings=settings_for(body),
            prices=prices,
        )
        return answer.to_dict()

    return app
