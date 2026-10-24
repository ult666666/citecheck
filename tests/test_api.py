from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from citecheck.api import create_app, status_for
from citecheck.errors import (
    CiteCheckError,
    ConfigError,
    CorpusError,
    FixtureNotFoundError,
    IndexFormatError,
    IndexNotFoundError,
    ModelAPIError,
    ModelResponseError,
    QueryError,
)
from citecheck.fixtures import FixtureClient
from citecheck.index import build_index
from tests.helpers import FIXTURES, HANDBOOK, ScriptedClient, text_response

CARRYOVER_Q = "How many unused PTO days carry over into the next year?"


@pytest.fixture(scope="module")
def index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("api") / "index.sqlite"
    build_index(HANDBOOK, path)
    return path


@pytest.fixture(scope="module")
def client(index_path: Path) -> TestClient:
    app = create_app(index_path=index_path, client=FixtureClient.from_directory(FIXTURES))
    return TestClient(app)


def test_home_page_is_an_accessible_form(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert '<html lang="en">' in html
    assert '<label for="question">' in html
    assert 'aria-live="polite"' in html
    assert "innerHTML" not in html  # server values are inserted as text only


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {
        "status": "ok",
        "documents": 25,
        "chunks": 103,
        "mode": "fixture",
    }


def test_search(client: TestClient) -> None:
    data = client.post("/api/search", json={"question": "hotel limit", "k": 2}).json()
    assert data["query"] == '"hotel" OR "limit"'
    assert len(data["offered"]) + len(data["below_threshold"]) <= 2
    assert data["offered"][0]["chunk_id"] == "workplace/travel.md#hotels"


def test_ask(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": CARRYOVER_Q})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert data["source"] == "fixture"
    assert [s["label"] for s in data["sentences"]] == ["supported", "supported"]


def test_ask_with_judge_and_k(client: TestClient) -> None:
    question = "Do I need a doctor's note when I take a sick day?"
    data = client.post("/api/ask", json={"question": question, "judge": True, "k": 3}).json()
    assert data["judge"]["verdicts"]["1"]["verdict"] == "not_supported"
    assert data["settings"]["k"] == 3
    assert len(data["passages"]) <= 3


def test_unknown_questions_in_fixture_mode_are_404(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "How do I expense a conference hotel?"})
    assert response.status_code == 404
    assert response.json()["error"] == "FixtureNotFoundError"


@pytest.mark.parametrize(
    "body",
    [{"question": ""}, {"question": "x", "k": 0}, {"question": "x", "unexpected": 1}, {}],
)
def test_invalid_bodies_are_422(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/api/ask", json=body).status_code == 422


def test_blank_questions_are_422(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["error"] == "QueryError"


def test_unusable_model_replies_are_502(index_path: Path) -> None:
    app = create_app(
        index_path=index_path, client=ScriptedClient([text_response(), text_response()])
    )
    response = TestClient(app).post("/api/ask", json={"question": CARRYOVER_Q})
    assert response.status_code == 502
    assert response.json()["error"] == "ModelResponseError"


def test_a_missing_index_fails_at_startup(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError):
        create_app(index_path=tmp_path / "missing.sqlite", client=ScriptedClient([]))


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (QueryError("x"), 422),
        (ConfigError("x"), 422),
        (FixtureNotFoundError("x"), 404),
        (IndexNotFoundError("x"), 503),
        (IndexFormatError("x"), 503),
        (ModelAPIError("x", retryable=True), 503),
        (ModelAPIError("x", retryable=False), 502),
        (ModelResponseError("x"), 502),
        (CorpusError("x"), 500),
        (CiteCheckError("x"), 500),
    ],
)
def test_status_codes(error: CiteCheckError, status: int) -> None:
    assert status_for(error) == status
