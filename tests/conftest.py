"""Shared test helpers.

No test calls the Anthropic API or reads an API key: the autouse fixture below
replaces ``anthropic.Anthropic`` with a constructor that fails the test, so a
live client cannot even be created by accident. Tests that exercise the live
adapter inject a stand-in SDK object instead.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import pytest

CorpusFactory = Callable[[dict[str, str]], Path]


@pytest.fixture(autouse=True)
def forbid_live_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tests must never construct a real anthropic.Anthropic client")

    monkeypatch.setattr(anthropic, "Anthropic", forbidden)


@pytest.fixture
def make_corpus(tmp_path: Path) -> CorpusFactory:
    """Return a function that writes ``{relative path: text}`` files to a new folder."""

    def make(files: dict[str, str]) -> Path:
        root = tmp_path / "corpus"
        root.mkdir(exist_ok=True)
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return root

    return make
