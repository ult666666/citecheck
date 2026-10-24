from typing import Any

import pytest

from citecheck.config import DEFAULT_MODEL, Settings
from citecheck.errors import ConfigError


def test_defaults_are_valid() -> None:
    settings = Settings()
    assert settings.model == DEFAULT_MODEL
    assert settings.weak_threshold <= settings.supported_threshold
    assert settings.effort == "medium"


def test_effort_can_be_omitted() -> None:
    assert Settings(effort=None).effort is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"k": 0}, "k must be"),
        ({"k": 51}, "k must be"),
        ({"min_score": -1.0}, "min_score"),
        ({"min_score": float("nan")}, "min_score"),
        ({"supported_threshold": 1.5}, "supported_threshold"),
        ({"weak_threshold": float("inf")}, "weak_threshold"),
        ({"weak_threshold": 0.9, "supported_threshold": 0.8}, "cannot exceed"),
        ({"model": "  "}, "model"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 200_000}, "max_tokens"),
        ({"effort": "extreme"}, "effort"),
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings(**overrides)
