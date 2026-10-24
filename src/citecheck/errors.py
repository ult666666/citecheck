"""Typed exceptions.

Every error that citecheck raises on purpose derives from :class:`CiteCheckError`,
so a caller (the CLI, the HTTP API, a test) can catch one base class and still
tell failure kinds apart. Each subclass points at a different fix: a bad input
file, a missing index, a malformed model reply, a failed API call, and so on.
"""


class CiteCheckError(Exception):
    """Base class for every error citecheck raises deliberately."""


class ConfigError(CiteCheckError, ValueError):
    """A setting is out of range or inconsistent with another setting."""


class CorpusError(CiteCheckError):
    """The document folder is missing, empty, or holds a file that cannot be read."""


class IndexNotFoundError(CiteCheckError):
    """The search index does not exist yet (run ``citecheck index DIR`` first)."""


class IndexFormatError(CiteCheckError):
    """The file is not a citecheck index, or was built by an incompatible version."""


class QueryError(CiteCheckError, ValueError):
    """The question is empty or too long to search for."""


class ModelResponseError(CiteCheckError):
    """The model's reply could not be turned into a cited answer.

    Raised when the reply has no tool call even after a retry, when the tool
    input does not match the schema, or when the reply was cut off at
    ``max_tokens``.
    """


class ModelAPIError(CiteCheckError):
    """The Anthropic API call failed (authentication, rate limit, network, server).

    Attributes:
        retryable: Whether trying again later could succeed (rate limits,
            overload, network errors) as opposed to a request that will fail
            the same way every time (bad key, unknown model, invalid request).
        status_code: The HTTP status code, when the API returned one.
    """

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        """Store the message together with retry guidance."""
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class FixtureError(CiteCheckError):
    """A fixture file (a hand-written synthetic model response) is malformed."""


class FixtureNotFoundError(FixtureError):
    """No fixture matches the request; fixture mode only replays prepared questions."""


class PricingError(CiteCheckError):
    """The price table file is missing or malformed."""


class EvalDataError(CiteCheckError):
    """The evaluation question set is malformed or names chunk IDs that do not exist."""
