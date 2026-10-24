"""Settings shared by the pipeline, the CLI, the HTTP API and the evaluation.

All knobs live in one frozen dataclass that validates itself on construction,
so an impossible combination (say, a "weak" threshold above the "supported"
threshold) fails immediately with a :class:`~citecheck.errors.ConfigError`
instead of producing confusing labels later.
"""

import math
from dataclasses import dataclass
from typing import Final, Literal, get_args

from citecheck.errors import ConfigError

Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: Final[tuple[str, ...]] = get_args(Effort)

#: The model used in live mode unless ``--model`` says otherwise.
DEFAULT_MODEL: Final = "claude-opus-5-5"
DEFAULT_K: Final = 5
#: Minimum BM25 score (higher is better) for a chunk to be offered to the model.
#: Chosen by inspecting score distributions on the bundled demo corpus (see
#: docs/DESIGN.md); BM25 scores depend on corpus statistics, so recalibrate it
#: for your own documents with ``citecheck search``.
DEFAULT_MIN_SCORE: Final = 3.0
DEFAULT_SUPPORTED_THRESHOLD: Final = 0.8
DEFAULT_WEAK_THRESHOLD: Final = 0.5
#: Generous on purpose: thinking tokens count toward max_tokens, and a reply
#: cut off mid tool call is an error, while unused headroom costs nothing.
DEFAULT_MAX_TOKENS: Final = 16_000
#: Set explicitly because Claude Opus 5.5 defaults to "medium" and other models
#: to "high"; pass effort=None to omit the parameter for models without it.
DEFAULT_EFFORT: Final[Effort] = "medium"
MAX_K: Final = 50
MAX_TOKENS_LIMIT: Final = 128_000


@dataclass(frozen=True)
class Settings:
    """Every tunable parameter of a citecheck run.

    Attributes:
        k: How many chunks to retrieve per question.
        min_score: Minimum BM25 score for a chunk to be offered to the model.
        supported_threshold: Minimum support score for the "supported" label.
        weak_threshold: Minimum support score for "weakly_supported"; below it
            a sentence is "unsupported".
        model: Claude model ID used in live mode.
        max_tokens: Output token cap per Messages API call.
        effort: ``output_config.effort`` sent to the API, or None to omit it.
        fallback: Request server-side refusal fallback (``fallbacks: "default"``).
        cache_system_prompt: Put a prompt-caching breakpoint after the stable
            system prompt and tool definition.
        judge: Run the optional LLM-judge pass after the lexical check.
    """

    k: int = DEFAULT_K
    min_score: float = DEFAULT_MIN_SCORE
    supported_threshold: float = DEFAULT_SUPPORTED_THRESHOLD
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: Effort | None = DEFAULT_EFFORT
    fallback: bool = True
    cache_system_prompt: bool = True
    judge: bool = False

    def __post_init__(self) -> None:
        """Reject out-of-range or inconsistent values."""
        if not 1 <= self.k <= MAX_K:
            raise ConfigError(f"k must be between 1 and {MAX_K}, got {self.k}")
        if not math.isfinite(self.min_score) or self.min_score < 0:
            raise ConfigError(f"min_score must be a non-negative number, got {self.min_score}")
        for name in ("supported_threshold", "weak_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} must be between 0 and 1, got {value}")
        if self.weak_threshold > self.supported_threshold:
            raise ConfigError(
                f"weak_threshold ({self.weak_threshold}) cannot exceed "
                f"supported_threshold ({self.supported_threshold})"
            )
        if not self.model.strip():
            raise ConfigError("model must be a non-empty model ID")
        if not 1 <= self.max_tokens <= MAX_TOKENS_LIMIT:
            raise ConfigError(
                f"max_tokens must be between 1 and {MAX_TOKENS_LIMIT}, got {self.max_tokens}"
            )
        if self.effort is not None and self.effort not in EFFORT_LEVELS:
            raise ConfigError(
                f"effort must be one of {', '.join(EFFORT_LEVELS)} or None, got {self.effort!r}"
            )
