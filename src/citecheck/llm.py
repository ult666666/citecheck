"""The boundary between citecheck and a language model.

The pipeline talks to a model only through the small :class:`ModelClient`
protocol: one method, ``complete(ModelRequest) -> ModelResponse``. There are two
implementations:

* :class:`AnthropicClient` calls the Anthropic Messages API through the official
  ``anthropic`` SDK (live mode; it needs your own credentials).
* :class:`citecheck.fixtures.FixtureClient` replays hand-written synthetic
  responses from JSON files. It is the default and never touches the network.

A :class:`ModelRequest` is a plain dataclass whose :meth:`~ModelRequest.to_api_params`
method renders the exact Messages API parameters. The same request can therefore
be printed (``citecheck ask --dry-run``), replayed from a fixture, or sent live,
and tests can assert on it without a network connection.

Live requests go through ``client.beta.messages.create`` because the server-side
refusal fallback parameter (``fallbacks: "default"``) lives in the SDK's beta
namespace. With ``fallback=False`` no beta header is sent and the call is an
ordinary Messages API request.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, Protocol

import anthropic
from anthropic.types.beta import (
    BetaFallbackMessageIterationUsage,
    BetaMessage,
    BetaMessageIterationUsage,
    BetaTextBlock,
    BetaToolUseBlock,
)

from citecheck.config import DEFAULT_EFFORT, Effort
from citecheck.errors import ModelAPIError, ModelResponseError

#: Beta header for the ``fallbacks: "default"`` form of server-side fallback.
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
#: One normal attempt plus one retry with a reminder when no tool call came back.
MAX_TOOL_ATTEMPTS: Final = 2

ClientSource = Literal["live", "fixture"]
Purpose = Literal["answer", "judge"]


@dataclass(frozen=True)
class ModelRequest:
    """One Messages API call that asks the model to answer by calling one tool.

    Attributes:
        purpose: "answer" for the cited answer, "judge" for the optional judge.
        replay_key: The question text. The fixture client uses it to find the
            prepared response; the live client ignores it.
        model: Claude model ID.
        max_tokens: Output token cap.
        system: The (constant, cacheable) system prompt.
        tool: The tool definition (name, description, strict JSON schema).
        user_message: The per-question message: passages plus the question.
        effort: ``output_config.effort``, or None to leave it out.
        cache_system_prompt: Mark the end of the system prompt as a prompt
            caching breakpoint, so ``tools`` + ``system`` can be cached.
        fallback: Ask for server-side refusal fallback.
    """

    purpose: Purpose
    replay_key: str
    model: str
    max_tokens: int
    system: str
    tool: Mapping[str, Any]
    user_message: str
    effort: Effort | None = DEFAULT_EFFORT
    cache_system_prompt: bool = True
    fallback: bool = True

    def to_api_params(self) -> dict[str, Any]:
        """Return the keyword arguments for ``client.beta.messages.create``.

        Notes on the choices made here:

        * ``tool_choice`` is ``auto`` because Claude Opus 5.5 and Claude Sonnet
          5.5 reject forced tool choice with HTTP 400. ``strict: true`` on the
          tool keeps the arguments schema-valid, the system prompt asks for the
          tool, and :func:`call_tool` retries once if it is not called.
          ``disable_parallel_tool_use`` means at most one tool call per reply.
        * No ``temperature``, ``top_p`` or ``top_k``: current models reject them.
        * No ``thinking`` parameter: current models think adaptively by default,
          and ``effort`` controls how much.
        """
        system_block: dict[str, Any] = {"type": "text", "text": self.system}
        if self.cache_system_prompt:
            system_block["cache_control"] = {"type": "ephemeral"}
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [system_block],
            "tools": [dict(self.tool)],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "messages": [{"role": "user", "content": self.user_message}],
        }
        if self.effort is not None:
            params["output_config"] = {"effort": self.effort}
        if self.fallback:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def with_reminder(self, reminder: str) -> "ModelRequest":
        """Return a copy with ``reminder`` appended to the user message."""
        return replace(self, user_message=self.user_message + reminder)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one model attempt, as reported in the API's ``usage``.

    ``input_tokens`` counts only uncached input; the full prompt size is
    ``input_tokens + cache_creation_input_tokens + cache_read_input_tokens``.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True)
class ToolCall:
    """A tool call the model made: the tool's name and its JSON arguments."""

    name: str
    input: Mapping[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    """The parts of a Messages API response that citecheck uses.

    Attributes:
        model: The model that produced the message (a fallback model, if one
            served the request).
        stop_reason: Why generation stopped ("tool_use", "end_turn",
            "max_tokens", "refusal", ...).
        tool_calls: Tool calls in the reply, in order.
        text: Any plain text in the reply, concatenated.
        usage: One entry per billed attempt. Usually one; more when a refusal
            fallback ran, because the API reports each attempt separately.
        usage_estimated: True when the token counts are a heuristic estimate
            (the fixture client) rather than numbers reported by the API.
        request_id: The API's request ID, for support tickets.
        fallback_used: Whether a fallback model served the response.
        refusal_category: The refusal category when ``stop_reason`` is "refusal".
    """

    model: str
    stop_reason: str | None
    tool_calls: tuple[ToolCall, ...]
    text: str
    usage: tuple[TokenUsage, ...]
    usage_estimated: bool = False
    request_id: str | None = None
    fallback_used: bool = False
    refusal_category: str | None = None


class ModelClient(Protocol):
    """Anything that can complete a :class:`ModelRequest`."""

    @property
    def source(self) -> ClientSource:
        """Where responses come from: "live" (the API) or "fixture" (replay)."""
        ...

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request and return the parsed response."""
        ...


def response_from_message(message: BetaMessage, *, requested_model: str) -> ModelResponse:
    """Convert an SDK ``BetaMessage`` into a :class:`ModelResponse`.

    Thinking blocks and fallback marker blocks are ignored: only tool calls and
    text matter here. Usage comes from ``usage.iterations`` when present (one
    entry per attempt, so a declined attempt and the fallback attempt that
    served the answer are billed separately, each by the model that ran it; an
    entry without a model is the requested model's). Otherwise usage comes from
    the top-level fields, which describe the attempt that produced the message.
    """
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    for block in message.content:
        if isinstance(block, BetaToolUseBlock):
            tool_calls.append(ToolCall(name=block.name, input=dict(block.input)))
        elif isinstance(block, BetaTextBlock):
            text_parts.append(block.text)

    usage = message.usage
    entries: list[TokenUsage] = []
    fallback_used = False
    for item in usage.iterations or []:
        if isinstance(item, BetaFallbackMessageIterationUsage):
            fallback_used = True
        if isinstance(item, BetaMessageIterationUsage | BetaFallbackMessageIterationUsage):
            entries.append(
                TokenUsage(
                    model=item.model or requested_model,
                    input_tokens=item.input_tokens,
                    output_tokens=item.output_tokens,
                    cache_creation_input_tokens=item.cache_creation_input_tokens,
                    cache_read_input_tokens=item.cache_read_input_tokens,
                )
            )
    if not entries:
        entries.append(
            TokenUsage(
                model=message.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            )
        )
    return ModelResponse(
        model=message.model,
        stop_reason=message.stop_reason,
        tool_calls=tuple(tool_calls),
        text="".join(text_parts),
        usage=tuple(entries),
        request_id=getattr(message, "_request_id", None),
        fallback_used=fallback_used,
        refusal_category=message.stop_details.category if message.stop_details else None,
    )


class AnthropicClient:
    """Live client: sends requests to the Anthropic Messages API.

    The SDK client is injected so tests can pass a stand-in object and never
    reach the network. :meth:`from_environment` builds the real one.
    """

    def __init__(self, sdk_client: anthropic.Anthropic) -> None:
        """Wrap an SDK client (``anthropic.Anthropic`` or a test double)."""
        self._sdk = sdk_client

    @classmethod
    def from_environment(
        cls, *, timeout: float = 120.0, max_retries: int = 2
    ) -> "AnthropicClient":
        """Build a client that finds credentials the way the SDK does.

        The SDK looks for ``ANTHROPIC_API_KEY``, then ``ANTHROPIC_AUTH_TOKEN``,
        then a profile saved by ``ant auth login``. citecheck never reads,
        stores or prints the key itself. The SDK already retries rate limits,
        overload errors and network failures ``max_retries`` times with
        exponential backoff.
        """
        return cls(anthropic.Anthropic(timeout=timeout, max_retries=max_retries))

    @property
    def source(self) -> ClientSource:
        """Always "live"."""
        return "live"

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send the request and convert SDK errors into :class:`ModelAPIError`.

        Errors are caught from most to least specific, so the caller learns
        whether retrying later could help (``ModelAPIError.retryable``).
        """
        try:
            message = self._sdk.beta.messages.create(**request.to_api_params())
        except anthropic.AuthenticationError as exc:
            raise ModelAPIError(
                "authentication failed: check ANTHROPIC_API_KEY or run `ant auth login`",
                retryable=False,
                status_code=exc.status_code,
            ) from exc
        except anthropic.NotFoundError as exc:
            raise ModelAPIError(
                f"model {request.model!r} was not found or is not available to your account",
                retryable=False,
                status_code=exc.status_code,
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ModelAPIError(
                "rate limited by the API (HTTP 429) even after automatic retries; "
                "wait and try again",
                retryable=True,
                status_code=exc.status_code,
            ) from exc
        except anthropic.BadRequestError as exc:
            raise ModelAPIError(
                f"the API rejected the request: {exc.message}",
                retryable=False,
                status_code=exc.status_code,
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ModelAPIError(
                f"API error (HTTP {exc.status_code}): {exc.message}",
                retryable=exc.status_code >= 500,
                status_code=exc.status_code,
            ) from exc
        except anthropic.APIConnectionError as exc:  # also covers APITimeoutError
            raise ModelAPIError(
                f"could not reach the Anthropic API: {exc}", retryable=True
            ) from exc
        except anthropic.AnthropicError as exc:  # e.g. CredentialsError
            raise ModelAPIError(f"Anthropic client error: {exc}", retryable=False) from exc
        except TypeError as exc:
            # With no credentials at all, the SDK raises a bare TypeError at
            # request time ("Could not resolve authentication method"); there is
            # no typed exception to catch, so this one is matched by message.
            if "authentication" not in str(exc):
                raise
            raise ModelAPIError(
                "no Anthropic credentials found: set ANTHROPIC_API_KEY or run `ant auth login`",
                retryable=False,
            ) from exc
        return response_from_message(message, requested_model=request.model)


@dataclass(frozen=True)
class ToolOutcome:
    """The result of asking the model for one tool call.

    Attributes:
        tool_input: The tool's arguments, or None if the model refused.
        responses: Every attempt made, so all of them can be billed.
        refusal_category: Set when the model refused.
    """

    tool_input: Mapping[str, Any] | None
    responses: tuple[ModelResponse, ...]
    refusal_category: str | None = None

    @property
    def refused(self) -> bool:
        """True if the model declined to respond."""
        return self.tool_input is None


def call_tool(
    client: ModelClient,
    request: ModelRequest,
    *,
    tool_name: str,
    retry_reminder: str,
    max_attempts: int = MAX_TOOL_ATTEMPTS,
) -> ToolOutcome:
    """Send ``request`` and return the arguments of the ``tool_name`` call.

    ``tool_choice: auto`` does not guarantee a tool call, so a reply without
    one is retried once as a fresh request with ``retry_reminder`` appended
    (the conversation is not continued, which keeps every request
    self-contained). A refusal is returned rather than raised: the pipeline
    turns it into an abstention.

    Raises:
        ModelResponseError: If the reply hit ``max_tokens`` (a tool call cut off
            mid-way must never be used) or no attempt produced the tool call.
    """
    responses: list[ModelResponse] = []
    current = request
    for _ in range(max_attempts):
        response = client.complete(current)
        responses.append(response)
        if response.stop_reason == "refusal":
            return ToolOutcome(None, tuple(responses), response.refusal_category)
        if response.stop_reason == "max_tokens":
            raise ModelResponseError(
                f"the model's reply was cut off at max_tokens={request.max_tokens}; "
                "raise --max-tokens and try again"
            )
        for call in response.tool_calls:
            if call.name == tool_name:
                return ToolOutcome(call.input, tuple(responses))
        current = request.with_reminder(retry_reminder)
    raise ModelResponseError(
        f"the model did not call the {tool_name} tool in {max_attempts} attempts"
    )
