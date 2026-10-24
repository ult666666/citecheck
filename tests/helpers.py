"""Test doubles shared by several test modules. Nothing here touches the network."""

from collections.abc import Iterable, Mapping
from typing import Any

from anthropic.types.beta import BetaMessage

from citecheck.llm import ClientSource, ModelRequest, ModelResponse, TokenUsage, ToolCall

MODEL = "claude-opus-5-5"


def tool_response(
    name: str,
    tool_input: Mapping[str, Any],
    *,
    model: str = MODEL,
    stop_reason: str = "tool_use",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> ModelResponse:
    """A response containing one tool call."""
    return ModelResponse(
        model=model,
        stop_reason=stop_reason,
        tool_calls=(ToolCall(name=name, input=dict(tool_input)),),
        text="",
        usage=(TokenUsage(model=model, input_tokens=input_tokens, output_tokens=output_tokens),),
    )


def text_response(text: str = "Here is my answer in prose.") -> ModelResponse:
    """A response that ignores the tool and answers in plain text."""
    return ModelResponse(
        model=MODEL,
        stop_reason="end_turn",
        tool_calls=(),
        text=text,
        usage=(TokenUsage(model=MODEL, input_tokens=100, output_tokens=10),),
    )


def refusal_response(category: str | None = "cyber") -> ModelResponse:
    """A response where the model declined (stop_reason "refusal")."""
    return ModelResponse(
        model=MODEL,
        stop_reason="refusal",
        tool_calls=(),
        text="",
        usage=(TokenUsage(model=MODEL, input_tokens=100, output_tokens=0),),
        refusal_category=category,
    )


class ScriptedClient:
    """A ModelClient that returns prepared responses in order and records requests."""

    def __init__(self, responses: Iterable[ModelResponse], source: ClientSource = "live") -> None:
        self._responses = list(responses)
        self._source: ClientSource = source
        self.requests: list[ModelRequest] = []

    @property
    def source(self) -> ClientSource:
        return self._source

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected extra model call")
        return self._responses.pop(0)


def beta_message(**overrides: Any) -> BetaMessage:
    """Build a real SDK BetaMessage from a minimal Messages API response body."""
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    body.update(overrides)
    return BetaMessage.model_validate(body)
