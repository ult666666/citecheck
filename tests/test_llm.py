import json
from typing import Any, cast

import anthropic
import httpx2
import pytest

from citecheck.answer import build_answer_request
from citecheck.config import Settings
from citecheck.errors import ModelAPIError, ModelResponseError
from citecheck.index import RetrievedChunk
from citecheck.llm import (
    FALLBACK_BETA,
    AnthropicClient,
    ModelRequest,
    TokenUsage,
    call_tool,
    response_from_message,
)
from tests.helpers import (
    MODEL,
    ScriptedClient,
    beta_message,
    refusal_response,
    text_response,
    tool_response,
)

ANSWER_INPUT = {"sentences": [], "insufficient_evidence": True}


def request(**overrides: Any) -> ModelRequest:
    fields: dict[str, Any] = {
        "purpose": "answer",
        "replay_key": "q?",
        "model": MODEL,
        "max_tokens": 1000,
        "system": "SYSTEM",
        "tool": {"name": "submit_answer", "input_schema": {"type": "object"}},
        "user_message": "USER",
    }
    fields.update(overrides)
    return ModelRequest(**fields)


def test_api_params_with_every_option_on() -> None:
    assert request(effort="low").to_api_params() == {
        "model": MODEL,
        "max_tokens": 1000,
        "system": [{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "submit_answer", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": [{"role": "user", "content": "USER"}],
        "output_config": {"effort": "low"},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }


def test_api_params_with_options_off() -> None:
    params = request(effort=None, fallback=False, cache_system_prompt=False).to_api_params()
    assert params["system"] == [{"type": "text", "text": "SYSTEM"}]
    assert "output_config" not in params
    assert "betas" not in params
    assert "fallbacks" not in params
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(params)


def test_with_reminder_appends_to_the_user_message() -> None:
    assert request().with_reminder(" Please.").user_message == "USER Please."


def test_cached_prefix_is_identical_across_questions() -> None:
    def passage(chunk_id: str) -> RetrievedChunk:
        return RetrievedChunk(chunk_id, "a.md", "A", 1, 2, f"text of {chunk_id}", 3.0)

    first = build_answer_request("Question one?", [passage("a.md#1")], Settings()).to_api_params()
    second = build_answer_request("Other?", [passage("a.md#2")], Settings()).to_api_params()
    for key in ("tools", "system", "model", "tool_choice"):
        assert json.dumps(first[key]) == json.dumps(second[key])
    assert first["messages"] != second["messages"]


def test_response_with_text_thinking_and_tool_use() -> None:
    message = beta_message(
        content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Calling the tool."},
            {"type": "tool_use", "id": "toolu_1", "name": "submit_answer", "input": ANSWER_INPUT},
        ],
        usage={
            "input_tokens": 50,
            "output_tokens": 30,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": 900,
        },
    )
    response = response_from_message(message, requested_model=MODEL)
    assert response.text == "Calling the tool."
    assert [(c.name, dict(c.input)) for c in response.tool_calls] == [
        ("submit_answer", ANSWER_INPUT)
    ]
    assert response.usage == (TokenUsage(MODEL, 50, 30, 0, 900),)
    assert response.fallback_used is False
    assert response.refusal_category is None


def test_response_served_by_a_fallback_model_bills_each_attempt() -> None:
    message = beta_message(
        model="claude-opus-4-8",
        content=[
            {
                "type": "fallback",
                "from": {"model": MODEL},
                "to": {"model": "claude-opus-4-8"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "tool_use", "id": "toolu_1", "name": "submit_answer", "input": ANSWER_INPUT},
        ],
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
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 5,
                },
            ],
        },
    )
    response = response_from_message(message, requested_model=MODEL)
    assert response.fallback_used is True
    assert response.model == "claude-opus-4-8"
    assert response.usage == (
        TokenUsage(MODEL, 60, 0, 0, 0),  # the declined attempt, by the requested model
        TokenUsage("claude-opus-4-8", 70, 25, 10, 5),
    )
    assert len(response.tool_calls) == 1


def test_refusal_category_is_reported() -> None:
    message = beta_message(
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    response = response_from_message(message, requested_model=MODEL)
    assert response.stop_reason == "refusal"
    assert response.refusal_category == "cyber"


class StubMessages:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class StubSDK:
    """Stands in for anthropic.Anthropic: only .beta.messages.create is used."""

    def __init__(self, outcome: object) -> None:
        self.messages = StubMessages(outcome)
        self.beta = self


def live_client(outcome: object) -> tuple[AnthropicClient, StubMessages]:
    stub = StubSDK(outcome)
    return AnthropicClient(cast(anthropic.Anthropic, stub)), stub.messages


def test_live_client_sends_exactly_the_rendered_params() -> None:
    message = beta_message(
        content=[{"type": "tool_use", "id": "t", "name": "submit_answer", "input": ANSWER_INPUT}]
    )
    client, messages = live_client(message)
    req = request()
    response = client.complete(req)
    assert messages.calls == [req.to_api_params()]
    assert response.tool_calls[0].name == "submit_answer"
    assert client.source == "live"


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    return cls(f"HTTP {status}", response=httpx2.Response(status, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    ("error", "retryable", "status", "fragment"),
    [
        (_status_error(anthropic.AuthenticationError, 401), False, 401, "authentication failed"),
        (_status_error(anthropic.NotFoundError, 404), False, 404, "was not found"),
        (_status_error(anthropic.RateLimitError, 429), True, 429, "rate limited"),
        (_status_error(anthropic.BadRequestError, 400), False, 400, "rejected the request"),
        (_status_error(anthropic.PermissionDeniedError, 403), False, 403, "HTTP 403"),
        (_status_error(anthropic.InternalServerError, 500), True, 500, "HTTP 500"),
        (anthropic.APIConnectionError(request=_REQUEST), True, None, "could not reach"),
        (anthropic.APITimeoutError(request=_REQUEST), True, None, "could not reach"),
        (anthropic.AnthropicError("no profile"), False, None, "client error"),
        (TypeError("Could not resolve authentication method."), False, None, "no Anthropic"),
    ],
)
def test_sdk_errors_become_typed_errors(
    error: BaseException, retryable: bool, status: int | None, fragment: str
) -> None:
    client, _ = live_client(error)
    with pytest.raises(ModelAPIError, match=fragment) as caught:
        client.complete(request())
    assert caught.value.retryable is retryable
    assert caught.value.status_code == status


def test_unrelated_type_errors_are_not_swallowed() -> None:
    client, _ = live_client(TypeError("unexpected keyword argument"))
    with pytest.raises(TypeError, match="unexpected keyword"):
        client.complete(request())


def test_from_environment_lets_the_sdk_find_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []

    def fake_constructor(**kwargs: Any) -> object:
        created.append(kwargs)
        return StubSDK(None)

    monkeypatch.setattr(anthropic, "Anthropic", fake_constructor)
    client = AnthropicClient.from_environment(timeout=5.0, max_retries=1)
    assert created == [{"timeout": 5.0, "max_retries": 1}]
    assert client.source == "live"


def test_call_tool_retries_once_with_a_reminder() -> None:
    client = ScriptedClient([text_response(), tool_response("submit_answer", ANSWER_INPUT)])
    outcome = call_tool(client, request(), tool_name="submit_answer", retry_reminder=" Use it.")
    assert outcome.tool_input == ANSWER_INPUT
    assert len(outcome.responses) == 2
    assert client.requests[0].user_message == "USER"
    assert client.requests[1].user_message == "USER Use it."


def test_call_tool_picks_the_named_tool() -> None:
    other = tool_response("something_else", {"x": 1})
    client = ScriptedClient([other, tool_response("submit_answer", ANSWER_INPUT)])
    outcome = call_tool(client, request(), tool_name="submit_answer", retry_reminder="!")
    assert outcome.tool_input == ANSWER_INPUT


def test_call_tool_gives_up_after_two_attempts() -> None:
    client = ScriptedClient([text_response(), text_response()])
    with pytest.raises(ModelResponseError, match="did not call the submit_answer tool"):
        call_tool(client, request(), tool_name="submit_answer", retry_reminder="!")


def test_call_tool_returns_refusals() -> None:
    client = ScriptedClient([refusal_response("bio")])
    outcome = call_tool(client, request(), tool_name="submit_answer", retry_reminder="!")
    assert outcome.refused
    assert outcome.refusal_category == "bio"


def test_call_tool_never_uses_a_truncated_tool_call() -> None:
    truncated = tool_response("submit_answer", ANSWER_INPUT, stop_reason="max_tokens")
    client = ScriptedClient([truncated])
    with pytest.raises(ModelResponseError, match="max_tokens"):
        call_tool(client, request(), tool_name="submit_answer", retry_reminder="!")
