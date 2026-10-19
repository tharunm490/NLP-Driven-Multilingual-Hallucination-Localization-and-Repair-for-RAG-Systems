"""Tests for the generation service (no real API calls)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import GenerationConfigurationError, GenerationError
from app.services.generator import (
    ChatMessage,
    MockGenerator,
    OpenAICompatibleGenerator,
    UnconfiguredGenerator,
    build_generator,
)

MESSAGES = [
    ChatMessage(role="system", content="SYSTEM RULES"),
    ChatMessage(role="user", content="QUESTION + EVIDENCE"),
]


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def make_generator(handler, **overrides) -> OpenAICompatibleGenerator:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    defaults = dict(
        provider="groq",
        model="test-model",
        api_key="secret-key",
        base_url="https://example.test/openai/v1",
        timeout_seconds=5.0,
        max_output_tokens=64,
    )
    defaults.update(overrides)
    return OpenAICompatibleGenerator(client=client, **defaults)


def ok_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "choices": [
                {"message": {"content": "  grounded answer  "}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
    )


# -- factory -----------------------------------------------------------------


def test_unconfigured_when_no_provider():
    generator = build_generator(make_settings(llm_provider=""))
    assert isinstance(generator, UnconfiguredGenerator)
    with pytest.raises(GenerationConfigurationError):
        generator.generate(MESSAGES)


def test_mock_provider_is_deterministic():
    generator = build_generator(make_settings(llm_provider="mock", llm_model="m"))
    assert isinstance(generator, MockGenerator)
    first = generator.generate(MESSAGES)
    second = generator.generate(MESSAGES)
    assert first.text == second.text
    assert first.provider == "mock"
    assert len(generator.calls) == 2


def test_groq_uses_documented_defaults():
    generator = build_generator(make_settings(llm_provider="groq", llm_api_key="k"))
    assert isinstance(generator, OpenAICompatibleGenerator)
    assert generator.provider == "groq"
    assert generator.model == "openai/gpt-oss-120b"


def test_missing_api_key_is_not_configured():
    generator = build_generator(make_settings(llm_provider="groq"))
    assert isinstance(generator, UnconfiguredGenerator)


def test_openai_compatible_requires_base_url():
    generator = build_generator(
        make_settings(llm_provider="openai-compatible", llm_api_key="k", llm_model="m")
    )
    assert isinstance(generator, UnconfiguredGenerator)


# -- OpenAI-compatible client ------------------------------------------------


def test_generate_success_parses_answer_and_tokens():
    generator = make_generator(ok_response)
    result = generator.generate(MESSAGES)
    assert result.text == "grounded answer"
    assert result.provider == "groq"
    assert result.model == "test-model"
    assert result.finish_reason == "stop"
    assert result.total_tokens == 14


def test_request_keeps_roles_separate_and_sends_auth_header():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return ok_response(request)

    make_generator(handler).generate(MESSAGES)
    assert [m["role"] for m in captured["body"]["messages"]] == ["system", "user"]
    assert captured["body"]["messages"][0]["content"] == "SYSTEM RULES"
    assert captured["auth"] == "Bearer secret-key"
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["max_tokens"] == 64
    assert captured["body"]["stream"] is False


def test_missing_key_at_call_time_raises_configuration_error():
    generator = make_generator(ok_response, api_key="")
    with pytest.raises(GenerationConfigurationError):
        generator.generate(MESSAGES)


def test_auth_failure_maps_to_configuration_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    with pytest.raises(GenerationConfigurationError) as exc:
        make_generator(handler).generate(MESSAGES)
    assert "Authentication" in exc.value.message
    assert "secret-key" not in exc.value.message


def test_rate_limit_maps_to_generation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(GenerationError) as exc:
        make_generator(handler).generate(MESSAGES)
    assert "rate" in exc.value.message.lower()


def test_server_error_maps_to_generation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    with pytest.raises(GenerationError):
        make_generator(handler).generate(MESSAGES)


def test_timeout_maps_to_generation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("too slow", request=request)

    with pytest.raises(GenerationError) as exc:
        make_generator(handler).generate(MESSAGES)
    assert "did not respond" in exc.value.message


def test_connection_error_maps_to_generation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(GenerationError):
        make_generator(handler).generate(MESSAGES)


def test_non_json_response_maps_to_generation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(GenerationError):
        make_generator(handler).generate(MESSAGES)


def test_empty_choices_maps_to_generation_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(GenerationError):
        make_generator(handler).generate(MESSAGES)
