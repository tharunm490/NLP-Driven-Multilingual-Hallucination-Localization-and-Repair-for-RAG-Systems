"""Generative LLM service.

Independent of FastAPI and of the retrieval implementation. The only real
provider implemented here speaks the OpenAI-compatible chat-completions API,
which covers Groq, OpenAI, OpenRouter, Together, and local Ollama-compatible
servers by changing ``LLM_BASE_URL``.

Credentials are read from backend environment variables only. The API key is
never logged, never returned in responses, and never sent to the frontend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import httpx

from app.core.config import Settings
from app.core.errors import GenerationConfigurationError, GenerationError

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]

# provider -> (default base URL, default model)
PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", ""),
    "together": ("https://api.together.xyz/v1", ""),
    "ollama": ("http://localhost:11434/v1", "llama3.1"),
    "openai-compatible": ("", ""),
    "custom": ("", ""),
}

_KEYLESS_PROVIDERS = {"ollama"}

_MAX_ERROR_EXCERPT = 300


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@runtime_checkable
class Generator(Protocol):
    """Replaceable generation backend."""

    provider: str
    model: str

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult: ...


# -- provider implementations ------------------------------------------------


class OpenAICompatibleGenerator:
    """Chat-completions client for any OpenAI-compatible provider."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 800,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        if not self._api_key:
            raise GenerationConfigurationError(
                f"LLM_API_KEY is not set for provider '{self.provider}'."
            )
        if not self.model:
            raise GenerationConfigurationError(
                f"LLM_MODEL is not set for provider '{self.provider}'."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        logger.info(
            "Calling LLM provider=%s model=%s messages=%s timeout=%ss",
            self.provider,
            self.model,
            len(messages),
            self._timeout,
        )
        try:
            response = self._client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise GenerationError(
                f"The LLM provider '{self.provider}' did not respond within "
                f"{self._timeout:.0f}s. Try again or increase LLM_TIMEOUT_SECONDS."
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"Could not reach the LLM provider '{self.provider}'. "
                "Check LLM_BASE_URL and your network connection."
            ) from exc

        self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise GenerationError(
                f"The LLM provider '{self.provider}' returned a non-JSON response."
            ) from exc

        return self._parse_response(data)

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        detail = self._error_excerpt(response)
        if status in {401, 403}:
            raise GenerationConfigurationError(
                f"Authentication with LLM provider '{self.provider}' failed "
                f"(HTTP {status}). Check that LLM_API_KEY is valid."
            )
        if status == 404:
            raise GenerationConfigurationError(
                f"LLM model '{self.model}' was not found at provider "
                f"'{self.provider}' (HTTP 404). Check LLM_MODEL and LLM_BASE_URL."
            )
        if status == 429:
            raise GenerationError(
                f"LLM provider '{self.provider}' rate-limited the request (HTTP 429). "
                "Wait and retry."
            )
        if status >= 500:
            raise GenerationError(
                f"LLM provider '{self.provider}' is unavailable (HTTP {status})." + detail
            )
        raise GenerationError(
            f"LLM provider '{self.provider}' rejected the request (HTTP {status})." + detail
        )

    @staticmethod
    def _error_excerpt(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return ""
        message = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
            elif isinstance(error, str):
                message = error
            if message is None and isinstance(body.get("detail"), str):
                message = body["detail"]
        if not message:
            return ""
        return f" Provider message: {str(message)[:_MAX_ERROR_EXCERPT]}"

    def _parse_response(self, data: Any) -> GenerationResult:
        if not isinstance(data, dict):
            raise GenerationError("The LLM provider returned an unexpected response shape.")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GenerationError("The LLM provider returned no answer choices.")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise GenerationError("The LLM provider returned an empty answer.")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return GenerationResult(
            text=text.strip(),
            provider=self.provider,
            model=str(data.get("model") or self.model),
            finish_reason=first.get("finish_reason"),
            prompt_tokens=_as_int(usage.get("prompt_tokens")),
            completion_tokens=_as_int(usage.get("completion_tokens")),
            total_tokens=_as_int(usage.get("total_tokens")),
        )


class MockGenerator:
    """Deterministic generator for tests and offline development.

    It never calls a network. The full message list of every call is recorded
    so tests can assert which evidence reached the generation service.
    """

    provider = "mock"

    def __init__(self, model: str = "mock-model", answer: str | None = None) -> None:
        self.model = model
        self._answer = answer
        self.calls: list[list[ChatMessage]] = []

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        self.calls.append(list(messages))
        if self._answer is not None:
            text = self._answer
        else:
            user_text = next((m.content for m in messages if m.role == "user"), "")
            text = f"[mock:{self.model}] {user_text[:280]}"
        return GenerationResult(text=text, provider=self.provider, model=self.model)


class UnconfiguredGenerator:
    """Placeholder returned when the LLM is not configured.

    The application still starts; the RAG endpoint raises a clear configuration
    error only when answer generation is actually requested.
    """

    provider = "unconfigured"
    model = ""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        raise GenerationConfigurationError(self.reason)


# -- factory -----------------------------------------------------------------


def build_generator(settings: Settings) -> Generator:
    """Construct the configured generator (never raises for missing config)."""
    provider = settings.llm_provider_normalised

    if provider in {"", "none"}:
        return UnconfiguredGenerator(
            "No LLM provider is configured. Set LLM_PROVIDER (for example 'groq'), "
            "LLM_MODEL, and LLM_API_KEY in backend/.env to enable answer generation."
        )

    if provider == "mock":
        logger.warning(
            "LLM_PROVIDER=mock: using the deterministic mock generator. "
            "This is for tests/development only and does not call a real model."
        )
        return MockGenerator(model=settings.llm_model or "mock-model")

    default_base, default_model = PROVIDER_DEFAULTS.get(provider, ("", ""))
    base_url = settings.llm_base_url.strip() or default_base
    model = settings.llm_model.strip() or default_model

    if not base_url:
        return UnconfiguredGenerator(
            f"LLM_PROVIDER='{provider}' requires LLM_BASE_URL to be set."
        )
    if not model:
        return UnconfiguredGenerator(
            f"LLM_MODEL is not set for provider '{provider}'."
        )
    if provider not in _KEYLESS_PROVIDERS and not settings.resolved_llm_api_key:
        return UnconfiguredGenerator(
            f"LLM_API_KEY is not set for provider '{provider}'. "
            "Add it to backend/.env (never commit it)."
        )

    logger.info("Configured LLM generator: provider=%s model=%s", provider, model)
    return OpenAICompatibleGenerator(
        provider=provider,
        model=model,
        api_key=settings.resolved_llm_api_key,
        base_url=base_url,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
