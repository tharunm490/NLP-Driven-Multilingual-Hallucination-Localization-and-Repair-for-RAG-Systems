"""Tests for the hosted NLI endpoint client (network mocked)."""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import NliError
from app.services.verification.nli import HostedNliModel, build_nli_model


class FakeTokenizer:
    model_max_length = 512

    def __call__(
        self,
        premise: str,
        hypothesis: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
    ) -> dict[str, list[int]]:
        count = len(premise.split()) + len(hypothesis.split()) + 3
        if truncation:
            count = min(count, max_length)
        return {"input_ids": list(range(count))}


def _row(entailment: float, neutral: float, contradiction: float) -> list[dict]:
    return [
        {"label": "entailment", "score": entailment},
        {"label": "neutral", "score": neutral},
        {"label": "contradiction", "score": contradiction},
    ]


def _model(
    handler,
    *,
    batch_size: int = 2,
    max_concurrency: int = 1,
    max_retries: int = 0,
    retry_backoff_seconds: float = 0.0,
) -> HostedNliModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HostedNliModel(
        model_name="test/model",
        endpoint_url="https://endpoint.test",
        api_key="test-secret",
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        max_tokens=128,
        timeout_seconds=5.0,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        tokenizer_factory=lambda _: FakeTokenizer(),
        client=client,
        sleep=lambda _: None,
    )


def test_hosted_single_pair_uses_premise_hypothesis_payload():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-secret"
        return httpx.Response(200, json=_row(0.1, 0.2, 0.7))

    model = _model(handler, max_retries=0)

    predictions = model.predict([("The policy is reviewed annually.", "The policy is reviewed yearly.")])

    assert requests == [
        {
            "inputs": {
                "text": "The policy is reviewed annually.",
                "text_pair": "The policy is reviewed yearly.",
            },
            "parameters": {
                "function_to_apply": "softmax",
                "top_k": 3,
            },
        }
    ]
    assert predictions[0].label == "contradiction"
    assert predictions[0].confidence == pytest.approx(0.7)
    assert predictions[0].probabilities["neutral"] == pytest.approx(0.2)


def test_hosted_batch_maps_labels_independent_of_response_order():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["inputs"] == [
            {"text": "evidence one", "text_pair": "claim one"},
            {"text": "evidence two", "text_pair": "claim two"},
        ]
        return httpx.Response(
            200,
            json=[
                [
                    {"label": "neutral", "score": 0.2},
                    {"label": "entailment", "score": 0.7},
                    {"label": "contradiction", "score": 0.1},
                ],
                [
                    {"label": "contradiction", "score": 0.8},
                    {"label": "neutral", "score": 0.1},
                    {"label": "entailment", "score": 0.1},
                ],
            ],
        )

    model = _model(handler)

    predictions = model.predict(
        [("evidence one", "claim one"), ("evidence two", "claim two")]
    )

    assert [item.label for item in predictions] == [
        "entailment",
        "contradiction",
    ]


def test_hosted_client_returns_neutral_prediction():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_row(0.1, 0.75, 0.15))

    model = _model(handler)

    predictions = model.predict([("premise", "hypothesis")])

    assert predictions[0].label == "neutral"
    assert predictions[0].confidence == pytest.approx(0.75)
    assert predictions[0].probabilities == pytest.approx(
        {"entailment": 0.1, "neutral": 0.75, "contradiction": 0.15}
    )


def test_hosted_client_deduplicates_identical_pairs_before_sending():
    payloads: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload["inputs"])
        return httpx.Response(
            200,
            json=[_row(0.8, 0.1, 0.1), _row(0.1, 0.1, 0.8)],
        )

    model = _model(handler, batch_size=4)

    predictions = model.predict(
        [
            ("same evidence", "same claim"),
            ("same evidence", "same claim"),
            ("other evidence", "other claim"),
        ]
    )

    assert len(payloads) == 1
    assert payloads[0] == [
        {"text": "same evidence", "text_pair": "same claim"},
        {"text": "other evidence", "text_pair": "other claim"},
    ]
    assert [item.label for item in predictions] == [
        "entailment",
        "entailment",
        "contradiction",
    ]


def test_hosted_client_uses_bounded_concurrency():
    barrier = threading.Barrier(2, timeout=2.0)

    def handler(request: httpx.Request) -> httpx.Response:
        barrier.wait()
        return httpx.Response(200, json=_row(0.8, 0.1, 0.1))

    model = _model(handler, batch_size=1, max_concurrency=2)

    predictions = model.predict(
        [("evidence one", "claim one"), ("evidence two", "claim two")]
    )

    assert [item.label for item in predictions] == [
        "entailment",
        "entailment",
    ]


def test_hosted_client_retries_transient_failure():
    calls: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        return httpx.Response(200, json=_row(0.8, 0.1, 0.1))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = HostedNliModel(
        model_name="test/model",
        endpoint_url="https://endpoint.test",
        api_key="test-secret",
        batch_size=1,
        max_tokens=128,
        timeout_seconds=5.0,
        max_retries=1,
        retry_backoff_seconds=0.0,
        tokenizer_factory=lambda _: FakeTokenizer(),
        client=client,
        sleep=sleeps.append,
    )

    predictions = model.predict([("premise", "hypothesis")])

    assert len(calls) == 2
    assert predictions[0].label == "entailment"


def test_hosted_client_does_not_retry_authentication_failure():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401)

    model = _model(handler, max_retries=3)

    with pytest.raises(NliError, match="authentication failed"):
        model.predict([("premise", "hypothesis")])

    assert len(calls) == 1


def test_hosted_client_retries_transport_error():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection dropped", request=request)
        return httpx.Response(200, json=_row(0.2, 0.3, 0.5))

    model = _model(handler, batch_size=1, max_retries=1)

    predictions = model.predict([("premise", "hypothesis")])

    assert len(calls) == 2
    assert predictions[0].label == "contradiction"


def test_hosted_client_retries_timeout():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("read timed out", request=request)
        return httpx.Response(200, json=_row(0.8, 0.1, 0.1))

    model = _model(handler, batch_size=1, max_retries=1)

    predictions = model.predict([("premise", "hypothesis")])

    assert len(calls) == 2
    assert predictions[0].label == "entailment"


def test_hosted_client_honours_retry_after_for_429():
    calls: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json=_row(0.1, 0.2, 0.7))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = HostedNliModel(
        model_name="test/model",
        endpoint_url="https://endpoint.test",
        api_key="test-secret",
        batch_size=1,
        max_concurrency=1,
        max_tokens=128,
        timeout_seconds=5.0,
        max_retries=1,
        retry_backoff_seconds=0.1,
        tokenizer_factory=lambda _: FakeTokenizer(),
        client=client,
        sleep=sleeps.append,
    )

    model.predict([("premise", "hypothesis")])

    assert len(calls) == 2
    assert sleeps[0] >= 0.25


def test_hosted_client_reports_quota_exhaustion_without_retry():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(402)

    model = _model(handler, max_retries=3)

    with pytest.raises(NliError, match="quota or credits"):
        model.predict([("premise", "hypothesis")])

    assert len(calls) == 1


def test_hosted_client_reports_missing_endpoint_without_retry():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404)

    model = _model(handler, max_retries=3)

    with pytest.raises(NliError, match="endpoint or model was not found"):
        model.predict([("premise", "hypothesis")])

    assert len(calls) == 1


def test_hosted_client_rejects_invalid_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    model = _model(handler)

    with pytest.raises(NliError, match="invalid JSON"):
        model.predict([("premise", "hypothesis")])


def test_hosted_client_rejects_batch_count_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_row(0.8, 0.1, 0.1)])

    model = _model(handler)

    with pytest.raises(NliError, match="different number of predictions"):
        model.predict([("evidence one", "claim one"), ("evidence two", "claim two")])


def test_hosted_client_rejects_missing_neutral_label():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"label": "entailment", "score": 0.6},
                {"label": "contradiction", "score": 0.4},
            ],
        )

    model = _model(handler)

    with pytest.raises(NliError, match="did not return entailment"):
        model.predict([("premise", "hypothesis")])


def test_hosted_client_rejects_unnormalized_scores():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_row(0.9, 0.9, 0.9))

    model = _model(handler)

    with pytest.raises(NliError, match="do not sum to one"):
        model.predict([("premise", "hypothesis")])


def test_hosted_client_counts_tokens_without_loading_the_model():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("token counting must not call the endpoint")

    model = _model(handler)

    assert model.count_pair_tokens("two words", "three words here") > 0
    assert model.max_pair_tokens == 128


def test_build_nli_model_selects_configured_backend_without_loading_models():
    local = build_nli_model(Settings(_env_file=None))
    assert local.__class__.__name__ == "TransformerNliModel"

    hosted = build_nli_model(
        Settings(
            _env_file=None,
            verification_nli_backend="hf_endpoint",
            verification_nli_endpoint_url="https://endpoint.test",
            verification_nli_api_key="test-secret",
        )
    )
    assert isinstance(hosted, HostedNliModel)
    assert hosted.model_name == "hf-endpoint:MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    assert not hosted.is_loaded


def test_hosted_api_key_is_redacted_from_settings_repr():
    settings = Settings(
        _env_file=None,
        verification_nli_backend="hf_endpoint",
        verification_nli_endpoint_url="https://endpoint.test",
        verification_nli_api_key="hf_super_secret_value",
    )

    assert "hf_super_secret_value" not in repr(settings)
    assert settings.resolved_verification_nli_api_key == "hf_super_secret_value"


def test_hosted_backend_requires_endpoint_and_api_key():
    with pytest.raises(ValueError, match="ENDPOINT_URL"):
        Settings(
            _env_file=None,
            verification_nli_backend="hf_endpoint",
        )

    with pytest.raises(ValueError, match="API_KEY"):
        Settings(
            _env_file=None,
            verification_nli_backend="hf_endpoint",
            verification_nli_endpoint_url="https://endpoint.test",
        )

    with pytest.raises(ValueError, match="https://"):
        Settings(
            _env_file=None,
            verification_nli_backend="hf_endpoint",
            verification_nli_endpoint_url="http://endpoint.test",
            verification_nli_api_key="test-secret",
        )
