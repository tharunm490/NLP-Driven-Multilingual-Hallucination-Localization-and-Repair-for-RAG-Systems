import math

import pytest

from app.core.config import Settings
from app.core.errors import EmbeddingError
from app.services.embedding_service import EmbeddingService
from tests.conftest import EMBEDDING_DIMENSION, FakeEncoderModel


def test_embed_documents_shape_and_dimension(settings: Settings):
    service = EmbeddingService(settings, model_factory=lambda: FakeEncoderModel())
    vectors = service.embed_documents(["first passage", "second passage"])
    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)


def test_embeddings_are_normalised(settings: Settings):
    service = EmbeddingService(settings, model_factory=lambda: FakeEncoderModel())
    vector = service.embed_documents(["normalise me"])[0]
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-6)


def test_batching_uses_configured_batch_size(settings: Settings):
    fake = FakeEncoderModel()
    service = EmbeddingService(settings, model_factory=lambda: fake)
    service.embed_documents(["a", "b", "c", "d", "e"])
    assert fake.encode_calls == [2, 2, 1]


def test_e5_prefixes_are_applied():
    settings = Settings(
        _env_file=None,
        embedding_model_name="intfloat/multilingual-e5-base",
        embedding_query_prefix="auto",
        embedding_passage_prefix="auto",
        embedding_batch_size=4,
    )
    fake = FakeEncoderModel()
    service = EmbeddingService(settings, model_factory=lambda: fake)
    service.embed_documents(["passage text"])
    service.embed_query("query text")
    assert fake.last_inputs == ["query: query text"]
    assert service._settings.effective_passage_prefix == "passage: "  # noqa: SLF001


def test_no_prefix_for_bge_style_models(settings: Settings):
    fake = FakeEncoderModel()
    service = EmbeddingService(settings, model_factory=lambda: fake)
    service.embed_query("plain query")
    assert fake.last_inputs == ["plain query"]


def test_empty_inputs_are_handled(settings: Settings):
    service = EmbeddingService(settings, model_factory=lambda: FakeEncoderModel())
    assert service.embed_documents([]) == []
    with pytest.raises(EmbeddingError):
        service.embed_documents(["   "])
    with pytest.raises(EmbeddingError):
        service.embed_query("   ")


def test_inference_failure_is_reported(settings: Settings):
    service = EmbeddingService(settings, model_factory=lambda: FakeEncoderModel(fail=True))
    with pytest.raises(EmbeddingError):
        service.embed_documents(["trigger failure"])


def test_dimension_mismatch_is_detected(settings: Settings):
    class WrongDimensionModel(FakeEncoderModel):
        def encode(self, texts, batch_size=None, normalize_embeddings=False, **kwargs):
            return [[0.1, 0.2] for _ in texts]

    service = EmbeddingService(settings, model_factory=lambda: WrongDimensionModel())
    with pytest.raises(EmbeddingError):
        service.embed_documents(["dimension mismatch"])


def test_lazy_loading_reuses_model(settings: Settings):
    fake = FakeEncoderModel()
    service = EmbeddingService(settings, model_factory=lambda: fake)
    assert not service.is_loaded
    service.embed_documents(["load once"])
    assert service.is_loaded
    assert service.dimension == EMBEDDING_DIMENSION
