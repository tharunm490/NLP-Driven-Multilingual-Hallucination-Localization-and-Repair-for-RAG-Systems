"""Configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings.

    Values are read from environment variables and from ``backend/.env``.
    Relative paths are resolved from the backend directory so the application
    behaves the same regardless of the shell's current working directory.
    """

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    app_version: str = "0.1.0"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    frontend_origin: str = "http://localhost:3000"
    log_level: str = "INFO"

    # Vector store
    chroma_persist_directory: Path = Path("./data/chroma")
    chroma_collection_name: str = "rag_documents"
    chroma_batch_size: int = Field(default=256, ge=1, le=4096)

    # Embeddings
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    embedding_batch_size: int = Field(default=4, ge=1, le=256)
    embedding_max_seq_length: int = Field(default=1024, ge=32, le=8192)
    embedding_normalize: bool = True
    embedding_query_prefix: str = "auto"
    embedding_passage_prefix: str = "auto"
    embedding_trust_remote_code: bool = False
    tokenizer_model_name: str = ""

    # Chunking
    chunk_size: int = Field(default=700, ge=50, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=4000)

    # Ingestion and retrieval
    top_k: int = Field(default=5, ge=1, le=50)
    max_upload_size_mb: int = Field(default=20, ge=1, le=1024)
    max_files_per_request: int = Field(default=5, ge=1, le=50)
    allow_duplicate_documents: bool = False
    language_detection_enabled: bool = True

    # Query language detection (Phase 3)
    query_language_min_chars: int = Field(default=3, ge=1, le=1000)
    query_language_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    query_language_min_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    query_language_fallback: str = ""

    # RAG pipeline (Phase 3)
    rag_max_question_length: int = Field(default=2000, ge=1, le=20000)
    rag_min_evidence_chunks: int = Field(default=1, ge=1, le=50)
    rag_distance_filter_enabled: bool = True
    rag_max_distance: float = Field(default=0.75, ge=0.0, le=2.0)
    rag_generation_without_evidence: bool = False

    # Generative LLM (Phase 3)
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = ""
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    llm_max_output_tokens: int = Field(default=800, ge=1, le=32000)

    # Storage
    data_directory: Path = Path("./data")
    document_registry_path: Path = Path("./data/document_registry.sqlite3")

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # -- Derived values -----------------------------------------------------

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()

    @property
    def resolved_data_directory(self) -> Path:
        return self._resolve(self.data_directory)

    @property
    def resolved_chroma_directory(self) -> Path:
        return self._resolve(self.chroma_persist_directory)

    @property
    def resolved_registry_path(self) -> Path:
        return self._resolve(self.document_registry_path)

    @property
    def frontend_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origin.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def normalised_log_level(self) -> str:
        return self.log_level.upper()

    @property
    def is_e5_model(self) -> bool:
        return "e5" in self.embedding_model_name.lower()

    @property
    def effective_query_prefix(self) -> str:
        if self.embedding_query_prefix != "auto":
            return self.embedding_query_prefix
        return "query: " if self.is_e5_model else ""

    @property
    def effective_passage_prefix(self) -> str:
        if self.embedding_passage_prefix != "auto":
            return self.embedding_passage_prefix
        return "passage: " if self.is_e5_model else ""

    @property
    def effective_tokenizer_model_name(self) -> str:
        return self.tokenizer_model_name or self.embedding_model_name

    # -- LLM helpers --------------------------------------------------------

    @property
    def llm_provider_normalised(self) -> str:
        return self.llm_provider.strip().lower()

    @property
    def resolved_llm_api_key(self) -> str:
        return self.llm_api_key.get_secret_value().strip() if self.llm_api_key else ""

    @property
    def llm_configured(self) -> bool:
        """True when a real provider and credentials are configured."""
        return self.llm_provider_normalised not in {"", "none", "mock"} and bool(
            self.resolved_llm_api_key
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
