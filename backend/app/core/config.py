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
    # Completed Ask responses are retained briefly in memory so the
    # Hallucination Checks page can reuse the exact answer without rerunning
    # retrieval or generation. Only the opaque result ID is stored client-side.
    rag_result_cache_size: int = Field(default=32, ge=1, le=500)
    rag_result_ttl_seconds: float = Field(default=3600.0, gt=0.0, le=86400.0)

    # Evidence sufficiency assessment (pre-generation decision layer)
    rag_evidence_assessment_mode: Literal["auto", "llm", "heuristic"] = "auto"
    rag_evidence_assessment_fallback_enabled: bool = True
    rag_assessment_max_chunks: int = Field(default=8, ge=1, le=50)
    rag_assessment_max_evidence_chars: int = Field(default=12000, ge=500, le=100000)
    # Only used by the conservative lexical fallback, never as the primary
    # sufficiency signal when an LLM judge is available.
    rag_lexical_sufficient_coverage: float = Field(default=0.6, ge=0.0, le=1.0)
    rag_lexical_partial_coverage: float = Field(default=0.35, ge=0.0, le=1.0)

    # Claim-level verification (Phase 4A)
    verification_enabled: bool = True
    # Claim extraction is deterministic and local. The extractor may find more
    # claims than the verifier checks, but the hard ceiling is 30 regardless of
    # this setting.
    verification_max_extracted_claims: int = Field(default=30, ge=1, le=500)
    verification_max_claims: int = Field(default=20, ge=1, le=200)
    # Safety ceiling only. The default matches the largest retrievable top_k so
    # that every retrieved chunk is compared with every checked claim.
    verification_max_evidence_per_claim: int = Field(default=50, ge=1, le=50)
    # Hard ceiling for expensive NLI work. Pairs beyond this limit remain
    # visible in the response as not evaluated instead of being silently lost.
    verification_max_nli_pairs: int = Field(default=250, ge=1, le=1000)
    # Maximum size of a focused evidence context around the best-matching
    # sentence when an entire chunk does not fit the NLI token limit.
    verification_primary_window_tokens: int = Field(default=256, ge=64, le=2048)
    verification_deadline_seconds: float = Field(default=120.0, gt=0.0, le=1800.0)
    # Minimum claim-term/semantic relevance before a raw NLI verdict can count
    # toward the final claim status.
    verification_min_evidence_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_max_concurrent_requests: int = Field(default=1, ge=1, le=16)
    verification_queue_wait_seconds: float = Field(default=5.0, ge=0.0, le=60.0)
    verification_max_queued_requests: int = Field(default=2, ge=0, le=64)
    verification_nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    verification_nli_backend: Literal["local", "onnx", "hf_endpoint"] = "local"
    verification_nli_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    verification_nli_batch_size: int = Field(default=8, ge=1, le=128)
    verification_nli_max_batch_tokens: int = Field(default=8192, ge=256, le=131072)
    verification_nli_cache_size: int = Field(default=256, ge=0, le=100000)
    verification_nli_num_threads: int = Field(default=0, ge=0, le=64)
    verification_nli_onnx_model_path: str = ""
    verification_nli_onnx_provider: Literal[
        "CPUExecutionProvider", "CUDAExecutionProvider"
    ] = "CPUExecutionProvider"
    verification_nli_onnx_intra_threads: int = Field(default=0, ge=0, le=64)
    verification_nli_onnx_inter_threads: int = Field(default=1, ge=0, le=64)
    verification_nli_hosted_batch_size: int = Field(default=1, ge=1, le=32)
    verification_nli_max_concurrency: int = Field(default=2, ge=1, le=8)
    verification_nli_max_tokens: int = Field(default=256, ge=64, le=2048)
    verification_nli_endpoint_url: str = ""
    verification_nli_api_key: SecretStr = SecretStr("")
    verification_nli_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    verification_nli_max_retries: int = Field(default=2, ge=0, le=5)
    verification_nli_retry_backoff_seconds: float = Field(
        default=0.5, ge=0.0, le=10.0
    )
    # Additional orchestration-level retries. A transient local-model failure
    # must not discard every remaining claim-evidence comparison.
    verification_nli_engine_retries: int = Field(default=2, ge=0, le=5)
    verification_nli_entailment_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_nli_contradiction_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    verification_ner_enabled: bool = True
    verification_ner_model_name: str = "Davlan/xlm-roberta-base-wikiann-ner"
    verification_ner_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    verification_ner_max_tokens: int = Field(default=512, ge=32, le=2048)

    # Dedicated hallucination groundedness detector. This is intentionally
    # separate from the NLI verifier so the two signals remain independent.
    hallucination_enabled: bool = True
    hallucination_model_name: str = "Horizon-Labs/hallucination-guard-base"
    hallucination_model_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    hallucination_model_local_files_only: bool = False
    hallucination_model_max_tokens: int = Field(default=2048, ge=32, le=8192)
    hallucination_model_batch_size: int = Field(default=8, ge=1, le=128)
    # Optional override for checkpoints whose config exposes generic labels
    # such as LABEL_0/LABEL_1 instead of UNSUPPORTED/SUPPORTED.
    hallucination_supported_label_index: int | None = Field(default=None, ge=0, le=1)
    hallucination_support_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    hallucination_review_margin: float = Field(default=0.05, ge=0.0, le=0.5)
    hallucination_max_claims: int = Field(default=64, ge=1, le=500)
    hallucination_max_evidence_per_claim: int = Field(default=5, ge=1, le=50)
    hallucination_max_evidence_chars: int = Field(
        default=4000, ge=200, le=100000
    )
    hallucination_max_pairs: int = Field(default=500, ge=1, le=2000)
    hallucination_deadline_seconds: float = Field(
        default=120.0, gt=0.0, le=1800.0
    )
    hallucination_cache_size: int = Field(default=32, ge=1, le=500)

    # Evidence-grounded iterative repair. This layer reuses the existing NLI
    # verifier and Hallucination Guard; it never changes their thresholds.
    repair_enabled: bool = True
    repair_max_attempts: Literal[3] = 3
    repair_max_evidence_per_claim: int = Field(default=3, ge=1, le=20)
    repair_deadline_seconds: float = Field(default=600.0, gt=0.0, le=3600.0)
    repair_result_cache_size: int = Field(default=32, ge=1, le=500)
    repair_result_ttl_seconds: float = Field(
        default=3600.0, gt=0.0, le=86400.0
    )

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
        if self.rag_lexical_partial_coverage > self.rag_lexical_sufficient_coverage:
            raise ValueError(
                "RAG_LEXICAL_PARTIAL_COVERAGE must not exceed "
                "RAG_LEXICAL_SUFFICIENT_COVERAGE"
            )
        return self

    @model_validator(mode="after")
    def validate_verification_windows(self) -> "Settings":
        if self.verification_primary_window_tokens > self.verification_nli_max_tokens:
            raise ValueError(
                "VERIFICATION_PRIMARY_WINDOW_TOKENS must not exceed "
                "VERIFICATION_NLI_MAX_TOKENS"
            )
        if self.verification_max_extracted_claims < self.verification_max_claims:
            raise ValueError(
                "VERIFICATION_MAX_EXTRACTED_CLAIMS must be at least "
                "VERIFICATION_MAX_CLAIMS so skipped claims can be reported"
            )
        return self

    @model_validator(mode="after")
    def validate_nli_backend(self) -> "Settings":
        if self.verification_nli_backend == "onnx":
            if not self.verification_nli_onnx_model_path.strip():
                raise ValueError(
                    "VERIFICATION_NLI_ONNX_MODEL_PATH is required when "
                    "VERIFICATION_NLI_BACKEND=onnx"
                )
            return self
        if self.verification_nli_backend != "hf_endpoint":
            return self
        endpoint = self.verification_nli_endpoint_url.strip()
        if not endpoint:
            raise ValueError(
                "VERIFICATION_NLI_ENDPOINT_URL is required when "
                "VERIFICATION_NLI_BACKEND=hf_endpoint"
            )
        if not endpoint.startswith("https://"):
            raise ValueError(
                "VERIFICATION_NLI_ENDPOINT_URL must start with https:// "
                "to protect the API key in transit"
            )
        if not self.resolved_verification_nli_api_key:
            raise ValueError(
                "VERIFICATION_NLI_API_KEY is required when "
                "VERIFICATION_NLI_BACKEND=hf_endpoint"
            )
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

    # -- NLI helpers --------------------------------------------------------

    @property
    def resolved_verification_nli_api_key(self) -> str:
        secret = self.verification_nli_api_key
        return secret.get_secret_value().strip() if secret else ""

    @property
    def resolved_verification_nli_onnx_model_path(self) -> Path:
        value = self.verification_nli_onnx_model_path.strip()
        if not value:
            return Path("")
        return self._resolve(Path(value))

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
