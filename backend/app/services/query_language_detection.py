"""Query language detection.

A dedicated, replaceable service. The detection *backend* (Lingua by default,
langdetect as a fallback) is isolated behind a small protocol so a stronger
model can be dropped in later without touching the RAG pipeline.

Important: a detected language is metadata only. It is not proof that the
embedding model or the generative LLM will perform equally well in that
language, and it is never used to translate or rewrite the user's query.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Languages we deliberately support/curate inside Lingua. Names are resolved
# defensively because Lingua's enum has changed between releases.
_LINGUA_LANGUAGE_NAMES: tuple[str, ...] = (
    "ENGLISH",
    "HINDI",
    "TAMIL",
    "TELUGU",
    "BENGALI",
    "MARATHI",
    "URDU",
    "GUJARATI",
    "KANNADA",
    "MALAYALAM",
    "PUNJABI",
    "FRENCH",
    "GERMAN",
    "SPANISH",
    "PORTUGUESE",
    "ITALIAN",
    "DUTCH",
    "RUSSIAN",
    "UKRAINIAN",
    "POLISH",
    "TURKISH",
    "ARABIC",
    "PERSIAN",
    "HEBREW",
    "CHINESE",
    "JAPANESE",
    "KOREAN",
    "VIETNAMESE",
    "THAI",
    "INDONESIAN",
    "SWAHILI",
)

# Unicode name based script buckets used only to flag *mixed-script* queries.
_SCRIPT_PREFIXES: dict[str, str] = {
    "LATIN": "latin",
    "DEVANAGARI": "devanagari",
    "TAMIL": "tamil",
    "TELUGU": "telugu",
    "BENGALI": "bengali",
    "GUJARATI": "gujarati",
    "GURMUKHI": "gurmukhi",
    "KANNADA": "kannada",
    "MALAYALAM": "malayalam",
    "CYRILLIC": "cyrillic",
    "ARABIC": "arabic",
    "HEBREW": "hebrew",
    "CJK": "han",
    "HIRAGANA": "japanese",
    "KATAKANA": "japanese",
    "HANGUL": "hangul",
    "THAI": "thai",
    "GREEK": "greek",
}

_MIN_LETTERS_PER_SCRIPT = 3


@dataclass(frozen=True)
class LanguageScore:
    """A single (language, confidence) candidate returned by a backend."""

    code: str
    confidence: float


@dataclass(frozen=True)
class LanguageDetection:
    """Outcome of query language detection."""

    code: str | None
    confidence: float | None
    reliable: bool
    method: str
    mixed: bool = False
    reason: str | None = None


@runtime_checkable
class LanguageBackend(Protocol):
    """Replaceable language-identification backend."""

    name: str

    def is_available(self) -> bool: ...

    def detect(self, text: str) -> list[LanguageScore]: ...


# -- backends ---------------------------------------------------------------


class LinguaBackend:
    """High-accuracy backend built on the Lingua library."""

    name = "lingua"

    def __init__(self) -> None:
        self._detector = None
        self._load_attempted = False

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from lingua import Language, LanguageDetectorBuilder

            languages = [
                language
                for language in (
                    getattr(Language, member, None) for member in _LINGUA_LANGUAGE_NAMES
                )
                if language is not None
            ]
            if not languages:
                raise RuntimeError("Lingua exposed no usable languages")
            self._detector = LanguageDetectorBuilder.from_languages(*languages).build()
            logger.info("Lingua language detection backend ready (%s languages)", len(languages))
        except Exception as exc:  # noqa: BLE001 - optional dependency
            logger.warning("Lingua backend unavailable: %s", exc)
            self._detector = None

    def is_available(self) -> bool:
        self._load()
        return self._detector is not None

    def detect(self, text: str) -> list[LanguageScore]:
        self._load()
        if self._detector is None:
            return []
        values = self._detector.compute_language_confidence_values(text)
        scores: list[LanguageScore] = []
        for value in values:
            code = _lingua_iso_code(value.language)
            if code:
                scores.append(LanguageScore(code=code, confidence=float(value.value)))
        return scores


def _lingua_iso_code(language: object) -> str | None:
    try:
        iso = language.iso_code_639_1  # type: ignore[attr-defined]
        name = getattr(iso, "name", None)
        if name:
            return str(name).lower()
    except Exception:  # noqa: BLE001
        return None
    return None


class LangdetectBackend:
    """Fallback backend using langdetect (already a Phase 2 dependency)."""

    name = "langdetect"

    def __init__(self) -> None:
        self._detect_langs = None
        self._load_attempted = False

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from langdetect import DetectorFactory, detect_langs

            DetectorFactory.seed = 0
            self._detect_langs = detect_langs
        except Exception as exc:  # noqa: BLE001 - optional dependency
            logger.warning("langdetect backend unavailable: %s", exc)
            self._detect_langs = None

    def is_available(self) -> bool:
        self._load()
        return self._detect_langs is not None

    def detect(self, text: str) -> list[LanguageScore]:
        self._load()
        if self._detect_langs is None:
            return []
        try:
            results = self._detect_langs(text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("langdetect failed: %s", exc)
            return []
        return [LanguageScore(code=result.lang, confidence=float(result.prob)) for result in results]


# -- service ----------------------------------------------------------------


class QueryLanguageDetector:
    """Detect the language of a user query with a configurable policy."""

    def __init__(
        self,
        enabled: bool = True,
        backend: LanguageBackend | None = None,
        fallback_backend: LanguageBackend | None = None,
        min_chars: int = 3,
        min_confidence: float = 0.5,
        min_margin: float = 0.15,
        fallback_language: str = "",
        max_chars: int = 1000,
    ) -> None:
        self.enabled = enabled
        self._backend = backend if backend is not None else LinguaBackend()
        self._fallback_backend = (
            fallback_backend if fallback_backend is not None else LangdetectBackend()
        )
        self._min_chars = max(1, min_chars)
        self._min_confidence = min_confidence
        self._min_margin = min_margin
        self._fallback_language = fallback_language.strip().lower()
        self._max_chars = max(100, max_chars)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._backend.is_available():
            return True
        return self._fallback_backend.is_available()

    def detect(self, text: str) -> LanguageDetection:
        if not self.enabled:
            return LanguageDetection(None, None, False, "disabled", reason="disabled")

        stripped = (text or "").strip()
        if not stripped:
            return LanguageDetection(None, None, False, self._backend.name, reason="empty")
        if len(stripped) < self._min_chars:
            return LanguageDetection(
                None, None, False, self._backend.name, reason="too_short"
            )
        if not _contains_letters(stripped):
            return LanguageDetection(
                None, None, False, self._backend.name, reason="no_letters"
            )

        sample = stripped[: self._max_chars]
        scores, method = self._detect_with_backends(sample)
        if not scores:
            if self._fallback_language:
                return LanguageDetection(
                    self._fallback_language,
                    None,
                    False,
                    f"{method}+fallback",
                    reason="no_candidates_fallback",
                )
            return LanguageDetection(None, None, False, method, reason="no_candidates")

        top = scores[0]
        runner_up = scores[1] if len(scores) > 1 else None
        margin = top.confidence - runner_up.confidence if runner_up else 1.0
        mixed = _is_mixed_script(stripped) or (
            runner_up is not None
            and runner_up.code != top.code
            and runner_up.confidence >= self._min_confidence
        )

        reliable = (
            top.confidence >= self._min_confidence
            and margin >= self._min_margin
            and not mixed
        )

        if reliable:
            return LanguageDetection(top.code, top.confidence, True, method, mixed=False)

        if not _is_mixed_script(stripped):
            confirmed = self._confirm_with_fallback(sample, top.code)
            if confirmed is not None:
                return confirmed

        reason = "mixed_language" if mixed else "low_confidence"
        if self._fallback_language:
            return LanguageDetection(
                self._fallback_language, top.confidence, False, f"{method}+fallback",
                mixed=mixed, reason=f"{reason}_fallback",
            )
        # Best-effort guess is still returned but flagged as not reliable; the
        # RAG response must not present it as a confident detection.
        return LanguageDetection(top.code, top.confidence, False, method, mixed=mixed, reason=reason)

    def _confirm_with_fallback(
        self, text: str, primary_code: str
    ) -> LanguageDetection | None:
        fallback = self._fallback_backend
        if fallback is self._backend or not fallback.is_available():
            return None
        try:
            scores = sorted(
                fallback.detect(text),
                key=lambda item: item.confidence,
                reverse=True,
            )
        except Exception as exc:  # noqa: BLE001 - detection must not break RAG
            logger.warning("Language backend '%s' raised: %s", fallback.name, exc)
            return None
        if not scores:
            return None
        top = scores[0]
        runner_up = scores[1] if len(scores) > 1 else None
        margin = top.confidence - runner_up.confidence if runner_up else 1.0
        if (
            top.code == primary_code
            and top.confidence >= self._min_confidence
            and margin >= self._min_margin
        ):
            return LanguageDetection(
                top.code,
                top.confidence,
                True,
                f"{self._backend.name}+{fallback.name}",
            )
        return None

    def _detect_with_backends(self, text: str) -> tuple[list[LanguageScore], str]:
        for backend in (self._backend, self._fallback_backend):
            if not backend.is_available():
                continue
            try:
                scores = backend.detect(text)
            except Exception as exc:  # noqa: BLE001 - detection must not break RAG
                logger.warning("Language backend '%s' raised: %s", backend.name, exc)
                continue
            if scores:
                return sorted(scores, key=lambda item: item.confidence, reverse=True), backend.name
        return [], self._backend.name


# -- helpers ----------------------------------------------------------------


def _contains_letters(text: str) -> bool:
    return any(char.isalpha() for char in text)


def _script_key(char: str) -> str | None:
    if not char.isalpha():
        return None
    try:
        unicode_name = unicodedata.name(char)
    except ValueError:
        return None
    for prefix, script in _SCRIPT_PREFIXES.items():
        if unicode_name.startswith(prefix):
            return script
    return None


def _is_mixed_script(text: str) -> bool:
    counts: dict[str, int] = {}
    for char in text:
        script = _script_key(char)
        if script:
            counts[script] = counts.get(script, 0) + 1
    significant = [script for script, count in counts.items() if count >= _MIN_LETTERS_PER_SCRIPT]
    return len(significant) >= 2
