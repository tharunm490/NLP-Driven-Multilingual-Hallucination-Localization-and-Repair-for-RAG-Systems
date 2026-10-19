"""Optional, conservative language detection for stored metadata.

Language codes are stored only when detection is reasonably stable. Short
chunks are not reliable enough, so detection runs on the full document text.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MIN_TEXT_LENGTH = 120


class LanguageDetector:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._detector_ready = False
        self._detect_langs = None

    def _load(self) -> None:
        if self._detector_ready:
            return
        self._detector_ready = True
        try:
            from langdetect import DetectorFactory, detect_langs

            DetectorFactory.seed = 0
            self._detect_langs = detect_langs
        except Exception as exc:  # noqa: BLE001 - metadata only
            logger.warning("Language detection unavailable: %s", exc)
            self._detect_langs = None

    def detect(self, text: str) -> str | None:
        if not self.enabled or len(text.strip()) < _MIN_TEXT_LENGTH:
            return None
        self._load()
        if self._detect_langs is None:
            return None
        try:
            results = self._detect_langs(text[:6000])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Language detection failed: %s", exc)
            return None
        if not results:
            return None
        top = results[0]
        margin = top.prob - results[1].prob if len(results) > 1 else 1.0
        if top.prob >= 0.85 and margin >= 0.15:
            return top.lang
        return None
