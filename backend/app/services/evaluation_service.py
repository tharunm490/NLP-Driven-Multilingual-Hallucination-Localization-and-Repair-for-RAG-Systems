"""Evaluation service: computes ROUGE-L and semantic similarity for a
candidate answer against a ground-truth reference.

Used by POST /api/rag/evaluate to compare RAG vs LLM-only answers.
"""

from __future__ import annotations

import numpy as np
from rouge_score import rouge_scorer

from app.services.embedding_service import EmbeddingService


class EvaluationService:
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedder = embedding_service
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, candidate: str, reference: str) -> dict[str, float]:
        """Return ROUGE-L F1 and semantic similarity for candidate vs reference."""
        rouge_l = self._scorer.score(reference, candidate)["rougeL"].fmeasure

        vecs = self._embedder.embed_documents([candidate, reference])
        a, b = np.array(vecs[0]), np.array(vecs[1])
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        semantic_sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0

        return {
            "rouge_l": round(rouge_l, 4),
            "semantic_similarity": round(semantic_sim, 4),
        }
