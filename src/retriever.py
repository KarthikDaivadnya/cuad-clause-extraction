"""
retriever.py
------------
Embedding-based retrieval over a contract's chunks.

Used in two places:
    1. clause_extractor.py — before asking the LLM "what is the termination
       clause?", we first retrieve the top-K chunks most semantically similar
       to a query like "termination of agreement, notice period...". This is
       a lightweight RAG pattern that keeps prompts short and grounded.
    2. semantic_search.py (the assignment's bonus feature) — lets a user type
       a free-text query and find the most relevant clause passages across
       one or many contracts.

We use sentence-transformers (all-MiniLM-L6-v2): small, CPU-friendly, no
API cost, and good enough quality for clause-level semantic matching.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from src import config
from src.chunker import Chunk

logger = logging.getLogger(__name__)

_model = None  # lazy-loaded singleton — avoids reloading the model per contract


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class ChunkRetriever:
    """Holds embeddings for one contract's chunks and answers similarity queries."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._embeddings = embed_texts([c.text for c in chunks]) if chunks else np.empty((0, 0))

    def top_k(self, query: str, k: int = config.TOP_K_CHUNKS_PER_CLAUSE) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_vec = embed_texts([query])[0]
        scores = self._embeddings @ query_vec  # cosine similarity (embeddings are normalized)
        top_idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]
