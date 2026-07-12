"""
semantic_search.py
-------------------
Bonus feature: "Implement semantic search over clauses using embeddings."

Builds a single FAISS index over every extracted clause (termination /
confidentiality / liability) across ALL processed contracts, so a user can
type something like "what happens if we miss a payment" and get back the
most relevant clauses + which contract they came from — regardless of
clause type or exact wording.

This is intentionally a separate, corpus-level index from retriever.py's
per-contract retrieval (which is used internally during extraction).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.retriever import embed_texts

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    contract_id: str
    clause_type: str
    clause_text: str
    score: float


class ClauseSearchIndex:
    def __init__(self):
        self._entries: List[dict] = []
        self._index = None

    @classmethod
    def from_results_json(cls, results_path: Path) -> "ClauseSearchIndex":
        """Build the index from the pipeline's output/results.json file."""
        idx = cls()
        records = json.loads(Path(results_path).read_text(encoding="utf-8"))
        for record in records:
            for clause_type in ("termination_clause", "confidentiality_clause", "liability_clause"):
                clause = record.get(clause_type) or {}
                text = clause.get("clause_text") if isinstance(clause, dict) else None
                if text:
                    idx._entries.append(
                        {"contract_id": record["contract_id"], "clause_type": clause_type, "clause_text": text}
                    )
        idx._build()
        return idx

    def _build(self):
        import faiss  # local import — optional heavy dependency, only needed for this bonus feature

        if not self._entries:
            logger.warning("No clauses available to index.")
            return
        vectors = embed_texts([e["clause_text"] for e in self._entries]).astype("float32")
        self._index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity via normalized dot product
        self._index.add(vectors)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if self._index is None or not self._entries:
            return []
        query_vec = embed_texts([query]).astype("float32")
        scores, ids = self._index.search(query_vec, min(k, len(self._entries)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            entry = self._entries[idx]
            results.append(SearchResult(
                contract_id=entry["contract_id"],
                clause_type=entry["clause_type"],
                clause_text=entry["clause_text"],
                score=float(score),
            ))
        return results
