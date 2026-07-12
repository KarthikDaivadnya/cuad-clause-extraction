import json

import pytest

from src.chunker import Chunk
from src.clause_extractor import _extract_json, extract_clause, extract_all_clauses
from src import config


class FakeRetriever:
    """Stands in for ChunkRetriever without requiring a real embedding model."""

    def __init__(self, chunks):
        self.chunks = chunks

    def top_k(self, query, k=4):
        return [(c, 1.0) for c in self.chunks[:k]]


class FakeProvider:
    """Stands in for LLMProvider; returns a canned response regardless of prompt."""

    model = "fake-model"

    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = 0

    def complete(self, prompt, system="", max_tokens=500, temperature=0.0):
        self.calls += 1
        return self.response_text


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------
def test_extract_json_plain():
    raw = '{"found": true, "clause_text": "abc", "key_terms": {}}'
    assert _extract_json(raw) == {"found": True, "clause_text": "abc", "key_terms": {}}


def test_extract_json_with_markdown_fence():
    raw = '```json\n{"found": false, "clause_text": "", "key_terms": {}}\n```'
    assert _extract_json(raw)["found"] is False


def test_extract_json_with_surrounding_prose():
    raw = 'Sure, here is the answer:\n{"found": true, "clause_text": "x", "key_terms": {}}\nHope that helps!'
    assert _extract_json(raw)["clause_text"] == "x"


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        _extract_json("I could not find any relevant clause.")


# ---------------------------------------------------------------------------
# extract_clause / extract_all_clauses
# ---------------------------------------------------------------------------
def test_extract_clause_happy_path():
    chunks = [Chunk(chunk_id=0, text="Either party may terminate with 30 days notice.", start_char=0)]
    retriever = FakeRetriever(chunks)
    provider = FakeProvider(json.dumps({
        "found": True,
        "clause_text": "30 days notice to terminate",
        "key_terms": {"notice_period": "30 days"},
    }))

    result = extract_clause(provider, retriever, "termination_clause")
    assert result["found"] is True
    assert "30 days" in result["clause_text"]
    assert provider.calls == 1


def test_extract_clause_no_chunks_returns_not_found_without_calling_llm():
    provider = FakeProvider("should not be called")
    result = extract_clause(provider, FakeRetriever([]), "liability_clause")
    assert result["found"] is False
    assert provider.calls == 0


def test_extract_clause_handles_malformed_llm_response_gracefully():
    chunks = [Chunk(chunk_id=0, text="some contract text", start_char=0)]
    provider = FakeProvider("The model rambled without returning JSON.")
    result = extract_clause(provider, FakeRetriever(chunks), "confidentiality_clause")
    assert result["found"] is False
    assert result.get("parse_error") is True


def test_extract_clause_rejects_unknown_clause_type():
    with pytest.raises(ValueError):
        extract_clause(FakeProvider("{}"), FakeRetriever([]), "not_a_real_clause")


def test_extract_all_clauses_calls_llm_once_per_clause_type():
    chunks = [Chunk(chunk_id=0, text="some contract text", start_char=0)]
    provider = FakeProvider(json.dumps({"found": True, "clause_text": "x", "key_terms": {}}))
    results = extract_all_clauses(provider, FakeRetriever(chunks))
    assert set(results.keys()) == set(config.CLAUSE_TYPES)
    assert provider.calls == len(config.CLAUSE_TYPES)
