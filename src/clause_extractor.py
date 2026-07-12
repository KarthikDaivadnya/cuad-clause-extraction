"""
clause_extractor.py
--------------------
Task 2, Part A: "Use an LLM to identify and extract key clauses: Termination
conditions, Confidentiality clauses, Liability clauses."

Pipeline per clause type:
    1. Retrieve the top-K chunks of the contract most relevant to that clause
       (via ChunkRetriever — see retriever.py). This is what lets this
       pipeline handle 40+ page contracts without truncating or blowing the
       context window.
    2. Build a prompt containing: instructions, a few-shot example, and the
       retrieved passages.
    3. Ask the LLM to respond in strict JSON so downstream code (CSV/JSON
       export) doesn't need brittle regex parsing of prose.
    4. Defensively parse the response; fall back to "not found" rather than
       crashing the whole pipeline on one bad contract.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict

from src import config
from src.llm_provider import LLMProvider
from src.retriever import ChunkRetriever
from prompts.few_shot_examples import FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a contract-review assistant used by a legal operations team. "
    "You extract clauses ONLY from the text you are given — never invent "
    "language that is not present in the passages. You always respond with "
    "a single valid JSON object and nothing else (no markdown fences, no "
    "commentary)."
)

_INSTRUCTION_TEMPLATE = """\
Clause type to extract: {clause_label}

Here is one worked example of the expected output format:
Passage:
\"\"\"{example_passage}\"\"\"
Expected JSON answer:
{example_answer}

Now extract the same kind of information from the passages below, which were
retrieved from a different contract (the passages may be out of order and may
include irrelevant surrounding text — ignore anything not related to
{clause_label_readable}).

Passages:
\"\"\"
{passages}
\"\"\"

Respond with a single JSON object with exactly these keys:
- "found": true or false — whether this clause is actually present in the passages
- "clause_text": a faithful, condensed restatement of the relevant clause (empty string if not found)
- "key_terms": a short JSON object of any extra structured details you can confidently identify (e.g. notice periods, caps, durations); use {{}} if none

Return ONLY the JSON object.
"""

_CLAUSE_READABLE = {
    "termination_clause": "contract termination conditions",
    "confidentiality_clause": "confidentiality / non-disclosure obligations",
    "liability_clause": "liability, indemnification, or limitation-of-damages terms",
}


def _extract_json(raw: str) -> Dict:
    """LLMs occasionally wrap JSON in markdown fences or add stray text —
    pull out the first {...} block defensively."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")
    return json.loads(match.group(0))


def extract_clause(
    provider: LLMProvider,
    retriever: ChunkRetriever,
    clause_type: str,
) -> Dict:
    """Extract a single clause type from a contract via retrieval + LLM call."""
    if clause_type not in config.CLAUSE_TYPES:
        raise ValueError(f"Unknown clause_type '{clause_type}'")

    query = config.CLAUSE_RETRIEVAL_QUERIES[clause_type]
    top_chunks = retriever.top_k(query, k=config.TOP_K_CHUNKS_PER_CLAUSE)

    if not top_chunks:
        return {"found": False, "clause_text": "", "key_terms": {}}

    passages = "\n---\n".join(c.text for c, _score in top_chunks)
    example = FEW_SHOT_EXAMPLES[clause_type][0]

    prompt = _INSTRUCTION_TEMPLATE.format(
        clause_label=clause_type,
        clause_label_readable=_CLAUSE_READABLE[clause_type],
        example_passage=example["passage"],
        example_answer=json.dumps(example["answer"], indent=2),
        passages=passages,
    )

    raw_response = provider.complete(prompt=prompt, system=_SYSTEM_PROMPT, max_tokens=500)

    try:
        return _extract_json(raw_response)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse JSON for %s: %s. Raw: %.200s", clause_type, exc, raw_response)
        return {"found": False, "clause_text": "", "key_terms": {}, "parse_error": True}


def extract_all_clauses(provider: LLMProvider, retriever: ChunkRetriever) -> Dict[str, Dict]:
    return {ct: extract_clause(provider, retriever, ct) for ct in config.CLAUSE_TYPES}
