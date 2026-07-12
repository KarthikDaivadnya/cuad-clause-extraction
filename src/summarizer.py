"""
summarizer.py
--------------
Task 2, Part B: "Generate a concise 100-150 word summary highlighting
purpose, key obligations of each party, and notable risks or penalties."

Uses a map-reduce strategy so long contracts don't blow context windows:
    MAP:    summarize each chunk individually into 2-3 bullet points.
    REDUCE: combine the bullet summaries into one 100-150 word narrative
            summary that hits the three required points.

For short contracts (a handful of chunks) this collapses to effectively one
map call + one reduce call, so there's no meaningful overhead on small docs.
"""
from __future__ import annotations

import logging
from typing import List

from src import config
from src.chunker import Chunk
from src.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

_MAP_SYSTEM = (
    "You are a contract-review assistant. Summarize ONLY what is stated in "
    "the passage — do not speculate. Be terse."
)
_MAP_PROMPT = """\
Extract, as 2-4 short bullet points, anything in this passage relevant to:
(a) the purpose of the agreement, (b) obligations of either party, or
(c) risks, penalties, or damages.
If none of these are present, respond with "N/A".

Passage:
\"\"\"{chunk_text}\"\"\"

Bullets:
"""

_REDUCE_SYSTEM = (
    "You are a contract-review assistant writing an executive summary for a "
    "legal operations team. Be precise and only use the information given."
)
_REDUCE_PROMPT = """\
Below are bullet-point notes extracted from different sections of one
contract. Write a single, coherent summary of the contract in
{min_words}-{max_words} words. The summary MUST cover:
  1. The purpose of the agreement.
  2. The key obligations of each party.
  3. Any notable risks or penalties.

Notes:
\"\"\"
{notes}
\"\"\"

Write the summary as plain prose (no bullet points, no headers, no preamble).
"""


def _map_chunk(provider: LLMProvider, chunk: Chunk) -> str:
    prompt = _MAP_PROMPT.format(chunk_text=chunk.text)
    return provider.complete(prompt=prompt, system=_MAP_SYSTEM, max_tokens=150)


def summarize_contract(provider: LLMProvider, chunks: List[Chunk]) -> str:
    """Map-reduce summary over all chunks of a contract."""
    if not chunks:
        return ""

    # MAP — summarize each chunk. Skip "N/A" results to keep the reduce
    # prompt focused on chunks that actually contained something useful.
    notes: List[str] = []
    for chunk in chunks:
        try:
            bullets = _map_chunk(provider, chunk)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Map step failed on chunk %d: %s", chunk.chunk_id, exc)
            continue
        if bullets.strip().upper() != "N/A":
            notes.append(bullets.strip())

    if not notes:
        return "No summarizable content was found in this contract."

    min_words, max_words = config.SUMMARY_WORD_RANGE
    reduce_prompt = _REDUCE_PROMPT.format(
        min_words=min_words, max_words=max_words, notes="\n\n".join(notes)
    )
    return provider.complete(prompt=reduce_prompt, system=_REDUCE_SYSTEM, max_tokens=350)
