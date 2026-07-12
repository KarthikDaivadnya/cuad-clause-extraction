"""
pipeline.py
-----------
Orchestrates the full flow described in the assignment README:

    Load PDFs -> Normalize -> Chunk -> Embed -> [Extract clauses | Summarize]
    -> Write CSV/JSON

Each contract is processed independently and defensively: if one contract
fails (bad PDF, LLM hiccup, etc.) the pipeline logs it and moves on rather
than aborting the whole batch — important when running over 50 real-world
SEC filings that vary wildly in formatting quality.

Per-contract results are cached to disk (.cache/<contract_id>.json) so
re-running the pipeline (e.g. after tweaking a prompt) doesn't re-spend
API calls on contracts that already succeeded, unless --no-cache is passed.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from src import config
from src.chunker import chunk_text
from src.clause_extractor import extract_all_clauses
from src.data_loader import load_contracts
from src.llm_provider import get_provider
from src.preprocessor import normalize_text
from src.retriever import ChunkRetriever
from src.summarizer import summarize_contract

logger = logging.getLogger(__name__)


def _cache_path(contract_id: str) -> Path:
    return config.CACHE_DIR / f"{contract_id}.json"


def process_contract(provider, contract, use_cache: bool = True) -> dict:
    cache_file = _cache_path(contract.contract_id)
    if use_cache and cache_file.exists():
        logger.info("Using cached result for %s", contract.contract_id)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    normalized = normalize_text(contract.raw_text)
    chunks = chunk_text(normalized)
    retriever = ChunkRetriever(chunks)

    clauses = extract_all_clauses(provider, retriever)
    summary = summarize_contract(provider, chunks)

    result = {
        "contract_id": contract.contract_id,
        "source_path": contract.source_path,
        "num_chunks": len(chunks),
        "summary": summary,
        **clauses,
    }

    cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    limit: Optional[int] = 50,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = get_provider(provider_name, model)
    logger.info("Using provider=%s model=%s", provider.__class__.__name__, provider.model)

    contracts = list(load_contracts(data_dir, limit=limit))
    results: List[dict] = []

    for contract in tqdm(contracts, desc="Processing contracts"):
        start = time.time()
        try:
            result = process_contract(provider, contract, use_cache=use_cache)
            result["processing_seconds"] = round(time.time() - start, 2)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 — keep the batch alive
            logger.error("FAILED on %s: %s", contract.contract_id, exc, exc_info=True)
            results.append({
                "contract_id": contract.contract_id,
                "source_path": contract.source_path,
                "error": str(exc),
            })

    # --- write JSON (full structured output, keeps nested key_terms) -------
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- write CSV (flat, matches the columns the assignment asks for) -----
    flat_rows = []
    for r in results:
        flat_rows.append({
            "contract_id": r.get("contract_id"),
            "summary": r.get("summary", ""),
            "termination_clause": (r.get("termination_clause") or {}).get("clause_text", ""),
            "confidentiality_clause": (r.get("confidentiality_clause") or {}).get("clause_text", ""),
            "liability_clause": (r.get("liability_clause") or {}).get("clause_text", ""),
            "error": r.get("error", ""),
        })
    df = pd.DataFrame(flat_rows)
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    logger.info("Wrote %s and %s", csv_path, json_path)
    return df
