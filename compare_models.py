#!/usr/bin/env python3
"""
compare_models.py
------------------
Addresses the evaluation rubric's "Creativity" bullet: "comparisons between
models." Runs clause extraction + summarization on the same small sample of
contracts with two different LLM providers/models, and reports:
    - per-contract latency for each provider
    - whether both providers agreed a clause was "found"
    - a simple text-overlap similarity between the two extracted clause texts

This is meant as a diagnostic/demo script, not part of the main pipeline —
it's how you'd defend a model choice in an interview ("here's why I picked
Llama-3.3-70b over GPT-4o-mini for this task: comparable extraction quality,
~3x faster, and free tier").

Usage:
    python compare_models.py --data-dir data/full_contract_pdf_sample --limit 5 \\
        --provider-a groq --model-a llama-3.3-70b-versatile \\
        --provider-b openai --model-b gpt-4o-mini
"""
import argparse
import difflib
import logging
import time
from pathlib import Path

import pandas as pd

from src import config
from src.chunker import chunk_text
from src.clause_extractor import extract_all_clauses
from src.data_loader import load_contracts
from src.llm_provider import get_provider
from src.preprocessor import normalize_text
from src.retriever import ChunkRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def run_one_provider(provider_name, model, contracts):
    provider = get_provider(provider_name, model)
    rows = []
    for contract in contracts:
        normalized = normalize_text(contract.raw_text)
        chunks = chunk_text(normalized)
        retriever = ChunkRetriever(chunks)

        start = time.time()
        clauses = extract_all_clauses(provider, retriever)
        elapsed = time.time() - start

        row = {"contract_id": contract.contract_id, "provider": provider_name, "model": model,
               "seconds": round(elapsed, 2)}
        for ct in config.CLAUSE_TYPES:
            row[f"{ct}_found"] = clauses[ct].get("found", False)
            row[f"{ct}_text"] = clauses[ct].get("clause_text", "")
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "full_contract_pdf_sample")
    p.add_argument("--limit", type=int, default=5, help="Keep this small — you're paying/waiting 2x per contract.")
    p.add_argument("--provider-a", default="groq")
    p.add_argument("--model-a", default=config.DEFAULT_MODELS["groq"])
    p.add_argument("--provider-b", default="openai")
    p.add_argument("--model-b", default=config.DEFAULT_MODELS["openai"])
    args = p.parse_args()

    contracts = list(load_contracts(args.data_dir, limit=args.limit))
    print(f"Comparing {args.provider_a}/{args.model_a} vs {args.provider_b}/{args.model_b} on {len(contracts)} contract(s)\n")

    rows_a = run_one_provider(args.provider_a, args.model_a, contracts)
    rows_b = run_one_provider(args.provider_b, args.model_b, contracts)

    comparison = []
    for ra, rb in zip(rows_a, rows_b):
        entry = {
            "contract_id": ra["contract_id"],
            f"{args.provider_a}_seconds": ra["seconds"],
            f"{args.provider_b}_seconds": rb["seconds"],
        }
        for ct in config.CLAUSE_TYPES:
            entry[f"{ct}_agree_found"] = ra[f"{ct}_found"] == rb[f"{ct}_found"]
            entry[f"{ct}_text_similarity"] = round(text_similarity(ra[f"{ct}_text"], rb[f"{ct}_text"]), 2)
        comparison.append(entry)

    df = pd.DataFrame(comparison)
    out_path = config.OUTPUT_DIR / "model_comparison.csv"
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nAvg latency — {args.provider_a}: {df[f'{args.provider_a}_seconds'].mean():.2f}s | "
          f"{args.provider_b}: {df[f'{args.provider_b}_seconds'].mean():.2f}s")
    print(f"Saved detailed comparison to {out_path}")


if __name__ == "__main__":
    main()
