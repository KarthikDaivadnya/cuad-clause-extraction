#!/usr/bin/env python3
"""
scripts/evaluate_accuracy.py
------------------------------
Answers the question this project has been missing an answer to: "how
accurate is the extraction, really?" -- measured against CUAD's own
expert-labeled gold data, not against another model's opinion.

This is a DIFFERENT question from `compare_models.py`, which measures
agreement BETWEEN two providers. Two providers can agree with each other and
both be wrong. This script instead checks each provider's output against
ground truth, so it answers both "is this accurate" and (run once per
provider) "which provider is more accurate" with the same evidence.

Ground truth source: the official CUAD GitHub repo (same one used by
scripts/prepare_demo_contracts.py and the demo notebook), which ships
`CUADv1.json` -- a SQuAD-format file with expert-annotated answer spans for
41 clause categories per contract. Two of those map cleanly onto this
project's clause types:

    termination_clause      <- CUAD category "Termination For Convenience"
    liability_clause        <- CUAD categories "Uncapped Liability" OR
                                "Cap On Liability" (present if either has
                                an answer)
    confidentiality_clause  <- NO direct CUAD category exists. Reported as
                                "not evaluable" rather than faking a number.

For each matched contract and evaluable clause type, this script reports:
    1. Detection accuracy  -- did "found: true/false" agree with whether
       CUAD's annotators found a gold span at all?
    2. Content overlap     -- for true positives (both say "found"), how much
       character-sequence overlap is there between the extracted clause_text
       and CUAD's gold span text? (difflib ratio, same technique already
       used in compare_models.py for cross-provider comparison.)

Usage:
    # Single run
    python scripts/evaluate_accuracy.py --run groq=output/results.json

    # Multi-provider comparison table (run main.py once per provider first,
    # saving/renaming each output/results.json between runs)
    python scripts/evaluate_accuracy.py \\
        --run groq=output/results_groq.json \\
        --run openai=output/results_openai.json \\
        --run anthropic=output/results_anthropic.json

    # No --run given: defaults to output/results.json labeled "pipeline"
    python scripts/evaluate_accuracy.py

Does not modify main.py, pipeline.py, or any other existing file -- it only
reads results.json files that the existing pipeline already produces.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # this script lives in scripts/, not the root

from src import config
CUAD_GITHUB_ZIP = "https://codeload.github.com/TheAtticusProject/cuad/zip/refs/heads/main"

# Map our clause types to the CUAD gold category name(s) that cover them.
# A list means "present if ANY of these categories has a gold answer".
GOLD_CATEGORY_MAP = {
    "termination_clause": ["Termination For Convenience"],
    "liability_clause": ["Uncapped Liability", "Cap On Liability"],
    # confidentiality_clause intentionally omitted -- no CUAD equivalent.
}
NOT_EVALUABLE_CLAUSES = set(config.CLAUSE_TYPES) - set(GOLD_CATEGORY_MAP)


# ---------------------------------------------------------------------------
# Step 1: fetch + parse CUAD's gold labels (same source as the demo notebook)
# ---------------------------------------------------------------------------
def load_cuad_gold(cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_zip = cache_dir / "cuad_repo.zip"
    if not repo_zip.exists():
        print("Downloading CUAD gold-label data from GitHub...")
        with requests.get(CUAD_GITHUB_ZIP, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(repo_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    with zipfile.ZipFile(repo_zip) as zf:
        inner_data_zip = [n for n in zf.namelist() if n.endswith("data.zip")][0]
        zf.extract(inner_data_zip, cache_dir)
    with zipfile.ZipFile(cache_dir / inner_data_zip) as zf:
        with zf.open("CUADv1.json") as f:
            cuad = json.load(f)
    return cuad["data"]


def extract_gold_answer(qas: list[dict], category: str) -> Optional[str]:
    """Return the concatenated gold answer text for a category, or None if
    CUAD's annotators marked it not present (is_impossible / no answers)."""
    for qa in qas:
        if qa["id"].split("__")[-1] != category:
            continue
        answers = qa.get("answers", [])
        if qa.get("is_impossible") or not answers:
            return None
        # CUAD sometimes has multiple non-contiguous spans for one category
        texts = [a["text"] for a in answers if a.get("text")]
        return " ... ".join(texts) if texts else None
    return None  # category not found in this contract's qas at all


def build_gold_lookup(cuad_data: list[dict]) -> dict[str, dict]:
    """contract_title -> {clause_type: gold_text_or_None}"""
    lookup = {}
    for contract in cuad_data:
        qas = contract["paragraphs"][0]["qas"]
        entry = {}
        for clause_type, categories in GOLD_CATEGORY_MAP.items():
            gold_text = None
            for cat in categories:
                found = extract_gold_answer(qas, cat)
                if found:
                    gold_text = found
                    break
            entry[clause_type] = gold_text
        lookup[contract["title"]] = entry
    return lookup


# ---------------------------------------------------------------------------
# Step 2: match our results.json contract_ids to CUAD gold titles
# ---------------------------------------------------------------------------
def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def match_to_gold(contract_id: str, gold_titles: list[str], norm_titles: dict[str, str],
                   threshold: float = 0.55) -> Optional[str]:
    """Best-effort fuzzy match: our PDF-derived contract_id vs CUAD's title
    strings. Filenames from download_cuad.py / prepare_demo_contracts.py
    aren't guaranteed to be byte-identical to CUAD's `title` field, so this
    matches on normalized substring containment first, then falls back to a
    similarity ratio. Returns None (excluded from scoring) below threshold."""
    norm_id = normalize(contract_id)
    best_title, best_score = None, 0.0
    for title in gold_titles:
        norm_title = norm_titles[title]
        if norm_id in norm_title or norm_title in norm_id:
            return title  # substring match -- treat as a confident match
        score = difflib.SequenceMatcher(None, norm_id, norm_title).ratio()
        if score > best_score:
            best_title, best_score = title, score
    return best_title if best_score >= threshold else None


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


# ---------------------------------------------------------------------------
# Step 3: score one run (one results.json) against gold
# ---------------------------------------------------------------------------
def evaluate_run(results_path: Path, gold_lookup: dict[str, dict]) -> pd.DataFrame:
    records = json.loads(results_path.read_text(encoding="utf-8"))
    gold_titles = list(gold_lookup.keys())
    norm_titles = {t: normalize(t) for t in gold_titles}

    rows = []
    for record in records:
        contract_id = record.get("contract_id", "")
        matched_title = match_to_gold(contract_id, gold_titles, norm_titles)

        row = {"contract_id": contract_id, "matched_gold_title": matched_title or ""}

        for clause_type in config.CLAUSE_TYPES:
            extracted = record.get(clause_type) or {}
            found = bool(extracted.get("found", False))
            clause_text = extracted.get("clause_text", "") or ""

            if clause_type in NOT_EVALUABLE_CLAUSES or matched_title is None:
                row[f"{clause_type}_gold_present"] = None
                row[f"{clause_type}_found"] = found
                row[f"{clause_type}_correct"] = None
                row[f"{clause_type}_overlap"] = None
                continue

            gold_text = gold_lookup[matched_title][clause_type]
            gold_present = gold_text is not None

            row[f"{clause_type}_gold_present"] = gold_present
            row[f"{clause_type}_found"] = found
            row[f"{clause_type}_correct"] = (found == gold_present)
            row[f"{clause_type}_overlap"] = (
                text_similarity(clause_text, gold_text) if (found and gold_present) else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_run(df: pd.DataFrame) -> dict:
    summary = {"n_contracts": len(df), "n_matched": int((df["matched_gold_title"] != "").sum())}
    for clause_type in config.CLAUSE_TYPES:
        if clause_type in NOT_EVALUABLE_CLAUSES:
            summary[clause_type] = "not evaluable (no CUAD gold category)"
            continue
        scored = df[df[f"{clause_type}_correct"].notna()]
        n = len(scored)
        if n == 0:
            summary[clause_type] = "no matched contracts"
            continue
        correct = int(scored[f"{clause_type}_correct"].sum())
        overlaps = scored[f"{clause_type}_overlap"].dropna()
        overlap_str = f", avg overlap {overlaps.mean():.2f}" if len(overlaps) else ""
        summary[clause_type] = f"{correct}/{n} ({100*correct/n:.0f}%){overlap_str}"
    return summary


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                         help="A results.json to score, e.g. --run groq=output/results.json. Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    args = parser.parse_args()

    runs = args.run or [f"pipeline={PROJECT_ROOT / 'output' / 'results.json'}"]
    parsed_runs = []
    for entry in runs:
        if "=" not in entry:
            raise SystemExit(f"--run must be LABEL=PATH, got: {entry!r}")
        label, path = entry.split("=", 1)
        parsed_runs.append((label, Path(path)))

    for label, path in parsed_runs:
        if not path.exists():
            raise SystemExit(f"Results file not found for run '{label}': {path}")

    print("Fetching CUAD gold labels (cached after first run)...")
    cuad_cache = PROJECT_ROOT / "data" / ".cuad_cache"
    cuad_data = load_cuad_gold(cuad_cache)
    gold_lookup = build_gold_lookup(cuad_data)
    print(f"Loaded gold labels for {len(gold_lookup)} CUAD contracts.\n")

    all_summaries = {}
    for label, path in parsed_runs:
        df = evaluate_run(path, gold_lookup)
        out_csv = args.output_dir / f"accuracy_report_{label}.csv"
        df.to_csv(out_csv, index=False)
        all_summaries[label] = summarize_run(df)
        print(f"[{label}] wrote {out_csv} ({len(df)} contracts, {int((df['matched_gold_title']!='').sum())} matched to gold)")

    # ---- comparison table -------------------------------------------------
    print("\n" + "=" * 78)
    print("ACCURACY SUMMARY  (detection accuracy = found vs CUAD gold presence)")
    print("=" * 78)
    table_rows = []
    for label, summary in all_summaries.items():
        table_rows.append({
            "provider": label,
            "termination_clause": summary["termination_clause"],
            "liability_clause": summary["liability_clause"],
            "confidentiality_clause": summary["confidentiality_clause"],
            "matched/total": f"{summary['n_matched']}/{summary['n_contracts']}",
        })
    summary_df = pd.DataFrame(table_rows)
    print(summary_df.to_string(index=False))

    summary_csv = args.output_dir / "accuracy_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote {summary_csv}")
    print("\nNote: confidentiality_clause has no CUAD gold category and is never")
    print("scored numerically -- spot-check it manually against source contracts.")


if __name__ == "__main__":
    main()
