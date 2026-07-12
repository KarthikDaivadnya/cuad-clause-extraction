#!/usr/bin/env python3
"""
main.py
-------
CLI entry point. Run the whole pipeline on a folder of CUAD contract PDFs.

Examples:
    # Smoke-test on 5 contracts with Groq's free/fast model
    python main.py --data-dir data/full_contract_pdf --limit 5

    # Full assignment run (50 contracts), explicit provider/model
    python main.py --data-dir data/full_contract_pdf --limit 50 \\
        --provider groq --model llama-3.3-70b-versatile

    # Re-run ignoring the cache (e.g. after editing a prompt)
    python main.py --data-dir data/full_contract_pdf --limit 50 --no-cache
"""
import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import run_pipeline
from src import config


def parse_args():
    p = argparse.ArgumentParser(description="CUAD contract clause extraction & summarization pipeline")
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "full_contract_pdf",
                    help="Folder containing CUAD contract PDFs (searched recursively).")
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR,
                    help="Where to write results.csv / results.json")
    p.add_argument("--limit", type=int, default=50, help="Max number of contracts to process (default: 50).")
    p.add_argument("--provider", type=str, default=None, choices=["groq", "openai", "anthropic"],
                    help="LLM provider to use (default: from LLM_PROVIDER env var, else groq).")
    p.add_argument("--model", type=str, default=None, help="Override the default model for the chosen provider.")
    p.add_argument("--no-cache", action="store_true", help="Ignore cached per-contract results.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.data_dir.exists():
        print(
            f"\n[!] Data directory not found: {args.data_dir}\n"
            f"    Run `python scripts/download_cuad.py` first, or pass --data-dir "
            f"pointing at your CUAD 'full_contract_pdf' folder.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    df = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        provider_name=args.provider,
        model=args.model,
        use_cache=not args.no_cache,
    )

    n_ok = df["error"].eq("").sum() if "error" in df.columns else len(df)
    print(f"\nDone. Processed {len(df)} contract(s), {n_ok} succeeded.")
    print(f"Results written to: {args.output_dir / 'results.csv'} and results.json")


if __name__ == "__main__":
    main()
