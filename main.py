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

    # Skip the automatic accuracy check (e.g. no internet access to fetch
    # CUAD's gold labels, or you just want the raw extraction run)
    python main.py --data-dir data/full_contract_pdf --limit 50 --skip-accuracy
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
    p.add_argument("--skip-accuracy", action="store_true",
                    help="Skip the automatic accuracy check against CUAD's gold labels after the run.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def print_accuracy_report(results_json_path: Path, provider_label: str):
    """Scores this run's output against CUAD's own expert gold labels and
    prints the summary straight to screen -- reuses scripts/evaluate_accuracy.py
    unmodified, so there is exactly one implementation of this logic."""
    try:
        from scripts.evaluate_accuracy import (
            build_gold_lookup, evaluate_run, load_cuad_gold, summarize_run,
        )
    except ImportError as exc:
        print(f"\n[!] Skipping accuracy check -- couldn't import evaluate_accuracy.py ({exc})")
        return

    print("\n" + "=" * 78)
    print("ACCURACY CHECK  (against CUAD's own expert-labeled gold data)")
    print("=" * 78)
    try:
        cuad_data = load_cuad_gold(config.DATA_DIR / ".cuad_cache")
        gold_lookup = build_gold_lookup(cuad_data)
        df = evaluate_run(results_json_path, gold_lookup)
        summary = summarize_run(df)

        report_csv = results_json_path.parent / f"accuracy_report_{provider_label}.csv"
        df.to_csv(report_csv, index=False)

        print(f"Matched {summary['n_matched']}/{summary['n_contracts']} contracts to CUAD gold labels.\n")
        print(f"  termination_clause:      {summary['termination_clause']}")
        print(f"  liability_clause:        {summary['liability_clause']}")
        print(f"  confidentiality_clause:  {summary['confidentiality_clause']}")
        print(f"\nPer-contract detail written to: {report_csv}")
    except Exception as exc:  # noqa: BLE001 -- never let this crash a successful pipeline run
        print(f"[!] Accuracy check failed ({type(exc).__name__}: {exc}) -- skipping.")
        print("    Your extraction results above are unaffected; run")
        print("    `python scripts/evaluate_accuracy.py` manually to retry.")


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

    if not args.skip_accuracy:
        provider_label = args.provider or config.DEFAULT_PROVIDER
        print_accuracy_report(args.output_dir / "results.json", provider_label)


if __name__ == "__main__":
    main()