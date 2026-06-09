"""
Fetch and parse 10-K filings from SEC EDGAR.

Reads data/sp500_companies.csv, downloads Item 1A and Item 7 for each company,
and saves the raw text to data/raw/. Checkpoint-aware: re-running skips
companies that already completed.

Usage:
    python scripts/ingest.py                  # all companies in CSV
    python scripts/ingest.py --ticker AAPL   # single company
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from src.ingestion.edgar import get_latest_10k, fetch_filing_html, load_ticker_cik_map
from src.ingestion.parser import parse_10k_sections

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CHECKPOINT_FILE = DATA_DIR / "checkpoints" / "ingestion_progress.json"
SP500_CSV = DATA_DIR / "sp500_companies.csv"


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()).get("completed", []))
    return set()


def save_checkpoint(completed: set[str]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps({"completed": sorted(completed)}, indent=2))


def save_section(record: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{record['ticker']}_{record['section']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


def process_company(ticker: str, company_name: str, cik: str) -> bool:
    print(f"\n[{ticker}] {company_name}")
    try:
        print("  Fetching filing metadata...")
        filing_meta = get_latest_10k(cik)
        print(f"  10-K filed {filing_meta['filing_date']} — {filing_meta['filing_url']}")

        print("  Downloading document...")
        html = fetch_filing_html(filing_meta["filing_url"])

        print("  Parsing sections...")
        sections = parse_10k_sections(
            html=html,
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            filing_date=filing_meta["filing_date"],
            accession_number=filing_meta["accession_number"],
        )

        if not sections:
            print(f"  ERROR: No sections extracted — skipping.")
            return False

        for record in sections:
            save_section(record)
            print(f"  Saved {record['section']}: {len(record['raw_text']):,} chars")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ingest S&P 500 10-K filings from EDGAR")
    parser.add_argument("--ticker", type=str, help="Process a single ticker (e.g. AAPL)")
    args = parser.parse_args()

    df = pd.read_csv(SP500_CSV, dtype=str).fillna("")

    if args.ticker:
        df = df[df["ticker"].str.upper() == args.ticker.upper()]
        if df.empty:
            print(f"Ticker {args.ticker} not found in {SP500_CSV}")
            sys.exit(1)
    else:
        print(f"Processing all {len(df)} companies")

    completed = load_checkpoint()
    if completed:
        print(f"Checkpoint: {len(completed)} already done, skipping.")

    # Resolve missing CIKs from EDGAR
    missing_cik = df[df["cik"].str.strip() == ""]
    if not missing_cik.empty:
        print(f"Resolving {len(missing_cik)} CIKs from EDGAR...")
        cik_map = load_ticker_cik_map()
        for idx, row in missing_cik.iterrows():
            t = row["ticker"].upper()
            if t in cik_map:
                df.at[idx, "cik"] = cik_map[t]
            else:
                print(f"  WARNING: CIK not found for {t}, skipping.")

    success_count = skip_count = fail_count = 0

    for row in tqdm(list(df.itertuples(index=False)), desc="Companies", unit="co"):
        ticker = row.ticker.upper()

        if not row.cik:
            print(f"\n[{ticker}] No CIK — skipping.")
            fail_count += 1
            continue

        if ticker in completed:
            skip_count += 1
            continue

        if process_company(ticker, row.company_name, row.cik):
            completed.add(ticker)
            save_checkpoint(completed)
            success_count += 1
        else:
            fail_count += 1

    print(f"\nDone — processed: {success_count}, skipped: {skip_count}, failed: {fail_count}")
    print(f"Raw files: {RAW_DIR}/")


if __name__ == "__main__":
    main()
