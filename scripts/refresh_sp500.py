"""
Refresh data/sp500_companies.csv from Wikipedia's S&P 500 constituent list.
CIKs are left blank and resolved at ingestion time via EDGAR.

Usage:
    python scripts/refresh_sp500.py
"""

import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("data/sp500_companies.csv")


def refresh():
    import requests
    from io import StringIO
    print("Fetching S&P 500 list from Wikipedia...")
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (research project)"},
    )
    sp500 = pd.read_html(StringIO(resp.text))[0]
    df = sp500[["Symbol", "Security"]].copy()
    df.columns = ["ticker", "company_name"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["cik"] = ""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} companies to {OUTPUT_PATH}")


if __name__ == "__main__":
    refresh()
