"""
SEC EDGAR API client.

Fetches 10-K filing metadata and HTML documents from EDGAR.
Sleeps 0.1s between requests to stay under the 10 req/s rate limit.
"""

import json
import time
import requests
from pathlib import Path

# SEC requires a descriptive User-Agent — they will block generic ones.
# Format: "Name email@domain.com"
USER_AGENT = "Dev Punjabi dev@devpunjabi.com"
REQUEST_DELAY = 0.1  # seconds between EDGAR requests

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _get(url: str) -> requests.Response:
    """Rate-limited GET with EDGAR-required headers."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp


def load_ticker_cik_map() -> dict[str, str]:
    """Download EDGAR's full ticker → CIK mapping."""
    data = _get(EDGAR_TICKERS_URL).json()
    return {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in data.values()
    }


def get_latest_10k(cik: str) -> dict:
    """Return metadata for the company's most recent 10-K filing. Raises ValueError if not found."""
    cik_padded = str(cik).zfill(10)
    data = _get(EDGAR_SUBMISSIONS_URL.format(cik=cik_padded)).json()

    recent = data["filings"]["recent"]
    idx = next((i for i, f in enumerate(recent["form"]) if f == "10-K"), None)
    if idx is None:
        raise ValueError(f"No 10-K found in recent filings for CIK {cik}")

    accession_raw = recent["accessionNumber"][idx]
    accession_clean = accession_raw.replace("-", "")
    primary_doc = recent["primaryDocument"][idx]
    filing_date = recent["filingDate"][idx]

    filing_url = EDGAR_ARCHIVES_URL.format(
        cik=str(int(cik)),
        accession=accession_clean,
        doc=primary_doc,
    )

    return {
        "cik": cik,
        "filing_date": filing_date,
        "accession_number": accession_raw,
        "primary_document": primary_doc,
        "filing_url": filing_url,
    }


def fetch_filing_html(filing_url: str) -> str:
    """Fetch the raw HTML of a 10-K filing document."""
    return _get(filing_url).text
