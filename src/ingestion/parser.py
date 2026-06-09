"""
HTML parser for SEC 10-K filings.

Extracts Item 1A (Risk Factors) and Item 7 (MD&A) from the raw HTML.
Uses regex on plain text rather than DOM traversal — more robust across
the wide variety of HTML structures EDGAR filings use.
"""

import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_SECTION_STARTS = {
    "risk_factors": [r"ITEM\s+1A[\.\s—–\-]", r"ITEM\s+1A$"],
    "mda":          [r"ITEM\s+7[\.\s—–\-](?!A)", r"ITEM\s+7$"],
}

_SECTION_ENDS = {
    "risk_factors": [r"ITEM\s+1B[\.\s—–\-]", r"ITEM\s+2[\.\s—–\-]"],
    "mda":          [r"ITEM\s+7A[\.\s—–\-]", r"ITEM\s+8[\.\s—–\-]"],
}

MAX_SECTION_CHARS = 150_000
# Minimum to skip Table of Contents entries and short forward-looking disclaimers
MIN_SECTION_CHARS = 5_000


def _html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=False)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_section(full_text: str, start_patterns: list[str], end_patterns: list[str]) -> str:
    """
    Find a section by trying each start-pattern match in document order.
    Skips TOC entries (they produce <MIN_SECTION_CHARS of content).
    """
    upper = full_text.upper()

    start_positions: list[int] = sorted(
        m.start()
        for pattern in start_patterns
        for m in re.finditer(pattern, upper)
    )

    if not start_positions:
        return ""

    for start_pos in start_positions:
        search_from = start_pos + 100
        end_pos = len(full_text)
        for pattern in end_patterns:
            m = re.search(pattern, upper[search_from:])
            if m:
                end_pos = search_from + m.start()
                break
        section_text = full_text[start_pos:end_pos].strip()
        if len(section_text) >= MIN_SECTION_CHARS:
            return section_text[:MAX_SECTION_CHARS]

    return ""


def parse_10k_sections(
    html: str,
    ticker: str,
    company_name: str,
    cik: str,
    filing_date: str,
    accession_number: str,
) -> list[dict]:
    """Extract Item 1A and Item 7 from a 10-K HTML document. Returns 0–2 section dicts."""
    full_text = _html_to_clean_text(html)
    results = []

    for section_key in ("risk_factors", "mda"):
        text = _extract_section(full_text, _SECTION_STARTS[section_key], _SECTION_ENDS[section_key])
        if not text:
            print(f"  WARNING: Could not extract {section_key} for {ticker}")
            continue
        results.append({
            "ticker": ticker,
            "company_name": company_name,
            "cik": str(cik),
            "filing_date": filing_date,
            "accession_number": accession_number,
            "section": section_key,
            "raw_text": text,
        })

    return results
