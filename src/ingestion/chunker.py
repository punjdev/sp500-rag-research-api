"""
Paragraph-level chunker for 10-K sections.

Splits each section into ≤500-token chunks with one-paragraph overlap between
consecutive chunks to avoid cutting context at boundaries.
"""

import re
from dataclasses import dataclass

import tiktoken

CHUNK_MAX_TOKENS = 500
_encoder = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    ticker: str
    company_name: str
    section: str
    filing_date: str
    accession_number: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "section": self.section,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def _split_paragraphs(text: str) -> list[str]:
    # EDGAR HTML → plain text produces single newlines; fall back if no double newlines
    separator = r"\n{2,}" if "\n\n" in text else r"\n"
    paragraphs = re.split(separator, text)
    return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) >= 15]


def chunk_section(record: dict) -> list[Chunk]:
    """Split a section record into overlapping chunks. Returns list of Chunk objects."""
    raw_text = record["raw_text"]
    paragraphs = _split_paragraphs(raw_text)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_paragraphs: list[str] = []
    current_tokens = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal chunk_index
        chunk_text = "\n\n".join(current_paragraphs)
        char_start = raw_text.find(current_paragraphs[0])
        chunks.append(Chunk(
            ticker=record["ticker"],
            company_name=record["company_name"],
            section=record["section"],
            filing_date=record["filing_date"],
            accession_number=record["accession_number"],
            chunk_index=chunk_index,
            text=chunk_text,
            char_start=max(char_start, 0),
            char_end=char_start + len(chunk_text),
        ))
        chunk_index += 1

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if current_tokens + para_tokens > CHUNK_MAX_TOKENS and current_paragraphs:
            flush()
            # Carry last paragraph into the next chunk as overlap
            last = current_paragraphs[-1]
            current_paragraphs = [last]
            current_tokens = _count_tokens(last)
        current_paragraphs.append(para)
        current_tokens += para_tokens

    if current_paragraphs:
        flush()

    return chunks
