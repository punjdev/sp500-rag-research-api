"""
BM25 sparse retrieval index.

Complements dense vector search — BM25 handles exact keyword matches
(tickers, financial terms) that embedding models may score poorly.

The index is persisted to disk and loaded into memory at API startup.
If pickle files are missing, the API rebuilds from Qdrant payloads.
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from settings import settings

_bm25: BM25Okapi | None = None
_chunks: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in tokens if t]


def build_index(chunk_payloads: list[dict]) -> tuple[BM25Okapi, list[dict]]:
    tokenized_corpus = [_tokenize(c["text"]) for c in chunk_payloads]
    return BM25Okapi(tokenized_corpus), chunk_payloads


def save_index(bm25: BM25Okapi, chunks: list[dict]) -> None:
    Path(settings.bm25_index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(settings.bm25_index_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(settings.bm25_chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"BM25 index saved: {len(chunks)} chunks")


def load_index() -> bool:
    """Load from disk into module state. Returns True on success."""
    global _bm25, _chunks
    if Path(settings.bm25_index_path).exists() and Path(settings.bm25_chunks_path).exists():
        with open(settings.bm25_index_path, "rb") as f:
            _bm25 = pickle.load(f)
        with open(settings.bm25_chunks_path, "rb") as f:
            _chunks = pickle.load(f)
        print(f"BM25 index loaded: {len(_chunks)} chunks")
        return True
    return False


def search(query: str, top_k: int = 50, ticker_filter: str | None = None) -> list[dict]:
    if _bm25 is None or _chunks is None:
        raise RuntimeError("BM25 index not loaded. Call load_index() first.")

    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        if score <= 0:
            break
        chunk = _chunks[idx]
        if ticker_filter and chunk.get("ticker", "").upper() != ticker_filter.upper():
            continue
        results.append({**chunk, "bm25_score": float(score)})
        if len(results) >= top_k:
            break

    return results
