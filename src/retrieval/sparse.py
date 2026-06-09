"""Sparse retrieval: BM25 keyword search."""

from src.indexing import bm25_index
from settings import settings


def retrieve_sparse(
    query: str,
    top_k: int | None = None,
    ticker_filter: str | None = None,
) -> list[dict]:
    """
    Search the in-memory BM25 index for the top_k most relevant chunks.

    Returns a list of chunk dicts sorted by BM25 score (highest first).
    """
    k = top_k or settings.sparse_top_k
    return bm25_index.search(query, top_k=k, ticker_filter=ticker_filter)
