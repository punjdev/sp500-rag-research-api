"""Dense retrieval: embed query → Qdrant cosine search."""

from src.indexing.embedder import embed_query
from src.indexing.qdrant_store import search_dense
from settings import settings


def retrieve_dense(
    query: str,
    top_k: int | None = None,
    ticker_filter: str | None = None,
) -> list[dict]:
    """
    Embed the query and search Qdrant for the top_k closest chunks.

    Returns a list of chunk dicts sorted by cosine similarity (highest first).
    """
    k = top_k or settings.dense_top_k
    query_vector = embed_query(query)
    return search_dense(query_vector, top_k=k, ticker_filter=ticker_filter)
