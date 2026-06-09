"""
Hybrid retrieval: Reciprocal Rank Fusion + Cohere reranking.

Pipeline: dense (top 50) + BM25 (top 50) → RRF → top 20 → reranker → top 5.
RRF formula: score(d) = Σ 1/(k + rank_i(d)), k=60 (Cormack 2009).
"""

import cohere
from settings import settings

_cohere_client = cohere.Client(api_key=settings.cohere_api_key)
RRF_K = 60


def _chunk_key(chunk: dict) -> str:
    return f"{chunk['ticker']}_{chunk['section']}_{chunk['chunk_index']}"


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """Merge two ranked lists via RRF. Chunks in both lists score higher."""
    scores: dict[str, float] = {}
    chunks_by_key: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunks_by_key[key] = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunks_by_key[key] = chunk

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [{**chunks_by_key[k], "rrf_score": scores[k]} for k in sorted_keys[:top_k]]


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """Cross-encoder reranking on the top RRF candidates."""
    if not candidates:
        return []
    response = _cohere_client.rerank(
        query=query,
        documents=[c["text"] for c in candidates],
        model=settings.cohere_rerank_model,
        top_n=top_n,
    )
    return [{**candidates[r.index], "rerank_score": r.relevance_score} for r in response.results]


def hybrid_retrieve(
    query: str,
    dense_results: list[dict],
    sparse_results: list[dict],
    rerank_top_k: int | None = None,
    final_top_k: int | None = None,
) -> list[dict]:
    """RRF fusion then reranking. Returns final chunks for the LLM."""
    fused = reciprocal_rank_fusion(
        dense_results, sparse_results, top_k=rerank_top_k or settings.rerank_top_k
    )
    return rerank(query, fused, top_n=final_top_k or settings.final_top_k)
