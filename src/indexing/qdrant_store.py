"""
Qdrant vector store client.

Collection: sp500_10k — 1024-dim cosine vectors, keyword payload index on ticker.
Each point stores the vector plus a payload containing the full chunk text and metadata.
"""

import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from settings import settings
from src.ingestion.chunker import Chunk

# Module-level client — reused across all calls
_client: QdrantClient | None = None

# How many points to send per Qdrant upsert call.
# Larger batches = fewer round trips = faster indexing.
UPSERT_BATCH_SIZE = 256


def get_client() -> QdrantClient:
    """Return (or create) the Qdrant client singleton."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


def ensure_collection() -> None:
    """Create the collection and ticker keyword index if they don't exist. Idempotent."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {settings.qdrant_collection}")
    else:
        print(f"Collection already exists: {settings.qdrant_collection}")

    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="ticker",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("Ticker payload index ensured.")


def _chunk_id(ticker: str, section: str, chunk_index: int) -> int:
    """Deterministic integer ID — same chunk always gets the same ID (upsert is idempotent)."""
    key = f"{ticker}_{section}_{chunk_index}"
    return int(hashlib.md5(key.encode()).hexdigest()[:16], 16) % (2**63)


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Store chunks and embeddings in Qdrant. Full text stored in payload for BM25 rebuild."""
    client = get_client()
    points = [
        PointStruct(
            id=_chunk_id(c.ticker, c.section, c.chunk_index),
            vector=embedding,
            payload={
                "ticker": c.ticker,
                "company_name": c.company_name,
                "section": c.section,
                "filing_date": c.filing_date,
                "accession_number": c.accession_number,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "char_start": c.char_start,
                "char_end": c.char_end,
            },
        )
        for c, embedding in zip(chunks, embeddings)
    ]

    # Upsert in batches
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)


def get_collection_info() -> dict:
    """Return basic stats about the collection (used by /health endpoint)."""
    client = get_client()
    info = client.get_collection(settings.qdrant_collection)
    return {
        "status": info.status,
        "chunk_count": info.points_count,
        "vector_size": info.config.params.vectors.size,
    }


def scroll_all_chunks(batch_size: int = 1000) -> list[dict]:
    """Retrieve all chunk payloads (used to rebuild BM25 index without re-fetching EDGAR)."""
    client = get_client()
    all_payloads: list[dict] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_payloads.extend(r.payload for r in results)
        offset = next_offset
        if next_offset is None:
            break

    return all_payloads


def search_dense(
    query_vector: list[float],
    top_k: int = 50,
    ticker_filter: str | None = None,
) -> list[dict]:
    """Cosine similarity search. Optional ticker_filter restricts to one company."""
    client = get_client()

    query_filter = None
    if ticker_filter:
        query_filter = Filter(
            must=[FieldCondition(key="ticker", match=MatchValue(value=ticker_filter.upper()))]
        )

    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "id": r.id,
            "score": r.score,
            **r.payload,
        }
        for r in results
    ]
