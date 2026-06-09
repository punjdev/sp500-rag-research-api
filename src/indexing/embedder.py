"""
Cohere embedding wrapper.

Uses embed-english-v3.0 (1024-dim). Documents must be embedded with
input_type="search_document" and queries with input_type="search_query" —
using the wrong type degrades retrieval accuracy.

Rate limiting: batch size 20 + 15s sleep keeps within Cohere's trial plan
limits. Increase EMBED_BATCH_SIZE and reduce EMBED_SLEEP_SECONDS on a paid plan.
"""

import time
import cohere
from cohere.errors import TooManyRequestsError
from settings import settings

_client = cohere.Client(api_key=settings.cohere_api_key)

EMBED_BATCH_SIZE = 20
EMBED_SLEEP_SECONDS = 15


def _embed_with_retry(texts: list[str], input_type: str) -> list[list[float]]:
    for attempt in range(4):
        try:
            response = _client.embed(
                texts=texts,
                model=settings.cohere_embed_model,
                input_type=input_type,
                embedding_types=["float"],
            )
            return response.embeddings.float_
        except TooManyRequestsError:
            if attempt == 3:
                raise
            print(f"\n  Rate limited — waiting 65s (attempt {attempt + 1}/4)...")
            time.sleep(65)
    return []


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document texts. Batched and rate-limited."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        all_embeddings.extend(_embed_with_retry(batch, input_type="search_document"))
        time.sleep(EMBED_SLEEP_SECONDS)
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single user query for retrieval."""
    return _embed_with_retry([query], input_type="search_query")[0]
