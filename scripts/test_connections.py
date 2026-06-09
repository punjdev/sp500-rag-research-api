"""
Quick smoke test — verifies all three external services are reachable.
Run this before doing any ingestion.

Usage: python scripts/test_connections.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings

def test_cohere():
    print("Testing Cohere...", end=" ", flush=True)
    import cohere
    co = cohere.Client(api_key=settings.cohere_api_key)
    resp = co.embed(
        texts=["hello world"],
        model=settings.cohere_embed_model,
        input_type="search_query",
        embedding_types=["float"],
    )
    vec = resp.embeddings.float_[0]
    assert len(vec) == 1024, f"Expected 1024-dim vector, got {len(vec)}"
    print(f"OK  (1024-dim embedding returned)")

def test_qdrant():
    print("Testing Qdrant...", end=" ", flush=True)
    from qdrant_client import QdrantClient
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    colls = client.get_collections()
    names = [c.name for c in colls.collections]
    print(f"OK  (connected, collections: {names or 'none yet'})")

def test_langfuse():
    print("Testing Langfuse...", end=" ", flush=True)
    from langfuse import Langfuse
    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    # auth_check() returns True if the keys are valid
    ok = lf.auth_check()
    assert ok, "Langfuse auth check failed — check your keys and host URL"
    lf.flush()
    print("OK  (authenticated)")

if __name__ == "__main__":
    print("=" * 50)
    print("Connection tests")
    print("=" * 50)
    errors = []
    for name, fn in [("Cohere", test_cohere), ("Qdrant", test_qdrant), ("Langfuse", test_langfuse)]:
        try:
            fn()
        except Exception as e:
            print(f"FAILED: {e}")
            errors.append(name)

    print("=" * 50)
    if errors:
        print(f"Failed: {errors}")
        sys.exit(1)
    else:
        print("All connections OK — ready to run ingestion.")
