"""
Rebuild the BM25 index from Qdrant payloads.

Use this when:
- You need to rebuild the BM25 index without re-running full ingestion.
- The BM25 pickle files were deleted (e.g., after a Railway deploy).
- You added new companies to Qdrant and want to update the BM25 index.

Usage:
    python scripts/build_bm25.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.qdrant_store import scroll_all_chunks
from src.indexing import bm25_index


def main():
    print("Fetching all chunk payloads from Qdrant...")
    chunks = scroll_all_chunks()

    if not chunks:
        print("No chunks found in Qdrant. Run build_index.py first.")
        sys.exit(1)

    print(f"Fetched {len(chunks):,} chunks. Building BM25 index...")
    index, chunk_list = bm25_index.build_index(chunks)
    bm25_index.save_index(index, chunk_list)
    print("Done.")


if __name__ == "__main__":
    main()
