"""
Build Qdrant vector and BM25 indexes from raw section data.

Run after scripts/ingest.py has populated data/raw/.

Usage:
    python scripts/build_index.py              # index everything in data/raw/
    python scripts/build_index.py --ticker AAPL
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.ingestion.chunker import chunk_section, Chunk
from src.indexing.embedder import embed_documents
from src.indexing.qdrant_store import ensure_collection, upsert_chunks
from src.indexing import bm25_index

RAW_DIR = Path("data/raw")
INDEX_BATCH_SIZE = 500


def load_raw_sections(ticker_filter: str | None = None) -> list[dict]:
    pattern = f"{ticker_filter.upper()}_*.json" if ticker_filter else "*.json"
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        print(f"No raw files found in {RAW_DIR}/")
        sys.exit(1)
    records = [json.loads(p.read_text()) for p in files]
    print(f"Loaded {len(records)} sections from {len(set(r['ticker'] for r in records))} companies")
    return records


def main():
    parser = argparse.ArgumentParser(description="Build Qdrant and BM25 indexes")
    parser.add_argument("--ticker", type=str, help="Index only this ticker")
    args = parser.parse_args()

    ensure_collection()
    records = load_raw_sections(args.ticker)

    # Chunk
    print("\nChunking sections...")
    all_chunks: list[Chunk] = []
    for record in tqdm(records, desc="Chunking", unit="section"):
        chunks = chunk_section(record)
        all_chunks.extend(chunks)
        if not chunks:
            print(f"  WARNING: 0 chunks from {record['ticker']} {record['section']}")
    print(f"Total chunks: {len(all_chunks):,}")

    # Embed + upsert to Qdrant
    print(f"\nEmbedding and indexing {len(all_chunks):,} chunks...")
    all_chunk_dicts: list[dict] = []

    for i in tqdm(range(0, len(all_chunks), INDEX_BATCH_SIZE), desc="Batches", unit="batch"):
        batch = all_chunks[i : i + INDEX_BATCH_SIZE]
        embeddings = embed_documents([c.text for c in batch])
        upsert_chunks(batch, embeddings)
        all_chunk_dicts.extend(c.to_dict() for c in batch)

    print(f"Qdrant: {len(all_chunk_dicts):,} chunks stored")

    # Build BM25
    print("\nBuilding BM25 index...")
    bm25, chunks = bm25_index.build_index(all_chunk_dicts)
    bm25_index.save_index(bm25, chunks)

    print(f"\nDone — {len(all_chunk_dicts):,} chunks in Qdrant and BM25")


if __name__ == "__main__":
    main()
