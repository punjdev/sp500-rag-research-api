"""
Interactive retrieval test — shows ranked results at each pipeline stage.

Usage:
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --query "What are Tesla's production risks?"
    python scripts/test_retrieval.py --query "Apple revenue" --ticker AAPL
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import bm25_index
from src.retrieval.dense import retrieve_dense
from src.retrieval.sparse import retrieve_sparse
from src.retrieval.fusion import hybrid_retrieve
from src.generation.rag import generate_answer


def main():
    parser = argparse.ArgumentParser(description="Test the RAG retrieval pipeline")
    parser.add_argument("--query", type=str, default="What are Apple's main cybersecurity risks?")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--no-generate", action="store_true", help="Skip generation, show only retrieved chunks")
    args = parser.parse_args()

    print(f"\nQuery: {args.query}")
    if args.ticker:
        print(f"Filter: {args.ticker}")
    print("=" * 60)

    # Load BM25 index
    if not bm25_index.load_index():
        print("ERROR: BM25 index not found. Run build_index.py first.")
        sys.exit(1)

    # --- Dense retrieval ---
    print("\n[1] Dense retrieval (top 5 of 50):")
    dense = retrieve_dense(args.query, ticker_filter=args.ticker)
    for i, chunk in enumerate(dense[:5]):
        print(f"  {i+1}. [{chunk['ticker']}] {chunk['section']} | score={chunk.get('score', 0):.3f}")
        print(f"     {chunk['text'][:120]}...")

    # --- Sparse retrieval ---
    print("\n[2] BM25 retrieval (top 5 of 50):")
    sparse = retrieve_sparse(args.query, ticker_filter=args.ticker)
    for i, chunk in enumerate(sparse[:5]):
        print(f"  {i+1}. [{chunk['ticker']}] {chunk['section']} | bm25={chunk.get('bm25_score', 0):.3f}")
        print(f"     {chunk['text'][:120]}...")

    # --- Hybrid: RRF + rerank ---
    print("\n[3] Hybrid retrieval (top 5 after RRF + rerank):")
    final = hybrid_retrieve(args.query, dense, sparse)
    for i, chunk in enumerate(final):
        print(f"  {i+1}. [{chunk['ticker']}] {chunk['section']} "
              f"| rerank={chunk.get('rerank_score', 0):.3f} rrf={chunk.get('rrf_score', 0):.4f}")
        print(f"     {chunk['text'][:200]}...")

    if args.no_generate:
        return

    # --- Generation ---
    print("\n[4] Generated answer:")
    print("-" * 60)
    result = generate_answer(args.query, final)
    print(result["answer"])

    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  [{c['ticker']}] {c['section']} ({c['filing_date'][:4]}): \"{c['cited_text'][:80]}...\"")


if __name__ == "__main__":
    main()
