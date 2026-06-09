"""
Benchmark runner — records observable performance and behaviour data.

Unlike the RAGAS eval (which uses an LLM to score quality), this script
captures hard metrics you can quote: latency breakdowns, citation counts,
section distribution, answer length. Results are saved as JSON so you can
load them into a spreadsheet or notebook.

Usage:
    python scripts/benchmark.py                          # all 30 questions, hybrid
    python scripts/benchmark.py --n-questions 5          # quick smoke test
    python scripts/benchmark.py --config dense           # specific retrieval config
    python scripts/benchmark.py --config all             # compare all three configs
    python scripts/benchmark.py --query "What are Apple's risks?" --ticker AAPL
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
from src.indexing import bm25_index
from src.indexing.embedder import embed_query
from src.indexing.qdrant_store import search_dense
from src.retrieval.dense import retrieve_dense
from src.retrieval.sparse import retrieve_sparse
from src.retrieval.fusion import hybrid_retrieve
from src.generation.rag import generate_answer

TEST_SET_PATH = Path("src/evaluation/test_set.json")
RESULTS_DIR = Path("data/benchmark_results")


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def retrieve(query: str, config: str, ticker: str | None = None) -> tuple[list[dict], float]:
    """Run retrieval for the given config. Returns (chunks, latency_ms)."""
    t0 = time.time()
    if config == "dense":
        chunks = search_dense(embed_query(query), top_k=settings.final_top_k, ticker_filter=ticker)
    elif config == "bm25":
        chunks = retrieve_sparse(query, top_k=settings.final_top_k, ticker_filter=ticker)
    elif config == "hybrid":
        dense = retrieve_dense(query, ticker_filter=ticker)
        sparse = retrieve_sparse(query, ticker_filter=ticker)
        chunks = hybrid_retrieve(query, dense, sparse)
    else:
        raise ValueError(f"Unknown config: {config}")
    return chunks, (time.time() - t0) * 1000


# ---------------------------------------------------------------------------
# Single query benchmark
# ---------------------------------------------------------------------------

def run_one(question: str, ticker: str | None, config: str, metadata: dict | None = None) -> dict:
    """Run a single query end-to-end and return a result record."""
    total_start = time.time()

    chunks, retrieval_ms = retrieve(question, config, ticker)

    gen_start = time.time()
    result = generate_answer(question, chunks) if chunks else {"answer": "", "citations": []}
    generation_ms = (time.time() - gen_start) * 1000

    total_ms = (time.time() - total_start) * 1000

    citations = result.get("citations", [])
    sections = [c["section"] for c in citations]
    tickers_cited = list({c["ticker"] for c in citations})
    section_counts = dict(Counter(sections))

    record = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "question": question,
        "ticker_filter": ticker,
        # Timing
        "total_ms": round(total_ms),
        "retrieval_ms": round(retrieval_ms),
        "generation_ms": round(generation_ms),
        # Retrieval
        "n_chunks": len(chunks),
        # Citations
        "n_citations": len(citations),
        "tickers_cited": sorted(tickers_cited),
        "sections_cited": sections,
        "section_counts": section_counts,
        # Answer
        "answer_length": len(result.get("answer", "")),
        "answer": result.get("answer", ""),
        # Optional metadata from test set
        **(metadata or {}),
    }
    return record


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(questions: list[dict], config: str) -> list[dict]:
    results = []
    print(f"\n{'─'*60}")
    print(f"  Config: {config.upper()}  |  {len(questions)} questions")
    print(f"{'─'*60}")

    for i, item in enumerate(questions):
        q = item["question"]
        ticker = item.get("expected_tickers", [None])[0] if item.get("expected_tickers") else None
        print(f"  [{i+1:02d}/{len(questions)}] {q[:65]}{'…' if len(q)>65 else ''}", end="", flush=True)

        record = run_one(
            question=q,
            ticker=None,   # no ticker filter in batch — tests open retrieval
            config=config,
            metadata={
                "id": item.get("id"),
                "category": item.get("category"),
                "expected_tickers": item.get("expected_tickers", []),
                "expected_sections": item.get("expected_sections", []),
            },
        )
        results.append(record)
        print(f"  {record['total_ms']}ms  |  {record['n_citations']} citations  |  {record['section_counts']}")
        # Cohere trial: 10 rerank + 10 generate calls/min each.
        # Sleep 7s between questions to stay safely under the limit.
        if i < len(questions) - 1:
            time.sleep(7)

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], config: str) -> None:
    if not results:
        return

    latencies = [r["total_ms"] for r in results]
    citation_counts = [r["n_citations"] for r in results]
    answer_lengths = [r["answer_length"] for r in results]
    all_sections = [s for r in results for s in r["sections_cited"]]
    section_dist = Counter(all_sections)

    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r["total_ms"])

    print(f"\n{'═'*60}")
    print(f"  SUMMARY — {config.upper()}")
    print(f"{'═'*60}")
    print(f"  Questions run:     {len(results)}")
    print(f"  Avg latency:       {sum(latencies)/len(latencies):.0f}ms")
    print(f"  Min / Max:         {min(latencies)}ms / {max(latencies)}ms")
    print(f"  Avg citations:     {sum(citation_counts)/len(citation_counts):.1f}")
    print(f"  Avg answer length: {sum(answer_lengths)/len(answer_lengths):.0f} chars")
    print(f"  Section breakdown: risk_factors={section_dist.get('risk_factors',0)}  mda={section_dist.get('mda',0)}")
    if categories:
        print(f"  Latency by category:")
        for cat, times in sorted(categories.items()):
            print(f"    {cat:<18} avg={sum(times)/len(times):.0f}ms  n={len(times)}")
    print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(all_results: dict[str, list[dict]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = RESULTS_DIR / f"benchmark_{ts}.json"
    payload = {
        "run_at": datetime.now().isoformat(),
        "configs": list(all_results.keys()),
        "total_queries": sum(len(v) for v in all_results.values()),
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the RAG pipeline.")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit to first N questions from the test set.")
    parser.add_argument("--config", choices=["dense", "bm25", "hybrid", "all"],
                        default="hybrid", help="Retrieval config to benchmark.")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single ad-hoc query instead of the test set.")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Ticker filter for --query mode.")
    parser.add_argument("--no-save", action="store_true",
                        help="Print results but don't save to disk.")
    args = parser.parse_args()

    if not bm25_index.load_index():
        print("ERROR: BM25 index not found. Run scripts/build_index.py first.")
        sys.exit(1)

    # ── Single ad-hoc query ──────────────────────────────────────────────────
    if args.query:
        config = "hybrid" if args.config == "all" else args.config
        print(f"\nQuery: {args.query}")
        print(f"Ticker: {args.ticker or 'none'} | Config: {config}")
        print("─" * 60)
        r = run_one(args.query, args.ticker, config)
        print(f"Answer ({r['answer_length']} chars):\n{r['answer']}\n")
        print(f"Citations: {r['n_citations']}  |  Sections: {r['section_counts']}")
        print(f"Latency: {r['total_ms']}ms  (retrieval={r['retrieval_ms']}ms  generation={r['generation_ms']}ms)")
        return

    # ── Batch from test set ──────────────────────────────────────────────────
    questions = json.loads(TEST_SET_PATH.read_text())["questions"]
    if args.n_questions:
        questions = questions[: args.n_questions]
    print(f"\nLoaded {len(questions)} questions from test set.")

    configs = ["dense", "bm25", "hybrid"] if args.config == "all" else [args.config]
    all_results: dict[str, list[dict]] = {}

    for config in configs:
        results = run_batch(questions, config)
        print_summary(results, config)
        all_results[config] = results

    if not args.no_save:
        out_path = save_results(all_results)
        print(f"Results saved → {out_path}")
    else:
        print("(--no-save: results not written to disk)")


if __name__ == "__main__":
    main()
