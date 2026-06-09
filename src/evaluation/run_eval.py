"""
RAGAS evaluation runner.

Evaluates the RAG pipeline across three retrieval configurations:
  dense   — Qdrant cosine search, top-5
  bm25    — BM25 keyword search, top-5
  hybrid  — dense + BM25 → RRF → Cohere rerank → top-5

Metrics (0–1, higher is better): faithfulness, context_precision,
context_recall, answer_relevancy. Uses Cohere as the judge LLM.

Usage:
    python src/evaluation/run_eval.py
    python src/evaluation/run_eval.py --n-questions 5
    python src/evaluation/run_eval.py --config hybrid
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from langchain_cohere import ChatCohere, CohereEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from settings import settings
from src.indexing import bm25_index
from src.indexing.qdrant_store import search_dense
from src.indexing.embedder import embed_query
from src.retrieval.sparse import retrieve_sparse
from src.retrieval.fusion import hybrid_retrieve
from src.generation.rag import generate_answer

TEST_SET_PATH = Path("src/evaluation/test_set.json")
RESULTS_PATH = Path("evaluation_results.md")

RetrievalConfig = Literal["dense", "bm25", "hybrid"]


def configure_ragas() -> None:
    """Configure RAGAS to use Cohere instead of OpenAI."""
    llm = LangchainLLMWrapper(
        ChatCohere(cohere_api_key=settings.cohere_api_key, model=settings.cohere_generation_model)
    )
    embeddings = LangchainEmbeddingsWrapper(
        CohereEmbeddings(cohere_api_key=settings.cohere_api_key, model=settings.cohere_embed_model)
    )
    for metric in [faithfulness, context_precision, context_recall, answer_relevancy]:
        metric.llm = llm
    answer_relevancy.embeddings = embeddings


def retrieve(query: str, config: RetrievalConfig, ticker_filter: str | None = None) -> list[dict]:
    if config == "dense":
        return search_dense(embed_query(query), top_k=5, ticker_filter=ticker_filter)
    if config == "bm25":
        return retrieve_sparse(query, top_k=5, ticker_filter=ticker_filter)
    if config == "hybrid":
        from src.retrieval.dense import retrieve_dense
        return hybrid_retrieve(query, retrieve_dense(query), retrieve_sparse(query))
    raise ValueError(f"Unknown config: {config}")


def run_evaluation(questions: list[dict], config: RetrievalConfig) -> dict:
    print(f"\n{'='*60}\n{config.upper()} — {len(questions)} questions\n{'='*60}")
    eval_rows = []

    for i, item in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {item['question'][:60]}...")
        chunks = retrieve(item["question"], config)
        if not chunks:
            print("    WARNING: no chunks retrieved — skipping")
            continue
        result = generate_answer(item["question"], chunks)
        eval_rows.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in chunks],
            "ground_truth": item["ground_truth"],
        })
        # Cohere trial tier: 10 rerank + 10 generate calls/min each.
        # Sleep 7s between questions to stay safely under the limit.
        if i < len(questions) - 1:
            time.sleep(7)

    if not eval_rows:
        return {}

    scores = evaluate(
        Dataset.from_list(eval_rows),
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
    )
    return dict(scores)


def format_results_table(results: dict[str, dict]) -> str:
    metrics = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    configs = ["dense", "bm25", "hybrid"]
    lines = [
        "# RAG Evaluation Results\n",
        "| Metric | Dense | BM25 | Hybrid + Rerank |",
        "|--------|-------|------|-----------------|",
    ]
    for metric in metrics:
        row = f"| {metric.replace('_', ' ').title()} |"
        for config in configs:
            val = results.get(config, {}).get(metric)
            row += f" {val:.3f} |" if val is not None else " N/A |"
        lines.append(row)
    lines.extend([
        "",
        "Metrics scored 0–1 (higher is better). Judge LLM: Cohere Command R+.",
        "- **Faithfulness**: claims in answer supported by retrieved context",
        "- **Context Precision**: retrieved chunks relevant to the question",
        "- **Context Recall**: retrieved chunks cover the ground-truth answer",
        "- **Answer Relevancy**: answer addresses what was asked",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--config", choices=["dense", "bm25", "hybrid", "all"], default="all")
    args = parser.parse_args()

    if not bm25_index.load_index():
        print("ERROR: BM25 index not found. Run build_index.py first.")
        sys.exit(1)

    configure_ragas()

    questions = json.loads(TEST_SET_PATH.read_text())["questions"]
    if args.n_questions:
        questions = questions[:args.n_questions]
    print(f"Loaded {len(questions)} questions")

    configs: list[RetrievalConfig] = ["dense", "bm25", "hybrid"] if args.config == "all" else [args.config]
    all_results: dict[str, dict] = {}

    for config in configs:
        scores = run_evaluation(questions, config)
        all_results[config] = scores
        print(f"\n{config.upper()}: {scores}")

    table = format_results_table(all_results)
    RESULTS_PATH.write_text(table)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("\n" + table)


if __name__ == "__main__":
    main()
