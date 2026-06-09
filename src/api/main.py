"""
FastAPI application — S&P 500 10-K RAG API.

Endpoints:
  POST /query      — hybrid RAG pipeline: answer + citations
  GET  /companies  — list indexed companies (used by frontend dropdown)
  GET  /health     — liveness / readiness probe
  GET  /           — test UI (development only, disabled in production)
"""

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from langfuse import Langfuse

from settings import settings
from src.api.models import (
    QueryRequest, QueryResponse, Citation,
    CompaniesResponse, Company, HealthResponse,
)
from src.indexing import bm25_index
from src.indexing.qdrant_store import scroll_all_chunks, get_collection_info, ensure_collection
from src.retrieval.dense import retrieve_dense
from src.retrieval.sparse import retrieve_sparse
from src.retrieval.fusion import hybrid_retrieve
from src.generation.rag import generate_answer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_sp500_all: dict[str, str] = {}           # ticker → company_name (full S&P 500)
_sp500_name_to_ticker: dict[str, str] = {}

limiter = Limiter(key_func=get_remote_address)

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)

# ---------------------------------------------------------------------------
# Company detection helpers
# ---------------------------------------------------------------------------

def _load_sp500_all() -> None:
    global _sp500_all, _sp500_name_to_ticker
    path = Path(settings.data_dir) / "sp500_all.csv"
    if not path.exists():
        logger.warning("data/sp500_all.csv not found — unindexed-company detection disabled.")
        return
    df = pd.read_csv(path, dtype=str)
    for _, row in df.iterrows():
        ticker = row["ticker"].strip()
        name = row["company_name"].strip()
        _sp500_all[ticker] = name
        clean = re.sub(r"[^a-z0-9 ]", "", name.lower())
        _sp500_name_to_ticker[clean] = ticker
    logger.info(f"Loaded {len(_sp500_all)} S&P 500 companies for query detection.")


# Short tickers that are also common English words — skip to avoid false positives.
_AMBIGUOUS_TICKERS = {
    "A", "T", "V", "F",
    "MA", "KO", "DG", "MS", "DE", "GE", "GM", "BA", "BK", "CB",
    "ARE", "ALL", "BIG", "CAR", "EAT", "GAS", "HAS", "HIT", "HOT",
    "KEY", "LOW", "MAR", "NEW", "NOW", "OLD", "OWL", "RUN", "SIX",
    "SKY", "TOP", "WIN",
}


def _detect_unindexed_company(query: str, indexed_tickers: set[str]) -> str | None:
    """
    Returns a human-readable company label if the query references an S&P 500
    company we haven't indexed, else None.

    Step 1: if ANY indexed company appears in the query, return None immediately
            so the RAG pipeline handles it.  This prevents "are" matching ticker
            ARE when the user asks "What are AMD risks…".
    Step 2: scan for unindexed companies, skipping short/ambiguous tickers.
    """
    if not _sp500_all:
        return None

    query_upper = query.upper()
    query_lower = re.sub(r"[^a-z0-9 ]", "", query.lower())

    # Step 1 — bail out if an indexed company is mentioned
    for ticker in indexed_tickers:
        if re.search(r"\b" + re.escape(ticker) + r"\b", query_upper):
            return None
    if bm25_index._chunks:
        seen: set[str] = set()
        for chunk in bm25_index._chunks:
            name = chunk.get("company_name", "").lower()
            if name and name not in seen:
                seen.add(name)
                clean = re.sub(r"[^a-z0-9 ]", "", name)
                if len(clean) > 5 and clean in query_lower:
                    return None

    # Step 2 — look for unindexed companies
    for ticker, company_name in _sp500_all.items():
        if ticker in indexed_tickers or ticker in _AMBIGUOUS_TICKERS:
            continue
        if not re.search(r"\b" + re.escape(ticker) + r"\b", query_upper):
            continue
        return f"{company_name} ({ticker})"

    for name_key, ticker in _sp500_name_to_ticker.items():
        if ticker in indexed_tickers:
            continue
        if len(name_key) > 8 and name_key in query_lower:
            return f"{_sp500_all.get(ticker, ticker)} ({ticker})"

    return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting RAG API (environment={settings.environment})")

    try:
        ensure_collection()
    except Exception as e:
        logger.warning(f"Could not ensure Qdrant collection: {e}")

    _load_sp500_all()

    if not bm25_index.load_index():
        logger.info("BM25 index not on disk — rebuilding from Qdrant (may take a minute)…")
        chunks = scroll_all_chunks()
        if chunks:
            idx, chunk_list = bm25_index.build_index(chunks)
            bm25_index.save_index(idx, chunk_list)
            bm25_index.load_index()
        else:
            logger.warning("Qdrant collection empty — run scripts/build_index.py first.")

    logger.info("API ready.")
    yield
    langfuse.flush()


app = FastAPI(
    title="S&P 500 10-K RAG API",
    description=(
        "Retrieval-augmented generation over S&P 500 annual filings. "
        "Answers are grounded in Item 1A (Risk Factors) and Item 7 (MD&A) "
        "sections from the most recent 10-K for each indexed company."
    ),
    version="1.0.0",
    # Disable interactive docs in production — they expose your API surface
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness + readiness probe. Used by Railway/Render health checks."""
    try:
        info = get_collection_info()
        chunk_count = info["chunk_count"] or 0
    except Exception:
        chunk_count = 0
    return HealthResponse(
        status="ok",
        chunk_count=chunk_count,
        bm25_loaded=bm25_index._bm25 is not None,
    )


@app.get("/companies", response_model=CompaniesResponse, tags=["data"])
async def list_companies():
    """
    Return all indexed companies with ticker, name, and filing date.
    Call this once on page load to populate a company filter if needed.
    """
    if bm25_index._chunks is None:
        return CompaniesResponse(companies=[], total=0)

    seen: dict[str, dict] = {}
    for chunk in bm25_index._chunks:
        ticker = chunk.get("ticker", "")
        if ticker not in seen or chunk.get("filing_date", "") > seen[ticker].get("filing_date", ""):
            seen[ticker] = chunk

    companies = [
        Company(
            ticker=d["ticker"],
            company_name=d.get("company_name", ""),
            filing_date=d.get("filing_date"),
        )
        for d in sorted(seen.values(), key=lambda x: x["ticker"])
    ]
    return CompaniesResponse(companies=companies, total=len(companies))


@app.post("/query", tags=["query"])
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    """
    Run the full RAG pipeline and return a grounded answer with citations.

    Set stream=true to receive a Server-Sent Events stream (text/event-stream).
    Set stream=false to receive a standard JSON response.
    """
    start_time = time.time()

    indexed_tickers: set[str] = set()
    if bm25_index._chunks:
        indexed_tickers = {c.get("ticker", "") for c in bm25_index._chunks}

    # Return a helpful message if a known-but-unindexed company is mentioned
    unindexed = _detect_unindexed_company(body.query, indexed_tickers)
    if unindexed:
        indexed_list = ", ".join(sorted(indexed_tickers)[:20]) + "…"
        return QueryResponse(
            answer=(
                f"I don't have 10-K data for **{unindexed}** yet. "
                f"I currently have filings indexed for {len(indexed_tickers)} companies "
                f"including: {indexed_list}\n\nMore companies will be added soon."
            ),
            citations=[],
            latency_ms=(time.time() - start_time) * 1000,
        )

    trace = langfuse.trace(
        name="rag_query",
        input={"query": body.query, "ticker": body.ticker},
    )

    try:
        t0 = time.time()
        span = trace.span(name="dense_retrieval")
        dense_results = retrieve_dense(body.query, ticker_filter=body.ticker)
        span.end(output={"n": len(dense_results)},
                 metadata={"ms": round((time.time() - t0) * 1000)})

        t0 = time.time()
        span = trace.span(name="sparse_retrieval")
        sparse_results = retrieve_sparse(body.query, ticker_filter=body.ticker)
        span.end(output={"n": len(sparse_results)},
                 metadata={"ms": round((time.time() - t0) * 1000)})

        t0 = time.time()
        span = trace.span(name="rerank")
        final_chunks = hybrid_retrieve(body.query, dense_results, sparse_results)
        span.end(output={"n": len(final_chunks)},
                 metadata={"ms": round((time.time() - t0) * 1000)})

        t0 = time.time()
        span = trace.span(name="generation")
        result = generate_answer(body.query, final_chunks)
        latency_ms = (time.time() - start_time) * 1000
        span.end(output={"chars": len(result["answer"])},
                 metadata={"ms": round((time.time() - t0) * 1000)})

        trace.update(
            output={"answer": result["answer"]},
            metadata={"total_ms": round(latency_ms)},
        )
        logger.info(f"Query answered in {round(latency_ms)}ms | '{body.query[:60]}'")

    except Exception as e:
        trace.update(metadata={"error": str(e)})
        raise

    citations = [Citation(**c) for c in result["citations"]]

    if body.stream:
        def event_stream():
            payload = {
                "answer": result["answer"],
                "citations": [c.model_dump() for c in citations],
                "latency_ms": latency_ms,
                "done": True,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return QueryResponse(answer=result["answer"], citations=citations, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Dev-only: built-in test UI
# ---------------------------------------------------------------------------

if not settings.is_production:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def serve_ui():
        """Local test UI — disabled in production."""
        ui_path = Path(__file__).parent.parent.parent / "static" / "index.html"
        return HTMLResponse(content=ui_path.read_text())
