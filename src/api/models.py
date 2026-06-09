"""Pydantic request/response models for the FastAPI endpoints."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language question")
    ticker: str | None = Field(None, description="Optional: restrict to a specific company ticker (e.g. 'AAPL')")
    stream: bool = Field(True, description="Stream the response as SSE events")


class Citation(BaseModel):
    cited_text: str
    ticker: str
    company_name: str
    section: str       # "risk_factors" or "mda"
    filing_date: str   # "YYYY-MM-DD"


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: float


class Company(BaseModel):
    ticker: str
    company_name: str
    filing_date: str | None = None


class CompaniesResponse(BaseModel):
    companies: list[Company]
    total: int


class HealthResponse(BaseModel):
    status: str
    chunk_count: int
    bm25_loaded: bool
