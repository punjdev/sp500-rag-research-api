"""
RAG generation with Cohere Command R+.

Passes retrieved chunks via the documents= parameter so Cohere handles
citation grounding natively. Temperature 0.1 for factual consistency.
"""

import cohere
from settings import settings

_client = cohere.Client(api_key=settings.cohere_api_key)

SYSTEM_PROMPT = """You are a financial research assistant. Answer questions based ONLY on the provided 10-K filing excerpts.

Rules:
- Cite the company name and section (Risk Factors or MD&A) for each claim.
- If the provided context does not contain enough information to answer the question, say "The provided filings do not contain sufficient information to answer this question."
- Do not speculate beyond what the filings state.
- Be concise and precise."""


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a cited answer from retrieved chunks using Cohere Command R+.

    Args:
        query:  The user's natural language question.
        chunks: The top-k retrieved chunks (from fusion.hybrid_retrieve).
                Each chunk must have at least "text", "ticker", "section",
                "company_name", and "filing_date" keys.

    Returns:
        {
            "answer": str,           # The generated answer text
            "citations": list[dict], # Each citation: {text, ticker, section, filing_date}
        }
    """
    # Format chunks as Cohere document dicts.
    # The "id" field is used in citation.document_ids to trace which chunk
    # each claim came from.
    documents = [
        {
            "id": f"doc_{i}",
            "text": chunk["text"],
            # Additional fields included in the document context
            "ticker": chunk.get("ticker", ""),
            "company_name": chunk.get("company_name", ""),
            "section": chunk.get("section", ""),
            "filing_date": chunk.get("filing_date", ""),
        }
        for i, chunk in enumerate(chunks)
    ]

    response = _client.chat(
        model=settings.cohere_generation_model,
        message=query,
        documents=documents,
        preamble=SYSTEM_PROMPT,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        # "accurate" mode instructs Cohere to generate inline citation spans
        # for each factual claim rather than paraphrasing without attribution.
        citation_quality="accurate",
    )

    # Map Cohere's citation objects to our simplified format
    citations = []
    if response.citations:
        for citation in response.citations:
            for doc_id in citation.document_ids:
                # doc_id is like "doc_0", "doc_1", etc.
                doc_index = int(doc_id.split("_")[1])
                source_chunk = chunks[doc_index]
                citations.append({
                    "cited_text": citation.text,
                    "ticker": source_chunk.get("ticker", ""),
                    "company_name": source_chunk.get("company_name", ""),
                    "section": source_chunk.get("section", ""),
                    "filing_date": source_chunk.get("filing_date", ""),
                })

    return {
        "answer": response.text,
        "citations": citations,
    }
