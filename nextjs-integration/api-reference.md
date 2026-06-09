# S&P 500 10-K RAG API — Website Integration Reference

For the Claude session working on devpunjabi.com. Covers what the API does, all endpoints, and how to call it from Next.js — both locally and against the deployed Railway instance.

---

## What the API does

A RAG (retrieval-augmented generation) API over 100 S&P 500 companies' most recent 10-K SEC filings. The user asks a natural language question; the API returns a grounded answer with citations pointing to the exact filing sections used.

- **Indexed sections:** Item 1A (Risk Factors) and Item 7 (MD&A) per company
- **100 companies indexed** — see the full list at `GET /companies`
- **Generation model:** Cohere Command R+ with native document citations
- **Retrieval:** Hybrid dense + BM25 with Cohere reranking

---

## Base URL

| Environment | URL |
|-------------|-----|
| Local dev   | `http://localhost:8000` |
| Production  | Set `NEXT_PUBLIC_RAG_API_URL` in Vercel dashboard once deployed to Railway |

In your Next.js code:
```ts
const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL ?? "http://localhost:8000";
```

To run the API locally:
```bash
# In the sp500-rag-research-api repo:
source .venv/bin/activate
python -m uvicorn src.api.main:app --reload
# Serves at http://localhost:8000
```

---

## Endpoints

### `POST /query` — ask a question

**Request:**
```ts
{
  query: string      // required, 1–1000 chars
  ticker?: string    // optional — restrict to one company, e.g. "AAPL"
  stream?: boolean   // false = JSON response (recommended); true = SSE stream
}
```

**Example:**
```ts
const res = await fetch(`${API_URL}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "What are NVIDIA's risks related to AI chip export controls?",
    ticker: "NVDA",
    stream: false,
  }),
});
const data = await res.json();
```

**Response:**
```json
{
  "answer": "NVIDIA faces significant risks from US export controls on advanced chips...",
  "citations": [
    {
      "cited_text": "The U.S. government has imposed export controls...",
      "ticker": "NVDA",
      "company_name": "NVIDIA Corporation",
      "section": "risk_factors",
      "filing_date": "2025-02-25"
    }
  ],
  "latency_ms": 4800
}
```

**Citation fields:**
| Field | Values |
|-------|--------|
| `section` | `"risk_factors"` (Item 1A) or `"mda"` (Item 7 MD&A) |
| `filing_date` | ISO date string — use `.slice(0, 4)` to display the year |

**Error responses:**
| Status | Meaning |
|--------|---------|
| `422` | Invalid request body |
| `429` | Rate limited (10 req/min per IP) — show a retry message to the user |
| `500` | Internal server error |

---

### `GET /companies` — list indexed companies

Call once on page load to populate any company selector.

**Response:**
```json
{
  "companies": [
    { "ticker": "AAPL", "company_name": "Apple Inc.", "filing_date": "2024-11-01" },
    { "ticker": "NVDA", "company_name": "NVIDIA Corporation", "filing_date": "2025-02-25" }
  ],
  "total": 100
}
```

---

### `GET /health` — liveness probe

```json
{ "status": "ok", "chunk_count": 11200, "bm25_loaded": true }
```

---

## Ready-made TypeScript client

`nextjs-integration/lib/rag-api.ts` is a fully typed client — copy it to your portfolio repo as `lib/rag-api.ts`.

```ts
import { queryRAG, getCompanies, formatSection, filingYear } from "@/lib/rag-api";

// Non-streaming query (simplest)
const result = await queryRAG("What are Apple's main risks?", { ticker: "AAPL" });
console.log(result.answer);
console.log(result.citations);

// Get all indexed companies
const companies = await getCompanies();
```

The full page and component examples are in:
- `nextjs-integration/app/research/page.tsx` — page wrapper
- `nextjs-integration/components/RagChat.tsx` — full chat component

---

## CORS

The API allows cross-origin requests from:
- `https://devpunjabi.com`
- `https://www.devpunjabi.com`
- `http://localhost:3000`

No authentication headers needed. Always set `Content-Type: application/json` on POST requests.

---

## Key behaviours to design for

**Latency is 3–8 seconds.** Always show a loading state. The `latency_ms` field in the response can be displayed as a "answered in Xs" note.

**No conversation memory.** Each `/query` call is fully independent. Don't send chat history — the API won't use it.

**Unindexed companies return a plain message, not an error.** If the user asks about a company not yet indexed, `answer` will be a helpful plain-text explanation and `citations` will be empty. Render it the same as a normal answer.

**`ticker` is optional but recommended.** When the user has selected a specific company, pass it. It significantly improves retrieval precision. Omit it for open-ended cross-company questions.

**Streaming sends one complete event, not tokens.** `stream: true` returns SSE but delivers the entire answer in a single event. If you want a typing effect, receive the full response then animate it client-side.

**Rate limit: 10 req/min per IP.** Handle `429` gracefully with a short user-facing message.
