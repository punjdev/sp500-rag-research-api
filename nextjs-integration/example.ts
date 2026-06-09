// Set in .env.local (dev) and Vercel environment variables (prod)
// NEXT_PUBLIC_RAG_API_URL=https://your-api.railway.app
const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL ?? "http://localhost:8000";

type Citation = {
  cited_text: string;
  ticker: string;
  company_name: string;
  section: "risk_factors" | "mda";
  filing_date: string;
};

type QueryResponse = {
  answer: string;
  citations: Citation[];
  latency_ms: number;
};

export async function askRAG(question: string): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: question, stream: false }),
  });

  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
