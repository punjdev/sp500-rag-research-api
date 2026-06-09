#!/usr/bin/env bash
# Full pipeline: ingest → clear old index → rebuild Qdrant + BM25.
# Usage:  nohup bash scripts/overnight_build.sh > logs/overnight.log 2>&1 &

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON=".venv/bin/python"
mkdir -p logs

echo "Build started: $(date)"

echo "[1/3] Ingesting filings..."
$PYTHON scripts/ingest.py
echo "  Done: $(ls data/raw/*.json 2>/dev/null | wc -l | tr -d ' ') section files"

echo "[2/3] Clearing old index..."
$PYTHON - << 'PYEOF'
from qdrant_client import QdrantClient
from settings import settings
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
try:
    client.delete_collection(settings.qdrant_collection)
    print(f"  Deleted: {settings.qdrant_collection}")
except Exception as e:
    print(f"  Not found (ok): {e}")
PYEOF
rm -f data/bm25_index.pkl data/bm25_chunks.pkl

echo "[3/3] Building index..."
$PYTHON scripts/build_index.py

echo "Build complete: $(date)"
echo "Restart the server to load the updated index."
