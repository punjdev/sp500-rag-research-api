FROM python:3.13-slim

WORKDIR /app

# System deps for lxml and other C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first — Docker caches this layer until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Railway / Render inject $PORT at runtime; default to 8000 locally
ENV PORT=8000

EXPOSE $PORT

# Health check so the platform knows when the app is ready
# Startup takes ~60s (BM25 rebuild from Qdrant) — allow 120s before failing
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
