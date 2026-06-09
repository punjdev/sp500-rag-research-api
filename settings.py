"""
Central configuration for the RAG system.

All values come from environment variables.
- Development: loaded from .env file automatically by pydantic-settings
- Production (Railway/Render): set as env vars in the platform dashboard

Required env vars: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY,
                   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # --- Environment ---
    # Set to "production" on Railway/Render to disable dev-only routes
    # (Swagger UI /docs, /redoc, and the built-in test HTML page).
    environment: str = "development"

    # --- Cohere ---
    cohere_api_key: str
    cohere_embed_model: str = "embed-english-v3.0"
    cohere_rerank_model: str = "rerank-english-v3.0"
    # command-r-plus was retired Sept 2025; pinned to last stable version.
    cohere_generation_model: str = "command-r-plus-08-2024"
    generation_temperature: float = 0.1
    generation_max_tokens: int = 600

    # --- Qdrant ---
    qdrant_url: str
    qdrant_api_key: str = ""
    qdrant_collection: str = "sp500_10k"

    # --- Langfuse ---
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://us.cloud.langfuse.com"

    # --- Retrieval ---
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rerank_top_k: int = 20
    final_top_k: int = 5

    # --- Paths ---
    data_dir: str = "data"
    bm25_index_path: str = "data/bm25_index.pkl"
    bm25_chunks_path: str = "data/bm25_chunks.pkl"
    sp500_csv_path: str = "data/sp500_companies.csv"
    checkpoint_path: str = "data/checkpoints/ingestion_progress.json"

    # --- CORS ---
    # Comma-separated list of allowed origins.
    # In production set: CORS_ORIGINS=https://devpunjabi.com,https://www.devpunjabi.com
    # In development the default below covers localhost.
    cors_origins_str: str = (
        "https://devpunjabi.com,"
        "https://www.devpunjabi.com,"
        "http://localhost:3000,"
        "http://localhost:8000,"
        "http://127.0.0.1:8000"
    )

    # --- Rate limiting ---
    rate_limit_per_minute: int = 10

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_str.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
