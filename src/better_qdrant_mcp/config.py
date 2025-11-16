import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    default_collection: str | None = os.getenv("COLLECTION_NAME")
    qdrant_timeout: float = float(os.getenv("QDRANT_TIMEOUT", "30"))
    # OpenAI embeddings
    openai_api_key: str | None = os.getenv("OPENAPI_API_KEY") or os.getenv(
        "OPENAI_API_KEY"
    )
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "30"))
    # Preferred vector name for multi-vector collections
    default_vector_name: str = os.getenv("DEFAULT_VECTOR_NAME", "dense")


def get_settings() -> Settings:
    return Settings()
