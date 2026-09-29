"""Central configuration. All secrets come from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # reads .env (does not override variables already set)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    # secrets
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    # models
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536  # fixed by text-embedding-3-small
    chat_model: str = "gpt-4o-mini"
    # pinecone serverless placement
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    # ingestion
    pdf_url: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    pdf_path: str = "data/Ebook-Agentic-AI.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # retrieval
    top_k: int = 3
    relevance_threshold: float = 0.30  # cosine score; tune on your own data


_REQUIRED = ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    missing = [name for name in _REQUIRED if not os.getenv(name, "").strip()]
    if missing:
        raise ConfigError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
    try:
        top_k = int(os.getenv("TOP_K", "3"))
        threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
    except ValueError as exc:
        raise ConfigError(f"TOP_K / RELEVANCE_THRESHOLD must be numeric: {exc}") from exc
    if top_k < 1:
        raise ConfigError("TOP_K must be >= 1")
    if not 0.0 <= threshold < 1.0:
        raise ConfigError("RELEVANCE_THRESHOLD must be in [0, 1)")

    return Settings(
        openai_api_key=os.environ["OPENAI_API_KEY"].strip(),
        pinecone_api_key=os.environ["PINECONE_API_KEY"].strip(),
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"].strip(),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        top_k=top_k,
        relevance_threshold=threshold,
    )
