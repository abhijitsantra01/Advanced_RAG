"""Centralized configuration for Enterprise RAG using Pydantic Settings.
Antigravity: Implement according to docs/LEVEL_1_SPEC.md Section 1.
"""
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = Field(default="", description="Google GenAI API Key")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant Cloud URL")
    qdrant_api_key: str = Field(default="", description="Qdrant Cloud API Key")
    logfire_token: str | None = Field(default=None, description="Pydantic Logfire Token")

    primary_embedding_model: str = Field(default="gemini-embedding-2")
    fallback_embedding_model: str = Field(default="gemini-embedding-001")
    embedding_dimension: int = Field(default=3072, gt=0)

    primary_llm_model: str = Field(default="gemini-3.5-flash-lite")
    fallback_llm_model: str = Field(default="gemini-3.1-flash-lite")

    chunk_size: int = Field(default=800, gt=50)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, gt=0)
    score_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> "RAGSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        return self
