"""Strict Pydantic data contracts for Documents, Chunks, and RAG Responses.

See docs/LEVEL_1_SPEC.md and docs/EXECUTION_PLAN.md Level 4.
"""
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    content: str
    source_path: str
    use_case: str = Field(default="general")
    content_hash: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    embedding_model: str
    use_case: str = Field(default="general")
    content_hash: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float


class QueryRequest(BaseModel):
    query: str
    use_case: str = Field(default="general")
    top_k: int = Field(default=5, gt=0)
    score_threshold: float = Field(default=0.65, ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[RetrievalResult]
    embedding_model_used: str
    generation_model_used: str
    fallback_triggered: bool = False
