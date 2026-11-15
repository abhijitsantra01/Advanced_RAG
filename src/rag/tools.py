"""Typed tool functions for Agentic RAG.

See docs/EXECUTION_PLAN.md Level 5.
"""
from typing import Any

import logfire
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.config import RAGSettings
from rag.embedder import EmbedderClient
from rag.vector_store import QdrantStore


class RAGTools:
    """Registry of pure, typed tools for Gemini tool calling."""

    def __init__(
        self,
        settings: RAGSettings | None = None,
        embedder: EmbedderClient | None = None,
        vector_store: QdrantStore | None = None,
    ) -> None:
        self.settings = settings or RAGSettings()
        self.embedder = embedder or EmbedderClient(settings=self.settings)
        self.vector_store = vector_store or QdrantStore(settings=self.settings)

    def search_use_case_kb(
        self,
        query: str,
        use_case: str = "general",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search enterprise knowledge base within a use_case namespace (default: 'general')."""
        with logfire.span(
            "tool_search_use_case_kb",
            query=query,
            use_case=use_case,
            top_k=top_k,
        ):
            query_embeddings = self.embedder.embed_texts([query], task_type="RETRIEVAL_QUERY")
            query_vector = query_embeddings[0]

            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                use_case=use_case,
            )

            return [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "chunk_index": r.chunk.chunk_index,
                    "text": r.chunk.text,
                    "score": round(r.score, 4),
                    "use_case": r.chunk.use_case,
                }
                for r in results
            ]

    def list_available_use_cases(self) -> list[str]:
        """List all available use-case domain namespaces currently indexed in the knowledge base."""
        with logfire.span("tool_list_available_use_cases"):
            try:
                records, _ = self.vector_store.client.scroll(
                    collection_name=self.vector_store.primary_collection,
                    limit=100,
                    with_payload=["use_case"],
                    with_vectors=False,
                )
                use_cases = sorted(
                    {
                        str(r.payload.get("use_case"))
                        for r in records
                        if r.payload and r.payload.get("use_case")
                    }
                )
                if use_cases:
                    return list(use_cases)
            except Exception:
                pass
            return ["architecture", "billing", "compliance", "general", "technical"]

    def get_adjacent_chunks(
        self,
        doc_id: str,
        chunk_index: int,
        window: int = 1,
    ) -> list[dict[str, Any]]:
        """Fetch adjacent chunks before and after a specific chunk to recover cut-off context."""
        with logfire.span(
            "tool_get_adjacent_chunks",
            doc_id=doc_id,
            chunk_index=chunk_index,
            window=window,
        ):
            target_indices = set(range(max(0, chunk_index - window), chunk_index + window + 1))
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
            try:
                records, _ = self.vector_store.client.scroll(
                    collection_name=self.vector_store.primary_collection,
                    scroll_filter=query_filter,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                records = []

            adjacent = []
            for r in records:
                payload = r.payload or {}
                if payload.get("chunk_index") in target_indices:
                    adjacent.append(
                        {
                            "chunk_id": payload.get("chunk_id"),
                            "doc_id": payload.get("doc_id"),
                            "chunk_index": payload.get("chunk_index"),
                            "text": payload.get("text"),
                            "use_case": payload.get("use_case"),
                        }
                    )
            adjacent.sort(key=lambda x: x.get("chunk_index", 0))
            return adjacent


_default_tools: RAGTools | None = None


def set_default_tools(tools: RAGTools) -> None:
    """Set the active default RAGTools instance."""
    global _default_tools
    _default_tools = tools


def get_default_tools() -> RAGTools:
    """Get or create singleton default RAGTools instance."""
    global _default_tools
    if _default_tools is None:
        _default_tools = RAGTools()
    return _default_tools


def search_use_case_kb(
    query: str,
    use_case: str = "general",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search enterprise knowledge base within a use_case namespace (default: 'general')."""
    return get_default_tools().search_use_case_kb(query=query, use_case=use_case, top_k=top_k)


def list_available_use_cases() -> list[str]:
    """List all available use-case domain namespaces currently indexed in the knowledge base."""
    return get_default_tools().list_available_use_cases()


def get_adjacent_chunks(
    doc_id: str,
    chunk_index: int,
    window: int = 1,
) -> list[dict[str, Any]]:
    """Fetch adjacent chunks before and after a specific chunk to recover cut-off context."""
    return get_default_tools().get_adjacent_chunks(
        doc_id=doc_id, chunk_index=chunk_index, window=window
    )
