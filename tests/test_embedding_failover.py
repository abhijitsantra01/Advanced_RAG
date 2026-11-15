"""Unit tests for Level 3 Embedding Failover and Blue-Green Re-Ingestion."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors

from rag.config import RAGSettings
from rag.embedder import EmbedderClient, EmbeddingFailoverRequired
from rag.pipeline import RAGPipeline
from rag.schemas import Chunk, RetrievalResult
from rag.vector_store import (
    ACTIVE_ALIAS,
    collection_name_for_model,
)


def test_collection_name_for_model() -> None:
    assert collection_name_for_model("gemini-embedding-2") == "rag_chunks_gemini_embedding_2"
    assert collection_name_for_model("gemini-embedding-001") == "rag_chunks_gemini_embedding_001"


def test_embedder_primary_429_triggers_failover_required() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_embedding_model="gemini-embedding-2",
        fallback_embedding_model="gemini-embedding-001",
        embedding_dimension=3072,
    )
    with patch("rag.embedder.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        err_429 = errors.APIError(429, {"error": "RESOURCE_EXHAUSTED"})
        mock_client.models.embed_content.side_effect = err_429

        embedder = EmbedderClient(settings=settings)
        assert embedder.active_model == "gemini-embedding-2"

        with pytest.raises(EmbeddingFailoverRequired) as exc_info:
            embedder.embed_texts(["test text"])

        assert exc_info.value.fallback_model == "gemini-embedding-001"
        assert embedder.active_model == "gemini-embedding-001"
        assert embedder.dimension == 768


def test_embedder_fallback_429_raises_without_looping() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_embedding_model="gemini-embedding-001",
        fallback_embedding_model="gemini-embedding-001",
        embedding_dimension=768,
    )
    with patch("rag.embedder.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        err_429 = errors.APIError(429, {"error": "RESOURCE_EXHAUSTED"})
        mock_client.models.embed_content.side_effect = err_429

        embedder = EmbedderClient(settings=settings)
        with pytest.raises(errors.APIError) as exc_info:
            embedder.embed_texts(["test text"])

        assert exc_info.value.code == 429


def test_reingest_all_creates_collection_and_swaps_alias(tmp_path: Path) -> None:
    # Setup sample doc in temp dir
    doc_path = tmp_path / "note.md"
    doc_path.write_text("Test architecture content for blue-green swap.", encoding="utf-8")

    settings = RAGSettings(
        gemini_api_key="test-key",
        qdrant_url="http://localhost:6333",
        primary_embedding_model="gemini-embedding-2",
        fallback_embedding_model="gemini-embedding-001",
        embedding_dimension=3072,
    )

    with (
        patch("rag.embedder.genai.Client"),
        patch("rag.generator.genai.Client"),
        patch("rag.vector_store.qdrant_client.QdrantClient") as mock_qdrant_cls,
    ):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False
        mock_qdrant_cls.return_value = mock_qdrant

        pipeline = RAGPipeline(settings=settings)

        # Set embedder to fallback model and dimension
        pipeline.embedder.active_model = "gemini-embedding-001"
        pipeline.embedder.dimension = 768
        pipeline.embedder.embed_texts = MagicMock(return_value=[[0.1] * 768])

        count = pipeline.reingest_all(source_dir=tmp_path, model_name="gemini-embedding-001")

        assert count > 0
        # Check that collection rag_chunks_gemini_embedding_001 was created
        mock_qdrant.create_collection.assert_called()
        call_kwargs = mock_qdrant.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "rag_chunks_gemini_embedding_001"
        assert call_kwargs["vectors_config"].size == 768

        # Check that alias was swapped atomically
        mock_qdrant.update_collection_aliases.assert_called_once()
        alias_call = mock_qdrant.update_collection_aliases.call_args.kwargs
        ops = alias_call["change_aliases_operations"]
        assert len(ops) == 1
        assert ops[0].create_alias.collection_name == "rag_chunks_gemini_embedding_001"
        assert ops[0].create_alias.alias_name == ACTIVE_ALIAS


def test_query_interrupted_by_429_triggers_reingest_and_completes() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        qdrant_url="http://localhost:6333",
        primary_embedding_model="gemini-embedding-2",
        fallback_embedding_model="gemini-embedding-001",
        embedding_dimension=3072,
    )

    with (
        patch("rag.embedder.genai.Client"),
        patch("rag.generator.genai.Client"),
        patch("rag.vector_store.qdrant_client.QdrantClient") as mock_qdrant_cls,
    ):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant_cls.return_value = mock_qdrant

        pipeline = RAGPipeline(settings=settings)

        # First call on query embedding throws EmbeddingFailoverRequired (simulating 429 on primary)
        failover_exc = EmbeddingFailoverRequired(fallback_model="gemini-embedding-001")
        pipeline.embedder.embed_texts = MagicMock(
            side_effect=[
                failover_exc,  # First query embedding fails
                [[0.2] * 768],  # Second query embedding succeeds on fallback model
            ]
        )

        # Mock reingest_all
        pipeline.reingest_all = MagicMock(return_value=1)

        # Mock vector search and generator
        dummy_chunk = Chunk(
            chunk_id="chunk-fallback-1",
            doc_id="arch-note",
            chunk_index=0,
            text="Grounded context from fallback collection.",
            start_char=0,
            end_char=42,
            embedding_model="gemini-embedding-001",
            use_case="general",
        )
        dummy_result = RetrievalResult(chunk=dummy_chunk, score=0.91)
        pipeline.vector_store.search = MagicMock(return_value=[dummy_result])
        pipeline.generator.generate = MagicMock(
            return_value=("Answer with [arch-note:0]", "gemini-3.5-flash-lite", False)
        )

        response = pipeline.query("How does failover work?")

        # Assert reingest_all was called
        pipeline.reingest_all.assert_called_once()
        # Assert pipeline returned response using fallback embedding model
        assert response.embedding_model_used == "gemini-embedding-001"
        assert "[arch-note:0]" in response.answer
