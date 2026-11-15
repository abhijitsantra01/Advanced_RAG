"""Unit tests for SHA-256 content hash caching and deduplication."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.config import RAGSettings
from rag.pipeline import RAGPipeline


def test_hash_caching_skips_unchanged_file(tmp_path: Path) -> None:
    sample_file = tmp_path / "cache_test.md"
    sample_file.write_text("Unique content for SHA-256 hash caching test.", encoding="utf-8")

    settings = RAGSettings(
        gemini_api_key="test-key",
        qdrant_url="http://localhost:6333",
        embedding_dimension=4,
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
        pipeline.embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        pipeline.vector_store.upsert_chunks = MagicMock()

        # Simulate first run: Qdrant has not seen this hash yet
        mock_count_res = MagicMock()
        mock_count_res.count = 0
        mock_qdrant.count.return_value = mock_count_res

        count1 = pipeline.ingest_directory(tmp_path)
        assert count1 == 1
        assert pipeline.embedder.embed_texts.call_count == 1
        assert pipeline.vector_store.upsert_chunks.call_count == 1

        # Simulate second run: Qdrant now contains this document with this hash
        mock_count_res.count = 1
        pipeline.embedder.embed_texts.reset_mock()
        pipeline.vector_store.upsert_chunks.reset_mock()

        count2 = pipeline.ingest_directory(tmp_path)
        # Should skip embedding and upserting completely
        assert count2 == 0
        assert pipeline.embedder.embed_texts.call_count == 0
        assert pipeline.vector_store.upsert_chunks.call_count == 0


def test_hash_caching_reembeds_when_content_modified(tmp_path: Path) -> None:
    sample_file = tmp_path / "modified_test.md"
    sample_file.write_text("Version 1 content.", encoding="utf-8")

    settings = RAGSettings(
        gemini_api_key="test-key",
        qdrant_url="http://localhost:6333",
        embedding_dimension=4,
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
        pipeline.embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        pipeline.vector_store.upsert_chunks = MagicMock()

        # Known hash for v1
        mock_count_res = MagicMock()
        mock_count_res.count = 1
        mock_qdrant.count.return_value = mock_count_res

        # Modify file content (Version 2)
        sample_file.write_text("Version 2 updated content with new facts.", encoding="utf-8")

        # Now Qdrant count returns 0 because the new hash doesn't exist
        mock_count_res.count = 0

        count = pipeline.ingest_directory(tmp_path)
        assert count == 1
        assert pipeline.embedder.embed_texts.call_count == 1
        assert pipeline.vector_store.upsert_chunks.call_count == 1


def test_force_flag_bypasses_hash_cache(tmp_path: Path) -> None:
    sample_file = tmp_path / "force_test.md"
    sample_file.write_text("Content that already exists in Qdrant.", encoding="utf-8")

    settings = RAGSettings(
        gemini_api_key="test-key",
        qdrant_url="http://localhost:6333",
        embedding_dimension=4,
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
        pipeline.embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        pipeline.vector_store.upsert_chunks = MagicMock()

        # Document is marked as already in Qdrant
        mock_count_res = MagicMock()
        mock_count_res.count = 1
        mock_qdrant.count.return_value = mock_count_res

        count = pipeline.ingest_directory(tmp_path, force=True)
        # force=True must bypass cache and re-embed
        assert count == 1
        assert pipeline.embedder.embed_texts.call_count == 1
