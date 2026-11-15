"""Offline tests for RAG pipeline with mocked external clients.

See docs/LEVEL_1_SPEC.md.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.config import RAGSettings
from rag.embedder import EmbedderClient
from rag.generator import GeneratorClient
from rag.pipeline import RAGPipeline
from rag.schemas import Chunk, RAGResponse, RetrievalResult


def test_embedder_client_embed_texts() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_embedding_model="gemini-embedding-2",
        embedding_dimension=4,
    )
    with patch("rag.embedder.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        mock_embedding_1 = MagicMock()
        mock_embedding_1.values = [0.1, 0.2, 0.3, 0.4]
        mock_embedding_2 = MagicMock()
        mock_embedding_2.values = [0.5, 0.6, 0.7, 0.8]

        mock_resp1 = MagicMock()
        mock_resp1.embeddings = [mock_embedding_1]
        mock_resp2 = MagicMock()
        mock_resp2.embeddings = [mock_embedding_2]
        mock_client.models.embed_content.side_effect = [mock_resp1, mock_resp2]

        embedder = EmbedderClient(settings=settings)
        result = embedder.embed_texts(["hello", "world"], task_type="RETRIEVAL_DOCUMENT")

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert result[1] == [0.5, 0.6, 0.7, 0.8]
        assert mock_client.models.embed_content.call_count == 2


def test_embedder_batching_over_100_texts() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_embedding_model="gemini-embedding-2",
        embedding_dimension=4,
    )
    with patch("rag.embedder.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        def mock_embed(model, contents, config):
            resp = MagicMock()
            item = MagicMock()
            item.values = [0.1, 0.2, 0.3, 0.4]
            resp.embeddings = [item]
            return resp

        mock_client.models.embed_content.side_effect = mock_embed

        embedder = EmbedderClient(settings=settings)
        # Generate 125 texts (exceeding single-call 100 limit)
        texts = [f"Text chunk number {i}" for i in range(125)]
        result = embedder.embed_texts(texts)

        # 125 texts embedded with 1-to-1 vectors
        assert mock_client.models.embed_content.call_count == 125
        assert len(result) == 125
        assert all(len(vec) == 4 for vec in result)


def test_generator_client_generate() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
    )
    with patch("rag.generator.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        mock_gen_response = MagicMock()
        mock_gen_response.text = "Based on [doc-1:0], the system uses Qdrant Cloud."
        mock_client.models.generate_content.return_value = mock_gen_response

        generator = GeneratorClient(settings=settings)
        dummy_chunk = Chunk(
            chunk_id="chunk-uuid-1",
            doc_id="doc-1",
            chunk_index=0,
            text="The system uses Qdrant Cloud for vector storage.",
            start_char=0,
            end_char=49,
            embedding_model="gemini-embedding-2",
            use_case="general",
        )
        retrieval_result = RetrievalResult(chunk=dummy_chunk, score=0.95)

        answer, model_used, fallback_triggered = generator.generate(
            query="What vector store is used?",
            retrieved_chunks=[retrieval_result],
        )

        assert "[doc-1:0]" in answer
        assert model_used == "gemini-3.5-flash-lite"
        assert fallback_triggered is False
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert "[doc-1:0] The system uses Qdrant Cloud" in call_kwargs["contents"]


def test_pipeline_ingest_directory(tmp_path: Path) -> None:
    # Create test markdown file
    sample_file = tmp_path / "sample.md"
    sample_file.write_text("First sentence for RAG test. " * 30, encoding="utf-8")

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

        # Mock embedding return
        pipeline.embedder.embed_texts = MagicMock(
            side_effect=lambda texts, **kwargs: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        )
        pipeline.vector_store.upsert_chunks = MagicMock()

        count = pipeline.ingest_directory(tmp_path)

        assert count > 0
        pipeline.embedder.embed_texts.assert_called_once()
        pipeline.vector_store.upsert_chunks.assert_called_once()


def test_pipeline_query() -> None:
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

        dummy_chunk = Chunk(
            chunk_id="chunk-uuid-1",
            doc_id="arch-note",
            chunk_index=0,
            text="Pipeline uses Gemini Embedding 2.",
            start_char=0,
            end_char=33,
            embedding_model="gemini-embedding-2",
            use_case="general",
        )
        dummy_result = RetrievalResult(chunk=dummy_chunk, score=0.92)

        pipeline.embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        pipeline.vector_store.search = MagicMock(return_value=[dummy_result])
        pipeline.generator.generate = MagicMock(
            return_value=(
                "According to [arch-note:0], the pipeline uses Gemini Embedding 2.",
                "gemini-3.5-flash-lite",
                False,
            )
        )

        response = pipeline.query("What embedding model is used?")

        assert isinstance(response, RAGResponse)
        assert response.query == "What embedding model is used?"
        assert "[arch-note:0]" in response.answer
        assert len(response.retrieved_chunks) == 1
        assert response.embedding_model_used == "gemini-embedding-2"
        assert response.generation_model_used == "gemini-3.5-flash-lite"
        assert response.fallback_triggered is False
