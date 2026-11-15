"""Unit tests for Level 4 use-case wise vector routing and metadata filtering."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from qdrant_client.models import PayloadSchemaType

from rag.config import RAGSettings
from rag.pipeline import RAGPipeline
from rag.schemas import Chunk, QueryRequest
from rag.vector_store import QdrantStore


def test_query_request_schema() -> None:
    req = QueryRequest(query="Explain architecture", use_case="architecture", score_threshold=0.75)
    assert req.query == "Explain architecture"
    assert req.use_case == "architecture"
    assert req.score_threshold == 0.75


def test_ensure_collection_exists_creates_keyword_payload_indexes() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        qdrant_url="http://localhost:6333",
        embedding_dimension=3072,
    )
    with patch("rag.vector_store.qdrant_client.QdrantClient") as mock_qdrant_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_qdrant_cls.return_value = mock_client

        store = QdrantStore(settings=settings)

        assert mock_client.create_collection.called
        assert mock_client.create_payload_index.call_count >= 2

        created_fields = [
            call.kwargs["field_name"] for call in mock_client.create_payload_index.call_args_list
        ]
        assert "use_case" in created_fields
        assert "doc_id" in created_fields

        created_schemas = [
            call.kwargs["field_schema"] for call in mock_client.create_payload_index.call_args_list
        ]
        assert all(s == PayloadSchemaType.KEYWORD for s in created_schemas)
        assert store.collection_name == "rag_chunks_gemini_embedding_2"


def test_search_applies_use_case_filter_and_score_threshold() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        qdrant_url="http://localhost:6333",
        embedding_dimension=4,
    )
    with patch("rag.vector_store.qdrant_client.QdrantClient") as mock_qdrant_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response
        mock_qdrant_cls.return_value = mock_client

        store = QdrantStore(settings=settings)
        store.search(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            top_k=5,
            use_case="technical",
            score_threshold=0.72,
        )

        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["score_threshold"] == 0.72

        query_filter = call_kwargs["query_filter"]
        assert query_filter is not None
        assert len(query_filter.must) == 1
        condition = query_filter.must[0]
        assert condition.key == "use_case"
        assert condition.match.value == "technical"


def test_strict_use_case_isolation_zero_cross_domain_leakage() -> None:
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

        technical_chunk = Chunk(
            chunk_id="tech-1",
            doc_id="tech-doc",
            chunk_index=0,
            text="Technical system architecture documentation.",
            start_char=0,
            end_char=43,
            embedding_model="gemini-embedding-2",
            use_case="technical",
            metadata={"domain": "engineering"},
        )
        scored_point = MagicMock()
        scored_point.score = 0.88
        scored_point.payload = technical_chunk.model_dump()

        mock_response = MagicMock()
        mock_response.points = [scored_point]
        mock_qdrant.query_points.return_value = mock_response
        mock_qdrant_cls.return_value = mock_qdrant

        pipeline = RAGPipeline(settings=settings)
        pipeline.embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        pipeline.generator.generate = MagicMock(
            return_value=("Technical answer with citations.", "gemini-3.5-flash-lite", False)
        )

        response = pipeline.query(
            question="How is the tech stack structured?",
            use_case="technical",
            score_threshold=0.65,
        )

        assert len(response.retrieved_chunks) == 1
        assert response.retrieved_chunks[0].chunk.use_case == "technical"
        assert response.retrieved_chunks[0].chunk.metadata == {"domain": "engineering"}
        assert response.retrieved_chunks[0].score >= 0.65


def test_pipeline_ingest_directory_with_custom_use_case(tmp_path: Path) -> None:
    sample_file = tmp_path / "billing_spec.md"
    sample_file.write_text("Billing invoice terms and subscription fees.", encoding="utf-8")

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

        count = pipeline.ingest_directory(tmp_path, use_case="billing")

        assert count == 1
        pipeline.vector_store.upsert_chunks.assert_called_once()
        upserted_items = pipeline.vector_store.upsert_chunks.call_args[0][0]
        assert upserted_items[0][0].use_case == "billing"
