"""Unit tests for GeneratorClient Level 2 fallback logic."""
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors

from rag.config import RAGSettings
from rag.generator import GeneratorClient
from rag.schemas import Chunk, RetrievalResult


@pytest.fixture
def dummy_retrieval_chunks() -> list[RetrievalResult]:
    chunk = Chunk(
        chunk_id="chunk-001",
        doc_id="doc-test",
        chunk_index=0,
        text="Qdrant Cloud provides vector storage.",
        start_char=0,
        end_char=38,
        embedding_model="gemini-embedding-2",
        use_case="general",
    )
    return [RetrievalResult(chunk=chunk, score=0.95)]


def test_primary_generation_succeeds(dummy_retrieval_chunks: list[RetrievalResult]) -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
        fallback_llm_model="gemini-3.1-flash-lite",
    )

    with patch("rag.generator.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = "Grounded answer from primary model."
        mock_client.models.generate_content.return_value = mock_resp

        generator = GeneratorClient(settings=settings)
        answer, model_used, fallback_triggered = generator.generate(
            query="Test query",
            retrieved_chunks=dummy_retrieval_chunks,
        )

        assert answer == "Grounded answer from primary model."
        assert model_used == "gemini-3.5-flash-lite"
        assert fallback_triggered is False
        mock_client.models.generate_content.assert_called_once()
        called_model = mock_client.models.generate_content.call_args.kwargs["model"]
        assert called_model == "gemini-3.5-flash-lite"


def test_primary_raises_429_triggers_fallback(
    dummy_retrieval_chunks: list[RetrievalResult],
) -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
        fallback_llm_model="gemini-3.1-flash-lite",
    )

    with patch("rag.generator.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        # Primary raises 429, fallback succeeds
        rate_limit_err = errors.APIError(429, {"error": "RESOURCE_EXHAUSTED"})
        fallback_resp = MagicMock()
        fallback_resp.text = "Answer generated via fallback model."

        mock_client.models.generate_content.side_effect = [
            rate_limit_err,
            fallback_resp,
        ]

        generator = GeneratorClient(settings=settings)
        answer, model_used, fallback_triggered = generator.generate(
            query="Test query",
            retrieved_chunks=dummy_retrieval_chunks,
        )

        assert answer == "Answer generated via fallback model."
        assert model_used == "gemini-3.1-flash-lite"
        assert fallback_triggered is True

        assert mock_client.models.generate_content.call_count == 2
        calls = mock_client.models.generate_content.call_args_list
        assert calls[0].kwargs["model"] == "gemini-3.5-flash-lite"
        assert calls[1].kwargs["model"] == "gemini-3.1-flash-lite"


def test_primary_raises_401_no_fallback(dummy_retrieval_chunks: list[RetrievalResult]) -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
        fallback_llm_model="gemini-3.1-flash-lite",
    )

    with patch("rag.generator.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        auth_err = errors.APIError(401, {"error": "UNAUTHENTICATED"})
        mock_client.models.generate_content.side_effect = auth_err

        generator = GeneratorClient(settings=settings)
        with pytest.raises(errors.APIError) as exc_info:
            generator.generate(
                query="Test query",
                retrieved_chunks=dummy_retrieval_chunks,
            )

        assert exc_info.value.code == 401
        # Fallback must NOT have been called
        mock_client.models.generate_content.assert_called_once()
        called_model = mock_client.models.generate_content.call_args.kwargs["model"]
        assert called_model == "gemini-3.5-flash-lite"


def test_both_primary_and_fallback_raise_429(dummy_retrieval_chunks: list[RetrievalResult]) -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
        fallback_llm_model="gemini-3.1-flash-lite",
    )

    with patch("rag.generator.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        err1 = errors.APIError(429, {"error": "RESOURCE_EXHAUSTED"})
        err2 = errors.APIError(429, {"error": "RESOURCE_EXHAUSTED"})
        mock_client.models.generate_content.side_effect = [err1, err2]

        generator = GeneratorClient(settings=settings)
        with pytest.raises(errors.APIError) as exc_info:
            generator.generate(
                query="Test query",
                retrieved_chunks=dummy_retrieval_chunks,
            )

        assert exc_info.value.code == 429
        assert mock_client.models.generate_content.call_count == 2


def test_primary_timeout_triggers_fallback(dummy_retrieval_chunks: list[RetrievalResult]) -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
        fallback_llm_model="gemini-3.1-flash-lite",
    )

    with patch("rag.generator.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        timeout_err = TimeoutError("Request timed out")
        fallback_resp = MagicMock()
        fallback_resp.text = "Answer from fallback after timeout."
        mock_client.models.generate_content.side_effect = [timeout_err, fallback_resp]

        generator = GeneratorClient(settings=settings)
        answer, model_used, fallback_triggered = generator.generate(
            query="Test query",
            retrieved_chunks=dummy_retrieval_chunks,
        )

        assert answer == "Answer from fallback after timeout."
        assert model_used == "gemini-3.1-flash-lite"
        assert fallback_triggered is True
        assert mock_client.models.generate_content.call_count == 2
