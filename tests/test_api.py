"""Unit tests for production FastAPI REST service."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag.agent import AgentResponse
from rag.api import _ingest_lock, _sessions, app
from rag.schemas import Chunk, RAGResponse, RetrievalResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear memory sessions and locks between tests."""
    _sessions.clear()
    if _ingest_lock.locked():
        try:
            _ingest_lock.release()
        except RuntimeError:
            pass


def test_health_endpoint_healthy() -> None:
    with (
        patch("rag.api.get_settings") as mock_settings,
        patch("rag.api.get_pipeline") as mock_pipe_fn,
    ):
        mock_settings.return_value.primary_llm_model = "gemini-3.5-flash-lite"
        mock_settings.return_value.fallback_llm_model = "gemini-3.1-flash-lite"
        mock_settings.return_value.primary_embedding_model = "gemini-embedding-2"

        mock_pipe = MagicMock()
        mock_pipe.vector_store.primary_collection = "rag_chunks_gemini_embedding_2"
        mock_pipe.vector_store.client.collection_exists.return_value = True
        mock_pipe_fn.return_value = mock_pipe

        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["qdrant"]["connected"] is True
        assert data["models"]["primary_llm"] == "gemini-3.5-flash-lite"


def test_health_endpoint_degraded() -> None:
    with (
        patch("rag.api.get_settings"),
        patch("rag.api.get_pipeline") as mock_pipe_fn,
    ):
        mock_pipe = MagicMock()
        mock_pipe.vector_store.primary_collection = "rag_chunks_gemini_embedding_2"
        mock_pipe.vector_store.client.collection_exists.side_effect = Exception("Connection lost")
        mock_pipe_fn.return_value = mock_pipe

        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["qdrant"]["connected"] is False


def test_catalog_endpoint(tmp_path: Path) -> None:
    with (
        patch("rag.api.load_catalog") as mock_load_cat,
        patch("rag.api.load_company_profile") as mock_load_prof,
    ):
        entry_mock = MagicMock()
        entry_mock.model_dump.return_value = {
            "doc_id": "test-doc",
            "filename": "test-doc.pdf",
            "use_case": "general",
            "chunk_count": 5,
            "preview": "Test preview",
        }
        mock_load_cat.return_value = [entry_mock]
        mock_load_prof.return_value = "# Test Company Scope"

        resp = client.get("/v1/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["catalog"]) == 1
        assert data["catalog"][0]["doc_id"] == "test-doc"
        assert "Test Company Scope" in data["company_profile"]


def test_query_endpoint() -> None:
    with patch("rag.api.get_pipeline") as mock_pipe_fn:
        mock_pipe = MagicMock()
        dummy_chunk = Chunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            chunk_index=0,
            text="Grounded content.",
            start_char=0,
            end_char=17,
            embedding_model="gemini-embedding-2",
            use_case="general",
        )
        mock_rag_resp = RAGResponse(
            query="What is this?",
            answer="Based on [doc-1:0], this is grounded content.",
            retrieved_chunks=[RetrievalResult(chunk=dummy_chunk, score=0.88)],
            generation_model_used="gemini-3.5-flash-lite",
            fallback_triggered=False,
            embedding_model_used="gemini-embedding-2",
        )
        mock_pipe.query.return_value = mock_rag_resp
        mock_pipe_fn.return_value = mock_pipe

        resp = client.post(
            "/v1/query",
            json={"query": "What is this?", "use_case": "general", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "[doc-1:0]" in data["answer"]
        assert data["generation_model_used"] == "gemini-3.5-flash-lite"
        assert len(data["retrieved_chunks"]) == 1


def test_chat_endpoint_new_session_and_continuity() -> None:
    with patch("rag.api.get_agent") as mock_agent_fn:
        mock_agent = MagicMock()

        # Turn 1
        resp1 = AgentResponse(
            query="Hello",
            answer="Hello! I am your knowledge assistant.",
            steps_taken=1,
            tools_called=[],
            model_used="gemini-3.5-flash-lite",
            fallback_triggered=False,
        )
        mock_agent.run.return_value = resp1
        mock_agent.chat_history = [("Hello", "Hello! I am your knowledge assistant.")]
        mock_agent_fn.return_value = mock_agent

        # Turn 1 without session_id
        res1 = client.post("/v1/chat", json={"query": "Hello"})
        assert res1.status_code == 200
        data1 = res1.json()
        session_id = data1["session_id"]
        assert session_id is not None
        assert "knowledge assistant" in data1["answer"]

        # Turn 2 reusing session_id
        resp2 = AgentResponse(
            query="What are the ICU limits?",
            answer="ICU is covered at actuals [doc:0].",
            steps_taken=2,
            tools_called=["search_use_case_kb"],
            model_used="gemini-3.5-flash-lite",
            fallback_triggered=False,
        )
        mock_agent.run.return_value = resp2
        mock_agent.chat_history = [
            ("Hello", "Hello! I am your knowledge assistant."),
            ("What are the ICU limits?", "ICU is covered at actuals [doc:0]."),
        ]

        res2 = client.post(
            "/v1/chat",
            json={"query": "What are the ICU limits?", "session_id": session_id},
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["session_id"] == session_id
        assert data2["tools_called"] == ["search_use_case_kb"]

        # Verify agent.run received the prior turn history
        call_kwargs = mock_agent.run.call_args.kwargs
        assert call_kwargs["chat_history"] == [("Hello", "Hello! I am your knowledge assistant.")]


def test_ingest_endpoint_enqueue_and_conflict(tmp_path: Path) -> None:
    valid_dir = tmp_path / "raw"
    valid_dir.mkdir()

    # 1. Invalid directory returns 400
    res_400 = client.post("/v1/ingest", json={"dir": "non_existent_folder_xyz"})
    assert res_400.status_code == 400

    # 2. Valid directory enqueues with 202 Accepted
    with patch("rag.api.execute_ingest_background"):
        res_202 = client.post("/v1/ingest", json={"dir": str(valid_dir)})
        assert res_202.status_code == 202
        data = res_202.json()
        assert data["status"] == "accepted"

    # 3. Concurrent ingest returns 409 Conflict
    _ingest_lock.acquire()
    try:
        res_409 = client.post("/v1/ingest", json={"dir": str(valid_dir)})
        assert res_409.status_code == 409
        assert "already running" in res_409.json()["detail"]
    finally:
        _ingest_lock.release()
