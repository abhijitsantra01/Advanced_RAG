"""Unit tests for Level 5 Enterprise Agentic RAG Controller."""
from unittest.mock import MagicMock, patch

from google.genai import errors, types

from rag.agent import AgenticRAG
from rag.config import RAGSettings
from rag.tools import RAGTools


def _make_mock_fc_response(tool_name: str, args: dict) -> MagicMock:
    fc = types.FunctionCall(name=tool_name, args=args)
    candidate = MagicMock()
    candidate.content = types.Content(
        role="model",
        parts=[types.Part(function_call=fc)],
    )
    resp = MagicMock()
    resp.function_calls = [fc]
    resp.candidates = [candidate]
    resp.text = None
    return resp


def _make_mock_text_response(text: str) -> MagicMock:
    candidate = MagicMock()
    candidate.content = types.Content(
        role="model",
        parts=[types.Part.from_text(text=text)],
    )
    resp = MagicMock()
    resp.function_calls = []
    resp.candidates = [candidate]
    resp.text = text
    return resp


def test_agent_simple_router_chat_direct_no_tools() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
    )

    with patch("rag.agent.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        # Direct conversational answer (no function calls)
        resp = _make_mock_text_response("Hello! I am your enterprise knowledge assistant.")
        mock_client.models.generate_content.return_value = resp

        mock_tools = MagicMock(spec=RAGTools)
        agent = AgenticRAG(settings=settings, tools=mock_tools, simple_router=True)
        res = agent.run("Hello, who are you?")

        assert res.steps_taken == 1
        assert res.tools_called == []
        assert "enterprise knowledge assistant" in res.answer
        assert res.model_used == "gemini-3.5-flash-lite"
        mock_tools.search_use_case_kb.assert_not_called()


def test_agent_selects_tool_and_synthesizes_cited_answer() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
        fallback_llm_model="gemini-3.1-flash-lite",
    )

    with patch("rag.agent.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        # Turn 1: model calls search_use_case_kb
        # Turn 2: model synthesizes final answer
        resp1 = _make_mock_fc_response(
            "search_use_case_kb",
            {"query": "architecture note", "use_case": "technical"},
        )
        resp2 = _make_mock_text_response(
            "Based on [arch-note:0], the system utilizes Qdrant Cloud."
        )
        mock_client.models.generate_content.side_effect = [resp1, resp2]

        mock_tools = MagicMock(spec=RAGTools)
        mock_tools.search_use_case_kb.return_value = [
            {"doc_id": "arch-note", "chunk_index": 0, "text": "Qdrant Cloud vectors."}
        ]
        mock_tools.list_available_use_cases.return_value = ["technical"]
        mock_tools.get_adjacent_chunks.return_value = []

        agent = AgenticRAG(settings=settings, tools=mock_tools, simple_router=True)
        res = agent.run("What vector DB is used in architecture?")

        assert res.steps_taken == 2
        assert res.tools_called == ["search_use_case_kb"]
        assert "[arch-note:0]" in res.answer
        assert res.model_used == "gemini-3.5-flash-lite"
        assert res.fallback_triggered is False
        mock_tools.search_use_case_kb.assert_called_once_with(
            query="architecture note", use_case="technical"
        )


def test_agent_query_refinement_on_empty_recall() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
    )

    with patch("rag.agent.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        # Turn 1: model calls tool with ambiguous query -> returns empty
        resp1 = _make_mock_fc_response(
            "search_use_case_kb",
            {"query": "fuzzy keyword", "use_case": "technical"},
        )
        # Turn 2: model refines query -> returns matching chunk
        resp2 = _make_mock_fc_response(
            "search_use_case_kb",
            {"query": "precise architecture", "use_case": "technical"},
        )
        # Turn 3: model synthesizes answer
        resp3 = _make_mock_text_response("Grounded answer from [doc-1:0].")
        mock_client.models.generate_content.side_effect = [resp1, resp2, resp3]

        mock_tools = MagicMock(spec=RAGTools)
        mock_tools.search_use_case_kb.side_effect = [
            [],  # Empty initial recall
            [{"doc_id": "doc-1", "chunk_index": 0, "text": "Precise content."}],
        ]

        agent = AgenticRAG(settings=settings, tools=mock_tools, max_steps=3, simple_router=False)
        res = agent.run("How does system scale?")

        assert res.steps_taken == 3
        assert len(res.tools_called) == 2
        assert mock_tools.search_use_case_kb.call_count == 2
        assert "[doc-1:0]" in res.answer


def test_agent_max_step_termination() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
    )

    with patch("rag.agent.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        # Model requests tools on all steps, forcing final synthesis
        resp1 = _make_mock_fc_response("list_available_use_cases", {})
        resp2 = _make_mock_fc_response("search_use_case_kb", {"query": "q", "use_case": "tech"})
        resp3 = _make_mock_fc_response("get_adjacent_chunks", {"doc_id": "d1", "chunk_index": 0})
        forced_final_resp = _make_mock_text_response(
            "Forced termination response with [d1:0]."
        )
        mock_client.models.generate_content.side_effect = [
            resp1,
            resp2,
            resp3,
            forced_final_resp,
        ]

        mock_tools = MagicMock(spec=RAGTools)
        mock_tools.list_available_use_cases.return_value = ["tech"]
        mock_tools.search_use_case_kb.return_value = []
        mock_tools.get_adjacent_chunks.return_value = []

        agent = AgenticRAG(settings=settings, tools=mock_tools, max_steps=3, simple_router=False)
        res = agent.run("Perform complex multi-step search")

        assert res.steps_taken == 3
        assert len(res.tools_called) == 3
        assert "Forced termination" in res.answer


def test_adjacent_chunks_tool() -> None:
    settings = RAGSettings(gemini_api_key="test-key")
    with (
        patch("rag.tools.EmbedderClient"),
        patch("rag.tools.QdrantStore") as mock_store_cls,
    ):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        rec0 = MagicMock()
        rec0.payload = {"doc_id": "doc-a", "chunk_index": 0, "text": "First chunk"}
        rec1 = MagicMock()
        rec1.payload = {"doc_id": "doc-a", "chunk_index": 1, "text": "Middle chunk"}
        rec2 = MagicMock()
        rec2.payload = {"doc_id": "doc-a", "chunk_index": 2, "text": "Trailing chunk"}

        mock_store.client.scroll.return_value = ([rec0, rec1, rec2], None)

        tools = RAGTools(settings=settings, vector_store=mock_store)
        res = tools.get_adjacent_chunks(doc_id="doc-a", chunk_index=1, window=1)

        assert len(res) == 3
        assert [c["chunk_index"] for c in res] == [0, 1, 2]


def test_agent_fallback_on_429() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
        fallback_llm_model="gemini-3.1-flash-lite",
    )

    with patch("rag.agent.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        # Primary model raises 429 immediately
        primary_err = errors.APIError(429, {"error": "RESOURCE_EXHAUSTED"})
        fallback_resp = _make_mock_text_response("Fallback model answered directly.")

        mock_client.models.generate_content.side_effect = [primary_err, fallback_resp]

        mock_tools = MagicMock(spec=RAGTools)
        agent = AgenticRAG(settings=settings, tools=mock_tools, max_steps=3)
        res = agent.run("Simple question")

        assert res.model_used == "gemini-3.1-flash-lite"
        assert res.fallback_triggered is True
        assert res.answer == "Fallback model answered directly."


def test_agent_multi_turn_history_preservation_and_cap() -> None:
    settings = RAGSettings(
        gemini_api_key="test-key",
        primary_llm_model="gemini-3.5-flash-lite",
    )

    with patch("rag.agent.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        mock_tools = MagicMock(spec=RAGTools)
        agent = AgenticRAG(
            settings=settings,
            tools=mock_tools,
            simple_router=True,
            max_history_turns=5,
        )

        # Turn 1
        resp1 = _make_mock_text_response("ICU expenses are covered at actuals.")
        mock_client.models.generate_content.return_value = resp1
        res1 = agent.run("What are the ICU limits?")
        assert res1.answer == "ICU expenses are covered at actuals."
        assert len(agent.chat_history) == 1

        # Turn 2: Follow-up question
        resp2 = _make_mock_text_response("For 5 days, ICU stays remain covered at actuals.")
        mock_client.models.generate_content.return_value = resp2
        res2 = agent.run("What about for 5 days?")
        assert res2.answer == "For 5 days, ICU stays remain covered at actuals."
        assert len(agent.chat_history) == 2

        # Verify that Turn 2's request contents included Turn 1 user query and model answer
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        sent_contents = call_kwargs["contents"]
        assert len(sent_contents) == 3  # (user turn 1, model turn 1, user turn 2)
        assert sent_contents[0].parts[0].text == "What are the ICU limits?"
        assert sent_contents[1].parts[0].text == "ICU expenses are covered at actuals."
        assert sent_contents[2].parts[0].text == "What about for 5 days?"

        # Verify capping at max_history_turns=5
        for i in range(3, 8):
            mock_client.models.generate_content.return_value = _make_mock_text_response(
                f"Answer {i}"
            )
            agent.run(f"Question {i}")

        assert len(agent.chat_history) == 5
        # The oldest turn ("What are the ICU limits?") should have been shifted out
        assert agent.chat_history[0][0] == "Question 3"
        assert agent.chat_history[-1][0] == "Question 7"

        # Verify clear_history()
        agent.clear_history()
        assert len(agent.chat_history) == 0

