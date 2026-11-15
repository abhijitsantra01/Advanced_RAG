"""Enterprise Agentic RAG Controller with tool use, router mode, and multi-turn memory.

See docs/EXECUTION_PLAN.md Level 5.
"""
import logfire
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from rag.catalog import build_system_memory
from rag.config import RAGSettings
from rag.tools import (
    RAGTools,
    get_adjacent_chunks,
    list_available_use_cases,
    search_use_case_kb,
    set_default_tools,
)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def get_router_system_instruction() -> str:
    """Build dynamic system instruction for router mode with active catalog memory."""
    memory = build_system_memory()
    return (
        f"{memory}\n\n"
        "### Assistant Router Directives:\n"
        "You have access to the tool `search_use_case_kb` to retrieve context from the "
        "enterprise knowledge base.\n"
        "1. For greetings, pleasantries, or general conversational questions (e.g. 'hi', "
        "'who are you'), reply politely and directly without calling any tools.\n"
        "2. When the user asks broad questions about what you know, what policies or documents "
        "are available, or how you can help (e.g. 'What documents do you have?',\n"
        "'How can you help me?'), answer directly using the Active Knowledge Base Catalog and "
        "Company Profile above WITHOUT calling any tools. Never invent unindexed topics.\n"
        "3. When the user asks about ANY document, policy, rules, coverage, limits, or "
        "architecture (even broadly or vaguely, such as 'tell me about the policy', "
        "'room rent rules', 'ICU limits'), YOU MUST CALL `search_use_case_kb`.\n"
        "4. Prior conversation turns are provided in the chat history. When the user asks a "
        "follow-up question (e.g. using pronouns or references like 'what about for 5 days?', "
        "'does it cover dental?'), resolve the subject from conversation history when "
        "calling `search_use_case_kb`.\n"
        "5. When answering from retrieved context, cite every fact using [doc_id:chunk_index]. "
        "If the context does not contain the answer, state that clearly without guessing."
    )


def get_agent_system_instruction() -> str:
    """Build dynamic system instruction for multi-step agent mode."""
    memory = build_system_memory()
    return (
        f"{memory}\n\n"
        "### Autonomous Agent Directives:\n"
        "You have access to tools to search knowledge bases, list domains, "
        "and fetch adjacent chunks.\n"
        "1. Identify the relevant domain and call search_use_case_kb.\n"
        "2. If retrieved chunks are incomplete or context is cut off, call get_adjacent_chunks.\n"
        "3. If recall is low or empty, refine your query and search again.\n"
        "4. Prior conversation turns are provided in the chat history. Resolve follow-up pronouns "
        "from context when formulating tool queries.\n"
        "5. Synthesize a grounded final response citing every fact using [doc_id:chunk_index].\n"
        "6. If information is missing, state clearly what is missing without guessing."
    )


class AgentResponse(BaseModel):
    """Structured response from the Agentic RAG Controller."""

    query: str
    answer: str
    steps_taken: int
    tools_called: list[str]
    model_used: str
    fallback_triggered: bool = False


class AgenticRAG:
    """Agentic RAG controller with multi-turn memory and automatic failover."""

    def __init__(
        self,
        settings: RAGSettings | None = None,
        tools: RAGTools | None = None,
        max_steps: int = 3,
        simple_router: bool = True,
        max_history_turns: int = 5,
    ) -> None:
        self.settings = settings or RAGSettings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.primary_model: str = self.settings.primary_llm_model
        self.fallback_model: str = self.settings.fallback_llm_model
        self.tools: RAGTools = tools or RAGTools(settings=self.settings)
        set_default_tools(self.tools)
        self.simple_router: bool = simple_router
        self.max_steps: int = 1 if simple_router else max_steps
        self.max_history_turns: int = max_history_turns
        self.chat_history: list[tuple[str, str]] = []

    def clear_history(self) -> None:
        """Clear rolling multi-turn conversation memory."""
        self.chat_history.clear()

    def _build_initial_contents(self, prompt: str) -> list[types.Content]:
        """Build message contents prepending clean (user_q, final_ans) history turns."""
        contents: list[types.Content] = []
        for user_q, model_ans in self.chat_history[-self.max_history_turns :]:
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=user_q)])
            )
            contents.append(
                types.Content(role="model", parts=[types.Part.from_text(text=model_ans)])
            )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        )
        return contents

    def _record_turn(self, prompt: str, answer: str) -> None:
        """Append Q&A pair to chat history bounded by max_history_turns."""
        if answer.strip():
            self.chat_history.append((prompt, answer))
            if len(self.chat_history) > self.max_history_turns:
                self.chat_history = self.chat_history[-self.max_history_turns :]

    def run(
        self,
        prompt: str,
        chat_history: list[tuple[str, str]] | None = None,
    ) -> AgentResponse:
        """Run the agent router or loop with automatic model fallback and history tracking."""
        if chat_history is not None:
            self.chat_history = list(chat_history[-self.max_history_turns :])

        runner = self._run_simple_router if self.simple_router else self._run_loop
        with logfire.span(
            "agentic_rag_run",
            prompt=prompt,
            simple_router=self.simple_router,
            max_steps=self.max_steps,
            history_turns=len(self.chat_history),
        ):
            try:
                resp = runner(prompt, model=self.primary_model, is_fallback=False)
            except errors.APIError as err:
                if err.code not in RETRYABLE_STATUS_CODES:
                    raise
                logfire.warn(
                    "Primary agent LLM failed; switching to fallback",
                    primary_model=self.primary_model,
                    fallback_model=self.fallback_model,
                    status_code=getattr(err, "code", None),
                    error=str(err),
                )
                resp = runner(prompt, model=self.fallback_model, is_fallback=True)
            except (TimeoutError, ConnectionError) as err:
                logfire.warn(
                    "Primary agent LLM failed; switching to fallback",
                    primary_model=self.primary_model,
                    fallback_model=self.fallback_model,
                    status_code=getattr(err, "code", None),
                    error=str(err),
                )
                resp = runner(prompt, model=self.fallback_model, is_fallback=True)

            self._record_turn(prompt, resp.answer)
            return resp

    def _run_simple_router(self, prompt: str, model: str, is_fallback: bool) -> AgentResponse:
        """Simple router mode: chat directly OR make a single search_use_case_kb call."""
        contents = self._build_initial_contents(prompt)
        tools_list = [search_use_case_kb]
        tool_map = {"search_use_case_kb": self.tools.search_use_case_kb}
        system_instruction = get_router_system_instruction()

        with logfire.span("agent_router_turn", model=model):
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools_list,
                    system_instruction=system_instruction,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )

            function_calls = getattr(response, "function_calls", None) or []
            if not function_calls:
                # Direct conversational or catalog response (0 tool calls)
                return AgentResponse(
                    query=prompt,
                    answer=response.text or "",
                    steps_taken=1,
                    tools_called=[],
                    model_used=model,
                    fallback_triggered=is_fallback,
                )

            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)

            tools_called: list[str] = []
            response_parts: list[types.Part] = []
            for fc in function_calls:
                name = fc.name
                tools_called.append(name)
                args = fc.args or {}
                fn = tool_map.get(name)

                with logfire.span("agent_tool_exec", tool=name, args=args):
                    if fn:
                        try:
                            tool_result = fn(**args)
                        except Exception as err:
                            tool_result = {"error": str(err)}
                    else:
                        tool_result = {"error": f"Unknown tool: {name}"}

                    response_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"result": tool_result},
                        )
                    )

            contents.append(types.Content(role="user", parts=response_parts))

            # Grounded answer synthesis after search_use_case_kb
            with logfire.span("agent_router_synthesis", model=model):
                final_resp = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
                return AgentResponse(
                    query=prompt,
                    answer=final_resp.text or "",
                    steps_taken=2,
                    tools_called=tools_called,
                    model_used=model,
                    fallback_triggered=is_fallback,
                )

    def _run_loop(self, prompt: str, model: str, is_fallback: bool) -> AgentResponse:
        """Multi-step agent loop with adjacent chunks and query refinement."""
        contents = self._build_initial_contents(prompt)
        tools_list = [
            search_use_case_kb,
            list_available_use_cases,
            get_adjacent_chunks,
        ]
        tool_map = {
            "search_use_case_kb": self.tools.search_use_case_kb,
            "list_available_use_cases": self.tools.list_available_use_cases,
            "get_adjacent_chunks": self.tools.get_adjacent_chunks,
        }
        system_instruction = get_agent_system_instruction()

        tools_called: list[str] = []
        steps_taken = 0

        for step in range(1, self.max_steps + 1):
            steps_taken = step
            with logfire.span("agent_step", step=step, model=model):
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=tools_list,
                        system_instruction=system_instruction,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )

                function_calls = getattr(response, "function_calls", None) or []

                if not function_calls:
                    # Model produced final text response without tool call
                    return AgentResponse(
                        query=prompt,
                        answer=response.text or "",
                        steps_taken=steps_taken,
                        tools_called=tools_called,
                        model_used=model,
                        fallback_triggered=is_fallback,
                    )

                # Append model content with function calls to history
                if response.candidates and response.candidates[0].content:
                    contents.append(response.candidates[0].content)

                # Execute requested function calls
                response_parts: list[types.Part] = []
                for fc in function_calls:
                    name = fc.name
                    tools_called.append(name)
                    args = fc.args or {}
                    fn = tool_map.get(name)

                    with logfire.span("agent_tool_exec", tool=name, args=args):
                        if fn:
                            try:
                                tool_result = fn(**args)
                            except Exception as err:
                                tool_result = {"error": str(err)}
                        else:
                            tool_result = {"error": f"Unknown tool: {name}"}

                        response_parts.append(
                            types.Part.from_function_response(
                                name=name,
                                response={"result": tool_result},
                            )
                        )

                contents.append(types.Content(role="user", parts=response_parts))

                # Step limit reached: force final synthesis
                if step == self.max_steps:
                    with logfire.span("agent_forced_synthesis", step=step):
                        final_prompt = types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(
                                    text=(
                                        "Step budget reached. Synthesize final answer now "
                                        "citing [doc_id:chunk_index]."
                                    )
                                )
                            ],
                        )
                        contents.append(final_prompt)
                        final_resp = self.client.models.generate_content(
                            model=model,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                    disable=True
                                ),
                            ),
                        )
                        return AgentResponse(
                            query=prompt,
                            answer=final_resp.text or "",
                            steps_taken=steps_taken,
                            tools_called=tools_called,
                            model_used=model,
                            fallback_triggered=is_fallback,
                        )

        return AgentResponse(
            query=prompt,
            answer="",
            steps_taken=steps_taken,
            tools_called=tools_called,
            model_used=model,
            fallback_triggered=is_fallback,
        )
