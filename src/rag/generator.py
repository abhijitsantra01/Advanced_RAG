"""Gemini grounded answer generator with resilient fallback.

See docs/EXECUTION_PLAN.md Level 2.
"""
import logfire
from google import genai
from google.genai import errors, types

from rag.catalog import build_system_memory
from rag.config import RAGSettings
from rag.schemas import RetrievalResult


def get_generator_system_instruction() -> str:
    """Build dynamic system instruction for generator with active catalog memory."""
    memory = build_system_memory()
    return (
        f"{memory}\n\n"
        "### Grounded Answer Directives:\n"
        "You are a precise enterprise knowledge assistant. Answer the user's question using "
        "ONLY the provided context chunks. Cite every fact using [doc_id:chunk_index]. "
        "If the context does not contain sufficient information to answer the question, "
        "state clearly what is missing without guessing."
    )


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class GeneratorClient:
    """Grounded LLM generator with automatic failover from primary to fallback model."""

    def __init__(self, settings: RAGSettings | None = None) -> None:
        self.settings = settings or RAGSettings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.primary_model: str = self.settings.primary_llm_model
        self.fallback_model: str = self.settings.fallback_llm_model
        self.model: str = self.primary_model

    def _call_model(self, model: str, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=get_generator_system_instruction(),
            ),
        )
        return response.text or ""

    def generate(
        self, query: str, retrieved_chunks: list[RetrievalResult]
    ) -> tuple[str, str, bool]:
        """Generate a grounded response, failing over to fallback model on retryable errors.

        Returns:
            tuple[str, str, bool]: (answer_text, model_used, fallback_triggered)
        """
        context_blocks: list[str] = []
        for res in retrieved_chunks:
            c = res.chunk
            context_blocks.append(f"[{c.doc_id}:{c.chunk_index}] {c.text}")

        context_str = "\n\n".join(context_blocks)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        with logfire.span(
            "generate_answer",
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            context_chunks=len(retrieved_chunks),
        ):
            try:
                with logfire.span("generate_attempt", model=self.primary_model):
                    answer = self._call_model(self.primary_model, prompt)
                    return answer, self.primary_model, False
            except errors.APIError as err:
                if err.code not in RETRYABLE_STATUS_CODES:
                    raise
                logfire.warn(
                    "Primary LLM failed; switching to fallback",
                    primary_model=self.primary_model,
                    fallback_model=self.fallback_model,
                    status_code=getattr(err, "code", None),
                    error=str(err),
                )
            except (TimeoutError, ConnectionError) as err:
                logfire.warn(
                    "Primary LLM failed; switching to fallback",
                    primary_model=self.primary_model,
                    fallback_model=self.fallback_model,
                    status_code=getattr(err, "code", None),
                    error=str(err),
                )

            # Fallback attempt
            with logfire.span("generate_attempt", model=self.fallback_model, is_fallback=True):
                answer = self._call_model(self.fallback_model, prompt)
                return answer, self.fallback_model, True
