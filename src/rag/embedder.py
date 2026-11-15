"""Gemini Embedding client with automatic quota failover, rate pacing, and retry.

See docs/EXECUTION_PLAN.md Level 3.
"""
import re
import time

import logfire
from google import genai
from google.genai import errors, types

from rag.config import RAGSettings


class EmbeddingFailoverRequired(Exception):
    """Raised when primary embedding quota is exhausted and corpus re-ingestion is required."""

    def __init__(self, fallback_model: str, message: str = "") -> None:
        self.fallback_model = fallback_model
        default_msg = (
            f"Embedding quota exhausted on primary model; failover to {fallback_model} required."
        )
        super().__init__(message or default_msg)


class EmbedderClient:
    """Client for generating vector embeddings via Gemini with quota failover."""

    def __init__(self, settings: RAGSettings | None = None) -> None:
        self.settings = settings or RAGSettings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.active_model: str = self.settings.primary_embedding_model
        self.fallback_model: str = self.settings.fallback_embedding_model
        self.dimension: int = self.settings.embedding_dimension

    def _extract_retry_delay(self, err: Exception) -> float:
        """Extract Google API retry delay from error message, returning 0.0 if not found."""
        match = re.search(r"retry in ([\d\.]+)s", str(err))
        if match:
            return min(float(match.group(1)) + 0.5, 60.0)
        return 0.0

    def _embed_single(self, text: str, task_type: str, max_retries: int = 3) -> list[float]:
        for attempt in range(max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.active_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimension,
                    ),
                )
                if response.embeddings and response.embeddings[0].values:
                    return list(response.embeddings[0].values)
                return []
            except errors.APIError as err:
                if err.code == 429:
                    delay = self._extract_retry_delay(err)
                    if delay > 0 and attempt < max_retries - 1:
                        logfire.warn(
                            "Rate limit 429 encountered; pausing for quota reset",
                            model=self.active_model,
                            retry_delay_seconds=delay,
                            attempt=attempt + 1,
                        )
                        time.sleep(delay)
                        continue

                    if self.active_model != self.fallback_model:
                        old_model = self.active_model
                        self.active_model = self.fallback_model
                        if self.fallback_model == "gemini-embedding-001":
                            self.dimension = 768

                        logfire.error(
                            "Embedding quota exhausted; switching to fallback model",
                            primary_model=old_model,
                            fallback_model=self.fallback_model,
                            status_code=err.code,
                            error=str(err),
                        )
                        raise EmbeddingFailoverRequired(fallback_model=self.fallback_model) from err
                raise
        return []

    def embed_texts(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 50,
    ) -> list[list[float]]:
        """Embed a list of text strings with rate pacing and return list of vector float lists."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        with logfire.span(
            "embed_texts",
            model=self.active_model,
            count=len(texts),
            task_type=task_type,
            batch_size=batch_size,
        ):
            for idx, text in enumerate(texts):
                vec = self._embed_single(text, task_type)
                all_embeddings.append(vec)
                # Polite pacing to stay under free tier 100 RPM limit
                if len(texts) > 5 and idx < len(texts) - 1:
                    time.sleep(0.7)

        return all_embeddings
