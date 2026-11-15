"""Observability and distributed tracing configuration using Pydantic Logfire.

See docs/LEVEL_1_SPEC.md Section 1.
"""
import logfire

from rag.config import RAGSettings


def setup_observability(settings: RAGSettings | None = None) -> None:
    """Configure Pydantic Logfire with fallback to local tracing if token is unset."""
    if settings is None:
        settings = RAGSettings()

    logfire.configure(
        token=settings.logfire_token,
        send_to_logfire="if-token-present",
    )
