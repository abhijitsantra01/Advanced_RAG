"""Baseline unit tests for deterministic fixed chunking and overlap."""
import pytest

from rag.chunker import FixedChunker
from rag.schemas import Document


def test_fixed_chunker_overlap_and_determinism() -> None:
    doc = Document(
        doc_id="doc-001",
        title="Test Doc",
        content="A" * 500 + "B" * 500 + "C" * 300,
        source_path="data/raw/test.md",
        use_case="technical",
    )
    chunker = FixedChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_document(doc, embedding_model="gemini-embedding-2")

    assert len(chunks) == 2
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 800
    assert chunks[1].start_char == 650  # 800 - 150 overlap
    assert chunks[1].end_char == 1300
    # Verify exact 150-char overlap content matches
    assert chunks[0].text[-150:] == chunks[1].text[:150]


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=500, chunk_overlap=500)
