"""Unit tests for document loader supporting txt, md, and pdf."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.loader import compute_file_hash, load_document_text


def test_compute_file_hash(tmp_path: Path) -> None:
    test_file = tmp_path / "hash_sample.txt"
    test_file.write_bytes(b"Exact binary bytes for hash.")
    h1 = compute_file_hash(test_file)
    h2 = compute_file_hash(test_file)
    assert h1 == h2
    assert len(h1) == 64


def test_load_text_and_markdown(tmp_path: Path) -> None:
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Title\n\nContent paragraph.", encoding="utf-8")
    loaded = load_document_text(md_file)
    assert "# Title" in loaded
    assert "Content paragraph." in loaded


def test_load_pdf_mocked(tmp_path: Path) -> None:
    pdf_file = tmp_path / "mock.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 mock content")

    page1 = MagicMock()
    page1.extract_text.return_value = "Page 1 coverage details."
    page2 = MagicMock()
    page2.extract_text.return_value = "Page 2 exclusions and limits."

    with patch("rag.loader.pypdf.PdfReader") as mock_reader_cls:
        mock_reader = MagicMock()
        mock_reader.pages = [page1, page2]
        mock_reader_cls.return_value = mock_reader

        loaded = load_document_text(pdf_file)

        assert "[Page 1]\nPage 1 coverage details." in loaded
        assert "[Page 2]\nPage 2 exclusions and limits." in loaded


def test_unsupported_file_extension_raises(tmp_path: Path) -> None:
    doc = tmp_path / "unsupported.docx"
    doc.write_bytes(b"PK")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document_text(doc)
