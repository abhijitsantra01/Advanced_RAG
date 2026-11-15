"""Document loader supporting .txt, .md, and .pdf documents with SHA-256 caching.

See execution rules.
"""
import hashlib
from pathlib import Path

import logfire
import pypdf


def compute_file_hash(path: Path) -> str:
    """Compute deterministic SHA-256 hex digest over raw file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_document_text(path: Path) -> str:
    """Load text content from .txt, .md, or .pdf file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        reader = pypdf.PdfReader(str(file_path))
        with logfire.span("extract_pdf", file=file_path.name, pages=len(reader.pages)):
            pages_text: list[str] = []
            for idx, page in enumerate(reader.pages):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages_text.append(f"[Page {idx + 1}]\n{page_text}")
            return "\n\n".join(pages_text)

    raise ValueError(f"Unsupported file format: {suffix}. Supported: .txt, .md, .pdf")
