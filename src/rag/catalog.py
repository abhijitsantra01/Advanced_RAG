"""Knowledge Base Catalog ('System Memory') manager.

Maintains data/kb_catalog.json and injects catalog context into agent/generator prompts.
"""
import json
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CATALOG_PATH = Path("data/kb_catalog.json")
DEFAULT_PROFILE_PATH = Path("data/company_profile.md")

DEFAULT_COMPANY_PROFILE = """# Enterprise Knowledge Scope & Company Profile
Organization: Enterprise Insurance & Technical Systems
Role: Official AI Knowledge Assistant
Scope:
- Inquiries regarding active health insurance policies, benefits, hospitalization coverage,
  room rent, ICU terms, and claim requirements.
- Technical architecture specifications, failover configurations, and system design notes.
- Strictly grounded in indexed enterprise documents. Never speculate or hallucinate topics outside
  indexed domains.
"""


class CatalogEntry(BaseModel):
    """Metadata summary of an indexed document in the knowledge base."""

    doc_id: str
    filename: str
    use_case: str
    chunk_count: int = Field(ge=0)
    preview: str


def load_catalog(catalog_path: Path | None = None) -> list[CatalogEntry]:
    """Load catalog entries from JSON file, returning empty list if missing or invalid."""
    path = catalog_path or DEFAULT_CATALOG_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [CatalogEntry.model_validate(item) for item in data]
    except Exception:
        return []


def save_catalog(entries: list[CatalogEntry], catalog_path: Path | None = None) -> None:
    """Save catalog entries to JSON file atomically with directory creation."""
    path = catalog_path or DEFAULT_CATALOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [entry.model_dump() for entry in entries]
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def record_catalog_entry(
    doc_id: str,
    filename: str,
    use_case: str,
    chunk_count: int,
    preview: str,
    catalog_path: Path | None = None,
) -> None:
    """Insert or update a catalog entry in data/kb_catalog.json."""
    entries = load_catalog(catalog_path)
    # Clean preview to single line or first 300 chars
    cleaned_preview = " ".join(preview[:300].split())

    updated = False
    for idx, item in enumerate(entries):
        if item.doc_id == doc_id:
            entries[idx] = CatalogEntry(
                doc_id=doc_id,
                filename=filename,
                use_case=use_case,
                chunk_count=chunk_count,
                preview=cleaned_preview,
            )
            updated = True
            break

    if not updated:
        entries.append(
            CatalogEntry(
                doc_id=doc_id,
                filename=filename,
                use_case=use_case,
                chunk_count=chunk_count,
                preview=cleaned_preview,
            )
        )

    save_catalog(entries, catalog_path)


def load_company_profile(profile_path: Path | None = None) -> str:
    """Load company profile markdown or return default profile text."""
    path = profile_path or DEFAULT_PROFILE_PATH
    if path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return DEFAULT_COMPANY_PROFILE.strip()
    return DEFAULT_COMPANY_PROFILE.strip()


def build_system_memory(
    catalog_path: Path | None = None,
    profile_path: Path | None = None,
) -> str:
    """Build injected memory combining company profile and document catalog summaries."""
    profile_text = load_company_profile(profile_path)
    entries = load_catalog(catalog_path)

    catalog_lines: list[str] = []
    if entries:
        for item in entries:
            catalog_lines.append(
                f"- Document: {item.filename} (ID: {item.doc_id}, Domain: '{item.use_case}', "
                f"Chunks: {item.chunk_count})\n"
                f"  Summary / Preview: {item.preview}"
            )
    else:
        catalog_lines.append("- No documents currently indexed in knowledge base catalog.")

    catalog_text = "\n".join(catalog_lines)

    return (
        f"{profile_text}\n\n"
        f"## Active Knowledge Base Catalog (Indexed Documents):\n"
        f"{catalog_text}"
    )
