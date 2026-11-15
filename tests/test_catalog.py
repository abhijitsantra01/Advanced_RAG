"""Unit tests for Knowledge Base Catalog ('System Memory')."""
from pathlib import Path

from rag.catalog import (
    CatalogEntry,
    build_system_memory,
    load_catalog,
    record_catalog_entry,
    save_catalog,
)


def test_catalog_save_and_load(tmp_path: Path) -> None:
    cat_file = tmp_path / "kb_catalog.json"
    entry1 = CatalogEntry(
        doc_id="policy-1",
        filename="policy-1.pdf",
        use_case="general",
        chunk_count=10,
        preview="Sample policy preview text.",
    )
    save_catalog([entry1], catalog_path=cat_file)

    loaded = load_catalog(cat_file)
    assert len(loaded) == 1
    assert loaded[0].doc_id == "policy-1"
    assert loaded[0].chunk_count == 10
    assert loaded[0].preview == "Sample policy preview text."


def test_record_catalog_entry_insert_and_update(tmp_path: Path) -> None:
    cat_file = tmp_path / "kb_catalog.json"

    # Insert entry
    record_catalog_entry(
        doc_id="doc-a",
        filename="doc-a.md",
        use_case="technical",
        chunk_count=5,
        preview="Initial architecture preview.",
        catalog_path=cat_file,
    )
    entries = load_catalog(cat_file)
    assert len(entries) == 1
    assert entries[0].chunk_count == 5

    # Update same entry
    record_catalog_entry(
        doc_id="doc-a",
        filename="doc-a.md",
        use_case="technical",
        chunk_count=8,
        preview="Updated architecture preview with more details.",
        catalog_path=cat_file,
    )
    entries = load_catalog(cat_file)
    assert len(entries) == 1
    assert entries[0].chunk_count == 8
    assert "Updated" in entries[0].preview


def test_build_system_memory_formatting(tmp_path: Path) -> None:
    cat_file = tmp_path / "kb_catalog.json"
    prof_file = tmp_path / "company_profile.md"

    prof_file.write_text("# Test Org\nRole: Test Bot", encoding="utf-8")
    record_catalog_entry(
        doc_id="doc-b",
        filename="doc-b.pdf",
        use_case="billing",
        chunk_count=12,
        preview="Billing terms and conditions.",
        catalog_path=cat_file,
    )

    memory = build_system_memory(catalog_path=cat_file, profile_path=prof_file)
    assert "# Test Org" in memory
    assert "Active Knowledge Base Catalog" in memory
    assert "doc-b.pdf" in memory
    assert "Billing terms and conditions." in memory
