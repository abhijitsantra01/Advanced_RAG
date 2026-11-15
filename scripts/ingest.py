"""Dedicated CLI tool for document ingestion with SHA-256 hash deduplication.

Usage:
    python scripts/ingest.py --dir data/raw --use-case general
    python scripts/ingest.py --force
"""
import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct CLI execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

from rag.config import RAGSettings
from rag.observability import setup_observability
from rag.pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant Cloud with SHA-256 hash deduplication."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/raw",
        help="Directory path containing .txt and .md files (default: data/raw).",
    )
    parser.add_argument(
        "--use-case",
        type=str,
        default="general",
        help="Use-case domain namespace (default: general).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-embedding even if document content hash already exists in Qdrant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    settings = RAGSettings()
    setup_observability(settings)

    pipeline = RAGPipeline(settings=settings)
    target_dir = Path(args.dir)

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Target directory does not exist: {target_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        [
            p
            for p in target_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
        ]
    )

    print(f"Scanning '{target_dir}' (found {len(files)} eligible files)...")
    print(f"Use-case namespace: '{args.use_case}' | Force re-embed: {args.force}")

    newly_embedded = pipeline.ingest_directory(
        directory_path=target_dir,
        use_case=args.use_case,
        force=args.force,
    )

    collection_name = pipeline.vector_store.collection_name
    print("\nIngestion Complete:")
    print(f"  - Files scanned: {len(files)}")
    print(f"  - Chunks newly embedded: {newly_embedded}")
    print(f"  - Target Qdrant collection: {collection_name}")
    if newly_embedded == 0 and len(files) > 0:
        print("  - All documents are up-to-date (skipped via SHA-256 hash matching).")


if __name__ == "__main__":
    main()
