"""Minimal live smoke test script (< 40 lines) for Level 1 RAG verification."""
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

from rag.config import RAGSettings
from rag.observability import setup_observability
from rag.pipeline import RAGPipeline


def main() -> None:
    load_dotenv()
    settings = RAGSettings()
    setup_observability(settings)

    pipeline = RAGPipeline(settings=settings)
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    count = pipeline.ingest_directory(raw_dir)
    print(f"Ingested {count} chunks into Qdrant ({pipeline.vector_store.collection_name}).")

    question = "What embedding and generation models does this architecture use?"
    response = pipeline.query(question=question, top_k=3)

    print("\n--- Question ---")
    print(response.query)
    print("\n--- Answer ---")
    print(response.answer)
    print(f"\nModel: {response.generation_model_used} | Embedding: {response.embedding_model_used}")


if __name__ == "__main__":
    main()
