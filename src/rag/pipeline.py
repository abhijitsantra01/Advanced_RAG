"""Pipeline orchestrator for Enterprise RAG with Blue-Green failover support.

See docs/EXECUTION_PLAN.md Level 3.
"""
from pathlib import Path

import logfire

from rag.catalog import load_catalog, record_catalog_entry
from rag.chunker import FixedChunker
from rag.config import RAGSettings
from rag.embedder import EmbedderClient, EmbeddingFailoverRequired
from rag.generator import GeneratorClient
from rag.loader import compute_file_hash, load_document_text
from rag.schemas import Document, RAGResponse
from rag.vector_store import QdrantStore, collection_name_for_model


class RAGPipeline:
    """End-to-end RAG orchestrator for document ingestion and grounded question answering."""

    def __init__(self, settings: RAGSettings | None = None) -> None:
        self.settings = settings or RAGSettings()
        self.chunker = FixedChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedder = EmbedderClient(settings=self.settings)
        self.vector_store = QdrantStore(settings=self.settings)
        self.generator = GeneratorClient(settings=self.settings)

    def ingest_directory(
        self,
        directory_path: Path,
        use_case: str = "general",
        force: bool = False,
    ) -> int:
        """Read all .txt and .md files in directory_path, chunk, embed, and upsert into Qdrant."""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        file_paths = sorted(
            [
                p
                for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
            ]
        )

        all_chunks = []
        for file_path in file_paths:
            doc_id = file_path.stem
            content_hash = compute_file_hash(file_path)

            if not force and self.vector_store.has_document_hash(doc_id, content_hash):
                logfire.info(
                    "Skipping unchanged document",
                    doc_id=doc_id,
                    content_hash=content_hash,
                )
                content = load_document_text(file_path)
                existing = {e.doc_id: e for e in load_catalog()}
                chunk_count = (
                    existing[doc_id].chunk_count
                    if doc_id in existing
                    else len(
                        self.chunker.chunk_document(
                            doc=Document(
                                doc_id=doc_id,
                                title=file_path.name,
                                content=content,
                                source_path=str(file_path),
                                use_case=use_case,
                                content_hash=content_hash,
                            ),
                            embedding_model=self.embedder.active_model,
                        )
                    )
                )
                record_catalog_entry(
                    doc_id=doc_id,
                    filename=file_path.name,
                    use_case=use_case,
                    chunk_count=chunk_count,
                    preview=content[:300],
                )
                continue

            content = load_document_text(file_path)
            doc = Document(
                doc_id=doc_id,
                title=file_path.name,
                content=content,
                source_path=str(file_path),
                use_case=use_case,
                content_hash=content_hash,
            )
            chunks = self.chunker.chunk_document(
                doc=doc,
                embedding_model=self.embedder.active_model,
            )
            for chunk in chunks:
                chunk.content_hash = content_hash
            all_chunks.extend(chunks)

            record_catalog_entry(
                doc_id=doc_id,
                filename=file_path.name,
                use_case=use_case,
                chunk_count=len(chunks),
                preview=content[:300],
            )

        if not all_chunks:
            return 0

        chunk_texts = [c.text for c in all_chunks]
        try:
            embeddings = self.embedder.embed_texts(chunk_texts, task_type="RETRIEVAL_DOCUMENT")
        except EmbeddingFailoverRequired as exc:
            logfire.warn(
                "Primary embedding quota exhausted; failing over to shadow collection",
                fallback_model=exc.fallback_model,
            )
            return self.reingest_all(
                source_dir=directory,
                model_name=exc.fallback_model,
                force=force,
            )

        chunk_vectors = list(zip(all_chunks, embeddings, strict=True))
        self.vector_store.upsert_chunks(chunk_vectors)

        return len(all_chunks)

    def reingest_all(self, source_dir: Path, model_name: str, force: bool = False) -> int:
        """Re-ingest all documents in source_dir into a model-specific collection and swap alias."""
        directory = Path(source_dir)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {source_dir}")

        target_collection = collection_name_for_model(model_name)
        dimension = self.embedder.dimension

        with logfire.span(
            "reingest_all",
            source_dir=str(directory),
            model_name=model_name,
            target_collection=target_collection,
        ):
            self.vector_store.ensure_collection_exists(
                collection_name=target_collection,
                dimension=dimension,
            )

            file_paths = sorted(
                [
                    p
                    for p in directory.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
                ]
            )

            all_chunks = []
            for file_path in file_paths:
                doc_id = file_path.stem
                content_hash = compute_file_hash(file_path)

                if not force and self.vector_store.has_document_hash(
                    doc_id, content_hash, collection_name=target_collection
                ):
                    logfire.info(
                        "Skipping unchanged document in fallback collection",
                        doc_id=doc_id,
                        collection=target_collection,
                    )
                    continue

                content = load_document_text(file_path)
                doc = Document(
                    doc_id=doc_id,
                    title=file_path.name,
                    content=content,
                    source_path=str(file_path),
                    use_case="general",
                    content_hash=content_hash,
                )
                chunks = self.chunker.chunk_document(
                    doc=doc,
                    embedding_model=model_name,
                )
                for chunk in chunks:
                    chunk.content_hash = content_hash
                all_chunks.extend(chunks)

                record_catalog_entry(
                    doc_id=doc_id,
                    filename=file_path.name,
                    use_case="general",
                    chunk_count=len(chunks),
                    preview=content[:300],
                )

            if all_chunks:
                chunk_texts = [c.text for c in all_chunks]
                embeddings = self.embedder.embed_texts(chunk_texts, task_type="RETRIEVAL_DOCUMENT")
                chunk_vectors = list(zip(all_chunks, embeddings, strict=True))
                self.vector_store.upsert_chunks(
                    chunk_vectors,
                    collection_name=target_collection,
                )

            # Atomically swap active alias ONLY after all chunks are upserted
            self.vector_store.swap_active_alias(target_collection=target_collection)

            return len(all_chunks)

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_case: str | None = "general",
        score_threshold: float = 0.40,
    ) -> RAGResponse:
        """Answer a question using embedded retrieval from Qdrant and Gemini generation."""
        with logfire.span(
            "rag_query_pipeline",
            question=question,
            top_k=top_k,
            use_case=use_case,
            score_threshold=score_threshold,
        ):
            try:
                query_embeddings = self.embedder.embed_texts(
                    [question], task_type="RETRIEVAL_QUERY"
                )
            except EmbeddingFailoverRequired as exc:
                self.embedder.active_model = exc.fallback_model
                raw_dir = Path("data/raw")
                self.reingest_all(source_dir=raw_dir, model_name=self.embedder.active_model)
                query_embeddings = self.embedder.embed_texts(
                    [question], task_type="RETRIEVAL_QUERY"
                )

            query_vector = query_embeddings[0]

            retrieved_chunks = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                use_case=use_case,
                score_threshold=score_threshold,
            )
            answer, model_used, fallback_triggered = self.generator.generate(
                query=question, retrieved_chunks=retrieved_chunks
            )

            return RAGResponse(
                query=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                embedding_model_used=self.embedder.active_model,
                generation_model_used=model_used,
                fallback_triggered=fallback_triggered,
            )
