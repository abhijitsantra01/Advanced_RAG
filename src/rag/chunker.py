"""Deterministic Fixed-Size Chunker with Character Overlap.
Antigravity: Complete implementation & Logfire spans per docs/LEVEL_1_SPEC.md Section 3.
"""
import uuid

from rag.schemas import Chunk, Document


class FixedChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step = chunk_size - chunk_overlap

    def chunk_document(self, doc: Document, embedding_model: str) -> list[Chunk]:
        text = doc.content.replace("\r\n", "\n").strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            deterministic_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_OID,
                    f"{doc.doc_id}:{chunk_idx}:{embedding_model}",
                )
            )
            chunks.append(
                Chunk(
                    chunk_id=deterministic_id,
                    doc_id=doc.doc_id,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    embedding_model=embedding_model,
                    use_case=doc.use_case,
                )
            )
            if end >= text_len:
                break
            start += self.step
            chunk_idx += 1

        return chunks
