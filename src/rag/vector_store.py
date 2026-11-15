"""Qdrant Cloud Vector Store integration with Blue-Green alias and use-case routing support.

See docs/EXECUTION_PLAN.md Levels 3 and 4.
"""
import logfire
import qdrant_client
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.config import RAGSettings
from rag.schemas import Chunk, RetrievalResult

ACTIVE_ALIAS = "rag_active_collection"


def collection_name_for_model(model_name: str) -> str:
    """Derive deterministic physical Qdrant collection name for an embedding model."""
    slug = model_name.replace("-", "_").replace(".", "_")
    return f"rag_chunks_{slug}"


class QdrantStore:
    """Wrapper around Qdrant Cloud client for indexing and similarity search."""

    def __init__(
        self,
        settings: RAGSettings | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = settings or RAGSettings()
        self.primary_collection = collection_name or collection_name_for_model(
            self.settings.primary_embedding_model
        )
        self.collection_name = self.primary_collection
        self.client = qdrant_client.QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
            timeout=60.0,
            check_compatibility=False,
        )
        self.ensure_collection_exists(self.primary_collection, self.settings.embedding_dimension)

    def ensure_collection_exists(
        self,
        collection_name: str | None = None,
        dimension: int | None = None,
    ) -> None:
        """Create collection with Cosine distance and KEYWORD indexes if it does not exist."""
        target_name = collection_name or self.primary_collection
        target_dim = dimension or self.settings.embedding_dimension
        if not self.client.collection_exists(collection_name=target_name):
            self.client.create_collection(
                collection_name=target_name,
                vectors_config=VectorParams(
                    size=target_dim,
                    distance=Distance.COSINE,
                ),
            )
        self.create_payload_indexes(target_name)

    def create_payload_indexes(self, collection_name: str) -> None:
        """Create KEYWORD payload indexes on use_case, doc_id, and content_hash."""
        for field in ("use_case", "doc_id", "content_hash"):
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def has_document_hash(
        self,
        doc_id: str,
        content_hash: str,
        collection_name: str | None = None,
    ) -> bool:
        """Check whether chunks for doc_id with the exact content_hash already exist in Qdrant."""
        target_name = collection_name or self.primary_collection
        try:
            if not self.client.collection_exists(collection_name=target_name):
                return False

            query_filter = Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
                ]
            )
            count_res = self.client.count(
                collection_name=target_name,
                count_filter=query_filter,
                exact=True,
            )
            return count_res.count > 0
        except Exception:
            return False

    def swap_active_alias(self, target_collection: str) -> None:
        """Atomically point rag_active_collection to target_collection."""
        with logfire.span("swap_active_alias", target_collection=target_collection):
            operation = CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=target_collection,
                    alias_name=ACTIVE_ALIAS,
                )
            )
            self.client.update_collection_aliases(change_aliases_operations=[operation])

    def upsert_chunks(
        self,
        chunk_vectors: list[tuple[Chunk, list[float]]],
        collection_name: str | None = None,
    ) -> None:
        """Upsert chunk vectors into a specified or primary physical Qdrant collection."""
        if not chunk_vectors:
            return

        target_name = collection_name or self.primary_collection
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in chunk_vectors
        ]

        with logfire.span("qdrant_upsert", collection=target_name, count=len(points)):
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch_points = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=target_name,
                    points=batch_points,
                    wait=True,
                )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        use_case: str | None = None,
        score_threshold: float | None = 0.65,
        target_collection: str | None = None,
    ) -> list[RetrievalResult]:
        """Search top_k most similar chunks with use_case filter and score_threshold."""
        with logfire.span(
            "qdrant_search",
            top_k=top_k,
            use_case=use_case,
            score_threshold=score_threshold,
        ):
            if target_collection:
                query_target = target_collection
            else:
                try:
                    if self.client.collection_exists(collection_name=ACTIVE_ALIAS):
                        query_target = ACTIVE_ALIAS
                    else:
                        query_target = self.primary_collection
                except Exception:
                    query_target = self.primary_collection

            query_filter = None
            if use_case:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="use_case",
                            match=MatchValue(value=use_case),
                        )
                    ]
                )

            try:
                response = self.client.query_points(
                    collection_name=query_target,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            except Exception:
                # Fallback to primary collection if alias query fails
                response = self.client.query_points(
                    collection_name=self.primary_collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )

            results: list[RetrievalResult] = []
            for scored_point in response.points:
                payload = scored_point.payload or {}
                chunk = Chunk.model_validate(payload)
                results.append(RetrievalResult(chunk=chunk, score=float(scored_point.score)))

            logfire.info(
                "qdrant_search_results",
                retrieved_count=len(results),
                use_case=use_case,
                scores=[r.score for r in results],
            )
            return results
