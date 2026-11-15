"""Production FastAPI REST service for Enterprise Agentic RAG.

Endpoints:
- POST /v1/chat    : Multi-turn conversational Agentic RAG with session isolation
- POST /v1/query   : Deterministic single-turn RAG retrieval and answer generation
- GET  /v1/catalog : Active knowledge base catalog and company profile scope
- POST /v1/ingest  : Trigger background folder re-sync with SHA-256 deduplication
- GET  /healthz    : Health check verifying Qdrant Cloud and Gemini configuration
"""
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.agent import AgenticRAG
from rag.catalog import load_catalog, load_company_profile
from rag.config import RAGSettings
from rag.observability import setup_observability
from rag.pipeline import RAGPipeline

MAX_SESSIONS = 500

_session_lock = threading.Lock()
_sessions: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()

_ingest_lock = threading.Lock()

_pipeline_instance: RAGPipeline | None = None
_agent_instance: AgenticRAG | None = None
_settings_instance: RAGSettings | None = None


def get_settings() -> RAGSettings:
    """Get or create singleton RAGSettings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = RAGSettings()
    return _settings_instance


def get_pipeline() -> RAGPipeline:
    """Get or create singleton RAGPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(settings=get_settings())
    return _pipeline_instance


def get_agent() -> AgenticRAG:
    """Get or create singleton AgenticRAG instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AgenticRAG(settings=get_settings())
    return _agent_instance


def get_session_history(session_id: str) -> list[tuple[str, str]]:
    """Retrieve chat history for a session_id, moving it to end (LRU)."""
    with _session_lock:
        if session_id in _sessions:
            _sessions.move_to_end(session_id)
            return list(_sessions[session_id])
        return []


def save_session_history(session_id: str, history: list[tuple[str, str]]) -> None:
    """Save chat history for a session_id, enforcing MAX_SESSIONS capacity."""
    with _session_lock:
        if session_id in _sessions:
            _sessions.move_to_end(session_id)
        elif len(_sessions) >= MAX_SESSIONS:
            _sessions.popitem(last=False)
        _sessions[session_id] = list(history[-5:])


def execute_ingest_background(dir_path: Path, use_case: str, force: bool) -> None:
    """Execute ingestion under ingest lock to prevent concurrent races."""
    with _ingest_lock:
        with logfire.span("api_background_ingest", dir=str(dir_path), use_case=use_case):
            pipeline = get_pipeline()
            pipeline.ingest_directory(directory_path=dir_path, use_case=use_case, force=force)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook initializing settings and observability."""
    load_dotenv()
    settings = get_settings()
    setup_observability(settings)
    yield


app = FastAPI(
    title="Enterprise Agentic RAG API",
    version="1.0.0",
    description="Production-grade Enterprise Agentic RAG Service with failover and deduplication.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Logfire FastAPI instrumentation
try:
    logfire.instrument_fastapi(app)
except Exception:
    pass


@app.middleware("http")
async def http_request_span_middleware(request: Request, call_next) -> Response:
    """Logfire request span wrapper."""
    with logfire.span(
        "http_request",
        method=request.method,
        path=request.url.path,
    ):
        return await call_next(request)


# --- Request & Response Models ---


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or statement.")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn history continuity.",
    )
    use_case: str = Field(default="general", description="Target domain namespace.")


class ChatResponse(BaseModel):
    query: str
    answer: str
    session_id: str
    model_used: str
    fallback_triggered: bool
    tools_called: list[str]
    steps_taken: int


class QueryRequestModel(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer.")
    use_case: str = Field(default="general", description="Use-case domain namespace.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    score_threshold: float = Field(default=0.40, ge=0.0, le=1.0)


class IngestRequest(BaseModel):
    dir: str = Field(default="data/raw", description="Directory path to scan and ingest.")
    use_case: str = Field(default="general", description="Use-case domain namespace.")
    force: bool = Field(default=False, description="Bypass SHA-256 content-hash cache.")


# --- Endpoints ---


@app.post("/v1/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """Multi-turn conversational Agentic RAG with session isolation and router mode."""
    session_id = req.session_id or uuid.uuid4().hex
    history = get_session_history(session_id)

    agent = get_agent()
    result = agent.run(prompt=req.query, chat_history=history)

    save_session_history(session_id, agent.chat_history)

    return ChatResponse(
        query=result.query,
        answer=result.answer,
        session_id=session_id,
        model_used=result.model_used,
        fallback_triggered=result.fallback_triggered,
        tools_called=result.tools_called,
        steps_taken=result.steps_taken,
    )


@app.post("/v1/query", status_code=status.HTTP_200_OK)
def query_endpoint(req: QueryRequestModel) -> dict:
    """Deterministic single-turn RAG retrieval and answer generation."""
    pipeline = get_pipeline()
    response = pipeline.query(
        question=req.query,
        top_k=req.top_k,
        use_case=req.use_case,
        score_threshold=req.score_threshold,
    )
    return response.model_dump()


@app.get("/v1/catalog", status_code=status.HTTP_200_OK)
def catalog_endpoint() -> dict:
    """Retrieve active knowledge base catalog and company profile scope."""
    entries = load_catalog()
    profile = load_company_profile()
    return {
        "catalog": [e.model_dump() for e in entries],
        "company_profile": profile,
    }


@app.post("/v1/ingest", status_code=status.HTTP_202_ACCEPTED)
def ingest_endpoint(req: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    """Trigger background document ingestion with SHA-256 deduplication."""
    target_dir = Path(req.dir)
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target directory does not exist: {req.dir}",
        )

    if _ingest_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion process is already running.",
        )

    background_tasks.add_task(
        execute_ingest_background,
        dir_path=target_dir,
        use_case=req.use_case,
        force=req.force,
    )

    return {
        "status": "accepted",
        "message": "Ingestion job enqueued in background.",
        "directory": req.dir,
        "use_case": req.use_case,
        "force": req.force,
    }


@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_endpoint() -> dict:
    """Health readiness probe verifying Qdrant Cloud connection and configuration."""
    settings = get_settings()
    pipeline = get_pipeline()

    qdrant_healthy = False
    error_msg = None
    try:
        qdrant_healthy = pipeline.vector_store.client.collection_exists(
            pipeline.vector_store.primary_collection
        )
    except Exception as exc:
        error_msg = str(exc)

    overall_status = "healthy" if qdrant_healthy else "degraded"
    return {
        "status": overall_status,
        "qdrant": {
            "connected": qdrant_healthy,
            "collection": pipeline.vector_store.primary_collection,
            "error": error_msg,
        },
        "models": {
            "primary_llm": settings.primary_llm_model,
            "fallback_llm": settings.fallback_llm_model,
            "primary_embedding": settings.primary_embedding_model,
        },
    }
