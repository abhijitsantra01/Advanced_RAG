# Enterprise Agentic RAG

A production-grade, zero-bloat enterprise RAG and Agentic RAG system built strictly on official SDKs with progressive failover resilience, strict multi-tenant use-case isolation, SHA-256 content-hash deduplication, and full observability.

---

## Key Capabilities

- **Zero-Bloat Architecture**: Built without heavy wrapper frameworks (no LangChain, no LlamaIndex). Uses direct, official SDKs: `google-genai`, `qdrant-client`, `logfire`, `pydantic`, and `pypdf`.
- **Multi-Format Ingestion with SHA-256 Caching**: Native support for `.pdf`, `.md`, and `.txt` files. Computes raw-byte SHA-256 digests to bypass embedding and vector upserts for unchanged documents (instant re-runs with zero API calls).
- **Rate-Pacing & Batch Resiliency**: Embedded in batches with deliberate rate-pacing (<100 RPM) to stay within Google AI Studio Free Tier limits; chunk uploads to Qdrant Cloud are batched (50 points/batch) with custom connection timeouts.
- **Failover & Blue-Green Aliasing**:
  - **Level 2 (Generator)**: Primary `gemini-3.5-flash-lite` automatically falls back to `gemini-3.1-flash-lite` on HTTP 429/503/timeout.
  - **Level 3 (Embedder & Blue-Green Collection Swap)**: Primary `gemini-embedding-2` (3,072 dimensions) falls back to `gemini-embedding-001` (768 dimensions) by auto-reingesting into a shadow collection and atomically swapping the `rag_active_collection` alias with zero downtime.
- **Use-Case Domain Routing**: Enforces strict use-case domain filtering (`general`, `billing`, `technical`, etc.) with `score_threshold=0.65` and `KEYWORD` payload indices to ensure zero cross-domain leakage.
- **Agentic RAG Mode**: Autonomous agent powered by Gemini Function Calling (`search_use_case_kb`, `list_available_use_cases`, `get_adjacent_chunks`) capable of multi-step retrieval, query reformulation, and reading neighboring document context.
- **End-to-End Tracing**: Instrumented throughout with [Pydantic Logfire](https://pydantic.dev/docs/logfire/) spanning chunking, PDF extraction, embeddings, Qdrant searches, and generation attempts.

---

## Repository Structure

```text
enterprise-rag/
├── .antigravityrules          # Architectural guardrails & banned package list
├── .env.example               # Environment variables template
├── pyproject.toml             # Pinned project dependencies & pytest settings
├── requirements.txt           # Locked pip dependencies
├── data/
│   └── raw/                   # Raw documents (.pdf, .md, .txt)
├── docs/
│   ├── ANTIGRAVITY_RULES.md   # Strict rules on dependencies and coding standards
│   ├── ARCHITECTURE.md        # Technical architecture & blue-green failover specs
│   ├── EXECUTION_PLAN.md      # Five-level evolutionary roadmap
│   └── LEVEL_1_SPEC.md        # Baseline specification
├── scripts/
│   ├── chat.py                # Interactive REPL supporting both RAG and Agent modes
│   ├── ingest.py              # CLI ingestion tool with SHA-256 hash deduplication
│   └── smoke_test_live.py     # Live end-to-end sanity check
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── agent.py           # Agentic loop with tool binding & fallback
│       ├── chunker.py         # Deterministic fixed-size chunker with UUID5 IDs
│       ├── config.py          # Pydantic BaseSettings configuration
│       ├── embedder.py        # Embedder client with rate pacing & failover
│       ├── generator.py       # Generator client with fallback & strict citations
│       ├── loader.py          # Multi-format reader (.txt, .md, .pdf) & SHA-256 hasher
│       ├── observability.py   # Logfire integration & span instrumentation
│       ├── pipeline.py        # End-to-end RAG pipeline & re-ingestion
│       ├── schemas.py         # Pydantic schemas (Document, Chunk, RetrievalResult, etc.)
│       └── vector_store.py    # Qdrant client, alias management & search
└── tests/                     # 34-test comprehensive unit test suite
    ├── test_agent.py
    ├── test_chunker.py
    ├── test_embedding_failover.py
    ├── test_generator_fallback.py
    ├── test_hash_caching.py
    ├── test_loader.py
    ├── test_pipeline.py
    └── test_use_case_routing.py
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.11+ (tested on Python 3.13)
- [Google AI Studio API Key](https://aistudio.google.com/)
- [Qdrant Cloud Cluster](https://cloud.qdrant.io/) (or local Qdrant instance)
- [Pydantic Logfire Token](https://logfire.pydantic.dev/) (optional, but recommended for observability)

### 2. Environment Setup

```bash
# Clone and enter workspace
git clone <repo-url>
cd enterprise-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .\.venv\Scripts\activate

# Install locked dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### 3. Configure `.env`
Update your `.env` file with your credentials:
```env
GEMINI_API_KEY="your-gemini-api-key"
QDRANT_URL="https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY="your-qdrant-api-key"
LOGFIRE_TOKEN="your-logfire-read-write-token"
```

---

## Usage

### 1. Ingest Documents
Place your `.pdf`, `.md`, or `.txt` files into `data/raw/` (or any custom folder), then run the ingestion CLI:

```bash
# Standard ingestion into the 'general' use-case
python scripts/ingest.py --dir data/raw --use-case general

# Ingest into a domain-isolated use-case (e.g. 'technical' or 'billing')
python scripts/ingest.py --dir data/raw --use-case technical

# Force re-embedding (bypass SHA-256 content-hash cache)
python scripts/ingest.py --dir data/raw --force
```

*Note: Unchanged documents are automatically skipped via SHA-256 hash checks with zero embedding API calls.*

### 2. Interactive Terminal Chat (REPL)
Launch the interactive terminal shell to query your documents:

```bash
# Start in standard RAG mode
python scripts/chat.py --mode rag --use-case general

# Start in Agentic RAG mode (autonomous multi-step search & context expansion)
python scripts/chat.py --mode agent --use-case general
```

**Commands inside the REPL**:
- `/mode <rag|agent>`: Switch between deterministic RAG and autonomous Agentic mode dynamically.
- `/usecase <name>`: Switch the active use-case namespace (e.g., `/usecase technical`).
- `/exit`: Exit the chat session.

### 3. Run FastAPI REST Service
Launch the production REST API server:

```bash
uvicorn rag.api:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI docs will be available at: `http://localhost:8000/docs`

**Core Endpoints**:
- `POST /v1/chat`: Conversational multi-turn Agentic RAG with `session_id` isolation.
- `POST /v1/query`: Single-turn deterministic RAG query.
- `GET /v1/catalog`: Retrieve active document catalog and company scope.
- `POST /v1/ingest`: Trigger background folder re-sync with SHA-256 deduplication.
- `GET /healthz`: Production readiness probe checking Qdrant Cloud connectivity.

### 4. Run Live Sanity Check
Verify your live connection to Gemini, Qdrant Cloud, and Logfire:

```bash
python scripts/smoke_test_live.py
```

---

## Testing & Code Quality

The project features comprehensive unit test coverage (45 tests) validating chunking overlap, Level 2 generator fallback, Level 3 embedding failover with blue-green alias swapping, PDF page-number preservation, use-case isolation, SHA-256 deduplication, agent function calling, catalog system memory, and FastAPI REST endpoints.

```bash
# Run the complete test suite
pytest -q

# Run Ruff linter
ruff check src/ tests/ scripts/
```

---

## Architectural Principles

1. **Strictly Official SDKs**: No wrapper frameworks (`langchain`, `llama-index`, etc.). Only direct client libraries are permitted.
2. **Deterministic Citations**: All generated answers cite sources using `[doc_id:chunk_index]`, verified before returning to the caller.
3. **Multi-Tenant Domain Isolation**: Qdrant queries strictly filter on `use_case` with a cosine similarity threshold of `0.65`, preventing unrelated context retrieval.
4. **Resilient Rate Handling**: Transparent pacing and retry-after backoff prevent API exhaustion, while dual-model fallbacks ensure zero user-facing downtime.
