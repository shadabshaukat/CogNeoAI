Architecture and logic summary of CogNeo AI 

High-level
- CogNeo is an agentic RAG platform with three runtime frontends:
  - FastAPI service (port 8000) exposing REST for search, RAG, chat (Ollama/OCI/Bedrock), ingestion orchestration metadata, models listing, Oracle DB proxy, health, optional metrics.
  - Streamlit app (port 8501) acting as a usability console: login, ingestion session management, hybrid search + RAG, and a chat UI. It also can launch background embedding workers on available GPUs.
  - Gradio UI (port 7866) providing tabs for hybrid/vector search, RAG, FTS, conversational chat, and agentic chat with provider selection (Ollama/OCI/Bedrock).
- Storage backends:
  - System of record DB: PostgreSQL (+pgvector) by default, optional Oracle 26ai.
  - Optional retrieval backend: OpenSearch KNN for serving vector, BM25/hybrid, and FTS while still keeping DB as the system of record (single-writer or dual-write).
- Embedding: Sentence-Transformers primary, HuggingFace AutoModel mean-pooling fallback. Dimensionality configurable; must match DB/index mapping.
- Ingestion paths:
  - Streamlit-driven worker (embedding_worker.py).
  - High-throughput multi-GPU “Beta” pipeline under ingest/ (orchestrator + worker + chunker), with dynamic sharding, resumability, metrics.
- RAG:
  - Ollama via HTTP generate/stream endpoints.
  - OCI GenAI via official SDK chat API.
  - AWS Bedrock via boto3 (model discovery and generation).
- Security/ops: Basic or JWT auth for API, CORS, optional simple rate limiting, optional Prometheus middleware.

Runtime entrypoints
- run_cogneo_stack.sh: starts FastAPI (uvicorn fastapi_app:app :8000), Gradio (python gradio_app.py :7866), Streamlit (app.py :8501), prints banner + URLs. stop_cogneo_stack.sh terminates using stored PIDs.
- FastAPI application: fastapi_app.py
- Streamlit application: app.py (+ streamlit pages/ for login/chat)
- Gradio application: gradio_app.py

Configuration and dependencies
- requirements.txt includes FastAPI/Uvicorn, Streamlit, Gradio, SQLAlchemy, psycopg2-binary, pgvector, sentence-transformers, transformers, torch, oci, oracledb, boto3, opensearch-py, PyJWT, starlette-exporter, html2text, bs4, etc.
- .env variables drive DB connectivity, auth mode, OpenSearch, Bedrock, OCI, and feature gating (AUTO_DDL, Prometheus, CORS, rate limits).
- DB connectivity:
  - PostgreSQL: either COGNEO_DB_URL or individual COGNEO_DB_HOST/PORT/USER/PASSWORD/NAME (connector_postgres enforces presence).
  - Oracle 26ai: via ORACLE_* or ORACLE_SQLALCHEMY_URL (when backend “oracle” is selected).
- Backend selectors:
  - db.store (dispatcher): COGNEO_DB_BACKEND=postgres|oracle selects which store_* module implements CRUD/search/DDL.
  - storage.registry (retrieval adapter): COGNEO_VECTOR_BACKEND=opensearch|postgres|oracle controls WHERE search calls go from the API/UI perspective. Defaults to Postgres. This is orthogonal; you can keep PostgreSQL/Oracle as system-of-record but serve KNN/BM25/FTS from OpenSearch.

Core modules and flow
- API server (fastapi_app.py):
  - Auth: Basic (default via FASTAPI_API_USER/PASS) or JWT (COGNEO_AUTH_MODE=jwt).
  - CORS middleware, optional Prometheus metrics on /metrics, optional IP-based rate limiting.
  - On startup optionally ensures schema via get_adapter().create_all_tables() (AUTO_DDL=1).
  - Endpoints:
    - Health: GET /health
    - Auth: POST /auth/token (JWT issuance)
    - Ingestion metadata: POST /ingest/start, GET /ingest/sessions, POST /ingest/stop (tracks sessions in DB; actual embedding is launched by Streamlit or ingest tools)
    - Documents: GET /documents, GET /documents/{id}
    - Search:
      - POST /search/vector (uses adapter.search_vector with Embedder for query embedding)
      - POST /search/hybrid (alpha fusion of vector presence/score + BM25 presence; returns metadata-rich hits)
      - POST /search/fts (FTS over documents and/or metadata)
      - POST /search/rerank (trivial top-k re-rank stub with model registry; download of HF XC models supported)
    - RAG:
      - POST /search/rag (Ollama)
      - POST /search/rag_stream (Ollama streaming plain text)
      - POST /search/oci_rag (OCI GenAI)
      - POST /search/bedrock_rag (AWS Bedrock)
    - Chat:
      - POST /chat/conversation (RAG per turn + sources; for Ollama/OCI/Bedrock)
      - POST /chat/agentic (Agentic/CoT formatting; performs adapter.search_hybrid to supply context first)
    - Model listings:
      - GET /models/ollama (via rag_pipeline.list_ollama_models)
      - GET /models/oci_genai (queries OCI Inference models; returns PRETRAINED/chat-capable)
      - GET /models/bedrock (?regions=..., include_current=bool)
    - Files: GET /files/ls for limited whitelisted roots
    - Oracle DB proxy: POST /db/oracle23ai_query (db/oracle23ai_connector)
- RAG backends:
  - rag/rag_pipeline.py: Ollama generate and streaming; formats context blocks with metadata; query(...) returns answer, contexts, sources, metadata. retrieve(...) is a placeholder; in practice contexts are supplied by API/UI using storage adapters.
  - rag/oci_rag_pipeline.py: Uses OCI Generative AI chat API; builds ChatDetails -> chat(...); returns answer plus echo of sources/contexts/metadata.
  - rag/bedrock_pipeline.py: Not shown here, but API wired similarly to instantiate and query Bedrock.
- UIs:
  - Streamlit (app.py):
    - Login gate (pages/login.py). After login, sidebars for dataset load session and HTML/PDF-to-text conversion (html2text_utils).
    - Hybrid search + RAG using FastAPI endpoints; citations and metadata cards shown.
    - RAG streaming chat (Ollama) plus OCI/Bedrock synchronous paths.
    - Ingestion session management (Start Ingestion). For single/multi-GPU, launches embedding_worker.py with optional per-GPU partition files. Tracks progress by polling DB session rows.
  - Gradio (gradio_app.py):
    - Login.
    - Tabs:
      - Hybrid Search (calls /search/hybrid and renders cards)
      - RAG (provider/model selection, calls RAG endpoints)
      - Full Text Search (calls /search/fts)
      - Conversational Chat (calls /chat/conversation)
      - Agentic RAG (calls /chat/agentic)
    - Ensures DB DDL on startup if AUTO_DDL=1.
    - Includes cached AWS Bedrock models listing by provider with TTL.

Storage and retrieval adapters
- storage/interfaces.py defines VectorSearchAdapter:
  - create_all_tables(), search_vector, search_bm25, search_hybrid, search_fts, index_chunks (optional bulk for serving stores).
- storage/registry.py chooses adapter by COGNEO_VECTOR_BACKEND or COGNEO_DB_BACKEND:
  - "opensearch" -> OpenSearchAdapter
  - "oracle" -> Oracle26aiAdapter (not reviewed here)
  - else -> PgVectorAdapter
- PgVectorAdapter wraps db.store_postgres.* for search + DDL.
  - db/store_postgres.py:
    - SQLAlchemy models: User, Document, Embedding, EmbeddingSession, EmbeddingSessionFile, ChatSession, ConversionFile; plus normalized legal metadata tables (Case/Legislation/Journal/Treaty families) for future or external loaders.
    - create_all_tables(): enables extensions (vector, pg_trgm, uuid-ossp, fuzzystrmatch), creates tables, sets up FTS column and triggers for documents, builds IVFFLAT index (unless LIGHT_INIT), adds essential indexes.
    - search_vector: uses vector <=> query syntax (distance ascending). Returns doc_id/chunk_index/score and metadata.
    - search_bm25: simplified ILIKE fallback; returns presence score 1.0.
    - search_hybrid: embeds query via Embedder, merges KNN results (vector_score) and BM25 presence, normalizes vector distance to a “higher is better” score: vector_score_norm = 1 - normalized_distance, then hybrid_score = alpha*vector_score_norm + (1-alpha)*bm25_presence. Adds citation for parity.
    - search_fts: FTS over documents.document_fts and/or to_tsvector(e.chunk_metadata::text), highlights snippets, dedups by url or doc_id.
    - CRUD and session helpers for ingestion progress tracking and chat session archiving.
  - db/store.py: dispatcher; COGNEO_DB_BACKEND selects store_postgres or store_oracle (Oracle not included in this scan).
- OpenSearchAdapter (storage/adapters/opensearch.py):
  - Client built from OPENSEARCH_* env. Reads OPENSEARCH_INDEX (default cogneo_chunks) and COGNEO_EMBED_DIM.
  - create_all_tables(): ensures index with knn_vector mapping, doc_id/chunk_index/citation/text/source/format/chunk_metadata (+ fallback chunk_metadata_text when flattened unsupported). Honors OPENSEARCH_NUMBER_OF_SHARDS/REPLICAS on first creation; logs diagnostics. Supports lucene/faiss/nmslib engines, HNSW params.
  - index_chunks(chunks, vectors,...): bulk or parallel_bulk with shard-aware effective concurrency. Computes stable _id = doc_id#chunk_index when present, or source#i. Adds citation = source#chunkN.
  - search_vector: modern/legacy KNN queries; returns vector_score (score), bm25_score 0.0, and metadata.
  - search_bm25: match on text; returns bm25_score=1.0 presence.
  - search_fts: query_string over chunk_metadata (flattened or text) and/or match on text; highlights snippet; falls back to text-only if no hits for metadata.
  - search_hybrid: embeds query internally (mentions Embedder), retrieves top-K from vector/BM25, normalizes vector scores to [0,1], hybrid_score = alpha*vector_norm + (1-alpha)*bm25_presence; returns metadata and citation.

Embedding and ingestion
- Embedding (embedding/embedder.py):
  - Resolution order: explicit arg -> COGNEO_EMBED_MODEL -> default nomic-ai/nomic-embed-text-v1.5.
  - Tries SentenceTransformers with optional trust_remote_code and revision/local-only flags; on failure uses HF AutoTokenizer/AutoModel and mean pooling with attention mask. Sets dimension from ST or HF config.hidden_size. Returns float32 numpy arrays.
  - Optional normalization commented (left off for score stability consistent with stores).
- Loader and chunker (ingest/loader.py):
  - walk_legal_files: recursive .txt/.html discovery.
  - parse_txt/parse_html: parse files, strip unwanted tags for HTML; supports a dashed metadata header block at document head and returns chunk_metadata and format.
  - Chunking:
    - All chunk sizes capped at 1500 characters; sentence-aware splitting on overflow.
    - Specialized chunkers: chunk_legislation (detect dashed section headers), chunk_journal (heading patterns), chunk_case (paragraph-based), chunk_generic fallback.
- Streamlit embedding worker (embedding_worker.py):
  - For each file: parse + chunk, embed as a batch, write one Document per chunk plus an Embedding row linked to that doc with chunk_index and chunk_metadata. Updates EmbeddingSessionFile and EmbeddingSession progress, supports resume by skipping completed.
  - For multi-GPU launches: streamlit creates per-GPU partition files and spawns workers with CUDA_VISIBLE_DEVICES set.
- Beta ingestion pipeline (ingest/):
  - Ingest/README.md documents a multi-GPU orchestrator with dynamic sharding, greedy-by-size balancing, pipelined CPU/GPU, timeouts and resumability, per-file metrics, and post-load SQL/index hardening.
  - End-to-end runbooks in docs/BetaDataLoad.md. Tools include logs/partition validation and resume flows.
  - Optional dual-write to OpenSearch during ingest is env-gated.

RAG and Chat flows
- Typical UI call path:
  - UI calls FastAPI /search/hybrid to get top_k chunks with text, citation, source, format, and chunk_metadata (via storage adapter).
  - For RAG:
    - /search/rag (Ollama): rag_pipeline builds prompt from context chunks + metadata blocks, calls Ollama generate, returns answer. /search/rag_stream streams plain text.
    - /search/oci_rag: oci_rag_pipeline composes GenericChatRequest and calls OCI chat API, returns answer.
    - /search/bedrock_rag: bedrock pipeline composes provider-specific payload via boto3 abstractions.
  - Conversational chat: /chat/conversation: per-turn hybrid search; passes context to RAG pipeline for selected provider; returns answer + context/sources for UI. Agentic chat adds enforced stepwise chain-of-thought formatting in the system prompt and similar retrieval.

Database model (Postgres store)
- Documents(id, source, content, format), Embeddings(id, doc_id, chunk_index, vector, chunk_metadata JSONB).
- Embedding sessions: embedding_sessions (progress counters, status), embedding_session_files (per-file status, unique(session_name, filepath)).
- Users, conversion_files, chat_sessions.
- Normalized legal entities tables present for future use (cases/legislation/journals/treaties + related tables).
- DDL: vector extension, FTS infrastructure on documents (document_fts), IVFFLAT index (lists=100) unless light init.

External services and environment
- Ollama: expected at http://localhost:11434 (rag_pipeline.list_ollama_models, generation endpoints).
- AWS Bedrock: AWS_REGION or AWS_DEFAULT_REGION, BEDROCK_MODEL_ID for default. Auth via environment, AWS CLI default profile, or IAM role.
- OCI Generative AI: requires user/tenancy/fingerprint/key_file and COMPARTMENT OCID + model OCID; region required.
- OpenSearch: OPENSEARCH_HOST, OPENSEARCH_INDEX, creds, timeout/retries, verify_certs, optional shards/replicas. Ensure COGNEO_EMBED_DIM matches mapping dimension.

How to run (typical)
1) Create venv and install dependencies:
   - python3 -m venv venv && source venv/bin/activate
   - pip install --upgrade pip
   - pip install -r requirements.txt
2) Set environment variables (.env or shell), minimally for Postgres:
   - COGNEO_DB_HOST/PORT/USER/PASSWORD/NAME or COGNEO_DB_URL
   - Optional: COGNEO_DB_BACKEND=postgres (default), COGNEO_VECTOR_BACKEND=opensearch (if serving OS)
   - FASTAPI_API_USER/FASTAPI_API_PASS (defaults in README)
   - COGNEO_EMBED_MODEL and COGNEO_EMBED_DIM must align (e.g., 768)
3) Launch:
   - bash run_cogneo_stack.sh
   - FastAPI: http://localhost:8000/health
   - Gradio: http://localhost:7866
   - Streamlit: http://localhost:8501
4) Ingest data:
   - From Streamlit: “Start Ingestion” (handles multi-GPU by partition)
   - Or use ingest/beta_orchestrator.py with session/shards options (see ingest/README.md)
   - Verify via /documents, /search endpoints.

Notable implementation details and pitfalls
- Postgres connector enforces required DB env vars at import time; Streamlit will error early if DB env not set. Use .env loading to avoid crashes.
- Score semantics:
  - Postgres vector search returns distance (lower is better); hybrid in store_postgres inverts and normalizes to compute vector_score_norm. BM25 path is a presence (1.0) not actual BM25—hybrid is “semantic+presence” rather than a true dual-score BM25 in PG (OpenSearch adapter does BM25 over text).
  - OpenSearch vector_score is already “higher is better”; hybrid normalizes directly.
- rag/rag_pipeline.retrieve is a stub; real retrieval is performed by API/UI with adapters, then passed as context.
- Embedding dimension mismatch will fail inserts; ensure COGNEO_EMBED_DIM matches the active model dimension and OpenSearch index mapping.
- OpenSearch index shard/replica count is immutable; to change, recreate with OPENSEARCH_FORCE_RECREATE=1 or a new index name and backfill with tools/reindex_to_opensearch.py.
- Gradio/Streamlit both can attempt AUTO_DDL; ensure DB permissions allow CREATE EXTENSION/DDL, or set COGNEO_AUTO_DDL=0 and pre-provision schema.
- Oracle/Bedrock/OCI code paths are env-gated; missing SDKs will be handled by informative errors but endpoints rely on those deps.

Data flow end-to-end
1) Ingest: Files (.txt/.html) -> loader.parse_* -> chunk_document (<=1500 chars per chunk) -> embed batch -> DB inserts (Documents + Embeddings w/ chunk_metadata) -> optional dual-write to OpenSearch (beta pipeline) or one-off backfill tool.
2) Query:
   - UI calls FastAPI search endpoints -> storage.registry adapter -> retrieval (DB or OpenSearch) -> returns fields including text/source/format/citation/chunk_metadata.
   - UI composes RAG request (context_chunks, sources, chunk_metadata) -> API /search/rag|oci_rag|bedrock_rag -> RAG pipeline -> LLM -> answer, with citations in output.

Files of interest (selected)
- fastapi_app.py: API endpoints, auth, CORS/Prometheus/rate-limit middleware, DB schema bootstrap
- app.py: Streamlit UI, ingestion runner, hybrid search + RAG, streaming chat
- gradio_app.py: Gradio UI with tabs for RAG/FTS/Chat/Agentic
- storage/registry.py, storage/interfaces.py, storage/adapters/{pgvector,opensearch}.py
- db/store.py (dispatcher), db/store_postgres.py (models + search + DDL), db/connector_postgres.py
- rag/rag_pipeline.py, rag/oci_rag_pipeline.py
- ingest/loader.py, embedding_worker.py, ingest/README.md (+ ingest beta pipeline modules)
- run_cogneo_stack.sh, stop_cogneo_stack.sh
- requirements.txt, README.md

Conclusion
- CogNeo provides a production-oriented, modular RAG stack with pluggable vector backends, multiple LLM providers, robust ingestion, and two UIs. PostgreSQL remains the system-of-record with optional OpenSearch serving parity. The APIs expose consistent result shapes enriched with chunk metadata to power legal-grade citations and agentic reasoning UIs.

This completes the scan and understanding of the codebase.