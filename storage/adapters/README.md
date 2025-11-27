CogNeo Storage Adapters

Overview
- Adapters implement a unified VectorSearchAdapter interface (storage/interfaces.py):
  - create_all_tables() or safe no-op bootstrap
  - search_vector(query_vec, top_k)
  - search_bm25(query, top_k)
  - search_hybrid(query, top_k, alpha)
  - search_fts(query, top_k, mode)
  - index_chunks(chunks, vectors, source_path, fmt) for serving engines that accept bulk indexing (e.g., OpenSearch)
- Adapter selection for serving retrieval is resolved in storage/registry.py by environment:
  - COGNEO_VECTOR_BACKEND wins (opensearch | oracle | postgres/pgvector)
  - Falls back to COGNEO_DB_BACKEND when COGNEO_VECTOR_BACKEND is unset
  - Defaults to pgvector (Postgres)

Return shape parity
Regardless of backend, search endpoints expect these keys as applicable:
- doc_id: int
- chunk_index: int
- score: float (engine-native score; vector distance or bm25)
- vector_score: float (vector-specific score/distance; may mirror score for KNN)
- bm25_score: float (presence score used in hybrid; 1.0 indicates BM25 presence)
- hybrid_score: float (alpha-weighted fusion; used by hybrid endpoints)
- citation: str (commonly "{source}#chunk{chunk_index}")
- text: str (chunk text; DB uses document.content; chunking strategy controls segment)
- source: str (file path or source identifier)
- format: str (txt, html, …)
- chunk_metadata: dict or null (URL, section indices, tokens_est, etc.)

Adapters

1) Postgres (pgvector) — storage/adapters/pgvector.py
- Delegates to db/store_postgres.py for:
  - create_all_tables: enables extensions vector/pg_trgm/etc., creates tables + FTS + triggers + optional IVFFLAT
  - search_vector: cosine distance over embeddings.vector, joined to documents
  - search_bm25: ILIKE-baseline search over documents.content
  - search_hybrid: performs vector + bm25, normalizes vector score, alpha-blend to hybrid_score; adds citation
  - search_fts: searches documents.document_fts and embeddings.chunk_metadata (text), dedups by URL/doc_id
- index_chunks: not implemented (DB is system of record)

2) Oracle 26ai — storage/adapters/oracle26ai.py
- Delegates to db/store_oracle.py (implementation not shown here)
- Mirrors the same public functions as Postgres for search and bootstrap
- index_chunks: not implemented (DB is system of record)

3) OpenSearch (KNN) — storage/adapters/opensearch.py
- Creates an index with mapping parity (create_all_tables):
  - Fields: doc_id (long), chunk_index (integer), citation (keyword), text (text), source (keyword), format (keyword), chunk_metadata (object, enabled), vector (knn_vector with hnsw/cosinesimil)
- Bulk indexing (index_chunks):
  - Accepts chunks + vectors aligned by position
  - Supports parity fields in chunk dict: doc_id, chunk_index, text, chunk_metadata
  - Computes stable _id="{doc_id}#{chunk_index}" when identifiers are present
  - Computes citation="{source}#chunk{chunk_index}"
- Search behavior:
  - search_vector: KNN over vector; returns doc_id, chunk_index, score=vector_score, bm25_score=0.0, citation, text, source, format, chunk_metadata
  - search_bm25: match on text; returns doc_id, chunk_index, score=bm25_score, bm25_score=1.0 presence signal, citation, text, source, format, chunk_metadata
  - search_hybrid: alpha-weighted fusion of normalized vector_score and bm25 presence; returns hybrid_score, citation, and parity fields
  - search_fts: multi_match over text + chunk_metadata.*; returns doc_id, chunk_index, source, content/text, chunk_metadata, snippet=None, search_area="both"

Serving modes
- DB as system of record (default):
  - COGNEO_DB_BACKEND=postgres | oracle
  - COGNEO_VECTOR_BACKEND unset or set to postgres/oracle
  - Ingestion writes to DB (documents + embeddings); retrieval served via DB adapter
- OpenSearch as retrieval layer:
  - COGNEO_VECTOR_BACKEND=opensearch
  - OPENSEARCH_HOST=http(s)://host:9200
  - OPENSEARCH_INDEX=cogneo_chunks_v2 (recommended name)
  - OPENSEARCH_USER / OPENSEARCH_PASS if required
  - COGNEO_EMBED_DIM must match the embedding dimension

One-off backfill to OpenSearch (non-disruptive)
- Tool: tools/reindex_to_opensearch.py
  - Instantiates OpenSearchAdapter directly (does NOT switch serving adapter)
  - Reads from the active DB backend (db/store.py selects postgres/oracle via COGNEO_DB_BACKEND)
  - Writes doc_id, chunk_index, text, chunk_metadata + vectors to OpenSearch
- Environment required:
  - OPENSEARCH_HOST, OPENSEARCH_INDEX, OPENSEARCH_* auth as needed
  - COGNEO_DB_BACKEND + DB credentials for reading
- Usage:
  - python tools/reindex_to_opensearch.py
  - If you had an older OS index without doc_id/chunk_index/citation, create a new index (e.g., cogneo_chunks_v2) before running

Optional ingestion dual-write (env-gated)
- Controlled by: COGNEO_VECTOR_DUAL_WRITE in {"1","true","yes","os","opensearch"}
- Implemented in: ingest/beta_worker.py (in _db_insert_with_retry path)
  - Primary path: embed -> insert into DB (documents + embeddings)
  - If dual-write enabled:
    - Worker obtains the doc_ids for current batch
    - Builds parity OS chunk docs: {doc_id, chunk_index, text, chunk_metadata}
    - Calls OpenSearchAdapter.index_chunks with aligned vectors and per-file source/format
  - Best-effort: OS errors are logged as warnings and ingestion continues
- Required env for dual-write:
  - OPENSEARCH_HOST, OPENSEARCH_INDEX, OPENSEARCH_USER/PASS when applicable
  - Ensure OpenSearch index exists (mapping is ensured on first call by OpenSearchAdapter.create_all_tables when using reindex tool; dual-write expects a pre-created index)

Hybrid retrieval parity details
- Postgres:
  - vector_hits: cosine distance ascending; normalized to [0,1] as vector_score_norm (lower distance -> higher norm)
  - bm25_hits: presence signal (bm25_score=1.0)
  - hybrid_score = alpha * vector_score_norm + (1 - alpha) * bm25_score
  - citation = f"{source}#chunk{chunk_index}"
- OpenSearch:
  - Uses KNN _score as vector_score, normalizes across the merged set
  - Uses bm25 presence (bm25_score=1.0) in the fusion
  - Computes hybrid_score the same way; provides citation the same way

TLS and connectivity notes (OpenSearch)
- For HTTPS endpoints:
  - Provide trusted certificates via your infra (LB/cluster) or use a proper CA/SAN; configure OPENSEARCH client verify/certs appropriately at deployment
- For SSH/bastion tunnels:
  - Ensure tunneling allows forwarding; verify with curl
  - Consider using a data node endpoint for HTTP if the LB enforces SNI/ALPN with public names

Troubleshooting
- Missing fields in OpenSearch results (doc_id/chunk_index/citation or hybrid fields):
  - Ensure you are using the updated adapter and reindexed with the new mapping (or created a fresh index)
- Hybrid returns only BM25:
  - Check embedding model availability; embedding failure path falls back to bm25-only
- Dual-write no-ops:
  - Verify COGNEO_VECTOR_DUAL_WRITE is set to a truthy value (1/true/yes/os/opensearch)
  - Verify OPENSEARCH_* env values and that the index exists and is reachable

Summary
- Keep DB (Postgres/Oracle) as the system of record for ingestion and application state
- Use OpenSearch as a retrieval layer either:
  - one-off backfill (tools/reindex_to_opensearch.py) while app keeps using DB, or
  - switch serving to OpenSearch with COGNEO_VECTOR_BACKEND=opensearch
- Optional dual-write provides near-real-time OpenSearch population without disrupting ingestion
