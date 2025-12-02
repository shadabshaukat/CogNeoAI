# OCI PostgreSQL + pgvector for Generative AI — Architecture & Best Practices (Slide Deck Outline)

Audience: Platform/Ops/DBA/Engineering  
Duration: 60 minutes (+ Q&A)  
Format: PowerPoint-friendly; copy/paste per slide. Speaker notes included.  
Repo Context: AUSLegalSearch v3 (Postgres + pgvector + pg_trgm + FTS)

--------------------------------------------------------------------------------
Slide 1 — Title

OCI Database with PostgreSQL + pgvector  
Operating GenAI Retrieval at Scale (Connection, Pooling, Performance, Indexing, FTS)

- Production patterns from AUSLegalSearch v3
- Focus on connection, pooling, performance, and search optimizations
- OCI‑specific guidance and operational guardrails

Speaker Notes:
- Frame as a practical, operations‑centric session for GenAI apps built on OCI PostgreSQL.

--------------------------------------------------------------------------------
Slide 2 — Agenda

- Why PostgreSQL + pgvector on OCI
- Architecture Overview
- Connection & Pooling Strategy
- Vector Indexing (IVFFLAT/HNSW) & Tunables
- Full‑Text Search & Trigram Shortlist
- Query Patterns & Hybrid Scoring
- Maintenance, Observability, and HA
- Security & Cost Optimization
- Roadmap & Q&A

Speaker Notes:
- Emphasize immediately actionable best practices.

--------------------------------------------------------------------------------
Slide 3 — Why PostgreSQL + pgvector on OCI

- Single operational surface: vectors, full text, metadata in one DB
- Mature ecosystem and extensions on OCI:
  - pgvector (ANN), pg_trgm (trigram), tsvector (FTS), uuid‑ossp, fuzzystrmatch
- Flexible: supports both pure vector search and hybrid (vector + FTS)
- Proven: Underpins AUSLegalSearch v3 ingestion and retrieval

Speaker Notes:
- Reduces infra complexity and speeds iteration versus polyglot stores.

--------------------------------------------------------------------------------
Slide 4 — Reference Architecture (AUSLegalSearch v3)

[Diagram placeholder: Client ➝ API (FastAPI/Streamlit/Gradio) ➝ OCI PostgreSQL + Extensions ➝ Ingestion Workers (GPU) ➝ Object Storage (optional)]

- OCI PostgreSQL: pgvector + pg_trgm + FTS
- Ingest Workers: chunk + embed + persist
- Application: hybrid retrieval; weighted scoring
- Tools: Benchmark (p50/p95), Delete‑by‑URL

Speaker Notes:
- Mention DB is central for both ingest and query.

--------------------------------------------------------------------------------
Slide 5 — Connection & Pooling (SQLAlchemy / Engine)

- Engine config (db/connector.py):
  - pool_pre_ping=True
  - pool_size, max_overflow, pool_recycle, pool_timeout
  - connect_args: keepalives, connect_timeout, statement_timeout (options)
- Keepalives (Linux):
  - keepalives=1, keepalives_idle=30s, keepalives_interval=10s, keepalives_count=5
- Statement Timeout:
  - Set per connection (e.g., AUSLEGALSEARCH_DB_STATEMENT_TIMEOUT_MS=60000) to kill outliers
- Consider PgBouncer for high‑concurrency read workloads (pooled mode)

Speaker Notes:
- Right‑size pool_size to app concurrency; avoid thousands of direct connections.

--------------------------------------------------------------------------------
Slide 6 — Pooling Best Practices (OCI)

- Per‑process pool sizing:
  - pool_size = CPU threads in API worker (start small, grow conservatively)
  - max_overflow for bursts (10–20 typical)
- pool_recycle ~ 1800s to avoid idle server disconnects
- pool_timeout ~ 30s to bound waiting time
- Don’t combine PgBouncer + ORM aggressive pooling without planning (double pooling)

Speaker Notes:
- Monitor wait events and connection spikes to tune.

--------------------------------------------------------------------------------
Slide 7 — pgvector Overview

- Distance operators:
  - <=> cosine distance (lower is better)
  - <-> Euclidean distance
  - <#> negative inner product
- Index types:
  - IVFFLAT (lists at build time; probes at query time)
  - HNSW (pgvector ≥ 0.7; ef_construction, ef_search tunables)
- Normalization:
  - For cosine, normalize vectors (unit length) to make distance behave well

Speaker Notes:
- Most GenAI apps use cosine.

--------------------------------------------------------------------------------
Slide 8 — Building Vector Indexes (IVFFLAT)

- Example:
  ```sql
  CREATE INDEX IF NOT EXISTS idx_embeddings_vector_ivfflat_cosine
  ON embeddings USING ivfflat (vector vector_cosine_ops)
  WITH (lists = 100);
  ```
- Query‑time probes:
  - SET ivfflat.probes = 10–20 for recall/perf trade‑off
- Build tips:
  - Large maintenance_work_mem for faster build
  - ANALYZE after build

Speaker Notes:
- Tune lists based on corpus size; more lists → better recall but larger index.

--------------------------------------------------------------------------------
Slide 9 — HNSW (When Available)

- Benefits:
  - Better p95 on large datasets; dynamic graph search
- Tunables:
  - ef_construction (build), ef_search (query)
- Trade‑offs:
  - Larger build time and memory
- Choose IVFFLAT vs HNSW based on N, QPS, and recall targets

Speaker Notes:
- For TB‑scale or very large N, HNSW often shines.

--------------------------------------------------------------------------------
Slide 10 — FTS (tsvector) Essentials

- Column: documents.document_fts (tsvector)
- Trigger: maintains document_fts := to_tsvector('english', coalesce(content,''))
- Rank:
  - Use ts_rank_cd (coverage density) or ts_rank with normalization flag (32 commonly used)
- Important:
  - Always constrain with @@ tsquery before ranking; else rank = 0 for non‑matches

Speaker Notes:
- Missing @@ filter is a common source of “always zero” ranks.

--------------------------------------------------------------------------------
Slide 11 — Trigram Shortlist (pg_trgm)

- Index:
  ```sql
  CREATE EXTENSION IF NOT EXISTS pg_trgm;
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_title_trgm
  ON embeddings USING GIN (md_title_lc gin_trgm_ops);
  ```
- Pattern:
  - SELECT set_limit(0.40);
  - WHERE md_title_lc % :q
  - ORDER BY md_title_lc <-> :q (KNN GIN)
- Avoid: similarity(a,b) > const alone (often no index; seq scan)

Speaker Notes:
- Use % operator with set_limit; then KNN <-> for top‑K.

--------------------------------------------------------------------------------
Slide 12 — Hybrid Retrieval Pattern

- Build shortlist on text (FTS or trigram), then apply vector within shortlist
- Weighted scoring:
  - score = w_vec * vec_sim + w_kw * ts_rank_cd
  - vec_sim = 1 − cosine_distance (if vectors normalized), else subtract distance
- Keep weights interpretable and tune per corpus

Speaker Notes:
- Two‑stage approach avoids full table vector KNN + expensive filters.

--------------------------------------------------------------------------------
Slide 13 — Query Patterns (Do/Don’t)

Do:
- WHERE document_fts @@ tsquery before ts_rank/ts_rank_cd
- set_limit(0.35–0.45) + % + ORDER BY <-> for fast trigram shortlist
- Use LIMIT shortlist (200–1000) before heavier ranking
- Use CROSS JOIN for tsquery once, not for every row

Don’t:
- similarity(a,b) > 0.7 alone (likely seq scan and 0 results)
- DISTINCT over entire embeddings prematurely (dedupe after shortlist)
- Unbounded ORDER BY vector distance without filters

Speaker Notes:
- Show a before/after EXPLAIN example if time permits.

--------------------------------------------------------------------------------
Slide 14 — Schema & Indexing Checklists

- documents(id PK, source, content, format, document_fts)
  - GIN on document_fts
- embeddings(id PK, doc_id FK, chunk_index, vector, chunk_metadata)
  - IVFFLAT/HNSW on vector
  - GIN trigram on md_title_lc
  - BTree filters (md_type, date ranges) as needed

Speaker Notes:
- Ensure ANALYZE after bulk loads.

--------------------------------------------------------------------------------
Slide 15 — Maintenance & Vacuum

- VACUUM (auto) / ANALYZE after large inserts
- Reindex only when necessary; prefer CONCURRENTLY for online
- Track bloat (pgstattuple), especially after deletes (e.g., Delete‑by‑URL)
- Consider partitioning by time/type for very large datasets

Speaker Notes:
- Routine ANALYZE significantly impacts planner quality.

--------------------------------------------------------------------------------
Slide 16 — Observability

- pg_stat_statements: top queries, normalize and tune
- auto_explain: catch slow plans intermittently
- EXPLAIN (ANALYZE, BUFFERS): verify index usage, rechecks
- Application:
  - request ID correlation across API/DB
  - Log ivfflat.probes and set_limit for reproducibility

Speaker Notes:
- Create dashboards for p50/p95, QPS, top queries, deadlocks, timeouts.

--------------------------------------------------------------------------------
Slide 17 — Timeouts & Error Handling

- Server:
  - statement_timeout (session/role)
- Client/Engine:
  - connect_timeout, pool_timeout
- App:
  - Retries with backoff for transient errors
  - Defensive deadlines around parse/chunk/embed/insert

Speaker Notes:
- Prevent “zombie” queries and slow burners impacting SLA.

--------------------------------------------------------------------------------
Slide 18 — Security & Access on OCI

- Network:
  - Private subnets, Security Lists/NSGs, LB/WAF in front of apps
- Auth:
  - Role‑based DB users; least privilege; rotate credentials
- Secrets:
  - Use OCI Vault/Secrets; avoid committing secrets
- TLS:
  - Enforce SSL for DB connections where required

Speaker Notes:
- Align with org standards; audit access and connection sources.

--------------------------------------------------------------------------------
Slide 19 — HA/DR & Backups

- OCI managed backups; PITR
- HA:
  - Multi‑AZ deployments; Read Replicas for heavy read
- DR:
  - Cross‑region replica/restore
- Plan drills; document RTO/RPO for legal workloads

Speaker Notes:
- Clarify expectations with stakeholders.

--------------------------------------------------------------------------------
Slide 20 — Cost & Capacity

- Right‑size shape and storage
- Offload heavy analytics to replicas
- Index sizing:
  - Balance lists (IVFFLAT) and recency of ANALYZE
- Model cache (HF_HOME) on fast disk; reuse across nodes if possible

Speaker Notes:
- Benchmark regularly with tools/bench_sql_latency.py.

--------------------------------------------------------------------------------
Slide 21 — Benchmarking (tools)

- SQL Latency Benchmark:
  - Scenarios for FTS, trigram shortlist, ANN + grouping
  - p50/p95 comparisons before/after index tuning
- Operational Levers:
  - ivfflat.probes, set_limit, shortlist size
  - GIN/HNSW build and ANALYZE state

Speaker Notes:
- Establish repeatable baselines and thresholds.

--------------------------------------------------------------------------------
Slide 22 — Delete‑by‑URL (Ops)

- Safe removal of content:
  - Delete embeddings by chunk_metadata->>'url'
  - Delete documents only if no embeddings remain
- Bulk mode + --dry-run:
  - Validate counts before destructive ops
- Re‑ingest: run ingestion pipeline after deletion

Speaker Notes:
- Reduce ghost references and ensure clean replacements.

--------------------------------------------------------------------------------
Slide 23 — Roadmap

- HNSW rollout and ef_search tuning
- Advanced reranking (MMR/learning‑to‑rank)
- Partitioning strategies for TB‑scale
- Better auto‑tuning of probes/thresholds
- Richer operational dashboards (DB + app)

Speaker Notes:
- Tie to org performance and reliability goals.

--------------------------------------------------------------------------------
Slide 24 — Key Takeaways

- PostgreSQL on OCI is a capable vector + FTS + trigram engine for GenAI
- Connection/pooling/timeout hygiene is critical
- Use index‑friendly patterns (set_limit/%/KNN <->; @@ before rank)
- Observe and iterate: stats, logs, and baseline benchmarks

Speaker Notes:
- End with action items: run benchmark, review slow queries, tune pools.

--------------------------------------------------------------------------------
Appendix — Commands & SQL Cheatsheet

- Enable extensions:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE EXTENSION IF NOT EXISTS pg_trgm;
  ```
- Vector index (IVFFLAT):
  ```sql
  CREATE INDEX IF NOT EXISTS idx_embeddings_vector_ivfflat_cosine
  ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
  ```
  SET ivfflat.probes = 10;
- FTS backfill:
  ```sql
  UPDATE documents
  SET document_fts = to_tsvector('english', coalesce(content, ''))
  WHERE document_fts IS NULL;
  ```
- Trigram shortlist:
  ```sql
  SELECT set_limit(0.40);
  SELECT * FROM embeddings
  WHERE md_title_lc % 'query'
  ORDER BY md_title_lc <-> 'query'
  LIMIT 500;
  ```

Speaker Notes:
- Include this slide as a leave‑behind for DBAs/ops.

--------------------------------------------------------------------------------
End of Slide Deck Outline
