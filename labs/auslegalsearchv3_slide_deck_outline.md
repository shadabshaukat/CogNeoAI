# AUSLegalSearch v3 — Solution Overview (Slide Deck Outline)

Audience: Technical and semi-technical stakeholders (Engineering, Ops, Product)  
Duration: 60 minutes (+ Q&A)  
Format: PowerPoint-friendly; copy/paste per slide. Speaker notes included.  
Repo: auslegalsearchv3

--------------------------------------------------------------------------------
Slide 1 — Title

AUSLegalSearch v3  
Agentic Multi‑Faceted Legal AI Platform (Ollama ⬄ OCI GenAI ⬄ Oracle 23ai)

- Unified ingestion, retrieval, and RAG platform
- High‑throughput ingest, FTS, trigram shortlist, vector ANN, hybrid scoring
- Tools for benchmarking and operational maintenance

Speaker Notes:
- Set context: platform enables legal search and RAG across case law, legislation, treaties, journals.
- Emphasize operational strengths: resumable ingest, multi‑GPU scaling, and powerful tools for ops.

--------------------------------------------------------------------------------
Slide 2 — Agenda

- Platform at a Glance
- Architecture & Components
- Data Model & Retrieval
- Ingestion: Chunking, Embedding, Persistence
- Operator Tooling (Benchmark, Delete‑by‑URL)
- Performance & Ranking Best Practices
- Live Demo (optional)
- Roadmap & Q&A

Speaker Notes:
- Our goal: Provide a deep yet accessible tour of features and operations.

--------------------------------------------------------------------------------
Slide 3 — Platform at a Glance

- End‑to‑end: ingestion ➝ storage ➝ retrieval ➝ RAG chat
- Multi‑model: Local (Ollama) or Oracle GenAI endpoints
- Retrieval modes:
  - Vector ANN (pgvector, cosine)
  - Full Text Search (Postgres tsvector)
  - Trigram shortlist (pg_trgm)
  - Hybrid (weighted)
- Operational features:
  - Multi‑GPU ingest with dynamic sharding
  - Resumability, rich logging, per‑file metrics
  - Tools: SQL latency benchmark; delete by URL

Speaker Notes:
- Emphasize flexibility and pragmatism: use what performs best for each task (hybrid works well for legal text).

--------------------------------------------------------------------------------
Slide 4 — Architecture Overview

[Diagram placeholder: Client UIs -> FastAPI/Gradio/Streamlit -> DB (Postgres+pgvector) + Ingest Workers + OCI/Ollama]

- UIs: Streamlit (chat), Gradio (demos), FastAPI (REST)
- Core DB: Postgres + pgvector + pg_trgm + FTS (tsvector)
- Ingest: Orchestrator (multi‑GPU) + Workers (CPU parse/chunk + GPU embed)
- LLM Providers: Ollama local; OCI GenAI
- Tools: Benchmark, Delete‑by‑URL, post‑load schema scripts

Speaker Notes:
- Stress modularity and that ops can scale independently (ingest vs query).

--------------------------------------------------------------------------------
Slide 5 — Repository Map (Key Paths)

- ingest/
  - beta_orchestrator.py (multi‑GPU orchestration, dynamic sharding)
  - beta_worker.py (pipelined worker)
  - semantic_chunker.py (token‑aware chunkers)
  - loader.py (txt/html parsers)
- db/
  - connector.py (engine, .env loader, pools)
  - store.py (ORM models, create_all_tables, FTS triggers)
- tools/
  - bench_sql_latency.py (scenarios, p50/p95)
  - delete_url_records.py (single/bulk delete by URL)
- docs/
  - perf_trigram_title_search.md (trigram best practices)
  - ops_handover_beta_ingestion_and_tools.md (ops runbook)

Speaker Notes:
- Keep this slide open during Q&A for quick pointers.

--------------------------------------------------------------------------------
Slide 6 — Core Data Model (Ingestion)

- documents
  - id (PK), source (path/URL), content (text), format (txt/html)
  - document_fts (tsvector; maintained by trigger)
- embeddings
  - id (PK), doc_id (FK→documents.id), chunk_index
  - vector (Vector(dim)), chunk_metadata (JSONB; includes url, type, title, section)
- embedding_sessions / embedding_session_files
  - Tracks ingest sessions and per‑file statuses

Speaker Notes:
- Explain data grain: one “document” row per chunk; embeddings aligned per chunk.

--------------------------------------------------------------------------------
Slide 7 — Retrieval & Ranking

- FTS: documents.document_fts + ts_rank_cd (coverage density)
- Trigram shortlist: GIN on md_title_lc; set_limit(0.35–0.45) + % + KNN <-> ORDER BY
- Vector ANN: pgvector cosine (<=> distance; lower is better)
- Hybrid: Weighted score = w_vec * vec_sim + w_kw * ts_rank_cd
  - vec_sim = 1 − cosine_distance (if vectors normalized); or subtract raw distance

Speaker Notes:
- Stress importance of WHERE document_fts @@ tsquery before ranking with ts_rank_cd.

--------------------------------------------------------------------------------
Slide 8 — Chunking Strategies

- Dashed‑header‑aware chunking (legislation/sections)
  - Keeps doc‑level metadata; section titles, identifiers, tokens_est
- Semantic chunking (headings→paragraphs→sentences)
  - Target_tokens, overlap, max_tokens tuned per corpus
- Optional RCTS fallback (LangChain) on timeouts
- Last‑resort character window when heavy content obstructs tokenization

Speaker Notes:
- Chunk sizes impact both embedding latency and retrieval quality.

--------------------------------------------------------------------------------
Slide 9 — Ingestion Pipeline (Flow)

1) Discover files (full vs sample/preview)
2) Parse (.txt/.html), extract metadata blocks
3) Chunk (semantic; dashed header‑aware)
4) Embed (GPU; batch with OOM backoff)
5) Persist documents + embeddings
6) Update session & per‑file status; metrics logging

Speaker Notes:
- Idempotent by design: per‑file status prevents reprocessing; resume workflows supported.

--------------------------------------------------------------------------------
Slide 10 — Multi‑GPU Orchestration

- Dynamic sharding: GPUs*4 default (configurable)
- Greedy size balancing (auto‑enable on high size skew)
- Work‑stealing: finished GPUs receive next shard
- Coverage validation:
  - Writes partition.manifest.json; validates no duplicates/misses
- Per‑child sessions: {session}-gpu<i>

Speaker Notes:
- This setup reduces tail latency and ensures balanced GPU use.

--------------------------------------------------------------------------------
Slide 11 — Operational Logs & Metrics

- Per‑worker logs:
  - {child}.success.log (TSV metrics) and {child}.error.log
  - {child}.errors.ndjson (structured; optional)
- Aggregated by orchestrator in wait mode:
  - {session}.success.log and {session}.error.log
- Partition manifests: session.partition.manifest.json

Speaker Notes:
- Recommend log rotation and separate high‑throughput disk for logs if needed.

--------------------------------------------------------------------------------
Slide 12 — Running Ingestion (Full)

Example:
- python -m ingest.beta_orchestrator --root "/data" --session "beta-$(date +%F-%H%M)" --gpus 4 --shards 16 --balance_by_size --log_dir "./logs"

Tips:
- Export .env first
- Enable size balancing for skewed datasets
- Monitor logs; confirm manifest and child log growth

Speaker Notes:
- Add organization‑specific paths and GPU counts before presenting.

--------------------------------------------------------------------------------
Slide 13 — Resume & Remaining Files

- Use success/error logs and original partition to compute remainder
- Launch worker on remaining list:
  - python -m ingest.beta_worker {child}-r1 --partition_file "…-remaining.txt" --log_dir "./logs"

Speaker Notes:
- Emphasize idempotency and careful naming to avoid conflicts.

--------------------------------------------------------------------------------
Slide 14 — FTS Correctness & Pitfalls

- Always include WHERE document_fts @@ tsquery before ts_rank/ts_rank_cd
- Backfill document_fts for legacy rows:
  - UPDATE documents SET document_fts = to_tsvector('english', coalesce(content, '')) WHERE document_fts IS NULL;
- Choose ts_rank_cd (coverage density) when compactness matters

Speaker Notes:
- Many “rank=0” issues arise from missing @@ filter or empty tsquery.

--------------------------------------------------------------------------------
Slide 15 — Trigram Best Practices (pg_trgm)

- CREATE EXTENSION pg_trgm; GIN on md_title_lc
- set_limit(0.35–0.45) and use % operator
- ORDER BY md_title_lc <-> :q for KNN GIN
- Avoid similarity(a,b) > const alone (typically seq scans)

Speaker Notes:
- Point to docs/perf_trigram_title_search.md for details and SQL templates.

--------------------------------------------------------------------------------
Slide 16 — Vector (pgvector) Do’s & Don’ts

- Cosine operator <=> = distance (lower better)
- Convert to similarity (1 − distance) or subtract distance in weighted score
- Ensure vector dim matches AUSLEGALSEARCH_EMBED_DIM
- Index builds: IVFFLAT lists tuning; HNSW optional in newer versions

Speaker Notes:
- Show before/after improvements from correcting vector term direction.

--------------------------------------------------------------------------------
Slide 17 — Tooling: SQL Latency Benchmark

- tools/bench_sql_latency.py
  - Scenarios: baseline, cases_by_citation, cases_by_name_trgm, legislation_title_trgm, ANN+filters+grouping
  - Output: p50/p95 latency, top results
- Command:
  - python3 tools/bench_sql_latency.py --scenario baseline --query "fiduciary duty in NSW" --top_k 10 --runs 5 --probes 12

Speaker Notes:
- Good for regression checks across index changes or provider swaps.

--------------------------------------------------------------------------------
Slide 18 — Tooling: Delete by URL (Single/Bulk)

- tools/delete_url_records.py
  - --url or --url-file (one per line); supports --dry-run, --yes
  - --show-sql prints literal SQL (cap via AUSLEGALSEARCH_SHOWSQL_MAXURLS)
- Safe orphan delete: removes documents only when no embeddings remain

Speaker Notes:
- Recommend always preview with --dry-run; snapshot DB before bulk deletes.

--------------------------------------------------------------------------------
Slide 19 — Security & Operations

- Secrets in environment; avoid committing secrets
- TLS/WAF/reverse proxy in front of UIs (FastAPI/Gradio/Streamlit)
- Backups/snapshots around index builds or bulk deletes
- Observability: DB metrics, ingestion throughput, GPU usage

Speaker Notes:
- Tie to org compliance standards and logging/retention policies.

--------------------------------------------------------------------------------
Slide 20 — Live Demo (Optional)

- Sample ingest (sample_per_folder)
- Benchmark scenario
- Delete one URL and reload
- Q&A

Speaker Notes:
- Prepare “known good” dataset subsets. Pre‑warm HF cache.

--------------------------------------------------------------------------------
Slide 21 — Roadmap

- Advanced reranking (MMR/learning-to-rank)
- Improved schema (generated columns, post‑load optimizations)
- More granular ops dashboards
- HNSW indices (pgvector >= 0.7) for very large N
- Additional content types and extraction pipelines

Speaker Notes:
- Align with stakeholder priorities and timelines.

--------------------------------------------------------------------------------
Slide 22 — Key Takeaways

- Production‑ready ingestion with dynamic multi‑GPU scheduling
- Hybrid retrieval combining strong FTS and ANN
- Operations tooling simplifies maintenance and tuning
- Clear best practices for performance and ranking

Speaker Notes:
- Reinforce value proposition and next steps.

--------------------------------------------------------------------------------
Slide 23 — Appendix: Commands Cheat‑Sheet

- Full ingest (4 GPUs, 16 shards):
  python -m ingest.beta_orchestrator --root "/data" --session "beta-$(date +%F-%H%M)" --gpus 4 --shards 16 --balance_by_size --log_dir "./logs"

- Sample ingest:
  python -m ingest.beta_orchestrator --root "/data" --session "beta-sample" --sample_per_folder --log_dir "./logs"

- Benchmark:
  python3 tools/bench_sql_latency.py --scenario baseline --query "fiduciary duty in NSW" --top_k 10 --runs 5 --probes 12

- Delete by URL:
  python -m tools.delete_url_records --url-file "/data/urls.txt" --dry-run
  python -m tools.delete_url_records --url-file "/data/urls.txt" --yes

Speaker Notes:
- Provide this slide as a leave‑behind.

--------------------------------------------------------------------------------
Slide 24 — Appendix: References

- docs/ops_handover_beta_ingestion_and_tools.md
- docs/perf_trigram_title_search.md
- ingest/README.md
- tools/README-bench-sql-latency.md
- tools/README-delete-url.md
- schema-post-load/README.md

Speaker Notes:
- Point attendees to these documents for deeper hands‑on details.

--------------------------------------------------------------------------------
Time Budget (60 minutes)

- Overview & Architecture: 10
- Data Model & Retrieval: 10
- Ingestion Deep Dive: 15
- Tooling: 10
- Performance & Ranking Practices: 10
- Q&A / Demo buffer: 5

Speaker Notes:
- Adjust according to audience interest (e.g., more time on trigram/FTS vs. ingest scaling).

--------------------------------------------------------------------------------
End of Slide Deck Outline
