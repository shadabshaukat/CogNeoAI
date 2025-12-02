# AUSLegalSearch v3 — Operations Handover (Beta Ingestion and Tools)

Version: 1.0  
Audience: Operations / SRE / DevOps  
Scope: Ingestion (ingest/), SQL Benchmark tool, Delete-by-URL tool  
Document type: Operational Guide (Word-friendly style; copy/paste into Word)

-------------------------------------------------------------------------------

1. EXECUTIVE SUMMARY

This handover explains how to run and maintain the Beta ingestion pipeline and tooling:

- Ingestion (ingest/):
  - Multi‑GPU orchestration with dynamic sharding and size‑aware scheduling
  - Per‑GPU workers with CPU/GPU pipelining, token‑aware chunking, batched embeddings, and DB persistence
  - Monitors progress and provides per‑file metrics/logs
- Tools (tools/):
  - SQL Latency Benchmark utility
  - Delete-by-URL utility (single/bulk) for removing source content by URL from the DB

You will find examples and commands to run full or sampled ingests, monitor logs, resume partially processed runs, and safely delete/reload content.

-------------------------------------------------------------------------------

2. SYSTEM & ENVIRONMENT

2.1. Prerequisites

- Python 3.10+ with virtualenv; project dependencies installed:
  - pip install -r requirements.txt
- PostgreSQL with:
  - pgvector extension installed (CREATE EXTENSION IF NOT EXISTS vector)
  - pg_trgm (trigram) extension (CREATE EXTENSION IF NOT EXISTS pg_trgm)
  - uuid-ossp, fuzzystrmatch recommended
- GPU(s) optional but recommended for speed (nvidia-smi available)
- Disk space for model cache (HF_HOME) and data/logs

2.2. Environment Variables (.env)

Load once per shell:
- set -a; source .env; set +a

Key variables (see .env for full list and descriptions):
- Database:
  - AUSLEGALSEARCH_DB_URL or AUSLEGALSEARCH_DB_HOST/PORT/USER/PASSWORD/NAME
  - AUSLEGALSEARCH_DB_POOL_SIZE, AUSLEGALSEARCH_DB_MAX_OVERFLOW, AUSLEGALSEARCH_DB_POOL_RECYCLE, AUSLEGALSEARCH_DB_POOL_TIMEOUT
- Embeddings:
  - AUSLEGALSEARCH_EMBED_MODEL (default nomic-ai/nomic-embed-text-v1.5)
  - AUSLEGALSEARCH_EMBED_DIM (default 768; must match model)
  - AUSLEGALSEARCH_EMBED_BATCH (per-GPU batch size; auto backoff on OOM)
- Ingestion/Worker:
  - AUSLEGALSEARCH_CPU_WORKERS (default min(cores-1,8))
  - AUSLEGALSEARCH_PIPELINE_PREFETCH (default 64; consider 96–128 on large RAM)
  - AUSLEGALSEARCH_SORT_WORKER_FILES=1 (size-desc per worker to reduce tails)
  - AUSLEGALSEARCH_SCHEMA_LIGHT_INIT=1 (skip heavy backfills on first init)
- Timeouts:
  - AUSLEGALSEARCH_TIMEOUT_PARSE/CHUNK/EMBED_BATCH/INSERT/SELECT
- Tools:
  - AUSLEGALSEARCH_SHOWSQL_MAXURLS (for delete-by-URL --show-sql bulk cap)
- GPU tuning (optional):
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  - OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1

-------------------------------------------------------------------------------

3. DATA MODEL & SCHEMA OVERVIEW

Primary ingestion tables (db/store.py):
- documents
  - id (PK), source (string path/URL), content (text), format (txt/html)
  - document_fts (tsvector) — FTS maintained by trigger (see 6.2)
- embeddings
  - id (PK), doc_id (FK->documents.id), chunk_index (int)
  - vector (Vector(EMBEDDING_DIM)), chunk_metadata (JSONB)
  - Stores per-chunk embeddings and metadata (e.g., url, type, title, section info)
- embedding_sessions
  - Track ingestion sessions (name, directory, times, status, totals/processed)
- embedding_session_files
  - One row per file per session; status pending/complete/error; unique(session_name, filepath)

Indices and extensions:
- CREATE EXTENSION vector;
- CREATE EXTENSION pg_trgm;
- documents.document_fts GIN trigram index maintained by trigger
- embeddings vector IVFFLAT index built unless LIGHT_INIT=1
- Additional indexes described and created in db/store.py during create_all_tables()

-------------------------------------------------------------------------------

4. CHUNKING, INGESTION & EMBEDDING LOGIC

4.1. Chunking

Paths: ingest/semantic_chunker.py
- chunk_legislation_dashed_semantic(doc_text, base_meta, cfg)
  - Parses dashed-header sections; preserves document-level metadata (PROTECTED_DOC_KEYS)
  - Attaches per-section info: section_idx/title, tokens_est, chunk_idx
- chunk_document_semantic(doc_text, base_meta, cfg)
  - Heading/paragraph/sentence token-aware merge with overlap; target/overlap/max tokens configurable
- chunk_generic_rcts (optional fallback)
  - LangChain RecursiveCharacterTextSplitter (enable via AUSLEGALSEARCH_USE_RCTS_GENERIC=1)
- Fallback on timeouts
  - Character-window chunker as last resort (controlled by env)

4.2. Ingestion Pipeline

Paths: ingest/beta_orchestrator.py, ingest/beta_worker.py
- Orchestrator (multi-GPU):
  - Detects GPU count (or use --gpus)
  - Splits file list into shards (default GPUs*4; control via --shards)
  - Dynamic scheduling: as GPUs finish shards, assigns next shard (reduces tail latency)
  - Size-aware shard formation (partition_by_size) auto-enabled when size skew high (Gini≥0.6)
  - Coverage validation before launch: ensures each file appears exactly once across shards
    - Writes {session}.partition.manifest.json and {session}.partition.validation.json (on failure)
  - Creates child embedding sessions: {session}-gpu<i>
- Worker (per-GPU):
  - CPU Stage (ProcessPool): parse + chunk with deadlines; per-file metrics (parse/chunk time, tokens)
  - GPU Stage: batched embed (auto OOM backoff)
  - DB Stage: insert documents + embeddings in a transaction; update per-file status and session progress
  - Resumable: skips files already marked “complete” for that session

4.3. Logging & Metrics

- logs/{child}.success.log: TSV lines with parse/chunk/embed/insert metrics by default
- logs/{child}.error.log: filepaths that failed
- logs/{child}.errors.ndjson: structured errors when AUSLEGALSEARCH_ERROR_DETAILS=1
- Orchestrator (wait mode): aggregates into {session}.success.log and {session}.error.log with headers/summary

-------------------------------------------------------------------------------

5. RUNBOOK — INGESTION

5.1. Full Ingest (Recommended)

Example (4 GPUs, dynamic sharding by size):
```bash
set -a; source .env; set +a

python -m ingest.beta_orchestrator \
  --root "/abs/path/to/data" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --gpus 4 \
  --shards 16 \
  --balance_by_size \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```

5.2. Sample/Preview Scan

Pick one file per folder (skip year directories):
```bash
python -m ingest.beta_orchestrator \
  --root "/abs/path/to/data" \
  --session "beta-sample-$(date +%Y%m%d-%H%M%S)" \
  --sample_per_folder \
  --gpus 2 \
  --log_dir "/abs/path/to/logs"
```

5.3. Resume/Remaining Files

Build remaining partition for a child (example child session: beta-full-YYYMMDD-HHMMSS-gpu3):
```bash
session=beta-full-YYYYMMDD-HHMMSS
child=${session}-gpu3
proj=/abs/path/auslegalsearchv3
logs="$proj/logs"
part="$proj/.beta-gpu-partition-${child}.txt"

awk -F'\t' '{print $1}' "$logs/${child}.success.log" 2>/dev/null | sed '/^#/d' > /tmp/processed.txt
cat "$logs/${child}.error.log" 2>/dev/null >> /tmp/processed.txt
sort -u /tmp/processed.txt -o /tmp/processed.txt
sort -u "$part" -o /tmp/part.txt
comm -23 /tmp/part.txt /tmp/processed.txt > "$proj/.beta-gpu-partition-${child}-remaining.txt"

CUDA_VISIBLE_DEVICES=3 \
python -m ingest.beta_worker ${child}-r1 \
  --partition_file "$proj/.beta-gpu-partition-${child}-remaining.txt" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "$logs"
```

-------------------------------------------------------------------------------

6. FULL TEXT SEARCH (FTS) & RANKING NOTES

6.1. FTS Column

- documents.document_fts is maintained by trigger in db/store.py:
  - On INSERT/UPDATE: document_fts := to_tsvector('english', coalesce(content,''))
- If backfill skipped (LIGHT_INIT=1), run once:
  ```sql
  UPDATE documents
  SET document_fts = to_tsvector('english', coalesce(content, ''))
  WHERE document_fts IS NULL;
  ```

6.2. ts_rank vs ts_rank_cd

- ts_rank_cd (coverage density) is recommended for legal passages where compact matching matters
- IMPORTANT: Ranking is meaningful only when you constrain to matches:
  - WHERE d.document_fts @@ plainto_tsquery('english', :q)

6.3. Trigram Performance

- Use pg_trgm GIN for title/substring shortlist:
  ```sql
  CREATE EXTENSION IF NOT EXISTS pg_trgm;
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_title_trgm
    ON embeddings USING GIN (md_title_lc gin_trgm_ops);
  ```
- Prefer:
  - set_limit(0.35–0.45) with % operator and KNN ordering (<->) for fast shortlist
  - Avoid plain similarity(a,b) > const; it typically doesn’t use GIN efficiently

See: docs/perf_trigram_title_search.md

-------------------------------------------------------------------------------

7. TOOLS — SQL LATENCY BENCHMARK

Docs: tools/README-bench-sql-latency.md  
Script: tools/bench_sql_latency.py

Purpose:
- Measure p50/p95 latency for vector/FTS/metadata queries
- Exercise optimized scenarios (cases by citation/name, legislation by title, hybrid ANN)

Quickstart:
```bash
python3 tools/bench_sql_latency.py --scenario baseline \
  --query "fiduciary duty in NSW" \
  --top_k 10 --runs 5 --probes 12
```

Prereqs:
- Post-load indexes (schema-post-load/README.md) recommended for best performance

-------------------------------------------------------------------------------

8. TOOLS — DELETE BY URL (SINGLE/BULK)

Docs: tools/README-delete-url.md  
Script: tools/delete_url_records.py

Purpose:
- Delete embeddings whose chunk_metadata->>'url' = URL
- Then delete documents for those doc_ids if no embeddings remain (safe-orphan removal)
- Supports single URL or bulk via a file (one URL per line)
- Show literal SQL via --show-sql (AUSLEGALSEARCH_SHOWSQL_MAXURLS limits bulk printing)

Usage:
```bash
# Preview (dry-run) for single URL
python -m tools.delete_url_records \
  --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/cth/HCA/2022/35.html" \
  --dry-run

# Delete single without prompt
python -m tools.delete_url_records \
  --url "https://austlii.edu.au/cgi-bin/viewdoc/au/cases/cth/HCA/2022/35.html" \
  --yes

# Bulk: urls.txt contains one URL per line (blank/# ignored)
python -m tools.delete_url_records --url-file "/abs/path/urls.txt" --dry-run
python -m tools.delete_url_records --url-file "/abs/path/urls.txt" --yes

# Show literal SQL (single or bulk; capped by AUSLEGALSEARCH_SHOWSQL_MAXURLS)
python -m tools.delete_url_records --url "..." --show-sql
python -m tools.delete_url_records --url-file "/abs/path/urls.txt" --show-sql
```

-------------------------------------------------------------------------------

9. OPERATIONAL PLAYBOOK

9.1. Provision/Init

- Ensure DB reachable; pgvector/pg_trgm enabled
- Run a small sample ingest to verify:
  - python -m ingest.beta_orchestrator --sample_per_folder ...
  - Inspect logs for success/error counts
  - Spot-check DB:
    ```sql
    SELECT count(*) FROM documents;
    SELECT count(*) FROM embeddings;
    ```

9.2. Full Ingest

- Run with proper --gpus and --shards (= GPUs*4 baseline; adjust for scale)
- Enable --balance_by_size for skewed corpora
- Monitor logs in real time (tail -f logs/*.log)
- Coverage: verify partition.manifest.json exists and looks sensible

9.3. Troubleshooting

- ts_rank_cd = 0:
  - Ensure WHERE document_fts @@ plainto_tsquery('english', :q) present before computing rank
  - Ensure document_fts is populated (run backfill if needed)
- ILIKE slow on titles:
  - Switch to trigram shortlist (set_limit() + % + ORDER BY <->; see doc)
- Vector distance vs similarity:
  - pgvector <=> is cosine distance (smaller is better); either convert to similarity (1 - distance) or subtract distance in weighted score
- Resume:
  - Build remaining partition and relaunch the child worker as shown in section 5.3

9.4. Safety

- Always run delete_url_records with --dry-run first
- Maintain encrypted backups/snapshots if supported in your environment
- Use LIGHT_INIT=1 on brand new instances to avoid long backfills; run backfills later during maintenance

-------------------------------------------------------------------------------

10. APPENDIX

10.1. Common Commands

- Full ingest (4 GPUs, 16 shards, size-balanced):
  ```bash
  python -m ingest.beta_orchestrator --root "/data" --session "beta-$(date +%F-%H%M)" \
    --gpus 4 --shards 16 --balance_by_size --log_dir "./logs"
  ```

- Sample ingest:
  ```bash
  python -m ingest.beta_orchestrator --root "/data" --session "beta-sample" \
    --sample_per_folder --log_dir "./logs"
  ```

- Delete URL (bulk):
  ```bash
  python -m tools.delete_url_records --url-file "/data/deletion-urls.txt" --dry-run
  python -m tools.delete_url_records --url-file "/data/deletion-urls.txt" --yes
  ```

10.2. Indexes & Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- FTS index is created in create_all_tables(), plus triggers for document_fts
```

10.3. Logs Location

- /abs/path/to/logs by default (override with --log_dir)

10.4. References

- ingest/README.md
- tools/README-bench-sql-latency.md
- tools/README-delete-url.md
- schema-post-load/README.md
- docs/perf_trigram_title_search.md

-------------------------------------------------------------------------------

END OF DOCUMENT
