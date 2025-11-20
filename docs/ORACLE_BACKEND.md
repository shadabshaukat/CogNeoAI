# Oracle Database 26ai Backend for AUSLegalSearch v3

This document explains how to run AUSLegalSearch with Oracle Database 26ai as the storage backend, while preserving the existing Postgres-first core logic.

## Overview

- Backend switch is controlled by environment variable `AUSLEGALSEARCH_DB_BACKEND`.
- Default remains Postgres. No behavior changes for existing setups.
- Oracle backend uses `python-oracledb` via SQLAlchemy and stores:
  - Embedding vectors in the native Oracle `VECTOR(dim, FLOAT32, DENSE)` column type.
  - Metadata as the native Oracle `JSON` column type (not CLOB).
- Similarity search is executed SQL-side using `vector_distance(e.vector, :qv)` (exact path).
  - Optional index creation via `DBMS_VECTOR.CREATE_INDEX` (HNSW/IVF) is supported (env-gated).
- Metadata FTS (baseline) uses substring search:
  - `WHERE LOWER(JSON_SERIALIZE(e.chunk_metadata RETURNING CLOB)) LIKE :q`
- The code paths for Postgres and Oracle are separated via dispatchers:
  - `db/store.py` and `db/connector.py` autoload `.env` and select the concrete backend modules (`store_postgres`/`connector_postgres` vs `store_oracle`/`connector_oracle`).

## Switching Backends

- Postgres (default)
  - `AUSLEGALSEARCH_DB_BACKEND=postgres`
- Oracle
  - `AUSLEGALSEARCH_DB_BACKEND=oracle`

## Environment Variables

### Option A: Single DSN URL (recommended when possible)

- `ORACLE_SQLALCHEMY_URL=oracle+oracledb://user:password@myadb_high`

### Option B: Individual fields

- `ORACLE_DB_USER=your_db_user`
- `ORACLE_DB_PASSWORD=your_db_password`
- `ORACLE_DB_DSN=myadb_high`            # TNS alias or EZConnect DSN (host:port/?service_name=...)
- `ORACLE_WALLET_LOCATION=/path/to/wallet/dir`  # Optional; sets TNS_ADMIN for Autonomous DB

### Pool/Timeout tuning (shared pattern with Postgres connector)

- `AUSLEGALSEARCH_DB_POOL_SIZE=10`
- `AUSLEGALSEARCH_DB_MAX_OVERFLOW=20`
- `AUSLEGALSEARCH_DB_POOL_RECYCLE=1800`
- `AUSLEGALSEARCH_DB_POOL_TIMEOUT=30`

### Oracle AI Vector Search (optional index/bootstrap)

- `AUSLEGALSEARCH_ORA_AUTO_VECTOR_INDEX=0|1`       # auto-create a vector index on `embeddings.vector` during bootstrap
- `AUSLEGALSEARCH_ORA_INDEX_TYPE=HNSW|IVF`
- `AUSLEGALSEARCH_ORA_DISTANCE=COSINE|EUCLIDEAN|EUCLIDEAN_SQUARED|DOT|MANHATTAN|HAMMING`
- `AUSLEGALSEARCH_ORA_ACCURACY=90`                 # target accuracy for approximate search
- `AUSLEGALSEARCH_ORA_INDEX_PARALLEL=1`            # parallelism for index build
- `AUSLEGALSEARCH_ORA_HNSW_NEIGHBORS=16`
- `AUSLEGALSEARCH_ORA_HNSW_EFCONSTRUCTION=200`
- `AUSLEGALSEARCH_ORA_IVF_PARTITIONS=100`

> Note: Query-time `APPROX` keyword is not used in the current SQL to avoid syntax/compat issues. The optimizer may still use a compatible vector index when present.

## File Changes in this backend

- `db/store.py`: Dispatcher that exports the same symbols as before, selecting backend by `AUSLEGALSEARCH_DB_BACKEND`
- `db/store_postgres.py`: Postgres models and helpers (pgvector + FTS + JSONB)
- `db/store_oracle.py`: Oracle models and helpers (native `VECTOR` + native `JSON`, SQL `vector_distance`, JSON_SERIALIZE for metadata LIKE)
  - `create_all_tables()` on Oracle creates only core tables by default (users, documents, embeddings, sessions, session_files, chat_sessions, conversion_files). Relational normalization tables are not auto-created.
  - PKs use `Identity()` on integer columns; a sequence+trigger fallback for legacy schemas is attempted when needed.
- `db/connector.py`: Dispatcher for connectors
- `db/connector_postgres.py`: Postgres connector
- `db/connector_oracle.py`: Oracle connector (`python-oracledb` via SQLAlchemy)
  - Exposes Oracle-native `JSON` through a SQLAlchemy type (`JSONType`) and `VECTOR` via a `UserDefinedType`.

## What works on Oracle (native JSON + VECTOR)

- Schema creation for core tables (users, documents, embeddings, sessions, etc.)
- Ingestion (documents + embeddings) using the same upper-layer code paths (env-driven dispatch)
  - Ingest writers serialize `chunk_metadata` for Oracle only (defensive); `JSONType` accepts dict/list binds.
- Embeddings stored in native Oracle `VECTOR(dim, FLOAT32, DENSE)`
- Vector search: SQL-side `vector_distance(e.vector, :qv)` exact ranking
- Optional HNSW/IVF vector indexes via `DBMS_VECTOR.CREATE_INDEX` (env-gated)
- BM25-like search: case-insensitive LIKE over `documents.content`
- Hybrid search: blends SQL vector_distance and LIKE-based results
- FTS endpoint (baseline): LIKE-based search over `documents` and `embeddings.chunk_metadata` using `JSON_SERIALIZE(... RETURNING CLOB)`

## Postgres-only functionality (not in Oracle baseline)

- pgvector-based vector operators/indexes (`<=>`, IVFFLAT/HNSW) and generated expression column patterns in `schema-post-load/`
- PostgreSQL FTS (`tsvector`, `ts_headline`) and trigram operators
- Post-load DDL and generated columns in `schema-post-load/`

## Verification

1) Set environment
- `export AUSLEGALSEARCH_DB_BACKEND=oracle`
- `export ORACLE_SQLALCHEMY_URL="oracle+oracledb://user:pass@myadb_high"`
  - or set `ORACLE_DB_USER` / `ORACLE_DB_PASSWORD` / `ORACLE_DB_DSN` (and `ORACLE_WALLET_LOCATION` if using wallet)

2) Ping and bootstrap
- The FastAPI/Gradio/Streamlit apps and ingestion workers will:
  - ping DB (`SELECT 1 FROM dual`)
  - create core tables automatically (if `AUSLEGALSEARCH_AUTO_DDL=1`)

3) Test minimal flows
- Run ingestion for a small folder to populate data
- Exercise `/search/vector`, `/search/bm25`, `/search/hybrid`, `/search/fts`
- Use Streamlit/Gradio UIs or FastAPI endpoints

## Performance Notes

- This backend ranks vectors SQL-side with `vector_distance` over native `VECTOR` columns.
- For larger datasets:
  - Create HNSW/IVF vector indexes via `DBMS_VECTOR.CREATE_INDEX` (env flags included above)
  - Use selective filters (e.g., equality/ranges over metadata keys) before vector ORDER BY to reduce candidates
  - Consider Oracle Text for advanced full-text features if required later

## Troubleshooting

- `oracledb` package missing: `pip install oracledb`
- Wallet/TNS issues: verify `ORACLE_WALLET_LOCATION` and `TNS_ADMIN`; test with SQL*Plus or SQL Developer
- Permissions: ensure the schema user can create tables and run queries
- Legacy tables (missing identity PK): the bootstrap attempts an identity alter, then a sequence+trigger fallback

## Rollback / Staying on Postgres

- Simply unset or set `AUSLEGALSEARCH_DB_BACKEND=postgres`
- Postgres codepaths and performance features (pgvector, FTS, trigram) remain unchanged
