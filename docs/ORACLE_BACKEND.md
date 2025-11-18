# Oracle 23ai Backend for AUSLegalSearch v3

This document explains how to run AUSLegalSearch with Oracle Autonomous Database 23ai as the storage backend, while preserving the existing Postgres-first core logic.

Overview
- Backend switch is controlled by environment variable AUSLEGALSEARCH_DB_BACKEND.
- Default remains Postgres. No behavior changes for existing setups.
- Oracle backend uses python-oracledb via SQLAlchemy and stores vectors as JSON (CLOB). Vector ranking is performed in Python for baseline functionality. Full Oracle VECTOR type + Oracle Text can be added later.

Switching Backends
- Postgres (default)
  - AUSLEGALSEARCH_DB_BACKEND=postgres
- Oracle
  - AUSLEGALSEARCH_DB_BACKEND=oracle

Environment Variables

Option A: Single DSN URL (recommended when possible)
- ORACLE_SQLALCHEMY_URL=oracle+oracledb://user:password@myadb_high

Option B: Individual fields
- ORACLE_DB_USER=your_db_user
- ORACLE_DB_PASSWORD=your_db_password
- ORACLE_DB_DSN=myadb_high            # TNS alias or EZConnect DSN
- ORACLE_WALLET_LOCATION=/path/to/wallet/dir  # Optional; sets TNS_ADMIN for Autonomous DB

Pool/Timeout tuning (shared with Postgres connector)
- AUSLEGALSEARCH_DB_POOL_SIZE=10
- AUSLEGALSEARCH_DB_MAX_OVERFLOW=20
- AUSLEGALSEARCH_DB_POOL_RECYCLE=1800
- AUSLEGALSEARCH_DB_POOL_TIMEOUT=30

Vector scan limit (Oracle only)
- AUSLEGALSEARCH_ORA_VECTOR_SCAN_LIMIT=5000  # limit rows scanned when ranking vectors in Python

File Changes in this branch
- db/store.py: Dispatcher that exports the same symbols as before, selecting backend by AUSLEGALSEARCH_DB_BACKEND
- db/store_postgres.py: Previous Postgres models and helpers (pgvector + FTS)
- db/store_oracle.py: Oracle models and helpers (JSON vectors + LIKE/py cosine)
- db/connector.py: Dispatcher for connectors
- db/connector_postgres.py: Previous Postgres connector
- db/connector_oracle.py: Oracle connector (python-oracledb via SQLAlchemy)

What works on Oracle (baseline)
- Schema creation for core tables (users, documents, embeddings, sessions, etc.)
- Ingestion (documents + embeddings) using the same code paths
- Vector search: fetch N rows and rank with Python cosine distance
- BM25-like search: case-insensitive LIKE over documents.content
- Hybrid search: blends vector and LIKE-based results
- FTS endpoint: LIKE-based fallback over documents and metadata JSON text

Postgres-only functionality (not in Oracle baseline)
- pgvector-based vector operators/indexes (<=>, IVFFLAT/HNSW)
- PostgreSQL FTS (tsvector/ts_headline) and trigram operators
- Post-load DDL and generated columns in schema-post-load/

Verification

1) Set environment
- export AUSLEGALSEARCH_DB_BACKEND=oracle
- export ORACLE_SQLALCHEMY_URL="oracle+oracledb://user:pass@myadb_high"
  or set ORACLE_DB_USER / ORACLE_DB_PASSWORD / ORACLE_DB_DSN (and ORACLE_WALLET_LOCATION if using wallet)

2) Ping and bootstrap
- The FastAPI/Gradio/Streamlit apps and ingestion workers will:
  - ping DB (SELECT 1 FROM dual)
  - create core tables automatically (if AUSLEGALSEARCH_AUTO_DDL=1)

3) Test minimal flows
- Run ingestion for a small folder to populate data
- Exercise /search/vector, /search/bm25, /search/hybrid, /search/fts
- Use Streamlit/Gradio UIs or FastAPI endpoints

Performance Notes
- The Oracle baseline ranks vectors client-side (Python). This is suitable for demos and smaller datasets.
- For production-scale vector search, plan a follow-up:
  - Use Oracle VECTOR column type and domain indexes
  - Replace client-side cosine with SQL-side vector operations
  - Add Oracle Text for full-text features comparable to PostgreSQL FTS

Troubleshooting
- oracledb package missing: pip install oracledb
- Wallet/TNS issues: verify ORACLE_WALLET_LOCATION and TNS_ADMIN; check connectivity using sqlplus or SQL Developer
- Permissions: ensure the schema user can create tables and run queries
- Large scans: reduce AUSLEGALSEARCH_ORA_VECTOR_SCAN_LIMIT, or migrate to Oracle VECTOR type and SQL ranking

Rollback / Staying on Postgres
- Simply unset or set AUSLEGALSEARCH_DB_BACKEND=postgres
- Postgres codepaths and performance features (pgvector, FTS, trigram) remain unchanged
