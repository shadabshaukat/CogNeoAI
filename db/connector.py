"""
DB connector dispatcher for AUSLegalSearch v3.

Selects the concrete connector at import-time based on environment:
  AUSLEGALSEARCH_DB_BACKEND=postgres | oracle
Default is 'postgres' to preserve current behavior.

Re-exports a stable surface:
  - engine, SessionLocal, DB_URL
  - Vector, JSONB, UUIDType
  - ensure_pgvector()  (no-op on Oracle)
"""

import os as _os

_BACKEND = (_os.environ.get("AUSLEGALSEARCH_DB_BACKEND", "postgres") or "postgres").lower()

if _BACKEND in ("oracle", "ora", "oracle23ai"):
    # Oracle backend (python-oracledb via SQLAlchemy)
    from db.connector_oracle import (
        engine,
        SessionLocal,
        DB_URL,
        Vector,
        JSONType as _ORACLE_JSON,
        UUIDType as _ORACLE_UUID,
    )
    # Compatibility aliases for callers expecting Postgres names
    JSONB = _ORACLE_JSON
    UUIDType = _ORACLE_UUID

    def ensure_pgvector():
        # Not applicable on Oracle backend; keep API surface compatible.
        return None

    BACKEND = "oracle"
else:
    # Postgres backend (psycopg2 + pgvector)
    from db.connector_postgres import (
        engine,
        SessionLocal,
        DB_URL,
        Vector,
        JSONB,
        UUIDType,
        ensure_pgvector,
    )
    BACKEND = "postgres"
