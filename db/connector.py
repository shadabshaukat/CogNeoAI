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

# Minimal .env loader (dependency-free) to support CLI scripts importing this module
def _load_dotenv_file():
    try:
        import os as __os
        here = __os.path.abspath(__os.path.dirname(__file__))
        candidates = [
            __os.path.abspath(__os.path.join(here, "..", ".env")),   # repo root
            __os.path.abspath(__os.path.join(__os.getcwd(), ".env")),  # current working dir
        ]
        for path in candidates:
            if __os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in __os.environ:
                            __os.environ[k] = v
                break
    except Exception:
        # Never fail on dotenv load
        pass

# Load .env before reading backend selector (helps when running ingest/* via python -m)
_load_dotenv_file()

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
