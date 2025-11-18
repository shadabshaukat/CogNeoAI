"""
DB store dispatcher for AUSLegalSearch v3.

Purpose:
- Preserve existing import surface (from db.store import ...).
- Selects the concrete backend at import-time based on environment:
    AUSLEGALSEARCH_DB_BACKEND=postgres | oracle
- Default is 'postgres' to maintain current behavior.

Backends:
- Postgres (pgvector/FTS): db.store_postgres
- Oracle 23ai (baseline, JSON vectors, LIKE search): db.store_oracle
"""

import os as _os

_BACKEND = (_os.environ.get("AUSLEGALSEARCH_DB_BACKEND", "postgres") or "postgres").lower()

if _BACKEND in ("oracle", "ora", "oracle23ai"):
    from db.store_oracle import *  # noqa: F401,F403
    BACKEND = "oracle"
else:
    from db.store_postgres import *  # noqa: F401,F403
    BACKEND = "postgres"
