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
    from db.store_oracle import *  # noqa: F401,F403
    BACKEND = "oracle"
else:
    from db.store_postgres import *  # noqa: F401,F403
    BACKEND = "postgres"
