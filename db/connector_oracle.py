"""
Oracle SQLAlchemy engine/session for AUSLegalSearch v3.

- Builds an SQLAlchemy engine for Oracle 26ai/23ai (Autonomous DB compatible) using python-oracledb.
- Mirrors pool/timeouts config style used by Postgres connector.
- Exposes: engine, SessionLocal, DB_URL, JSONType, UUIDType (String proxy), Vector (UserDefinedType emitting Oracle VECTOR).

Notes:
- Requires Oracle AI Vector Search with COMPATIBLE >= 23.4.0 to use the native VECTOR data type and vector_distance().
- The custom Vector UserDefinedType emits "VECTOR(dim, FLOAT32, DENSE)" and binds dense literals like "[1.0,2.0,...]".
- Use DBMS_VECTOR to create HNSW/IVF indexes and enable APPROX vector_distance() for index usage.

Env variables (either ORACLE_SQLALCHEMY_URL or the individual fields must be provided):
- ORACLE_SQLALCHEMY_URL                # e.g. oracle+oracledb://user:pass@myadb_high
- ORACLE_DB_USER
- ORACLE_DB_PASSWORD
- ORACLE_DB_DSN                        # e.g. myadb_high (TNS name) or host/service_name
- ORACLE_WALLET_LOCATION               # optional; sets TNS_ADMIN for Autonomous DB wallet

Pool/timeouts (optional):
- AUSLEGALSEARCH_DB_POOL_SIZE          # default 10
- AUSLEGALSEARCH_DB_MAX_OVERFLOW       # default 20
- AUSLEGALSEARCH_DB_POOL_RECYCLE       # default 1800s
- AUSLEGALSEARCH_DB_POOL_TIMEOUT       # default 30s
"""

import os
import json
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import UserDefinedType, Text
try:
    from sqlalchemy import JSON as _SAJSON  # SQLAlchemy generic JSON (may map to CLOB on Oracle)
except Exception:
    _SAJSON = None
from sqlalchemy import String

# Minimal .env loader (same behavior as Postgres connector)
def _load_dotenv_file():
    try:
        here = os.path.abspath(os.path.dirname(__file__))
        candidates = [
            os.path.abspath(os.path.join(here, "..", ".env")),   # repo root
            os.path.abspath(os.path.join(os.getcwd(), ".env")),  # current working dir
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
    except Exception:
        pass

_load_dotenv_file()

# Wallet (Autonomous DB)
WALLET = os.environ.get("ORACLE_WALLET_LOCATION")
if WALLET:
    os.environ["TNS_ADMIN"] = WALLET

# Build SQLAlchemy URL
ORACLE_SQLALCHEMY_URL = os.environ.get("ORACLE_SQLALCHEMY_URL")
if not ORACLE_SQLALCHEMY_URL:
    ORA_USER = os.environ.get("ORACLE_DB_USER")
    ORA_PASS = os.environ.get("ORACLE_DB_PASSWORD")
    ORA_DSN = os.environ.get("ORACLE_DB_DSN")
    required = {"ORACLE_DB_USER": ORA_USER, "ORACLE_DB_PASSWORD": ORA_PASS, "ORACLE_DB_DSN": ORA_DSN}
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing required Oracle env vars: " + ", ".join(missing) +
            ". Provide ORACLE_SQLALCHEMY_URL or ORACLE_DB_USER/ORACLE_DB_PASSWORD/ORACLE_DB_DSN."
        )
    # Percent-encode creds for URL safety
    user_q = quote_plus(ORA_USER)
    pwd_q = quote_plus(ORA_PASS)
    # DSN is taken as-is (TNS alias like 'myadb_high' or EZConnect 'host:port/?service_name=...').
    ORACLE_SQLALCHEMY_URL = f"oracle+oracledb://{user_q}:{pwd_q}@{ORA_DSN}"

# Pool configuration (mirrors Postgres style)
POOL_SIZE = int(os.environ.get("AUSLEGALSEARCH_DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.environ.get("AUSLEGALSEARCH_DB_MAX_OVERFLOW", "20"))
POOL_RECYCLE = int(os.environ.get("AUSLEGALSEARCH_DB_POOL_RECYCLE", "1800"))  # seconds
POOL_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_DB_POOL_TIMEOUT", "30"))    # seconds

# Connect args for oracledb via SQLAlchemy are limited compared to psycopg2;
# keep minimal and rely on database/sqlnet configs for timeouts/keepalives.
engine = create_engine(
    ORACLE_SQLALCHEMY_URL,
    pool_pre_ping=True,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_recycle=POOL_RECYCLE,
    pool_timeout=POOL_TIMEOUT,
    # No generic connect_args widely applicable here; use tnsnames/sqlnet.ora for advanced config.
)
SessionLocal = sessionmaker(bind=engine)
DB_URL = ORACLE_SQLALCHEMY_URL

# Type aliases to match Postgres store expectations
JSONType = _SAJSON if _SAJSON is not None else Text  # Fallback to Text on dialects without JSON
UUIDType = String  # UUIDs stored as VARCHAR2(36) in Oracle backend

class Vector(UserDefinedType):
    """
    Oracle 26ai VECTOR column type for embeddings.

    Usage:
        Column(Vector(dim))  -> VECTOR(dim, FLOAT32, DENSE)

    - get_col_spec() emits VECTOR(dim, FLOAT32, DENSE) (or flexible '*' if dim is None)
    - bind_processor converts Python list/numpy array to textual dense literal: "[1.0,2.0,...]"
    - result_processor returns value unchanged (vector values are rarely selected in this app)
    """
    cache_ok = True

    def __init__(self, dim: int = None, fmt: str = "FLOAT32", storage: str = "DENSE"):
        self.dim = dim
        self.fmt = (fmt or "FLOAT32").upper()
        self.storage = (storage or "DENSE").upper()

    def get_col_spec(self, **kw):
        dim = "*" if not self.dim else str(int(self.dim))
        fmt = self.fmt if self.fmt in ("INT8", "FLOAT32", "FLOAT64", "BINARY", "*") else "FLOAT32"
        stor = self.storage if self.storage in ("DENSE", "SPARSE", "*") else "DENSE"
        return f"VECTOR({dim}, {fmt}, {stor})"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            # Accept list-like / numpy arrays and emit dense textual literal: [v0,v1,...]
            try:
                # Try numpy first without hard dependency
                if hasattr(value, "tolist"):
                    seq = value.tolist()
                else:
                    seq = list(value)
                return "[" + ",".join(str(float(x)) for x in seq) + "]"
            except Exception:
                # If already a string literal like "[...]" pass through
                if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                    return value
                # Fallback empty vector
                return "[]"
        return process

    def result_processor(self, dialect, coltype):
        # Leave as-is (the app rarely selects the vector column itself)
        return lambda val: val
