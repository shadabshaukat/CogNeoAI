"""
Oracle SQLAlchemy engine/session for AUSLegalSearch v3.

- Builds an SQLAlchemy engine for Oracle 23ai (Autonomous DB compatible) using python-oracledb.
- Mirrors pool/timeouts config style used by Postgres connector.
- Exposes: engine, SessionLocal, JSONType, UUIDType (String proxy), Vector (JSON-backed fallback).

Notes:
- For full Oracle VECTOR type support in pure SQL, you'd typically use Oracle's VECTOR(…) column type and SQL functions.
  SQLAlchemy lacks a first-class Oracle VECTOR type today, so this connector provides a JSON-backed Vector type
  (TypeDecorator) as a minimal compatibility layer. Vector search in db/store_oracle.py computes distances in Python.
- This is intended as a functional baseline to enable Oracle backend without changing app imports;
  performance/tight SQL integration can be incrementally added (e.g., using Oracle VECTOR and DOMAIN INDEX).

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
from sqlalchemy.types import TypeDecorator, Text
from sqlalchemy.dialects.oracle import JSON as OracleJSON
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
JSONType = OracleJSON
UUIDType = String  # UUIDs stored as VARCHAR2(36) in Oracle backend

class Vector(TypeDecorator):
    """
    Fallback Vector type for Oracle backend: stores Python list[float] as JSON text (CLOB).
    This enables minimal parity with the Postgres Vector column, at the cost of SQL-only vector ops.
    Vector distance computation is performed in Python in db/store_oracle.py.
    """
    impl = Text
    cache_ok = True

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        try:
            # Accept list-like, numpy arrays, etc.
            return json.dumps([float(x) for x in list(value)])
        except Exception:
            # As last resort, stringify
            return json.dumps([])

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        try:
            arr = json.loads(value)
            # leave as Python list[float]
            return arr
        except Exception:
            return None
