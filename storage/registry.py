from __future__ import annotations
import os
from typing import Optional
from storage.interfaces import VectorSearchAdapter

_singleton: Optional[VectorSearchAdapter] = None

def get_adapter() -> VectorSearchAdapter:
    global _singleton
    if _singleton is not None:
        return _singleton

    # COGNEO_VECTOR_BACKEND wins; fallback to COGNEO_DB_BACKEND; defaults to pgvector
    backend = (os.environ.get("COGNEO_VECTOR_BACKEND")
               or os.environ.get("COGNEO_DB_BACKEND")
               or "postgres").lower()

    if backend in ("oracle", "ora", "oracle26ai"):
        from storage.adapters.oracle26ai import Oracle26aiAdapter
        _singleton = Oracle26aiAdapter()
    elif backend in ("opensearch", "os", "opensearch-knn"):
        from storage.adapters.opensearch import OpenSearchAdapter
        _singleton = OpenSearchAdapter()
    else:
        from storage.adapters.pgvector import PgVectorAdapter
        _singleton = PgVectorAdapter()
    return _singleton
