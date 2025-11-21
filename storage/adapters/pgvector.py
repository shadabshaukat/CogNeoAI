from __future__ import annotations
from typing import Any, Dict, List
from storage.interfaces import VectorSearchAdapter
from db import store_postgres as store

class PgVectorAdapter(VectorSearchAdapter):
    name = "pgvector"

    def create_all_tables(self) -> None:
        store.create_all_tables()

    def search_vector(self, query_vec, top_k: int = 5) -> List[Dict[str, Any]]:
        return store.search_vector(query_vec, top_k=top_k)

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return store.search_bm25(query, top_k=top_k)

    def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        return store.search_hybrid(query, top_k=top_k, alpha=alpha)

    def search_fts(self, query: str, top_k: int = 10, mode: str = "both") -> List[Dict[str, Any]]:
        return store.search_fts(query, top_k=top_k, mode=mode)
