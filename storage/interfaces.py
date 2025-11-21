from __future__ import annotations
from typing import Any, Dict, List

class VectorSearchAdapter:
    """Abstract interface for vector/hybrid/fts search + schema bootstrap."""
    name: str = "abstract"

    # Schema/bootstrap (no-op for engines without DDL)
    def create_all_tables(self) -> None:
        raise NotImplementedError

    # Search interfaces
    def search_vector(self, query_vec, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search_fts(self, query: str, top_k: int = 10, mode: str = "both") -> List[Dict[str, Any]]:
        raise NotImplementedError

    def index_chunks(self, chunks, vectors, source_path: str, fmt: str) -> int:
        """
        Optional bulk indexing hook for engines that can act as a serving vector store (e.g., OpenSearch).
        Implementations should align chunks with vectors by position and return number of indexed items.
        """
        raise NotImplementedError
