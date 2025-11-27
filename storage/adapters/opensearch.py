from __future__ import annotations
from typing import Any, Dict, List
import os
from opensearchpy import OpenSearch, helpers
from storage.interfaces import VectorSearchAdapter


def _os_client() -> OpenSearch:
    host = os.environ.get("OPENSEARCH_HOST", "https://localhost:9200")
    user = os.environ.get("OPENSEARCH_USER")
    pwd = os.environ.get("OPENSEARCH_PASS")
    if user and pwd:
        return OpenSearch(hosts=[host], http_auth=(user, pwd), verify_certs=True)
    return OpenSearch(hosts=[host], verify_certs=True)

    """
    For OCI OpenSearch set verify_certs=True
    For all HTTPS enabled services set verify_certs=True
    For HTTP service set verify_certs=False
    """


def _index_name() -> str:
    return os.environ.get("OPENSEARCH_INDEX", "cogneo_chunks")


def _dim() -> int:
    try:
        return int(os.environ.get("COGNEO_EMBED_DIM", "768"))
    except Exception:
        return 768


class OpenSearchAdapter(VectorSearchAdapter):
    """
    OpenSearch backend adapter:
    - Ensures a KNN (HNSW cosine) index with correct dimension
    - Indexes chunks + vectors via bulk API
    - Supports vector/BM25/FTS searches
    """
    name = "opensearch"

    def __init__(self) -> None:
        self.client = _os_client()
        self.index = _index_name()
        self.dim = _dim()

    def create_all_tables(self) -> None:
        """
        Ensure the OpenSearch index exists with a knn_vector mapping.
        """
        try:
            if self.client.indices.exists(index=self.index):
                return
        except Exception:
            # For OS < 2.5 the API may differ; ignore existence check errors and attempt create
            pass

        body = {
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "format": {"type": "keyword"},
                    "chunk_metadata": {"type": "object", "enabled": True},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 200,
                                "m": 16
                            }
                        }
                    }
                }
            }
        }
        self.client.indices.create(index=self.index, body=body, ignore=400)

    def index_chunks(self, chunks: List[Dict[str, Any]], vectors, source_path: str, fmt: str) -> int:
        """
        Bulk index chunks into OpenSearch, aligned with provided vectors.
        """
        actions = []
        for i, ch in enumerate(chunks):
            vec = vectors[i].tolist() if hasattr(vectors[i], "tolist") else list(vectors[i])
            actions.append({
                "_op_type": "index",
                "_index": self.index,
                "_id": f"{source_path}#{i}",
                "text": ch.get("text", ""),
                "source": source_path,
                "format": fmt,
                "chunk_metadata": ch.get("chunk_metadata") or {},
                "vector": vec
            })
        if not actions:
            return 0
        helpers.bulk(self.client, actions, refresh=False)
        return len(actions)

    def search_vector(self, query_vec, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        KNN vector search using the configured vector field.
        """
        body = {
            "size": top_k,
            "knn": {
                "field": "vector",
                "query_vector": query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec),
                "k": top_k,
                "num_candidates": max(100, top_k * 10),
            },
            "_source": True
        }
        res = self.client.search(index=self.index, body=body)
        hits = []
        for h in res.get("hits", {}).get("hits", []):
            s = h.get("_source", {})
            hits.append({
                "doc_id": None,
                "chunk_index": None,
                "score": float(h.get("_score", 0.0)),
                "text": s.get("text"),
                "source": s.get("source"),
                "format": s.get("format"),
                "chunk_metadata": s.get("chunk_metadata"),
            })
        return hits

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        BM25 text search on the chunk text field.
        """
        body = {
            "size": top_k * 2,
            "query": {"match": {"text": {"query": query}}},
            "_source": True
        }
        res = self.client.search(index=self.index, body=body)
        out = []
        for h in res.get("hits", {}).get("hits", []):
            s = h.get("_source", {})
            out.append({
                "doc_id": None,
                "chunk_index": None,
                "score": float(h.get("_score", 0.0)),
                "text": s.get("text"),
                "source": s.get("source"),
                "format": s.get("format"),
                "chunk_metadata": s.get("chunk_metadata"),
            })
        return out[:top_k]

    def search_fts(self, query: str, top_k: int = 10, mode: str = "both") -> List[Dict[str, Any]]:
        """
        Full text search across chunk text and metadata fields.
        """
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "chunk_metadata.*"]
                }
            },
            "_source": True
        }
        res = self.client.search(index=self.index, body=body)
        out = []
        for h in res.get("hits", {}).get("hits", []):
            s = h.get("_source", {})
            out.append({
                "doc_id": None,
                "chunk_index": None,
                "source": s.get("source"),
                "content": s.get("text"),
                "text": s.get("text"),
                "chunk_metadata": s.get("chunk_metadata"),
                "snippet": None,
                "search_area": "both"
            })
        return out[:top_k]

    def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Minimal hybrid: use BM25 baseline.
        (Upstream fusion with vector KNN can be added later if needed.)
        """
        return self.search_bm25(query, top_k=top_k)
