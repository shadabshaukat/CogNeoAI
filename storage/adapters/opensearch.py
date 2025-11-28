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
                    "doc_id": {"type": "long"},
                    "chunk_index": {"type": "integer"},
                    "citation": {"type": "keyword"},
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
        Accepts optional doc_id and chunk_index inside each chunk dict to preserve parity with DB records.
        """
        actions = []
        for i, ch in enumerate(chunks):
            vec = vectors[i].tolist() if hasattr(vectors[i], "tolist") else list(vectors[i])
            doc_id = ch.get("doc_id")
            chunk_idx = ch.get("chunk_index")
            source_val = source_path
            fmt_val = fmt

            citation = None
            if doc_id is not None and chunk_idx is not None:
                try:
                    doc_id_val = int(doc_id)
                except Exception:
                    doc_id_val = None
                try:
                    chunk_idx_val = int(chunk_idx)
                except Exception:
                    chunk_idx_val = None
                if doc_id_val is not None and chunk_idx_val is not None:
                    _id = f"{doc_id_val}#{chunk_idx_val}"
                    citation = f"{source_val}#chunk{chunk_idx_val}"
                else:
                    _id = f"{source_val}#{i}"
            else:
                _id = f"{source_val}#{i}"

            doc = {
                "_op_type": "index",
                "_index": self.index,
                "_id": _id,
                "text": ch.get("text", ""),
                "source": source_val,
                "format": fmt_val,
                "chunk_metadata": ch.get("chunk_metadata") or {},
                "vector": vec
            }
            if doc_id is not None:
                try:
                    doc["doc_id"] = int(doc_id)
                except Exception:
                    pass
            if chunk_idx is not None:
                try:
                    doc["chunk_index"] = int(chunk_idx)
                except Exception:
                    pass
            if citation:
                doc["citation"] = citation

            actions.append(doc)

        if not actions:
            return 0
        helpers.bulk(self.client, actions, refresh=False)
        return len(actions)

    def search_vector(self, query_vec, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        KNN vector search using the configured vector field.
        Tries modern top-level 'knn' syntax first, falls back to legacy 'query':{'knn':{...}} syntax for older clusters.
        """
        qv = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
        # Modern syntax (some OS versions accept top-level 'knn')
        body_modern = {
            "size": top_k,
            "knn": {
                "field": "vector",
                "query_vector": qv,
                "k": top_k,
                "num_candidates": max(100, top_k * 10),
            },
            "_source": True
        }
        # Legacy syntax (OpenSearch 1.x/early 2.x) — variant A (field + query_vector)
        body_legacy = {
            "size": top_k,
            "query": {
                "knn": {
                    "field": "vector",
                    "query_vector": qv,
                    "k": top_k,
                    "num_candidates": max(100, top_k * 10),
                }
            },
            "_source": True
        }
        # Legacy syntax variant B (older knn plugin shape: index 'vector' field object)
        body_legacy_alt = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": qv,
                        "k": top_k
                    }
                }
            },
            "_source": True
        }
        try:
            res = self.client.search(index=self.index, body=body_modern)
        except Exception:
            # Fallback to legacy knn query structures
            try:
                res = self.client.search(index=self.index, body=body_legacy)
            except Exception:
                try:
                    res = self.client.search(index=self.index, body=body_legacy_alt)
                except Exception:
                    # As last resort, return empty hits to let callers degrade gracefully (e.g., hybrid -> BM25-only)
                    return []

        hits = []
        for h in res.get("hits", {}).get("hits", []):
            s = h.get("_source", {})
            doc_id = s.get("doc_id")
            chunk_idx = s.get("chunk_index")
            vector_score = float(h.get("_score", 0.0))
            hits.append({
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "score": vector_score,
                "vector_score": vector_score,
                "bm25_score": 0.0,
                "text": s.get("text"),
                "source": s.get("source"),
                "format": s.get("format"),
                "chunk_metadata": s.get("chunk_metadata"),
                "citation": s.get("citation") or (f"{s.get('source')}#chunk{chunk_idx}" if (s.get("source") and chunk_idx is not None) else None),
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
            doc_id = s.get("doc_id")
            chunk_idx = s.get("chunk_index")
            bm25 = float(h.get("_score", 0.0))
            out.append({
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "score": bm25,
                "vector_score": 0.0,
                "bm25_score": 1.0,  # presence signal (parity with Postgres hybrid)
                "text": s.get("text"),
                "source": s.get("source"),
                "format": s.get("format"),
                "chunk_metadata": s.get("chunk_metadata"),
                "citation": s.get("citation") or (f"{s.get('source')}#chunk{chunk_idx}" if (s.get("source") and chunk_idx is not None) else None),
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
                "doc_id": s.get("doc_id"),
                "chunk_index": s.get("chunk_index"),
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
        Hybrid fusion parity with Postgres:
        - Embed query for KNN search
        - Combine KNN and BM25 presence with alpha-weighted score
        - Provide doc_id, chunk_index, citation, vector_score, bm25_score, hybrid_score
        """
        try:
            from embedding.embedder import Embedder
            embedder = Embedder()
            query_vec = embedder.embed([query])[0]
        except Exception:
            # Fallback to BM25-only if embedding fails
            return self.search_bm25(query, top_k=top_k)

        k = max(20, top_k * 2)
        vector_hits = self.search_vector(query_vec, top_k=k)
        bm25_hits = self.search_bm25(query, top_k=k)

        all_hits: Dict[Any, Dict[str, Any]] = {}
        for h in vector_hits:
            key = (h.get("doc_id"), h.get("chunk_index"))
            all_hits[key] = {
                **h,
                "vector_score": h.get("vector_score", h.get("score", 0.0)),
                "bm25_score": 0.0,
                "hybrid_score": 0.0,
            }
        for h in bm25_hits:
            key = (h.get("doc_id"), h.get("chunk_index"))
            if key in all_hits:
                all_hits[key]["bm25_score"] = 1.0
            else:
                all_hits[key] = {
                    **h,
                    "vector_score": 0.0,
                    "bm25_score": 1.0,
                    "hybrid_score": 0.0,
                }

        scores = [v.get("vector_score", 0.0) for v in all_hits.values()]
        if scores:
            minv, maxv = min(scores), max(scores)
            for v in all_hits.values():
                if maxv != minv:
                    v["vector_score_norm"] = (v.get("vector_score", 0.0) - minv) / (maxv - minv)
                else:
                    v["vector_score_norm"] = 1.0
        else:
            for v in all_hits.values():
                v["vector_score_norm"] = 0.0

        for v in all_hits.values():
            v["hybrid_score"] = alpha * v["vector_score_norm"] + (1 - alpha) * v.get("bm25_score", 0.0)
            if not v.get("citation"):
                src = v.get("source")
                ci = v.get("chunk_index")
                v["citation"] = f"{src}#chunk{ci}" if src and ci is not None else None

        results = sorted(all_hits.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
        return results
