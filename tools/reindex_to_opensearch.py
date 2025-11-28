from __future__ import annotations
"""
Backfill existing chunks from the database (Postgres/Oracle) into OpenSearch serving index.

Usage:
  # Ensure the API/adapter would resolve to OpenSearch (for consistency with config):
  export COGNEO_VECTOR_BACKEND=opensearch
  export OPENSEARCH_HOST=http://localhost:9200
  export OPENSEARCH_INDEX=cogneo_chunks

  # Then run:
  python tools/reindex_to_opensearch.py

Notes:
- Reads Document + Embedding rows via db.store SessionLocal
- Converts embedding vectors to plain python lists (works for pgvector and Oracle VECTOR(CLOB) text)
- Uses storage.registry.get_adapter().index_chunks() to bulk index to OpenSearch
"""

import ast
import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path when running as a script (python tools/reindex_to_opensearch.py)
_sys_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

from storage.adapters.opensearch import OpenSearchAdapter
from db.store import SessionLocal

# Import ORM models via store facade (works for both postgres/oracle stores)
from db.store import Document, Embedding  # type: ignore


def _to_float_list(vec: Any) -> List[float]:
    """
    Convert various vector representations to python List[float].
    Supports:
      - numpy arrays (via .tolist())
      - plain lists/tuples
      - string like "[1.23, 4.56, ...]"
    """
    if vec is None:
        return []
    try:
        if hasattr(vec, "tolist"):
            return [float(x) for x in vec.tolist()]
    except Exception:
        pass
    if isinstance(vec, (list, tuple)):
        try:
            return [float(x) for x in vec]
        except Exception:
            return []
    if isinstance(vec, (bytes, bytearray)):
        try:
            vec = vec.decode("utf-8", "ignore")
        except Exception:
            vec = str(vec)
    if isinstance(vec, str):
        txt = vec.strip()
        # Ensure it's list-like text
        try:
            if txt.startswith("[") and txt.endswith("]"):
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, (list, tuple)):
                    return [float(x) for x in parsed]
        except Exception:
            return []
    # Fallback: unknown type
    return []


def main(batch_size: int = 500) -> None:
    # Instantiate OpenSearch adapter directly so the serving backend can remain Postgres/Oracle.
    adapter = OpenSearchAdapter()
    try:
        adapter.create_all_tables()
    except Exception as e:
        print(f"[WARN] OpenSearch index ensure failed: {e}")

    total = 0
    with SessionLocal() as s:
        # Stream rows, ordered to group by document so source/format remain correct per batch
        q = (
            s.query(Embedding, Document)
             .join(Document, Embedding.doc_id == Document.id)
             .order_by(Embedding.doc_id, Embedding.chunk_index)
        )

        cur_chunks: List[Dict[str, Any]] = []
        cur_vecs: List[List[float]] = []
        cur_src: str = ""
        cur_fmt: str = "txt"
        cur_doc_id = None

        def _flush():
            nonlocal total, cur_chunks, cur_vecs, cur_src, cur_fmt, cur_doc_id
            if cur_chunks:
                try:
                    adapter.index_chunks(cur_chunks, cur_vecs, cur_src or "unknown", cur_fmt or "txt")
                    total += len(cur_chunks)
                except Exception as e:
                    print(f"[WARN] Indexing batch failed: {e}")
                finally:
                    cur_chunks, cur_vecs = [], []
            cur_doc_id = None

        for emb, doc in q.yield_per(batch_size):
            # Start new group when doc changes or batch is full
            if cur_doc_id is None:
                cur_doc_id = emb.doc_id
                cur_src = doc.source
                cur_fmt = doc.format if getattr(doc, "format", None) else "txt"
            elif emb.doc_id != cur_doc_id or len(cur_chunks) >= batch_size:
                _flush()
                cur_doc_id = emb.doc_id
                cur_src = doc.source
                cur_fmt = doc.format if getattr(doc, "format", None) else "txt"

            # Build chunk document including identifiers for parity
            cur_chunks.append({
                "doc_id": emb.doc_id,
                "chunk_index": emb.chunk_index,
                "text": doc.content,
                "chunk_metadata": emb.chunk_metadata if isinstance(emb.chunk_metadata, dict) else None,
            })
            cur_vecs.append(_to_float_list(emb.vector))

        # flush remaining group
        _flush()

    print(f"Indexed {total} chunks to OpenSearch.")


if __name__ == "__main__":
    try:
        bs = int(os.environ.get("REINDEX_BATCH_SIZE", "500"))
    except Exception:
        bs = 500
    main(batch_size=bs)
