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
from typing import Any, Dict, List, Tuple

from storage.registry import get_adapter
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
    adapter = get_adapter()
    if getattr(adapter, "name", "") != "opensearch":
        print("ERROR: Set COGNEO_VECTOR_BACKEND=opensearch to run this tool.")
        return

    total = 0
    with SessionLocal() as s:
        # Stream rows in chunks
        q = s.query(Embedding, Document).join(Document, Embedding.doc_id == Document.id)
        cur_chunks: List[Dict[str, Any]] = []
        cur_vecs: List[List[float]] = []
        cur_src: str = ""
        cur_fmt: str = "txt"

        def _flush():
            nonlocal total, cur_chunks, cur_vecs, cur_src, cur_fmt
            if cur_chunks:
                try:
                    adapter.index_chunks(cur_chunks, cur_vecs, cur_src or "unknown", cur_fmt or "txt")
                    total += len(cur_chunks)
                except Exception as e:
                    print(f"[WARN] Indexing batch failed: {e}")
                finally:
                    cur_chunks, cur_vecs = [], []

        for emb, doc in q.yield_per(batch_size):
            # Build chunk document
            cur_chunks.append({
                "text": doc.content,
                "chunk_metadata": emb.chunk_metadata if isinstance(emb.chunk_metadata, dict) else None
            })
            cur_vecs.append(_to_float_list(emb.vector))
            cur_src = doc.source
            cur_fmt = doc.format if getattr(doc, "format", None) else "txt"

            if len(cur_chunks) >= batch_size:
                _flush()

        # flush remaining
        _flush()

    print(f"Indexed {total} chunks to OpenSearch.")


if __name__ == "__main__":
    main()
