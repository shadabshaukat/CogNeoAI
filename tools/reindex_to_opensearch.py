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
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path when running as a script (python tools/reindex_to_opensearch.py)
_sys_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

# Auto-load .env so OPENSEARCH_NUMBER_OF_SHARDS/REPLICAS and other settings are respected
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

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
        # Print effective shard/replica settings for visibility (requires appropriate privileges)
        if os.environ.get("OPENSEARCH_DEBUG", "0") == "1":
            try:
                s = adapter.client.indices.get_settings(index=adapter.index)
                idx = list(s.keys())[0] if isinstance(s, dict) and s else adapter.index
                settings = s.get(idx, {}).get("settings", {}).get("index", {}) or {}
                nsh = settings.get("number_of_shards")
                nrepl = settings.get("number_of_replicas")
                print(f"[reindex] Index '{adapter.index}' settings: number_of_shards={nsh}, number_of_replicas={nrepl}")
            except Exception as es:
                print(f"[reindex] Note: could not read index settings: {es}")
    except Exception as e:
        # Abort rather than continuing to bulk index, otherwise auto_create_index can silently create a 1-shard index,
        # leading to very slow ingestion and parity issues.
        print(f"[ERROR] OpenSearch index ensure failed, aborting reindex to avoid 1-shard auto-create: {e}")
        sys.exit(1)

    # Verify index exists and shard count before proceeding to avoid slow single-shard auto-create scenarios
    try:
        if not adapter.client.indices.exists(index=adapter.index):
            print(f"[reindex] ERROR: index '{adapter.index}' does not exist after ensure step. Aborting.")
            sys.exit(1)
        s = adapter.client.indices.get_settings(index=adapter.index)
        idx = list(s.keys())[0] if isinstance(s, dict) and s else adapter.index
        settings = s.get(idx, {}).get("settings", {}).get("index", {}) or {}
        effective_shards = int(str(settings.get("number_of_shards") or "1"))
        want_shards = int(os.environ.get("OPENSEARCH_NUMBER_OF_SHARDS", "0") or "0")
        if want_shards and effective_shards != want_shards:
            print(f"[reindex] ERROR: effective shards={effective_shards}, wanted={want_shards}. Aborting to avoid slow ingestion.")
            print("[reindex] Hint: set OPENSEARCH_FORCE_RECREATE=1 and rerun, or choose a new OPENSEARCH_INDEX.")
            sys.exit(1)
    except Exception as ei:
        print(f"[reindex] WARN: could not validate index existence/shards: {ei}")

    total = 0
    with SessionLocal() as s:
        # Stream rows, ordered to group by document so source/format remain correct per batch
        q = (
            s.query(Embedding, Document)
             .join(Document, Embedding.doc_id == Document.id)
        )

        # Optional partitioning for parallel runs: process only rows where
        # doc_id % REINDEX_PARTITIONS == REINDEX_PARTITION_ID
        # Example:
        #   REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=0 python3 -m tools.reindex_to_opensearch
        #   REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=1 python3 -m tools.reindex_to_opensearch
        #   ...
        # SQLAlchemy will translate modulo appropriately for Postgres/Oracle dialects.
        try:
            _parts = int(os.environ.get("REINDEX_PARTITIONS", "0"))
            _pid = int(os.environ.get("REINDEX_PARTITION_ID", "0"))
        except Exception:
            _parts, _pid = 0, 0
        if _parts and _parts > 1:
            q = q.filter((Embedding.doc_id % _parts) == _pid)

        q = q.order_by(Embedding.doc_id, Embedding.chunk_index)

        cur_chunks: List[Dict[str, Any]] = []
        cur_vecs: List[List[float]] = []
        cur_src: str = ""
        cur_fmt: str = "txt"
        cur_doc_id = None

        def _flush():
            nonlocal total, cur_chunks, cur_vecs, cur_src, cur_fmt, cur_doc_id
            if cur_chunks:
                try:
                    sz = len(cur_chunks)
                    adapter.index_chunks(cur_chunks, cur_vecs, cur_src or "unknown", cur_fmt or "txt")
                    total += sz
                    print(f"[reindex] flushed batch size={sz} total_indexed={total}")
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
            cm = emb.chunk_metadata
            if isinstance(cm, str):
                try:
                    cm = json.loads(cm)
                except Exception:
                    cm = {}
            elif cm is None:
                cm = {}
            elif not isinstance(cm, dict):
                # Fallback: ensure dict to comply with OpenSearch 'flattened' mapping
                try:
                    cm = json.loads(str(cm))
                    if not isinstance(cm, dict):
                        cm = {}
                except Exception:
                    cm = {}
            cur_chunks.append({
                "doc_id": emb.doc_id,
                "chunk_index": emb.chunk_index,
                "text": doc.content,
                "chunk_metadata": cm,
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
