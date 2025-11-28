from __future__ import annotations
"""
Backfill existing chunks from the database (Postgres/Oracle) into OpenSearch serving index.

Usage:
  # Ensure the API/adapter would resolve to OpenSearch (for consistency with config):
  export COGNEO_VECTOR_BACKEND=opensearch
  export OPENSEARCH_HOST=http(s)://host:9200
  export OPENSEARCH_INDEX=cogneo_chunks_v2

  # Optional high-throughput settings (see .env docs):
  # OPENSEARCH_NUMBER_OF_SHARDS=5
  # OPENSEARCH_NUMBER_OF_REPLICAS=1
  # OPENSEARCH_FORCE_RECREATE=1
  # OPENSEARCH_ENFORCE_SHARDS=1
  # OPENSEARCH_BULK_CHUNK_SIZE=600
  # OPENSEARCH_BULK_MAX_BYTES=104857600
  # OPENSEARCH_BULK_CONCURRENCY=4
  # OPENSEARCH_BULK_QUEUE_SIZE=8
  # OPENSEARCH_CONCURRENCY_OVERSUB=1
  # OPENSEARCH_TIMEOUT=120
  # OPENSEARCH_MAX_RETRIES=8
  # OPENSEARCH_HTTP_COMPRESS=1
  # OPENSEARCH_KNN_ENGINE=lucene
  # OPENSEARCH_KNN_SPACE=cosinesimil
  # OPENSEARCH_KNN_METHOD=hnsw
  # REINDEX_BATCH_SIZE=2000
  # REINDEX_YIELD_PER=3000
  # OPENSEARCH_TUNE_INDEX=1   (temporarily set refresh_interval=-1 and replicas=0, restore after reindex)

  # Then run:
  python tools/reindex_to_opensearch.py

Notes:
- Reads Document + Embedding rows via db.store SessionLocal
- Converts embedding vectors to python lists (works for pgvector and Oracle VECTOR text)
- Uses storage.adapters.opensearch.OpenSearchAdapter.index_chunks() to bulk index to OpenSearch
"""

import ast
import json
import os
import sys
from typing import Any, Dict, List, Tuple, Optional

# Ensure project root is on sys.path when running as a script (python tools/reindex_to_opensearch.py)
_sys_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

# Auto-load .env so OPENSEARCH_* and REINDEX_* are respected
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
        try:
            if txt.startswith("[") and txt.endswith("]"):
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, (list, tuple)):
                    return [float(x) for x in parsed]
        except Exception:
            return []
    return []


def _truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on", "y")


def _get_index_settings(client, index: str) -> Dict[str, Any]:
    try:
        s = client.indices.get_settings(index=index)
        idx = list(s.keys())[0] if isinstance(s, dict) and s else index
        return (s.get(idx, {}).get("settings", {}) or {})
    except Exception:
        return {}


def _get_index_mapping(client, index: str) -> Dict[str, Any]:
    try:
        m = client.indices.get_mapping(index=index)
        return (m.get(index, {}).get("mappings", {}) or {})
    except Exception:
        return {}


def _put_index_settings(client, index: str, settings: Dict[str, Any]) -> None:
    # settings should be {"index": {...}}
    try:
        client.indices.put_settings(index=index, body=settings)
    except Exception as e:
        print(f"[reindex] WARN: put_settings failed: {e}")


def _tune_index_for_bulk(client, index: str) -> Dict[str, Any]:
    """
    If OPENSEARCH_TUNE_INDEX=1: set refresh_interval=-1 and number_of_replicas=0 temporarily.
    Returns a dict with original values so they can be restored.
    """
    restore: Dict[str, Any] = {}
    try:
        if not _truthy(os.environ.get("OPENSEARCH_TUNE_INDEX")):
            return restore
        cur = _get_index_settings(client, index)
        idx_settings = cur.get("index", {}) if isinstance(cur, dict) else {}
        orig_refresh = idx_settings.get("refresh_interval")
        orig_repl = idx_settings.get("number_of_replicas")
        restore_vals: Dict[str, Any] = {}
        if orig_refresh is not None:
            restore_vals["refresh_interval"] = orig_refresh
        if orig_repl is not None:
            restore_vals["number_of_replicas"] = orig_repl
        if restore_vals:
            restore["index"] = restore_vals
        # Apply tuning
        print(f"[reindex] Tuning index '{index}' for bulk: refresh_interval=-1, number_of_replicas=0 (will restore after)")
        _put_index_settings(client, index, {"index": {"refresh_interval": "-1", "number_of_replicas": "0"}})
    except Exception as e:
        print(f"[reindex] WARN: index tuning (pre) failed: {e}")
    return restore


def _restore_index_settings(client, index: str, restore: Dict[str, Any]) -> None:
    try:
        if not restore:
            return
        print(f"[reindex] Restoring index '{index}' settings: {restore}")
        _put_index_settings(client, index, restore)
    except Exception as e:
        print(f"[reindex] WARN: index tuning (post-restore) failed: {e}")


def main(batch_size: int = 500) -> None:
    # Load dynamic batch/streaming controls
    try:
        batch_size_env = int(os.environ.get("REINDEX_BATCH_SIZE", str(batch_size)))
        if batch_size_env > 0:
            batch_size = batch_size_env
    except Exception:
        pass
    try:
        yield_per = int(os.environ.get("REINDEX_YIELD_PER", str(max(batch_size, 3000))))
    except Exception:
        yield_per = max(batch_size, 3000)

    # Instantiate OpenSearch adapter directly so the serving backend can remain Postgres/Oracle.
    adapter = OpenSearchAdapter()
    # Ensure index exists (adapter handles 3.x compatibility, flattened fallback, shard enforcement, etc.)
    try:
        adapter.create_all_tables()
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
        print(f"[ERROR] OpenSearch index ensure failed, aborting reindex to avoid 1-shard auto-create: {e}")
        sys.exit(1)

    # Verify index exists and shard count before proceeding to avoid slow single-shard scenarios
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

    # Optional pre-bulk tuning and planned restore
    restore_settings: Dict[str, Any] = {}
    try:
        restore_settings = _tune_index_for_bulk(adapter.client, adapter.index)
    except Exception:
        restore_settings = {}

    total = 0
    try:
        with SessionLocal() as s:
            # Stream rows, allow grouping across multiple documents to fill large batches
            q = (
                s.query(Embedding, Document)
                 .join(Document, Embedding.doc_id == Document.id)
                 .order_by(Embedding.doc_id, Embedding.chunk_index)
            )

            # Optional partitioning for parallel runs
            try:
                _parts = int(os.environ.get("REINDEX_PARTITIONS", "0"))
                _pid = int(os.environ.get("REINDEX_PARTITION_ID", "0"))
            except Exception:
                _parts, _pid = 0, 0
            if _parts and _parts > 1:
                q = q.filter((Embedding.doc_id % _parts) == _pid)

            # Apply DB streaming chunk size
            q = q.yield_per(yield_per)

            cur_chunks: List[Dict[str, Any]] = []
            cur_vecs: List[List[float]] = []

            def _flush():
                nonlocal total, cur_chunks, cur_vecs
                if cur_chunks:
                    try:
                        sz = len(cur_chunks)
                        adapter.index_chunks(cur_chunks, cur_vecs, cur_chunks[0].get("source", "unknown"), cur_chunks[0].get("format", "txt"))
                        total += sz
                        print(f"[reindex] flushed batch size={sz} total_indexed={total}")
                    except Exception as e:
                        print(f"[WARN] Indexing batch failed: {e}")
                    finally:
                        cur_chunks, cur_vecs = [], []

            for emb, doc in q:
                # Build per-chunk document with source/format present to allow cross-document bulks
                cm = emb.chunk_metadata
                if isinstance(cm, str):
                    try:
                        cm = json.loads(cm)
                    except Exception:
                        cm = {}
                elif cm is None:
                    cm = {}
                elif not isinstance(cm, dict):
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
                    "source": doc.source,
                    "format": (doc.format if getattr(doc, "format", None) else "txt"),
                })
                cur_vecs.append(_to_float_list(emb.vector))

                if len(cur_chunks) >= batch_size:
                    _flush()

            # flush remaining
            _flush()
    finally:
        # Post restore index settings (replicas, refresh)
        try:
            _restore_index_settings(adapter.client, adapter.index, restore_settings)
        except Exception:
            pass

    print(f"Indexed {total} chunks to OpenSearch.")


if __name__ == "__main__":
    try:
        bs = int(os.environ.get("REINDEX_BATCH_SIZE", "2000"))
    except Exception:
        bs = 2000
    main(batch_size=bs)
