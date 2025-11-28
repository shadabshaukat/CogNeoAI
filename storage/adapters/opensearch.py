from __future__ import annotations
from typing import Any, Dict, List
import os
from opensearchpy import OpenSearch, helpers
from storage.interfaces import VectorSearchAdapter


def _os_client() -> OpenSearch:
    host = (os.environ.get("OPENSEARCH_HOST", "https://localhost:9200") or "").rstrip("/")
    user = os.environ.get("OPENSEARCH_USER")
    pwd = os.environ.get("OPENSEARCH_PASS")
    verify = os.environ.get("OPENSEARCH_VERIFY_CERTS", "1") != "0"
    timeout = int(os.environ.get("OPENSEARCH_TIMEOUT", "20"))
    max_retries = int(os.environ.get("OPENSEARCH_MAX_RETRIES", "2"))

    kwargs = dict(
        hosts=[host],
        verify_certs=verify,
        timeout=timeout,
        max_retries=max_retries,
        retry_on_timeout=True,
    )
    if user and pwd:
        kwargs["http_auth"] = (user, pwd)
    return OpenSearch(**kwargs)


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
        self.timeout = int(os.environ.get("OPENSEARCH_TIMEOUT", "20"))
        self._debug = os.environ.get("OPENSEARCH_DEBUG", "0") == "1"
        self._primary_shards = None  # cached primary shard count for effective concurrency
        self._md_field_mode = "flattened"  # 'flattened' (preferred) or 'text' (fallback: chunk_metadata_text)
        if self._debug:
            host_env = (os.environ.get("OPENSEARCH_HOST", "") or "").rstrip("/")
            verify_env = os.environ.get("OPENSEARCH_VERIFY_CERTS", "1") != "0"
            print(f"[OpenSearchAdapter] host={host_env} index={self.index} timeout={self.timeout}s verify_certs={verify_env}")

    def _get_primary_shard_count(self) -> int:
        try:
            if self._primary_shards is not None:
                return int(self._primary_shards)
            s = self.client.indices.get_settings(index=self.index)
            idx = list(s.keys())[0] if isinstance(s, dict) and s else self.index
            settings = s.get(idx, {}).get("settings", {}).get("index", {}) or {}
            nsh = int(str(settings.get("number_of_shards")))
            self._primary_shards = nsh
            return nsh if nsh > 0 else 1
        except Exception:
            return 1

    def _cm_to_text(self, cm: Any) -> str:
        """
        Flatten chunk_metadata dict into a space-joined text suitable for 'chunk_metadata_text' field.
        Keeps values and small key:value pairs; safe for clusters without 'flattened' type support.
        """
        try:
            out: List[str] = []
            def _walk(v, k=None):
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        _walk(vv, kk)
                elif isinstance(v, list):
                    for it in v:
                        _walk(it, k)
                else:
                    try:
                        sval = str(v)
                    except Exception:
                        sval = ""
                    if sval:
                        if k:
                            out.append(f"{k}:{sval}")
                        else:
                            out.append(sval)
            if isinstance(cm, dict):
                _walk(cm)
            else:
                # attempt to parse string json
                if isinstance(cm, str):
                    import json as _json
                    try:
                        obj = _json.loads(cm)
                        if isinstance(obj, dict):
                            _walk(obj)
                    except Exception:
                        out.append(cm)
            return " ".join([s for s in out if s])[:32766]
        except Exception:
            return ""

    def create_all_tables(self) -> None:
        """
        Ensure the OpenSearch index exists with a knn_vector mapping.
        - Shard count in OpenSearch is immutable after index creation.
          If you need to change shards, either:
            * use a new OPENSEARCH_INDEX name, or
            * set OPENSEARCH_FORCE_RECREATE=1 to delete and recreate (DANGEROUS: drops existing index).
        """
        try:
            exists = self.client.indices.exists(index=self.index)
            if exists:
                force_recreate = str(os.environ.get("OPENSEARCH_FORCE_RECREATE", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
                if force_recreate:
                    if self._debug:
                        print(f"[OpenSearchAdapter] OPENSEARCH_FORCE_RECREATE=1 -> deleting existing index {self.index}")
                    try:
                        self.client.indices.delete(index=self.index, ignore=[404])
                    except Exception as e_del:
                        print(f"[OpenSearchAdapter] WARN: failed to delete index {self.index}: {e_del}")
                else:
                    # Log current settings for visibility and explain shard immutability
                    if self._debug:
                        try:
                            s = self.client.indices.get_settings(index=self.index)
                            idx = list(s.keys())[0] if isinstance(s, dict) and s else self.index
                            settings = s.get(idx, {}).get("settings", {}).get("index", {}) or {}
                            nsh = settings.get("number_of_shards")
                            nrepl = settings.get("number_of_replicas")
                            print(f"[OpenSearchAdapter] index '{self.index}' exists; number_of_shards={nsh}, number_of_replicas={nrepl}. Shards are immutable after creation. To change shards, set OPENSEARCH_FORCE_RECREATE=1 or use a new OPENSEARCH_INDEX.")
                        except Exception:
                            print(f"[OpenSearchAdapter] index '{self.index}' exists. Shards are immutable; use OPENSEARCH_FORCE_RECREATE=1 to recreate with new shard count.")
                    return
        except Exception:
            # For OS < 2.5 the API may differ; ignore existence check errors and attempt create
            pass

        # Diagnostics: list index templates that may affect shard/replica counts
        if self._debug:
            try:
                # Composable templates (v2)
                tmpl = self.client.transport.perform_request("GET", "/_index_template")
                matched = []
                if isinstance(tmpl, dict) and "index_templates" in tmpl:
                    import fnmatch
                    for t in tmpl.get("index_templates", []):
                        it = t.get("index_template") or {}
                        patterns = it.get("index_patterns") or it.get("pattern_list") or []
                        for pat in patterns:
                            if fnmatch.fnmatch(self.index, pat):
                                settings = (((it.get("template") or {}).get("settings") or {}).get("index") or {})
                                matched.append((t.get("name"), pat, settings.get("number_of_shards"), settings.get("number_of_replicas")))
                # Legacy templates (v1)
                try:
                    legacy = self.client.transport.perform_request("GET", "/_template")
                    if isinstance(legacy, dict):
                        import fnmatch
                        for name, spec in legacy.items():
                            patterns = spec.get("index_patterns") or spec.get("template") or [name]
                            if isinstance(patterns, str):
                                patterns = [patterns]
                            for pat in patterns:
                                if fnmatch.fnmatch(self.index, pat):
                                    settings = ((spec.get("settings") or {}).get("index") or {})
                                    matched.append((name, pat, settings.get("number_of_shards"), settings.get("number_of_replicas")))
                except Exception:
                    pass
                if matched:
                    print(f"[OpenSearchAdapter] templates matching '{self.index}': " + ", ".join([f"{n}({p}) shards={s} repl={r}" for (n,p,s,r) in matched]))
            except Exception as _te:
                print(f"[OpenSearchAdapter] template diagnostics failed: {_te}")

        # Cluster defaults diagnostics
        if self._debug:
            try:
                cfg = self.client.transport.perform_request("GET", "/_cluster/settings", params={"include_defaults": "true"})
                def _pluck(dct, *keys):
                    cur = dct or {}
                    for k in keys:
                        cur = (cur.get(k) if isinstance(cur, dict) else None) or {}
                    return cur
                idx_defaults = _pluck(cfg, "defaults", "index")
                idx_transient = _pluck(cfg, "transient", "index")
                idx_persistent = _pluck(cfg, "persistent", "index")
                aci = None
                try:
                    aci = (_pluck(cfg, "transient", "action") or {}).get("auto_create_index") or (_pluck(cfg, "persistent", "action") or {}).get("auto_create_index")
                except Exception:
                    aci = None
                print(f"[OpenSearchAdapter] cluster index defaults shards={idx_defaults.get('number_of_shards')} repl={idx_defaults.get('number_of_replicas')} auto_create_index={aci}")
                if idx_transient:
                    print(f"[OpenSearchAdapter] cluster transient index settings: {idx_transient}")
                if idx_persistent:
                    print(f"[OpenSearchAdapter] cluster persistent index settings: {idx_persistent}")
            except Exception as _ce:
                print(f"[OpenSearchAdapter] cluster settings diagnostics failed: {_ce}")

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
                    "chunk_metadata": {"type": "flattened"},
                    "chunk_metadata_text": {"type": "text"},
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
        # Optional shard/replica overrides from env
        shards = os.environ.get("OPENSEARCH_NUMBER_OF_SHARDS")
        replicas = os.environ.get("OPENSEARCH_NUMBER_OF_REPLICAS")
        if shards:
            try:
                body["settings"]["index"]["number_of_shards"] = str(int(shards))
            except Exception:
                pass
        if replicas:
            try:
                body["settings"]["index"]["number_of_replicas"] = str(int(replicas))
            except Exception:
                pass
        if self._debug:
            try:
                sdict = body.get("settings", {}).get("index", {}) if isinstance(body, dict) else {}
                print(f"[OpenSearchAdapter] creating index '{self.index}' with settings: shards={sdict.get('number_of_shards','(default)')}, replicas={sdict.get('number_of_replicas','(default)')}")
            except Exception:
                pass
        try:
            # Avoid ignoring errors so we don't silently fall back to auto_create_index (which would default to 1 shard).
            resp = self.client.indices.create(
                index=self.index,
                body=body,
                params={"wait_for_active_shards": "1"}
            )
            self._md_field_mode = "flattened"
            if self._debug:
                try:
                    print(f"[OpenSearchAdapter] indices.create ack={resp.get('acknowledged')} shards_ack={resp.get('shards_acknowledged')} index={resp.get('index')}")
                except Exception:
                    pass
        except Exception as e:
            # Fallback for clusters that do not support 'flattened' mapping
            emsg = str(e)
            if "No handler for type [flattened]" in emsg or "mapper_parsing_exception" in emsg and "flattened" in emsg:
                if self._debug:
                    print(f"[OpenSearchAdapter] retrying without 'flattened' (fallback to chunk_metadata_text). Reason: {e}")
                try:
                    # Rebuild mapping: disable object indexing (no field explosion) and rely on 'chunk_metadata_text'
                    props = body.get("mappings", {}).get("properties", {})
                    if isinstance(props, dict):
                        props["chunk_metadata"] = {"type": "object", "enabled": False}
                        props["chunk_metadata_text"] = {"type": "text"}
                    resp = self.client.indices.create(
                        index=self.index,
                        body=body,
                        params={"wait_for_active_shards": "1"}
                    )
                    self._md_field_mode = "text"
                    if self._debug:
                        try:
                            print(f"[OpenSearchAdapter] indices.create (fallback) ack={resp.get('acknowledged')} shards_ack={resp.get('shards_acknowledged')} index={resp.get('index')}")
                        except Exception:
                            pass
                except Exception as e2:
                    print(f"[OpenSearchAdapter] ERROR creating index '{self.index}' with fallback mapping: {e2}")
                    raise
            else:
                print(f"[OpenSearchAdapter] ERROR creating index '{self.index}': {e}")
                raise
        # Validate effective settings post-create; optionally enforce match
        try:
            s = self.client.indices.get_settings(index=self.index)
            idx = list(s.keys())[0] if isinstance(s, dict) and s else self.index
            settings = s.get(idx, {}).get("settings", {}).get("index", {}) or {}
            nsh = str(settings.get("number_of_shards"))
            nrepl = str(settings.get("number_of_replicas"))
            try:
                self._primary_shards = int(nsh)
            except Exception:
                self._primary_shards = self._primary_shards or 1
            want_shards = str(int(shards)) if shards else None
            want_repl = str(int(replicas)) if replicas else None
            if self._debug:
                print(f"[OpenSearchAdapter] created index '{self.index}' effective: number_of_shards={nsh}, number_of_replicas={nrepl}")
            enforce = str(os.environ.get("OPENSEARCH_ENFORCE_SHARDS", "0")).strip().lower() in ("1","true","yes","on","y")
            if enforce and ((want_shards and nsh != want_shards) or (want_repl and nrepl != want_repl)):
                raise RuntimeError(f"Index created with unexpected shard/replica (got {nsh}/{nrepl}, wanted {want_shards}/{want_repl}). Check index templates or auto_create_index.")
            # Detect mapping mode by inspecting mapping
            try:
                mp = self.client.indices.get_mapping(index=self.index)
                mprops = (((mp or {}).get(self.index) or {}).get("mappings") or {}).get("properties") or {}
                if "chunk_metadata_text" in mprops:
                    self._md_field_mode = "text"
                else:
                    self._md_field_mode = "flattened"
                if self._debug:
                    print(f"[OpenSearchAdapter] metadata field mode = {self._md_field_mode}")
            except Exception:
                pass
        except Exception as es:
            print(f"[OpenSearchAdapter] WARN: could not validate index settings: {es}")

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
            # Fallback metadata text for clusters without 'flattened' support
            try:
                cm_text_val = self._cm_to_text(ch.get("chunk_metadata") or {})
                if cm_text_val:
                    doc["chunk_metadata_text"] = cm_text_val
            except Exception:
                pass
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
        # Bulk tuning via env:
        # - OPENSEARCH_BULK_CHUNK_SIZE: number of docs per HTTP bulk sub-request (default 500)
        # - OPENSEARCH_BULK_MAX_BYTES: max bytes per bulk sub-request (default 100MB)
        try:
            chunk_size = int(os.environ.get("OPENSEARCH_BULK_CHUNK_SIZE", "500"))
        except Exception:
            chunk_size = 500
        try:
            max_chunk_bytes = int(os.environ.get("OPENSEARCH_BULK_MAX_BYTES", str(100 * 1024 * 1024)))
        except Exception:
            max_chunk_bytes = 100 * 1024 * 1024
        # Optional parallel bulk concurrency for multi-shard clusters
        try:
            requested_conc = int(os.environ.get("OPENSEARCH_BULK_CONCURRENCY", "1"))
        except Exception:
            requested_conc = 1
        if requested_conc <= 0:
            requested_conc = 1

        # Cap concurrency by primary shard count to avoid stalls on single-shard indices
        shard_count = self._get_primary_shard_count()
        eff_conc = min(requested_conc, shard_count) if shard_count > 0 else requested_conc

        if self._debug:
            try:
                print(f"[OpenSearchAdapter] bulk params: requested_concurrency={requested_conc} effective_concurrency={eff_conc} shard_count={shard_count} chunk_size={chunk_size} max_chunk_bytes={max_chunk_bytes}")
            except Exception:
                pass

        if eff_conc > 1:
            # Consume parallel_bulk and collect success/error stats for diagnostics
            ok_cnt = 0
            err_cnt = 0
            first_err = None
            for ok, info in helpers.parallel_bulk(
                self.client,
                actions,
                thread_count=eff_conc,
                chunk_size=chunk_size,
                max_chunk_bytes=max_chunk_bytes,
                request_timeout=self.timeout,
                refresh=False,
            ):
                if ok:
                    ok_cnt += 1
                else:
                    err_cnt += 1
                    if first_err is None:
                        first_err = info
            if err_cnt and self._debug:
                try:
                    print(f"[OpenSearchAdapter] parallel_bulk finished ok={ok_cnt} err={err_cnt}; first_err={first_err}")
                except Exception:
                    print(f"[OpenSearchAdapter] parallel_bulk finished ok={ok_cnt} err={err_cnt}")
        else:
            # Bulk with stats so we can see failures without raising
            try:
                succ, failed = helpers.bulk(
                    self.client,
                    actions,
                    refresh=False,
                    request_timeout=self.timeout,
                    chunk_size=chunk_size,
                    max_chunk_bytes=max_chunk_bytes,
                    stats_only=True,
                    raise_on_error=False,
                    raise_on_exception=False,
                )
                if failed and self._debug:
                    print(f"[OpenSearchAdapter] bulk finished succ={succ} failed={failed}")
            except Exception as e:
                # Ensure we surface the failure reason in debug mode
                if self._debug:
                    print(f"[OpenSearchAdapter] bulk error: {e}")
                raise
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
        try:
            res = self.client.search(index=self.index, body=body, request_timeout=self.timeout)
        except Exception as e:
            if self._debug:
                print(f"[OpenSearchAdapter] BM25 search error on index={self.index}: {e}")
            return []
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
        - mode: "documents" -> search only text
                "metadata"  -> search only chunk_metadata.*
                "both" (default) -> search both with a boolean should clause
        Adds snippet from highlight(text) when available.
        Falls back to text-only match when no hits are found (to align UX with Postgres/Oracle behavior).
        """
        # Build component queries
        q_text = {"match": {"text": {"query": query}}}
        # Choose metadata fields based on mapping mode; fallback uses chunk_metadata_text
        meta_fields = ["chunk_metadata", "chunk_metadata.*"] if getattr(self, "_md_field_mode", "flattened") == "flattened" else ["chunk_metadata_text"]
        q_meta = {
            "query_string": {
                "query": query,
                "fields": meta_fields,
                "lenient": True,
                "default_operator": "AND"
            }
        }
        mode_lc = (mode or "both").strip().lower()
        if mode_lc == "documents":
            q = q_text
        elif mode_lc == "metadata":
            q = q_meta
        else:
            q = {"bool": {"should": [q_text, q_meta], "minimum_should_match": 1}}
            mode_lc = "both"

        body = {
            "size": top_k,
            "query": q,
            "_source": True,
            "highlight": {"fields": {"text": {}}}
        }

        def _to_results(resp, search_area: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for h in resp.get("hits", {}).get("hits", []):
                s = h.get("_source", {})
                hl = h.get("highlight", {}) or {}
                snippet = None
                if isinstance(hl.get("text"), list) and hl["text"]:
                    snippet = hl["text"][0]
                out.append({
                    "doc_id": s.get("doc_id"),
                    "chunk_index": s.get("chunk_index"),
                    "source": s.get("source"),
                    "content": s.get("text"),
                    "text": s.get("text"),
                    "chunk_metadata": s.get("chunk_metadata"),
                    "snippet": snippet,
                    "search_area": search_area
                })
            return out[:top_k]

        try:
            res = self.client.search(index=self.index, body=body, request_timeout=self.timeout)
            results = _to_results(res, mode_lc)
            # Fallback to text-only match if no hits (improves robustness when metadata fields are not text-mapped)
            if not results and mode_lc != "documents":
                fb_body = {
                    "size": top_k,
                    "query": q_text,
                    "_source": True,
                    "highlight": {"fields": {"text": {}}}
                }
                try:
                    fb_res = self.client.search(index=self.index, body=fb_body, request_timeout=self.timeout)
                    results = _to_results(fb_res, "documents")
                except Exception as e2:
                    if self._debug:
                        print(f"[OpenSearchAdapter] FTS fallback error on index={self.index}: {e2}")
                    results = []
            return results
        except Exception as e:
            if self._debug:
                print(f"[OpenSearchAdapter] FTS search error on index={self.index}: {e}")
            return []

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
