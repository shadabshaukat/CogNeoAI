"""
OpenSearch Full Ingestion Worker for CogNeo

Purpose:
- Parse (.txt/.html), perform semantic token-aware chunking, embed on GPU, and bulk index directly to OpenSearch.
- NO database dependency: no EmbeddingSession rows, no DB pings. Progress tracked via per-worker logs only.
- High-throughput pipeline with CPU parse+chunk in subprocesses, GPU embedding in main, and batched OpenSearch bulk API.

New capabilities:
- Resume mode: skip files listed in previous success logs (glob: *.success.log in --log_dir) when --resume is set (or OS_RESUME_FROM_LOGS=1).
- Metrics NDJSON export: per-file structured metrics written to {session}.metrics.ndjson when OS_METRICS_NDJSON=1.
- Error NDJSON export: structured error logs written to {session}.errors.ndjson when COGNEO_ERROR_DETAILS=1 (optional stack traces via COGNEO_ERROR_TRACE=1).
- Optional OpenSearch-backed progress KV: when OS_INGEST_STATE_ENABLE=1, upsert per-file status and session summary into OPENSEARCH_INGEST_STATE_INDEX (default: cogneo_ingest_state).

Usage (typically invoked by an orchestrator):
    python -m ingest.os_worker SESSION_NAME \\
        --root "/abs/path/to/corpus" \\
        --partition_file ".os-gpu-partition-SESSION_NAME-gpu0.txt" \\
        --model "nomic-ai/nomic-embed-text-v1.5" \\
        --target_tokens 512 --overlap_tokens 64 --max_tokens 640 \\
        --log_dir "./logs" \\
        --resume

Environment (selected):
- COGNEO_EMBED_MODEL, COGNEO_EMBED_DIM, COGNEO_EMBED_BATCH
- COGNEO_TIMEOUT_PARSE, COGNEO_TIMEOUT_CHUNK, COGNEO_TIMEOUT_EMBED_BATCH
- COGNEO_CPU_WORKERS, COGNEO_PIPELINE_PREFETCH, COGNEO_SORT_WORKER_FILES
- OPENSEARCH_* (host, index, user/pass, shards/replicas, bulk concurrency/size, etc.)
- OPENSEARCH_INGEST_BATCH_SIZE (accumulate chunks across files before each bulk; default 1000)
- OS_RESUME_FROM_LOGS=1 (alternate to --resume)
- OS_METRICS_NDJSON=1 (enable metrics NDJSON)
- OS_INGEST_STATE_ENABLE=1 (enable OS KV state)
- OPENSEARCH_INGEST_STATE_INDEX=cogneo_ingest_state (state index name)
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import signal
import contextlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import glob
from datetime import datetime

# Chunking stack (semantic; same as beta pipeline quality)
from ingest.semantic_chunker import (
    chunk_document_semantic,
    ChunkingConfig,
    detect_doc_type,
    chunk_legislation_dashed_semantic,
    chunk_generic_rcts,
)

# Optional domain pack (best-effort)
def _try_domain_chunk(text: str, base_meta: Dict[str, Any], cfg: ChunkingConfig):
    try:
        from ingest.domains import load_domain  # type: ignore
        _dm = load_domain()
        _dc = _dm.chunk_document(text, base_meta=base_meta, cfg=cfg)
        if _dc:
            return _dc, "domain"
    except Exception:
        pass
    return None, None

from embedding.embedder import Embedder
from storage.adapters.opensearch import OpenSearchAdapter

# Try to speed up fork behavior to avoid CUDA context issues
try:
    mp.set_start_method("spawn", force=False)
except RuntimeError:
    pass

# Supported file types and simple discover
SUPPORTED_EXTS = {".txt", ".html"}

def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s or "")]

def find_all_supported_files(root_dir: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = sorted(dirnames, key=_natural_sort_key)
        for f in sorted(filenames, key=_natural_sort_key):
            if Path(f).suffix.lower() in SUPPORTED_EXTS:
                out.append(os.path.abspath(os.path.join(dirpath, f)))
    # dedupe + sort
    return sorted(list(dict.fromkeys(out)), key=_natural_sort_key)

def read_partition_file(fname: str) -> List[str]:
    with open(fname, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _sort_by_size_desc(paths: List[str]) -> List[str]:
    def _sz(p: str) -> int:
        try:
            return int(os.path.getsize(p))
        except Exception:
            return 0
    return sorted(paths, key=_sz, reverse=True)

# Timeouts and pipeline settings
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

PARSE_TIMEOUT = _int_env("COGNEO_TIMEOUT_PARSE", 60)
CHUNK_TIMEOUT = _int_env("COGNEO_TIMEOUT_CHUNK", 90)
EMBED_BATCH_TIMEOUT = _int_env("COGNEO_TIMEOUT_EMBED_BATCH", 180)

_CORES = os.cpu_count() or 2
CPU_WORKERS = _int_env("COGNEO_CPU_WORKERS", 0)
if CPU_WORKERS <= 0:
    CPU_WORKERS = max(1, min(8, _CORES - 1))
PIPELINE_PREFETCH = _int_env("COGNEO_PIPELINE_PREFETCH", 64)
if PIPELINE_PREFETCH <= 0:
    PIPELINE_PREFETCH = 64

OPENSEARCH_INGEST_BATCH_SIZE = _int_env("OPENSEARCH_INGEST_BATCH_SIZE", 1000)

def _truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on", "y")

class _Timeout(Exception):
    pass

@contextlib.contextmanager
def _deadline(seconds: int):
    if seconds is None or seconds <= 0:
        yield
        return
    def _handler(signum, frame):
        raise _Timeout(f"operation exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

def _read_text_utf8_latin1(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return ""

def parse_txt(filepath: str) -> Dict[str, Any]:
    # Import lightweight dashed header parser from loader to keep compatibility
    try:
        from ingest.loader import extract_metadata_block  # type: ignore
    except Exception:
        extract_metadata_block = None
    text = _read_text_utf8_latin1(filepath)
    if not text:
        return {}
    meta, body = ({}, text)
    if extract_metadata_block:
        try:
            meta, body = extract_metadata_block(text)
        except Exception:
            meta, body = {}, text
    detected_type = (meta.get("type", "") or "").strip().lower() if isinstance(meta, dict) else ""
    return {
        "text": body,
        "source": filepath,
        "format": detected_type if detected_type else "txt",
        "chunk_metadata": meta or None
    }

def parse_html(filepath: str) -> Dict[str, Any]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        BeautifulSoup = None  # type: ignore
    html = _read_text_utf8_latin1(filepath)
    if not html:
        return {}
    text = ""
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except Exception:
            text = ""
    if not text:
        # fallback: plain text from HTML file
        text = html
    try:
        from ingest.loader import extract_metadata_block  # type: ignore
        meta, body = extract_metadata_block(text)
    except Exception:
        meta, body = {}, text
    detected_type = (meta.get("type", "") or "").strip().lower() if isinstance(meta, dict) else ""
    return {
        "text": body,
        "source": filepath,
        "format": detected_type if detected_type else "html",
        "chunk_metadata": meta or None
    }

_YEAR_DIR_RE = re.compile(r"^(19|20)\d{2}$")  # 1900-2099

def derive_path_metadata(file_path: str, root_dir: str) -> Dict[str, Any]:
    root_dir = os.path.abspath(root_dir) if root_dir else ""
    file_path = os.path.abspath(file_path)
    rel_path = os.path.relpath(file_path, root_dir) if root_dir and file_path.startswith(root_dir) else file_path
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    parts_no_years = [p for p in parts if not _YEAR_DIR_RE.match(p or "")]
    jurisdiction_guess = parts_no_years[0].lower() if parts_no_years else None
    court_guess = None
    if parts_no_years:
        if len(parts_no_years) >= 2:
            last_non_year = parts_no_years[-2] if "." in parts_no_years[-1] else parts_no_years[-1]
            court_guess = last_non_year
    series_guess = "/".join(parts_no_years[1:3]).lower() if len(parts_no_years) >= 3 else None
    return {
        "dataset_root": root_dir,
        "rel_path": rel_path,
        "rel_path_no_years": "/".join(parts_no_years),
        "path_parts": parts,
        "path_parts_no_years": parts_no_years,
        "jurisdiction_guess": jurisdiction_guess,
        "court_guess": court_guess,
        "series_guess": series_guess,
        "filename": os.path.basename(file_path),
        "ext": Path(file_path).suffix.lower(),
    }

def _append_log_line(log_dir: str, session_name: str, file_path: str, success: bool) -> None:
    os.makedirs(log_dir, exist_ok=True)
    fname = f"{session_name}.success.log" if success else f"{session_name}.error.log"
    fpath = os.path.join(log_dir, fname)
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")

def _append_error_detail(
    log_dir: str,
    session_name: str,
    file_path: str,
    stage: str,
    error_type: str,
    message: str,
    duration_ms: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    tb: Optional[str] = None
) -> None:
    # Mirror beta pipeline toggles for parity
    if os.environ.get("COGNEO_ERROR_DETAILS", "1") != "1":
        return
    try:
        rec = {
            "session": session_name,
            "file": file_path,
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "duration_ms": duration_ms,
            "meta": meta or {},
            "ts": int(time.time() * 1000),
        }
        if os.environ.get("COGNEO_ERROR_TRACE", "0") == "1":
            rec["traceback"] = tb or ""
        path = os.path.join(log_dir, f"{session_name}.errors.ndjson")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _append_metrics_detail(
    log_dir: str,
    session_name: str,
    file_path: str,
    detail: Dict[str, Any]
) -> None:
    if os.environ.get("OS_METRICS_NDJSON", "0") != "1":
        return
    try:
        rec = {
            "session": session_name,
            "file": file_path,
            "ts": int(time.time() * 1000),
            **detail
        }
        path = os.path.join(log_dir, f"{session_name}.metrics.ndjson")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _metrics_enabled() -> bool:
    return os.environ.get("COGNEO_LOG_METRICS", "1") != "0"

def _append_success_metrics_line(
    log_dir: str,
    session_name: str,
    file_path: str,
    chunks_count: int,
    text_len: int,
    cfg: ChunkingConfig,
    strategy: str,
    detected_type: Optional[str] = None,
    section_count: Optional[int] = None,
    tokens_est_total: Optional[int] = None,
    tokens_est_mean: Optional[int] = None,
    parse_ms: Optional[int] = None,
    chunk_ms: Optional[int] = None,
    embed_ms: Optional[int] = None,
    index_ms: Optional[int] = None,
) -> None:
    try:
        os.makedirs(log_dir, exist_ok=True)
        fpath = os.path.join(log_dir, f"{session_name}.success.log")
        if not _metrics_enabled():
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(str(file_path) + "\n")
            return
        parts = [
            str(file_path),
            f"chunks={int(chunks_count)}",
            f"text_len={int(text_len)}",
            f"strategy={strategy or ''}",
            f"target_tokens={int(cfg.target_tokens)}",
            f"overlap_tokens={int(cfg.overlap_tokens)}",
            f"max_tokens={int(cfg.max_tokens)}",
        ]
        if detected_type:
            parts.append(f"type={detected_type}")
        if section_count is not None:
            parts.append(f"section_count={int(section_count)}")
        if tokens_est_total is not None:
            parts.append(f"tokens_est_total={int(tokens_est_total)}")
        if tokens_est_mean is not None:
            parts.append(f"tokens_est_mean={int(tokens_est_mean)}")
        if parse_ms is not None:
            parts.append(f"parse_ms={int(parse_ms)}")
        if chunk_ms is not None:
            parts.append(f"chunk_ms={int(chunk_ms)}")
        if embed_ms is not None:
            parts.append(f"embed_ms={int(embed_ms)}")
        if index_ms is not None:
            parts.append(f"index_ms={int(index_ms)}")
        line = "\t".join(parts)
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def _write_logs(log_dir: str, session_name: str, successes: List[str], failures: List[str]) -> Dict[str, str]:
    os.makedirs(log_dir, exist_ok=True)
    succ_path = os.path.join(log_dir, f"{session_name}.success.log")
    fail_path = os.path.join(log_dir, f"{session_name}.error.log")
    try:
        open(succ_path, "a", encoding="utf-8").close()
    except Exception:
        pass
    try:
        open(fail_path, "a", encoding="utf-8").close()
    except Exception:
        pass
    successes = list(dict.fromkeys(successes))
    failures = list(dict.fromkeys(failures))
    try:
        with open(succ_path, "a", encoding="utf-8") as f:
            f.write(f"# summary files_ok={len(successes)}\n")
    except Exception:
        pass
    try:
        with open(fail_path, "a", encoding="utf-8") as f:
            f.write(f"# summary files_failed={len(failures)}\n")
    except Exception:
        pass
    return {"success_log": succ_path, "error_log": fail_path}

class _OSState:
    def __init__(self, adapter: OpenSearchAdapter, enabled: bool):
        self.enabled = enabled
        self.client = adapter.client if enabled else None
        self.index = os.environ.get("OPENSEARCH_INGEST_STATE_INDEX", "cogneo_ingest_state")
        if not self.enabled or self.client is None:
            return
        try:
            if not self.client.indices.exists(index=self.index):
                body = {
                    "mappings": {
                        "properties": {
                            "type": {"type": "keyword"},
                            "session": {"type": "keyword"},
                            "file": {"type": "keyword"},
                            "status": {"type": "keyword"},
                            "chunks": {"type": "integer"},
                            "text_len": {"type": "integer"},
                            "embed_ms": {"type": "integer"},
                            "index_ms": {"type": "integer"},
                            "started_at": {"type": "date"},
                            "ended_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "files_ok": {"type": "integer"},
                            "files_failed": {"type": "integer"},
                            "total_indexed": {"type": "integer"}
                        }
                    }
                }
                self.client.indices.create(index=self.index, body=body)
        except Exception as e:
            print(f"[os_worker] WARN: could not ensure state index '{self.index}': {e}", flush=True)
            self.enabled = False

    def upsert_session_start(self, session: str, total_files: int):
        if not self.enabled:
            return
        try:
            doc = {
                "type": "session",
                "session": session,
                "status": "running",
                "total_files": int(total_files),
                "started_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            self.client.index(index=self.index, id=f"{session}::summary", body=doc, refresh=False)
        except Exception:
            pass

    def upsert_file_status(self, session: str, file_path: str, status: str, meta: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        try:
            doc = {
                "type": "file",
                "session": session,
                "file": file_path,
                "status": status,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            if meta:
                doc.update(meta)
            self.client.index(index=self.index, id=f"{session}::{file_path}", body=doc, refresh=False)
        except Exception:
            pass

    def upsert_session_end(self, session: str, files_ok: int, files_failed: int, total_indexed: int):
        if not self.enabled:
            return
        try:
            doc = {
                "type": "session",
                "session": session,
                "status": "complete",
                "files_ok": int(files_ok),
                "files_failed": int(files_failed),
                "total_indexed": int(total_indexed),
                "ended_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            self.client.index(index=self.index, id=f"{session}::summary", body=doc, refresh=False)
        except Exception:
            pass

def _embed_in_batches(embedder: Embedder, texts: List[str], batch_size: int) -> List:
    if batch_size <= 0:
        batch_size = 64
    all_vecs = []
    i = 0
    cur_bs = int(batch_size)
    n = len(texts)
    while i < n:
        sub = texts[i:i + cur_bs]
        try:
            with _deadline(EMBED_BATCH_TIMEOUT):
                vecs = embedder.embed(sub)  # ndarray [batch, dim]
            for j in range(vecs.shape[0]):
                all_vecs.append(vecs[j])
            i += cur_bs
            if cur_bs < batch_size:
                cur_bs = min(batch_size, max(1, cur_bs * 2))
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                next_bs = max(1, cur_bs // 2)
                if next_bs == cur_bs:
                    raise
                print(f"[os_worker] embed OOM/backoff: reducing batch_size {cur_bs} -> {next_bs}", flush=True)
                cur_bs = next_bs
                time.sleep(0.5)
            else:
                raise
    return all_vecs

def _fallback_chunk_text(text: str, base_meta: Dict[str, Any], cfg: ChunkingConfig) -> List[Dict[str, Any]]:
    chars_per_chunk = _int_env("COGNEO_FALLBACK_CHARS_PER_CHUNK", 4000)
    overlap_chars = _int_env("COGNEO_FALLBACK_OVERLAP_CHARS", 200)
    if chars_per_chunk <= 0:
        chars_per_chunk = 4000
    if overlap_chars < 0:
        overlap_chars = 0
    step = max(1, chars_per_chunk - overlap_chars)
    chunks: List[Dict[str, Any]] = []
    n = len(text or "")
    i = 0
    idx = 0
    while i < n:
        j = min(n, i + chars_per_chunk)
        slice_text = text[i:j]
        md = dict(base_meta or {})
        md.setdefault("strategy", "fallback-naive")
        md["fallback"] = True
        md["start_char"] = i
        md["end_char"] = j
        chunks.append({"text": slice_text, "chunk_metadata": md})
        idx += 1
        i += step
    return chunks

def _cpu_prepare_file(
    filepath: str,
    root_dir: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
) -> Dict[str, Any]:
    try:
        t0 = time.time()
        ext = Path(filepath).suffix.lower()
        if ext == ".txt":
            base_doc = parse_txt(filepath)
        elif ext == ".html":
            base_doc = parse_html(filepath)
        else:
            return {"filepath": filepath, "status": "skip"}
        text_len = len(base_doc.get("text", "") or "")
        parse_ms = int((time.time() - t0) * 1000)
        if not base_doc or not base_doc.get("text"):
            return {
                "filepath": filepath,
                "status": "empty",
                "text_len": 0,
                "chunk_count": 0,
                "chunk_strategy": "no-chunks",
                "parse_ms": parse_ms,
                "chunk_ms": None,
            }

        path_meta = derive_path_metadata(filepath, root_dir or "")
        base_meta = dict(base_doc.get("chunk_metadata") or {})
        base_meta.update(path_meta)

        detected_type = detect_doc_type(base_meta, base_doc.get("text", ""))

        cfg = ChunkingConfig(
            target_tokens=int(token_target),
            overlap_tokens=int(token_overlap),
            max_tokens=int(token_max),
        )

        chunk_strategy = None
        t1 = time.time()
        file_chunks = []

        # Domain-pack first
        dc, st = _try_domain_chunk(base_doc["text"], base_meta, cfg)
        if dc:
            file_chunks = dc
            chunk_strategy = st

        # Chunk with deadlines
        with _deadline(CHUNK_TIMEOUT):
            if not file_chunks:
                # dashed header aware
                file_chunks = chunk_legislation_dashed_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
                if file_chunks:
                    chunk_strategy = "dashed-semantic"
            if not file_chunks:
                file_chunks = chunk_document_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
                if file_chunks:
                    chunk_strategy = "semantic"
            if (not file_chunks) and os.environ.get("COGNEO_USE_RCTS_GENERIC", "0") == "1":
                rcts_chunks = chunk_generic_rcts(base_doc["text"], base_meta=base_meta, cfg=cfg)
                if rcts_chunks:
                    file_chunks = rcts_chunks
                    chunk_strategy = "rcts-generic"

        chunk_ms = int((time.time() - t1) * 1000)

        if not file_chunks:
            return {
                "filepath": filepath,
                "status": "zero_chunks",
                "text_len": text_len,
                "chunk_count": 0,
                "detected_type": detected_type,
                "chunk_strategy": chunk_strategy or "no-chunks",
                "parse_ms": parse_ms,
                "chunk_ms": chunk_ms,
            }

        # Light metrics
        section_idxs = set()
        token_vals = []
        for c in file_chunks:
            md = c.get("chunk_metadata") or {}
            if isinstance(md, dict):
                si = md.get("section_idx")
                if si is not None:
                    section_idxs.add(si)
                tv = md.get("tokens_est")
                if isinstance(tv, int):
                    token_vals.append(tv)
        section_count = len(section_idxs) if section_idxs else 0
        tokens_est_total = sum(token_vals) if token_vals else None
        tokens_est_mean = int(round(tokens_est_total / len(token_vals))) if token_vals else None

        return {
            "filepath": filepath,
            "status": "ok",
            "chunks": file_chunks,
            "text_len": text_len,
            "chunk_count": len(file_chunks),
            "detected_type": detected_type,
            "chunk_strategy": chunk_strategy or "semantic",
            "section_count": section_count,
            "tokens_est_total": tokens_est_total,
            "tokens_est_mean": tokens_est_mean,
            "parse_ms": parse_ms,
            "chunk_ms": chunk_ms,
            "format": base_doc.get("format", ext.strip(".")),
            "source": filepath,
        }
    except _Timeout:
        try:
            if "base_doc" in locals() and isinstance(base_doc, dict) and base_doc.get("text"):
                cfg = ChunkingConfig(
                    target_tokens=int(token_target),
                    overlap_tokens=int(token_overlap),
                    max_tokens=int(token_max),
                )
                fb_chunks = _fallback_chunk_text(base_doc.get("text", ""), base_meta, cfg)  # type: ignore
                section_idxs = set()
                token_vals = []
                for c in fb_chunks:
                    md = c.get("chunk_metadata") or {}
                    if isinstance(md, dict):
                        si = md.get("section_idx")
                        if si is not None:
                            section_idxs.add(si)
                        tv = md.get("tokens_est")
                        if isinstance(tv, int):
                            token_vals.append(tv)
                section_count = len(section_idxs) if section_idxs else 0
                tokens_est_total = sum(token_vals) if token_vals else None
                tokens_est_mean = int(round(tokens_est_total / len(token_vals))) if token_vals else None
                return {
                    "filepath": filepath,
                    "status": "fallback_ok",
                    "chunks": fb_chunks,
                    "text_len": len(base_doc.get("text","")),  # type: ignore
                    "chunk_count": len(fb_chunks),
                    "detected_type": detect_doc_type({}, base_doc.get("text","")),  # type: ignore
                    "chunk_strategy": "fallback-naive",
                    "section_count": section_count,
                    "tokens_est_total": tokens_est_total,
                    "tokens_est_mean": tokens_est_mean,
                    "parse_ms": None,
                    "chunk_ms": None,
                    "format": base_doc.get("format", Path(filepath).suffix.lower().strip(".")),
                    "source": filepath,
                }
        except Exception as e2:
            return {"filepath": filepath, "status": "error", "error": f"Timeout + fallback error: {e2}"}
        return {"filepath": filepath, "status": "error", "error": "Timeout in chunking"}
    except Exception as e:
        return {"filepath": filepath, "status": "error", "error": str(e)}

def run_worker_pipelined(
    session_name: str,
    root_dir: Optional[str],
    partition_file: Optional[str],
    embedding_model: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
    log_dir: str,
    resume: bool
) -> None:
    print(f"[os_worker] start session={session_name} cwd={os.getcwd()}", flush=True)

    # Resolve files
    if partition_file:
        files = read_partition_file(partition_file)
    else:
        if not root_dir:
            raise ValueError("Either --partition_file or --root must be provided")
        files = find_all_supported_files(root_dir)

    # Resume: skip files present in any *.success.log within log_dir
    if resume or _truthy(os.environ.get("OS_RESUME_FROM_LOGS")):
        try:
            completed: set[str] = set()
            for path in glob.glob(os.path.join(log_dir or ".", "*.success.log")):
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        s = (ln or "").strip()
                        if not s or s.startswith("#"):
                            continue
                        completed.add(s)
            before = len(files)
            files = [fp for fp in files if fp not in completed]
            print(f"[os_worker] {session_name}: resume enabled -> skipping {before - len(files)} already-completed files", flush=True)
        except Exception as e:
            print(f"[os_worker] {session_name}: resume scan failed (continuing without resume): {e}", flush=True)

    if os.environ.get("COGNEO_SORT_WORKER_FILES", "1") != "0":
        try:
            files = _sort_by_size_desc(files)
            print(f"[os_worker] {session_name}: sorted {len(files)} files by size desc", flush=True)
        except Exception as e:
            print(f"[os_worker] {session_name}: note: could not sort by size: {e}", flush=True)

    os.makedirs(log_dir, exist_ok=True)
    succ_path = os.path.join(log_dir, f"{session_name}.success.log")
    err_path = os.path.join(log_dir, f"{session_name}.error.log")
    print(f"[os_worker] {session_name}: logs: success->{succ_path}, error->{err_path}", flush=True)

    # Instantiate embedder after pool creation to avoid CUDA context inheritance
    adapter = OpenSearchAdapter()  # ensures index exists (mapping/shards, etc.)
    print(f"[os_worker] {session_name}: OpenSearch index={adapter.index}", flush=True)
    state = _OSState(adapter, enabled=_truthy(os.environ.get("OS_INGEST_STATE_ENABLE")))
    state.upsert_session_start(session_name, total_files=len(files))

    successes: List[str] = []
    failures: List[str] = []

    # Accumulators for OS index bulks
    ingest_batch = int(os.environ.get("OPENSEARCH_INGEST_BATCH_SIZE", str(OPENSEARCH_INGEST_BATCH_SIZE)))
    cur_chunks: List[Dict[str, Any]] = []
    cur_vecs: List = []
    total_indexed = 0

    def _flush_os():
        nonlocal cur_chunks, cur_vecs, total_indexed
        if not cur_chunks:
            return 0
        t0 = time.time()
        try:
            adapter.index_chunks(cur_chunks, cur_vecs, cur_chunks[0].get("source", "unknown"), cur_chunks[0].get("format", "txt"))
            dt = int((time.time() - t0) * 1000)
            total_indexed += len(cur_chunks)
            print(f"[os_worker] {session_name}: indexed batch size={len(cur_chunks)} total={total_indexed} ({dt}ms)", flush=True)
            flushed = len(cur_chunks)
        finally:
            cur_chunks = []
            cur_vecs = []
        return flushed

    with ProcessPoolExecutor(max_workers=CPU_WORKERS) as pool:
        embedder = Embedder(embedding_model) if embedding_model else Embedder()
        batch_size = _int_env("COGNEO_EMBED_BATCH", 64)
        # submitter
        file_iter = iter(files)
        inflight = {}
        submitted = 0
        done_count = 0
        total_files = len(files)
        files_ok = 0
        files_failed = 0
        total_indexed_local = 0

        def _submit_next():
            nonlocal submitted
            while len(inflight) < PIPELINE_PREFETCH:
                try:
                    fp = next(file_iter)
                except StopIteration:
                    return
                submitted += 1
                print(f"[os_worker] {session_name}: start {submitted}/{total_files} -> {fp}", flush=True)
                fut = pool.submit(_cpu_prepare_file, fp, root_dir, int(token_target), int(token_overlap), int(token_max))
                inflight[fut] = fp

        _submit_next()

        while inflight:
            for fut in as_completed(list(inflight.keys()), timeout=None):
                fp = inflight.pop(fut)
                done_count += 1
                try:
                    res = fut.result()
                except Exception as e:
                    failures.append(fp)
                    files_failed += 1
                    _append_log_line(log_dir, session_name, fp, success=False)
                    _append_error_detail(
                        log_dir, session_name, fp,
                        "prep", type(e).__name__, str(e),
                        None, None, None
                    )
                    state.upsert_file_status(session_name, fp, "error", {"stage": "prep"})
                    print(f"[os_worker] {session_name}: FAILED (prep) {done_count}/{total_files} -> {fp} :: {e}", flush=True)
                    break

                status = res.get("status")
                if status in ("skip", "empty", "zero_chunks"):
                    successes.append(fp)
                    files_ok += 1
                    _append_success_metrics_line(
                        log_dir=log_dir,
                        session_name=session_name,
                        file_path=fp,
                        chunks_count=0,
                        text_len=res.get("text_len") or 0,
                        cfg=ChunkingConfig(target_tokens=int(token_target), overlap_tokens=int(token_overlap), max_tokens=int(token_max)),
                        strategy=res.get("chunk_strategy") or "no-chunks",
                        detected_type=res.get("detected_type"),
                        section_count=0,
                        tokens_est_total=0,
                        tokens_est_mean=0,
                        parse_ms=res.get("parse_ms"),
                        chunk_ms=res.get("chunk_ms"),
                        embed_ms=None,
                        index_ms=None,
                    )
                    _append_metrics_detail(
                        log_dir=log_dir,
                        session_name=session_name,
                        file_path=fp,
                        detail={
                            "status": "complete",
                            "chunks": 0,
                            "text_len": res.get("text_len") or 0,
                            "parse_ms": res.get("parse_ms"),
                            "chunk_ms": res.get("chunk_ms"),
                            "embed_ms": None,
                            "index_ms": None,
                        }
                    )
                    state.upsert_file_status(session_name, fp, "complete", {"chunks": 0, "text_len": res.get("text_len") or 0})
                    print(f"[os_worker] {session_name}: OK (0 chunks) {done_count}/{total_files} -> {fp}", flush=True)
                    break

                if status == "error":
                    failures.append(fp)
                    files_failed += 1
                    _append_log_line(log_dir, session_name, fp, success=False)
                    _append_error_detail(
                        log_dir, session_name, fp,
                        "prep", "Error", str(res.get("error") or ""),
                        None, None, None
                    )
                    state.upsert_file_status(session_name, fp, "error", {"stage": "prep"})
                    print(f"[os_worker] {session_name}: FAILED (prep) {done_count}/{total_files} -> {fp} :: {res.get('error')}", flush=True)
                    break

                file_chunks = res.get("chunks") or []
                current_chunk_count = int(res.get("chunk_count") or len(file_chunks))
                current_text_len = int(res.get("text_len") or 0)
                chunk_strategy = res.get("chunk_strategy") or "semantic"
                detected_type = res.get("detected_type")
                fmt = res.get("format") or "txt"
                source_val = res.get("source") or fp

                texts = [c["text"] for c in file_chunks]
                print(f"[os_worker] {session_name}: parse done {res.get('parse_ms')}ms len={current_text_len}", flush=True)
                print(f"[os_worker] {session_name}: chunk done {res.get('chunk_ms')}ms chunks={current_chunk_count}", flush=True)

                # embed
                print(f"[os_worker] {session_name}: embed start batch={batch_size} texts={len(texts)}", flush=True)
                tE = time.time()
                vecs = _embed_in_batches(embedder, texts, batch_size=batch_size)
                embed_ms = int((time.time() - tE) * 1000)
                print(f"[os_worker] {session_name}: embed done {embed_ms}ms", flush=True)

                # accumulate for OS bulk
                tI = time.time()
                for idx, ch in enumerate(file_chunks):
                    cm = ch.get("chunk_metadata") or {}
                    if isinstance(cm, str):
                        try:
                            cm = json.loads(cm)
                        except Exception:
                            cm = {}
                    cur_chunks.append({
                        # no doc_id available (DB-less); adapter will fallback to _id="source#i"
                        "doc_id": None,
                        "chunk_index": idx,
                        "text": ch.get("text", ""),
                        "chunk_metadata": cm,
                        "source": source_val,
                        "format": fmt,
                    })
                    cur_vecs.append(vecs[idx])
                    if len(cur_chunks) >= ingest_batch:
                        _flush_os()
                index_ms = int((time.time() - tI) * 1000)
                total_indexed_local += current_chunk_count

                successes.append(fp)
                files_ok += 1
                _append_success_metrics_line(
                    log_dir=log_dir,
                    session_name=session_name,
                    file_path=fp,
                    chunks_count=current_chunk_count or 0,
                    text_len=current_text_len or 0,
                    cfg=ChunkingConfig(target_tokens=int(token_target), overlap_tokens=int(token_overlap), max_tokens=int(token_max)),
                    strategy=chunk_strategy,
                    detected_type=detected_type,
                    section_count=res.get("section_count"),
                    tokens_est_total=res.get("tokens_est_total"),
                    tokens_est_mean=res.get("tokens_est_mean"),
                    parse_ms=res.get("parse_ms"),
                    chunk_ms=res.get("chunk_ms"),
                    embed_ms=embed_ms,
                    index_ms=index_ms,
                )

                _append_metrics_detail(
                    log_dir=log_dir,
                    session_name=session_name,
                    file_path=fp,
                    detail={
                        "status": "complete",
                        "chunks": current_chunk_count or 0,
                        "text_len": current_text_len or 0,
                        "parse_ms": res.get("parse_ms"),
                        "chunk_ms": res.get("chunk_ms"),
                        "embed_ms": embed_ms,
                        "index_ms": index_ms,
                    }
                )
                state.upsert_file_status(session_name, fp, "complete", {
                    "chunks": current_chunk_count or 0,
                    "text_len": current_text_len or 0,
                    "embed_ms": embed_ms,
                    "index_ms": index_ms
                })
                if done_count % 10 == 0:
                    print(f"[os_worker] {session_name}: OK {done_count}/{total_files}, total_indexed={total_indexed}+{len(cur_chunks)} (pending)", flush=True)
                break  # process next completion

            _submit_next()

    # final flush
    _flush_os()

    paths = _write_logs(log_dir, session_name, successes, failures)
    state.upsert_session_end(session_name, files_ok=len(successes), files_failed=len(failures), total_indexed=total_indexed)
    print(f"[os_worker] Session {session_name} complete. Files OK: {len(successes)}, failed: {len(failures)}, total_indexed={total_indexed}", flush=True)
    print(f"[os_worker] Success log: {paths['success_log']}", flush=True)
    print(f"[os_worker] Error log:   {paths['error_log']}", flush=True)

def run_worker(
    session_name: str,
    root_dir: Optional[str],
    partition_file: Optional[str],
    embedding_model: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
    log_dir: str,
    resume: bool
) -> None:
    # Always use pipelined mode for throughput
    return run_worker_pipelined(session_name, root_dir, partition_file, embedding_model, token_target, token_overlap, token_max, log_dir, resume)

def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    import argparse
    ap = argparse.ArgumentParser(description="OpenSearch full-ingest worker: semantic chunking + embeddings + OpenSearch bulk.")
    ap.add_argument("session_name", help="Child session name (e.g., os-sess-...-gpu0)")
    ap.add_argument("--root", default=None, help="Root directory (used if --partition_file not provided)")
    ap.add_argument("--partition_file", default=None, help="Text file listing file paths for this worker")
    ap.add_argument("--model", default=None, help="Embedding model (optional)")
    ap.add_argument("--target_tokens", type=int, default=512, help="Chunking target tokens (default 512)")
    ap.add_argument("--overlap_tokens", type=int, default=64, help="Chunk overlap tokens (default 64)")
    ap.add_argument("--max_tokens", type=int, default=640, help="Hard max per chunk (default 640)")
    ap.add_argument("--log_dir", default="./logs", help="Directory to write per-worker success/error logs")
    ap.add_argument("--resume", action="store_true", help="Skip files already present in *.success.log files under --log_dir")
    return vars(ap.parse_args(argv))

if __name__ == "__main__":
    args = _parse_cli_args(sys.argv[1:])
    run_worker(
        session_name=args["session_name"],
        root_dir=args.get("root"),
        partition_file=args.get("partition_file"),
        embedding_model=args.get("model"),
        token_target=args.get("target_tokens") or 512,
        token_overlap=args.get("overlap_tokens") or 64,
        token_max=args.get("max_tokens") or 640,
        log_dir=args.get("log_dir") or "./logs",
        resume=bool(args.get("resume")),
    )
