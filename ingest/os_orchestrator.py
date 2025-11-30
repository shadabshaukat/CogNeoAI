"""
Multi-GPU Orchestrator for OpenSearch Full Ingestion (DB-less)

Purpose:
- Discover corpus files, partition dynamically into shards, and launch one os_worker per GPU.
- Ensures/tunes the OpenSearch index once per run (optional tuning via env).
- Aggregates per-worker logs into master success/error logs.
- No PostgreSQL/Oracle dependency: this pipeline writes vectors and chunks directly to OpenSearch.

Usage (full ingest across detected GPUs):
    python -m ingest.os_orchestrator \\
      --root "/abs/path/to/corpus" \\
      --session "os-full-$(date +%Y%m%d-%H%M%S)" \\
      --model "nomic-ai/nomic-embed-text-v1.5" \\
      --log_dir "./logs"

Options:
    --gpus 0|N        # 0 = auto-detect, else cap to detected
    --sample_per_folder
    --no_skip_years_in_sample
    --target_tokens 512 --overlap_tokens 64 --max_tokens 640
    --balance_by_size
    --shards 0        # 0 = auto GPUs*4 (dynamic scheduling)
    --no_wait         # exit after launch (no aggregation)

Environment (selected):
- OPENSEARCH_* (host, index, user/pass, shards/replicas, timeout/retries, debug, etc.)
- OPENSEARCH_TUNE_INDEX=1  # Orchestrator will set refresh_interval=-1 & replicas=0 pre-run, then restore.
- OPENSEARCH_NUMBER_OF_SHARDS / OPENSEARCH_NUMBER_OF_REPLICAS (for first-time index create)
- OPENSEARCH_FORCE_RECREATE=1 (drop existing before create; dangerous)
- OPENSEARCH_ENFORCE_SHARDS=1 (error if created shards != desired)
- OPENSEARCH_INGEST_BATCH_SIZE (worker-side batch accumulator)
- COGNEO_* embedding/timeout/CPU workers/pipeline prefetch knobs

Notes:
- File coverage and partition manifest are written under --log_dir.
- Partition files: .os-gpu-partition-{child}.txt
- Children launched as: python -m ingest.os_worker ...
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

def _natural_sort_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s or "")]

# Reuse listing helpers from beta stack for parity
try:
    from ingest.beta_worker import find_all_supported_files  # type: ignore
except Exception:
    # Minimal fallback
    def find_all_supported_files(root_dir: str) -> List[str]:
        SUPPORTED_EXTS = {".txt", ".html"}
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = sorted(dirnames, key=_natural_sort_key)
            files = [f for f in filenames if Path(f).suffix.lower() in SUPPORTED_EXTS]
            for f in sorted(files, key=_natural_sort_key):
                out.append(os.path.abspath(os.path.join(dirpath, f)))
        return sorted(list(dict.fromkeys(out)), key=_natural_sort_key)

try:
    from ingest.beta_scanner import find_sample_files  # type: ignore
except Exception:
    def find_sample_files(root_dir: str, skip_year_dirs: bool = True) -> List[str]:
        # Simple fallback: one file per leaf folder
        SUPPORTED_EXTS = {".txt", ".html"}
        seen = set()
        out = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = sorted(dirnames, key=_natural_sort_key)
            files = [f for f in sorted(filenames, key=_natural_sort_key) if Path(f).suffix.lower() in SUPPORTED_EXTS]
            if files:
                fp = os.path.abspath(os.path.join(dirpath, files[0]))
                if fp not in seen:
                    seen.add(fp)
                    out.append(fp)
        return sorted(out, key=_natural_sort_key)

def get_num_gpus() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        gpus = [line for line in result.stdout.split("\n") if "GPU" in line]
        return max(1, len(gpus))
    except Exception:
        return 1

def partition(items: List[str], n: int) -> List[List[str]]:
    if n <= 1:
        return [items]
    k, m = divmod(len(items), n)
    parts = [items[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
    return [p for p in parts if p]

def partition_by_size(items: List[str], n: int) -> List[List[str]]:
    if n <= 1:
        return [items]
    try:
        def _sz(p: str) -> int:
            try:
                return int(os.path.getsize(p))
            except Exception:
                return 0
        sized = sorted(items, key=_sz, reverse=True)
        bins: List[List[str]] = [[] for _ in range(n)]
        bin_sizes = [0] * n
        for path in sized:
            j = min(range(n), key=lambda i: bin_sizes[i])
            bins[j].append(path)
            bin_sizes[j] += _sz(path)
        return [b for b in bins if b]
    except Exception:
        return partition(items, n)

def _file_sizes(paths: List[str]) -> List[int]:
    out = []
    for p in paths:
        try:
            out.append(int(os.path.getsize(p)))
        except Exception:
            out.append(0)
    return out

def _gini(values: List[int]) -> float:
    vals = [v for v in values if v >= 0]
    if not vals:
        return 0.0
    vals.sort()
    n = len(vals)
    cum = 0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    total = sum(vals)
    if total == 0:
        return 0.0
    return (2 * cum) / (n * total) - (n + 1) / n

# OpenSearch index ensure/tune
def _truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on", "y")

def _get_index_settings(client, index: str) -> Dict[str, Any]:
    try:
        s = client.indices.get_settings(index=index)
        idx = list(s.keys())[0] if isinstance(s, dict) and s else index
        return (s.get(idx, {}).get("settings", {}) or {})
    except Exception:
        return {}

def _put_index_settings(client, index: str, settings: Dict[str, Any]) -> None:
    try:
        client.indices.put_settings(index=index, body=settings)
    except Exception as e:
        print(f"[os_orchestrator] WARN: put_settings failed: {e}", flush=True)

def _tune_index_for_bulk(adapter) -> Dict[str, Any]:
    restore: Dict[str, Any] = {}
    try:
        if not _truthy(os.environ.get("OPENSEARCH_TUNE_INDEX")):
            return restore
        cur = _get_index_settings(adapter.client, adapter.index)
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
        print(f"[os_orchestrator] Tuning index '{adapter.index}' for bulk: refresh_interval=-1, number_of_replicas=0", flush=True)
        _put_index_settings(adapter.client, adapter.index, {"index": {"refresh_interval": "-1", "number_of_replicas": "0"}})
    except Exception as e:
        print(f"[os_orchestrator] WARN: index tuning (pre) failed: {e}", flush=True)
    return restore

def _restore_index_settings(adapter, restore: Dict[str, Any]) -> None:
    try:
        if not restore:
            return
        print(f"[os_orchestrator] Restoring index '{adapter.index}' settings: {restore}", flush=True)
        _put_index_settings(adapter.client, adapter.index, restore)
    except Exception as e:
        print(f"[os_orchestrator] WARN: index tuning (restore) failed: {e}", flush=True)

def write_partition_file(part: List[str], fname: str) -> None:
    with open(fname, "w", encoding="utf-8") as f:
        for p in part:
            f.write(p + "\n")

def launch_worker(
    child_session: str,
    root_dir: str,
    partition_file: str,
    embedding_model: Optional[str],
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
    log_dir: str,
    resume: bool = False,
    gpu_index: Optional[int] = None,
    python_exec: Optional[str] = None
) -> subprocess.Popen:
    python_exec = python_exec or os.environ.get("PYTHON_EXEC", sys.executable)
    cmd = [
        python_exec, "-m", "ingest.os_worker",
        child_session,
        "--root", root_dir,
        "--partition_file", partition_file,
        "--target_tokens", str(int(target_tokens)),
        "--overlap_tokens", str(int(overlap_tokens)),
        "--max_tokens", str(int(max_tokens)),
        "--log_dir", log_dir,
    ]
    if resume:
        cmd.append("--resume")
    if embedding_model:
        cmd.extend(["--model", embedding_model])
    env = dict(os.environ)
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    print(f"[os_orchestrator] launching: {' '.join(cmd)} (GPU={env.get('CUDA_VISIBLE_DEVICES','N/A')})", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    return proc

def orchestrate(
    root_dir: str,
    session_name: str,
    embedding_model: Optional[str],
    num_gpus: Optional[int],
    sample_per_folder: bool,
    skip_year_dirs_in_sample: bool,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
    log_dir: str,
    resume: bool = False,
    shards: int = 0,
    balance_by_size: bool = False,
    wait: bool = True
) -> Dict[str, Any]:
    print(f"[os_orchestrator] start session={session_name}", flush=True)
    print(f"[os_orchestrator] root={root_dir}, model={embedding_model}, log_dir={log_dir}", flush=True)
    print(f"[os_orchestrator] cwd={os.getcwd()}", flush=True)

    # Ensure/tune index once
    from storage.adapters.opensearch import OpenSearchAdapter
    adapter = OpenSearchAdapter()
    adapter.create_all_tables()
    restore_settings = _tune_index_for_bulk(adapter)

    log_root = Path(log_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    start_iso = datetime.utcnow().isoformat() + "Z"

    # Resolve file list
    print(f"[os_orchestrator] Resolving files (sample_mode={bool(sample_per_folder)})...", flush=True)
    if sample_per_folder:
        files = find_sample_files(root_dir, skip_year_dirs=skip_year_dirs_in_sample)
    else:
        files = find_all_supported_files(root_dir)
    files = sorted(list(dict.fromkeys(files)), key=_natural_sort_key)
    total_files = len(files)
    print(f"[os_orchestrator] Found files={total_files}", flush=True)

    if total_files == 0:
        raise RuntimeError(f"No supported files found under root: {root_dir}")

    detected = get_num_gpus()
    if num_gpus is None or num_gpus <= 0:
        num = detected
    else:
        num = max(1, min(num_gpus, detected))
    print(f"[os_orchestrator] GPUs detected={detected}, using={num}, parent CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}", flush=True)

    shards_count = int(shards) if shards and shards > 0 else max(num * 4, num)
    shards_count = min(shards_count, max(1, len(files)))

    if not balance_by_size:
        sizes = _file_sizes(files)
        gini = _gini(sizes)
        if gini >= 0.6 and num > 1:
            balance_by_size = True
            print(f"[os_orchestrator] Detected skewed size distribution (gini={gini:.2f}); enabling size-balanced partitioning.", flush=True)

    if balance_by_size:
        print(f"[os_orchestrator] Partitioning by total size into {shards_count} shard(s) over {num} GPU(s)...", flush=True)
        parts = partition_by_size(files, shards_count)
    else:
        print(f"[os_orchestrator] Partitioning evenly into {shards_count} shard(s) over {num} GPU(s)...", flush=True)
        parts = partition(files, shards_count)
    if not parts:
        parts = [files]

    # Coverage validation
    flat = [p for sub in parts for p in sub]
    unique = set(flat)
    if len(flat) != len(files) or len(unique) != len(files):
        extra = [p for p in flat if flat.count(p) > 1]
        missing = [p for p in files if p not in unique]
        manifest = {
            "total_files": len(files),
            "assigned_total": len(flat),
            "unique_assigned": len(unique),
            "duplicates": sorted(list(dict.fromkeys(extra))),
            "missing_sample": missing[:100],
        }
        man_path = Path(log_root) / f"{session_name}.partition.validation.json"
        try:
            with man_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
        raise RuntimeError(f"Partition coverage validation failed. See {man_path} for details.")
    # Manifest
    try:
        manifest = {
            "total_files": len(files),
            "shards": [{"index": i, "count": len(p)} for i, p in enumerate(parts)],
            "assigned_total": sum(len(p) for p in parts),
            "unique_assigned": len(unique),
        }
        man_path = Path(log_root) / f"{session_name}.partition.manifest.json"
        with man_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[os_orchestrator] Partition manifest written: {man_path}", flush=True)
    except Exception as e:
        print(f"[os_orchestrator] Could not write partition manifest: {e}", flush=True)

    child_sessions: List[str] = []
    part_files: List[str] = []
    procs: List[subprocess.Popen] = []
    active: List[Dict[str, Any]] = []
    next_idx = 0

    def _launch_shard(gpu_index: int, shard_idx: int) -> None:
        nonlocal child_sessions, part_files, procs, active
        file_part = parts[shard_idx]
        child = f"{session_name}-gpu{gpu_index}-shard{shard_idx}"
        pfile = f".os-gpu-partition-{child}.txt"
        write_partition_file(file_part, pfile)
        print(f"[os_orchestrator] child={child}, files={len(file_part)}, partition={pfile}", flush=True)
        proc = launch_worker(
            child_session=child,
            root_dir=os.path.abspath(root_dir),
            partition_file=pfile,
            embedding_model=embedding_model,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
            log_dir=str(log_root),
            resume=resume,
            gpu_index=gpu_index if num > 1 else None
        )
        child_sessions.append(child)
        part_files.append(pfile)
        procs.append(proc)
        active.append({"proc": proc, "gpu": gpu_index, "child": child})

    # Seed
    for gpu_i in range(min(num, len(parts))):
        _launch_shard(gpu_i, next_idx)
        next_idx += 1

    # Dynamic schedule
    exit_codes: List[int] = []
    while active:
        i = 0
        while i < len(active):
            item = active[i]
            proc = item["proc"]
            gpu_i = item["gpu"]
            if proc.poll() is None:
                i += 1
                continue
            exit_codes.append(proc.returncode if hasattr(proc, "returncode") else 0)
            active.pop(i)
            if next_idx < len(parts):
                _launch_shard(gpu_i, next_idx)
                next_idx += 1
        if active:
            time.sleep(0.5)

    summary = {
        "root": os.path.abspath(root_dir),
        "session": session_name,
        "child_sessions": child_sessions,
        "partition_files": part_files,
        "total_files": total_files,
        "gpus_used": num,
        "shards": len(parts),
        "detected_gpus": detected,
        "log_dir": str(log_root),
    }

    if not wait:
        # Restore index settings on early exit as well
        try:
            _restore_index_settings(adapter, restore_settings)
        except Exception:
            pass
        return summary

    # Wait for completion
    exit_codes = []
    for proc in procs:
        code = proc.wait()
        exit_codes.append(code)
    summary["exit_codes"] = exit_codes

    # Aggregate logs
    def _read_lines(path: Path) -> List[str]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    all_success: List[str] = []
    all_error: List[str] = []
    for child in child_sessions:
        succ = _read_lines(log_root / f"{child}.success.log")
        err = _read_lines(log_root / f"{child}.error.log")
        all_success.extend(succ)
        all_error.extend(err)

    all_success = list(dict.fromkeys(all_success))
    all_error = list(dict.fromkeys(all_error))

    master_success = log_root / f"{session_name}.success.log"
    master_error = log_root / f"{session_name}.error.log"

    end_iso = datetime.utcnow().isoformat() + "Z"
    duration_sec = int(time.time() - t_start)

    with master_success.open("w", encoding="utf-8") as f:
        f.write(f"# session={session_name}\n")
        f.write(f"# started_at={start_iso}\n")
        f.write(f"# ended_at={end_iso}\n")
        f.write(f"# duration_sec={duration_sec}\n")
        f.write(f"# child_sessions={len(child_sessions)}\n")
        f.write(f"# files_ok={len(all_success)}\n")
        f.write("# --- aggregated child success entries ---\n")
        for ln in all_success:
            f.write(ln + "\n")

    with master_error.open("w", encoding="utf-8") as f:
        f.write(f"# session={session_name}\n")
        f.write(f"# started_at={start_iso}\n")
        f.write(f"# ended_at={end_iso}\n")
        f.write(f"# duration_sec={duration_sec}\n")
        f.write(f"# child_sessions={len(child_sessions)}\n")
        f.write(f"# files_failed={len(all_error)}\n")
        f.write("# --- aggregated child error entries ---\n")
        for ln in all_error:
            f.write(ln + "\n")

    try:
        for child in child_sessions:
            c_succ = log_root / f"{child}.success.log"
            with c_succ.open("a", encoding="utf-8") as f:
                f.write(f"# master_summary session={session_name} files_ok={len(all_success)} files_failed={len(all_error)}\n")
                f.write(f"# master_success_log={master_success}\n")
                f.write(f"# master_error_log={master_error}\n")
    except Exception as e:
        print(f"[os_orchestrator] Note: failed to append master summary to child logs: {e}", flush=True)

    print(f"[os_orchestrator] Aggregated logs for session {session_name}", flush=True)
    print(f"[os_orchestrator] Success log: {master_success} (files: {len(all_success)})", flush=True)
    print(f"[os_orchestrator] Error log:   {master_error} (files: {len(all_error)})", flush=True)

    summary["aggregated_success_log"] = str(master_success)
    summary["aggregated_error_log"] = str(master_error)
    summary["success_count"] = len(all_success)
    summary["error_count"] = len(all_error)
    summary["started_at"] = start_iso
    summary["ended_at"] = end_iso
    summary["duration_sec"] = duration_sec

    # Restore index settings post-run
    try:
        _restore_index_settings(adapter, restore_settings)
    except Exception:
        pass

    return summary

def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    import argparse
    ap = argparse.ArgumentParser(description="Multi-GPU orchestrator for OpenSearch full ingestion (semantic chunking + embeddings + OS bulk).")
    ap.add_argument("--root", required=True, help="Root directory of the corpus")
    ap.add_argument("--session", required=True, help="Base session name (child sessions created as {session}-gpu<i>-shard<j>)")
    ap.add_argument("--model", default=None, help="Embedding model name (optional)")
    ap.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (0 = auto-detect)")
    ap.add_argument("--sample_per_folder", action="store_true", help="Use sample mode: one file per folder, skip year dirs (preview)")
    ap.add_argument("--no_skip_years_in_sample", action="store_true", help="In sample mode, do not skip year directories")
    ap.add_argument("--target_tokens", type=int, default=512, help="Chunk target tokens (default 512)")
    ap.add_argument("--overlap_tokens", type=int, default=64, help="Chunk overlap tokens (default 64)")
    ap.add_argument("--max_tokens", type=int, default=640, help="Hard max per chunk (default 640)")
    ap.add_argument("--log_dir", default="./logs", help="Directory to write per-worker and aggregated logs")
    ap.add_argument("--no_wait", action="store_true", help="Do not wait for workers; exit after launch (no aggregation)")
    ap.add_argument("--balance_by_size", action="store_true", help="Greedy size-balanced partitions across GPUs (better load balance)")
    ap.add_argument("--shards", type=int, default=0, help="Number of shards to split the file list into (0=auto: GPUs*4). Enables dynamic scheduling across GPUs.")
    ap.add_argument("--resume", action="store_true", help="Pass --resume to workers to skip files already present in *.success.log")
    return vars(ap.parse_args(argv))

if __name__ == "__main__":
    args = _parse_cli_args(sys.argv[1:])
    summary = orchestrate(
        root_dir=args["root"],
        session_name=args["session"],
        embedding_model=args.get("model"),
        num_gpus=args.get("gpus"),
        sample_per_folder=bool(args.get("sample_per_folder")),
        skip_year_dirs_in_sample=not bool(args.get("no_skip_years_in_sample")),
        target_tokens=args.get("target_tokens") or 512,
        overlap_tokens=args.get("overlap_tokens") or 64,
        max_tokens=args.get("max_tokens") or 640,
        log_dir=args.get("log_dir") or "./logs",
        resume=bool(args.get("resume")),
        shards=args.get("shards") or 0,
        balance_by_size=bool(args.get("balance_by_size")),
        wait=not bool(args.get("no_wait")),
    )
    print(json.dumps(summary, indent=2))
