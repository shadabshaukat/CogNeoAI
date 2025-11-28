# CogNeo — OpenSearch Backfill and High‑Throughput Tuning (OpenSearch 3.x)

This guide covers how to backfill an OpenSearch serving index from your relational DB (Postgres or Oracle) and how to tune ingestion for large corpora. It also documents all relevant environment variables and provides ready‑to‑run commands.

Contents
- Quick start
- Prerequisites
- Environment variables (reference)
- Create/tune the index
- Single‑process reindex (fast path)
- Multi‑process reindex (scale‑out)
- What each flag does
- Monitoring, verification, and troubleshooting
- Safety notes and backend compatibility

## Quick start

```bash
# Load your environment
set -a; source .env; set +a

# One‑time index creation with tuned settings (optional: adjust to your shard/replica needs)
export OPENSEARCH_FORCE_RECREATE=1
export OPENSEARCH_ENFORCE_SHARDS=1
export OPENSEARCH_NUMBER_OF_SHARDS=5
export OPENSEARCH_NUMBER_OF_REPLICAS=1

# Throughput knobs (adjust incrementally)
export OPENSEARCH_BULK_CHUNK_SIZE=600
export OPENSEARCH_BULK_CONCURRENCY=4
export OPENSEARCH_BULK_QUEUE_SIZE=16
export OPENSEARCH_TIMEOUT=120
export OPENSEARCH_MAX_RETRIES=8
export OPENSEARCH_HTTP_COMPRESS=1

# Reindex streaming/batching
export REINDEX_BATCH_SIZE=3000
export REINDEX_YIELD_PER=5000
export OPENSEARCH_TUNE_INDEX=1   # temporarily set refresh_interval=-1 and replicas=0; restored after

# Run the reindex
python3 -m tools.reindex_to_opensearch

# IMPORTANT: after first create, disable FORCE_RECREATE
export OPENSEARCH_FORCE_RECREATE=0
```

## Prerequisites

- DB backend reachable and configured in `.env` (Postgres or Oracle).
- OpenSearch 2.x/3.x endpoint (managed or self‑hosted) and credentials configured in `.env`.
- The reindex tool reads from your DB and writes to OpenSearch; the DB remains the system of record.

Script entrypoint:
```bash
python3 -m tools.reindex_to_opensearch
```

## Environment variables (reference)

Core target
- OPENSEARCH_HOST: http(s)://host:9200 (no trailing slash)
- OPENSEARCH_INDEX: serving index name (e.g., cogneo_chunks_v2)
- OPENSEARCH_USER / OPENSEARCH_PASS: basic auth if required
- OPENSEARCH_VERIFY_CERTS: 1 to verify TLS certs (recommended)

Index shape and creation
- OPENSEARCH_NUMBER_OF_SHARDS, OPENSEARCH_NUMBER_OF_REPLICAS: applied on first create only
- OPENSEARCH_FORCE_RECREATE: 1 to drop and recreate index (danger: data loss) to apply shard changes
- OPENSEARCH_ENFORCE_SHARDS: 1 to fail‑fast if templates/defaults override requested shards/replicas

k‑NN/vector config (3.x‑safe)
- OPENSEARCH_KNN_ENGINE: lucene (default) | faiss | nmslib (deprecated in 3.x)
- OPENSEARCH_KNN_SPACE: cosinesimil (default) | l2 | innerproduct
- OPENSEARCH_KNN_METHOD: hnsw (default)
- COGNEO_EMBED_DIM: must match embedder dimension (e.g., 768)

HTTP and retry behavior
- OPENSEARCH_TIMEOUT: request timeout (suggest 60–120 for managed/LB endpoints)
- OPENSEARCH_MAX_RETRIES: retry count for transient timeouts
- OPENSEARCH_HTTP_COMPRESS: 1 to enable HTTP compression (recommended for LBs)

Bulk sizing and concurrency
- OPENSEARCH_BULK_CHUNK_SIZE: docs per bulk sub‑request (typical 300–1200)
- OPENSEARCH_BULK_MAX_BYTES: max bytes per sub‑request (default 104857600 = 100MB)
- OPENSEARCH_BULK_CONCURRENCY: parallel_bulk threads per process (2–5 typical)
- OPENSEARCH_BULK_QUEUE_SIZE: producer queue feeding worker threads (8–32)

Reindex streaming and tuning
- REINDEX_BATCH_SIZE: docs per flush to OpenSearch (e.g., 2000–5000)
- REINDEX_YIELD_PER: ORM `.yield_per()` window (use ≥ batch size; e.g., 3000–5000)
- OPENSEARCH_TUNE_INDEX: 1 to temporarily set refresh_interval=-1 and replicas=0, restored after

## Create/tune the index

One‑time creation (first run) with explicit shards/replicas:
```bash
set -a; source .env; set +a

export OPENSEARCH_FORCE_RECREATE=1
export OPENSEARCH_ENFORCE_SHARDS=1
export OPENSEARCH_NUMBER_OF_SHARDS=5
export OPENSEARCH_NUMBER_OF_REPLICAS=1

python3 -m tools.reindex_to_opensearch

# Disable recreate to avoid accidental drops
export OPENSEARCH_FORCE_RECREATE=0
```

Recommended throughput knobs (increase gradually while monitoring):
```bash
export OPENSEARCH_BULK_CHUNK_SIZE=600       # consider 800–1200 if stable
export OPENSEARCH_BULK_CONCURRENCY=4        # do not exceed shard count initially
export OPENSEARCH_BULK_QUEUE_SIZE=16
export OPENSEARCH_TIMEOUT=120
export OPENSEARCH_MAX_RETRIES=8
export OPENSEARCH_HTTP_COMPRESS=1
```

Reindex streaming/batching:
```bash
export REINDEX_BATCH_SIZE=3000
export REINDEX_YIELD_PER=5000
export OPENSEARCH_TUNE_INDEX=1
```

## Single‑process reindex (fast path)

```bash
set -a; source .env; set +a

export OPENSEARCH_BULK_CHUNK_SIZE=600
export OPENSEARCH_BULK_CONCURRENCY=4
export OPENSEARCH_BULK_QUEUE_SIZE=16
export REINDEX_BATCH_SIZE=3000
export REINDEX_YIELD_PER=5000
export OPENSEARCH_TUNE_INDEX=1

python3 -m tools.reindex_to_opensearch
```

Expected logs:
- `indices.create ack=… shards_ack=…` (first run only)
- `created index … effective: number_of_shards=5`
- `[reindex] flushed batch size=… total_indexed=…`
- `bulk params: … effective_concurrency=… shard_count=… queue_size=…`

## Multi‑process reindex (scale‑out)

Partition by `doc_id % REINDEX_PARTITIONS == REINDEX_PARTITION_ID`.

Same machine (4 processes):
```bash
# Terminal 1
REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=0 \
REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 \
OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 \
python3 -m tools.reindex_to_opensearch

# Terminal 2
REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=1 \
REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 \
OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 \
python3 -m tools.reindex_to_opensearch

# Terminal 3
REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=2 \
REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 \
OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 \
python3 -m tools.reindex_to_opensearch

# Terminal 4
REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=3 \
REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 \
OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 \
python3 -m tools.reindex_to_opensearch
```

Same machine (background + logs):
```bash
for k in 0 1 2 3; do
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=$k \
  REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 \
  OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 \
  python3 -m tools.reindex_to_opensearch > reindex_$k.log 2>&1 &
done

tail -F reindex_*.log
```

Multiple machines (one slice per host):
```bash
# Host A
REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=0 python3 -m tools.reindex_to_opensearch
# Host B
REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=1 python3 -m tools.reindex_to_opensearch
# Host C
REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=2 python3 -m tools.reindex_to_opensearch
# Host D
REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=3 python3 -m tools.reindex_to_opensearch
```

## What each flag does

- REINDEX_PARTITIONS: total slices of the corpus (N). All processes must use the same N.
- REINDEX_PARTITION_ID: slice ID for this process (0..N‑1). Each process must be unique.
- REINDEX_BATCH_SIZE: docs per flush to OpenSearch; bigger batches reduce Python/HTTP overhead.
- REINDEX_YIELD_PER: ORM streaming window; reduce DB roundtrips by keeping this ≥ batch size.
- OPENSEARCH_TUNE_INDEX: 1 to set `refresh_interval=-1` and `number_of_replicas=0` during bulk; restored afterward.
- OPENSEARCH_BULK_CONCURRENCY: parallel_bulk threads per process; keep modest (2–5), scale with more processes.
- OPENSEARCH_BULK_QUEUE_SIZE: producer queue feeding threads; increase on high‑latency links so workers stay fed.
- OPENSEARCH_BULK_CHUNK_SIZE: docs per HTTP bulk sub‑request; raise gradually while watching for 413/429/timeouts.

## Monitoring, verification, and troubleshooting

Mapping and settings:
```bash
curl -u "$OPENSEARCH_USER:$OPENSEARCH_PASS" -k "$OPENSEARCH_HOST/$OPENSEARCH_INDEX/_mapping?pretty"
curl -u "$OPENSEARCH_USER:$OPENSEARCH_PASS" -k "$OPENSEARCH_HOST/$OPENSEARCH_INDEX/_settings?pretty"
```

Throughput/logs:
```bash
tail -F reindex_*.log
```

Common issues
- Concurrency stuck at 1:
  - Effective primary shards are 1. Recreate with more shards (danger: drops data) or use a new index name.
- Managed clusters/LB:
  - Increase `OPENSEARCH_TIMEOUT`, `OPENSEARCH_MAX_RETRIES`, `OPENSEARCH_BULK_QUEUE_SIZE`; keep `OPENSEARCH_HTTP_COMPRESS=1`.
- 413/429/gateway timeouts:
  - Lower `OPENSEARCH_BULK_CHUNK_SIZE` or `OPENSEARCH_BULK_MAX_BYTES`; keep concurrency moderate.

## Safety notes and backend compatibility

- DB (Postgres/Oracle) remains the system of record; reindex mirrors into OpenSearch.
- Retrieval parity:
  - Vector (KNN), BM25, FTS, and hybrid search are supported in the OpenSearch adapter.
  - Metadata mapping is resilient:
    - Preferred: `flattened`
    - Fallback: a single `chunk_metadata_text` field (auto‑generated by the adapter) for clusters without `flattened` support
- Dual‑write ingestion is optional and best‑effort; DB insert is never blocked by OpenSearch failures.
