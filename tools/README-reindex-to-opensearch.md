# CogNeo — OpenSearch Backfill and High-Throughput Tuning (OpenSearch 3.x Ready)

This document explains how to backfill an OpenSearch serving index directly from the relational DB (Postgres or Oracle) and how to tune ingestion for very large corpora (millions of chunks). It also documents all environment variables controlling behavior, including OpenSearch 3.x compatibility changes.

Contents
- Overview
- OpenSearch 3.x compatibility
- Mapping and field explosion avoidance
- Environment variables (complete reference)
- High-throughput reindexing guide
- Multi-process partitioning (horizontal scale-out)
- Verification and troubleshooting
- Safety notes and compatibility with Postgres/Oracle backends

Overview
- Goal: Populate a serving OpenSearch index with parity fields (doc_id, chunk_index, citation, text, source, format, chunk_metadata, vector) while the app continues to serve retrieval from DB or OS as configured.
- System of record remains the database. This tool and optional dual-write mirror data into OpenSearch for retrieval workloads.
- Script entrypoint:
  - python3 -m tools.reindex_to_opensearch

OpenSearch 3.x compatibility
- Engine change: nmslib is deprecated for new index creation on OpenSearch 3.x.
  - Default engine is now lucene (configurable).
- Mapping fallback: Some clusters may not support the flattened type.
  - The adapter first attempts a mapping with chunk_metadata: flattened.
  - If unsupported, it falls back to chunk_metadata: {type: object, enabled: false} plus a single text field chunk_metadata_text. The adapter auto-generates this text at index time.
- No field explosion: Both mapping modes avoid creating per-key subfields under chunk_metadata.

Mapping and field explosion avoidance
- Preferred mapping (if supported by your cluster):
  - chunk_metadata: flattened
  - chunk_metadata_text: text (kept for compatibility and query convenience)
- Fallback mapping:
  - chunk_metadata: object, enabled: false (not indexed -> avoids dynamic keys)
  - chunk_metadata_text: text (single field used for metadata search)
- Full-text search queries adapt automatically:
  - If flattened is active: FTS queries metadata via ["chunk_metadata", "chunk_metadata.*"].
  - If fallback: FTS queries ["chunk_metadata_text"] only.

Environment variables (complete reference)
- Core OpenSearch target
  - OPENSEARCH_HOST: http(s)://host:9200
  - OPENSEARCH_INDEX: serving index name (e.g., cogneo_chunks_v2)
  - OPENSEARCH_NUMBER_OF_SHARDS / OPENSEARCH_NUMBER_OF_REPLICAS: applied on first creation only
  - OPENSEARCH_FORCE_RECREATE: 1 to drop and recreate the index on startup (danger: data loss) to apply new shard settings
  - OPENSEARCH_ENFORCE_SHARDS: 1 to fail-fast if templates/defaults override requested shard/replica counts
  - OPENSEARCH_VERIFY_CERTS: TLS validation (1=on)
  - OPENSEARCH_TIMEOUT, OPENSEARCH_MAX_RETRIES: increase for managed/high-latency clusters

- OpenSearch 3.x k-NN configuration
  - OPENSEARCH_KNN_ENGINE: lucene (default, 3.x-safe) | faiss | nmslib (deprecated on 3.x)
  - OPENSEARCH_KNN_SPACE: cosinesimil (default) | l2 | innerproduct
  - OPENSEARCH_KNN_METHOD: hnsw (default)

- Bulk ingestion sizing and concurrency
  - OPENSEARCH_BULK_CHUNK_SIZE: docs per HTTP bulk sub-request (typical 300–1200)
  - OPENSEARCH_BULK_MAX_BYTES: max bytes per bulk request (default 104857600 = 100MB)
  - OPENSEARCH_BULK_CONCURRENCY: worker threads (parallel_bulk). The adapter uses:
    - effective_concurrency = min(OPENSEARCH_BULK_CONCURRENCY, primary_shards * OPENSEARCH_CONCURRENCY_OVERSUB)
  - OPENSEARCH_BULK_QUEUE_SIZE: producer queue feeding parallel_bulk worker threads (default 8; consider 12–32 for LB/managed)
  - OPENSEARCH_CONCURRENCY_OVERSUB: modest oversubscription factor relative to shard count (default 1; try 2 if queues are low)
  - OPENSEARCH_HTTP_COMPRESS: 1 enables HTTP compression to reduce wire size on managed/LB endpoints

- Reindex tool streaming and tuning
  - REINDEX_BATCH_SIZE: documents per flush to OpenSearch (default 2000; consider 3000–5000 if memory allows)
  - REINDEX_YIELD_PER: ORM .yield_per() streaming window (default=max(REINDEX_BATCH_SIZE, 3000))
  - REINDEX_PARTITIONS / REINDEX_PARTITION_ID: partition dataset by doc_id modulo for multi-process scale-out
  - OPENSEARCH_TUNE_INDEX: 1 to temporarily set refresh_interval=-1 and number_of_replicas=0 before bulk and restore afterward

High-throughput reindexing guide
- One-time index creation with tuned settings
  - In .env:
    - OPENSEARCH_KNN_ENGINE=lucene
    - OPENSEARCH_NUMBER_OF_SHARDS=5 (or more, if you need more parallelism)
    - OPENSEARCH_NUMBER_OF_REPLICAS=1 (use 0 during bulk for maximum write speed; restore to 1 afterward)
    - OPENSEARCH_FORCE_RECREATE=1 (on first run to apply shard/replica settings)
    - OPENSEARCH_ENFORCE_SHARDS=1
  - For throughput (tune incrementally):
    - OPENSEARCH_BULK_CHUNK_SIZE=600 (raise slowly to 800–1200 if cluster/LB permit)
    - OPENSEARCH_BULK_CONCURRENCY=4–5 (not above shard count unless OPENSEARCH_CONCURRENCY_OVERSUB>1)
    - OPENSEARCH_BULK_QUEUE_SIZE=16
    - OPENSEARCH_CONCURRENCY_OVERSUB=1 (or 2 if threadpools remain underutilized)
    - OPENSEARCH_HTTP_COMPRESS=1
    - OPENSEARCH_TIMEOUT=120, OPENSEARCH_MAX_RETRIES=8
  - Reindex streaming:
    - REINDEX_BATCH_SIZE=3000 (or 5000 if memory allows)
    - REINDEX_YIELD_PER=5000
    - OPENSEARCH_TUNE_INDEX=1 (the tool will restore original settings after bulk)

- Start the reindex
  - set -a; source .env; set +a
  - python3 -m tools.reindex_to_opensearch
  - Logs to expect:
    - “indices.create ack=… shards_ack=…”
    - “created index … effective: number_of_shards=5”
    - Frequent “[reindex] flushed batch size=… total_indexed=…”
    - “bulk params: … effective_concurrency=… shard_count=… queue_size=… oversub=…”

- Horizontal scaling (multi-process)
  - Partition by doc_id modulo:
    - REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=0 python3 -m tools.reindex_to_opensearch
    - …
    - REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=3 python3 -m tools.reindex_to_opensearch
  - Keep per-process OPENSEARCH_BULK_CONCURRENCY moderate (2–4); scale throughput via more processes.

Multi-process partitioning (horizontal scale-out)
- Variables:
  - REINDEX_PARTITIONS=N: total partitions
  - REINDEX_PARTITION_ID=K: 0..N-1 for each process
- Strategy:
  - The query filters where doc_id % N == K, enabling independent processes to safely backfill different slices.
  - You can run these on the same machine or multiple machines pointing to the same DB and OpenSearch.

Examples (same machine)
- 4-process fan-out in 4 terminals (or background with &):
  # Terminal 1
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=0 REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 python3 -m tools.reindex_to_opensearch
  # Terminal 2
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=1 REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 python3 -m tools.reindex_to_opensearch
  # Terminal 3
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=2 REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 python3 -m tools.reindex_to_opensearch
  # Terminal 4
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=3 REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 python3 -m tools.reindex_to_opensearch

- Background on same machine (start all, then tail logs):
  for k in 0 1 2 3; do \
    REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=$k REINDEX_BATCH_SIZE=3000 REINDEX_YIELD_PER=5000 OPENSEARCH_TUNE_INDEX=1 OPENSEARCH_BULK_CONCURRENCY=4 OPENSEARCH_BULK_QUEUE_SIZE=16 python3 -m tools.reindex_to_opensearch > reindex_$k.log 2>&1 & \
  done
  # Inspect progress
  tail -F reindex_*.log

Examples (multiple machines)
- Ensure all hosts point to the same DB and OPENSEARCH_* target. Launch one partition per host:
  # Host A
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=0 ... python3 -m tools.reindex_to_opensearch
  # Host B
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=1 ... python3 -m tools.reindex_to_opensearch
  # Host C
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=2 ... python3 -m tools.reindex_to_opensearch
  # Host D
  REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=3 ... python3 -m tools.reindex_to_opensearch

What each flag does
- REINDEX_PARTITIONS: total slices of the corpus (by doc_id modulo). All processes must agree on the same N.
- REINDEX_PARTITION_ID: which slice this process indexes (0..N-1). Each process must use a unique K.
- REINDEX_BATCH_SIZE: number of documents per HTTP flush to OpenSearch; increase to reduce Python/HTTP overhead.
- REINDEX_YIELD_PER: ORM streaming window (rows fetched per DB roundtrip); increase to reduce DB overhead.
- OPENSEARCH_TUNE_INDEX: when 1, temporarily sets refresh_interval=-1 and replicas=0 before bulk and restores afterward.
- OPENSEARCH_BULK_CONCURRENCY: threads per process for parallel_bulk; keep modest (2–4) when running multiple processes.
- OPENSEARCH_BULK_QUEUE_SIZE: producer queue feeding parallel_bulk threads; raise when latency is high so workers stay fed.

Notes and recommendations
- Keep per-process OPENSEARCH_BULK_CONCURRENCY moderate; scale total throughput primarily by adding processes (REINDEX_PARTITIONS/ID).
- Consider temporarily setting number_of_replicas=0 for the target index (via OPENSEARCH_TUNE_INDEX=1 or admin CLI) during the bulk and restore afterward.
- Monitor OpenSearch bulk/threadpool metrics for 429 (too_many_requests) and queue saturation; adjust CHUNK_SIZE, CONCURRENCY, and QUEUE_SIZE accordingly.
- Ensure OPENSEARCH_ENFORCE_SHARDS=1 so the tool fails fast if a template overrides your desired shard/replica settings.

Verification and troubleshooting
- Mapping and settings
  - curl -u "$OPENSEARCH_USER:$OPENSEARCH_PASS" -k "$OPENSEARCH_HOST/$OPENSEARCH_INDEX/_mapping?pretty"
  - curl -u "$OPENSEARCH_USER:$OPENSEARCH_PASS" -k "$OPENSEARCH_HOST/$OPENSEARCH_INDEX/_settings?pretty"
  - Expect:
    - Either flattened mode:
      - chunk_metadata: flattened and chunk_metadata_text: text
    - Or fallback mode (no flattened):
      - chunk_metadata: object, enabled: false and chunk_metadata_text: text
- Concurrency slow at 1 thread
  - Means your effective primary shards=1. Recreate with OPENSEARCH_FORCE_RECREATE=1 (danger: drops data) or choose a new OPENSEARCH_INDEX with higher shard count.
  - The tool now aborts if effective shards != desired shards to avoid implicit 1-shard slow scenarios.
- Managed clusters and LBs
  - Increase OPENSEARCH_TIMEOUT, OPENSEARCH_MAX_RETRIES, and OPENSEARCH_BULK_QUEUE_SIZE.
  - Use HTTP compression (OPENSEARCH_HTTP_COMPRESS=1).
  - Consider temporarily setting number_of_replicas=0 during bulk and restore afterward (via OPENSEARCH_TUNE_INDEX or admin operation).
- 413/429/gateway timeouts
  - Lower OPENSEARCH_BULK_CHUNK_SIZE or OPENSEARCH_BULK_MAX_BYTES.
  - Keep OPENSEARCH_BULK_CONCURRENCY moderate.
- Dual-write path
  - DB remains system of record; dual-write logs warnings on failures without blocking inserts.
  - Ensure target OpenSearch index exists with proper mapping before enabling.

Safety notes and cross-backend compatibility
- Postgres/Oracle backends unaffected:
  - The OpenSearch adapter changes are isolated. Postgres/Oracle retrieval and hybrid logic remain the same.
- Retrieval features:
  - Vector (KNN), BM25, FTS, and hybrid search work as before.
  - With fallback mapping, FTS uses chunk_metadata_text automatically; BM25 and vector queries are unchanged.

References
- Storage adapter configuration and behavior: storage/adapters/README.md
- Ingestion pipeline details: ingest/README.md
- Root project configuration: README.md
