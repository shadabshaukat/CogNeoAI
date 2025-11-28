# CogNeo v3 — Tools Index

This directory contains developer tools for CogNeo v3.

Contents
- SQL Latency Benchmark (p50/p95; vector/FTS/metadata; optimized SQL scenarios)
  - Readme: README-bench-sql-latency.md
  - Script: bench_sql_latency.py
- Delete by URL Utility (single and bulk; literal --show-sql; safe orphan document deletion)
  - Readme: README-delete-url.md
  - Script: delete_url_records.py
- OpenSearch Backfill (one-off, non-disruptive reindex from DB to OpenSearch)
  - Script: reindex_to_opensearch.py
  - Purpose: populate a serving OpenSearch index with parity fields (doc_id, chunk_index, citation, text, chunk_metadata, vector) from the relational DB (Postgres/Oracle) without changing how the app serves retrieval today.

Setup
- All tools inherit configuration via the unified .env loading in db/store.py and connectors (db/connector.py).
- Recommended shell prep:
  ```bash
  set -a; source .env; set +a
  ```
- For details and examples, see each tool’s README where applicable.

OpenSearch Backfill (tools/reindex_to_opensearch.py)
- Goal: Populate an OpenSearch index from the relational DB (Postgres/Oracle) as a one-off operation, while the app can continue to serve retrieval from the DB.
- Key properties:
  - Does NOT switch the serving adapter. The script instantiates the OpenSearch adapter directly.
  - Streams Document + Embedding rows from the active DB backend (db/store.py chooses Postgres or Oracle via COGNEO_DB_BACKEND).
  - Writes parity fields for retrieval consistency:
    - doc_id, chunk_index, citation, text, source, format, chunk_metadata, vector
    - Stable _id format: "{doc_id}#{chunk_index}"
- Environment requirements:
  - Database (source of truth)
    - COGNEO_DB_BACKEND=postgres | oracle
    - DB variables as per your backend (see root README and db/README.md)
  - OpenSearch (target serving index)
    - OPENSEARCH_HOST=http(s)://host:9200
    - OPENSEARCH_INDEX=cogneo_chunks_v2          # recommended new name if upgrading mapping
    - OPENSEARCH_USER / OPENSEARCH_PASS          # if required
    - Optional first-time index settings (adapter-created only; ignored if index already exists):
      - OPENSEARCH_NUMBER_OF_SHARDS
      - OPENSEARCH_NUMBER_OF_REPLICAS
    - COGNEO_EMBED_DIM must match embedding dimension (e.g., 768)
- Usage:
  ```bash
  # Ensure env is loaded as above
  source .env
  python3 -m tools.reindex_to_opensearch
  ```

Performance on multi-shard clusters
- Adapter parallel bulk (single-process, multithreaded)
  - Controls: OPENSEARCH_BULK_CONCURRENCY (threads), OPENSEARCH_BULK_CHUNK_SIZE (docs), OPENSEARCH_BULK_MAX_BYTES (bytes)
  - Recommendations:
    - Start with OPENSEARCH_BULK_CONCURRENCY=2–4 on multi-shard indices
    - Keep chunk size moderate (300–1000) as you raise concurrency to avoid overwhelming coordinator/bulk threadpools
    - Increase OPENSEARCH_TIMEOUT (e.g., 60–120) and OPENSEARCH_MAX_RETRIES (e.g., 5–8) for high-latency managed clusters
    - For very heavy one-off loads, consider temporarily setting index replicas=0 and restoring after the load (manual cluster op)

- Multi-process partitioning (horizontal scaling)
  - Environment variables:
    - REINDEX_PARTITIONS=N            # total partitions
    - REINDEX_PARTITION_ID=K          # 0-based partition id for this process
    - REINDEX_BATCH_SIZE=500          # stream size from DB to OS per flush (optional)
  - Example (4 parallel processes on the same machine or different machines):
    ```bash
    # process 0
    REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=0 OPENSEARCH_BULK_CONCURRENCY=3 python3 -m tools.reindex_to_opensearch
    # process 1
    REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=1 OPENSEARCH_BULK_CONCURRENCY=3 python3 -m tools.reindex_to_opensearch
    # process 2
    REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=2 OPENSEARCH_BULK_CONCURRENCY=3 python3 -m tools.reindex_to_opensearch
    # process 3
    REINDEX_PARTITIONS=4 REINDEX_PARTITION_ID=3 OPENSEARCH_BULK_CONCURRENCY=3 python3 -m tools.reindex_to_opensearch
    ```
  - Under the hood, the tool partitions rows by doc_id % REINDEX_PARTITIONS == REINDEX_PARTITION_ID and streams each partition independently.
  - Tune REINDEX_BATCH_SIZE (DB stream group) alongside OPENSEARCH_BULK_CHUNK_SIZE/MAX_BYTES (HTTP bulk group) for best throughput.

- Checklist for high-throughput loads
  - Pre-size shards and replicas for target volume (first-time creation via OPENSEARCH_NUMBER_OF_SHARDS/REPLICAS)
  - Disable replicas temporarily if allowed, re-enable after (cluster setting)
  - Warm I/O and increase OS client timeout/retries for managed endpoints
  - Monitor OpenSearch bulk/threadpool queues and adjust concurrency/chunk sizes accordingly

- Migration tip:
  - If you previously created an OpenSearch index without doc_id/chunk_index/citation, create a new index (e.g., cogneo_chunks_v2) and run the backfill tool again.

Optional Dual-Write during Ingestion (env-gated)
- Purpose: Keep DB as system of record while also mirroring each ingested batch into OpenSearch in near-real-time.
- Control via .env:
  - COGNEO_VECTOR_DUAL_WRITE=1       # accepted truthy: 1, true, yes, os, opensearch
  - OPENSEARCH_HOST / OPENSEARCH_INDEX / OPENSEARCH_USER / OPENSEARCH_PASS
- Behavior:
  - The worker inserts rows into DB first (unchanged flow).
  - If dual-write is enabled, it best-effort indexes the same batch into OpenSearch with parity fields (doc_id + chunk_index).
  - OpenSearch errors are logged as warnings; ingestion continues.
- Notes:
  - Ensure the OpenSearch index exists and uses the updated mapping (with doc_id/chunk_index/citation).
  - See storage/adapters/README.md and the root README OpenSearch section for parity details.

Troubleshooting (Tools)
- Module import errors when running a tool:
  - Run from project root or add the repo root to PYTHONPATH:
    ```bash
    cd /path/to/cogneo
    python tools/reindex_to_opensearch.py
    ```
- OpenSearch mapping missing fields:
  - Create a fresh index (e.g., cogneo_chunks_v2) and re-run the backfill.
- SSL/Tunnel errors to OpenSearch:
  - Verify your endpoint is reachable with curl.
  - For tunnels, ensure forwarding is allowed on the bastion and that your OPENSEARCH_* settings match the endpoint (scheme, auth, certs).
- DB read errors:
  - Confirm DB connectivity and credentials in .env and that COGNEO_DB_BACKEND matches your deployment.

References
- Root capabilities and configuration: README.md
- Adapter parity and behaviors (Postgres | Oracle | OpenSearch): storage/adapters/README.md
- Ingestion pipeline details and performance guidance: ingest/README.md
