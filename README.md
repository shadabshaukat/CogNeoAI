# CogNeo — Agentic Multi-Faceted Legal AI Platform (Ollama ⬄ OCI GenAI ⬄ Oracle Database 26ai)

---

## Quick Deployment Guide

1) System prerequisites
```sh
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv python3-pip git postgresql libpq-dev gcc unzip curl -y
```

2) Clone, prepare, and install dependencies
```sh
git clone https://github.com/shadabshaukat/cogneo.git
cd cogneo
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Notes:
- Requirements include oci and oracledb for full Oracle GenAI and Oracle Database 26ai DB coverage.
- pgvector must be installed/enabled on your PostgreSQL target.

3) Configure environment variables

Database (Postgres):
```sh
export COGNEO_DB_HOST=localhost
export COGNEO_DB_PORT=5432
export COGNEO_DB_USER=postgres
export COGNEO_DB_PASSWORD='YourPasswordHere'
export COGNEO_DB_NAME=postgres
# Optional single DSN override:
# export COGNEO_DB_URL='postgresql+psycopg2://user:pass@host:5432/dbname'
```

API/Backend:
```sh
export FASTAPI_API_USER=legal_api
export FASTAPI_API_PASS=letmein
export COGNEO_API_URL=http://localhost:8000
```

Oracle Cloud GenAI (optional):
```sh
export OCI_USER_OCID='ocid1.user.oc1...'
export OCI_TENANCY_OCID='ocid1.tenancy.oc1...'
export OCI_KEY_FILE='/path/to/oci_api_key.pem'
export OCI_KEY_FINGERPRINT='xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx'
export OCI_REGION='ap-sydney-1'
export OCI_COMPARTMENT_OCID='ocid1.compartment.oc1...'
export OCI_GENAI_MODEL_OCID='ocid1.generativeaiocid...'
```

Oracle Database 26ai DB integration (optional):
```sh
export ORACLE_DB_USER='your_db_user'
export ORACLE_DB_PASSWORD='your_db_password'
export ORACLE_DB_DSN='your_db_high'
export ORACLE_WALLET_LOCATION='/path/to/wallet/dir'
```

Backend switch (Postgres default, Oracle optional):
```sh
# Default backend remains Postgres
export COGNEO_DB_BACKEND=postgres

# To use Oracle Database 26ai backend:
# Either provide a single SQLAlchemy DSN:
# export ORACLE_SQLALCHEMY_URL='oracle+oracledb://user:pass@myadb_high'
# Or set individual fields:
# export ORACLE_DB_USER='...'
# export ORACLE_DB_PASSWORD='...'
# export ORACLE_DB_DSN='myadb_high'
# export ORACLE_WALLET_LOCATION='/path/to/wallet/dir'    # if using an Autonomous DB wallet
```

Embedding model (defaults are sensible):
```sh
export COGNEO_EMBED_MODEL='nomic-ai/nomic-embed-text-v1.5'
export COGNEO_EMBED_DIM=768
export COGNEO_EMBED_BATCH=64
```

Domain selection (future‑ready):
```sh
export COGNEO_DOMAIN=legal   # legal | healthcare | telecom | finance | ...
```

4) Network and ports
- Open: 8000 (FastAPI), 7866–7879 (Gradio), 8501 (Streamlit).
```sh
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 7866:7879 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
```

5) Launch the stack
```sh
bash run_cogneo_stack.sh
# To stop:
bash stop_cogneo_stack.sh
# Gradio:   http://localhost:7866
# FastAPI:  http://localhost:8000
# Streamlit http://localhost:8501
```

For production, secure endpoints behind WAF/reverse proxy and TLS. Store secrets in environment or a secret manager.

---

## Documentation Index

- Ingestion (Beta) Pipeline: [ingest/README.md](ingest/README.md)
- Embedding Subsystem: [embedding/README.md](embedding/README.md)
- Database Layer (Postgres + pgvector + FTS + Oracle Database 26ai connector): [db/README.md](db/README.md)
- Multi‑Domain Readiness: [docs/MULTI_DOMAIN_READINESS.md](docs/MULTI_DOMAIN_READINESS.md)
- RAG Pipelines (Ollama and OCI GenAI): [rag/README.md](rag/README.md)
- Streamlit UI (Login + Chat): [pages/README.md](pages/README.md)
- Tools: SQL Latency Benchmark (p50/p95, vector/FTS/metadata, and optimized SQL scenarios): [tools/README-bench-sql-latency.md](tools/README-bench-sql-latency.md)
- Tools: Delete by URL utility (single/bulk, with literal --show-sql): [tools/README-delete-url.md](tools/README-delete-url.md)
- Post-load Indexing & Metadata Strategy (TB-scale): [schema-post-load/README.md](schema-post-load/README.md)
- Optimized SQL templates (citation/name/title/source, ANN, grouping): [schema-post-load/optimized_sql.sql](schema-post-load/optimized_sql.sql)
- Beta Data Load Guide (end-to-end ingest runbook): [docs/BetaDataLoad.md](docs/BetaDataLoad.md)

Other helpful docs:
- Gradio API/UX: [docs/API_SPEC_GRADIO.md](docs/API_SPEC_GRADIO.md)
- Deployment notes (TLS, Nginx, Let’s Encrypt): [docs/setup_letsencrypt_streamlit_nginx.md](docs/setup_letsencrypt_streamlit_nginx.md), [docs/setup_letsencrypt_oracle_lb.md](docs/setup_letsencrypt_oracle_lb.md)
- AWS/OCI helpers: [docs/awscli_install_instructions.md](docs/awscli_install_instructions.md), [oci_models_debug.json](oci_models_debug.json)

---

## Platform Overview

- Agentic Chain-of-Thought RAG
  - Stepwise, explainable legal answers with Thought/Action/Evidence/Reasoning and Final Conclusion structure; auditable and reproducible.
- Model endpoints
  - Switch between local Ollama models and Oracle Cloud GenAI seamlessly.
- Retrieval modes
  - Vector (pgvector cosine), BM25-like (ILIKE), Hybrid, FTS (tsvector); metadata-aware filtering and chunk metadata search.
- Applications
  - FastAPI REST API for search, RAG, agentic chat, Oracle Database 26ai proxy.
  - Streamlit chat UI with hybrid retrieval and source cards.
  - Gradio UI for hybrid/vector/OCI GenAI demos.

### Main components

- Ingestion (Beta)
  - Multi-GPU orchestrator and per-GPU worker with CPU/GPU pipelining, token-aware semantic chunking (dashed-header aware), batched embedding, DB persistence, per-file metrics, and resumability.
- Embeddings
  - Sentence-Transformers primary; HuggingFace AutoModel fallback with mean pooling; configurable batch size and revisions; dimension must match DB.
- Database
  - PostgreSQL + pgvector, FTS maintenance via trigger; post-load DDLs surface md_* generated/expression columns for optimized filters; optional Oracle Database 26ai connector.
- RAG
  - Ollama and OCI GenAI pipelines format metadata-rich context and enforce legal-grade prompts; Agentic CoT endpoints exposed via FastAPI.
- UIs
  - Streamlit (login + chat) and Gradio (LLM/Cloud tabs); professional UX with progress, error handling, and citations.
- Tools
  - Benchmark utility for p50/p95 latency across vector/FTS/metadata, plus optimized SQL scenarios (citations, names, titles, sources, ANN + grouping).

---

## Architecture and Workflow

<img width="2548" height="2114" alt="AusLegalSearchv3" src="https://github.com/user-attachments/assets/d90a5d18-d769-44b9-a6b8-5e75ba47727c" />

### System Overview

```
                                              [ Route53 (DNS) ]
                                                      |
                                      auslegal.oraclecloudserver.com (or custom domain)
                                                      |
                                  +--------------------+--------------------+
                                  |                                         |
                          [OCI Public Load Balancer]                [Nginx (public VM, alt)]
                                  |                                         |
                                   -------- WAF (Web Application Firewall) ---
                                  |
                          [Backend on Ubuntu VM]  (Private IP: e.g. 10.150.1.82)
                                  |
    +----------+-----------+----------+----------------------------------------+
    |          |           |                                  |          |
[FastAPI :8000]  [Gradio :7866+]  [Streamlit :8501]    [PGVector/PostgreSQL] [Oracle Database 26ai]
    |          |           |                                  |          |
    |          |           +---Modern LLM/Cloud UI (Gradio tabs: Hybrid, Chat, OCI GenAI, Agentic)----+ 
    |          +---Multisource LLM-driven Chat, Hybrid Search, OCI GenAI---+                 
    +---REST API: ingestion, retrieval, RAG (Ollama and OCI), DB bridge, agentic reasoning----+                 
```

### Agentic RAG Chain-of-Thought Workflow

<img width="2000" height="12600" alt="image" src="https://github.com/user-attachments/assets/05148d8c-1327-47da-99ee-5c4e6421e7f8" />

---

## Ingestion Pipeline (Beta) — Quickstart

- Multi-GPU orchestrator launches one worker per GPU; each worker:
  - Parses (.txt/.html), semantic-chunks (with dashed-header support), embeds (GPU), writes to Postgres/pgvector.
  - Records per-file status in EmbeddingSessionFile for resumable ingestion.
  - Appends per-file metrics and timings (parse_ms, chunk_ms, embed_ms, insert_ms) to success logs.

Environment variables (core)
```
# Postgres
COGNEO_DB_HOST=localhost
COGNEO_DB_PORT=5432
COGNEO_DB_USER=postgres
COGNEO_DB_PASSWORD='YourPasswordHere'
COGNEO_DB_NAME=postgres
# Optional: full DSN override
# COGNEO_DB_URL='postgresql+psycopg2://user:pass@host:5432/dbname'

# Embedding model and vector dimension
COGNEO_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
COGNEO_EMBED_DIM=768

# Worker timeouts and batch size
COGNEO_EMBED_BATCH=64
COGNEO_TIMEOUT_PARSE=30
COGNEO_TIMEOUT_CHUNK=60
COGNEO_TIMEOUT_EMBED_BATCH=180
COGNEO_TIMEOUT_INSERT=120

# Optional features
COGNEO_LOG_METRICS=1
# COGNEO_USE_RCTS_GENERIC=1
```

Production database pooling/timeouts (optional)
```
COGNEO_DB_POOL_SIZE=10
COGNEO_DB_MAX_OVERFLOW=20
COGNEO_DB_POOL_RECYCLE=1800
COGNEO_DB_POOL_TIMEOUT=30
# COGNEO_DB_STATEMENT_TIMEOUT_MS=60000
```

Run orchestrator (full dataset)
```sh
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --gpus 4 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```

Dynamic sharding and size balancing (reduces tail latency on skewed corpora)
```sh
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-sharded-$(date +%Y%m%d-%H%M%S)" \
  --gpus 4 --shards 16 --balance_by_size \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```
Notes:
- --shards splits the file list into many shards (default GPUs*4) and dynamically schedules them across GPUs to reduce stragglers.
- --balance_by_size greedily balances shards by total file size; orchestration auto-enables this when size skew is high.
- Per-worker file ordering by size is enabled by env: COGNEO_SORT_WORKER_FILES=1 (default). Set 0 to keep natural order.

Resume a stuck child on “remaining files” only
```sh
session=beta-full-YYYYMMDD-HHMMSS
child=${session}-gpu3
proj=/abs/path/cogneo
logs="$proj/logs"
part=".beta-gpu-partition-${child}.txt"

awk -F'\t' '{print $1}' "$logs/${child}.success.log" 2>/dev/null | sed '/^#/d' > /tmp/processed_g3.txt
cat "$logs/${child}.error.log" 2>/dev/null >> /tmp/processed_g3.txt
sort -u /tmp/processed_g3.txt -o /tmp/processed_g3.txt
sort -u "$part" -o /tmp/partition_g3.txt
comm -23 /tmp/partition_g3.txt /tmp/processed_g3.txt > "$proj/.beta-gpu-partition-${child}-remaining.txt"

CUDA_VISIBLE_DEVICES=3 \
python3 -m ingest.beta_worker ${child}-r1 \
  --partition_file "$proj/.beta-gpu-partition-${child}-remaining.txt" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "$logs"
```

---

## API Specification

### POST /chat/agentic — Agentic Chain-of-Thought LLM Endpoint

Request:
```json
{
  "llm_source": "ollama",
  "model": "llama3",
  "message": "Explain the procedure for contesting a will in NSW.",
  "chat_history": [],
  "system_prompt": "...",
  "temperature": 0.15,
  "top_p": 0.9,
  "max_tokens": 1920,
  "repeat_penalty": 1.07,
  "top_k": 12,
  "oci_config": {"compartment_id": "...", "model_id": "...", "region": "ap-sydney-1"}
}
```

Response:
```json
{
  "answer": "Step 1 - Thought: ...\nStep 2 - Action: ...\nFinal Conclusion: ...",
  "sources": ["Succession Act 2006 (NSW) s 96", "Practical Law ..."],
  "chunk_metadata": [{"citation": "Succession Act 2006 (NSW) s 96", "url": "https://..."}],
  "context_chunks": ["Section 96 of the Act provides that...", "..."]
}
```

Notes:
- Each CoT step is prefixed as "Step X - [Label]:" to enable reliable parsing and UI rendering.
- Sources/citations and context are aligned to reasoning steps for legal traceability.

---

## Security and Operations

- Always secure endpoints behind WAF/proxy and TLS.
- Store credentials outside the repo.
- Rotate secrets; follow enterprise policy and auditing standards.

---

## Contribution and Support

Raise issues, feature requests, or PRs at:  
https://github.com/shadabshaukat/cogneo

---

## Startup Banner and URLs

- The run script now prints a rainbow ASCII CogNeo AI banner and the URLs for all three web stacks.
- You can set COGNEO_HOST_DISPLAY in your environment (or .env) to control the host part in the printed URLs (default: localhost).

Example output (after bash run_cogneo_stack.sh):
- FastAPI:  http://localhost:8000/health
- Gradio:   http://localhost:7866
- Streamlit: http://localhost:8501

Stop all services:
- bash stop_cogneo_stack.sh

---

## New: Vector Backend — OpenSearch (KNN)

You can serve retrieval from OpenSearch while keeping Postgres/Oracle as the system of record.

Environment (.env):
- COGNEO_VECTOR_BACKEND=opensearch
- OPENSEARCH_HOST=http://localhost:9200
- OPENSEARCH_INDEX=cogneo_chunks
- Optional: OPENSEARCH_USER and OPENSEARCH_PASS
- Ensure COGNEO_EMBED_DIM matches your embedding model dimension (e.g., 768)

Backfill from DB to OpenSearch index:
- python tools/reindex_to_opensearch.py
  - Streams Document + Embedding rows from the DB and indexes to OpenSearch via the new adapter.

Notes:
- Registry is lazy-loaded; OpenSearch is imported only when selected.
- Hybrid/BM25/FTS are provided by the OpenSearch adapter. Vector KNN uses HNSW (cosine).

---

## New: LLM Backend — AWS Bedrock

In addition to Ollama and OCI GenAI, Bedrock is available via boto3.

Environment (.env):
- AWS_REGION=us-east-1
- BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

Credentials (pick one):
- AWS CLI default profile (aws configure) -> ~/.aws/credentials
- Environment: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY [/ AWS_SESSION_TOKEN]
- IAM role (EC2/ECS/EKS) with bedrock:InvokeModel

API usage:
- POST /search/bedrock_rag
- Chat endpoints: use "llm_source": "bedrock"

---

## Enterprise Controls (Env-Gated)

Security & Auth:
- COGNEO_AUTH_MODE=basic|jwt
- FASTAPI_API_USER, FASTAPI_API_PASS
- JWT settings (when using jwt):
  - COGNEO_JWT_SECRET, COGNEO_JWT_ALG, COGNEO_JWT_EXPIRE_MIN
- Token endpoint: POST /auth/token (returns Bearer token)  

CORS:
- COGNEO_CORS_ENABLE, COGNEO_CORS_ALLOW_ORIGINS, COGNEO_CORS_ALLOW_METHODS, COGNEO_CORS_ALLOW_HEADERS, COGNEO_CORS_ALLOW_CREDENTIALS

Rate Limiting (simple, per client IP):
- COGNEO_RATE_LIMIT_ENABLE, COGNEO_RATE_LIMIT_REQUESTS, COGNEO_RATE_LIMIT_WINDOW_S

Observability:
- COGNEO_PROMETHEUS_ENABLE (exposes /metrics via starlette-exporter)

All of the above are optional and default-safe for local dev.

---

## Ingestion Notes (Plain Folders, No Metadata)

- Point ingestion at any folder of .txt and/or .html. Dashed headers/metadata are optional.
- The pipeline will:
  - parse (parse_txt/parse_html),
  - chunk (semantic with fallbacks),
  - embed,
  - persist to DB.

Helpful toggles in .env:
- COGNEO_USE_RCTS_GENERIC=1
- COGNEO_FALLBACK_CHARS_PER_CHUNK=4000
- COGNEO_FALLBACK_OVERLAP_CHARS=200
- Optional: COGNEO_FALLBACK_CHUNK_ON_TIMEOUT=1

FastAPI ingestion endpoints:
- POST /ingest/start (body: {"directory": "/abs/path", "session_name": "ingest-YYYYMMDD"})
- GET /ingest/sessions
- POST /ingest/stop?session_name=...

Authentication:
- Basic (default) using FASTAPI_API_USER/PASS or JWT (see Enterprise Controls).

---

## Backend Selection Matrix

- Default: Postgres + pgvector
  - COGNEO_DB_BACKEND=postgres
- Oracle Database 26ai:
  - COGNEO_DB_BACKEND=oracle (+ ORACLE_* variables)
- OpenSearch for serving:
  - COGNEO_VECTOR_BACKEND=opensearch (overrides DB backend for retrieval layer)
  - Use tools/reindex_to_opensearch.py to populate the serving index from DB

---

**CogNeo — Enterprise-grade, agentic, explainable legal AI built for the modern legal practice.**
