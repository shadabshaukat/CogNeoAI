# CogNeo — Agentic Multi-Faceted AI Platform (Ollama ⬄ OCI GenAI ⬄ AWS Bedrock ⬄ Oracle Database 26ai ⬄ PostgreSQL)

```sh
 ████████╗ ██████╗  ██████╗ ███╗  ██╗██████   ██████╗       █████╗   ██████╗ 
██╔══════╝██╔═══██╗██╔════╝ ███╗  ██║██╔════╝██╔═══██╗     ██╔══██╗  ╚═██╔═╝ 
██║       ██║   ██║██║  ███╗█╔██╗ ██║█████╗  ██║   ██║     ███████║    ██║   
██║       ██║   ██║██║   ██║█║╚██╗██║██╔══╝  ██║   ██║     ██╔══██║    ██║   
╚███████╗ ╚██████╔╝╚██████╔╝█║ ╚████║███████╗╚██████╔╝     ██║  ██║  ██████║
  ╚═════╝  ╚═════╝  ╚═════╝ ═╝  ╚═══╝╚══════╝ ╚═════╝      ╚═╝  ╚═╝  ╚═════╝ 
```

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

<img width="1523" height="882" alt="CogNeoAI" src="https://github.com/user-attachments/assets/d8ee6b1e-403a-4329-b62e-405d068319d4" />


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

Optional tuning (if desired) :
- Increase shard count for large corpora to improve balance, e.g. --shards GPUs*16 (for 4 GPUs and ~100k files, try 64–128).

- Keep/enable size-balanced partitioning for skewed file sizes: --balance_by_size (auto-enables when Gini ≥ 0.6).

- If you want faster shard turnover, reduce the poll sleep to 0.2s (currently 0.5s) to shave reassign latency.


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
- Optional (first-time index creation by adapter): OPENSEARCH_NUMBER_OF_SHARDS, OPENSEARCH_NUMBER_OF_REPLICAS
- Tuning (client): OPENSEARCH_TIMEOUT=30, OPENSEARCH_MAX_RETRIES=3, OPENSEARCH_VERIFY_CERTS=1
- Ensure COGNEO_EMBED_DIM matches your embedding model dimension (e.g., 768)

Backfill from DB to OpenSearch index:
- python tools/reindex_to_opensearch.py
  - Streams Document + Embedding rows from the DB and indexes to OpenSearch via the adapter.
  - Optional: REINDEX_BATCH_SIZE=1000 to increase streaming group size.
  - First-time index creation honors OPENSEARCH_NUMBER_OF_SHARDS and OPENSEARCH_NUMBER_OF_REPLICAS; if the index already exists, it is used as-is (you may pre-create it manually).

Notes:
- Registry is lazy-loaded; OpenSearch is imported only when selected.
- Hybrid/BM25/FTS are provided by the OpenSearch adapter. Vector KNN uses HNSW (cosine).

Retrieval parity:

- Vector search: HNSW cosine KNN over knn_vector; returns doc_id/chunk_index and vector_score.
- BM25: match on text; output includes doc_id/chunk_index and a bm25 presence score (1.0) for hybrid fusion parity.
- Hybrid: alpha-weighted fusion of normalized vector_score and bm25 presence identical to DB approach.
- FTS: multi_match on text + chunk_metadata.*; parallels DB FTS over content and metadata fields conceptually. Scoring/ranking models aren't byte-identical across engines but response shape and functionality are aligned for UI/API.

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

## API Testing via curl (Complete)

Auth
- Basic (default): use -u FASTAPI_API_USER:FASTAPI_API_PASS (defaults legal_api:letmein)
- JWT (optional): obtain from POST /auth/token then pass Authorization: Bearer <token>

Quick env (optional helpers)
```sh
export API=http://localhost:8000
export AUTH="legal_api:letmein"
export AWS_REGION=${AWS_REGION:-us-east-1}
export BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-anthropic.claude-3-haiku-20240307-v1:0}
# OCI helpers (if using OCI endpoints)
export OCI_COMP=${OCI_COMPARTMENT_OCID:-"ocid1.compartment.oc1..xxxxx"}
export OCI_MODEL=${OCI_GENAI_MODEL_OCID:-"ocid1.generativeai.oc1..xxxxx"}
export OCI_REGION=${OCI_REGION:-"ap-sydney-1"}
```

Utility
- Health:
```sh
curl -u $AUTH $API/health
```
- Files (restricted roots):
```sh
curl -u $AUTH "$API/files/ls?path=$(pwd)"
```

Auth (JWT mode only)
```sh
# Issue token
curl -s -X POST $API/auth/token -H 'Content-Type: application/json' \
  -d '{"username":"legal_api","password":"letmein"}'
# Then use: -H "Authorization: Bearer <access_token>"
```

Models
- Ollama models:
```sh
curl -u $AUTH $API/models/ollama
```
- OCI GenAI models (pretrained/chat-capable):
```sh
curl -u $AUTH $API/models/oci_genai
```
- AWS Bedrock on-demand models (current region):
```sh
curl -u $AUTH "$API/models/bedrock"
```
- AWS Bedrock cross-region (example: us-east-1 and us-west-2):
```sh
curl -u $AUTH "$API/models/bedrock?regions=us-east-1,us-west-2"
```

Documents
```sh
curl -u $AUTH $API/documents
curl -u $AUTH $API/documents/1
```

Ingestion
```sh
# Start
curl -u $AUTH -X POST $API/ingest/start -H 'Content-Type: application/json' \
  -d '{"directory":"/abs/path/to/data","session_name":"ingest-'"$(date +%Y%m%d-%H%M%S)"'"}'
# List active sessions
curl -u $AUTH $API/ingest/sessions
# Stop by name
curl -u $AUTH -X POST "$API/ingest/stop?session_name=ingest-20250101-120000"
```

Search: Vector, Hybrid, FTS, Reranker
```sh
# Vector
curl -u $AUTH -X POST $API/search/vector -H 'Content-Type: application/json' \
  -d '{"query":"privacy act exemptions","top_k":5}'

# Hybrid (vector+bm25)
curl -u $AUTH -X POST $API/search/hybrid -H 'Content-Type: application/json' \
  -d '{"query":"privacy act exemptions","top_k":8,"alpha":0.5}'

# FTS (both, documents, or metadata)
curl -u $AUTH -X POST $API/search/fts -H 'Content-Type: application/json' \
  -d '{"query":"law enforcement agency","top_k":10,"mode":"both"}'

# Reranker info
curl -u $AUTH $API/models/reranker
curl -u $AUTH $API/models/rerankers
# (Optional) Download a reranker (async)
curl -u $AUTH -X POST $API/models/reranker/download -H 'Content-Type: application/json' \
  -d '{"name":"mxbai-rerank-xsmall","hf_repo":"mixedbread-ai/mxbai-rerank-xsmall"}'

# Rerank (re-score top vector hits with a given cross-encoder)
curl -u $AUTH -X POST $API/search/rerank -H 'Content-Type: application/json' \
  -d '{"query":"privacy act exemptions","top_k":5,"model":"mxbai-rerank-xsmall"}'
```

RAG (Ollama, OCI, AWS Bedrock)
```sh
# RAG (Ollama)
curl -u $AUTH -X POST $API/search/rag -H 'Content-Type: application/json' \
  -d '{
    "question":"Summarize penalties under the Spam Act",
    "model":"llama3",
    "context_chunks":["..."],
    "chunk_metadata":[{}],
    "temperature":0.1, "top_p":0.9, "max_tokens":1024, "repeat_penalty":1.1
  }'

# RAG streaming (Ollama) — plain text stream
curl -u $AUTH -N -X POST $API/search/rag_stream -H 'Content-Type: application/json' \
  -d '{
    "question":"Outline exemptions in Australian Privacy Principles",
    "model":"llama3",
    "context_chunks":["..."],
    "chunk_metadata":[{}],
    "temperature":0.1, "top_p":0.9, "max_tokens":1024, "repeat_penalty":1.1
  }'

# RAG (OCI GenAI)
curl -u $AUTH -X POST $API/search/oci_rag -H 'Content-Type: application/json' \
  -d "{
    \"oci_config\": {\"compartment_id\":\"$OCI_COMP\",\"model_id\":\"$OCI_MODEL\",\"region\":\"$OCI_REGION\"},
    \"question\":\"List mandatory reportable data breaches\",
    \"context_chunks\":[\"...\"], \"chunk_metadata\":[{}],
    \"temperature\":0.1, \"top_p\":0.9, \"max_tokens\":1024, \"repeat_penalty\":1.1
  }"

# RAG (AWS Bedrock) — current region + selected model
curl -u $AUTH -X POST $API/search/bedrock_rag -H 'Content-Type: application/json' \
  -d "{
    \"region\":\"$AWS_REGION\",
    \"model_id\":\"$BEDROCK_MODEL_ID\",
    \"question\":\"Provide a summary of consent requirements under APP\",
    \"context_chunks\":[\"...\"], \"chunk_metadata\":[{}],
    \"temperature\":0.1, \"top_p\":0.9, \"max_tokens\":1024, \"repeat_penalty\":1.1
  }"
```

Chat (Conversational)
```sh
# Conversational (Ollama)
curl -u $AUTH -X POST $API/chat/conversation -H 'Content-Type: application/json' \
  -d '{
    "llm_source":"ollama", "model":"llama3",
    "message":"What is a Notifiable Data Breach?",
    "chat_history":[], "system_prompt":"...","temperature":0.1,"top_p":0.9,"max_tokens":1024,"repeat_penalty":1.1,"top_k":10
  }'

# Conversational (OCI GenAI)
curl -u $AUTH -X POST $API/chat/conversation -H 'Content-Type: application/json' \
  -d "{
    \"llm_source\":\"oci_genai\",
    \"message\":\"Which APPs cover direct marketing?\", \"chat_history\":[],
    \"oci_config\":{\"compartment_id\":\"$OCI_COMP\",\"model_id\":\"$OCI_MODEL\",\"region\":\"$OCI_REGION\"},
    \"system_prompt\":\"...\",\"temperature\":0.1,\"top_p\":0.9,\"max_tokens\":1024,\"repeat_penalty\":1.1,\"top_k\":10
  }"

# Conversational (AWS Bedrock)
curl -u $AUTH -X POST $API/chat/conversation -H 'Content-Type: application/json' \
  -d "{
    \"llm_source\":\"bedrock\",
    \"model\":\"$BEDROCK_MODEL_ID\",
    \"message\":\"Summarize the OAIC guidance on consent.\",
    \"chat_history\":[], \"system_prompt\":\"...\",
    \"temperature\":0.1,\"top_p\":0.9,\"max_tokens\":1024,\"repeat_penalty\":1.1,\"top_k\":10
  }"
```

Chat (Agentic / Chain-of-Thought)
```sh
# Agentic (Ollama)
curl -u $AUTH -X POST $API/chat/agentic -H 'Content-Type: application/json' \
  -d '{
    "llm_source":"ollama","model":"llama3",
    "message":"Explain implied consent with legal citations.",
    "chat_history":[], "system_prompt":"...", "temperature":0.1,"top_p":0.9,"max_tokens":1024,"repeat_penalty":1.1,"top_k":10
  }'

# Agentic (OCI GenAI)
curl -u $AUTH -X POST $API/chat/agentic -H 'Content-Type: application/json' \
  -d "{
    \"llm_source\":\"oci_genai\",
    \"message\":\"Outline APP 7 with sources.\",
    \"chat_history\":[],
    \"oci_config\":{\"compartment_id\":\"$OCI_COMP\",\"model_id\":\"$OCI_MODEL\",\"region\":\"$OCI_REGION\"},
    \"system_prompt\":\"...\",\"temperature\":0.1,\"top_p\":0.9,\"max_tokens\":1024,\"repeat_penalty\":1.1,\"top_k\":10
  }"

# Agentic (AWS Bedrock)
curl -u $AUTH -X POST $API/chat/agentic -H 'Content-Type: application/json' \
  -d "{
    \"llm_source\":\"bedrock\",
    \"model\":\"$BEDROCK_MODEL_ID\",
    \"message\":\"Detail cross-border disclosure rules with citations.\",
    \"chat_history\":[], \"system_prompt\":\"...\",
    \"temperature\":0.1,\"top_p\":0.9,\"max_tokens\":1024,\"repeat_penalty\":1.1,\"top_k\":10
  }"
```

Oracle Database 26ai (optional)
```sh
curl -u $AUTH -X POST $API/db/oracle23ai_query -H 'Content-Type: application/json' \
  -d '{
    "user":"scott","password":"tiger","dsn":"myadb_high","wallet_location":"/path/to/wallet",
    "sql":"select 1 from dual"
  }'
```

Notes
- Bedrock provider payloads are auto-formatted based on modelId (Anthropic/Meta/Mistral/Cohere/Titan); ensure your model is available in AWS_REGION.
- For streaming (Ollama /search/rag_stream), -N keeps curl output unbuffered.
- All endpoints are protected; use -u for Basic (or JWT with Authorization header).

**CogNeo — Enterprise-grade, agentic, explainable legal AI built for the modern legal practice.**

---

## OpenSearch Parity + One-off Load + Optional Dual-Write (Env-gated)

This release adds full retrieval-shape parity for the OpenSearch adapter and a safe path to populate OpenSearch without changing how Streamlit/Gradio/FastAPI operate on Postgres/Oracle.

What’s new
- OpenSearch index mapping now includes:
  - doc_id (long), chunk_index (integer), citation (keyword), text (text), source (keyword), format (keyword), chunk_metadata (object), vector (knn_vector)
- OpenSearch search responses now align with DB stores (Postgres/Oracle):
  - search_vector: returns doc_id, chunk_index, score=vector_score, bm25_score=0.0, citation, text, source, format, chunk_metadata
  - search_bm25: returns doc_id, chunk_index, score=bm25_score, bm25_score=1.0 presence signal, citation, text, source, format, chunk_metadata
  - search_hybrid: alpha-weighted blend of normalized vector_score and bm25 presence -> hybrid_score; includes citation and all fields expected by APIs/UI
  - search_fts: returns doc_id, chunk_index, source, content/text, chunk_metadata, snippet=None, search_area="both"

One-off, non-disruptive backfill to OpenSearch (keep app on DB)
- Configure only OpenSearch connectivity in your .env/shell:
  - OPENSEARCH_HOST=http(s)://host:9200
  - OPENSEARCH_INDEX=cogneo_chunks_v2        # recommended new name if upgrading mapping
  - OPENSEARCH_USER / OPENSEARCH_PASS        # if required
- Run the tool (no need to switch serving backend):
  - python tools/reindex_to_opensearch.py
  - It reads Document + Embedding from the current DB backend (Postgres/Oracle) and writes to OpenSearch via the adapter.
  - It passes doc_id and chunk_index to OpenSearch for retrieval parity and computes stable _id="{doc_id}#{chunk_index}".

Serve from OpenSearch (optional)
- If you want the API/UI to retrieve from OpenSearch:
  - Set COGNEO_VECTOR_BACKEND=opensearch
  - Keep DB vars for non-retrieval features; OpenSearch is only the retrieval layer.

Optional ingestion dual-write (env-gated)
- To also push chunks to OpenSearch during ingest while keeping the DB as system of record:
  - COGNEO_VECTOR_DUAL_WRITE=1            # accepted truthy: 1, true, yes, os, opensearch
  - OPENSEARCH_HOST/INDEX/USER/PASS as above
- Behavior:
  - The worker inserts to DB as before.
  - It then best-effort indexes the same batch into OpenSearch (doc_id + chunk_index parity).
  - Any OpenSearch errors are logged as warnings and DO NOT fail the ingestion job.

Notes and migration tip
- If you previously created an OpenSearch index without doc_id/chunk_index/citation, create a new index (e.g., cogneo_chunks_v2) with the new mapping and run the backfill tool.
- Ensure COGNEO_EMBED_DIM matches your embedding model (e.g., 768).
- For HTTPS endpoints that require CA verification/SNI, pass proper OPENSEARCH_* TLS options at your infra level; otherwise use a trusted endpoint.
