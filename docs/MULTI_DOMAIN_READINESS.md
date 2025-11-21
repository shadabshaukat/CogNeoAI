# Multi‑Domain Readiness — Architecture Notes

Status: Proposal/architecture only. No multi‑domain code changes have been implemented. This document is a deep‑dive strategic outline for a future platform approach and is intended for review and alignment before any implementation.

This document outlines how CogNeo can generalize into a multi‑industry/domain RAG/Agentic AI search platform (health, telecom, finance/banking, etc.) with minimal changes to the core.

## Goals

- Add new domains without forking the core stack.
- Keep ingestion/chunking/embedding pipelines pluggable per domain.
- Preserve backend DB separation (Postgres vs Oracle Database 26ai) via env‑driven dispatchers.
- Retain UI/API routes while allowing domain‑specific behavior through configuration.

## Current Separation (Already Implemented)

- Backend dispatchers:
  - `db/connector.py` and `db/store.py` autoload `.env` and pick backend by `COGNEO_DB_BACKEND` (`postgres` default, `oracle` optional).
  - Postgres codepaths: `db/connector_postgres.py`, `db/store_postgres.py` (pgvector + JSONB + tsvector).
  - Oracle codepaths: `db/connector_oracle.py`, `db/store_oracle.py` (native `VECTOR` + native `JSON`, `vector_distance`, JSON_SERIALIZE for JSON LIKE).
- Ingestion pipeline (beta):
  - Orchestrator/worker design decoupled from DB specifics. Domain logic is primarily in parsing/chunking and metadata enrichment.

## Proposed Domain Architecture

Introduce a “domain pack” concept to localize domain‑specific pieces:

```
ingest/
  domains/
    legal/
      chunkers.py        # e.g., dashed headers, legislation/case patterns
      schema_hints.md    # domain metadata keys and typical filters
    healthcare/
      chunkers.py        # HIPAA-aware parsing, clinical sections, ICD/CPT hints
    telecom/
      chunkers.py        # RFC/spec headings, incident reports, OSS/BSS artifacts
    finance/
      chunkers.py        # SEC filings sections, statements, transaction logs
```

- Each `chunkers.py` exports a uniform API:
  - `def detect_doc_type(meta: dict, text: str) -> Optional[str]`
  - `def chunk_document(text: str, base_meta: dict, cfg: ChunkingConfig) -> list[dict]`
  - Optionally specialized helpers (e.g., dashed‑header, schedule/section, YAML preambles).

- Existing generic helpers remain at:
  - `ingest/semantic_chunker.py` (token‑aware semantic chunking + utilities)
  - `ingest/loader.py` (parse .txt/.html, extract base metadata block)

## Domain Selection

Add one environment var to select the domain pack at runtime (defaults to `legal`):

```
COGNEO_DOMAIN=legal        # legal | healthcare | telecom | finance | ...
```

Worker/orchestrator dispatch:

```python
# Pseudocode inside ingest.beta_worker / ingest.beta_ingest
domain = os.environ.get("COGNEO_DOMAIN", "legal")
if domain == "healthcare":
    from ingest.domains.healthcare.chunkers import detect_doc_type as _detect, chunk_document as _chunk
elif domain == "telecom":
    from ingest.domains.telecom.chunkers import detect_doc_type as _detect, chunk_document as _chunk
# ...
else:
    from ingest.domains.legal.chunkers import detect_doc_type as _detect, chunk_document as _chunk

detected_type = _detect(base_meta, base_doc.get("text", ""))
file_chunks = _chunk(base_doc["text"], base_meta=base_meta, cfg=cfg)
```

- Fallback: if a domain pack does not provide a custom chunker, call the generic semantic chunker in `ingest/semantic_chunker.py`.

## Metadata & Filters

- Keep `embeddings.chunk_metadata` as JSON (JSONB on Postgres, JSON on Oracle Database 26ai).
- Encourage consistent keys across domains (e.g., `type`, `jurisdiction`, `year`, `title`, `url`, `section_idx`), plus domain‑specific keys as needed.
- Postgres can surface `md_*` generated columns or expression indexes to accelerate filters. Oracle can use functional indexes and/or Oracle Text as an optional follow‑up.

## DB Backends

- Postgres (default):
  - pgvector for embeddings; tsvector/ts_headline for FTS; JSONB for metadata.
  - Post‑load indexing patterns under `schema-post-load/`.
- Oracle Database 26ai (optional):
  - Native `VECTOR(dim, FLOAT32, DENSE)` and native `JSON` column types.
  - `vector_distance(e.vector, :qv)` for SQL‑side ranking; optional `DBMS_VECTOR.CREATE_INDEX` for HNSW/IVF.
  - Baseline metadata search via `JSON_SERIALIZE(... RETURNING CLOB)` LIKE; Oracle Text can be added later.

## UI/API Considerations

- The RAG and Agentic endpoints (`/search/hybrid`, `/search/vector`, `/search/fts`, `/chat/*`) already accept and propagate `chunk_metadata` lists; they are domain agnostic.
- Gradio/Streamlit renderers treat `chunk_metadata` opaquely (dict or string), sufficient for multi‑domain sources.

## Configuration Summary

```
# Backend database selection (kept)
COGNEO_DB_BACKEND=postgres | oracle

# Domain pack selection (new)
COGNEO_DOMAIN=legal           # legal|healthcare|telecom|finance|...

# Ingestion/embedding (existing)
COGNEO_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
COGNEO_EMBED_DIM=768
COGNEO_EMBED_BATCH=64
```

## Migration Path

1) Establish directory structure under `ingest/domains/*`.
2) Move legal‑specific chunk logic from `ingest/semantic_chunker.py` into `ingest/domains/legal/chunkers.py` (keep generic functions in place).
3) Introduce domain import dispatch in worker/orchestrator.
4) Add README snippets for each domain pack as they are created.

## Testing Matrix

- Postgres + legal (baseline)
- Oracle Database 26ai + legal (baseline)
- Postgres + healthcare (domain pack prototype)
- Oracle Database 26ai + healthcare (domain pack prototype)

Each matrix pair should validate:
- Ingestion success (schema bootstrap + inserts).
- `/search/vector`, `/search/hybrid`, `/search/fts` perf/function.
- UI rendering of sources and metadata cards.

## Enterprise Strategy (Deep Dive)

This section captures a “platform at scale” approach — how a billion‑dollar company would structure a multi‑domain, multi‑database RAG/Agentic platform to evolve safely, predictably, and rapidly. This is an architectural north star to guide future work; it does not change any code today.

### Core principles

- Platform over product
  - Clear separation between Core (shared runtime, orchestration, contracts) and Domain Packs (industry logic).
  - Minimize cross‑domain coupling with strict interfaces and versioned contracts.
- Extensibility first
  - Plug‑in architecture for domain chunking/ingestion, retrieval transforms, and downstream actions (e.g., enrichment).
- Explicit data contracts
  - Versioned schemas for chunk_metadata, ingestion events, and service responses; automated contract tests.
- Environment and capability flags
  - Feature gates and capability discovery (DB features, tokenizer limits, vector ops) set behavior at runtime without code forks.
- Safe changes at scale
  - Backward compatibility, migration playbooks, and canary rollouts as first‑class concerns.

### Layered architecture (logical)

1) Experience layer
   - UIs (Gradio/Streamlit), API (FastAPI), SDKs (Python/CLI). Domain‑agnostic inputs/outputs; no vendor logic here.
2) Application layer
   - Orchestration, request shaping, retrieval policy, ranking, post‑processing; domain‑agnostic core flows.
3) Domain layer (plugin)
   - Domain Packs implement interfaces: parse, detect_doc_type, chunk_document, metadata enrichment, redaction rules.
   - Each pack ships its own tests, fixtures, docs, and optional post‑ingest analytics.
4) Data layer
   - Storage adapters and repositories with strict interfaces; adapters for Postgres/pgvector, Oracle Database 26ai, etc.
   - Migrations and index management per backend, encapsulated.

### Domain packs (as packages)

- Packaging
  - Each domain under ingest/domains/<domain>/ packaged as an installable module (optional separate repo for governance).
- Interfaces
  - detect_doc_type(meta: dict, text: str) -> Optional[str]
  - chunk_document(text: str, base_meta: dict, cfg: ChunkingConfig) -> List[Chunk]
  - Optional preprocess/postprocess hooks (e.g., PHI redaction in healthcare).
- Versioning
  - Pack version drives contract versions and CI matrices; pack update does not force platform update if contracts are stable.

### Data contracts and compatibility

- Chunk metadata contract (platform‑wide)
  - Stable core keys (type, title, url, year, section_idx, tokens_est), plus domain‑specific extensions under a namespaced key (e.g., x_healthcare).
- Schema evolution
  - Backward compatible writes; readers tolerate unknown fields.
  - Deprecate keys via feature flags and dual‑write periods.
- Test enforcement
  - Contract tests run per domain pack and per backend adapter.

### Backend abstraction (multi‑DB)

- Repository pattern
  - Domain and application layers depend on interfaces (search, insert, fts) not concrete engines.
- Storage adapters
  - postgres_adapter: JSONB, pgvector, tsvector/ts_headline pipeline.
  - oracle26ai_adapter: native JSON, VECTOR, JSON_SERIALIZE LIKE baseline; optional Oracle Text later.
- Capability registry
  - At runtime, register capabilities (e.g., supports_tsheadline=false, supports_native_json=true, supports_vector_indexes=true) to toggle strategies safely.

### Configuration and rollout

- Runtime configuration
  - COGNEO_DB_BACKEND, COGNEO_DOMAIN, feature flags (e.g., RAG_STRICT_CITATION, ENABLE_ORACLE_TEXT).
- Progressive delivery
  - Env‑scoped flags (dev/stage/prod), canary cohorts, and rollback levers; config stored centrally (e.g., Consul/ConfigMap/Secrets Manager).

### Observability, SLOs, and governance

- Metrics and traces
  - Request/ingest latencies, index health, vector search quality counters, domain‑specific parsing success rates.
- SLOs
  - Per route and per backend (p50/p95 latency, error budgets).
- Governance
  - DRI per domain pack; approval gates on contract changes; artifact signing and provenance tracking.

### Security, tenancy, and compliance

- Multi‑tenant isolation
  - Namespaced schemas or DBs, per‑tenant encryption keys, RBAC at API and storage layers.
- Compliance
  - Domain‑specific redaction pipelines (e.g., PHI in healthcare), audit logs, retention policies; optional immutable storage for regulated logs.

### CI/CD and release

- Monorepo or multi‑repo with pinned versions for domain packs.
- Matrix builds across domains x backends; mutation tests for contract compatibility.
- Release trains with changelogs per pack and per backend adapter.

### Migration playbooks

- Index and schema migrations treated like code
  - Preflight checks, online creation (CONCURRENT where possible), backfills off‑peak, safe fallbacks.
- Data contract migrations
  - Announce, dual‑write, monitor, then cutover with kill‑switch.

### No‑op implementation note

- This document is strategic; no multi‑domain code has been added or changed now.
- Current system remains “legal” domain by default with env‑driven backend selection (Postgres default, Oracle Database 26ai optional).

### Open questions (for later alignment)

- Domain granularity: mono pack per industry vs sub‑packs per content type (e.g., filings vs transcripts).
- Storage tiering: hot vs warm storage strategies per domain (cost/perf).
- Central registry: domain pack discovery, version pinning, and provenance.

## Summary

This approach keeps DB and app layers stable while allowing per‑domain chunking/ingestion to evolve independently. It preserves env‑driven backend dispatch and enables controlled expansion into non‑legal domains with minimal friction.
