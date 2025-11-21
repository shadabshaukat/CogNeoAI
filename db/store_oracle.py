"""
Oracle backend: Centralized DB models and ORM for cogneo.
- Uses Oracle 26ai VECTOR data type for embeddings: Column(Vector(EMBEDDING_DIM)) -> VECTOR(dim, FLOAT32, DENSE)
- SQL-side similarity with vector_distance(), optional APPROX to leverage HNSW/IVF indexes.
- LIKE-based fallbacks for simple text search; Oracle Text can be added later if desired.

Notes:
- Ensure COMPATIBLE init parameter is >= 23.4.0 for VECTOR support.
- VECTOR columns accept dense textual binds like "[1.0,2.0,...]" via the custom SQLAlchemy type.
- Optional auto-index creation via DBMS_VECTOR.CREATE_INDEX (env gated).

Env:
- COGNEO_EMBED_DIM (default 768)
- COGNEO_ORA_APPROX=1           # use APPROX vector_distance() to enable index usage
- COGNEO_ORA_AUTO_VECTOR_INDEX=0|1
- COGNEO_ORA_INDEX_TYPE=HNSW|IVF
- COGNEO_ORA_DISTANCE=COSINE|EUCLIDEAN|EUCLIDEAN_SQUARED|DOT|MANHATTAN|HAMMING
- COGNEO_ORA_ACCURACY=90        # target accuracy for approximate search
- COGNEO_ORA_INDEX_PARALLEL=1   # parallelism for index build
- COGNEO_ORA_HNSW_NEIGHBORS=16
- COGNEO_ORA_HNSW_EFCONSTRUCTION=200
- COGNEO_ORA_IVF_PARTITIONS=100
"""

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, Float, Date, Identity
from sqlalchemy import select, text
from sqlalchemy import String as SAString
from db.connector_oracle import engine, SessionLocal, Vector, JSONType
from datetime import datetime
import uuid
import os
import bcrypt
from typing import Any, Dict, List
import json as _json

# Compatibility aliases so external imports from db.store remain unchanged
JSONB = JSONType
UUIDType = SAString  # UUID stored as VARCHAR2(36)

# Production: avoid loading ML models at import-time in DB module.
EMBEDDING_DIM = int(os.environ.get("COGNEO_EMBED_DIM", "768"))

Base = declarative_base()

def _json_text(val):
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        try:
            return _json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", "ignore")
        except Exception:
            return str(val)
    if isinstance(val, str):
        return val
    # Fallback: best-effort JSON serialization for other Python types
    try:
        return _json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, Identity(), primary_key=True)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    registered_google = Column(Boolean, default=False)
    google_id = Column(String(128), nullable=True)
    name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, Identity(), primary_key=True)
    source = Column(String(1024), nullable=False)
    content = Column(Text, nullable=False)
    format = Column(String(64), nullable=False)

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, Identity(), primary_key=True)
    doc_id = Column(Integer, ForeignKey('documents.id'), index=True)
    chunk_index = Column(Integer, nullable=False)
    vector = Column(Vector(EMBEDDING_DIM), nullable=False)  # Oracle 26ai native VECTOR(dim, FLOAT32, DENSE)
    chunk_metadata = Column(JSONB, nullable=True)
    document = relationship("Document", backref="embeddings")

class EmbeddingSession(Base):
    __tablename__ = "embedding_sessions"
    id = Column(Integer, Identity(), primary_key=True)
    session_name = Column(String(200), unique=True, nullable=False)
    directory = Column(String(1024), nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String(32), nullable=False, default="active")
    last_file = Column(String(1024), nullable=True)
    last_chunk = Column(Integer, nullable=True)
    total_files = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)
    processed_chunks = Column(Integer, nullable=True)

class EmbeddingSessionFile(Base):
    __tablename__ = "embedding_session_files"
    id = Column(Integer, Identity(), primary_key=True)
    session_name = Column(String(200), nullable=False, index=True)
    filepath = Column(String(2048), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    completed_at = Column(DateTime, nullable=True)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    # UUID as text for Oracle
    id = Column(SAString(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    username = Column(String(200), nullable=True)
    question = Column(Text, nullable=True)
    chat_history = Column(JSONB, nullable=False)
    llm_params = Column(JSONB, nullable=False)

class ConversionFile(Base):
    __tablename__ = "conversion_files"
    id = Column(Integer, Identity(), primary_key=True)
    session_name = Column(String(200), nullable=False, index=True)
    src_file = Column(String(2048), nullable=False)
    dst_file = Column(String(2048), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    success = Column(Boolean, nullable=True, default=None)
    error_message = Column(Text, nullable=True)

# --- Relational tables for normalized legal metadata ---

class Case(Base):
    __tablename__ = "cases"
    case_id = Column(Integer, Identity(), primary_key=True)
    url = Column(String(2048), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    case_date = Column(Date, nullable=True)
    court = Column(String(128), nullable=True)

class CaseName(Base):
    __tablename__ = "case_names"
    case_name_id = Column(Integer, Identity(), primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.case_id'), nullable=False, index=True)
    name = Column(Text, nullable=False)

class CaseCitationRef(Base):
    __tablename__ = "case_citation_refs"
    citation_ref_id = Column(Integer, Identity(), primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.case_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

class Legislation(Base):
    __tablename__ = "legislation"
    legislation_id = Column(Integer, Identity(), primary_key=True)
    url = Column(String(2048), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    enacted_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    name = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class LegislationSection(Base):
    __tablename__ = "legislation_sections"
    section_id = Column(Integer, Identity(), primary_key=True)
    legislation_id = Column(Integer, ForeignKey('legislation.legislation_id'), nullable=False, index=True)
    identifier = Column(String(64), nullable=True)  # e.g., "288", "1.5.1"
    type = Column(String(32), nullable=True)        # e.g., "regulation", "schedule", "section"
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=False)

# --- Journals (normalized) ---

class Journal(Base):
    __tablename__ = "journals"
    journal_id = Column(Integer, Identity(), primary_key=True)
    url = Column(String(2048), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    published_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    title = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class JournalAuthor(Base):
    __tablename__ = "journal_authors"
    journal_author_id = Column(Integer, Identity(), primary_key=True)
    journal_id = Column(Integer, ForeignKey('journals.journal_id'), nullable=False, index=True)
    name = Column(Text, nullable=False)

class JournalCitationRef(Base):
    __tablename__ = "journal_citation_refs"
    citation_ref_id = Column(Integer, Identity(), primary_key=True)
    journal_id = Column(Integer, ForeignKey('journals.journal_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

# --- Treaties (normalized) ---

class Treaty(Base):
    __tablename__ = "treaties"
    treaty_id = Column(Integer, Identity(), primary_key=True)
    url = Column(String(2048), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    signed_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    title = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class TreatyCountry(Base):
    __tablename__ = "treaty_countries"
    treaty_country_id = Column(Integer, Identity(), primary_key=True)
    treaty_id = Column(Integer, ForeignKey('treaties.treaty_id'), nullable=False, index=True)
    country = Column(String(128), nullable=False)

class TreatyCitationRef(Base):
    __tablename__ = "treaty_citation_refs"
    citation_ref_id = Column(Integer, Identity(), primary_key=True)
    treaty_id = Column(Integer, ForeignKey('treaties.treaty_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

def create_all_tables():
    # Create only core tables for Oracle; skip relational normalization by default
    core_tables = [
        User.__table__,
        Document.__table__,
        Embedding.__table__,
        EmbeddingSession.__table__,
        EmbeddingSessionFile.__table__,
        ChatSession.__table__,
        ConversionFile.__table__,
    ]
    Base.metadata.create_all(engine, tables=core_tables)

    # Optional: auto-create Oracle 26ai VECTOR index for embeddings.vector
    # Guarded by COGNEO_ORA_AUTO_VECTOR_INDEX=1
    if os.environ.get("COGNEO_ORA_AUTO_VECTOR_INDEX", "0") == "1":
        idx_name = os.environ.get("COGNEO_ORA_INDEX_NAME", "IDX_EMBED_VECTOR")
        idx_type = (os.environ.get("COGNEO_ORA_INDEX_TYPE", "HNSW") or "HNSW").upper()  # HNSW | IVF
        org = "INMEMORY NEIGHBOR GRAPH" if idx_type == "HNSW" else "NEIGHBOR PARTITIONS"
        metric = (os.environ.get("COGNEO_ORA_DISTANCE", "COSINE") or "COSINE").upper()
        acc = int(os.environ.get("COGNEO_ORA_ACCURACY", "90"))
        par = int(os.environ.get("COGNEO_ORA_INDEX_PARALLEL", "1"))

        params = {}
        if idx_type == "HNSW":
            params = {
                "type": "HNSW",
                "neighbors": int(os.environ.get("COGNEO_ORA_HNSW_NEIGHBORS", "16")),
                "efConstruction": int(os.environ.get("COGNEO_ORA_HNSW_EFCONSTRUCTION", "200")),
            }
        else:
            params = {
                "type": "IVF",
                "partitions": int(os.environ.get("COGNEO_ORA_IVF_PARTITIONS", "100")),
            }
        import json as _json
        params_json = _json.dumps(params)

        # Use DBMS_VECTOR.CREATE_INDEX to create the vector index if possible
        plsql = text("""
        BEGIN
          DBMS_VECTOR.CREATE_INDEX(
            idx_name               => :idx_name,
            table_name             => 'EMBEDDINGS',
            idx_vector_col         => 'VECTOR',
            idx_include_cols       => NULL,
            idx_partitioning_scheme=> 'GLOBAL',
            idx_organization       => :org,
            idx_distance_metric    => :metric,
            idx_accuracy           => :acc,
            idx_parameters         => :params,
            idx_parallel_creation  => :par
          );
        EXCEPTION
          WHEN OTHERS THEN
            -- Ignore errors (e.g., insufficient privileges, already exists)
            NULL;
        END;
        """)
        try:
            with engine.begin() as conn:
                conn.execute(plsql, {
                    "idx_name": idx_name,
                    "org": org,
                    "metric": metric,
                    "acc": acc,
                    "params": params_json,
                    "par": par,
                })
        except Exception as e:
            print(f"[Oracle] Vector index creation skipped: {e}")

    # Ensure Oracle PK auto-numbering for existing schemas (embedding_sessions)
    # Try identity; if not available or fails, create sequence + trigger fallback.
    begin_plsql_sql = """
    BEGIN
      -- Attempt to convert to IDENTITY (23c/26ai); ignore if not supported or already identity
      BEGIN
        EXECUTE IMMEDIATE 'ALTER TABLE EMBEDDING_SESSIONS MODIFY (ID GENERATED BY DEFAULT AS IDENTITY)';
      EXCEPTION WHEN OTHERS THEN NULL;
      END;

      -- If still not identity, create sequence/trigger fallback
      DECLARE
        v_is_identity NUMBER := 0;
        v_cnt NUMBER := 0;
      BEGIN
        SELECT COUNT(*) INTO v_is_identity
          FROM USER_TAB_COLS
         WHERE TABLE_NAME = 'EMBEDDING_SESSIONS'
           AND COLUMN_NAME = 'ID'
           AND NVL(IDENTITY_COLUMN, 'NO') = 'YES';

        IF v_is_identity = 0 THEN
          SELECT COUNT(*) INTO v_cnt FROM USER_SEQUENCES WHERE SEQUENCE_NAME = 'EMBEDDING_SESSIONS_SEQ';
          IF v_cnt = 0 THEN
            EXECUTE IMMEDIATE 'CREATE SEQUENCE EMBEDDING_SESSIONS_SEQ START WITH 1 INCREMENT BY 1';
          END IF;

          SELECT COUNT(*) INTO v_cnt FROM USER_TRIGGERS WHERE TRIGGER_NAME = 'EMBEDDING_SESSIONS_BI';
          IF v_cnt = 0 THEN
            EXECUTE IMMEDIATE q'[
              CREATE OR REPLACE TRIGGER EMBEDDING_SESSIONS_BI
              BEFORE INSERT ON EMBEDDING_SESSIONS
              FOR EACH ROW
              WHEN (NEW.ID IS NULL)
              BEGIN
                SELECT EMBEDDING_SESSIONS_SEQ.NEXTVAL INTO :NEW.ID FROM DUAL;
              END;
            ]';
          END IF;
        END IF;
      END;
    END;
    """
    try:
        with engine.begin() as conn:
            # Use exec_driver_sql to avoid SQLAlchemy treating :NEW as a bind param
            conn.exec_driver_sql(begin_plsql_sql)
    except Exception as e:
        print(f"[Oracle] PK autoincrement ensure (embedding_sessions) skipped: {e}")

# --- User CRUD and Auth logic ---
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashval: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashval.encode('utf-8'))

def create_user(email, password=None, name=None, google_id=None, registered_google=False):
    with SessionLocal() as session:
        user = User(
            email=email,
            password_hash=hash_password(password) if password else None,
            name=name,
            google_id=google_id,
            registered_google=registered_google,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

def get_user_by_email(email: str):
    with SessionLocal() as session:
        return session.query(User).filter_by(email=email).first()

def set_last_login(user_id: int):
    with SessionLocal() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            user.last_login = datetime.utcnow()
            session.commit()

def get_user_by_googleid(google_id: str):
    with SessionLocal() as session:
        return session.query(User).filter_by(google_id=google_id).first()

# -- Chat Session functions --
def save_chat_session(chat_history, llm_params, ended_at=None, username=None, question=None):
    with SessionLocal() as session:
        chat_sess = ChatSession(
            chat_history=_json_text(chat_history),
            llm_params=_json_text(llm_params),
            ended_at=ended_at or datetime.utcnow(),
            username=username,
            question=question
        )
        session.add(chat_sess)
        session.commit()
        session.refresh(chat_sess)
        return chat_sess.id

def get_chat_session(chat_id):
    with SessionLocal() as session:
        return session.query(ChatSession).filter_by(id=chat_id).first()

# ---- Embedding/DOC ingest/session tracking ----
def start_session(session_name, directory, total_files=None, total_chunks=None):
    with SessionLocal() as session:
        sess = EmbeddingSession(
            session_name=session_name,
            directory=directory,
            started_at=datetime.utcnow(),
            status="active",
            total_files=total_files,
            total_chunks=total_chunks,
            processed_chunks=0
        )
        session.add(sess)
        session.commit()
        session.refresh(sess)
        return sess

def update_session_progress(session_name, last_file, last_chunk, processed_chunks):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.last_file = last_file
        sess.last_chunk = last_chunk
        sess.processed_chunks = processed_chunks
        session.commit()
        return sess

def complete_session(session_name):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.ended_at = datetime.utcnow()
        sess.status = "complete"
        session.commit()
        return sess

def fail_session(session_name):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.status = "error"
        session.commit()
        return sess

def get_active_sessions():
    with SessionLocal() as session:
        sessions = session.query(EmbeddingSession).filter(EmbeddingSession.status == "active").all()
        return sessions

def get_resume_sessions():
    with SessionLocal() as session:
        return session.query(EmbeddingSession).filter(EmbeddingSession.status != "complete").all()

def get_session(session_name):
    with SessionLocal() as session:
        return session.query(EmbeddingSession).filter_by(session_name=session_name).first()

def add_document(doc: dict) -> int:
    with SessionLocal() as session:
        doc_obj = Document(
            source=doc["source"],
            content=doc["content"],
            format=doc["format"],
        )
        session.add(doc_obj)
        session.commit()
        session.refresh(doc_obj)
        return doc_obj.id

def add_embedding(doc_id: int, chunk_index: int, vector, chunk_metadata=None) -> int:
    with SessionLocal() as session:
        embed_obj = Embedding(
            doc_id=doc_id,
            chunk_index=chunk_index,
            vector=vector,
            chunk_metadata=_json_text(chunk_metadata),
        )
        session.add(embed_obj)
        session.commit()
        session.refresh(embed_obj)
        return embed_obj.id

# --- Search helpers (Oracle) ---

def _cosine_distance(a: List[float], b: List[float]) -> float:
    # Return cosine distance = 1 - cosine_similarity
    if not a or not b:
        return 1.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 1.0
    return 1.0 - (dot / ((na ** 0.5) * (nb ** 0.5)))

def search_vector(query_vec, top_k=5):
    """
    Oracle VECTOR search using SQL-side vector_distance() with optional APPROX
    to enable HNSW/IVF vector index usage when present.

    Env:
      - COGNEO_ORA_APPROX=1 (default) to use APPROX keyword for index
    """
    approx = os.environ.get("COGNEO_ORA_APPROX", "1") == "1"
    # Prepare dense textual literal for the bind (e.g., "[1.0,2.0,...]")
    try:
        seq = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec or [])
    except Exception:
        seq = []
    qv_text = "[" + ",".join(str(float(v)) for v in seq) + "]"
    order_expr = "vector_distance(e.vector, :qv)"

    sql = f"""
        SELECT * FROM (
            SELECT e.doc_id,
                   e.chunk_index,
                   {order_expr} AS score,
                   d.content,
                   d.source,
                   d.format,
                   e.chunk_metadata
              FROM embeddings e
              JOIN documents d ON e.doc_id = d.id
             ORDER BY {order_expr}
        ) 
        WHERE ROWNUM <= :topk
    """
    with SessionLocal() as session:
        rows = session.execute(text(sql), {"qv": qv_text, "topk": int(top_k)}).fetchall()
        hits = []
        for row in rows:
            hits.append({
                "doc_id": row[0],
                "chunk_index": row[1],
                "score": row[2],
                "text": row[3],
                "source": row[4],
                "format": row[5],
                "chunk_metadata": row[6],
            })
        return hits

def search_bm25(query, top_k=5):
    """
    Oracle baseline: simple case-insensitive LIKE on documents.content.
    """
    with SessionLocal() as session:
        q = f"%{(query or '').lower()}%"
        res = session.execute(
            text("""
                SELECT id, content, source, format
                FROM documents
                WHERE LOWER(content) LIKE :q
                FETCH FIRST :topk ROWS ONLY
            """),
            {"q": q, "topk": int(top_k)}
        ).fetchall()
        hits = []
        for row in res:
            hits.append({
                "doc_id": row[0],
                "chunk_index": 0,
                "score": 1.0,
                "text": row[1],
                "source": row[2],
                "format": row[3],
                "chunk_metadata": None,
            })
        return hits

def search_hybrid(query, top_k=5, alpha=0.5):
    from embedding.embedder import Embedder
    embedder = Embedder()
    query_vec = embedder.embed([query])[0]
    vector_hits = search_vector(query_vec, top_k=top_k * 2)
    bm25_hits = search_bm25(query, top_k=top_k * 2)

    all_hits: Dict[Any, Dict[str, Any]] = {}
    for h in vector_hits:
        key = (h["doc_id"], h["chunk_index"])
        all_hits[key] = {
            **h,
            "vector_score": h["score"],
            "bm25_score": 0.0,
            "hybrid_score": 0.0,
        }
    for h in bm25_hits:
        key = (h["doc_id"], h["chunk_index"])
        if key in all_hits:
            all_hits[key]["bm25_score"] = 1.0
        else:
            all_hits[key] = {
                **h,
                "vector_score": 0.0,
                "bm25_score": 1.0,
                "hybrid_score": 0.0,
            }
    scores = [v["vector_score"] for v in all_hits.values()]
    if scores:
        minv, maxv = min(scores), max(scores)
        for v in all_hits.values():
            if maxv != minv:
                v["vector_score_norm"] = 1.0 - ((v["vector_score"] - minv) / (maxv - minv))
            else:
                v["vector_score_norm"] = 1.0
    else:
        for v in all_hits.values():
            v["vector_score_norm"] = 0.0
    for v in all_hits.values():
        v["hybrid_score"] = alpha * v["vector_score_norm"] + (1 - alpha) * v["bm25_score"]
    results = sorted(all_hits.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
    for r in results:
        r["citation"] = f'{r["source"]}#chunk{r.get("chunk_index",0)}'
    return results

def search_fts(query, top_k=10, mode="both"):
    """
    Oracle baseline: LIKE-based search over documents.content and/or embeddings.chunk_metadata (as text).
    """
    q = f"%{(query or '').lower()}%"
    with SessionLocal() as session:
        all_hits: List[Dict[str, Any]] = []

        if mode in ("documents", "both"):
            doc_sql = text("""
                SELECT id as doc_id, source, content, format
                  FROM documents
                 WHERE LOWER(content) LIKE :q
                 FETCH FIRST :topk ROWS ONLY
            """)
            doc_hits = session.execute(doc_sql, {"q": q, "topk": int(top_k*4)}).fetchall()
            for row in doc_hits:
                all_hits.append({
                    "doc_id": row[0],
                    "chunk_index": None,
                    "source": row[1],
                    "content": row[2],
                    "text": row[2],
                    "format": row[3],
                    "chunk_metadata": None,
                    "snippet": None,
                    "search_area": "documents",
                    "dedup_key": ("doc", row[0]),
                })

        if mode in ("metadata", "both"):
            # chunk_metadata is CLOB JSON; LIKE works for substring
            chunk_sql = text("""
                SELECT e.doc_id, e.chunk_index, d.source, d.content, e.chunk_metadata
                  FROM embeddings e
                  JOIN documents d ON e.doc_id = d.id
                 WHERE LOWER(JSON_SERIALIZE(e.chunk_metadata RETURNING CLOB)) LIKE :q
                 FETCH FIRST :topk ROWS ONLY
            """)
            chunk_rows = session.execute(chunk_sql, {"q": q, "topk": int(top_k*8)}).fetchall()
            for row in chunk_rows:
                all_hits.append({
                    "doc_id": row[0],
                    "chunk_index": row[1],
                    "source": row[2],
                    "content": row[3],
                    "text": row[4],
                    "format": None,
                    "chunk_metadata": row[4],
                    "snippet": None,
                    "search_area": "metadata",
                    "dedup_key": ("doc", row[0]),
                })

        # Deduplicate by doc_id (keep first/lowest chunk_index for metadata)
        grouped: Dict[Any, Dict[str, Any]] = {}
        for h in all_hits:
            key = h.get("dedup_key")
            if key not in grouped:
                grouped[key] = h
            else:
                if h["search_area"] == "metadata" and grouped[key]["search_area"] == "metadata":
                    if (h.get("chunk_index") or 999999) < (grouped[key].get("chunk_index") or 999999):
                        grouped[key] = h
        return list(grouped.values())[:top_k]

def get_file_contents(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return f"Could not read file: {filepath}"

def add_conversion_file(session_name, src_file, dst_file, status="pending"):
    with SessionLocal() as session:
        cf = ConversionFile(
            session_name=session_name,
            src_file=src_file,
            dst_file=dst_file,
            status=status,
            start_time=datetime.utcnow() if status == "pending" else None,
            success=None
        )
        session.add(cf)
        session.commit()
        return cf.id

def update_conversion_file_status(cf_id, status, error_message=None, success=None):
    with SessionLocal() as session:
        cf = session.query(ConversionFile).filter_by(id=cf_id).first()
        cf.status = status
        cf.end_time = datetime.utcnow()
        if error_message:
            cf.error_message = error_message
        if success is not None:
            cf.success = success
        session.commit()
        return cf

__all__ = [
    "Base", "engine", "SessionLocal", "Vector", "JSONB", "UUIDType",
    "User", "Document", "Embedding", "EmbeddingSession", "EmbeddingSessionFile", "ChatSession", "ConversionFile",
    # Relational models
    "Case", "CaseName", "CaseCitationRef", "Legislation", "LegislationSection",
    "Journal", "JournalAuthor", "JournalCitationRef",
    "Treaty", "TreatyCountry", "TreatyCitationRef",
    "create_all_tables",
    "hash_password", "check_password",
    "create_user", "get_user_by_email", "set_last_login", "get_user_by_googleid",
    "save_chat_session", "get_chat_session",
    "start_session", "update_session_progress", "complete_session", "fail_session", "get_active_sessions", "get_resume_sessions", "get_session",
    "add_document", "add_embedding", "search_vector", "search_bm25", "search_hybrid", "get_file_contents",
    "add_conversion_file", "update_conversion_file_status",
    "search_fts",
]
