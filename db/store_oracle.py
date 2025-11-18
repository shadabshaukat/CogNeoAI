"""
Oracle backend: Centralized DB models and ORM for auslegalsearchv3.
- Mirrors the Postgres models but uses Oracle-compatible types and SQL.
- Vectors are stored as JSON (CLOB) via a TypeDecorator in connector_oracle.Vector.
- FTS/pg_trgm/pgvector features are not used; searches fall back to LIKE and Python-side vector scoring.

Notes:
- This baseline aims for functional parity with minimal core logic changes.
- Optimize later using Oracle 23ai VECTOR type and Oracle Text, if available.

Env:
- AUSLEGALSEARCH_EMBED_DIM (default 768)
- AUSLEGALSEARCH_ORA_VECTOR_SCAN_LIMIT (default 5000)  # max rows scanned for Python-side vector ranking
"""

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, Float, Date
from sqlalchemy import select, text
from sqlalchemy import String as SAString
from db.connector_oracle import engine, SessionLocal, Vector, JSONType
from datetime import datetime
import uuid
import os
import bcrypt
from typing import Any, Dict, List

# Compatibility aliases so external imports from db.store remain unchanged
JSONB = JSONType
UUIDType = SAString  # UUID stored as VARCHAR2(36)

# Production: avoid loading ML models at import-time in DB module.
EMBEDDING_DIM = int(os.environ.get("AUSLEGALSEARCH_EMBED_DIM", "768"))

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    registered_google = Column(Boolean, default=False)
    google_id = Column(String(128), nullable=True)
    name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source = Column(String(1024), nullable=False)
    content = Column(Text, nullable=False)
    format = Column(String(64), nullable=False)

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey('documents.id'), index=True)
    chunk_index = Column(Integer, nullable=False)
    vector = Column(Vector(EMBEDDING_DIM), nullable=False)  # stored as JSON text via TypeDecorator
    chunk_metadata = Column(JSONB, nullable=True)
    document = relationship("Document", backref="embeddings")

class EmbeddingSession(Base):
    __tablename__ = "embedding_sessions"
    id = Column(Integer, primary_key=True)
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
    id = Column(Integer, primary_key=True)
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
    id = Column(Integer, primary_key=True)
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
    case_id = Column(Integer, primary_key=True)
    url = Column(String(2048), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    case_date = Column(Date, nullable=True)
    court = Column(String(128), nullable=True)

class CaseName(Base):
    __tablename__ = "case_names"
    case_name_id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.case_id'), nullable=False, index=True)
    name = Column(Text, nullable=False)

class CaseCitationRef(Base):
    __tablename__ = "case_citation_refs"
    citation_ref_id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.case_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

class Legislation(Base):
    __tablename__ = "legislation"
    legislation_id = Column(Integer, primary_key=True)
    url = Column(String(2048), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    enacted_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    name = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class LegislationSection(Base):
    __tablename__ = "legislation_sections"
    section_id = Column(Integer, primary_key=True)
    legislation_id = Column(Integer, ForeignKey('legislation.legislation_id'), nullable=False, index=True)
    identifier = Column(String(64), nullable=True)  # e.g., "288", "1.5.1"
    type = Column(String(32), nullable=True)        # e.g., "regulation", "schedule", "section"
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=False)

# --- Journals (normalized) ---

class Journal(Base):
    __tablename__ = "journals"
    journal_id = Column(Integer, primary_key=True)
    url = Column(String(2048), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    published_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    title = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class JournalAuthor(Base):
    __tablename__ = "journal_authors"
    journal_author_id = Column(Integer, primary_key=True)
    journal_id = Column(Integer, ForeignKey('journals.journal_id'), nullable=False, index=True)
    name = Column(Text, nullable=False)

class JournalCitationRef(Base):
    __tablename__ = "journal_citation_refs"
    citation_ref_id = Column(Integer, primary_key=True)
    journal_id = Column(Integer, ForeignKey('journals.journal_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

# --- Treaties (normalized) ---

class Treaty(Base):
    __tablename__ = "treaties"
    treaty_id = Column(Integer, primary_key=True)
    url = Column(String(2048), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    signed_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    title = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class TreatyCountry(Base):
    __tablename__ = "treaty_countries"
    treaty_country_id = Column(Integer, primary_key=True)
    treaty_id = Column(Integer, ForeignKey('treaties.treaty_id'), nullable=False, index=True)
    country = Column(String(128), nullable=False)

class TreatyCitationRef(Base):
    __tablename__ = "treaty_citation_refs"
    citation_ref_id = Column(Integer, primary_key=True)
    treaty_id = Column(Integer, ForeignKey('treaties.treaty_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

def create_all_tables():
    # Oracle baseline: just create tables if missing (no extensions/FTS triggers).
    Base.metadata.create_all(engine)

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
            chat_history=chat_history,
            llm_params=llm_params,
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
            chunk_metadata=chunk_metadata,
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
    Oracle baseline: fetch up to N embeddings and rank by Python-side cosine distance.
    N defaults to AUSLEGALSEARCH_ORA_VECTOR_SCAN_LIMIT (5000).
    """
    scan_limit = int(os.environ.get("AUSLEGALSEARCH_ORA_VECTOR_SCAN_LIMIT", "5000"))
    qv = [float(v) for v in list(query_vec or [])]

    with SessionLocal() as session:
        # Limited scan to avoid full table scan at large scale
        sql = text("""
            SELECT e.doc_id, e.chunk_index, e.vector, e.chunk_metadata, d.content, d.source, d.format
            FROM embeddings e
            JOIN documents d ON e.doc_id = d.id
            WHERE ROWNUM <= :lim
        """)
        rows = session.execute(sql, {"lim": scan_limit}).fetchall()

        ranked = []
        for row in rows:
            doc_id = row[0]
            chunk_index = row[1]
            vec = row[2] or []
            score = _cosine_distance(vec, qv)
            ranked.append({
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "score": score,
                "text": row[4],
                "source": row[5],
                "format": row[6],
                "chunk_metadata": row[3],
            })
        ranked.sort(key=lambda x: x["score"])
        return ranked[:top_k]

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
                 WHERE LOWER(e.chunk_metadata) LIKE :q
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
