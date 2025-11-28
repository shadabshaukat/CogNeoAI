"""
CogNeo — Chat (Conversation RAG with Streaming and Session History)
- ChatGPT-like interface with per-user login gate
- Streaming RAG via FastAPI (/search/rag_stream) for Local Ollama
- OCI GenAI and AWS Bedrock support (non-streaming) via FastAPI endpoints
- Conversation history persisted to DB (both Postgres and Oracle supported via db.store)
"""

import os
import json
import time
import uuid
import requests
import streamlit as st
from datetime import datetime, timezone
from rag.rag_pipeline import list_ollama_models

# DB facade (env-dispatched: postgres/oracle)
from db.store import (
    save_chat_session, get_chat_session, SessionLocal, ChatSession  # ChatSession exported by db.store.*
)

# AUTH WALL
if "user" not in st.session_state:
    st.warning("You must login to continue.")
    if hasattr(st, "switch_page"):
        st.switch_page("pages/login.py")
    else:
        st.stop()

st.set_page_config(page_title="CogNeo Chat", layout="wide")

API_ROOT = os.environ.get("COGNEO_API_URL", "http://localhost:8000")
AUTH_TUPLE = (os.environ.get("FASTAPI_API_USER", "cogneo_api"), os.environ.get("FASTAPI_API_PASS", "letmein"))

SYSTEM_PROMPT = """You are an expert research and compliance AI assistant.
Answer strictly from the provided sources and context. Always cite the source section/citation for every statement. If you do not know the answer from the context, reply: "Not found in the provided documents."
When summarizing, be neutral and factual. Never invent advice."""

# State init
if "chat_msgs" not in st.session_state:
    st.session_state["chat_msgs"] = []   # list of {"role":"user|assistant", "content": str}
if "chat_session_id" not in st.session_state:
    st.session_state["chat_session_id"] = str(uuid.uuid4())
if "session_started_at" not in st.session_state:
    st.session_state["session_started_at"] = datetime.now(timezone.utc).isoformat()
if "chat_top_k" not in st.session_state:
    st.session_state["chat_top_k"] = 10
if "llm_temperature" not in st.session_state:
    st.session_state["llm_temperature"] = 0.1
if "llm_top_p" not in st.session_state:
    st.session_state["llm_top_p"] = 0.9
if "llm_max_tokens" not in st.session_state:
    st.session_state["llm_max_tokens"] = 1024
if "llm_repeat_penalty" not in st.session_state:
    st.session_state["llm_repeat_penalty"] = 1.1

# UI styles
st.markdown("""
<style>
.chat-ct { max-width: 900px; margin: 0 auto; }
.user-bubble { background: #eef5ff; border-radius: 16px; padding: 12px 16px; margin: 8px 0; border-left: 4px solid #6aa4f5; }
.assistant-bubble { background: #f6fff4; border-radius: 16px; padding: 12px 16px; margin: 8px 0; border-left: 4px solid #78c97a; }
.streaming { color:#2f6ebd; font-size:1.05em; }
.sidebar-caption { color:#6b7a85; font-size: 0.94em; }
.session-item { font-size: 0.92em; }
</style>
""", unsafe_allow_html=True)

# Helpers
def list_recent_chat_sessions(limit=10):
    """List most recent chat sessions from DB with a preview of the first user question."""
    out = []
    try:
        with SessionLocal() as s:
            q = s.query(ChatSession).order_by(ChatSession.started_at.desc())
            if hasattr(q, "limit"):
                q = q.limit(limit)
            rows = q.all()
            for r in rows:
                # started timestamp
                try:
                    started = r.started_at.isoformat() if r.started_at else ""
                except Exception:
                    started = str(r.started_at)
                # preview: prefer explicit question column, else parse first user turn from chat_history JSON
                preview = ""
                try:
                    if getattr(r, "question", None):
                        preview = str(r.question)
                    else:
                        hist = getattr(r, "chat_history", None)
                        if isinstance(hist, (str, bytes)):
                            try:
                                hist = json.loads(hist)
                            except Exception:
                                hist = []
                        if isinstance(hist, list):
                            for msg in hist:
                                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                                    preview = str(msg.get("content"))[:60]
                                    break
                except Exception:
                    preview = ""
                if not preview:
                    preview = "(no question)"
                out.append({
                    "id": r.id,
                    "started_at": started or "",
                    "username": getattr(r, "username", None) or "",
                    "title": preview
                })
    except Exception as e:
        # If anything blows up, return empty list, we won't break the chat page
        print(f"[chat] list_recent_chat_sessions error: {e}")
    return out

def build_history_prompt(max_turns=10):
    """
    Build a compact transcript from the last N turns to preserve conversational context.
    One turn = User + Assistant. Use only text (omit citations list).
    """
    msgs = st.session_state.get("chat_msgs", [])
    # Extract last 2*max_turns messages (user+assistant pairs)
    tail = msgs[-2*max_turns:] if len(msgs) > 0 else []
    lines = []
    for m in tail:
        role = m.get("role")
        if role == "user":
            lines.append(f"User: {m.get('content','')}")
        elif role == "assistant":
            # Use only assistant content; ignore citations block if present
            content = m.get("content","")
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)

def extract_citations(hits, max_cites=10):
    """
    From hybrid hits, compile up to 10 citation strings: prefer URL, fallback to source/citation.
    """
    cites = []
    seen = set()
    for h in hits:
        meta = h.get("chunk_metadata") or {}
        url = meta.get("url")
        citation = h.get("citation")
        source = h.get("source")
        label = url or citation or source
        if not label:
            continue
        if label in seen:
            continue
        seen.add(label)
        cites.append(label)
        if len(cites) >= max_cites:
            break
    return cites

def save_current_session_snapshot():
    """Persist current chat session to DB (JSON), storing last ~20 turns."""
    try:
        user = st.session_state.get("user")
        username = user["email"] if isinstance(user, dict) and "email" in user else str(user)
        # Keep only last 20 messages to control size; 1 turn= user+assistant, so 40 items
        msgs = st.session_state.get("chat_msgs", [])
        if len(msgs) > 40:
            msgs = msgs[-40:]
            st.session_state["chat_msgs"] = msgs
        # first question if exists
        first_q = None
        for m in msgs:
            if m.get("role") == "user" and m.get("content"):
                first_q = m["content"]
                break
        llm_params = {
            "llm_source": st.session_state.get("llm_source", "Local Ollama"),
            "temperature": float(st.session_state.get("llm_temperature", 0.1)),
            "top_p": float(st.session_state.get("llm_top_p", 0.9)),
            "max_tokens": int(st.session_state.get("llm_max_tokens", 1024)),
            "repeat_penalty": float(st.session_state.get("llm_repeat_penalty", 1.1)),
            "top_k": int(st.session_state.get("chat_top_k", 10)),
            "custom_prompt": SYSTEM_PROMPT,
        }
        save_chat_session(
            chat_history=msgs,
            llm_params=llm_params,
            ended_at=datetime.now(timezone.utc),
            username=username,
            question=first_q
        )
    except Exception as e:
        print(f"[chat] save_current_session_snapshot error: {e}")

# Sidebar controls
with st.sidebar:
    st.markdown("### Account")
    _user = st.session_state.get("user")
    if _user:
        st.caption(f"Signed in as {_user.get('email','')}")
    if st.button("Logout", key="logout_btn_sidebar"):
        st.session_state.pop("user", None)
        if hasattr(st, "switch_page"):
            st.switch_page("pages/login.py")
        else:
            st.rerun()

    st.markdown("### Session", help="Your chat session details", unsafe_allow_html=True)
    st.markdown(
        f"<div class='session-item'>ID: <b>{st.session_state['chat_session_id']}</b><br>"
        f"Started: <b>{st.session_state['session_started_at'].split('.')[0].replace('T',' ')}</b></div>",
        unsafe_allow_html=True
    )
    if st.button("Start New Chat"):
        save_current_session_snapshot()
        st.session_state["chat_msgs"] = []
        st.session_state["chat_session_id"] = str(uuid.uuid4())
        st.session_state["session_started_at"] = datetime.now(timezone.utc).isoformat()
        st.rerun()

    st.markdown("---")
    st.markdown("### LLM Source")
    llm_source = st.selectbox("Select LLM", ["Local Ollama", "OCI GenAI", "AWS Bedrock"], index=0, key="llm_source")
    # When using Local Ollama, allow selecting a local model; warn if none available
    ollama_models = []
    selected_ollama = None
    if st.session_state["llm_source"] == "Local Ollama":
        try:
            ollama_models = list_ollama_models()
        except Exception:
            ollama_models = []
        if ollama_models:
            st.selectbox("Ollama model", ollama_models, index=0, key="ollama_model")
        else:
            st.session_state["ollama_model"] = None
            st.warning("No local Ollama models found. Please load a model to your local host to use as a LLM Source")

    # AWS Bedrock: Provider + Model selection (region-aware)
    if st.session_state["llm_source"] == "AWS Bedrock":
        try:
            resp = requests.get(
                f"{API_ROOT}/models/bedrock?include_current=true",
                auth=AUTH_TUPLE,
                timeout=25
            )
            data = resp.json() if resp.ok else {}
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            region_models = []
            if isinstance(data, dict):
                regions = data.get("regions") or {}
                if isinstance(regions, dict):
                    if region and region in regions:
                        region_models = (regions.get(region) or {}).get("models") or []
                    else:
                        for _reg, payload in regions.items():
                            region_models = (payload or {}).get("models") or []
                            break
            provider_to_models = {}
            for m in region_models:
                prov = str(m.get("providerName") or "Provider").strip()
                name = str(m.get("modelName") or m.get("modelId") or "Unknown").strip()
                mid = str(m.get("modelId") or "").strip()
                label = f"{name} ({mid})"
                provider_to_models.setdefault(prov, []).append((label, mid))
            # Sort models within each provider
            for prov in list(provider_to_models.keys()):
                provider_to_models[prov].sort(key=lambda x: x[0].lower())
            providers = sorted(provider_to_models.keys(), key=lambda x: x.lower())
            # Prefer provider of default model first
            default_id = os.environ.get("BEDROCK_MODEL_ID", "")
            if default_id:
                for prov, lst in provider_to_models.items():
                    if any(mid == default_id for (_lbl, mid) in lst):
                        providers = [prov] + [p for p in providers if p != prov]
                        break
            sel_prov = st.selectbox("AWS Bedrock Provider", providers, index=0 if providers else None, key="bedrock_provider")
            models_list = provider_to_models.get(sel_prov, [])
            if default_id and models_list:
                models_list = sorted(models_list, key=lambda x: (0 if x[1] == default_id else 1, x[0].lower()))
            model_labels = [lbl for (lbl, _mid) in models_list]
            sel_label = st.selectbox("AWS Bedrock Model", model_labels, index=0 if model_labels else None, key="bedrock_model_label")
            sel_id = None
            for lbl, mid in models_list:
                if lbl == sel_label:
                    sel_id = mid
                    break
            st.session_state["bedrock_model_id"] = sel_id or default_id
        except Exception as e:
            st.warning(f"Failed to load Bedrock models: {e}")
            st.session_state["bedrock_model_id"] = os.environ.get("BEDROCK_MODEL_ID","")

    st.markdown("### Generation Controls")
    st.session_state["chat_top_k"] = st.number_input("Top K context chunks", min_value=3, max_value=100, value=int(st.session_state["chat_top_k"]), step=1)
    st.session_state["llm_temperature"] = st.slider("Temperature", 0.0, 2.0, float(st.session_state["llm_temperature"]), step=0.05)
    st.session_state["llm_top_p"] = st.slider("Top-p", 0.0, 1.0, float(st.session_state["llm_top_p"]), step=0.01)
    st.session_state["llm_max_tokens"] = st.number_input("Max tokens", min_value=128, max_value=4096, value=int(st.session_state["llm_max_tokens"]), step=32)
    st.session_state["llm_repeat_penalty"] = st.slider("Repeat penalty", 1.0, 2.0, float(st.session_state["llm_repeat_penalty"]), step=0.05)
    st.caption("These parameters affect the LLM's decoding behavior.", help=None)

    st.markdown("---")
    st.markdown("### Recent Conversations")
    recent = list_recent_chat_sessions(limit=10)
    recent_opts = [f"{x['title']} — {x['started_at']} — {x['username']}" for x in recent] or ["No recent sessions"]
    sel = st.selectbox("Load a previous conversation", recent_opts, index=0)
    if st.button("Load Conversation", disabled=(not recent or "No recent sessions" in sel)):
        try:
            idx = recent_opts.index(sel)
            chat_id = recent[idx]["id"]
            rec = get_chat_session(chat_id)
            hist = rec.chat_history if rec else None
            # Parse JSON text or dict into python list
            if isinstance(hist, (str, bytes)):
                try:
                    hist = json.loads(hist)
                except Exception:
                    hist = []
            if not isinstance(hist, list):
                hist = []
            st.session_state["chat_msgs"] = hist[-40:] if len(hist) > 40 else hist
            st.session_state["chat_session_id"] = chat_id
            st.session_state["session_started_at"] = (rec.started_at.isoformat() if rec and rec.started_at else datetime.now(timezone.utc).isoformat())
            st.success(f"Loaded conversation {chat_id}")
        except Exception as e:
            st.error(f"Failed to load conversation: {e}")

st.markdown("## Chat", unsafe_allow_html=True)
st.markdown("<div class='chat-ct'>", unsafe_allow_html=True)

# Render chat history (ChatGPT-like: question, answer, citations)
for m in st.session_state["chat_msgs"]:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(m["content"])
            cites = m.get("citations") or []
            if cites:
                st.markdown("**Citations (Top 10):**")
                for i, c in enumerate(cites, 1):
                    if isinstance(c, str) and c.startswith("http"):
                        st.markdown(f"{i}. [{c}]({c})")
                    else:
                        st.markdown(f"{i}. {c}")

# Chat input
user_msg = st.chat_input("Ask a question (citations required)…")
if user_msg:
    # Append user turn
    st.session_state["chat_msgs"].append({"role": "user", "content": user_msg})
    # Immediately display the user message under previous citations (ChatGPT-like order)
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Retrieve context via FastAPI hybrid search
    hits = []
    context_chunks = []
    chunk_metadata = []
    try:
        r_ctx = requests.post(
            f"{API_ROOT}/search/hybrid",
            json={"query": user_msg, "top_k": int(st.session_state['chat_top_k']), "alpha": 0.5},
            auth=AUTH_TUPLE, timeout=30
        )
        r_ctx.raise_for_status()
        hits = r_ctx.json() if r_ctx.ok else []
        context_chunks = [h.get("text", "") for h in hits]
        chunk_metadata = [h.get("chunk_metadata") or {} for h in hits]
    except Exception as e:
        hits, context_chunks, chunk_metadata = [], [], []

    # Build citations list (top 10)
    citations = extract_citations(hits, max_cites=10)

    # Compose history-augmented prompt
    history_txt = build_history_prompt(max_turns=10)
    custom_prompt = SYSTEM_PROMPT + ("\n\nChat History:\n" + history_txt if history_txt else "")

    # Assistant response area (stream for Ollama; others as full text)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_text = ""

        if st.session_state["llm_source"] == "Local Ollama":
            # Stream from /search/rag_stream
            try:
                selected_model = st.session_state.get("ollama_model")
                if not selected_model:
                    assistant_text += "No local Ollama models found. Please load a model to your local host to use as a LLM Source"
                    placeholder.markdown(f"<div class='assistant-bubble streaming'>{assistant_text}</div>", unsafe_allow_html=True)
                else:
                    payload = {
                        "question": user_msg,
                        "context_chunks": context_chunks or [],
                        "chunk_metadata": chunk_metadata or [],
                        "custom_prompt": SYSTEM_PROMPT,
                        "temperature": float(st.session_state["llm_temperature"]),
                        "top_p": float(st.session_state["llm_top_p"]),
                        "max_tokens": int(st.session_state["llm_max_tokens"]),
                        "repeat_penalty": float(st.session_state["llm_repeat_penalty"]),
                        "model": selected_model
                    }
                    with requests.post(f"{API_ROOT}/search/rag_stream", json=payload, auth=AUTH_TUPLE, stream=True, timeout=300) as resp:
                        resp.raise_for_status()
                        for chunk in resp.iter_lines(decode_unicode=True):
                            if not chunk:
                                continue
                            assistant_text += chunk
                            placeholder.markdown(assistant_text)
            except Exception as e:
                assistant_text += f"\n[stream-error] {e}"
                placeholder.markdown(assistant_text)

        elif st.session_state["llm_source"] == "OCI GenAI":
            try:
                payload = {
                    "oci_config": {
                        "compartment_id": os.environ.get("OCI_COMPARTMENT_OCID",""),
                        "model_id": os.environ.get("OCI_GENAI_MODEL_OCID",""),
                        "region": os.environ.get("OCI_REGION","ap-sydney-1")
                    },
                    "question": user_msg,
                    "context_chunks": context_chunks or [],
                    "sources": [],
                    "chunk_metadata": chunk_metadata or [],
                    "custom_prompt": custom_prompt,
                    "temperature": float(st.session_state["llm_temperature"]),
                    "top_p": float(st.session_state["llm_top_p"]),
                    "max_tokens": int(st.session_state["llm_max_tokens"]),
                    "repeat_penalty": float(st.session_state["llm_repeat_penalty"])
                }
                r = requests.post(f"{API_ROOT}/search/oci_rag", json=payload, auth=AUTH_TUPLE, timeout=180)
                data = r.json() if r.ok else {}
                assistant_text = data.get("answer","") or ""
                placeholder.markdown(assistant_text)
            except Exception as e:
                assistant_text = f"[oci-error] {e}"
                placeholder.markdown(assistant_text)

        else:  # AWS Bedrock
            try:
                payload = {
                    "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
                    "model_id": st.session_state.get("bedrock_model_id") or os.environ.get("BEDROCK_MODEL_ID",""),
                    "question": user_msg,
                    "context_chunks": context_chunks or [],
                    "sources": [],
                    "chunk_metadata": chunk_metadata or [],
                    "custom_prompt": custom_prompt,
                    "temperature": float(st.session_state["llm_temperature"]),
                    "top_p": float(st.session_state["llm_top_p"]),
                    "max_tokens": int(st.session_state["llm_max_tokens"]),
                    "repeat_penalty": float(st.session_state["llm_repeat_penalty"])
                }
                r = requests.post(f"{API_ROOT}/search/bedrock_rag", json=payload, auth=AUTH_TUPLE, timeout=180)
                data = r.json() if r.ok else {}
                assistant_text = data.get("answer","") or ""
                placeholder.markdown(assistant_text)
            except Exception as e:
                assistant_text = f"[bedrock-error] {e}"
                placeholder.markdown(assistant_text)

        # After final answer, show citations below
        if citations:
            st.markdown("**Citations (Top 10):**")
            for i, c in enumerate(citations, 1):
                if isinstance(c, str) and c.startswith("http"):
                    st.markdown(f"{i}. [{c}]({c})")
                else:
                    st.markdown(f"{i}. {c}")

        # Detailed context chunks (expanders), similar to Hybrid Search & RAG
        if hits:
            st.markdown("**Context Chunks Used (Details):**")
            for i, h in enumerate(hits, 1):
                with st.expander(f"{i}. {h.get('citation','?')} | Score: {h.get('hybrid_score',0):.3f}"):
                    meta = h.get("chunk_metadata") or {}
                    if meta:
                        st.markdown("**Metadata:**")
                        for k, v in meta.items():
                            st.write(f"- {k}: {v}")
                    st.write(f"**Source:** {h.get('source','?')}\n**Chunk:** {h.get('chunk_index','?')}\n**Format:** {h.get('format','?')}")
                    text = h.get('text','')
                    st.write(f"**Text:**\n{text[:1200]}{'...' if len(text)>1200 else ''}")

    # Commit assistant turn with citations and save snapshot
    st.session_state["chat_msgs"].append({"role": "assistant", "content": assistant_text, "citations": citations})
    # Keep last 20 turns (40 messages)
    if len(st.session_state["chat_msgs"]) > 40:
        st.session_state["chat_msgs"] = st.session_state["chat_msgs"][-40:]
    save_current_session_snapshot()

st.markdown("</div>", unsafe_allow_html=True)
