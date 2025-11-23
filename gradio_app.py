"""
CogNeo Gradio UI:
- Tabs for Hybrid Search, Vector Search, RAG (with supersystem prompt), Conversational Chat, and Agentic Chat
- Supports Local Ollama, OCI GenAI, and AWS Bedrock with provider+model selection for Bedrock
"""

# Always load .env if present (so AUTO_DDL and DB_* vars are available even if shell didn't export them)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import gradio as gr
import requests
import os
import html
import json
import re

from db.store import create_all_tables

# Ensure DB schema on UI startup when enabled
if os.environ.get("COGNEO_AUTO_DDL", "1") == "1":
    try:
        create_all_tables()
        print("[gradio] DB schema ensured (AUTO_DDL=1)")
    except Exception as e:
        print(f"[gradio] DB schema ensure failed: {e}")

API_ROOT = os.environ.get("COGNEO_API_URL", "http://localhost:8000")
SESS = type("Session", (), {"user": None, "auth": None})()
SESS.auth = None

DEFAULT_SYSTEM_PROMPT = """You are an expert Australian legal research and compliance AI assistant.
Answer strictly from the provided sources and context. Always cite the source section/citation for every statement. If you do not know the answer from the context, reply: "Not found in the provided legal documents."
When summarizing, be neutral and factual. Never invent legal advice."""

def login_fn(username, password):
    try:
        r = requests.get(f"{API_ROOT}/health", auth=(username, password), timeout=10)
        if r.ok:
            SESS.auth = (username, password)
            return gr.update(visible=False), gr.update(visible=True), f"Welcome, {username}!", ""
        else:
            return gr.update(visible=True), gr.update(visible=False), "", "Invalid login."
    except Exception:
        return gr.update(visible=True), gr.update(visible=False), "", "Invalid login."

def fetch_oci_models():
    try:
        resp = requests.get(f"{API_ROOT}/models/oci_genai", auth=SESS.auth, timeout=30)
        data = resp.json() if resp.ok else {}
        all_models = data.get("all", []) if isinstance(data, dict) else []
        return [
            (
                m.get("display_name", m.get("model_name", m.get("id", "Unknown"))),
                json.dumps(m)
            )
            for m in all_models
            if m.get("id") or m.get("ocid") or m.get("model_id")
        ] or [("No OCI GenAI models found. Check Oracle Console/config.", "")]
    except Exception as exc:
        return [(f"Error loading Oracle models: {exc}", "")]

def fetch_ollama_models():
    try:
        resp = requests.get(f"{API_ROOT}/models/ollama", auth=SESS.auth, timeout=15)
        data = resp.json() if resp.ok else []
        if isinstance(data, list):
            return [(m, m) for m in data]
        return []
    except Exception:
        return []

# ===== AWS Bedrock helpers =====

def fetch_bedrock_models_grouped():
    """
    Returns dict provider -> list of (display, modelId) for current region or the first region returned.
    Display format: 'Provider — ModelName (modelId)' but we store only 'ModelName (modelId)' in display for provider-specific dropdown.
    """
    grouped = {}
    try:
        resp = requests.get(f"{API_ROOT}/models/bedrock?include_current=true", auth=SESS.auth, timeout=25)
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
        for m in region_models:
            prov = str(m.get("providerName") or "Provider").strip()
            name = str(m.get("modelName") or m.get("modelId") or "Unknown").strip()
            mid = str(m.get("modelId") or "").strip()
            disp = f"{name} ({mid})"
            grouped.setdefault(prov, []).append((disp, mid))
        # Sort within each provider by display
        for prov in list(grouped.keys()):
            grouped[prov].sort(key=lambda x: x[0].lower())
        return grouped
    except Exception:
        return {}

def bedrock_providers():
    """
    Returns provider list with the default model's provider (if resolvable) first.
    """
    grouped = fetch_bedrock_models_grouped()
    if not grouped:
        return []
    providers = sorted(grouped.keys(), key=lambda x: x.lower())
    default_id = os.environ.get("BEDROCK_MODEL_ID", "")
    if default_id:
        try:
            # Find provider for default model
            for prov, lst in grouped.items():
                if any(mid == default_id for (_disp, mid) in lst):
                    providers = [prov] + [p for p in providers if p != prov]
                    break
        except Exception:
            pass
    return providers

def bedrock_models_for_provider(provider: str | None):
    """
    Returns display list for given provider; if None, use first provider or provider of default model.
    """
    grouped = fetch_bedrock_models_grouped()
    if not grouped:
        return []
    providers = bedrock_providers()
    use_prov = provider if (provider and provider in grouped) else (providers[0] if providers else None)
    if not use_prov:
        return []
    default_id = os.environ.get("BEDROCK_MODEL_ID", "")
    items = grouped.get(use_prov, [])
    if default_id:
        items.sort(key=lambda x: (0 if x[1] == default_id else 1, x[0].lower()))
    return [disp for (disp, _mid) in items]

def bedrock_default_selection():
    """
    Returns (default_provider, default_model_display) based on BEDROCK_MODEL_ID when resolvable; otherwise first provider/model.
    """
    grouped = fetch_bedrock_models_grouped()
    if not grouped:
        return None, None
    providers = bedrock_providers()
    default_provider = providers[0] if providers else None
    default_model_disp = None
    default_id = os.environ.get("BEDROCK_MODEL_ID", "")
    if default_id:
        for prov, lst in grouped.items():
            for disp, mid in lst:
                if mid == default_id:
                    default_provider = prov
                    default_model_disp = disp
                    break
            if default_model_disp:
                break
    # If no explicit default model display, pick first model of default provider
    if not default_model_disp and default_provider and default_provider in grouped and grouped[default_provider]:
        default_model_disp = grouped[default_provider][0][0]
    return default_provider, default_model_disp

# ===== Misc helpers =====

def format_context_cards(sources, chunk_metadata, context_chunks):
    if not sources and not context_chunks:
        return ""
    cards = []
    for idx in range(max(len(sources), len(context_chunks))):
        meta = (chunk_metadata[idx] if idx < len(chunk_metadata) else {}) or {}
        citation = meta.get("citation") or meta.get("url") or (sources[idx] if idx < len(sources) else "?")
        url = meta.get("url")
        url_part = f'<a href="{url}" target="_blank">{html.escape(str(citation))}</a>' if url else html.escape(str(citation))
        text = context_chunks[idx] if idx < len(context_chunks) else ""
        excerpt = html.escape(text[:600]) + ("..." if len(text) > 600 else "")
        card = f"""
        <div class='chat-source-card'>
            <div class='chat-source-cite'>{url_part}</div>
            <div class='chat-source-excerpt'>{excerpt}</div>
        </div>
        """
        cards.append(card)
    return "<div class='chat-source-cards-ct'>" + "".join(cards) + "</div>"

def hybrid_search_fn(query, top_k, alpha):
    reranker_model = "mxbai-rerank-xsmall"
    try:
        resp = requests.post(
            f"{API_ROOT}/search/hybrid",
            json={"query": query, "top_k": top_k, "alpha": alpha, "reranker": reranker_model}, auth=SESS.auth, timeout=20
        )
        resp.raise_for_status()
        hits = resp.json()
    except Exception:
        hits = []
    return format_context_cards([h.get("citation","?") for h in hits], [h.get("chunk_metadata") or {} for h in hits], [h.get("text","") for h in hits])

def vector_search_fn(query, top_k):
    try:
        resp = requests.post(
            f"{API_ROOT}/search/vector",
            json={"query": query, "top_k": top_k}, auth=SESS.auth, timeout=20
        )
        resp.raise_for_status()
        hits = resp.json()
    except Exception:
        hits = []
    return "<ul>" + "".join(f"<li>{html.escape(h.get('text','')[:150])}{'...' if len(h.get('text',''))>150 else ''}</li>" for h in hits) + "</ul>"

# ===== RAG and Chat call functions =====

def rag_chatbot(question, llm_source, ollama_model, oci_model_info_json, bedrock_provider, bedrock_model, rag_top_k, system_prompt, temperature, top_p, max_tokens, repeat_penalty):
    if not question:
        return "", "", ""
    try:
        resp = requests.post(
            f"{API_ROOT}/search/hybrid",
            json={"query": question, "top_k": rag_top_k, "alpha": 0.5, "reranker": "mxbai-rerank-xsmall"},
            auth=SESS.auth, timeout=30
        )
        resp.raise_for_status()
        hits = resp.json()
        context_chunks = [h.get("text", "") for h in hits]
        sources = [h.get("citation") or ((h.get("chunk_metadata") or {}).get("url") if (h.get("chunk_metadata") or {}) else "?") for h in hits]
        chunk_metadata = [h.get("chunk_metadata") or {} for h in hits]
    except Exception:
        context_chunks, sources, chunk_metadata = [], [], []
    answer = ""
    params = {
        "context_chunks": context_chunks or [],
        "sources": sources or [],
        "chunk_metadata": chunk_metadata or [],
        "custom_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty
    }
    if llm_source == "OCI GenAI":
        oci_region = os.environ.get("OCI_REGION", "ap-sydney-1")
        try:
            model_info = json.loads(oci_model_info_json) if oci_model_info_json else {}
        except Exception:
            model_info = {}
        model_id = model_info.get("id") or model_info.get("ocid") or model_info.get("model_id") or os.environ.get("OCI_GENAI_MODEL_OCID", "")
        oci_payload = {
            "model_info": model_info,
            "oci_config": {
                "compartment_id": os.environ.get("OCI_COMPARTMENT_OCID", ""),
                "model_id": model_id,
                "region": oci_region
            },
            "question": question,
            **params,
        }
        try:
            r_oci = requests.post(f"{API_ROOT}/search/oci_rag", json=oci_payload, auth=SESS.auth, timeout=50)
            oci_data = r_oci.json() if r_oci.ok else {}
            answer = oci_data.get("answer", "")
            if answer and "does not support TextGeneration" in answer:
                answer += "<br><span style='color:#c42;font-size:1.03em;'>This OCI model does not support text generation.</span>"
        except Exception as e:
            answer = f"Error querying OCI GenAI: {e}"
    elif llm_source == "AWS Bedrock":
        # Extract model_id from dropdown selection like 'ModelName (modelId)'
        m = re.search(r"\(([^)]+)\)\s*$", str(bedrock_model)) if bedrock_model else None
        selected_mid = m.group(1) if m else (bedrock_model or os.environ.get("BEDROCK_MODEL_ID", ""))
        bed_payload = {
            "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
            "model_id": selected_mid,
            "question": question,
            **params
        }
        try:
            r_bed = requests.post(f"{API_ROOT}/search/bedrock_rag", json=bed_payload, auth=SESS.auth, timeout=60)
            bed_data = r_bed.json() if r_bed.ok else {}
            answer = bed_data.get("answer", "")
        except Exception as e:
            answer = f"Error querying Bedrock: {e}"
    else:
        rag_payload = {
            "question": question,
            **params,
            "model": ollama_model or "llama3",
            "reranker_model": "mxbai-rerank-xsmall",
            "top_k": rag_top_k
        }
        try:
            r = requests.post(f"{API_ROOT}/search/rag", json=rag_payload, auth=SESS.auth, timeout=35)
            rag_data = r.json() if r.ok else {}
            answer = rag_data.get("answer", "")
        except Exception as e:
            answer = f"Error querying Ollama: {e}"
    answer_html = f"<div style='color:#10890b;font-size:1.1em;font-family:Menlo,Monaco,monospace;margin-top:0.7em;white-space:pre-wrap'>{answer or '[No answer returned]'}"
    answer_html += "</div>"
    context_html = format_context_cards(sources, chunk_metadata, context_chunks)
    return answer_html, context_html, sources

def conversational_chat_fn(message, llm_source, ollama_model, oci_model_info_json, bedrock_provider, bedrock_model, top_k, history, system_prompt, temperature, top_p, max_tokens, repeat_penalty):
    chat_history = history or []
    req = {
        "llm_source": "ollama" if llm_source == "Local Ollama" else ("oci_genai" if llm_source == "OCI GenAI" else "bedrock"),
        "model": ollama_model,
        "message": message,
        "chat_history": chat_history,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "top_k": top_k,
        "oci_config": {},
    }
    if llm_source == "AWS Bedrock":
        m = re.search(r"\(([^)]+)\)\s*$", str(bedrock_model)) if bedrock_model else None
        req["model"] = (m.group(1) if m else (bedrock_model or os.environ.get("BEDROCK_MODEL_ID", "")))
    if llm_source == "OCI GenAI":
        try:
            model_info = json.loads(oci_model_info_json) if oci_model_info_json else {}
        except Exception:
            model_info = {}
        req["oci_config"] = {
            "compartment_id": os.environ.get("OCI_COMPARTMENT_OCID", ""),
            "model_id": model_info.get("id") or model_info.get("ocid") or model_info.get("model_id") or os.environ.get("OCI_GENAI_MODEL_OCID", ""),
            "region": os.environ.get("OCI_REGION", "ap-sydney-1"),
        }
    try:
        resp = requests.post(f"{API_ROOT}/chat/conversation", json=req, auth=SESS.auth, timeout=120)
        data = resp.json()
        answer = data.get("answer", "") if isinstance(data, dict) else str(data)
        sources = data.get("sources", [])
        chunk_metadata = data.get("chunk_metadata", [])
        context_chunks = data.get("context_chunks", [])
        formatted_answer = html.escape(answer).replace("\\n", "<br>").replace("\n", "<br>")
    except Exception as e:
        formatted_answer = f"Error querying chatbot API: {e}"
        sources = []
        chunk_metadata = []
        context_chunks = []
    if not isinstance(chat_history, list):
        chat_history = []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({
        "role": "assistant",
        "content": formatted_answer,
        "cards": format_context_cards(sources, chunk_metadata, context_chunks)
    })
    def render_history(hist):
        html_out = "<div class='chatbox-ct'>"
        idx = 0
        while idx < len(hist):
            msg = hist[idx]
            if msg["role"] == "user":
                html_out += f"<div class='bubble user-bubble'><b>User:</b> {msg['content']}</div>"
            elif msg["role"] == "assistant":
                html_out += (
                    "<div class='bubble assistant-bubble'><b>Assistant:</b> "
                    f"{msg['content']}</div>"
                )
                if msg.get("cards"):
                    html_out += f"<div>{msg['cards']}</div>"
            idx += 1
        html_out += "</div>"
        return html_out or "<i>No conversation yet.</i>"
    return render_history(chat_history), chat_history

def parse_agentic_markdown_to_steps(md_answer):
    steps = []
    pattern = re.compile(r"(Step\s+\d+\s*-\s*[^\n:]+:\s*.*?)(?=Step\s+\d+\s*-\s*[^\n:]+:|Final\s+Conclusion:|$)", re.DOTALL|re.IGNORECASE)
    matches = pattern.findall(md_answer)
    for step_block in matches:
        m = re.match(r"Step\s+(\d+)\s*-\s*([^\n:]+):\s*(.+)", step_block, re.DOTALL|re.IGNORECASE)
        if m:
            step_num, label, content = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            steps.append((f"Step {step_num} - {label}", label, content))
    concl = re.search(r"Final\s*Conclusion:\s*(.+)", md_answer, re.DOTALL|re.IGNORECASE)
    if concl:
        steps.append(("Final Conclusion", "Conclusion", concl.group(1).strip()))
    return steps

def agentic_chat_fn(message, llm_source, ollama_model, oci_model_info_json, bedrock_provider, bedrock_model, top_k, history, system_prompt, temperature, top_p, max_tokens, repeat_penalty):
    chat_history = history or []
    req = {
        "llm_source": "ollama" if llm_source == "Local Ollama" else ("oci_genai" if llm_source == "OCI GenAI" else "bedrock"),
        "model": ollama_model,
        "message": message,
        "chat_history": chat_history,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "top_k": top_k,
        "oci_config": {},
    }
    if llm_source == "AWS Bedrock":
        m = re.search(r"\(([^)]+)\)\s*$", str(bedrock_model)) if bedrock_model else None
        req["model"] = (m.group(1) if m else (bedrock_model or os.environ.get("BEDROCK_MODEL_ID", "")))
    if llm_source == "OCI GenAI":
        try:
            model_info = json.loads(oci_model_info_json) if oci_model_info_json else {}
        except Exception:
            model_info = {}
        req["oci_config"] = {
            "compartment_id": os.environ.get("OCI_COMPARTMENT_OCID", ""),
            "model_id": model_info.get("id") or model_info.get("ocid") or model_info.get("model_id") or os.environ.get("OCI_GENAI_MODEL_OCID", ""),
            "region": os.environ.get("OCI_REGION", "ap-sydney-1"),
        }
    try:
        resp = requests.post(f"{API_ROOT}/chat/agentic", json=req, auth=SESS.auth, timeout=180)
        data = resp.json()
        answer = data.get("answer", "") if isinstance(data, dict) else str(data)
        steps = parse_agentic_markdown_to_steps(answer)
        final_ans = ""
        if steps and steps[-1][0].lower().startswith("final conclusion"):
            final_ans = steps[-1][2]
        else:
            final_ans = answer
        step_html = "<div class='agentic-cot-timeline'>"
        for (title, label, content) in steps:
            label_class = {
                "Thought": "cot-thought",
                "Action": "cot-action",
                "Evidence": "cot-evidence",
                "Reasoning": "cot-reason",
                "Conclusion": "cot-conclusion"
            }.get(label.capitalize(), "cot-other")
            step_html += f"""
            <div class='cot-card {label_class}'>
                <div class='cot-card-title'>{html.escape(title)}</div>
                <div class='cot-card-content'>{html.escape(content).replace("\\n", "<br>").replace("\n", "<br>")}</div>
            </div>
            """
        step_html += "</div>"
        answer_html = (
            f"<div class='llm-answer-main'>{html.escape(final_ans).replace('\\n','<br>').replace('\n','<br>')}</div>"
            + step_html
        )
    except Exception as e:
        answer_html = f"Error querying agentic chat API: {e}"

    if not isinstance(chat_history, list):
        chat_history = []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({
        "role": "assistant",
        "content": answer_html
    })
    def render_history(hist):
        html_out = "<div class='chatbox-ct'>"
        if len(hist) >= 2:
            last_user = hist[-2]
            last_assistant = hist[-1]
            if last_user["role"] == "user":
                html_out += f"<div class='bubble user-bubble'><b>User:</b> {last_user['content']}</div>"
            if last_assistant["role"] == "assistant":
                html_out += f"<div class='bubble assistant-bubble'><b>Assistant:</b> {last_assistant['content']}</div>"
        else:
            for msg in hist:
                if msg["role"] == "user":
                    html_out += f"<div class='bubble user-bubble'><b>User:</b> {msg['content']}</div>"
                elif msg["role"] == "assistant":
                    html_out += f"<div class='bubble assistant-bubble'><b>Assistant:</b> {msg['content']}</div>"
        html_out += "</div>"
        return html_out or "<i>No conversation yet.</i>"
    return render_history(chat_history), chat_history

# ===== FTS search wrapper =====

def fts_search_fn(query, top_k):
    try:
        resp = requests.post(
            f"{API_ROOT}/search/fts",
            json={"query": query, "top_k": int(top_k)},
            auth=SESS.auth,
            timeout=15
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        return f"<div style='color:#c22'>FTS Error: {e}</div>"
    if not results:
        return "<div style='color:#888;'><i>No matches found.</i></div>"
    out = ""
    for hit in results:
        out += f"""
        <div class="fts-result-card" style="border:1.1px solid #ccd; border-radius:7px; padding:9px 14px; margin:12px 0;">
          <div><span style="color:#296b8b;font-weight:bold">Source:</span> <span>{html.escape(str(hit.get('source','')))}</span></div>
          <div><span style="color:#1d6842;font-weight:bold">Snippet:</span> <span>{hit.get('snippet','')}</span></div>
          <div><span style="color:#333;font-size:90%;">Doc ID: {hit.get('doc_id','')}</span>
          {'| Chunk Index: '+str(hit.get('chunk_index','')) if hit.get('chunk_index') is not None else ''}</div>
          {f"<div style='color:#aaa;font-size:90%;margin-top:3px;'><b>Metadata:</b> {html.escape(str(hit.get('chunk_metadata','') or ''))}</div>" if hit.get('chunk_metadata') else ""}
        </div>
        """
    return out

# ===== UI =====

with gr.Blocks(title="CogNeo RAG UI", css="""
#llm-answer-box {
    color: #10890b !important;
    font-size: 1.13em;
    font-family: Menlo, Monaco, 'SFMono-Regular', monospace;
    background: #f2fff4 !important;
    border-radius: 7px;
    border: 2px solid #c5ebd3;
    margin-bottom:12px;
    min-height: 32px;
}
.llm-answer-main {
    color: #143b2b !important;
    font-weight: bold;
    font-size: 1.16em;
    margin-bottom: 9px;
    padding-top: 3px;
}
.agentic-cot-timeline {
    display: flex;
    flex-direction: column;
    margin-bottom: 16px;
    gap: 7px;
}
.cot-card {
    border-radius: 7px;
    padding: 8px 13px 6px 13px;
    margin-bottom: 2px;
    font-size: .99em;
    font-family: Menlo, Monaco, 'SFMono-Regular', monospace;
    border: 1.7px solid #ededf7;
}
.cot-card-title {
    font-weight: bold;
    color: #1a469f;
    background: #f5f8ff;
    padding: 2px 7px;
    margin-bottom: 2.2px;
    border-radius: 3px;
    font-size: .99em;
    display: inline-block;
}
.cot-thought    { border-left: 6px solid #3366cc; }
.cot-action     { border-left: 6px solid #378a08; }
.cot-evidence   { border-left: 6px solid #d68611; }
.cot-reason     { border-left: 6px solid #9104b6; }
.cot-conclusion { border-left: 6px solid #b12a2a; }
.cot-other      { border-left: 6px solid #555; }
.chatbox-ct {
    display: flex;
    flex-direction: column;
    gap: 1.2em;
    max-width: 700px;
    margin: 0 auto;
}
.bubble {
    border-radius: 12px;
    padding: 16px 20px;
    max-width: 95%;
    margin: 2px 0;
    font-size: 1.08em;
    box-shadow: 0 2.5px 8px #e7f4e9;
    transition: border .15s;
}
.user-bubble {
    background: #f0f5fd;
    align-self: flex-start;
    border-bottom-left-radius: 0;
}
.assistant-bubble {
    background: #eafbe5;
    align-self: flex-end;
    border-bottom-right-radius: 0;
}
.chat-source-cards-ct {
    display: flex;
    flex-wrap: wrap;
    gap: .7em;
    margin-top: 12px;
}
.chat-source-card {
    border: 1.7px solid #ebeff3;
    background: #fff;
    border-radius: 7px;
    padding: 7.5px 13px 9.5px 13px;
    margin-right: 0;
    min-width: 215px;
    max-width: 315px;
    min-height: 68px;
    font-size: .98em;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.chat-source-cite {
    color: #276188;
    font-weight: bold;
    margin-bottom: 2px;
}
.chat-source-excerpt {
    color: #18311a;
    font-family: Menlo, Monaco, "SFMono-Regular", monospace;
    background: #f8fafd;
    padding: 2.8px 5px 2.8px 5px;
    border-radius: 4px;
}
.spinner {
    display:inline-block;
    width:1.1em;
    height:1.1em;
    border:2.7px solid #abbada;
    border-radius:50%;
    border-top-color:#378a08;
    animation: spin 0.77s linear infinite;
    vertical-align:middle;
}
@keyframes spin {
  to {transform: rotate(360deg);}
}
""") as demo:
    gr.Markdown("# CogNeo RAG Platform")

    login_box = gr.Row(visible=True)
    with login_box:
        gr.Markdown("## Login to continue")
        username = gr.Textbox(label="Username", value="legal_api")
        password = gr.Textbox(label="Password", type="password")
        login_err = gr.Markdown("")
        login_btn = gr.Button("Login")

    with gr.Row(visible=False) as app_panel:
        with gr.Tabs():
            # ===== Hybrid Search tab =====
            with gr.Tab("Hybrid Search"):
                hybrid_query = gr.Textbox(label="Enter a research question", lines=2)
                hybrid_top_k = gr.Number(label="Top K Results", value=10, precision=0)
                hybrid_alpha = gr.Slider(label="Hybrid weighting (0 = keyword, 1 = semantic)", value=0.5, minimum=0.0, maximum=1.0)
                hybrid_btn = gr.Button("Hybrid Search")
                hybrid_results = gr.HTML(label="Results", value="", show_label=False)
                hybrid_btn.click(
                    hybrid_search_fn,
                    inputs=[hybrid_query, hybrid_top_k, hybrid_alpha],
                    outputs=[hybrid_results]
                )
            # ===== RAG tab =====
            with gr.Tab("RAG"):
                gr.Markdown("#### RAG-Powered Chat")
                rag_llm_source = gr.Dropdown(label="LLM Source", choices=["Local Ollama", "OCI GenAI", "AWS Bedrock"], value="Local Ollama")
                rag_ollama_model = gr.Dropdown(label="Ollama Model", choices=[], visible=True)
                rag_oci_model = gr.Dropdown(label="OCI GenAI Model", choices=[], visible=False)
                rag_bedrock_provider = gr.Dropdown(label="AWS Bedrock Provider", choices=[], visible=False)
                rag_bedrock_model = gr.Dropdown(label="AWS Bedrock Model", choices=[], visible=False)

                def update_rag_model_dropdowns(src):
                    if src == "Local Ollama":
                        return (
                            gr.update(choices=fetch_ollama_models(), visible=True, value=None),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False)
                        )
                    elif src == "AWS Bedrock":
                        providers = bedrock_providers()
                        default_provider, default_model_disp = bedrock_default_selection()
                        initial_models = bedrock_models_for_provider(default_provider) if default_provider else []
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(choices=providers, visible=True, value=default_provider),
                            gr.update(choices=initial_models, visible=True, value=default_model_disp)
                        )
                    else:  # OCI GenAI
                        return (
                            gr.update(visible=False),
                            gr.update(choices=[], visible=False),  # keep hidden; configured via env
                            gr.update(visible=False),
                            gr.update(visible=False)
                        )

                def update_rag_bedrock_models(provider):
                    # When provider changes, if default model belongs to this provider select it, else first model
                    models = bedrock_models_for_provider(provider)
                    _, default_model_disp = bedrock_default_selection()
                    default_value = default_model_disp if (default_model_disp and default_model_disp in models) else (models[0] if models else None)
                    return gr.update(choices=models, visible=True, value=default_value)

                rag_llm_source.change(update_rag_model_dropdowns, inputs=[rag_llm_source], outputs=[rag_ollama_model, rag_oci_model, rag_bedrock_provider, rag_bedrock_model])
                rag_bedrock_provider.change(update_rag_bedrock_models, inputs=[rag_bedrock_provider], outputs=[rag_bedrock_model])

                rag_top_k = gr.Number(label="Top K Context Chunks", value=10, precision=0)
                rag_system_prompt = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3)
                rag_temperature = gr.Slider(label="Temperature", value=0.1, minimum=0.0, maximum=1.5, step=0.01)
                rag_top_p = gr.Slider(label="Top P", value=0.9, minimum=0.0, maximum=1.0, step=0.01)
                rag_max_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                rag_repeat_penalty = gr.Slider(label="Repeat Penalty", value=1.1, minimum=0.5, maximum=2.0, step=0.01)

                rag_question = gr.Textbox(label="Enter your question", lines=2)
                rag_ask_btn = gr.Button("Ask")
                rag_answer = gr.HTML(label="Answer", elem_id="llm-answer-box", value="")
                rag_context = gr.HTML(label="Context / Sources", value="", show_label=False)
                rag_ask_btn.click(
                    rag_chatbot,
                    inputs=[rag_question, rag_llm_source, rag_ollama_model, rag_oci_model, rag_bedrock_provider, rag_bedrock_model, rag_top_k, rag_system_prompt, rag_temperature, rag_top_p, rag_max_tokens, rag_repeat_penalty],
                    outputs=[rag_answer, rag_context, gr.State()]
                )
            # ===== FTS tab =====
            with gr.Tab("Full Text Search"):
                gr.Markdown("**Full Text Search** — phrase and stemmed search across documents and/or all indexed metadata fields.")
                fts_q = gr.Textbox(label="Search Query", lines=2)
                fts_top_k = gr.Number(label="Max Results", value=10, precision=0)
                fts_mode = gr.Dropdown(
                    label="Search in",
                    choices=["Both", "Documents", "Chunk Metadata"],
                    value="Both"
                )
                fts_results = gr.HTML(value="", show_label=False)
                def fts_search_fn_user(q, k, mode):
                    mode_map = {"Both": "both", "Documents": "documents", "Chunk Metadata": "metadata"}
                    mode_val = mode_map.get(mode, "both")
                    try:
                        resp = requests.post(
                            f"{API_ROOT}/search/fts",
                            json={"query": q, "top_k": int(k), "mode": mode_val},
                            auth=SESS.auth,
                            timeout=15
                        )
                        resp.raise_for_status()
                        results = resp.json()
                    except Exception as e:
                        return f"<div style='color:#c22'>FTS Error: {e}</div>"
                    if not results:
                        return "<div style='color:#888;'><i>No matches found.</i></div>"
                    out = ""
                    for hit in results:
                        url = None
                        meta = {}
                        val = hit.get('chunk_metadata')
                        if isinstance(val, dict):
                            meta = val
                        elif isinstance(val, str):
                            try:
                                meta = json.loads(val)
                            except Exception:
                                meta = {}
                        else:
                            meta = {}
                        url = meta.get('url')
                        if url:
                            source_html = f'<a href="{html.escape(url)}" target="_blank">{html.escape(url)}</a>'
                        else:
                            source_html = html.escape(str(hit.get("source","")))
                        out += f"""
                        <div class="fts-result-card" style="border:1.1px solid #ccd; border-radius:7px; padding:9px 14px; margin:12px 0;">
                          <div><span style="color:#296b8b;font-weight:bold">Source:</span> <span>{source_html}</span></div>
                          <div><span style="color:#1d6842;font-weight:bold">Snippet:</span> <span>{hit.get('snippet','')}</span></div>
                          <div><span style="color:#333;font-size:90%;">Doc ID: {hit.get('doc_id','')}</span>
                          {'| Chunk Index: '+str(hit.get('chunk_index','')) if hit.get('chunk_index') is not None else ''}</div>
                          {f"<div style='color:#aaa;font-size:90%;margin-top:3px;'><b>Metadata:</b> {html.escape(str(hit.get('chunk_metadata','') or ''))}</div>" if hit.get('chunk_metadata') else ""}
                        </div>
                        """
                    return out
                fts_btn = gr.Button("Full Text Search")
                fts_btn.click(fts_search_fn_user, [fts_q, fts_top_k, fts_mode], [fts_results])

            # ===== Conversational Chat tab =====
            with gr.Tab("Conversational Chat"):
                gr.Markdown("#### Conversational Chatbot (RAG-style per turn)")
                chat_llm_source = gr.Dropdown(label="LLM Source", choices=["Local Ollama", "OCI GenAI", "AWS Bedrock"], value="Local Ollama")
                chat_ollama_model = gr.Dropdown(label="Ollama Model", choices=[], visible=True)
                chat_oci_model = gr.Dropdown(label="OCI GenAI Model", choices=[], visible=False)
                chat_bedrock_provider = gr.Dropdown(label="AWS Bedrock Provider", choices=[], visible=False)
                chat_bedrock_model = gr.Dropdown(label="AWS Bedrock Model", choices=[], visible=False)
                chat_top_k = gr.Number(label="Top K Context Chunks", value=10, precision=0)
                chat_system_prompt = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3)
                chat_temperature = gr.Slider(label="Temperature", value=0.1, minimum=0.0, maximum=1.5, step=0.01)
                chat_top_p = gr.Slider(label="Top P", value=0.9, minimum=0.0, maximum=1.0, step=0.01)
                chat_max_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                chat_repeat_penalty = gr.Slider(label="Repeat Penalty", value=1.1, minimum=0.5, maximum=2.0, step=0.01)
                chat_history = gr.State([])

                def update_chat_model_dropdowns(src):
                    if src == "Local Ollama":
                        return (
                            gr.update(choices=fetch_ollama_models(), visible=True, value=None),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False)
                        )
                    elif src == "AWS Bedrock":
                        providers = bedrock_providers()
                        default_provider, default_model_disp = bedrock_default_selection()
                        initial_models = bedrock_models_for_provider(default_provider) if default_provider else []
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(choices=providers, visible=True, value=default_provider),
                            gr.update(choices=initial_models, visible=True, value=default_model_disp)
                        )
                    else:  # OCI GenAI
                        return (
                            gr.update(visible=False),
                            gr.update(choices=[], visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False)
                        )

                def update_chat_bedrock_models(provider):
                    models = bedrock_models_for_provider(provider)
                    _, default_model_disp = bedrock_default_selection()
                    default_value = default_model_disp if (default_model_disp and default_model_disp in models) else (models[0] if models else None)
                    return gr.update(choices=models, visible=True, value=default_value)

                chat_llm_source.change(update_chat_model_dropdowns, inputs=[chat_llm_source], outputs=[chat_ollama_model, chat_oci_model, chat_bedrock_provider, chat_bedrock_model])
                chat_bedrock_provider.change(update_chat_bedrock_models, inputs=[chat_bedrock_provider], outputs=[chat_bedrock_model])

                chat_message = gr.Textbox(label="Your message", lines=2)
                send_btn = gr.Button("Send")
                conversation_html = gr.HTML(label="Conversation", value="", show_label=False)

                def show_in_progress(*_):
                    return (
                        "<div class='chatbox-ct'><div class='bubble user-bubble'>Sending message...</div><div class='bubble assistant-bubble' style='color:#aaa;'><span class='spinner'></span> Fetching response...</div></div>",
                        gr.update()
                    )

                send_btn.click(
                    show_in_progress,
                    inputs=[chat_message, chat_llm_source, chat_ollama_model, chat_oci_model, chat_bedrock_provider, chat_bedrock_model,
                            chat_top_k, chat_history, chat_system_prompt, chat_temperature, chat_top_p, chat_max_tokens, chat_repeat_penalty],
                    outputs=[conversation_html, chat_history],
                    queue=False
                ).then(
                    conversational_chat_fn,
                    inputs=[chat_message, chat_llm_source, chat_ollama_model, chat_oci_model, chat_bedrock_provider, chat_bedrock_model,
                            chat_top_k, chat_history, chat_system_prompt, chat_temperature, chat_top_p, chat_max_tokens, chat_repeat_penalty],
                    outputs=[conversation_html, chat_history]
                )

            # ===== Agentic RAG tab =====
            with gr.Tab("Agentic RAG"):
                gr.Markdown("#### Agentic RAG/Chain-of-Thought Chat")
                agent_llm_source = gr.Dropdown(label="LLM Source", choices=["Local Ollama", "OCI GenAI", "AWS Bedrock"], value="Local Ollama")
                agent_ollama_model = gr.Dropdown(label="Ollama Model", choices=[], visible=True)
                agent_oci_model = gr.Dropdown(label="OCI GenAI Model", choices=[], visible=False)
                agent_bedrock_provider = gr.Dropdown(label="AWS Bedrock Provider", choices=[], visible=False)
                agent_bedrock_model = gr.Dropdown(label="AWS Bedrock Model", choices=[], visible=False)
                agent_top_k = gr.Number(label="Top K Context Chunks", value=10, precision=0)
                agent_system_prompt = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3)
                agent_temperature = gr.Slider(label="Temperature", value=0.1, minimum=0.0, maximum=1.5, step=0.01)
                agent_top_p = gr.Slider(label="Top P", value=0.9, minimum=0.0, maximum=1.0, step=0.01)
                agent_max_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                agent_repeat_penalty = gr.Slider(label="Repeat Penalty", value=1.1, minimum=0.5, maximum=2.0, step=0.01)
                agent_history = gr.State([])

                def update_agent_model_dropdowns(src):
                    if src == "Local Ollama":
                        return (
                            gr.update(choices=fetch_ollama_models(), visible=True, value=None),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False)
                        )
                    elif src == "AWS Bedrock":
                        providers = bedrock_providers()
                        default_provider, default_model_disp = bedrock_default_selection()
                        initial_models = bedrock_models_for_provider(default_provider) if default_provider else []
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(choices=providers, visible=True, value=default_provider),
                            gr.update(choices=initial_models, visible=True, value=default_model_disp)
                        )
                    else:  # OCI GenAI
                        return (
                            gr.update(visible=False),
                            gr.update(choices=[], visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False)
                        )

                def update_agent_bedrock_models(provider):
                    models = bedrock_models_for_provider(provider)
                    _, default_model_disp = bedrock_default_selection()
                    default_value = default_model_disp if (default_model_disp and default_model_disp in models) else (models[0] if models else None)
                    return gr.update(choices=models, visible=True, value=default_value)

                agent_llm_source.change(update_agent_model_dropdowns, inputs=[agent_llm_source], outputs=[agent_ollama_model, agent_oci_model, agent_bedrock_provider, agent_bedrock_model])
                agent_bedrock_provider.change(update_agent_bedrock_models, inputs=[agent_bedrock_provider], outputs=[agent_bedrock_model])

                agent_message = gr.Textbox(label="Your message", lines=2)
                agent_send_btn = gr.Button("Send")
                agent_conversation_html = gr.HTML(label="Conversation", value="", show_label=False)

                def show_agentic_in_progress(*args):
                    user_msg = str(args[0]) if (args and len(args) > 0 and args[0] is not None) else ""
                    progress_html = (
                        "<div class='chatbox-ct'>"
                        "<div class='bubble user-bubble'><b>User:</b> " + html.escape(user_msg) + "</div>"
                        "<div class='bubble assistant-bubble'><span class='spinner'></span> <span style='color:#888;'>Thinking…</span></div>"
                        "</div>"
                    )
                    return progress_html, []

                agent_send_btn.click(
                    show_agentic_in_progress,
                    inputs=[
                        agent_message, agent_llm_source, agent_ollama_model, agent_oci_model, agent_bedrock_provider, agent_bedrock_model,
                        agent_top_k, agent_history, agent_system_prompt, agent_temperature, agent_top_p, agent_max_tokens, agent_repeat_penalty
                    ],
                    outputs=[agent_conversation_html, agent_history],
                    queue=False
                ).then(
                    agentic_chat_fn,
                    inputs=[
                        agent_message, agent_llm_source, agent_ollama_model, agent_oci_model, agent_bedrock_provider, agent_bedrock_model,
                        agent_top_k, agent_history, agent_system_prompt, agent_temperature, agent_top_p, agent_max_tokens, agent_repeat_penalty
                    ],
                    outputs=[agent_conversation_html, agent_history]
                )

    login_btn.click(
        login_fn,
        inputs=[username, password],
        outputs=[login_box, app_panel, login_err, login_err]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866)
