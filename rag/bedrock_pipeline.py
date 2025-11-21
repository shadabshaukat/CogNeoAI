"""
AWS Bedrock RAG pipeline for CogNeo.
- Uses boto3 bedrock-runtime to invoke the selected model.
- Formats context, sources, and metadata similarly to other RAG pipelines.
"""

from __future__ import annotations
import os
from typing import List, Optional, Any, Dict

try:
    import boto3  # type: ignore
except ImportError:
    boto3 = None


class BedrockPipeline:
    def __init__(self, region: Optional[str] = None, model_id: Optional[str] = None, profile: Optional[str] = None):
        """
        Args:
            region: AWS region where Bedrock is available (e.g., us-east-1)
            model_id: Bedrock model ID (e.g., anthropic.claude-3-haiku-20240307-v1:0)
            profile: Optional AWS profile name for credential resolution
        """
        if not boto3:
            raise ImportError("boto3 is not installed. Please add 'boto3' to requirements.txt and pip install.")
        region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        sess_kwargs = {}
        if profile:
            sess_kwargs["profile_name"] = profile
        session = boto3.Session(region_name=region, **sess_kwargs)
        self.client = session.client("bedrock-runtime")
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

    def _provider(self) -> str:
        mid = (self.model_id or "").lower()
        if mid.startswith("anthropic."):
            return "anthropic"
        if mid.startswith("meta."):
            return "meta"
        if mid.startswith("mistral."):
            return "mistral"
        if mid.startswith("cohere."):
            return "cohere"
        if mid.startswith("amazon.") or "titan" in mid:
            return "titan"
        return "unknown"

    def _format_blocks(self, context_chunks: Optional[List[str]], chunk_metadata: Optional[List[Dict[str, Any]]]) -> List[str]:
        blocks: List[str] = []
        if chunk_metadata is None:
            return context_chunks or []
        else:
            for text, meta in zip(context_chunks or [], chunk_metadata or []):
                meta_str = "\n".join(f"{k}: {v}" for k, v in (meta or {}).items())
                block = (meta_str + "\n---\n" if meta_str else "") + (text or "")
                blocks.append(block)
        return blocks

    def query(
        self,
        question: str,
        context_chunks: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
        custom_prompt: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        repeat_penalty: float = 1.1,
        chat_history: Optional[List[dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Calls AWS Bedrock with context+question, returns dict with answer, sources, and context metadata.
        """
        sys_prompt = custom_prompt or system_prompt or "You are a helpful RAG assistant. Answer from the provided context and cite sources."
        blocks = self._format_blocks(context_chunks, chunk_metadata)
        context = "\n\n---\n\n".join(blocks)
        prompt = f"{sys_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"

        try:
            import json as _json

            provider = self._provider()
            body_dict: Dict[str, Any]

            if provider == "anthropic":
                # Claude 3 family expects messages + anthropic_version
                body_dict = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                }
            elif provider == "meta":
                # Meta Llama 3 instruct accepts "prompt", max_gen_len, temperature, top_p
                inst_prompt = f"[INST] <<SYS>>{sys_prompt}<</SYS>>\n{context}\n\n{question} [/INST]"
                body_dict = {
                    "prompt": inst_prompt,
                    "max_gen_len": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                }
            elif provider == "mistral":
                # Mistral instruct accepts "prompt", max_tokens, temperature, top_p
                body_dict = {
                    "prompt": prompt,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                }
            elif provider == "cohere":
                # Cohere Command family - bedrock often expects "prompt" & "max_tokens"
                body_dict = {
                    "prompt": prompt,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    # Cohere uses "p" not top_p; pass both to be safe if backend ignores unknown
                    "p": float(top_p),
                    "top_p": float(top_p),
                }
            else:
                # Default to Titan Text format
                body_dict = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "temperature": float(temperature),
                        "topP": float(top_p),
                        "maxTokenCount": int(max_tokens),
                    }
                }

            req = {
                "modelId": self.model_id,
                "contentType": "application/json",
                "accept": "application/json",
                "body": _json.dumps(body_dict),
            }

            resp = self.client.invoke_model(**req)
            data = _json.loads(resp.get("body").read().decode("utf-8"))

            # Parse provider-specific responses
            answer = None
            if provider == "anthropic":
                # content: [{"type":"text","text":"..."}]
                try:
                    content = data.get("content") or []
                    if content and isinstance(content, list):
                        first = content[0]
                        if isinstance(first, dict):
                            answer = first.get("text")
                except Exception:
                    answer = None
            if not answer and "generation" in data and isinstance(data.get("generation"), str):
                answer = data.get("generation")
            if not answer and "outputText" in data:
                answer = data.get("outputText")
            if not answer and "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
                # Mistral often returns {"outputs":[{"text":"..."}], ...}
                out0 = data["outputs"][0]
                if isinstance(out0, dict):
                    answer = out0.get("text") or out0.get("outputText")
            if not answer and "generations" in data and isinstance(data["generations"], list) and data["generations"]:
                # Cohere often returns {"generations":[{"text":"..."}]}
                answer = data["generations"][0].get("text")

            if not answer:
                answer = str(data)
        except Exception as e:
            answer = f"Error querying Bedrock: {e}"

        return {
            "answer": answer,
            "sources": sources or [],
            "contexts": context_chunks or [],
            "chunk_metadata": chunk_metadata or [],
        }
