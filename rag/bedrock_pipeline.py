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
            # Generic text generation request (model-specific request bodies vary across providers)
            req = {
                "modelId": self.model_id,
                "contentType": "application/json",
                "accept": "application/json",
                "body": (
                    '{"inputText": ' + repr(prompt) + ', "textGenerationConfig": '
                    + '{"temperature": %s, "topP": %s, "maxTokenCount": %s}}'
                ) % (float(temperature), float(top_p), int(max_tokens))
            }
            resp = self.client.invoke_model(**req)
            import json as _json
            data = _json.loads(resp.get("body").read().decode("utf-8"))
            # Common fields; Bedrock providers may return slightly different envelopes
            answer = data.get("outputText") or data.get("generation") or str(data)
        except Exception as e:
            answer = f"Error querying Bedrock: {e}"

        return {
            "answer": answer,
            "sources": sources or [],
            "contexts": context_chunks or [],
            "chunk_metadata": chunk_metadata or [],
        }
