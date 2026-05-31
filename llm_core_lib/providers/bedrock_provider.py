"""AWS Bedrock adapter (Anthropic-shaped body).

v1 targets Anthropic models on Bedrock — Claude 3.x family — because
that's both the most common deploy and the spec's example model id
(``anthropic.claude-3-5-sonnet-20241022-v2:0``). Other model families
on Bedrock have different request/response shapes; adding them is
adapter-local future work.

``boto3`` is imported lazily; tests pass a fake client through the
``client`` kwarg.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from llm_core_lib.errors import LlmConfigError, LlmProviderError, LlmError
from llm_core_lib.provider import LlmProvider
from llm_core_lib.types import (
    LlmChatRequest,
    LlmChatResponse,
    LlmUsage,
)


_BEDROCK_ANTHROPIC_VERSION = 'bedrock-2023-05-31'
_BEDROCK_DEFAULT_MAX_TOKENS = 4096


class BedrockLlmProvider(LlmProvider):
    """``boto3.client('bedrock-runtime').invoke_model``-shaped adapter."""

    id = 'bedrock'

    def __init__(
        self,
        region: str,
        model: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        if not region and client is None:
            raise LlmConfigError('bedrock provider requires region')
        if not model:
            raise LlmConfigError('bedrock provider requires model')
        self._region = region
        self._model = model
        self._access_key = access_key
        self._secret_key = secret_key
        self._endpoint_url = endpoint_url
        self._client = client

    def _resolved_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover — env-dependent
            raise LlmConfigError(
                'boto3 not installed; pip install boto3'
            ) from exc
        kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': self._region,
        }
        if self._endpoint_url:
            kwargs['endpoint_url'] = self._endpoint_url
        if self._access_key and self._secret_key:
            kwargs['aws_access_key_id'] = self._access_key
            kwargs['aws_secret_access_key'] = self._secret_key
        self._client = boto3.client(**kwargs)
        return self._client

    def chat(self, request: LlmChatRequest) -> LlmChatResponse:
        model_id = request.model or self._model
        body = _build_anthropic_body(request)

        try:
            raw = self._resolved_client().invoke_model(
                modelId=model_id,
                body=json.dumps(body).encode('utf-8'),
                contentType='application/json',
                accept='application/json',
            )
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'bedrock invoke_model failed: {exc}') from exc

        return _normalize_bedrock_response(raw, model_id)


def _build_anthropic_body(request: LlmChatRequest) -> dict:
    messages = [
        {
            'role': m.role,
            'content': [{'type': 'text', 'text': m.content}],
        }
        for m in request.messages
    ]
    body = {
        'anthropic_version': _BEDROCK_ANTHROPIC_VERSION,
        'max_tokens': request.max_tokens or _BEDROCK_DEFAULT_MAX_TOKENS,
        'messages': messages,
    }
    if request.system:
        body['system'] = request.system
    if request.temperature is not None:
        body['temperature'] = request.temperature
    if request.stop is not None:
        body['stop_sequences'] = (
            request.stop if isinstance(request.stop, list) else [request.stop]
        )
    if request.extra:
        body.update(request.extra)
    return body


def _normalize_bedrock_response(raw: Any, requested_model: str) -> LlmChatResponse:
    # invoke_model returns a dict with a streaming-body ``body`` key
    # whose ``.read()`` yields the full JSON. Mirror that contract in
    # the fake client used by tests.
    body = raw['body'].read()
    payload = json.loads(body)

    text = ''
    content = payload.get('content')
    if isinstance(content, list):
        text = ''.join(
            part.get('text', '') for part in content if isinstance(part, dict)
        )
    elif payload.get('completion'):
        text = payload['completion']

    usage_payload = payload.get('usage') or {}
    usage = None
    if usage_payload:
        input_tokens = int(usage_payload.get('input_tokens', 0) or 0)
        output_tokens = int(usage_payload.get('output_tokens', 0) or 0)
        usage = LlmUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    return LlmChatResponse(
        content=text,
        model=requested_model,
        finish_reason=payload.get('stop_reason'),
        usage=usage,
    )
