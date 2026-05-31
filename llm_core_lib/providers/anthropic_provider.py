"""Anthropic Messages-API adapter.

The ``anthropic`` SDK is imported lazily for the same reason as the
OpenAI adapter — and the tests inject a fake client so we never need
the SDK installed to exercise this module.
"""
from __future__ import annotations

from typing import Any, Optional

from llm_core_lib.errors import LlmConfigError, LlmProviderError, LlmError
from llm_core_lib.provider import LlmProvider
from llm_core_lib.types import (
    LlmChatRequest,
    LlmChatResponse,
    LlmUsage,
)


_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


class AnthropicLlmProvider(LlmProvider):
    """``anthropic.Anthropic().messages.create``-shaped adapter."""

    id = 'anthropic'

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        if not api_key and client is None:
            raise LlmConfigError('anthropic provider requires api_key')
        if not model:
            raise LlmConfigError('anthropic provider requires model')
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = client

    def _resolved_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover — env-dependent
            raise LlmConfigError(
                'anthropic SDK not installed; pip install anthropic'
            ) from exc
        kwargs = {'api_key': self._api_key}
        if self._base_url:
            kwargs['base_url'] = self._base_url
        self._client = Anthropic(**kwargs)
        return self._client

    def chat(self, request: LlmChatRequest) -> LlmChatResponse:
        # Anthropic's API takes the system prompt as a top-level field,
        # NOT as a message in the messages array. Everything else maps
        # 1:1 to role/content.
        messages = [
            {'role': m.role, 'content': m.content} for m in request.messages
        ]
        kwargs = {
            'model': request.model or self._model,
            'messages': messages,
            'max_tokens': request.max_tokens or _ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        if request.system:
            kwargs['system'] = request.system
        if request.temperature is not None:
            kwargs['temperature'] = request.temperature
        if request.stop is not None:
            kwargs['stop_sequences'] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        if request.extra:
            kwargs.update(request.extra)

        try:
            raw = self._resolved_client().messages.create(**kwargs)
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'anthropic chat failed: {exc}') from exc

        return _normalize_anthropic_response(raw, kwargs['model'])


def _normalize_anthropic_response(raw: Any, requested_model: str) -> LlmChatResponse:
    # Anthropic returns ``content`` as a list of content blocks; we
    # concatenate every text block in order.
    text_parts = []
    for block in getattr(raw, 'content', None) or []:
        if getattr(block, 'type', None) == 'text':
            text_parts.append(getattr(block, 'text', '') or '')
    content = ''.join(text_parts)

    usage_obj = getattr(raw, 'usage', None)
    usage = None
    if usage_obj is not None:
        input_tokens = int(getattr(usage_obj, 'input_tokens', 0) or 0)
        output_tokens = int(getattr(usage_obj, 'output_tokens', 0) or 0)
        usage = LlmUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    return LlmChatResponse(
        content=content,
        model=getattr(raw, 'model', requested_model) or requested_model,
        finish_reason=getattr(raw, 'stop_reason', None),
        usage=usage,
    )
