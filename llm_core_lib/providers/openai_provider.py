"""OpenAI chat-completions adapter.

The ``openai`` SDK is imported lazily — installs that only use Anthropic
or Bedrock don't pay for it at import time. Tests pass a fake client
through the ``client`` kwarg to avoid touching the network or
requiring the SDK to be installed.
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


class OpenAiLlmProvider(LlmProvider):
    """``openai.OpenAI().chat.completions.create``-shaped adapter."""

    id = 'openai'

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        if not api_key and client is None:
            # A test client implies "use the injected fake regardless of
            # api_key"; only require the key when we'd build the real SDK.
            raise LlmConfigError('openai provider requires api_key')
        if not model:
            raise LlmConfigError('openai provider requires model')
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._organization = organization
        self._client = client

    def _resolved_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover — env-dependent
            raise LlmConfigError(
                'openai SDK not installed; pip install openai'
            ) from exc
        kwargs = {'api_key': self._api_key}
        if self._base_url:
            kwargs['base_url'] = self._base_url
        if self._organization:
            kwargs['organization'] = self._organization
        self._client = OpenAI(**kwargs)
        return self._client

    def chat(self, request: LlmChatRequest) -> LlmChatResponse:
        messages = [
            {'role': m.role, 'content': m.content} for m in request.messages
        ]
        if request.system:
            # Prepend a system message — openai chat-completions accepts
            # `system` only via the messages array.
            messages = [{'role': 'system', 'content': request.system}] + messages

        payload = {
            'model': request.model or self._model,
            'messages': messages,
        }
        if request.temperature is not None:
            payload['temperature'] = request.temperature
        if request.max_tokens is not None:
            payload['max_tokens'] = request.max_tokens
        if request.stop is not None:
            payload['stop'] = request.stop
        if request.extra:
            payload.update(request.extra)

        try:
            raw = self._resolved_client().chat.completions.create(**payload)
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrap any SDK exception
            raise LlmProviderError(f'openai chat failed: {exc}') from exc

        return _normalize_openai_response(raw, payload['model'])


def _normalize_openai_response(raw: Any, requested_model: str) -> LlmChatResponse:
    choices = getattr(raw, 'choices', None) or []
    if not choices:
        raise LlmProviderError('openai response carried no choices')
    choice = choices[0]
    message = getattr(choice, 'message', None)
    content = getattr(message, 'content', None) or '' if message is not None else ''
    finish_reason = getattr(choice, 'finish_reason', None)

    usage_obj = getattr(raw, 'usage', None)
    usage = None
    if usage_obj is not None:
        usage = LlmUsage(
            prompt_tokens=int(getattr(usage_obj, 'prompt_tokens', 0) or 0),
            completion_tokens=int(getattr(usage_obj, 'completion_tokens', 0) or 0),
            total_tokens=int(getattr(usage_obj, 'total_tokens', 0) or 0),
        )

    return LlmChatResponse(
        content=content,
        model=getattr(raw, 'model', requested_model) or requested_model,
        finish_reason=finish_reason,
        usage=usage,
    )
