"""``AnthropicConnection`` — per-call wrapper around an
``anthropic.Anthropic`` client.

The companion :class:`AnthropicConnectionFactory` in
``anthropic_connection_factory.py`` owns SDK-client construction.

Anthropic has no native embeddings endpoint, so ``embed`` is
deliberately absent — callers wanting embeddings should register an
OpenAI or Bedrock connection alongside.
"""
import base64
from typing import Any, Optional

from core_lib.connection.connection import Connection

from llm_core_lib.errors import LlmError, LlmProviderError
from llm_core_lib.types import LlmCompletion


ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


class AnthropicConnection(Connection):
    """Wraps ``anthropic.Anthropic`` for the Messages API.

    Implements the ``core_lib.connection.Connection`` context-manager
    contract so callers can write::

        with factory.get() as conn:
            completion = conn.complete_text('hello')
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        vision_model_id: str,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ):
        self._client = client
        self._model_id = model_id
        self._vision_model_id = vision_model_id
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vision_model_id(self) -> str:
        return self._vision_model_id

    def complete_text(
        self, prompt: str, system: Optional[str] = None
    ) -> LlmCompletion:
        content = [{'type': 'text', 'text': prompt}]
        return self._invoke(self._model_id, content, system)

    def complete_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        image_mime: str = 'image/png',
        system: Optional[str] = None,
    ) -> LlmCompletion:
        # Anthropic vision: image first, text second per their docs.
        content = [
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': image_mime,
                    'data': base64.b64encode(image_bytes).decode('ascii'),
                },
            },
            {'type': 'text', 'text': prompt},
        ]
        return self._invoke(self._vision_model_id, content, system)

    def close(self) -> None:
        # The anthropic SDK client doesn't require explicit close.
        pass

    def __enter__(self) -> 'AnthropicConnection':
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        # Always call close(); let any exception propagate by returning None.
        self.close()

    def _invoke(
        self, model_id: str, content: list, system: Optional[str],
    ) -> LlmCompletion:
        kwargs = {
            'model': model_id,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': self._max_tokens,
            'temperature': self._temperature,
        }
        if system:
            kwargs['system'] = system

        try:
            raw = self._client.messages.create(**kwargs)
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'anthropic chat failed: {exc}') from exc

        text = ''.join(
            getattr(block, 'text', '') or ''
            for block in (getattr(raw, 'content', None) or [])
            if getattr(block, 'type', None) == 'text'
        )

        usage_obj = getattr(raw, 'usage', None)
        usage = None
        if usage_obj is not None:
            input_tokens = int(getattr(usage_obj, 'input_tokens', 0) or 0)
            output_tokens = int(getattr(usage_obj, 'output_tokens', 0) or 0)
            usage = {
                'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
            }

        return LlmCompletion(
            text=text,
            model=getattr(raw, 'model', model_id) or model_id,
            usage=usage,
        )
