"""
Anthropic Messages-API connection factory. Mirrors the
``BedrockConnectionFactory`` / ``OpenAiConnectionFactory`` shape so the
cross-provider registry hands callers a uniform interface.

The ``anthropic`` SDK import is local to ``_build_client``; tests
inject a fake client via ``config['client']``.

Note: native Anthropic has no embeddings endpoint, so
``AnthropicConnection`` deliberately omits ``embed`` — callers wanting
embeddings should register an OpenAI or Bedrock connection alongside.
"""
import base64
from typing import Any, Mapping, Optional

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.errors import LlmConfigError, LlmError, LlmProviderError
from llm_core_lib.types import LlmCompletion


_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


class AnthropicConnection(object):
    """Wraps ``anthropic.Anthropic`` for the Messages API."""

    def __init__(
        self,
        client: Any,
        model_id: str,
        vision_model_id: str,
        max_tokens: int = _ANTHROPIC_DEFAULT_MAX_TOKENS,
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


class AnthropicConnectionFactory(ConnectionFactory):
    """One Anthropic SDK client per process. ``get()`` returns a fresh
    :class:`AnthropicConnection` wrapping that shared client."""

    def __init__(self, config: Mapping[str, Any]):
        model_id = config.get('model') or config.get('model_id')
        if not model_id:
            raise LlmConfigError(
                'anthropic connection requires model (or model_id)'
            )
        injected_client = config.get('client')
        if injected_client is None and not config.get('api_key'):
            raise LlmConfigError('anthropic connection requires api_key')

        self._config = config
        self._model_id = model_id
        self._vision_model_id = (
            config.get('vision_model') or config.get('vision_model_id') or model_id
        )
        self._max_tokens = int(
            config.get('max_tokens', _ANTHROPIC_DEFAULT_MAX_TOKENS)
        )
        self._temperature = float(config.get('temperature', 0.0))
        self._client = injected_client or self._build_client(config)

    def get(self, *args, **kwargs) -> AnthropicConnection:
        return AnthropicConnection(
            self._client,
            self._model_id,
            self._vision_model_id,
            self._max_tokens,
            self._temperature,
        )

    @staticmethod
    def _build_client(config: Mapping[str, Any]) -> Any:
        # Integration-only path; unit tests inject ``client``.
        try:  # pragma: no cover — requires the real anthropic SDK
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise LlmConfigError(
                'anthropic SDK not installed; pip install anthropic'
            ) from exc
        kwargs = {'api_key': config['api_key']}  # pragma: no cover
        if config.get('base_url'):  # pragma: no cover
            kwargs['base_url'] = config['base_url']
        return Anthropic(**kwargs)  # pragma: no cover
