"""
OpenAI connection factory. Mirrors the ``BedrockConnectionFactory``
shape: ``__init__(config)`` builds a single shared client; ``get()``
hands back an ``OpenAiConnection`` whose ``complete_text`` /
``complete_vision`` / ``embed`` methods return the shared
:class:`LlmCompletion` envelope.

The ``openai`` SDK import is local to ``_build_client`` so installs
that only use Anthropic or Bedrock don't pay the import cost.
"""
import base64
from typing import Any, List, Mapping, Optional

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.errors import LlmConfigError, LlmError, LlmProviderError
from llm_core_lib.types import LlmCompletion


class OpenAiConnection(object):
    """Wraps ``openai.OpenAI`` for chat-completions + embeddings."""

    def __init__(
        self,
        client: Any,
        model_id: str,
        vision_model_id: str,
        embedding_model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self._client = client
        self._model_id = model_id
        self._vision_model_id = vision_model_id
        self._embedding_model = embedding_model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vision_model_id(self) -> str:
        return self._vision_model_id

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def complete_text(
        self, prompt: str, system: Optional[str] = None
    ) -> LlmCompletion:
        messages = self._build_messages(prompt, system, image=None)
        return self._invoke_chat(self._model_id, messages)

    def complete_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        image_mime: str = 'image/png',
        system: Optional[str] = None,
    ) -> LlmCompletion:
        messages = self._build_messages(
            prompt, system, image=(image_bytes, image_mime),
        )
        return self._invoke_chat(self._vision_model_id, messages)

    def embed(self, text: str) -> List[float]:
        if not self._embedding_model:
            raise LlmConfigError(
                'openai connection has no embedding_model configured'
            )
        try:
            raw = self._client.embeddings.create(
                model=self._embedding_model, input=text,
            )
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'openai embed failed: {exc}') from exc
        return list(raw.data[0].embedding)

    def close(self) -> None:
        # The openai SDK client doesn't require explicit close.
        pass

    def _invoke_chat(self, model_id: str, messages: list) -> LlmCompletion:
        payload = {
            'model': model_id,
            'messages': messages,
            'temperature': self._temperature,
        }
        if self._max_tokens:
            payload['max_tokens'] = self._max_tokens
        try:
            raw = self._client.chat.completions.create(**payload)
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'openai chat failed: {exc}') from exc

        choices = getattr(raw, 'choices', None) or []
        if not choices:
            raise LlmProviderError('openai response carried no choices')
        choice = choices[0]
        message = getattr(choice, 'message', None)
        text = ''
        if message is not None:
            text = getattr(message, 'content', None) or ''

        usage_obj = getattr(raw, 'usage', None)
        usage = None
        if usage_obj is not None:
            usage = {
                'prompt_tokens': int(getattr(usage_obj, 'prompt_tokens', 0) or 0),
                'completion_tokens': int(
                    getattr(usage_obj, 'completion_tokens', 0) or 0
                ),
                'total_tokens': int(getattr(usage_obj, 'total_tokens', 0) or 0),
            }

        return LlmCompletion(
            text=text,
            model=getattr(raw, 'model', model_id) or model_id,
            usage=usage,
        )

    @staticmethod
    def _build_messages(
        prompt: str,
        system: Optional[str],
        image: Optional[tuple],
    ) -> list:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        if image is not None:
            image_bytes, image_mime = image
            data_url = (
                f'data:{image_mime};base64,'
                + base64.b64encode(image_bytes).decode('ascii')
            )
            messages.append(
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': data_url}},
                    ],
                }
            )
        else:
            messages.append({'role': 'user', 'content': prompt})
        return messages


class OpenAiConnectionFactory(ConnectionFactory):
    """One OpenAI SDK client per process. ``get()`` returns a fresh
    :class:`OpenAiConnection` wrapping that shared client."""

    def __init__(self, config: Mapping[str, Any]):
        # Either the cross-provider ``model`` or OpenAI-native
        # ``model_id`` selects the chat model. The tests inject
        # ``client`` directly so the SDK import path stays untouched.
        model_id = config.get('model') or config.get('model_id')
        if not model_id:
            raise LlmConfigError(
                'openai connection requires model (or model_id)'
            )
        injected_client = config.get('client')
        if injected_client is None and not config.get('api_key'):
            raise LlmConfigError('openai connection requires api_key')

        self._config = config
        self._model_id = model_id
        self._vision_model_id = (
            config.get('vision_model') or config.get('vision_model_id') or model_id
        )
        self._embedding_model = (
            config.get('embedding_model')
            or config.get('embedding_model_id')
            or ''
        )
        self._max_tokens = int(config.get('max_tokens', 4096))
        self._temperature = float(config.get('temperature', 0.0))
        self._client = injected_client or self._build_client(config)

    def get(self, *args, **kwargs) -> OpenAiConnection:
        return OpenAiConnection(
            self._client,
            self._model_id,
            self._vision_model_id,
            self._embedding_model,
            self._max_tokens,
            self._temperature,
        )

    @staticmethod
    def _build_client(config: Mapping[str, Any]) -> Any:
        # Integration-only path; unit tests inject ``client`` so the
        # openai SDK isn't a hard test dep.
        try:  # pragma: no cover — requires the real openai SDK
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LlmConfigError(
                'openai SDK not installed; pip install openai'
            ) from exc
        kwargs = {'api_key': config['api_key']}  # pragma: no cover
        if config.get('base_url'):  # pragma: no cover
            kwargs['base_url'] = config['base_url']
        if config.get('organization'):  # pragma: no cover
            kwargs['organization'] = config['organization']
        return OpenAI(**kwargs)  # pragma: no cover
