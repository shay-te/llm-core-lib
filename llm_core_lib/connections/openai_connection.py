"""``OpenAiConnection`` — per-call wrapper around an ``openai.OpenAI`` client.

The companion :class:`OpenAiConnectionFactory` in
``openai_connection_factory.py`` owns SDK-client construction; this
module owns only the per-call request/response shape so callers can
read it in isolation.
"""
import base64
from typing import Any, List, Optional

from core_lib.connection.connection import Connection

from llm_core_lib.errors import LlmConfigError, LlmError, LlmProviderError
from llm_core_lib.types import LlmCompletion


class OpenAiConnection(Connection):
    """Wraps ``openai.OpenAI`` for chat-completions + embeddings.

    A fresh connection is returned by every
    :meth:`OpenAiConnectionFactory.get` call but the underlying SDK
    client is shared — connections are cheap, the client is not.

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

    def __enter__(self) -> 'OpenAiConnection':
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        # Always call close(); let any exception propagate by returning None.
        self.close()

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
            # OpenAI vision: text first, image second per their docs.
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
