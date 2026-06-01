"""``BedrockConnection`` — per-call wrapper around a ``boto3``
``bedrock-runtime`` client (Anthropic-shaped request body).

The companion :class:`BedrockConnectionFactory` in
``bedrock_connection_factory.py`` owns SDK-client construction.
"""
import base64
import json
from typing import Any, List, Optional, Tuple

from core_lib.connection.connection import Connection

from llm_core_lib.errors import LlmConfigError, LlmError, LlmProviderError
from llm_core_lib.types import LlmCompletion


_ANTHROPIC_VERSION = 'bedrock-2023-05-31'


class BedrockConnection(Connection):
    """
    Thin provider-abstracted wrapper. The methods are intentionally narrow
    — chat completion, vision, and embedding generation. Anything more
    exotic (tool use, streaming, etc.) belongs in a dedicated client at
    the edge, not in services.

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
        body = self._build_messages_body(prompt, system, image_bytes=None)
        text, usage = self._invoke(self._model_id, body)
        return LlmCompletion(text=text, model=self._model_id, usage=usage)

    def complete_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        image_mime: str = 'image/png',
        system: Optional[str] = None,
    ) -> LlmCompletion:
        # Bedrock (Anthropic-shaped) vision: image first, text second.
        body = self._build_messages_body(prompt, system, image_bytes, image_mime)
        text, usage = self._invoke(self._vision_model_id, body)
        return LlmCompletion(text=text, model=self._vision_model_id, usage=usage)

    def embed(self, text: str) -> List[float]:
        if not self._embedding_model:
            # Mirror the OpenAI adapter — raise a clear config error
            # rather than letting boto3 fail with an empty modelId.
            raise LlmConfigError(
                'bedrock connection has no embedding_model configured'
            )

        body = json.dumps({'inputText': text}).encode('utf-8')
        try:
            resp = self._client.invoke_model(
                modelId=self._embedding_model,
                body=body,
                contentType='application/json',
                accept='application/json',
            )
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'bedrock embed failed: {exc}') from exc
        payload = json.loads(resp['body'].read())
        # 1. fetch — every key we may consume out of the JSON response.
        # Titan returns `embedding`; Cohere-on-Bedrock returns
        # `embeddings`. Pull both up front, decide which shape we got
        # next, then return last.
        embedding = payload.get('embedding')
        embeddings_array = payload.get('embeddings')

        # 2. decide which shape is populated.
        if embedding:
            chosen = embedding
        elif embeddings_array:
            chosen = embeddings_array[0]
        else:
            chosen = []

        # 3. use.
        return chosen

    def close(self) -> None:
        # boto3 clients don't require explicit close.
        pass

    def __enter__(self) -> 'BedrockConnection':
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        # Always call close(); let any exception propagate by returning None.
        self.close()

    def _invoke(self, model_id: str, body_dict: dict) -> Tuple[str, Optional[dict]]:
        try:
            resp = self._client.invoke_model(
                modelId=model_id,
                body=json.dumps(body_dict, default=_json_default).encode('utf-8'),
                contentType='application/json',
                accept='application/json',
            )
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'bedrock invoke_model failed: {exc}') from exc

        payload = json.loads(resp['body'].read())
        # 1. fetch — pull every field _invoke touches out of the
        # response payload into named locals.
        payload_is_dict = isinstance(payload, dict)
        usage = payload.get('usage') if payload_is_dict else None

        # 2. normalize text via the shape-aware extractor.
        text = _extract_text(payload)

        # 3. use.
        return text, usage

    def _build_messages_body(
        self,
        prompt: str,
        system: Optional[str],
        image_bytes: Optional[bytes],
        image_mime: str = 'image/png',
    ) -> dict:
        content = []
        if image_bytes is not None:
            content.append(
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': image_mime,
                        'data': base64.b64encode(image_bytes).decode('ascii'),
                    },
                }
            )
        content.append({'type': 'text', 'text': prompt})

        body = {
            'anthropic_version': _ANTHROPIC_VERSION,
            'max_tokens': self._max_tokens,
            'temperature': self._temperature,
            'messages': [{'role': 'user', 'content': content}],
        }
        if system:
            body['system'] = system
        return body


def _extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ''

    # 1. fetch — every key we might consume, pulled into named locals.
    content_blocks = payload.get('content')
    legacy_completion = payload.get('completion')
    titan_results = payload.get('results')

    # 2. decide which response shape is populated.
    # Bedrock Anthropic-shaped → content is a list of blocks.
    if isinstance(content_blocks, list):
        text_parts = []
        for part in content_blocks:
            if isinstance(part, dict):
                part_text = part.get('text', '')
                text_parts.append(part_text)
        return ''.join(text_parts)

    # Legacy / Titan-shaped fallback.
    if legacy_completion:
        return legacy_completion
    if titan_results:
        first_result = titan_results[0] if titan_results else {}
        first_output = first_result.get('outputText', '') if isinstance(first_result, dict) else ''
        return first_output or ''

    # 3. use — no recognised shape.
    return ''


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('ascii')
    raise TypeError(
        f'Object of type {type(obj).__name__} is not JSON serializable'
    )
