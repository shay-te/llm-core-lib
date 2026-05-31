"""
AWS Bedrock connection factory. All LLM / vision / embedding calls go
through this edge — the model is swappable via config (region, model_id,
embedding_model). Services depend only on the small interface exposed by
``BedrockConnection``.

The boto3 Bedrock SDK import is local to ``_build_client`` so the rest of
the codebase stays SDK-free.

Mirrors the pattern from ``library-core-lib`` so consumers that already
use that style move over without API churn — and so the cross-provider
``LlmConnectionRegistry`` can hand back a uniform ``ConnectionFactory``
regardless of backend.
"""
from typing import Any, List, Mapping, Optional

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.errors import LlmConfigError, LlmError, LlmProviderError
from llm_core_lib.types import LlmCompletion


class BedrockConnection(object):
    """
    Thin provider-abstracted wrapper. The methods are intentionally narrow
    — chat completion, vision, and embedding generation. Anything more
    exotic (tool use, streaming, etc.) belongs in a dedicated client at
    the edge, not in services.
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
        body = self._build_messages_body(prompt, system, image_bytes, image_mime)
        text, usage = self._invoke(self._vision_model_id, body)
        return LlmCompletion(text=text, model=self._vision_model_id, usage=usage)

    def embed(self, text: str) -> List[float]:
        import json

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
        # Titan returns `embedding`; Cohere-on-Bedrock returns
        # `embeddings`. Accept either, fall back to empty list if the
        # model returns neither (defensive — log + caller can react).
        return payload.get('embedding') or (payload.get('embeddings') or [[]])[0]

    def close(self) -> None:
        # boto3 clients don't require explicit close.
        pass

    def _invoke(self, model_id: str, body_dict: dict) -> tuple:
        import json

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
        text = _extract_text(payload)
        usage = payload.get('usage') if isinstance(payload, dict) else None
        return text, usage

    def _build_messages_body(
        self,
        prompt: str,
        system: Optional[str],
        image_bytes: Optional[bytes],
        image_mime: str = 'image/png',
    ) -> dict:
        import base64

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
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': self._max_tokens,
            'temperature': self._temperature,
            'messages': [{'role': 'user', 'content': content}],
        }
        if system:
            body['system'] = system
        return body


def _extract_text(payload: Any) -> str:
    # Bedrock Anthropic-shaped response — content is a list of blocks.
    if isinstance(payload, dict) and isinstance(payload.get('content'), list):
        return ''.join(
            part.get('text', '')
            for part in payload['content']
            if isinstance(part, dict)
        )
    # Legacy / Titan-shaped fallback.
    if isinstance(payload, dict):
        return (
            payload.get('completion')
            or (payload.get('results') or [{}])[0].get('outputText', '')
            or ''
        )
    return ''


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        import base64

        return base64.b64encode(obj).decode('ascii')
    raise TypeError(
        f'Object of type {type(obj).__name__} is not JSON serializable'
    )


class BedrockConnectionFactory(ConnectionFactory):
    """
    One Bedrock runtime client per process. ``get()`` hands back a
    ``BedrockConnection`` configured with the model ids from the config
    so callers can do ``conn.complete_text(...)`` / ``conn.embed(...)``
    without knowing about boto3.
    """

    def __init__(self, config: Mapping[str, Any]):
        if not config.get('region'):
            raise LlmConfigError('bedrock connection requires region')
        # Accept either ``model_id`` (Bedrock-native) or ``model`` (the
        # cross-provider registry key) so the same factory works
        # whether you wire it from a hand-rolled Bedrock yaml or from
        # an LlmConnectionConfig that uses ``model``.
        model_id = config.get('model_id') or config.get('model')
        if not model_id:
            raise LlmConfigError(
                'bedrock connection requires model_id (or model)'
            )
        self._config = config
        self._model_id = model_id
        self._vision_model_id = (
            config.get('vision_model_id') or config.get('vision_model') or model_id
        )
        self._embedding_model = (
            config.get('embedding_model')
            or config.get('embedding_model_id')
            or ''
        )
        self._max_tokens = int(config.get('max_tokens', 4096))
        self._temperature = float(config.get('temperature', 0.0))
        self._client = config.get('client') or self._build_client(config)

    def get(self, *args, **kwargs) -> BedrockConnection:
        return BedrockConnection(
            self._client,
            self._model_id,
            self._vision_model_id,
            self._embedding_model,
            self._max_tokens,
            self._temperature,
        )

    @staticmethod
    def _build_client(config: Mapping[str, Any]) -> Any:
        # Integration-only path; unit tests inject ``client`` in the
        # config so boto3 isn't a hard test dep.
        import boto3  # pragma: no cover — requires boto3 + real AWS creds

        kwargs = {  # pragma: no cover
            'service_name': 'bedrock-runtime',
            'region_name': config.get('region', 'us-east-1'),
        }
        if config.get('endpoint_url'):  # pragma: no cover
            kwargs['endpoint_url'] = config['endpoint_url']
        if config.get('access_key') and config.get('secret_key'):  # pragma: no cover
            kwargs['aws_access_key_id'] = config['access_key']
            kwargs['aws_secret_access_key'] = config['secret_key']
        return boto3.client(**kwargs)  # pragma: no cover
