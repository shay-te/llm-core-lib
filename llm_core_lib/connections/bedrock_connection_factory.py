"""``BedrockConnectionFactory`` — builds the shared
``boto3.client('bedrock-runtime')`` once, hands back per-call
:class:`BedrockConnection` instances.

The Connection class lives in :mod:`bedrock_connection`; this module
only owns SDK-client construction + config validation.

The boto3 import is local to ``_build_client`` so installs that only
use OpenAI or Anthropic don't pay the import cost.
"""
from typing import Any, Mapping

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.connections.bedrock_connection import BedrockConnection
from llm_core_lib.errors import LlmConfigError


class BedrockConnectionFactory(ConnectionFactory):
    """
    One Bedrock runtime client per process. ``get()`` hands back a
    :class:`BedrockConnection` configured with the model ids from the
    config so callers can do ``conn.complete_text(...)`` /
    ``conn.embed(...)`` without knowing about boto3.
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
