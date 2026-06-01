"""``BedrockConnectionFactory`` — builds the shared
``boto3.client('bedrock-runtime')`` once, hands back per-call
:class:`BedrockConnection` instances.

The Connection class lives in :mod:`bedrock_connection`; this module
only owns SDK-client construction + config validation.

The boto3 import is local to ``_build_client`` so installs that only
use OpenAI or Anthropic don't pay the import cost.
"""
from typing import Any

from omegaconf import DictConfig

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

    def __init__(self, config: DictConfig):
        # 1. fetch — pull everything out of config into named locals first.
        # Accept either ``model_id`` (Bedrock-native) or ``model`` (the
        # cross-provider registry key) so the same factory works whether
        # you wire it from a hand-rolled Bedrock yaml or from an
        # LlmConnectionConfig that uses ``model``.
        region = config.get('region')
        model_id = config.get('model_id') or config.get('model')
        vision_model_id = (
            config.get('vision_model_id') or config.get('vision_model') or model_id
        )
        embedding_model = (
            config.get('embedding_model')
            or config.get('embedding_model_id')
            or ''
        )
        max_tokens = int(config.get('max_tokens', 4096))
        temperature = float(config.get('temperature', 0.0))
        injected_client = config.get('client')

        # 2. validate — every required value, in one place.
        if not region:
            raise LlmConfigError('bedrock connection requires region')
        if not model_id:
            raise LlmConfigError(
                'bedrock connection requires model_id (or model)'
            )

        # 3. use — assign + lazy-build the SDK client.
        self._config = config
        self._model_id = model_id
        self._vision_model_id = vision_model_id
        self._embedding_model = embedding_model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = injected_client or self._build_client(config)

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
    def _build_client(config: DictConfig) -> Any:
        # Integration-only path; unit tests inject ``client`` in the
        # config so boto3 isn't a hard test dep.
        import boto3  # pragma: no cover — requires boto3 + real AWS creds

        region = config.get('region')  # pragma: no cover
        endpoint_url = config.get('endpoint_url')  # pragma: no cover
        access_key = config.get('access_key')  # pragma: no cover
        secret_key = config.get('secret_key')  # pragma: no cover

        if region is None:  # pragma: no cover
            raise LlmConfigError('bedrock connection requires region')
        if endpoint_url is None:  # pragma: no cover
            raise LlmConfigError('bedrock connection requires endpoint_url')
        if access_key is None:  # pragma: no cover
            raise LlmConfigError('bedrock connection requires access_key')
        if secret_key is None:  # pragma: no cover
            raise LlmConfigError('bedrock connection requires secret_key')

        return boto3.client(  # pragma: no cover
            service_name='bedrock-runtime',
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
