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
        # 1. fetch — every value, one canonical key per value, no fallback.
        region = config.get('region')
        model = config.get('model')
        vision_model = config.get('vision_model')
        embedding_model = config.get('embedding_model')
        max_tokens = config.get('max_tokens')
        temperature = config.get('temperature')
        injected_client = config.get('client')

        # 2. validate — every required key fails fast if missing. Strings
        # use truthy-check (rejects None AND empty string); numerics use
        # is-None (so e.g. max_tokens=0 stays valid).
        if not region:
            raise LlmConfigError('bedrock connection requires region')
        if not model:
            raise LlmConfigError('bedrock connection requires model')
        if not vision_model:
            raise LlmConfigError('bedrock connection requires vision_model')
        if not embedding_model:
            raise LlmConfigError('bedrock connection requires embedding_model')
        if max_tokens is None:
            raise LlmConfigError('bedrock connection requires max_tokens')
        if temperature is None:
            raise LlmConfigError('bedrock connection requires temperature')

        # 3. use — assign + lazy-build the SDK client.
        self._config = config
        self._model_id = model
        self._vision_model_id = vision_model
        self._embedding_model = embedding_model
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
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
