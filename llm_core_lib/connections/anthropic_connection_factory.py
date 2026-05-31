"""``AnthropicConnectionFactory`` — builds the shared
``anthropic.Anthropic`` client once, hands back per-call
:class:`AnthropicConnection` instances.

The Connection class lives in :mod:`anthropic_connection`; this module
only owns SDK-client construction + config validation.

The ``anthropic`` SDK import is local to ``_build_client``; tests
inject a fake client via ``config['client']``.
"""
from typing import Any

from omegaconf import DictConfig

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.connections.anthropic_connection import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    AnthropicConnection,
)
from llm_core_lib.errors import LlmConfigError


class AnthropicConnectionFactory(ConnectionFactory):
    """One Anthropic SDK client per process. ``get()`` returns a fresh
    :class:`AnthropicConnection` wrapping that shared client."""

    def __init__(self, config: DictConfig):
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
            config.get('max_tokens', ANTHROPIC_DEFAULT_MAX_TOKENS)
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
    def _build_client(config: DictConfig) -> Any:
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
