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
        # 1. fetch
        model_id = config.get('model') or config.get('model_id')
        vision_model_id = (
            config.get('vision_model') or config.get('vision_model_id') or model_id
        )
        max_tokens = int(config.get('max_tokens', ANTHROPIC_DEFAULT_MAX_TOKENS))
        temperature = float(config.get('temperature', 0.0))
        injected_client = config.get('client')
        api_key = config.get('api_key')

        # 2. validate
        if not model_id:
            raise LlmConfigError(
                'anthropic connection requires model (or model_id)'
            )
        if injected_client is None and not api_key:
            raise LlmConfigError('anthropic connection requires api_key')

        # 3. use
        self._config = config
        self._model_id = model_id
        self._vision_model_id = vision_model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
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

        api_key = config.get('api_key')  # pragma: no cover
        base_url = config.get('base_url')  # pragma: no cover

        if api_key is None:  # pragma: no cover
            raise LlmConfigError('anthropic connection requires api_key')
        if base_url is None:  # pragma: no cover
            raise LlmConfigError('anthropic connection requires base_url')

        return Anthropic(api_key=api_key, base_url=base_url)  # pragma: no cover
