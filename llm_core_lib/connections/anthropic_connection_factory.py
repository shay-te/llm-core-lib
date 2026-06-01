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

from llm_core_lib.connections.anthropic_connection import AnthropicConnection
from llm_core_lib.errors import LlmConfigError


class AnthropicConnectionFactory(ConnectionFactory):
    """One Anthropic SDK client per process. ``get()`` returns a fresh
    :class:`AnthropicConnection` wrapping that shared client."""

    def __init__(self, config: DictConfig):
        # 1. fetch — every value, one canonical key per value, no fallback.
        model = config.get('model')
        vision_model = config.get('vision_model')
        max_tokens = config.get('max_tokens')
        temperature = config.get('temperature')
        injected_client = config.get('client')
        api_key = config.get('api_key')

        # 2. validate — every required key fails fast if missing. Strings
        # use truthy-check (rejects None AND empty string); numerics use
        # is-None (so e.g. max_tokens=0 stays valid).
        if not model:
            raise LlmConfigError('anthropic connection requires model')
        if not vision_model:
            raise LlmConfigError('anthropic connection requires vision_model')
        if max_tokens is None:
            raise LlmConfigError('anthropic connection requires max_tokens')
        if temperature is None:
            raise LlmConfigError('anthropic connection requires temperature')
        # api_key is only required when no SDK client is injected; tests
        # inject ``client`` directly to skip the SDK build path.
        if injected_client is None and not api_key:
            raise LlmConfigError('anthropic connection requires api_key')

        # 3. use
        self._config = config
        self._model_id = model
        self._vision_model_id = vision_model
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
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
