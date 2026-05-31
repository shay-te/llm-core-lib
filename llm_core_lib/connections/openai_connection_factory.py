"""``OpenAiConnectionFactory`` — builds the shared ``openai.OpenAI``
client once, hands back per-call :class:`OpenAiConnection` instances.

The Connection class lives in :mod:`openai_connection` so the per-call
shape can be read in isolation; this module only owns SDK-client
construction + config validation.

The ``openai`` SDK import is local to ``_build_client`` so installs
that only use Anthropic or Bedrock don't pay the import cost.
"""
from typing import Any

from omegaconf import DictConfig

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.connections.openai_connection import OpenAiConnection
from llm_core_lib.errors import LlmConfigError


class OpenAiConnectionFactory(ConnectionFactory):
    """One OpenAI SDK client per process. ``get()`` returns a fresh
    :class:`OpenAiConnection` wrapping that shared client."""

    def __init__(self, config: DictConfig):
        # Either the cross-provider ``model`` or OpenAI-native
        # ``model_id`` selects the chat model. Tests inject ``client``
        # directly so the SDK import path stays untouched.
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
    def _build_client(config: DictConfig) -> Any:
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
