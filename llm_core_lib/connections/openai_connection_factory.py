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
        # 1. fetch — every value, one canonical key per value, no fallback.
        model = config.get('model')
        vision_model = config.get('vision_model')
        embedding_model = config.get('embedding_model')
        max_tokens = config.get('max_tokens')
        temperature = config.get('temperature')
        injected_client = config.get('client')
        api_key = config.get('api_key')

        # 2. validate — every required key fails fast if missing. Strings
        # use truthy-check (rejects None AND empty string); numerics use
        # is-None (so e.g. max_tokens=0 stays valid).
        if not model:
            raise LlmConfigError('openai connection requires model')
        if not vision_model:
            raise LlmConfigError('openai connection requires vision_model')
        if not embedding_model:
            raise LlmConfigError('openai connection requires embedding_model')
        if max_tokens is None:
            raise LlmConfigError('openai connection requires max_tokens')
        if temperature is None:
            raise LlmConfigError('openai connection requires temperature')
        # api_key is only required when no SDK client is injected; tests
        # inject ``client`` directly to skip the SDK build path.
        if injected_client is None and not api_key:
            raise LlmConfigError('openai connection requires api_key')

        # 3. use
        self._config = config
        self._model_id = model
        self._vision_model_id = vision_model
        self._embedding_model = embedding_model
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
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

    def raw_client(self) -> Any:
        """Return the shared underlying ``openai.OpenAI`` client.

        Escape hatch for callers that need provider-native features the
        normalized connection surface intentionally does not cover —
        e.g. the OpenAI Responses API with tool/function calling. Prefer
        ``get()`` + ``complete_text`` / ``embed`` for ordinary use.
        """
        return self._client

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

        api_key = config.get('api_key')  # pragma: no cover
        base_url = config.get('base_url')  # pragma: no cover
        organization = config.get('organization')  # pragma: no cover

        if api_key is None:  # pragma: no cover
            raise LlmConfigError('openai connection requires api_key')
        if base_url is None:  # pragma: no cover
            raise LlmConfigError('openai connection requires base_url')
        if organization is None:  # pragma: no cover
            raise LlmConfigError('openai connection requires organization')

        return OpenAI(  # pragma: no cover
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
