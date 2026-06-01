"""Build a backend ``*ConnectionFactory`` from an
:class:`LlmConnectionConfig`.

This is the single seam that knows the union of supported backends.
Tests can still construct individual factories directly (passing a
fake ``client`` through the config dict), but anything reading the
cross-provider :class:`LlmConnectionConfig` shape should go through
here so backend selection lives in one place.
"""
from __future__ import annotations

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.connections.anthropic_connection_factory import (
    AnthropicConnectionFactory,
)
from llm_core_lib.connections.bedrock_connection_factory import (
    BedrockConnectionFactory,
)
from llm_core_lib.connections.openai_connection_factory import (
    OpenAiConnectionFactory,
)
from llm_core_lib.errors import LlmInvalidProviderError
from llm_core_lib.types import LLM_PROVIDER_IDS, LlmConnectionConfig


def create_connection_factory(
    config: LlmConnectionConfig,
) -> ConnectionFactory:
    """Return the backend ``*ConnectionFactory`` for ``config.provider``.

    Raises:
        LlmInvalidProviderError: ``config.provider`` isn't one of
            :data:`llm_core_lib.types.LLM_PROVIDER_IDS`.
        LlmConfigError: the matched factory rejected the config
            (e.g. OpenAI without ``api_key``, Bedrock without
            ``region``).
    """
    provider_id = (config.provider or '').strip().lower()
    factory_config = config.as_factory_config()

    if provider_id == 'openai':
        return OpenAiConnectionFactory(factory_config)
    if provider_id == 'anthropic':
        return AnthropicConnectionFactory(factory_config)
    if provider_id == 'bedrock':
        return BedrockConnectionFactory(factory_config)

    raise LlmInvalidProviderError(
        f'unknown provider {config.provider!r}; '
        f'supported: {list(LLM_PROVIDER_IDS)}'
    )
