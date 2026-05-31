"""Build an :class:`LlmProvider` from an :class:`LlmProviderConfig`.

The factory is the *only* code path that converts a config into a
concrete adapter. Tests can still construct adapters directly (and pass
their own ``client=`` for mocking), but anything reading config should
go through here so validation lives in one place.
"""
from __future__ import annotations

from llm_core_lib.errors import LlmInvalidProviderError
from llm_core_lib.provider import LlmProvider
from llm_core_lib.providers.anthropic_provider import AnthropicLlmProvider
from llm_core_lib.providers.bedrock_provider import BedrockLlmProvider
from llm_core_lib.providers.openai_provider import OpenAiLlmProvider
from llm_core_lib.types import LLM_PROVIDER_IDS, LlmProviderConfig


def create_llm_provider(config: LlmProviderConfig) -> LlmProvider:
    """Return the adapter that matches ``config.provider``.

    Raises:
        LlmInvalidProviderError: ``config.provider`` is not one of
            :data:`llm_core_lib.types.LLM_PROVIDER_IDS`.
        LlmConfigError: the matched adapter rejected the config (e.g.
            OpenAI without ``api_key``, Bedrock without ``region``).
    """
    provider_id = (config.provider or '').strip().lower()

    if provider_id == 'openai':
        return OpenAiLlmProvider(
            api_key=config.api_key or '',
            model=config.model,
            base_url=config.base_url,
            organization=config.organization,
        )

    if provider_id == 'anthropic':
        return AnthropicLlmProvider(
            api_key=config.api_key or '',
            model=config.model,
            base_url=config.base_url,
        )

    if provider_id == 'bedrock':
        return BedrockLlmProvider(
            region=config.region or '',
            model=config.model,
            access_key=config.access_key,
            secret_key=config.secret_key,
            endpoint_url=config.endpoint_url,
        )

    raise LlmInvalidProviderError(
        f'unknown provider {config.provider!r}; '
        f'supported: {list(LLM_PROVIDER_IDS)}'
    )
