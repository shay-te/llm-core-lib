"""Shared LLM provider abstraction.

Public surface (everything below is importable directly from
``llm_core_lib``):

    LlmCoreLib                — composition root (subclass of ``CoreLib``)
    LlmConnectionRegistry     — in-memory named connection registry
    create_llm_provider       — factory: config → adapter
    LlmProvider               — base ABC every adapter implements

    OpenAiLlmProvider         — OpenAI adapter
    AnthropicLlmProvider      — Anthropic adapter
    BedrockLlmProvider        — AWS Bedrock adapter (Anthropic-shaped body)

    LlmProviderId             — Literal['openai', 'anthropic', 'bedrock']
    LLM_PROVIDER_IDS          — runtime tuple of provider ids
    LlmRole                   — Literal of message roles
    LlmMessage                — one chat message
    LlmChatRequest            — chat call input
    LlmChatResponse           — normalized chat response
    LlmStreamEvent            — stream event envelope
    LlmUsage                  — normalized token counts
    LlmProviderConfig         — factory input
    LlmConnectionConfig       — registry input (provider config + id)

    LlmError                  — base exception
    LlmConfigError            — bad provider / connection config
    LlmInvalidProviderError   — factory got an unknown provider id
    LlmDuplicateConnectionError — registry got a duplicate id
    LlmMissingConnectionError — registry asked for an unregistered id
    LlmProviderError          — wrapped error from an underlying SDK call
"""

__version__ = '0.1.0'

from llm_core_lib.errors import (
    LlmConfigError,
    LlmDuplicateConnectionError,
    LlmError,
    LlmInvalidProviderError,
    LlmMissingConnectionError,
    LlmProviderError,
)
from llm_core_lib.factory import create_llm_provider
from llm_core_lib.llm_core_lib import LlmCoreLib
from llm_core_lib.provider import LlmProvider
from llm_core_lib.providers.anthropic_provider import AnthropicLlmProvider
from llm_core_lib.providers.bedrock_provider import BedrockLlmProvider
from llm_core_lib.providers.openai_provider import OpenAiLlmProvider
from llm_core_lib.registry import LlmConnectionRegistry
from llm_core_lib.types import (
    LLM_PROVIDER_IDS,
    LlmChatRequest,
    LlmChatResponse,
    LlmConnectionConfig,
    LlmMessage,
    LlmProviderConfig,
    LlmProviderId,
    LlmRole,
    LlmStreamEvent,
    LlmUsage,
)

__all__ = [
    '__version__',
    # composition root + registry + factory
    'LlmCoreLib',
    'LlmConnectionRegistry',
    'create_llm_provider',
    # interface + adapters
    'LlmProvider',
    'OpenAiLlmProvider',
    'AnthropicLlmProvider',
    'BedrockLlmProvider',
    # types
    'LLM_PROVIDER_IDS',
    'LlmProviderId',
    'LlmRole',
    'LlmMessage',
    'LlmChatRequest',
    'LlmChatResponse',
    'LlmStreamEvent',
    'LlmUsage',
    'LlmProviderConfig',
    'LlmConnectionConfig',
    # errors
    'LlmError',
    'LlmConfigError',
    'LlmInvalidProviderError',
    'LlmDuplicateConnectionError',
    'LlmMissingConnectionError',
    'LlmProviderError',
]
