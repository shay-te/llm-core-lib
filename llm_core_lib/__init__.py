"""Shared LLM provider abstraction.

Public surface (everything below is importable directly from
``llm_core_lib``):

    LlmCoreLib                — composition root (subclass of ``CoreLib``)
    LlmConnectionRegistry     — in-memory named connection registry
    create_connection_factory — factory: config → ConnectionFactory

    OpenAiConnectionFactory   — OpenAI factory
    OpenAiConnection          — OpenAI per-call wrapper
    AnthropicConnectionFactory — Anthropic factory
    AnthropicConnection       — Anthropic per-call wrapper
    BedrockConnectionFactory  — AWS Bedrock factory
    BedrockConnection         — Bedrock per-call wrapper

    LlmCompletion             — normalized completion envelope
    LlmConnectionConfig       — registry input
    LlmProviderId             — Literal['openai', 'anthropic', 'bedrock']
    LLM_PROVIDER_IDS          — runtime tuple of provider ids

    LlmError                  — base exception
    LlmConfigError            — bad provider / connection config
    LlmInvalidProviderError   — factory got an unknown provider id
    LlmDuplicateConnectionError — registry got a duplicate id
    LlmMissingConnectionError — registry asked for an unregistered id
    LlmProviderError          — wrapped error from an underlying SDK call

Mirrors ``library-core-lib``'s ``BedrockConnectionFactory`` shape so a
caller who already knows that pattern picks this up unchanged.
"""

__version__ = '0.2.0'

from llm_core_lib.connections.anthropic_connection import AnthropicConnection
from llm_core_lib.connections.anthropic_connection_factory import (
    AnthropicConnectionFactory,
)
from llm_core_lib.connections.bedrock_connection import BedrockConnection
from llm_core_lib.connections.bedrock_connection_factory import (
    BedrockConnectionFactory,
)
from llm_core_lib.connections.openai_connection import OpenAiConnection
from llm_core_lib.connections.openai_connection_factory import (
    OpenAiConnectionFactory,
)
from llm_core_lib.errors import (
    LlmConfigError,
    LlmDuplicateConnectionError,
    LlmError,
    LlmInvalidProviderError,
    LlmMissingConnectionError,
    LlmProviderError,
)
from llm_core_lib.factory import create_connection_factory
from llm_core_lib.llm_core_lib import LlmCoreLib
from llm_core_lib.registry import LlmConnectionRegistry
from llm_core_lib.types import (
    LLM_PROVIDER_IDS,
    LlmCompletion,
    LlmConnectionConfig,
    LlmProviderId,
)

__all__ = [
    '__version__',
    # composition root + registry + factory
    'LlmCoreLib',
    'LlmConnectionRegistry',
    'create_connection_factory',
    # connection factories + per-call connections
    'OpenAiConnectionFactory',
    'OpenAiConnection',
    'AnthropicConnectionFactory',
    'AnthropicConnection',
    'BedrockConnectionFactory',
    'BedrockConnection',
    # types
    'LLM_PROVIDER_IDS',
    'LlmProviderId',
    'LlmCompletion',
    'LlmConnectionConfig',
    # errors
    'LlmError',
    'LlmConfigError',
    'LlmInvalidProviderError',
    'LlmDuplicateConnectionError',
    'LlmMissingConnectionError',
    'LlmProviderError',
]
