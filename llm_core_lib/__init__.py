# Top-level re-exports: consumers (library-core-lib + downstream) write
# ``from llm_core_lib import BedrockConnectionFactory`` instead of the
# deep submodule path. Removing one of these IS a public-API break.
from llm_core_lib.connections.anthropic_connection_factory import AnthropicConnectionFactory
from llm_core_lib.connections.bedrock_connection_factory import BedrockConnectionFactory
from llm_core_lib.connections.openai_connection_factory import OpenAiConnectionFactory

__version__ = '0.2.0'

__all__ = [
    '__version__',
    'AnthropicConnectionFactory',
    'BedrockConnectionFactory',
    'OpenAiConnectionFactory',
]
