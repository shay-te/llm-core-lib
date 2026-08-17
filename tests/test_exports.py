"""Lock the public surface — every public symbol must be importable from
its module (the package uses deep imports; the root ``__init__`` is
version-only, matching the other core libs)."""
import unittest


class TestPublicExports(unittest.TestCase):
    def test_factories_and_root_are_classes(self):
        from llm_core_lib.connections.anthropic_connection_factory import AnthropicConnectionFactory
        from llm_core_lib.connections.bedrock_connection_factory import BedrockConnectionFactory
        from llm_core_lib.connections.openai_connection_factory import OpenAiConnectionFactory
        from llm_core_lib.llm_core_lib import LlmCoreLib
        from llm_core_lib.registry import LlmConnectionRegistry

        for cls in (
            LlmCoreLib,
            LlmConnectionRegistry,
            OpenAiConnectionFactory,
            AnthropicConnectionFactory,
            BedrockConnectionFactory,
        ):
            self.assertTrue(isinstance(cls, type), f'{cls!r} should be a class')

    def test_connections_are_classes(self):
        from llm_core_lib.connections.anthropic_connection import AnthropicConnection
        from llm_core_lib.connections.bedrock_connection import BedrockConnection
        from llm_core_lib.connections.openai_connection import OpenAiConnection

        for cls in (OpenAiConnection, AnthropicConnection, BedrockConnection):
            self.assertTrue(isinstance(cls, type), f'{cls!r} should be a class')

    def test_types_and_errors_import(self):
        from llm_core_lib.types import (
            LLM_PROVIDER_IDS,
            LlmCompletion,
            LlmConnectionConfig,
            LlmProviderId,
        )
        from llm_core_lib.errors import (
            LlmConfigError,
            LlmDuplicateConnectionError,
            LlmError,
            LlmInvalidProviderError,
            LlmMissingConnectionError,
            LlmProviderError,
        )
        self.assertTrue(isinstance(LlmCompletion, type))
        self.assertTrue(issubclass(LlmConfigError, LlmError))

    def test_create_connection_factory_is_callable(self):
        from llm_core_lib.factory import create_connection_factory
        self.assertTrue(callable(create_connection_factory))

    def test_provider_ids_match_literal(self):
        from llm_core_lib.types import LLM_PROVIDER_IDS
        self.assertEqual(set(LLM_PROVIDER_IDS), {'openai', 'anthropic', 'bedrock'})

    def test_version_is_exposed(self):
        import llm_core_lib
        self.assertTrue(isinstance(llm_core_lib.__version__, str))


if __name__ == '__main__':
    unittest.main()
