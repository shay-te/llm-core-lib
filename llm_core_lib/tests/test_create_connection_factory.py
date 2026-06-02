"""Tests for :func:`llm_core_lib.create_connection_factory`."""
import unittest

from llm_core_lib import (
    AnthropicConnectionFactory,
    BedrockConnectionFactory,
    LlmConfigError,
    LlmConnectionConfig,
    LlmInvalidProviderError,
    OpenAiConnectionFactory,
    create_connection_factory,
)
from llm_core_lib.tests.mock.anthropic_client import MockAnthropicClient
from llm_core_lib.tests.mock.bedrock_client import MockBedrockClient
from llm_core_lib.tests.mock.openai_client import MockOpenAIClient


# Fake clients are injected through ``extra={'client': ...}`` so these
# tests stay deterministic regardless of which SDKs are installed.

# Every config requires the full keyset now — the factories fail fast on
# anything missing. ``_full(...)`` returns a config that satisfies them.
def _full(provider, **overrides):
    base = {
        'id': '_',
        'provider': provider,
        'model': 'm',
        'vision_model': 'm',
        'embedding_model': 'emb',
        'max_tokens': 4096,
        'temperature': 0.0,
    }
    base.update(overrides)
    return LlmConnectionConfig(**base)


class TestCreateConnectionFactoryHappyPath(unittest.TestCase):
    def test_openai(self):
        factory = create_connection_factory(_full(
            'openai', model='gpt-4o-mini', vision_model='gpt-4o-mini',
            api_key='sk-test', extra={'client': MockOpenAIClient()},
        ))
        self.assertIsInstance(factory, OpenAiConnectionFactory)

    def test_anthropic(self):
        factory = create_connection_factory(_full(
            'anthropic',
            model='claude-3-5-sonnet-latest',
            vision_model='claude-3-5-sonnet-latest',
            api_key='sk-test',
            extra={'client': MockAnthropicClient()},
        ))
        self.assertIsInstance(factory, AnthropicConnectionFactory)

    def test_bedrock(self):
        factory = create_connection_factory(_full(
            'bedrock',
            model='anthropic.claude-3-5-sonnet-20241022-v2:0',
            vision_model='anthropic.claude-3-5-sonnet-20241022-v2:0',
            region='us-east-1',
            extra={'client': MockBedrockClient()},
        ))
        self.assertIsInstance(factory, BedrockConnectionFactory)

    def test_provider_name_is_case_insensitive(self):
        factory = create_connection_factory(_full(
            'OpenAI', model='gpt-4o-mini', vision_model='gpt-4o-mini',
            api_key='sk-test', extra={'client': MockOpenAIClient()},
        ))
        self.assertIsInstance(factory, OpenAiConnectionFactory)


class TestRawClientExposesInjectedClient(unittest.TestCase):
    """`raw_client()` returns the same shared SDK client `get()` wraps —
    the escape hatch host apps use for provider-native tool-calling."""

    def test_openai_raw_client_is_injected_client(self):
        client = MockOpenAIClient()
        factory = create_connection_factory(_full(
            'openai', model='gpt-4o-mini', vision_model='gpt-4o-mini',
            api_key='sk-test', extra={'client': client},
        ))
        self.assertIs(factory.raw_client(), client)

    def test_anthropic_raw_client_is_injected_client(self):
        client = MockAnthropicClient()
        factory = create_connection_factory(_full(
            'anthropic', model='claude-x', vision_model='claude-x',
            api_key='sk-test', extra={'client': client},
        ))
        self.assertIs(factory.raw_client(), client)

    def test_bedrock_raw_client_is_injected_client(self):
        client = MockBedrockClient()
        factory = create_connection_factory(_full(
            'bedrock', model='anthropic.claude-x', vision_model='anthropic.claude-x',
            region='us-east-1', extra={'client': client},
        ))
        self.assertIs(factory.raw_client(), client)


class TestCreateConnectionFactoryRejects(unittest.TestCase):
    def test_unknown_provider_raises_invalid_provider(self):
        with self.assertRaises(LlmInvalidProviderError):
            create_connection_factory(_full('cohere'))

    def test_empty_provider_raises_invalid_provider(self):
        with self.assertRaises(LlmInvalidProviderError):
            create_connection_factory(_full(''))

    def test_openai_without_api_key_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_connection_factory(_full('openai', model='gpt-4o-mini'))

    def test_anthropic_without_api_key_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_connection_factory(_full('anthropic', model='claude-x'))

    def test_bedrock_without_region_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_connection_factory(_full('bedrock', model='anthropic.claude-x'))

    def test_openai_without_model_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_connection_factory(_full(
                'openai', model='', api_key='sk-x',
            ))


if __name__ == '__main__':
    unittest.main()
