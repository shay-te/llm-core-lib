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
from llm_core_lib.tests.fakes import (
    FakeAnthropicClient,
    FakeBedrockClient,
    FakeOpenAIClient,
)


# Fake clients are injected through ``extra={'client': ...}`` so these
# tests stay deterministic regardless of which SDKs are installed.

class TestCreateConnectionFactoryHappyPath(unittest.TestCase):
    def test_openai(self):
        factory = create_connection_factory(LlmConnectionConfig(
            id='_', provider='openai', model='gpt-4o-mini', api_key='sk-test',
            extra={'client': FakeOpenAIClient()},
        ))
        self.assertIsInstance(factory, OpenAiConnectionFactory)

    def test_anthropic(self):
        factory = create_connection_factory(LlmConnectionConfig(
            id='_', provider='anthropic',
            model='claude-3-5-sonnet-latest', api_key='sk-test',
            extra={'client': FakeAnthropicClient()},
        ))
        self.assertIsInstance(factory, AnthropicConnectionFactory)

    def test_bedrock(self):
        factory = create_connection_factory(LlmConnectionConfig(
            id='_', provider='bedrock',
            model='anthropic.claude-3-5-sonnet-20241022-v2:0',
            region='us-east-1',
            extra={'client': FakeBedrockClient()},
        ))
        self.assertIsInstance(factory, BedrockConnectionFactory)

    def test_provider_name_is_case_insensitive(self):
        factory = create_connection_factory(LlmConnectionConfig(
            id='_', provider='OpenAI', model='gpt-4o-mini', api_key='sk-test',
            extra={'client': FakeOpenAIClient()},
        ))
        self.assertIsInstance(factory, OpenAiConnectionFactory)


class TestCreateConnectionFactoryRejects(unittest.TestCase):
    def test_unknown_provider_raises_invalid_provider(self):
        with self.assertRaises(LlmInvalidProviderError):
            create_connection_factory(LlmConnectionConfig(
                id='_', provider='cohere', model='c4',
            ))

    def test_empty_provider_raises_invalid_provider(self):
        with self.assertRaises(LlmInvalidProviderError):
            create_connection_factory(LlmConnectionConfig(
                id='_', provider='', model='x',
            ))

    def test_openai_without_api_key_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_connection_factory(LlmConnectionConfig(
                id='_', provider='openai', model='gpt-4o-mini',
            ))

    def test_anthropic_without_api_key_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_connection_factory(LlmConnectionConfig(
                id='_', provider='anthropic', model='claude-x',
            ))

    def test_bedrock_without_region_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_connection_factory(LlmConnectionConfig(
                id='_', provider='bedrock', model='anthropic.claude-x',
            ))

    def test_openai_without_model_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_connection_factory(LlmConnectionConfig(
                id='_', provider='openai', model='', api_key='sk-x',
            ))


if __name__ == '__main__':
    unittest.main()
