"""Tests for :func:`llm_core_lib.create_llm_provider`."""
import unittest

from llm_core_lib import (
    AnthropicLlmProvider,
    BedrockLlmProvider,
    LlmConfigError,
    LlmInvalidProviderError,
    LlmProviderConfig,
    OpenAiLlmProvider,
    create_llm_provider,
)


class TestFactoryHappyPath(unittest.TestCase):
    def test_openai(self):
        provider = create_llm_provider(
            LlmProviderConfig(
                provider='openai',
                model='gpt-4o-mini',
                api_key='sk-test',
            )
        )
        self.assertIsInstance(provider, OpenAiLlmProvider)
        self.assertEqual(provider.id, 'openai')

    def test_anthropic(self):
        provider = create_llm_provider(
            LlmProviderConfig(
                provider='anthropic',
                model='claude-3-5-sonnet-latest',
                api_key='sk-test',
            )
        )
        self.assertIsInstance(provider, AnthropicLlmProvider)
        self.assertEqual(provider.id, 'anthropic')

    def test_bedrock(self):
        provider = create_llm_provider(
            LlmProviderConfig(
                provider='bedrock',
                model='anthropic.claude-3-5-sonnet-20241022-v2:0',
                region='us-east-1',
            )
        )
        self.assertIsInstance(provider, BedrockLlmProvider)
        self.assertEqual(provider.id, 'bedrock')

    def test_provider_name_is_case_insensitive(self):
        provider = create_llm_provider(
            LlmProviderConfig(
                provider='OpenAI',
                model='gpt-4o-mini',
                api_key='sk-test',
            )
        )
        self.assertIsInstance(provider, OpenAiLlmProvider)


class TestFactoryRejectsBadInput(unittest.TestCase):
    def test_unknown_provider_raises_invalid_provider(self):
        with self.assertRaises(LlmInvalidProviderError):
            create_llm_provider(
                LlmProviderConfig(provider='cohere', model='c4')
            )

    def test_empty_provider_raises_invalid_provider(self):
        with self.assertRaises(LlmInvalidProviderError):
            create_llm_provider(LlmProviderConfig(provider='', model='x'))

    def test_openai_without_api_key_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_llm_provider(
                LlmProviderConfig(provider='openai', model='gpt-4o-mini')
            )

    def test_anthropic_without_api_key_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_llm_provider(
                LlmProviderConfig(provider='anthropic', model='claude-x')
            )

    def test_bedrock_without_region_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_llm_provider(
                LlmProviderConfig(provider='bedrock', model='anthropic.claude-x')
            )

    def test_openai_without_model_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            create_llm_provider(
                LlmProviderConfig(provider='openai', model='', api_key='sk-x')
            )


if __name__ == '__main__':
    unittest.main()
