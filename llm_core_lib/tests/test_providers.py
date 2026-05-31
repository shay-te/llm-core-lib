"""Tests for the three concrete adapter classes.

All tests run against in-memory fake clients (``llm_core_lib.tests.fakes``);
nothing touches the network and none of the SDKs need to be installed.
"""
import unittest

from llm_core_lib import (
    AnthropicLlmProvider,
    BedrockLlmProvider,
    LlmChatRequest,
    LlmConfigError,
    LlmMessage,
    LlmProviderError,
    OpenAiLlmProvider,
)
from llm_core_lib.tests.fakes import (
    FakeAnthropicClient,
    FakeBedrockClient,
    FakeOpenAIClient,
)


class TestOpenAiProvider(unittest.TestCase):
    def test_normalizes_response(self):
        fake = FakeOpenAIClient(content='hello world', model='gpt-x')
        provider = OpenAiLlmProvider(
            api_key='sk-test', model='gpt-x', client=fake,
        )
        response = provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
        ))
        self.assertEqual(response.content, 'hello world')
        self.assertEqual(response.model, 'gpt-x')
        self.assertEqual(response.finish_reason, 'stop')
        self.assertIsNotNone(response.usage)
        self.assertEqual(response.usage.prompt_tokens, 5)
        self.assertEqual(response.usage.completion_tokens, 7)
        self.assertEqual(response.usage.total_tokens, 12)

    def test_passes_request_knobs_through(self):
        fake = FakeOpenAIClient()
        provider = OpenAiLlmProvider(api_key='k', model='m', client=fake)
        provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
            temperature=0.2,
            max_tokens=128,
            stop=['STOP'],
            system='be brief',
            extra={'top_p': 0.9},
        ))
        sent = fake.calls[0]
        self.assertEqual(sent['model'], 'm')
        self.assertEqual(sent['temperature'], 0.2)
        self.assertEqual(sent['max_tokens'], 128)
        self.assertEqual(sent['stop'], ['STOP'])
        self.assertEqual(sent['top_p'], 0.9)
        # System prompt is prepended as a message, not a top-level key.
        self.assertEqual(sent['messages'][0]['role'], 'system')
        self.assertEqual(sent['messages'][0]['content'], 'be brief')

    def test_request_model_overrides_provider_default(self):
        fake = FakeOpenAIClient()
        provider = OpenAiLlmProvider(api_key='k', model='default-model', client=fake)
        provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
            model='override-model',
        ))
        self.assertEqual(fake.calls[0]['model'], 'override-model')

    def test_sdk_exception_is_wrapped(self):
        fake = FakeOpenAIClient(raise_exc=RuntimeError('boom'))
        provider = OpenAiLlmProvider(api_key='k', model='m', client=fake)
        with self.assertRaises(LlmProviderError):
            provider.chat(LlmChatRequest(
                messages=[LlmMessage(role='user', content='hi')],
            ))

    def test_response_without_usage(self):
        fake = FakeOpenAIClient(omit_usage=True)
        provider = OpenAiLlmProvider(api_key='k', model='m', client=fake)
        response = provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
        ))
        self.assertIsNone(response.usage)

    def test_constructor_rejects_missing_api_key(self):
        with self.assertRaises(LlmConfigError):
            OpenAiLlmProvider(api_key='', model='m')

    def test_constructor_rejects_missing_model(self):
        with self.assertRaises(LlmConfigError):
            OpenAiLlmProvider(api_key='k', model='')


class TestAnthropicProvider(unittest.TestCase):
    def test_normalizes_response(self):
        fake = FakeAnthropicClient(content='hi', model='claude-x')
        provider = AnthropicLlmProvider(api_key='k', model='claude-x', client=fake)
        response = provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hello')],
        ))
        self.assertEqual(response.content, 'hi')
        self.assertEqual(response.model, 'claude-x')
        self.assertEqual(response.finish_reason, 'end_turn')
        self.assertEqual(response.usage.prompt_tokens, 11)
        self.assertEqual(response.usage.completion_tokens, 4)
        self.assertEqual(response.usage.total_tokens, 15)

    def test_system_goes_top_level_not_in_messages(self):
        fake = FakeAnthropicClient()
        provider = AnthropicLlmProvider(api_key='k', model='m', client=fake)
        provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
            system='be brief',
        ))
        sent = fake.calls[0]
        self.assertEqual(sent['system'], 'be brief')
        # The system prompt must NOT appear inside the messages list.
        for msg in sent['messages']:
            self.assertNotEqual(msg['role'], 'system')

    def test_stop_string_normalized_to_list(self):
        fake = FakeAnthropicClient()
        provider = AnthropicLlmProvider(api_key='k', model='m', client=fake)
        provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
            stop='STOP',
        ))
        self.assertEqual(fake.calls[0]['stop_sequences'], ['STOP'])

    def test_default_max_tokens_filled_in(self):
        fake = FakeAnthropicClient()
        provider = AnthropicLlmProvider(api_key='k', model='m', client=fake)
        provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
        ))
        # max_tokens is required by Anthropic — adapter must default it.
        self.assertGreater(fake.calls[0]['max_tokens'], 0)

    def test_concatenates_multiple_text_blocks(self):
        fake = FakeAnthropicClient(content_blocks=[
            {'type': 'text', 'text': 'hello '},
            {'type': 'thinking', 'text': 'IGNORED'},
            {'type': 'text', 'text': 'world'},
        ])
        provider = AnthropicLlmProvider(api_key='k', model='m', client=fake)
        response = provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
        ))
        self.assertEqual(response.content, 'hello world')

    def test_sdk_exception_is_wrapped(self):
        fake = FakeAnthropicClient(raise_exc=RuntimeError('boom'))
        provider = AnthropicLlmProvider(api_key='k', model='m', client=fake)
        with self.assertRaises(LlmProviderError):
            provider.chat(LlmChatRequest(
                messages=[LlmMessage(role='user', content='hi')],
            ))

    def test_constructor_rejects_missing_api_key(self):
        with self.assertRaises(LlmConfigError):
            AnthropicLlmProvider(api_key='', model='m')


class TestBedrockProvider(unittest.TestCase):
    def test_normalizes_response(self):
        fake = FakeBedrockClient(content='br hi')
        provider = BedrockLlmProvider(
            region='us-east-1',
            model='anthropic.claude-3-5-sonnet-20241022-v2:0',
            client=fake,
        )
        response = provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
        ))
        self.assertEqual(response.content, 'br hi')
        self.assertEqual(response.finish_reason, 'end_turn')
        self.assertEqual(response.usage.prompt_tokens, 9)
        self.assertEqual(response.usage.completion_tokens, 4)
        self.assertEqual(response.usage.total_tokens, 13)

    def test_builds_anthropic_shaped_body(self):
        fake = FakeBedrockClient()
        provider = BedrockLlmProvider(
            region='us-east-1', model='anthropic.x', client=fake,
        )
        provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
            system='be brief',
            temperature=0.3,
            max_tokens=512,
        ))
        sent_body = fake.calls[0]['body']
        self.assertEqual(sent_body['anthropic_version'], 'bedrock-2023-05-31')
        self.assertEqual(sent_body['system'], 'be brief')
        self.assertEqual(sent_body['temperature'], 0.3)
        self.assertEqual(sent_body['max_tokens'], 512)
        # Messages must wrap content in the {'type': 'text', 'text': ...} shape.
        self.assertEqual(
            sent_body['messages'][0]['content'][0],
            {'type': 'text', 'text': 'hi'},
        )

    def test_legacy_completion_shape(self):
        # Pre-Messages-API Anthropic-on-Bedrock returns ``completion``
        # at the top level; the adapter should still surface it.
        fake = FakeBedrockClient(
            legacy_completion='legacy-text', omit_usage=True,
        )
        provider = BedrockLlmProvider(
            region='us-east-1', model='anthropic.x', client=fake,
        )
        response = provider.chat(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
        ))
        self.assertEqual(response.content, 'legacy-text')
        self.assertIsNone(response.usage)

    def test_sdk_exception_is_wrapped(self):
        fake = FakeBedrockClient(raise_exc=RuntimeError('aws down'))
        provider = BedrockLlmProvider(
            region='us-east-1', model='anthropic.x', client=fake,
        )
        with self.assertRaises(LlmProviderError):
            provider.chat(LlmChatRequest(
                messages=[LlmMessage(role='user', content='hi')],
            ))

    def test_constructor_rejects_missing_region(self):
        with self.assertRaises(LlmConfigError):
            BedrockLlmProvider(region='', model='anthropic.x')


class TestStreamFallback(unittest.TestCase):
    def test_default_stream_emits_delta_then_stop(self):
        fake = FakeOpenAIClient(content='abc', finish_reason='stop')
        provider = OpenAiLlmProvider(api_key='k', model='m', client=fake)
        events = list(provider.stream(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
        )))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].type, 'delta')
        self.assertEqual(events[0].content, 'abc')
        self.assertEqual(events[1].type, 'stop')
        self.assertEqual(events[1].finish_reason, 'stop')

    def test_default_stream_emits_error_on_chat_failure(self):
        fake = FakeOpenAIClient(raise_exc=RuntimeError('boom'))
        provider = OpenAiLlmProvider(api_key='k', model='m', client=fake)
        events = list(provider.stream(LlmChatRequest(
            messages=[LlmMessage(role='user', content='hi')],
        )))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, 'error')
        self.assertIn('boom', events[0].content)


if __name__ == '__main__':
    unittest.main()
