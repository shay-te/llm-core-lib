"""Tests for the three ``*ConnectionFactory`` / ``*Connection`` pairs.

All tests inject a fake SDK client via ``config['client']``; nothing
touches the network and none of the real SDKs need to be installed.
"""
import unittest

from llm_core_lib import (
    AnthropicConnectionFactory,
    BedrockConnectionFactory,
    LlmCompletion,
    LlmConfigError,
    LlmProviderError,
    OpenAiConnectionFactory,
)
from llm_core_lib.tests.fakes import (
    FakeAnthropicClient,
    FakeBedrockClient,
    FakeOpenAIClient,
)


# ---- OpenAI -----------------------------------------------------------


class TestOpenAiConnection(unittest.TestCase):
    def _factory(self, **overrides):
        fake = overrides.pop('client', FakeOpenAIClient())
        cfg = {
            'model': 'gpt-x',
            'api_key': 'sk-test',
            'embedding_model': 'text-embedding-3-small',
            'client': fake,
        }
        cfg.update(overrides)
        return OpenAiConnectionFactory(cfg), fake

    def test_complete_text_normalizes(self):
        factory, fake = self._factory(client=FakeOpenAIClient(content='hello', model='gpt-x'))
        completion = factory.get().complete_text('hi')
        self.assertIsInstance(completion, LlmCompletion)
        self.assertEqual(completion.text, 'hello')
        self.assertEqual(completion.model, 'gpt-x')
        self.assertEqual(completion.usage['prompt_tokens'], 5)
        self.assertEqual(completion.usage['completion_tokens'], 7)
        self.assertEqual(completion.usage['total_tokens'], 12)

    def test_complete_text_sends_one_user_message(self):
        factory, fake = self._factory()
        factory.get().complete_text('hi')
        sent = fake.chat_calls[0]
        self.assertEqual(sent['model'], 'gpt-x')
        self.assertEqual(sent['messages'], [{'role': 'user', 'content': 'hi'}])

    def test_complete_text_prepends_system_message(self):
        factory, fake = self._factory()
        factory.get().complete_text('hi', system='be brief')
        msgs = fake.chat_calls[0]['messages']
        self.assertEqual(msgs[0], {'role': 'system', 'content': 'be brief'})
        self.assertEqual(msgs[1], {'role': 'user', 'content': 'hi'})

    def test_complete_vision_builds_image_payload(self):
        factory, fake = self._factory(vision_model='gpt-vision')
        factory.get().complete_vision('describe', b'\x89PNG\r\n\x1a\nfake')
        sent = fake.chat_calls[0]
        self.assertEqual(sent['model'], 'gpt-vision')
        # Vision payload is a list of content blocks.
        content = sent['messages'][0]['content']
        self.assertEqual(content[0]['type'], 'text')
        self.assertEqual(content[0]['text'], 'describe')
        self.assertEqual(content[1]['type'], 'image_url')
        self.assertTrue(
            content[1]['image_url']['url'].startswith('data:image/png;base64,'),
        )

    def test_complete_vision_with_system(self):
        # System message prepended for vision calls too.
        factory, fake = self._factory()
        factory.get().complete_vision('hi', b'x', system='be brief')
        msgs = fake.chat_calls[0]['messages']
        self.assertEqual(msgs[0]['role'], 'system')

    def test_embed_passes_text_and_returns_vector(self):
        factory, fake = self._factory(
            client=FakeOpenAIClient(embedding=[0.7, 0.8, 0.9]),
        )
        vec = factory.get().embed('something')
        self.assertEqual(vec, [0.7, 0.8, 0.9])
        self.assertEqual(fake.embedding_calls[0]['input'], 'something')
        self.assertEqual(
            fake.embedding_calls[0]['model'], 'text-embedding-3-small',
        )

    def test_embed_without_embedding_model_raises_config_error(self):
        factory, _ = self._factory(embedding_model=None)
        with self.assertRaises(LlmConfigError):
            factory.get().embed('something')

    def test_embed_wraps_sdk_exception(self):
        factory, _ = self._factory(
            client=FakeOpenAIClient(raise_exc=RuntimeError('boom')),
        )
        with self.assertRaises(LlmProviderError):
            factory.get().embed('x')

    def test_embed_passes_through_llm_error(self):
        factory, _ = self._factory(
            client=FakeOpenAIClient(raise_exc=LlmProviderError('inner')),
        )
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().embed('x')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_chat_wraps_sdk_exception(self):
        factory, _ = self._factory(
            client=FakeOpenAIClient(raise_exc=RuntimeError('boom')),
        )
        with self.assertRaises(LlmProviderError):
            factory.get().complete_text('x')

    def test_chat_passes_through_llm_error(self):
        factory, _ = self._factory(
            client=FakeOpenAIClient(raise_exc=LlmProviderError('inner')),
        )
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().complete_text('x')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_response_without_usage(self):
        factory, _ = self._factory(
            client=FakeOpenAIClient(omit_usage=True),
        )
        completion = factory.get().complete_text('x')
        self.assertIsNone(completion.usage)

    def test_empty_choices_raises(self):
        # Custom client with empty choices — must refuse rather than
        # silently return empty content.
        from types import SimpleNamespace

        class _ZeroChoices(object):
            def __init__(self):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kw: SimpleNamespace(
                            choices=[], model='m', usage=None,
                        ),
                    ),
                )
                self.embeddings = SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(data=[]),
                )

        factory, _ = self._factory(client=_ZeroChoices())
        with self.assertRaises(LlmProviderError):
            factory.get().complete_text('hi')

    def test_factory_requires_model(self):
        with self.assertRaises(LlmConfigError):
            OpenAiConnectionFactory({'api_key': 'sk', 'client': FakeOpenAIClient()})

    def test_factory_requires_api_key_when_no_client_injected(self):
        with self.assertRaises(LlmConfigError):
            OpenAiConnectionFactory({'model': 'gpt-x'})

    def test_factory_model_id_alias_accepted(self):
        # Bedrock-native ``model_id`` key works too — convenience for
        # hand-rolled yaml configs that mirror the Bedrock factory.
        factory = OpenAiConnectionFactory({
            'model_id': 'gpt-x',
            'client': FakeOpenAIClient(),
        })
        self.assertIsNotNone(factory.get())

    def test_connection_close_is_noop(self):
        factory, _ = self._factory()
        conn = factory.get()
        self.assertIsNone(conn.close())

    def test_connection_properties_expose_models(self):
        factory, _ = self._factory(vision_model='gpt-v', embedding_model='emb')
        conn = factory.get()
        self.assertEqual(conn.model_id, 'gpt-x')
        self.assertEqual(conn.vision_model_id, 'gpt-v')
        self.assertEqual(conn.embedding_model, 'emb')

    def test_chat_with_zero_max_tokens_omits_the_field(self):
        # ``max_tokens=0`` is treated as "let the model decide" — the
        # payload should NOT carry max_tokens at all.
        factory, fake = self._factory(max_tokens=0)
        factory.get().complete_text('hi')
        self.assertNotIn('max_tokens', fake.chat_calls[0])

    def test_chat_handles_choice_without_message_attribute(self):
        # Defensive: if a future SDK returns a choice that lacks a
        # ``message`` attr, treat content as empty rather than crashing.
        from types import SimpleNamespace

        class _NoMessage(object):
            def __init__(self):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kw: SimpleNamespace(
                            choices=[SimpleNamespace(finish_reason='stop')],
                            model='m',
                            usage=None,
                        ),
                    ),
                )

        factory, _ = self._factory(client=_NoMessage())
        completion = factory.get().complete_text('hi')
        self.assertEqual(completion.text, '')


# ---- Anthropic --------------------------------------------------------


class TestAnthropicConnection(unittest.TestCase):
    def _factory(self, **overrides):
        fake = overrides.pop('client', FakeAnthropicClient())
        cfg = {
            'model': 'claude-x',
            'api_key': 'sk-test',
            'client': fake,
        }
        cfg.update(overrides)
        return AnthropicConnectionFactory(cfg), fake

    def test_complete_text_normalizes(self):
        factory, _ = self._factory(
            client=FakeAnthropicClient(content='hi', model='claude-x'),
        )
        completion = factory.get().complete_text('hello')
        self.assertEqual(completion.text, 'hi')
        self.assertEqual(completion.model, 'claude-x')
        self.assertEqual(completion.usage['prompt_tokens'], 11)
        self.assertEqual(completion.usage['completion_tokens'], 4)
        self.assertEqual(completion.usage['total_tokens'], 15)

    def test_complete_text_sends_user_content_block(self):
        factory, fake = self._factory()
        factory.get().complete_text('hi')
        sent = fake.calls[0]
        self.assertEqual(sent['model'], 'claude-x')
        self.assertEqual(
            sent['messages'][0]['content'],
            [{'type': 'text', 'text': 'hi'}],
        )

    def test_complete_text_system_goes_top_level(self):
        factory, fake = self._factory()
        factory.get().complete_text('hi', system='be brief')
        sent = fake.calls[0]
        self.assertEqual(sent['system'], 'be brief')
        # System must NOT appear inside messages.
        for msg in sent['messages']:
            self.assertNotEqual(msg['role'], 'system')

    def test_complete_vision_image_block(self):
        factory, fake = self._factory(vision_model='claude-vision')
        factory.get().complete_vision('describe', b'\x89PNG\r\n\x1a\nfake')
        sent = fake.calls[0]
        self.assertEqual(sent['model'], 'claude-vision')
        blocks = sent['messages'][0]['content']
        self.assertEqual(blocks[0]['type'], 'image')
        self.assertEqual(blocks[0]['source']['type'], 'base64')
        self.assertEqual(blocks[0]['source']['media_type'], 'image/png')
        self.assertEqual(blocks[1], {'type': 'text', 'text': 'describe'})

    def test_complete_vision_with_system(self):
        factory, fake = self._factory()
        factory.get().complete_vision('hi', b'x', system='be brief')
        self.assertEqual(fake.calls[0]['system'], 'be brief')

    def test_default_max_tokens_filled(self):
        factory, fake = self._factory()
        factory.get().complete_text('hi')
        self.assertGreater(fake.calls[0]['max_tokens'], 0)

    def test_concatenates_text_blocks(self):
        factory, _ = self._factory(client=FakeAnthropicClient(content_blocks=[
            {'type': 'text', 'text': 'hello '},
            {'type': 'thinking', 'text': 'IGNORED'},
            {'type': 'text', 'text': 'world'},
        ]))
        self.assertEqual(factory.get().complete_text('x').text, 'hello world')

    def test_chat_wraps_sdk_exception(self):
        factory, _ = self._factory(
            client=FakeAnthropicClient(raise_exc=RuntimeError('boom')),
        )
        with self.assertRaises(LlmProviderError):
            factory.get().complete_text('x')

    def test_chat_passes_through_llm_error(self):
        factory, _ = self._factory(
            client=FakeAnthropicClient(raise_exc=LlmProviderError('inner')),
        )
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().complete_text('x')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_response_without_usage(self):
        factory, _ = self._factory(client=FakeAnthropicClient(omit_usage=True))
        completion = factory.get().complete_text('x')
        self.assertIsNone(completion.usage)

    def test_factory_requires_model(self):
        with self.assertRaises(LlmConfigError):
            AnthropicConnectionFactory({
                'api_key': 'sk', 'client': FakeAnthropicClient(),
            })

    def test_factory_requires_api_key_when_no_client_injected(self):
        with self.assertRaises(LlmConfigError):
            AnthropicConnectionFactory({'model': 'claude-x'})

    def test_connection_close_is_noop(self):
        factory, _ = self._factory()
        self.assertIsNone(factory.get().close())

    def test_connection_properties_expose_models(self):
        factory, _ = self._factory(vision_model='claude-vision')
        conn = factory.get()
        self.assertEqual(conn.model_id, 'claude-x')
        self.assertEqual(conn.vision_model_id, 'claude-vision')


# ---- Bedrock ----------------------------------------------------------


class TestBedrockConnection(unittest.TestCase):
    def _factory(self, **overrides):
        fake = overrides.pop('client', FakeBedrockClient())
        cfg = {
            'model': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
            'region': 'us-east-1',
            'client': fake,
        }
        cfg.update(overrides)
        return BedrockConnectionFactory(cfg), fake

    def test_complete_text_normalizes(self):
        factory, _ = self._factory(client=FakeBedrockClient(content='br hi'))
        completion = factory.get().complete_text('hi')
        self.assertEqual(completion.text, 'br hi')
        self.assertEqual(
            completion.model,
            'anthropic.claude-3-5-sonnet-20241022-v2:0',
        )
        self.assertEqual(completion.usage['input_tokens'], 9)
        self.assertEqual(completion.usage['output_tokens'], 4)

    def test_anthropic_shaped_body(self):
        factory, fake = self._factory()
        factory.get().complete_text('hi', system='be brief')
        body = fake.calls[0]['body']
        self.assertEqual(body['anthropic_version'], 'bedrock-2023-05-31')
        self.assertEqual(body['system'], 'be brief')
        # Single user message with a text block.
        self.assertEqual(
            body['messages'][0]['content'][0],
            {'type': 'text', 'text': 'hi'},
        )

    def test_complete_vision_inserts_image_block_first(self):
        factory, fake = self._factory(vision_model='bedrock.vision')
        factory.get().complete_vision('describe', b'fakepng', image_mime='image/jpeg')
        body = fake.calls[0]['body']
        self.assertEqual(fake.calls[0]['modelId'], 'bedrock.vision')
        content = body['messages'][0]['content']
        self.assertEqual(content[0]['type'], 'image')
        self.assertEqual(content[0]['source']['media_type'], 'image/jpeg')
        self.assertEqual(content[1], {'type': 'text', 'text': 'describe'})

    def test_legacy_completion_shape(self):
        factory, _ = self._factory(client=FakeBedrockClient(
            legacy_completion='legacy-text',
        ))
        completion = factory.get().complete_text('hi')
        self.assertEqual(completion.text, 'legacy-text')

    def test_titan_results_shape(self):
        factory, _ = self._factory(client=FakeBedrockClient(
            titan_results=['titan-text'],
        ))
        completion = factory.get().complete_text('hi')
        self.assertEqual(completion.text, 'titan-text')

    def test_embed_returns_embedding_vector(self):
        factory, _ = self._factory(
            embedding_model='amazon.titan-embed-text-v1',
            client=FakeBedrockClient(embedding=[0.4, 0.5, 0.6]),
        )
        vec = factory.get().embed('hi')
        self.assertEqual(vec, [0.4, 0.5, 0.6])

    def test_embed_falls_back_to_embeddings_array(self):
        # When the SDK returns ``embeddings: [[...]]`` (Cohere-on-Bedrock
        # shape) the adapter should still surface the first vector.
        factory, _ = self._factory(
            embedding_model='cohere.embed-english-v3',
            client=FakeBedrockClient(),  # default → embeddings: [[0.1, 0.2]]
        )
        vec = factory.get().embed('hi')
        self.assertEqual(vec, [0.1, 0.2])

    def test_embed_without_embedding_model_raises_config_error(self):
        # Mirrors OpenAI's behavior: missing embedding_model → clear
        # config error rather than letting boto3 fail on empty modelId.
        factory, _ = self._factory()  # default config has no embedding_model
        with self.assertRaises(LlmConfigError):
            factory.get().embed('hi')

    def test_embed_wraps_sdk_exception(self):
        factory, _ = self._factory(
            embedding_model='amazon.titan-embed-text-v1',
            client=FakeBedrockClient(raise_exc=RuntimeError('aws down')),
        )
        with self.assertRaises(LlmProviderError):
            factory.get().embed('hi')

    def test_embed_passes_through_llm_error(self):
        factory, _ = self._factory(
            embedding_model='amazon.titan-embed-text-v1',
            client=FakeBedrockClient(raise_exc=LlmProviderError('inner')),
        )
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().embed('hi')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_chat_wraps_sdk_exception(self):
        factory, _ = self._factory(client=FakeBedrockClient(
            raise_exc=RuntimeError('aws down'),
        ))
        with self.assertRaises(LlmProviderError):
            factory.get().complete_text('hi')

    def test_chat_passes_through_llm_error(self):
        factory, _ = self._factory(client=FakeBedrockClient(
            raise_exc=LlmProviderError('inner'),
        ))
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().complete_text('hi')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_factory_requires_region(self):
        with self.assertRaises(LlmConfigError):
            BedrockConnectionFactory({'model': 'anthropic.x', 'client': FakeBedrockClient()})

    def test_factory_requires_model(self):
        with self.assertRaises(LlmConfigError):
            BedrockConnectionFactory({'region': 'us-east-1', 'client': FakeBedrockClient()})

    def test_factory_accepts_model_id_alias(self):
        # Bedrock-native ``model_id`` key works alongside ``model``.
        factory = BedrockConnectionFactory({
            'model_id': 'anthropic.x',
            'region': 'us-east-1',
            'client': FakeBedrockClient(),
        })
        self.assertEqual(factory.get().model_id, 'anthropic.x')

    def test_factory_accepts_embedding_model_id_alias(self):
        factory = BedrockConnectionFactory({
            'model': 'anthropic.x',
            'region': 'us-east-1',
            'embedding_model_id': 'amazon.titan-embed-text-v1',
            'client': FakeBedrockClient(embedding=[0.1]),
        })
        self.assertEqual(factory.get().embedding_model, 'amazon.titan-embed-text-v1')

    def test_factory_accepts_vision_model_id_alias(self):
        factory = BedrockConnectionFactory({
            'model': 'anthropic.x',
            'region': 'us-east-1',
            'vision_model_id': 'bedrock.vision',
            'client': FakeBedrockClient(),
        })
        self.assertEqual(factory.get().vision_model_id, 'bedrock.vision')

    def test_connection_close_is_noop(self):
        factory, _ = self._factory()
        self.assertIsNone(factory.get().close())

    def test_json_default_encodes_bytes(self):
        from llm_core_lib.connections.bedrock_connection import _json_default
        encoded = _json_default(b'hi')
        self.assertEqual(encoded, 'aGk=')

    def test_json_default_rejects_other_types(self):
        from llm_core_lib.connections.bedrock_connection import _json_default
        with self.assertRaises(TypeError):
            _json_default(object())

    def test_extract_text_handles_non_dict_payload(self):
        from llm_core_lib.connections.bedrock_connection import _extract_text
        # Defensive — a malformed payload (e.g. a list at the top
        # level) returns empty text rather than crashing the call.
        self.assertEqual(_extract_text(['unexpected']), '')


if __name__ == '__main__':
    unittest.main()
