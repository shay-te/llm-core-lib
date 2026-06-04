"""Tests for the three ``*ConnectionFactory`` / ``*Connection`` pairs.

All tests inject a fake SDK client via ``config['client']``; nothing
touches the network and none of the real SDKs need to be installed.
"""
import unittest

from llm_core_lib.connections.anthropic_connection import AnthropicConnection
from llm_core_lib.connections.anthropic_connection_factory import AnthropicConnectionFactory
from llm_core_lib.connections.bedrock_connection import BedrockConnection
from llm_core_lib.connections.bedrock_connection_factory import BedrockConnectionFactory
from llm_core_lib.connections.openai_connection import OpenAiConnection
from llm_core_lib.connections.openai_connection_factory import OpenAiConnectionFactory
from llm_core_lib.errors import LlmConfigError, LlmProviderError
from llm_core_lib.types import LlmCompletion
from llm_core_lib.tests.mock.anthropic_client import MockAnthropicClient
from llm_core_lib.tests.mock.bedrock_client import MockBedrockClient
from llm_core_lib.tests.mock.openai_client import MockOpenAIClient


# ---- OpenAI -----------------------------------------------------------


class TestOpenAiConnection(unittest.TestCase):
    def _factory(self, **overrides):
        fake = overrides.pop('client', MockOpenAIClient())
        cfg = {
            'model': 'gpt-x',
            'vision_model': 'gpt-x',
            'embedding_model': 'text-embedding-3-small',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'sk-test',
            'client': fake,
        }
        cfg.update(overrides)
        return OpenAiConnectionFactory(cfg), fake

    def test_complete_text_normalizes(self):
        factory, fake = self._factory(client=MockOpenAIClient(content='hello', model='gpt-x'))
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
            client=MockOpenAIClient(embedding=[0.7, 0.8, 0.9]),
        )
        vec = factory.get().embed('something')
        self.assertEqual(vec, [0.7, 0.8, 0.9])
        self.assertEqual(fake.embedding_calls[0]['input'], 'something')
        self.assertEqual(
            fake.embedding_calls[0]['model'], 'text-embedding-3-small',
        )

    def test_factory_rejects_missing_embedding_model(self):
        # Strict-required at __init__ now — the embed-time fallback is gone.
        with self.assertRaises(LlmConfigError):
            self._factory(embedding_model=None)

    def test_embed_wraps_sdk_exception(self):
        factory, _ = self._factory(
            client=MockOpenAIClient(raise_exc=RuntimeError('boom')),
        )
        with self.assertRaises(LlmProviderError):
            factory.get().embed('x')

    def test_embed_passes_through_llm_error(self):
        factory, _ = self._factory(
            client=MockOpenAIClient(raise_exc=LlmProviderError('inner')),
        )
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().embed('x')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_chat_wraps_sdk_exception(self):
        factory, _ = self._factory(
            client=MockOpenAIClient(raise_exc=RuntimeError('boom')),
        )
        with self.assertRaises(LlmProviderError):
            factory.get().complete_text('x')

    def test_chat_passes_through_llm_error(self):
        factory, _ = self._factory(
            client=MockOpenAIClient(raise_exc=LlmProviderError('inner')),
        )
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().complete_text('x')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_response_without_usage(self):
        factory, _ = self._factory(
            client=MockOpenAIClient(omit_usage=True),
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
            self._factory(model=None)

    def test_factory_requires_vision_model(self):
        with self.assertRaises(LlmConfigError):
            self._factory(vision_model=None)

    def test_factory_requires_max_tokens(self):
        with self.assertRaises(LlmConfigError):
            self._factory(max_tokens=None)

    def test_factory_requires_temperature(self):
        with self.assertRaises(LlmConfigError):
            self._factory(temperature=None)

    def test_factory_requires_api_key_when_no_client_injected(self):
        with self.assertRaises(LlmConfigError):
            OpenAiConnectionFactory({
                'model': 'gpt-x',
                'vision_model': 'gpt-x',
                'embedding_model': 'emb',
                'max_tokens': 4096,
                'temperature': 0.0,
            })

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

    def test_embed_requires_embedding_model_configured(self):
        # The factory's own validation already rejects an empty
        # ``embedding_model`` at construction; we exercise the
        # connection-level guard directly here (the factory layer's
        # rejection is tested in test_create_connection_factory.py).
        connection = OpenAiConnection(
            client=MockOpenAIClient(),
            model_id='gpt-x',
            vision_model_id='gpt-x',
            embedding_model='',  # empty by direct construction
        )
        with self.assertRaises(LlmConfigError):
            connection.embed('hello')

    def test_response_model_falls_back_to_request_model(self):
        # SDK omits ``model`` on the response — we fall back to the
        # model we sent in the request rather than returning ``None``.
        factory, _ = self._factory(
            client=MockOpenAIClient(content='hi', model=None),
        )
        completion = factory.get().complete_text('hi')
        self.assertEqual(completion.model, 'gpt-x')


# ---- Anthropic --------------------------------------------------------


class TestAnthropicConnection(unittest.TestCase):
    def _factory(self, **overrides):
        fake = overrides.pop('client', MockAnthropicClient())
        cfg = {
            'model': 'claude-x',
            'vision_model': 'claude-x',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'sk-test',
            'client': fake,
        }
        cfg.update(overrides)
        return AnthropicConnectionFactory(cfg), fake

    def test_complete_text_normalizes(self):
        factory, _ = self._factory(
            client=MockAnthropicClient(content='hi', model='claude-x'),
        )
        completion = factory.get().complete_text('hello')
        self.assertEqual(completion.text, 'hi')
        self.assertEqual(completion.model, 'claude-x')
        self.assertEqual(completion.usage['prompt_tokens'], 11)
        self.assertEqual(completion.usage['completion_tokens'], 4)
        self.assertEqual(completion.usage['total_tokens'], 15)

    def test_response_model_falls_back_to_request_model(self):
        # Same shape as the OpenAI fallback: if the SDK omits ``model``
        # on the response, we use the model we sent in the request.
        factory, _ = self._factory(
            client=MockAnthropicClient(content='hi', model=None),
        )
        completion = factory.get().complete_text('hello')
        self.assertEqual(completion.model, 'claude-x')

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

    def test_configured_max_tokens_is_forwarded_to_sdk(self):
        # No silent default anymore — whatever max_tokens the config
        # carries is what the SDK call receives verbatim.
        factory, fake = self._factory(max_tokens=2048)
        factory.get().complete_text('hi')
        self.assertEqual(fake.calls[0]['max_tokens'], 2048)

    def test_concatenates_text_blocks(self):
        factory, _ = self._factory(client=MockAnthropicClient(content_blocks=[
            {'type': 'text', 'text': 'hello '},
            {'type': 'thinking', 'text': 'IGNORED'},
            {'type': 'text', 'text': 'world'},
        ]))
        self.assertEqual(factory.get().complete_text('x').text, 'hello world')

    def test_chat_wraps_sdk_exception(self):
        factory, _ = self._factory(
            client=MockAnthropicClient(raise_exc=RuntimeError('boom')),
        )
        with self.assertRaises(LlmProviderError):
            factory.get().complete_text('x')

    def test_chat_passes_through_llm_error(self):
        factory, _ = self._factory(
            client=MockAnthropicClient(raise_exc=LlmProviderError('inner')),
        )
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().complete_text('x')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_response_without_usage(self):
        factory, _ = self._factory(client=MockAnthropicClient(omit_usage=True))
        completion = factory.get().complete_text('x')
        self.assertIsNone(completion.usage)

    def test_factory_requires_model(self):
        with self.assertRaises(LlmConfigError):
            self._factory(model=None)

    def test_factory_requires_vision_model(self):
        with self.assertRaises(LlmConfigError):
            self._factory(vision_model=None)

    def test_factory_requires_max_tokens(self):
        with self.assertRaises(LlmConfigError):
            self._factory(max_tokens=None)

    def test_factory_requires_temperature(self):
        with self.assertRaises(LlmConfigError):
            self._factory(temperature=None)

    def test_factory_requires_api_key_when_no_client_injected(self):
        with self.assertRaises(LlmConfigError):
            AnthropicConnectionFactory({
                'model': 'claude-x',
                'vision_model': 'claude-x',
                'max_tokens': 4096,
                'temperature': 0.0,
            })

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
        fake = overrides.pop('client', MockBedrockClient())
        cfg = {
            'model': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
            'vision_model': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
            'embedding_model': 'amazon.titan-embed-text-v1',
            'max_tokens': 4096,
            'temperature': 0.0,
            'region': 'us-east-1',
            'client': fake,
        }
        cfg.update(overrides)
        return BedrockConnectionFactory(cfg), fake

    def test_complete_text_normalizes(self):
        factory, _ = self._factory(client=MockBedrockClient(content='br hi'))
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
        factory, _ = self._factory(client=MockBedrockClient(
            legacy_completion='legacy-text',
        ))
        completion = factory.get().complete_text('hi')
        self.assertEqual(completion.text, 'legacy-text')

    def test_titan_results_shape(self):
        factory, _ = self._factory(client=MockBedrockClient(
            titan_results=['titan-text'],
        ))
        completion = factory.get().complete_text('hi')
        self.assertEqual(completion.text, 'titan-text')

    def test_embed_returns_embedding_vector(self):
        factory, _ = self._factory(
            embedding_model='amazon.titan-embed-text-v1',
            client=MockBedrockClient(embedding=[0.4, 0.5, 0.6]),
        )
        vec = factory.get().embed('hi')
        self.assertEqual(vec, [0.4, 0.5, 0.6])

    def test_embed_falls_back_to_embeddings_array(self):
        # When the SDK returns ``embeddings: [[...]]`` (Cohere-on-Bedrock
        # shape) the adapter should still surface the first vector.
        factory, _ = self._factory(
            embedding_model='cohere.embed-english-v3',
            client=MockBedrockClient(),  # default → embeddings: [[0.1, 0.2]]
        )
        vec = factory.get().embed('hi')
        self.assertEqual(vec, [0.1, 0.2])

    def test_factory_rejects_missing_embedding_model(self):
        # Strict-required at __init__ now — the embed-time fallback is gone.
        with self.assertRaises(LlmConfigError):
            self._factory(embedding_model=None)

    def test_embed_wraps_sdk_exception(self):
        factory, _ = self._factory(
            embedding_model='amazon.titan-embed-text-v1',
            client=MockBedrockClient(raise_exc=RuntimeError('aws down')),
        )
        with self.assertRaises(LlmProviderError):
            factory.get().embed('hi')

    def test_embed_passes_through_llm_error(self):
        factory, _ = self._factory(
            embedding_model='amazon.titan-embed-text-v1',
            client=MockBedrockClient(raise_exc=LlmProviderError('inner')),
        )
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().embed('hi')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_chat_wraps_sdk_exception(self):
        factory, _ = self._factory(client=MockBedrockClient(
            raise_exc=RuntimeError('aws down'),
        ))
        with self.assertRaises(LlmProviderError):
            factory.get().complete_text('hi')

    def test_chat_passes_through_llm_error(self):
        factory, _ = self._factory(client=MockBedrockClient(
            raise_exc=LlmProviderError('inner'),
        ))
        with self.assertRaises(LlmProviderError) as ctx:
            factory.get().complete_text('hi')
        self.assertEqual(str(ctx.exception), 'inner')

    def test_factory_requires_region(self):
        with self.assertRaises(LlmConfigError):
            self._factory(region=None)

    def test_factory_requires_model(self):
        with self.assertRaises(LlmConfigError):
            self._factory(model=None)

    def test_factory_requires_vision_model(self):
        with self.assertRaises(LlmConfigError):
            self._factory(vision_model=None)

    def test_factory_requires_max_tokens(self):
        with self.assertRaises(LlmConfigError):
            self._factory(max_tokens=None)

    def test_factory_requires_temperature(self):
        with self.assertRaises(LlmConfigError):
            self._factory(temperature=None)

    def test_factory_rejects_legacy_model_id_alias(self):
        # The Bedrock-native ``model_id`` alias was removed — only the
        # canonical ``model`` key is accepted. A config that uses the
        # old key (and omits ``model``) must fail at __init__.
        with self.assertRaises(LlmConfigError):
            BedrockConnectionFactory({
                'model_id': 'anthropic.x',
                'vision_model': 'anthropic.x',
                'embedding_model': 'emb',
                'max_tokens': 4096,
                'temperature': 0.0,
                'region': 'us-east-1',
                'client': MockBedrockClient(),
            })

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

    def test_property_accessors_return_configured_values(self):
        # ``model_id`` / ``vision_model_id`` / ``embedding_model`` are
        # the three read-only accessors callers reach for when they
        # need to know which model a connection is bound to.
        factory, _ = self._factory(
            vision_model='anthropic.claude-3-haiku-20240307-v1:0',
            embedding_model='amazon.titan-embed-text-v1',
        )
        connection = factory.get()
        self.assertEqual(
            connection.model_id,
            'anthropic.claude-3-5-sonnet-20241022-v2:0',
        )
        self.assertEqual(
            connection.vision_model_id,
            'anthropic.claude-3-haiku-20240307-v1:0',
        )
        self.assertEqual(
            connection.embedding_model,
            'amazon.titan-embed-text-v1',
        )

    def test_embed_requires_embedding_model_configured(self):
        # Factory rejects an empty ``embedding_model`` at construction;
        # we exercise the connection-level guard directly. Mirrors the
        # OpenAI adapter's behavior.
        connection = BedrockConnection(
            client=MockBedrockClient(),
            model_id='anthropic.claude',
            vision_model_id='anthropic.claude',
            embedding_model='',
        )
        with self.assertRaises(LlmConfigError):
            connection.embed('hello')

    def test_embed_returns_empty_list_when_response_has_no_shape(self):
        # Neither ``embedding`` nor ``embeddings`` populated in the
        # response payload — the fall-through branch returns ``[]``
        # rather than raising. Locks the third "decide which shape"
        # branch.
        class _EmptyBedrockClient(object):
            def invoke_model(self, **kwargs):
                import json

                class _Body(object):
                    def read(_self):
                        return json.dumps({}).encode('utf-8')

                return {'body': _Body()}

        factory, _ = self._factory(client=_EmptyBedrockClient())
        self.assertEqual(factory.get().embed('hello'), [])

    def test_extract_text_returns_empty_for_unrecognised_shape(self):
        from llm_core_lib.connections.bedrock_connection import _extract_text
        # A payload that's a dict but carries no ``content`` /
        # ``completion`` / ``results`` keys — the final fall-through
        # returns ``''``.
        self.assertEqual(_extract_text({'random': 'shape'}), '')

    def test_extract_text_skips_non_dict_items_in_content_blocks(self):
        from llm_core_lib.connections.bedrock_connection import _extract_text
        # ``content`` is a list, but one of its items isn't a dict — the
        # per-item ``isinstance`` guard must skip it without breaking
        # the loop. Locks the inner ``False``-branch of the per-block
        # ``isinstance(part, dict)`` check.
        payload = {
            'content': [
                {'type': 'text', 'text': 'first'},
                'not a dict — must be skipped',
                {'type': 'text', 'text': 'second'},
            ],
        }
        self.assertEqual(_extract_text(payload), 'firstsecond')


# ---- Context-manager protocol ----------------------------------------


class TestConnectionContextManager(unittest.TestCase):
    """Every ``*Connection`` extends ``core_lib.connection.Connection``
    so callers can use ``with factory.get() as conn:`` ergonomics.
    The exit always calls ``close()`` and lets exceptions propagate."""

    def test_openai_with_block_yields_connection(self):
        factory = OpenAiConnectionFactory({
            'model': 'gpt-x',
            'vision_model': 'gpt-x',
            'embedding_model': 'emb',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'sk',
            'client': MockOpenAIClient(),
        })
        with factory.get() as conn:
            self.assertIsInstance(conn, OpenAiConnection)
            completion = conn.complete_text('hi')
        self.assertEqual(completion.text, 'hello')

    def test_anthropic_with_block_yields_connection(self):
        factory = AnthropicConnectionFactory({
            'model': 'claude-x',
            'vision_model': 'claude-x',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'sk',
            'client': MockAnthropicClient(),
        })
        with factory.get() as conn:
            self.assertIsInstance(conn, AnthropicConnection)
            completion = conn.complete_text('hi')
        self.assertEqual(completion.text, 'hi from anthropic')

    def test_bedrock_with_block_yields_connection(self):
        factory = BedrockConnectionFactory({
            'model': 'anthropic.x',
            'vision_model': 'anthropic.x',
            'embedding_model': 'emb',
            'max_tokens': 4096,
            'temperature': 0.0,
            'region': 'us-east-1',
            'client': MockBedrockClient(),
        })
        with factory.get() as conn:
            self.assertIsInstance(conn, BedrockConnection)
            completion = conn.complete_text('hi')
        self.assertEqual(completion.text, 'bedrock reply')

    def _openai_factory(self):
        return OpenAiConnectionFactory({
            'model': 'gpt-x',
            'vision_model': 'gpt-x',
            'embedding_model': 'emb',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'sk',
            'client': MockOpenAIClient(),
        })

    def test_with_block_exit_calls_close(self):
        # Spy on close() to confirm the context-manager exit actually
        # calls it (rather than relying on the no-op default firing).
        conn = self._openai_factory().get()
        closed = {'count': 0}

        def _spy_close():
            closed['count'] += 1

        conn.close = _spy_close
        with conn:
            pass
        self.assertEqual(closed['count'], 1)

    def test_with_block_propagates_exceptions(self):
        # The exit must NOT suppress exceptions raised inside the
        # ``with`` block.
        factory = self._openai_factory()

        class _Boom(RuntimeError):
            pass

        with self.assertRaises(_Boom):
            with factory.get():
                raise _Boom('inner failure')


if __name__ == '__main__':
    unittest.main()
