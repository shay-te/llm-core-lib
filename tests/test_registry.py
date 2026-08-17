"""Tests for :class:`llm_core_lib.LlmConnectionRegistry`."""
import unittest

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.connections.openai_connection_factory import OpenAiConnectionFactory
from llm_core_lib.errors import LlmConfigError, LlmDuplicateConnectionError, LlmInvalidProviderError, LlmMissingConnectionError
from llm_core_lib.registry import LlmConnectionRegistry
from llm_core_lib.types import LlmConnectionConfig
from tests.mock.openai_client import MockOpenAIClient


def _ok_openai(_id: str = 'openai-default') -> LlmConnectionConfig:
    # Inject a fake client via ``extra`` so the test stays deterministic
    # regardless of whether the real openai SDK is installed in the env.
    return LlmConnectionConfig(
        id=_id,
        provider='openai',
        model='gpt-4o-mini',
        vision_model='gpt-4o-mini',
        embedding_model='text-embedding-3-small',
        max_tokens=4096,
        temperature=0.0,
        api_key='sk-test',
        extra={'client': MockOpenAIClient()},
    )


def _config(**overrides) -> LlmConnectionConfig:
    base = {
        'id': '_',
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'vision_model': 'gpt-4o-mini',
        'embedding_model': 'emb',
        'max_tokens': 4096,
        'temperature': 0.0,
        'api_key': 'sk-test',
    }
    base.update(overrides)
    return LlmConnectionConfig(**base)


class TestRegistryHappyPath(unittest.TestCase):
    def setUp(self):
        self.registry = LlmConnectionRegistry()

    def test_register_then_get_returns_factory(self):
        self.registry.register(_ok_openai())
        factory = self.registry.get('openai-default')
        self.assertIsInstance(factory, ConnectionFactory)
        self.assertIsInstance(factory, OpenAiConnectionFactory)

    def test_list_returns_configs_in_order(self):
        self.registry.register(_ok_openai('a'))
        self.registry.register(_ok_openai('b'))
        ids = [c.id for c in self.registry.list()]
        self.assertEqual(ids, ['a', 'b'])

    def test_get_returns_cached_factory_instance(self):
        # Two get() calls return the same factory — build happens
        # once at register().
        self.registry.register(_ok_openai())
        f1 = self.registry.get('openai-default')
        f2 = self.registry.get('openai-default')
        self.assertIs(f1, f2)


class TestRegistryErrors(unittest.TestCase):
    def setUp(self):
        self.registry = LlmConnectionRegistry()

    def test_register_without_id_raises_config_error(self):
        with self.assertRaises(LlmConfigError):
            self.registry.register(_config(id=''))

    def test_duplicate_register_raises(self):
        self.registry.register(_ok_openai())
        with self.assertRaises(LlmDuplicateConnectionError):
            self.registry.register(_ok_openai())

    def test_get_missing_raises(self):
        with self.assertRaises(LlmMissingConnectionError):
            self.registry.get('nope')

    def test_bad_provider_propagates_invalid_provider(self):
        with self.assertRaises(LlmInvalidProviderError):
            self.registry.register(_config(id='bad', provider='cohere'))

    def test_bad_provider_config_propagates_config_error(self):
        # Strip api_key (and the extra client) — factory must reject.
        with self.assertRaises(LlmConfigError):
            self.registry.register(_config(
                id='nokey', api_key=None,
            ))


if __name__ == '__main__':
    unittest.main()
