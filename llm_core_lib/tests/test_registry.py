"""Tests for :class:`llm_core_lib.LlmConnectionRegistry`."""
import unittest

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib import (
    LlmConfigError,
    LlmConnectionConfig,
    LlmConnectionRegistry,
    LlmDuplicateConnectionError,
    LlmInvalidProviderError,
    LlmMissingConnectionError,
    OpenAiConnectionFactory,
)
from llm_core_lib.tests.fakes import FakeOpenAIClient


def _ok_openai(_id: str = 'openai-default') -> LlmConnectionConfig:
    # Inject a fake client via ``extra`` so the test stays deterministic
    # regardless of whether the real openai SDK is installed in the env.
    return LlmConnectionConfig(
        id=_id,
        provider='openai',
        model='gpt-4o-mini',
        api_key='sk-test',
        extra={'client': FakeOpenAIClient()},
    )


class TestRegistryHappyPath(unittest.TestCase):
    def setUp(self):
        self.registry = LlmConnectionRegistry()

    def test_register_then_get_returns_factory(self):
        self.registry.register(_ok_openai())
        factory = self.registry.get('openai-default')
        self.assertIsInstance(factory, ConnectionFactory)
        self.assertIsInstance(factory, OpenAiConnectionFactory)

    def test_has_reports_membership(self):
        self.assertFalse(self.registry.has('x'))
        self.registry.register(_ok_openai('x'))
        self.assertTrue(self.registry.has('x'))

    def test_list_returns_configs_in_order(self):
        self.registry.register(_ok_openai('a'))
        self.registry.register(_ok_openai('b'))
        ids = [c.id for c in self.registry.list()]
        self.assertEqual(ids, ['a', 'b'])

    def test_unregister_removes(self):
        self.registry.register(_ok_openai('z'))
        self.registry.unregister('z')
        self.assertFalse(self.registry.has('z'))

    def test_clear_drops_everything(self):
        self.registry.register(_ok_openai('a'))
        self.registry.register(_ok_openai('b'))
        self.registry.clear()
        self.assertEqual(self.registry.list(), [])
        self.assertFalse(self.registry.has('a'))

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
            self.registry.register(
                LlmConnectionConfig(
                    id='',
                    provider='openai',
                    model='gpt-4o-mini',
                    api_key='sk-test',
                )
            )

    def test_duplicate_register_raises(self):
        self.registry.register(_ok_openai())
        with self.assertRaises(LlmDuplicateConnectionError):
            self.registry.register(_ok_openai())

    def test_get_missing_raises(self):
        with self.assertRaises(LlmMissingConnectionError):
            self.registry.get('nope')

    def test_unregister_missing_raises(self):
        with self.assertRaises(LlmMissingConnectionError):
            self.registry.unregister('nope')

    def test_bad_provider_propagates_invalid_provider(self):
        with self.assertRaises(LlmInvalidProviderError):
            self.registry.register(
                LlmConnectionConfig(
                    id='bad',
                    provider='cohere',
                    model='c4',
                )
            )

    def test_bad_provider_config_propagates_config_error(self):
        with self.assertRaises(LlmConfigError):
            self.registry.register(
                LlmConnectionConfig(
                    id='nokey',
                    provider='openai',
                    model='gpt-4o-mini',
                    # no api_key
                )
            )


if __name__ == '__main__':
    unittest.main()
