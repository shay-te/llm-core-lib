"""Tests for :class:`llm_core_lib.LlmConnectionRegistry`."""
import unittest

from llm_core_lib import (
    LlmConfigError,
    LlmConnectionConfig,
    LlmConnectionRegistry,
    LlmDuplicateConnectionError,
    LlmInvalidProviderError,
    LlmMissingConnectionError,
    OpenAiLlmProvider,
)


def _ok_openai(_id: str = 'openai-default') -> LlmConnectionConfig:
    return LlmConnectionConfig(
        id=_id,
        provider='openai',
        model='gpt-4o-mini',
        api_key='sk-test',
    )


class TestRegistryHappyPath(unittest.TestCase):
    def setUp(self):
        self.registry = LlmConnectionRegistry()

    def test_register_then_get(self):
        self.registry.register(_ok_openai())
        provider = self.registry.get('openai-default')
        self.assertIsInstance(provider, OpenAiLlmProvider)

    def test_has_reports_membership(self):
        self.assertFalse(self.registry.has('x'))
        self.registry.register(_ok_openai('x'))
        self.assertTrue(self.registry.has('x'))

    def test_list_returns_registered_configs(self):
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

    def test_get_returns_cached_instance(self):
        # Two get() calls must return the same provider — registering
        # builds once and caches.
        self.registry.register(_ok_openai())
        p1 = self.registry.get('openai-default')
        p2 = self.registry.get('openai-default')
        self.assertIs(p1, p2)


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
