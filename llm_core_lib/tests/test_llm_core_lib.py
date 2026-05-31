"""Tests for the :class:`llm_core_lib.LlmCoreLib` composition root."""
import unittest

from llm_core_lib import (
    LlmConfigError,
    LlmCoreLib,
    LlmInvalidProviderError,
    OpenAiLlmProvider,
)


class TestLlmCoreLibBootstrap(unittest.TestCase):
    def test_empty_constructor_yields_empty_registry(self):
        core = LlmCoreLib()
        self.assertEqual(core.registry.list(), [])

    def test_none_config_yields_empty_registry(self):
        core = LlmCoreLib(None)
        self.assertEqual(core.registry.list(), [])

    def test_dict_config_hydrates_connections(self):
        cfg = {
            'core_lib': {
                'llm': {
                    'connections': [
                        {
                            'id': 'openai-default',
                            'provider': 'openai',
                            'model': 'gpt-4o-mini',
                            'api_key': 'sk-test',
                        },
                    ],
                },
            },
        }
        core = LlmCoreLib(cfg)
        self.assertTrue(core.registry.has('openai-default'))
        provider = core.registry.get('openai-default')
        self.assertIsInstance(provider, OpenAiLlmProvider)

    def test_missing_llm_section_is_tolerated(self):
        # A host that doesn't use this lib's config slot must still be
        # able to construct the CoreLib without exploding.
        core = LlmCoreLib({'core_lib': {'data': {'db': {}}}})
        self.assertEqual(core.registry.list(), [])

    def test_bad_connection_in_config_propagates(self):
        cfg = {
            'core_lib': {
                'llm': {
                    'connections': [
                        {
                            'id': 'bad',
                            'provider': 'cohere',  # unsupported
                            'model': 'c4',
                        },
                    ],
                },
            },
        }
        with self.assertRaises(LlmInvalidProviderError):
            LlmCoreLib(cfg)

    def test_missing_provider_config_propagates(self):
        cfg = {
            'core_lib': {
                'llm': {
                    'connections': [
                        {
                            'id': 'nokey',
                            'provider': 'openai',
                            'model': 'gpt-4o-mini',
                            # no api_key
                        },
                    ],
                },
            },
        }
        with self.assertRaises(LlmConfigError):
            LlmCoreLib(cfg)

    def test_namespace_object_config_hydrates(self):
        # Hydra DictConfig supports both attr and item access; emulate
        # that with a SimpleNamespace tree to confirm _attr_or_item
        # handles both paths.
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            core_lib=SimpleNamespace(
                llm=SimpleNamespace(
                    connections=[
                        SimpleNamespace(
                            id='a',
                            provider='openai',
                            model='m',
                            api_key='sk-x',
                            base_url=None,
                            organization=None,
                            region=None,
                            access_key=None,
                            secret_key=None,
                            endpoint_url=None,
                            extra={},
                        ),
                    ],
                ),
            ),
        )
        core = LlmCoreLib(cfg)
        self.assertTrue(core.registry.has('a'))


if __name__ == '__main__':
    unittest.main()
