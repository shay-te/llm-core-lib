"""Tests for the :class:`llm_core_lib.LlmCoreLib` composition root."""
import unittest

from llm_core_lib import (
    LlmConfigError,
    LlmCoreLib,
    LlmInvalidProviderError,
    OpenAiConnectionFactory,
)


class TestLlmCoreLibBootstrap(unittest.TestCase):
    def test_empty_constructor_yields_empty_registry(self):
        core = LlmCoreLib()
        self.assertEqual(core.registry.list(), [])

    def test_none_config_yields_empty_registry(self):
        core = LlmCoreLib(None)
        self.assertEqual(core.registry.list(), [])

    def test_dict_config_hydrates_connections(self):
        # ``extra.client`` injects a fake so the test doesn't require
        # the openai SDK to be installed in CI.
        from llm_core_lib.tests.fakes import FakeOpenAIClient
        cfg = {
            'core_lib': {
                'llm': {
                    'connections': [
                        {
                            'id': 'openai-default',
                            'provider': 'openai',
                            'model': 'gpt-4o-mini',
                            'api_key': 'sk-test',
                            'extra': {'client': FakeOpenAIClient()},
                        },
                    ],
                },
            },
        }
        core = LlmCoreLib(cfg)
        self.assertTrue(core.registry.has('openai-default'))
        factory = core.registry.get('openai-default')
        self.assertIsInstance(factory, OpenAiConnectionFactory)

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
        from llm_core_lib.tests.fakes import FakeOpenAIClient
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
                            extra={'client': FakeOpenAIClient()},
                        ),
                    ],
                ),
            ),
        )
        core = LlmCoreLib(cfg)
        self.assertTrue(core.registry.has('a'))

    def test_connections_set_to_none_is_tolerated(self):
        # `connections: null` (or missing the key) must not raise.
        core = LlmCoreLib({'core_lib': {'llm': {'connections': None}}})
        self.assertEqual(core.registry.list(), [])

    def test_connections_set_to_non_iterable_is_tolerated(self):
        # `connections: 12345` is treated as empty rather than crashing
        # the host on boot.
        core = LlmCoreLib({'core_lib': {'llm': {'connections': 12345}}})
        self.assertEqual(core.registry.list(), [])

    def test_connection_from_mapping_with_bad_extra_falls_back_to_empty(self):
        # ``extra`` that isn't dict-coercible silently becomes {} on
        # the resulting LlmConnectionConfig. Tested at the helper
        # boundary so we don't have to spin up a real SDK client to
        # observe the behavior.
        from llm_core_lib.llm_core_lib import _connection_from_mapping
        entry = {
            'id': 'x',
            'provider': 'openai',
            'model': 'm',
            'api_key': 'sk-x',
            'extra': 12345,  # not iterable for dict()
        }
        cfg = _connection_from_mapping(entry)
        self.assertEqual(cfg.extra, {})
        self.assertEqual(cfg.id, 'x')


class TestAttrOrItemHelper(unittest.TestCase):
    """Direct coverage for the defensive config resolver. Public flow
    tests exercise the Mapping + attribute paths; this class pins the
    None and item-access fallbacks so a future refactor doesn't
    silently widen behavior."""

    def setUp(self):
        from llm_core_lib.llm_core_lib import _attr_or_item
        self._attr_or_item = _attr_or_item

    def test_returns_none_when_obj_is_none(self):
        self.assertIsNone(self._attr_or_item(None, 'anything'))

    def test_falls_back_to_item_access_for_subscript_only(self):
        # Object exposes __getitem__ but not the attribute by name —
        # exercises the try/except item path.
        class _SubscriptOnly(object):
            def __getitem__(self, key):
                if key == 'foo':
                    return 'bar'
                raise KeyError(key)

        self.assertEqual(self._attr_or_item(_SubscriptOnly(), 'foo'), 'bar')

    def test_item_access_missing_key_returns_none(self):
        class _AlwaysRaises(object):
            def __getitem__(self, key):
                raise KeyError(key)

        self.assertIsNone(self._attr_or_item(_AlwaysRaises(), 'foo'))


if __name__ == '__main__':
    unittest.main()
