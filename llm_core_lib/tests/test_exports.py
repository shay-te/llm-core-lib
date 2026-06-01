"""Lock the package's public surface — every name listed in
``llm_core_lib.__all__`` must be importable from the package root."""
import unittest

import llm_core_lib


class TestPublicExports(unittest.TestCase):
    def test_every_all_name_resolves(self):
        for name in llm_core_lib.__all__:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(llm_core_lib, name),
                    f'llm_core_lib is missing public export {name!r}',
                )

    def test_factories_are_classes(self):
        from llm_core_lib import (
            AnthropicConnectionFactory,
            BedrockConnectionFactory,
            LlmConnectionRegistry,
            LlmCoreLib,
            OpenAiConnectionFactory,
        )

        for cls in (
            LlmCoreLib,
            LlmConnectionRegistry,
            OpenAiConnectionFactory,
            AnthropicConnectionFactory,
            BedrockConnectionFactory,
        ):
            self.assertTrue(isinstance(cls, type), f'{cls!r} should be a class')

    def test_create_connection_factory_is_callable(self):
        from llm_core_lib import create_connection_factory
        self.assertTrue(callable(create_connection_factory))

    def test_provider_ids_match_literal(self):
        from llm_core_lib import LLM_PROVIDER_IDS
        self.assertEqual(set(LLM_PROVIDER_IDS), {'openai', 'anthropic', 'bedrock'})


if __name__ == '__main__':
    unittest.main()
