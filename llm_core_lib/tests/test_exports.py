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

    def test_top_level_imports_are_callable_or_class(self):
        # Sanity: a representative sample resolves to the expected kind.
        from llm_core_lib import (
            AnthropicLlmProvider,
            BedrockLlmProvider,
            LlmConnectionRegistry,
            LlmCoreLib,
            LlmProvider,
            OpenAiLlmProvider,
            create_llm_provider,
        )

        for cls in (
            LlmCoreLib,
            LlmConnectionRegistry,
            LlmProvider,
            OpenAiLlmProvider,
            AnthropicLlmProvider,
            BedrockLlmProvider,
        ):
            self.assertTrue(isinstance(cls, type), f'{cls!r} should be a class')
        self.assertTrue(callable(create_llm_provider))

    def test_provider_ids_match_literal(self):
        # LLM_PROVIDER_IDS is the runtime mirror of the LlmProviderId
        # Literal — keep them in sync. The check here pins the literal
        # values; if the Literal grows, this test grows with it.
        from llm_core_lib import LLM_PROVIDER_IDS

        self.assertEqual(set(LLM_PROVIDER_IDS), {'openai', 'anthropic', 'bedrock'})


if __name__ == '__main__':
    unittest.main()
