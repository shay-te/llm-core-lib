"""``LLMView`` — class identity + subclass-able contract.

The transport-layer marker must be importable as a class so the gate
can ``isinstance``-check against it, and must be subclassable so a
concrete view (the Pydantic-backed one in agent-core-lib, or the
stdlib stub here) can satisfy ``model_dump``.

Per the workspace-wide "one TestCase per file" rule (see
``architecture.md`` and this repo's ``AGENTS.md``), this file owns
exactly one TestCase. Shared fixtures live in
``safety_llm_view_helpers``.
"""
from __future__ import annotations

import unittest

from llm_core_lib.safety.llm_view import LLMView

from llm_core_lib.tests.safety_llm_view_helpers import StubLLMView


class TestLLMViewIsAClass(unittest.TestCase):
    def test_llm_view_is_a_class(self):
        self.assertTrue(isinstance(LLMView, type))

    def test_llm_view_can_be_subclassed(self):
        view = StubLLMView(id='u1', display_name='Jane')
        self.assertEqual(view.model_dump(), {'id': 'u1', 'display_name': 'Jane'})


if __name__ == '__main__':
    unittest.main()
