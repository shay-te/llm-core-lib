"""``LLMView`` — ``isinstance`` semantics the gate relies on.

The safety gate (``llm_core_lib.safety.payload_gate.to_llm_payload``)
uses ``isinstance(item, LLMView)`` to decide whether a tool result is
allowlist-projectable. These tests lock the semantic from both
directions: a subclass passes, anything else (a non-subclass object,
a raw dict, ``None``) fails.

Per the workspace-wide "one TestCase per file" rule (see
``architecture.md`` and this repo's ``AGENTS.md``), this file owns
exactly one TestCase. Shared fixtures live in
``safety_llm_view_helpers``.
"""
from __future__ import annotations

import unittest

from llm_core_lib.safety.llm_view import LLMView

from tests.helpers.safety_llm_view_helpers import StubLLMView


class TestLLMViewIsinstanceCheck(unittest.TestCase):
    """The gate uses ``isinstance(item, LLMView)`` — these tests lock
    that semantic."""

    def test_subclass_instance_passes(self):
        view = StubLLMView(id='u1')
        self.assertIsInstance(view, LLMView)

    def test_non_subclass_fails(self):
        class _NotAView(object):
            def model_dump(self):
                return {}

        self.assertNotIsInstance(_NotAView(), LLMView)

    def test_raw_dict_fails(self):
        self.assertNotIsInstance({'id': 'u1'}, LLMView)

    def test_none_fails(self):
        self.assertNotIsInstance(None, LLMView)


if __name__ == '__main__':
    unittest.main()
