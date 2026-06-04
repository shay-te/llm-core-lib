"""Tests for :class:`llm_core_lib.safety.llm_view.LLMView`.

The transport-layer marker. Stdlib-only, no Pydantic dependency.
What the gate ``isinstance``-checks against; the actual allowlist
enforcement (``extra='forbid'``, ``frozen=True``, field declarations)
lives one layer up in :class:`agent_core_lib.safety.llm_view.LLMView`
where the Pydantic dep lives — those behaviors are tested in
``agent-core-lib/tests/test_safety_llm_view.py`` instead.

Three invariants locked here:

1. The marker is importable and is a class.
2. A stdlib subclass passes the ``isinstance`` check.
3. A class that does NOT subclass the marker fails the check.

That's the entire contract this layer owns. Everything else is the
concrete agent-layer view's job.
"""
from __future__ import annotations

import unittest

from llm_core_lib.safety.llm_view import LLMView


class _StubLLMView(LLMView):
    """A non-Pydantic concrete subclass — exercises the contract the
    transport marker advertises (``model_dump`` is the one method the
    gate calls)."""

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self):
        return dict(self._fields)


class TestLLMViewIsAClass(unittest.TestCase):
    def test_llm_view_is_a_class(self):
        self.assertTrue(isinstance(LLMView, type))

    def test_llm_view_can_be_subclassed(self):
        view = _StubLLMView(id='u1', display_name='Jane')
        self.assertEqual(view.model_dump(), {'id': 'u1', 'display_name': 'Jane'})


class TestLLMViewIsinstanceCheck(unittest.TestCase):
    """The gate uses ``isinstance(item, LLMView)`` — these tests lock
    that semantic."""

    def test_subclass_instance_passes(self):
        view = _StubLLMView(id='u1')
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
