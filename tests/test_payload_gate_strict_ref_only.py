"""Strict-mode tests for :func:`llm_core_lib.safety.payload_gate.to_llm_payload`.

The gate now refuses ANYTHING that is not a :class:`RefLLMView`
subclass — even a rich :class:`LLMView` carrying entity data (name,
email, status) is rejected. The structural rule is what guarantees PII
cannot reach the model: a tool author can't decide on their own to
ship "just one display field"; the type system stops them.

These tests pin the contract:

  * ``RefLLMView`` (id-only) → dumped to ``{'id': N}``.
  * ``list[RefLLMView]`` → dumped to ``[{'id': N}, ...]``.
  * ``None`` → passes through.
  * Plain ``LLMView`` (the transport marker) carrying entity data →
    raises :class:`UnsafeToolResultError`. This is the regression
    guard: the older gate accepted any ``LLMView``; the strict gate
    must not.
  * Raw dict / primitive → raises (same as before).
"""
from __future__ import annotations

import unittest

from llm_core_lib.safety.llm_view import LLMView, RefLLMView
from llm_core_lib.safety.payload_gate import (
    UnsafeToolResultError,
    to_llm_payload,
)


class _StubRefLLMView(RefLLMView):
    """Stdlib RefLLMView subclass — the gate's ``isinstance`` check
    passes; ``model_dump`` returns id-only."""

    def __init__(self, id):
        self.id = id

    def model_dump(self):
        return {'id': self.id}


class _RichLLMView(LLMView):
    """Stdlib non-Ref ``LLMView`` carrying display fields. The
    transport marker's ``isinstance(item, LLMView)`` succeeds but the
    strict gate's ``isinstance(item, RefLLMView)`` must NOT — this
    fixture exercises the regression guard."""

    def __init__(self, id, display_name):
        self.id = id
        self.display_name = display_name

    def model_dump(self):
        return {'id': self.id, 'display_name': self.display_name}


class TestToLlmPayloadAcceptsRefLLMView(unittest.TestCase):
    def test_single_ref_returns_id_only_dict(self):
        self.assertEqual(to_llm_payload(_StubRefLLMView(id=1)), {'id': 1})

    def test_list_of_refs_returns_list_of_id_dicts(self):
        self.assertEqual(
            to_llm_payload([_StubRefLLMView(id=1), _StubRefLLMView(id=2)]),
            [{'id': 1}, {'id': 2}],
        )

    def test_none_passes_through(self):
        self.assertIsNone(to_llm_payload(None))


class TestToLlmPayloadRefusesRichViews(unittest.TestCase):
    """The regression guard — a plain ``LLMView`` carrying entity
    data must NOT cross the gate. The previous (lenient) version of
    the gate accepted any ``LLMView``; the strict version rejects."""

    def test_rich_llm_view_raises(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(_RichLLMView(id=1, display_name='Jane'))

    def test_list_with_rich_view_raises_on_first_unsafe_item(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload([
                _StubRefLLMView(id=1),
                _RichLLMView(id=2, display_name='Jane'),
            ])


class TestToLlmPayloadRefusesUnsafePrimitives(unittest.TestCase):
    """Anything that isn't a ``RefLLMView`` subclass — primitive,
    plain dict, or raw object — raises."""

    def test_plain_dict_raises(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload({'id': 1})

    def test_int_primitive_raises(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(123)

    def test_string_primitive_raises(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload('hello')

    def test_list_with_plain_dict_raises(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload([{'id': 1}])


if __name__ == '__main__':
    unittest.main()
