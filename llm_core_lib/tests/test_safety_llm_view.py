"""Tests for :class:`llm_core_lib.safety.llm_view.LLMView`.

The view base is the allowlist mechanism — anything that escapes the
field list it declares can leak to the LLM. These tests lock the three
invariants: unknown init kwargs are rejected, post-init assignment is
rejected, and ``to_dict`` returns exactly the declared field set.
"""
from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, dataclass

from llm_core_lib.safety.llm_view import LLMView


@dataclass(frozen=True)
class _UserLLMView(LLMView):
    id: str
    display_name: str


@dataclass(frozen=True)
class _OrderLLMView(LLMView):
    id: str
    total_cents: int


class TestLLMViewRejectsUnknownFields(unittest.TestCase):
    def test_unknown_init_kwarg_raises_at_construction(self):
        with self.assertRaises(TypeError):
            _UserLLMView(id='u1', display_name='Jane', email='jane@example.com')  # type: ignore[call-arg]

    def test_post_init_assignment_is_rejected(self):
        view = _UserLLMView(id='u1', display_name='Jane')
        with self.assertRaises(FrozenInstanceError):
            view.email = 'jane@example.com'  # type: ignore[attr-defined]


class TestLLMViewToDict(unittest.TestCase):
    def test_to_dict_returns_exactly_declared_fields(self):
        view = _UserLLMView(id='u1', display_name='Jane')
        self.assertEqual(view.to_dict(), {'id': 'u1', 'display_name': 'Jane'})

    def test_to_dict_on_non_dataclass_subclass_raises(self):
        class _BadView(LLMView):
            pass

        with self.assertRaises(TypeError):
            _BadView().to_dict()


class TestLLMViewAllowedFieldNames(unittest.TestCase):
    """Locks the contract behind ``test_user_view_fields_are_locked``
    in the task description — a reviewer can pin the allowlist and the
    test fails the moment someone widens it without updating the test."""

    def test_user_view_fields_are_locked(self):
        self.assertEqual(
            _UserLLMView.allowed_field_names(),
            frozenset({'id', 'display_name'}),
        )

    def test_order_view_fields_are_locked(self):
        self.assertEqual(
            _OrderLLMView.allowed_field_names(),
            frozenset({'id', 'total_cents'}),
        )

    def test_allowed_field_names_on_non_dataclass_subclass_raises(self):
        class _BadView(LLMView):
            pass

        with self.assertRaises(TypeError):
            _BadView.allowed_field_names()


if __name__ == '__main__':
    unittest.main()
