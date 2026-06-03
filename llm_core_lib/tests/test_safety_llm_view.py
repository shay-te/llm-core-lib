"""Tests for :class:`llm_core_lib.safety.llm_view.LLMView`.

The view base is the allowlist mechanism — anything that escapes the
field list it declares can leak to the LLM. These tests lock the four
invariants the Pydantic v2 ``ConfigDict(extra='forbid', frozen=True)``
contract gives us:

1. Unknown init kwargs raise :class:`pydantic.ValidationError` at
   construction.
2. Post-init assignment raises :class:`pydantic.ValidationError`
   (``frozen=True``).
3. ``to_dict()`` / ``model_dump()`` returns exactly the declared field
   set.
4. ``allowed_field_names()`` is the operator-readable allowlist that
   reviewer-pinned tests can lock against.
"""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from llm_core_lib.safety.llm_view import LLMView


class _UserLLMView(LLMView):
    id: str
    display_name: str


class _OrderLLMView(LLMView):
    id: str
    total_cents: int


class TestLLMViewRejectsUnknownFields(unittest.TestCase):
    def test_unknown_init_kwarg_raises_at_construction(self):
        with self.assertRaises(ValidationError):
            _UserLLMView(id='u1', display_name='Jane', email='jane@example.com')

    def test_post_init_assignment_is_rejected(self):
        view = _UserLLMView(id='u1', display_name='Jane')
        with self.assertRaises(ValidationError):
            view.display_name = 'somebody else'


class TestLLMViewToDict(unittest.TestCase):
    def test_to_dict_returns_exactly_declared_fields(self):
        view = _UserLLMView(id='u1', display_name='Jane')
        self.assertEqual(view.to_dict(), {'id': 'u1', 'display_name': 'Jane'})

    def test_model_dump_returns_exactly_declared_fields(self):
        # Pydantic v2's idiomatic API — to_dict is a thin wrapper, but
        # callers that prefer model_dump should get the same result.
        view = _UserLLMView(id='u1', display_name='Jane')
        self.assertEqual(view.model_dump(), {'id': 'u1', 'display_name': 'Jane'})


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

    def test_model_fields_match_allowed_names(self):
        # The classmethod is a thin wrapper over Pydantic's
        # ``model_fields``; lock that they agree.
        self.assertEqual(
            frozenset(_UserLLMView.model_fields.keys()),
            _UserLLMView.allowed_field_names(),
        )


class TestLLMViewProject(unittest.TestCase):
    """``project`` is the adapter every tool's last line calls.

    The contract: take an arbitrary upstream dict (an ORM row that's
    been ``ResultToDict``'d, an Elasticsearch hit, a service return),
    drop every key not on the view's allowlist, build the view from
    what's left. Locked behaviors:
    """

    def test_drops_keys_not_on_allowlist(self):
        # Upstream returned an email but the view's allowlist doesn't
        # declare one — email must NOT make it onto the view.
        raw = {'id': 'u1', 'display_name': 'Jane', 'email': 'jane@example.com'}
        view = _UserLLMView.project(raw)
        self.assertEqual(view.to_dict(), {'id': 'u1', 'display_name': 'Jane'})
        self.assertFalse(hasattr(view, 'email'))

    def test_none_input_returns_none(self):
        self.assertIsNone(_UserLLMView.project(None))

    def test_missing_allowlisted_key_is_omitted_not_defaulted(self):
        # Pydantic v2 with no explicit default on a field would raise
        # on missing input. The fields in the test view are required,
        # so a dict missing ``display_name`` must raise — caller's job
        # to declare ``Optional[...] = None`` if they want tolerance.
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            _UserLLMView.project({'id': 'u1'})

    def test_attribute_shaped_input_is_supported(self):
        # An ORM row / SimpleNamespace passes through too — the same
        # gate handles both shapes so tools don't need a manual dict-cast.
        class _Row(object):
            id = 'u1'
            display_name = 'Jane'
            email = 'jane@example.com'  # not on the allowlist — must be dropped

        view = _UserLLMView.project(_Row())
        self.assertEqual(view.to_dict(), {'id': 'u1', 'display_name': 'Jane'})


class TestLLMViewProjectList(unittest.TestCase):
    def test_empty_or_none_returns_empty_list(self):
        self.assertEqual(_UserLLMView.project_list(None), [])
        self.assertEqual(_UserLLMView.project_list([]), [])

    def test_each_item_is_projected(self):
        raws = [
            {'id': 'u1', 'display_name': 'Jane', 'email': 'jane@example.com'},
            {'id': 'u2', 'display_name': 'John', 'phone': '+1 555 1234'},
        ]
        views = _UserLLMView.project_list(raws)
        self.assertEqual(len(views), 2)
        # Per-view allowlist drops both the email and the phone.
        self.assertEqual(views[0].to_dict(), {'id': 'u1', 'display_name': 'Jane'})
        self.assertEqual(views[1].to_dict(), {'id': 'u2', 'display_name': 'John'})

    def test_single_item_input_is_promoted_to_list(self):
        # A function that returned a single record (not a list) should
        # still be project_list-able for a uniform callsite.
        views = _UserLLMView.project_list({'id': 'u1', 'display_name': 'Jane'})
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].to_dict(), {'id': 'u1', 'display_name': 'Jane'})

    def test_none_items_inside_list_are_skipped(self):
        views = _UserLLMView.project_list([
            {'id': 'u1', 'display_name': 'Jane'},
            None,
            {'id': 'u2', 'display_name': 'John'},
        ])
        self.assertEqual(len(views), 2)


class TestLLMViewConfigDictContract(unittest.TestCase):
    """The base's ConfigDict is the contract — lock its values directly
    so a future refactor that loosens ``extra='forbid'`` or drops
    ``frozen=True`` is caught at the test layer, not in production."""

    def test_extra_is_forbid(self):
        self.assertEqual(LLMView.model_config.get('extra'), 'forbid')

    def test_frozen_is_true(self):
        self.assertTrue(LLMView.model_config.get('frozen'))


if __name__ == '__main__':
    unittest.main()
