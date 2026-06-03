"""Adversarial tests for the structural safety boundary.

Parallel to ``agent_core_lib/tests/test_pii_adversarial.py`` but for
the *structural* defense, not the regex set. The bypass surface here
is **subclass tricks** a tool author could (accidentally or
deliberately) use to slip data past the gate:

* :class:`~llm_core_lib.safety.llm_view.LLMView` — Pydantic v2 base
  with ``ConfigDict(extra='forbid', frozen=True)``. Subclasses MUST
  inherit those flags; if a subclass overrides ``model_config`` and
  drops one, the safety guarantee silently weakens.
* :func:`~llm_core_lib.safety.payload_gate.to_llm_payload` — the
  choke point. Rejects non-``LLMView`` inputs at the OUTER level;
  but recursive-content enforcement (an ``Any``-typed field carrying
  a raw dict) isn't a feature it can provide without per-field
  type-walking.

Tests are categorized:

* ``test_bypass_*`` — proves an actual smuggling path. These are the
  RED tests that an attacker (or a sloppy refactor) could use; locking
  the current behavior surfaces them as known limitations.
* ``test_gate_rejects_*`` — proves the gate refuses a particular
  shape. Lock for regressions.
* ``test_project_*`` / ``test_run_tool_*`` / ``test_flow_*`` — edge
  cases in the projection helpers and run_tool wrapper.
* ``test_no_leak_*`` — round-trip checks that the LLM-visible payload
  carries nothing it shouldn't.
"""
from __future__ import annotations

import logging
import unittest
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

# Importing the test package first activates the core_lib stub from
# ``llm_core_lib/tests/__init__.py`` — needed locally so the gate's
# StatusCodeException base resolves without the framework installed.
import llm_core_lib.tests  # noqa: F401

from llm_core_lib.safety.llm_view import LLMView
from llm_core_lib.safety.payload_gate import (
    UnsafeToolResultError,
    new_error_ref,
    run_tool,
    sanitized_error_payload,
    to_llm_payload,
)


# Canonical fixture used across the suite.
class _UserView(LLMView):
    id: str
    display_name: str


# ===========================================================================
# BYPASS: subclass tricks that weaken the contract
# ===========================================================================


class TestBypassConfigDictOverride(unittest.TestCase):
    """The single biggest footgun: a subclass that redefines
    ``model_config`` without preserving ``extra='forbid'`` or
    ``frozen=True``. Pydantic v2 does NOT merge ``model_config`` across
    the MRO — the most-derived class's value wins outright. So a tool
    author who writes ``model_config = ConfigDict(arbitrary_types_allowed=True)``
    on their view silently drops both safety flags."""

    def test_bypass_subclass_dropping_extra_forbid_accepts_unknown_fields(self):
        # A subclass that overrides ``model_config`` without
        # ``extra='forbid'`` silently lets extras through. The gate
        # then dumps them as part of ``model_dump()``.
        class _LeakyView(LLMView):
            id: str
            model_config = ConfigDict(extra='allow')  # SAFETY DROPPED

        # Construction allows the unknown field.
        view = _LeakyView(id='u1', email='jane@example.com')
        # And ``model_dump`` exposes it — the gate forwards it.
        payload = to_llm_payload(view)
        self.assertEqual(payload.get('email'), 'jane@example.com')
        # Lock this as a KNOWN bypass — the long-term mitigation is
        # either (a) freezing ``LLMView.model_config`` via a metaclass
        # check, or (b) a unit-test sweep across every LLMView
        # subclass asserting ``extra='forbid'`` is still set.

    def test_bypass_subclass_dropping_frozen_allows_post_init_mutation(self):
        class _MutableView(LLMView):
            id: str
            note: str = ''
            model_config = ConfigDict(extra='forbid', frozen=False)

        view = _MutableView(id='u1', note='clean')
        # Without ``frozen=True`` we can splice PII onto an
        # already-built view AFTER any upstream allowlist check.
        view.note = 'now contains jane@example.com'
        payload = to_llm_payload(view)
        self.assertIn('jane@example.com', payload['note'])

    def test_catch_subclass_keeping_both_flags_still_safe(self):
        # The reference safe pattern — any new view should look like
        # this. Locks the contract for the rest of the codebase.
        class _SafeView(LLMView):
            id: str
            model_config = ConfigDict(extra='forbid', frozen=True)

        with self.assertRaises(ValidationError):
            _SafeView(id='u1', email='jane@example.com')


class TestBypassComputedFieldLeaks(unittest.TestCase):
    """Pydantic v2's ``@computed_field`` decorator exposes a derived
    value through ``model_dump()`` even though it is NOT in
    ``model_fields``. So ``LLMView.allowed_field_names()`` (which reads
    ``model_fields``) reports an under-count, and the
    "fields are locked" style tests miss the computed leak."""

    def test_bypass_computed_field_exposes_data_not_in_allowlist(self):
        class _ComputedLeakView(LLMView):
            id: str
            _email: str = 'jane@example.com'  # private; not a field

            model_config = ConfigDict(extra='forbid', frozen=True)

            @computed_field
            @property
            def derived_contact(self) -> str:
                # Reads from a private attribute and exposes it on dump.
                return self._email

        view = _ComputedLeakView(id='u1')
        payload = to_llm_payload(view)
        # The computed field is in the dumped payload.
        self.assertIn('derived_contact', payload)
        self.assertEqual(payload['derived_contact'], 'jane@example.com')
        # And the "allowlist" misses it — allowed_field_names() only
        # walks ``model_fields``.
        self.assertNotIn('derived_contact', view.allowed_field_names())


class TestBypassAnyTypedNestedDict(unittest.TestCase):
    """The gate enforces the OUTER allowlist but does not recursively
    type-walk nested values. A field typed ``Any`` / ``Dict[str, Any]``
    / ``List[Dict[str, Any]]`` accepts arbitrary content — and that
    content survives ``model_dump()`` verbatim."""

    def test_bypass_any_field_smuggles_raw_dict(self):
        class _BagView(LLMView):
            id: str
            data: Any = None

        view = _BagView(id='u1', data={
            'email': 'jane@example.com',
            'ssn': '123-45-6789',
            'card': '4242 4242 4242 4242',
        })
        payload = to_llm_payload(view)
        # The raw PII-bearing dict is in the payload as-is.
        self.assertEqual(payload['data']['email'], 'jane@example.com')
        # Mitigation: this is what the scrub_pii backstop in the host
        # app is for — the gate alone can't prevent this and shouldn't
        # try (recursive type-walking would require per-field schema
        # introspection across arbitrarily-nested generic types).

    def test_bypass_any_field_smuggles_list_of_dicts(self):
        class _ListBagView(LLMView):
            id: str
            items: List[Dict[str, Any]] = Field(default_factory=list)

        view = _ListBagView(id='u1', items=[
            {'email': 'a@b.com'},
            {'phone': '+1 555 123 4567'},
        ])
        payload = to_llm_payload(view)
        self.assertEqual(payload['items'][0]['email'], 'a@b.com')

    def test_bypass_nested_non_llmview_basemodel(self):
        # A field typed as another (non-LLMView) BaseModel — Pydantic
        # nests cleanly, but the gate has no idea the nested model is
        # exposing fields outside the safety hierarchy.
        class _RawNested(BaseModel):
            email: str
            ssn: str

        class _OuterView(LLMView):
            id: str
            nested: _RawNested

        view = _OuterView(
            id='u1',
            nested=_RawNested(email='jane@example.com', ssn='123-45-6789'),
        )
        payload = to_llm_payload(view)
        self.assertEqual(payload['nested']['email'], 'jane@example.com')


class TestBypassFrozenViaObjectSetattr(unittest.TestCase):
    """Pydantic v2's ``frozen=True`` overrides ``__setattr__`` but
    ``object.__setattr__`` bypasses that. Hostile in-process code
    can still mutate. This is a Python-language limitation, not
    something the safety layer can defend against — but lock the
    behavior so it's a known surface."""

    def test_bypass_object_setattr_circumvents_frozen(self):
        view = _UserView(id='u1', display_name='Jane')
        # Normal assignment is blocked.
        with self.assertRaises(ValidationError):
            view.display_name = 'somebody else'  # type: ignore[misc]
        # But the language-level escape hatch isn't.
        object.__setattr__(view, 'display_name', 'spliced after build')
        self.assertEqual(view.display_name, 'spliced after build')
        # No mitigation at the type-system level; this is "trust the
        # process boundary". Audit log + code review are the controls.


class TestBypassRootModelExposesAnyRoot(unittest.TestCase):
    """A subclass of Pydantic's ``RootModel`` (or a workaround mimicking
    one) carries a single ``root`` value of arbitrary type. If someone
    builds an ``LLMView`` whose only field is ``root: Any``, that view
    is effectively unbounded."""

    def test_bypass_root_any_field_is_unbounded(self):
        class _RootAnyView(LLMView):
            root: Any

        view = _RootAnyView(root={'email': 'jane@example.com', 'card': '4242...'})
        payload = to_llm_payload(view)
        self.assertEqual(payload['root']['email'], 'jane@example.com')


# ===========================================================================
# GATE REJECTS: structural boundary keeps these out
# ===========================================================================


class TestGateRejectsNonLLMViewInputs(unittest.TestCase):
    """Positive-side locking — the gate refuses these. Tightens the
    perimeter; a regression that accepted any of them would be a
    real safety bug."""

    def test_gate_rejects_bare_basemodel(self):
        class _NotAView(BaseModel):
            email: str

        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(_NotAView(email='jane@example.com'))

    def test_gate_rejects_str(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload('jane@example.com')

    def test_gate_rejects_int(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(42)

    def test_gate_rejects_bool(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(True)

    def test_gate_rejects_tuple(self):
        # ``to_llm_payload`` treats only ``list`` as a collection input.
        # A tuple goes through the single-item branch and fails the
        # isinstance check. Lock that — if we ever decide tuples are
        # OK too, this test flips.
        view = _UserView(id='u1', display_name='Jane')
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload((view,))

    def test_gate_rejects_set_of_views(self):
        view = _UserView(id='u1', display_name='Jane')
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload({view})  # type: ignore[arg-type]

    def test_gate_rejects_generator_yielding_views(self):
        def _gen():
            yield _UserView(id='u1', display_name='Jane')
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(_gen())

    def test_gate_rejects_class_object_not_instance(self):
        # Passing the LLMView CLASS itself (not an instance) is the
        # canonical foot-gun for a forgetful ``return MyView`` instead
        # of ``return MyView(...)``.
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(_UserView)

    def test_gate_rejects_list_containing_none(self):
        view = _UserView(id='u1', display_name='Jane')
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload([view, None])

    def test_gate_rejects_mixed_list_views_and_dicts(self):
        view = _UserView(id='u1', display_name='Jane')
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload([view, {'id': 'u2'}])

    def test_gate_rejects_dict_keyed_by_views(self):
        view = _UserView(id='u1', display_name='Jane')
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload({view: 'value'})  # type: ignore[arg-type]


class TestGateAccepts(unittest.TestCase):
    """Locked positive cases — what the gate intentionally lets
    through. None passes for "nothing to report"; empty list returns
    empty list."""

    def test_gate_accepts_none(self):
        self.assertIsNone(to_llm_payload(None))

    def test_gate_accepts_empty_list(self):
        self.assertEqual(to_llm_payload([]), [])

    def test_gate_accepts_single_view(self):
        view = _UserView(id='u1', display_name='Jane')
        self.assertEqual(to_llm_payload(view), {'id': 'u1', 'display_name': 'Jane'})

    def test_gate_accepts_list_of_views(self):
        views = [
            _UserView(id='u1', display_name='Jane'),
            _UserView(id='u2', display_name='John'),
        ]
        payload = to_llm_payload(views)
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]['id'], 'u1')


# ===========================================================================
# PROJECT: edge cases in LLMView.project / project_list
# ===========================================================================


class TestProjectEdgeCases(unittest.TestCase):
    def test_project_drops_keys_not_in_allowlist(self):
        raw = {'id': 'u1', 'display_name': 'Jane', 'email': 'leak@x.com'}
        view = _UserView.project(raw)
        self.assertEqual(view.to_dict(), {'id': 'u1', 'display_name': 'Jane'})

    def test_project_raises_on_missing_required_field(self):
        # Required field absent → Pydantic raises. Caller's job to
        # declare ``Optional[...] = None`` if they want tolerance.
        with self.assertRaises(ValidationError):
            _UserView.project({'id': 'u1'})  # display_name required

    def test_project_handles_attribute_shaped_input(self):
        class _Row:
            id = 'u1'
            display_name = 'Jane'
            email = 'leak@x.com'
        view = _UserView.project(_Row())
        self.assertEqual(view.to_dict(), {'id': 'u1', 'display_name': 'Jane'})

    def test_project_attribute_branch_uses_hasattr_check(self):
        # An attribute-shaped input MISSING a required field hits the
        # branch where ``hasattr`` returns False; the missing kwarg
        # surfaces as a ValidationError. Lock both branches.
        class _Partial:
            id = 'u1'  # display_name absent
        with self.assertRaises(ValidationError):
            _UserView.project(_Partial())

    def test_project_with_explicit_none_value_uses_none(self):
        # An upstream key present with a None value passes None to
        # the constructor — which for a *required* field is a
        # ValidationError. Lock that — silent-coerce would mask
        # data-quality issues upstream.
        with self.assertRaises(ValidationError):
            _UserView.project({'id': 'u1', 'display_name': None})

    def test_project_none_input_returns_none(self):
        self.assertIsNone(_UserView.project(None))

    def test_project_list_with_none_input_returns_empty(self):
        self.assertEqual(_UserView.project_list(None), [])

    def test_project_list_filters_none_items_silently(self):
        out = _UserView.project_list([
            {'id': 'u1', 'display_name': 'Jane'},
            None,
            {'id': 'u2', 'display_name': 'John'},
        ])
        self.assertEqual(len(out), 2)

    def test_project_list_promotes_single_dict_to_list(self):
        out = _UserView.project_list({'id': 'u1', 'display_name': 'Jane'})
        self.assertEqual(len(out), 1)

    def test_project_list_promotes_attribute_shaped_object_to_list(self):
        class _Row:
            id = 'u1'
            display_name = 'Jane'
        out = _UserView.project_list(_Row())
        self.assertEqual(len(out), 1)

    def test_project_list_tuple_input(self):
        out = _UserView.project_list((
            {'id': 'u1', 'display_name': 'Jane'},
            {'id': 'u2', 'display_name': 'John'},
        ))
        self.assertEqual(len(out), 2)


# ===========================================================================
# RUN_TOOL: edge cases
# ===========================================================================


class TestRunToolEdgeCases(unittest.TestCase):
    def test_run_tool_with_named_logger(self):
        captured = []

        class _Logger:
            def exception(self, *args, **_kwargs) -> None:
                captured.append(args)

        def _boom() -> None:
            raise RuntimeError('User jane@example.com not found')

        payload = run_tool(_boom, logger=_Logger())
        self.assertEqual(payload['status'], 'error')
        self.assertNotIn('jane@example.com', str(payload))
        # The logger received the full detail.
        self.assertTrue(captured)

    def test_run_tool_default_logger_when_none_supplied(self):
        # No logger arg — falls back to ``logging.getLogger('llm_core_lib.safety')``
        # and the assertion stays the same: payload generic, raw msg gone.
        def _boom() -> None:
            raise RuntimeError('User jane@example.com not found')
        with self.assertLogs('llm_core_lib.safety', level='ERROR'):
            payload = run_tool(_boom)
        self.assertEqual(payload['status'], 'error')

    def test_run_tool_passes_through_args_and_kwargs(self):
        captured = {}

        def _spy(a, b, *, c) -> Optional[_UserView]:
            captured['a'], captured['b'], captured['c'] = a, b, c
            return None

        payload = run_tool(_spy, 1, 2, c=3)
        self.assertEqual(captured, {'a': 1, 'b': 2, 'c': 3})
        self.assertIsNone(payload)

    def test_run_tool_catches_unsafe_tool_result_error(self):
        # When a tool forgets to project, the gate raises
        # ``UnsafeToolResultError`` inside the try; the catch turns
        # that into the same generic envelope as any other failure.
        def _forgot() -> Any:
            return {'id': 1, 'email': 'jane@example.com'}
        with self.assertLogs('llm_core_lib.safety', level='ERROR'):
            payload = run_tool(_forgot)
        self.assertEqual(payload['status'], 'error')
        self.assertNotIn('jane@example.com', str(payload))

    def test_run_tool_callable_without_name_uses_repr(self):
        # Lambdas have ``__name__ == '<lambda>'``; a more interesting
        # case is a callable instance without ``__name__``. The gate
        # falls back to ``repr(fn)`` so the log line still has SOMETHING.
        class _CallableNoName:
            def __call__(self) -> None:
                raise RuntimeError('boom')

        with self.assertLogs('llm_core_lib.safety', level='ERROR'):
            payload = run_tool(_CallableNoName())
        self.assertEqual(payload['status'], 'error')


# ===========================================================================
# NEW_ERROR_REF / SANITIZED_ERROR_PAYLOAD
# ===========================================================================


class TestNewErrorRef(unittest.TestCase):
    def test_returns_8_char_hex(self):
        ref = new_error_ref()
        self.assertEqual(len(ref), 8)
        # All chars are hex.
        int(ref, 16)  # raises ValueError if not hex

    def test_successive_refs_are_distinct(self):
        # Collision is astronomically unlikely (~1e-9 for 8-hex = 32
        # bits worth), so a deterministic check across a few hundred
        # is a reasonable sanity test.
        refs = {new_error_ref() for _ in range(500)}
        self.assertEqual(len(refs), 500)


class TestSanitizedErrorPayload(unittest.TestCase):
    def test_shape_is_status_detail_ref(self):
        payload = sanitized_error_payload('abc12345')
        self.assertEqual(set(payload.keys()), {'status', 'detail', 'ref'})

    def test_detail_string_is_fixed(self):
        # The literal is the LLM-facing contract — must not change
        # casually (operators may have matched on it in logs / metrics).
        self.assertEqual(
            sanitized_error_payload('x')['detail'],
            'The request could not be completed.',
        )

    def test_detail_is_pii_free(self):
        # The detail text itself must carry no PII shape. If a future
        # change introduces e.g. "contact admin@example.com" in the
        # detail, this catches it.
        text = sanitized_error_payload('abc12345')['detail']
        self.assertNotIn('@', text)
        self.assertNotIn('http', text.lower())


# ===========================================================================
# NO LEAK: round-trip checks on the gate
# ===========================================================================


class TestNoLeakRoundTrip(unittest.TestCase):
    """End-to-end: a payload that's been through the gate carries
    nothing it shouldn't."""

    def test_no_leak_failure_payload_carries_no_arbitrary_data(self):
        class _Boom(Exception):
            def __init__(self) -> None:
                super().__init__('User jane@example.com SSN 123-45-6789')
                self.user_id = 42
                self.email = 'jane@example.com'

        def _bad() -> None:
            raise _Boom()

        with self.assertLogs('llm_core_lib.safety', level='ERROR'):
            payload = run_tool(_bad)

        blob = str(payload)
        # None of the exception's args, attributes, or class name
        # leak into the LLM-visible payload.
        self.assertNotIn('jane@example.com', blob)
        self.assertNotIn('123-45-6789', blob)
        self.assertNotIn('42', blob.replace(payload['ref'], ''))
        self.assertNotIn('_Boom', blob)

    def test_no_leak_gate_error_does_not_include_type_name_in_payload(self):
        # ``UnsafeToolResultError`` carries the offending type's name
        # in its message — that detail must stay in the log line, not
        # the LLM-bound payload.
        class _SecretOrm:
            secret = 'hidden'

        def _leak() -> Any:
            return _SecretOrm()

        with self.assertLogs('llm_core_lib.safety', level='ERROR'):
            payload = run_tool(_leak)
        self.assertEqual(payload['status'], 'error')
        self.assertNotIn('_SecretOrm', str(payload))
        self.assertNotIn('hidden', str(payload))

    def test_no_leak_view_to_dict_only_exposes_declared_fields(self):
        # Build a view, mutate around the boundary, dump.
        view = _UserView(id='u1', display_name='Jane')
        # ``to_dict`` is the documented contract entry point.
        self.assertEqual(set(view.to_dict().keys()), {'id', 'display_name'})
        # And ``model_dump`` (the underlying Pydantic API) returns the
        # same set — no synthetic fields, no class-attr leakage.
        self.assertEqual(set(view.model_dump().keys()), {'id', 'display_name'})


# ===========================================================================
# DOCUMENTED KNOWN-LIMITATIONS REGISTRY
# ===========================================================================


class TestKnownLimitationsCatalog(unittest.TestCase):
    """Roll-up — categories of structural bypass we know about. If
    this set drifts, the discussion in this file (and the comment
    block in ``pii_patterns.py`` about the layered defense) should
    drift too."""

    KNOWN_BYPASS_CATEGORIES = frozenset({
        # subclass-level overrides
        'subclass_overrides_model_config_dropping_extra_forbid',
        'subclass_overrides_model_config_dropping_frozen',
        # field-level escapes
        'computed_field_exposes_value_outside_allowlist',
        'any_typed_field_carries_arbitrary_dict',
        'list_of_any_dict_field',
        'nested_non_llmview_basemodel',
        'root_typed_any',
        # language-level
        'object_setattr_bypasses_frozen',
    })

    def test_catalog_is_locked(self):
        self.assertEqual(len(self.KNOWN_BYPASS_CATEGORIES), 8)


if __name__ == '__main__':
    unittest.main()
