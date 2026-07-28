"""Adversarial tests for the transport-layer safety boundary.

This file owns only the probes that exercise the **transport-layer**
defense — the :class:`LLMView` marker (a plain class with no Pydantic
dep) and the gate (:func:`to_llm_payload` + :func:`run_tool`). The
subclass-tricks-against-Pydantic family (``ConfigDict`` overrides,
computed-field leaks, ``Any``-typed nested dicts, frozen-via-setattr,
``RootModel`` with ``Any`` root, ``project`` / ``project_list`` edge
cases) is the agent-layer's concern and lives in
``agent-core-lib/tests/test_safety_adversarial.py`` next to the
Pydantic-backed concrete view.

Categories:

* ``test_gate_rejects_*`` — proves the gate refuses a particular
  non-``LLMView`` shape. Locks regressions.
* ``test_gate_accepts_*`` — proves the gate accepts the legitimate
  shapes (single view, list of views, ``None``, empty list).
* ``test_run_tool_*`` — edge cases in the wrapper.
* ``test_new_error_ref`` / ``test_sanitized_error_payload_*`` — the
  generic error envelope shape contract.
* ``test_no_leak_*`` — round-trip: nothing the tool authored should
  reach the LLM-bound payload beyond the declared field list.
"""
from __future__ import annotations

import logging
import unittest

# Importing the test package first activates the core_lib stub from
# ``llm_core_lib/tests/__init__.py`` — needed locally so the gate's
# StatusCodeException base resolves without the framework installed.
import llm_core_lib.tests  # noqa: F401

from llm_core_lib.safety.llm_view import RefLLMView
from llm_core_lib.safety.payload_gate import (
    UnsafeToolResultError,
    new_error_ref,
    run_tool,
    sanitized_error_payload,
    to_llm_payload,
)


class _UserView(RefLLMView):
    """Stdlib subclass of the transport marker — the gate only sees
    ``isinstance(item, LLMView)`` and ``item.model_dump()``, so this
    minimal class is enough to exercise every gate-behavior probe.
    The Pydantic-backed equivalent (with ``ConfigDict`` enforcement)
    lives in ``agent_core_lib.safety.llm_view`` and the bypass probes
    against it live in agent-core-lib's adversarial test file."""

    def __init__(self, id, display_name):
        self.id = id
        self.display_name = display_name

    def model_dump(self):
        return {'id': self.id, 'display_name': self.display_name}


# ===========================================================================
# GATE — refusal probes
# ===========================================================================


class TestGateRejectsNonLLMViewInputs(unittest.TestCase):
    """Every non-``LLMView`` input shape that could plausibly reach the
    gate. Each one must raise ``UnsafeToolResultError`` — silent
    coercion is the bug class we're defending against."""

    def test_gate_rejects_bare_object(self):
        class _NotAView(object):
            email = 'jane@example.com'

        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(_NotAView())

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
        # Tuples and other iterables fall through to the single-item
        # ``isinstance`` check and are refused — locks the contract.
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
        # Someone passes the View *class* instead of an instance — the
        # class object itself is not an instance of LLMView (it IS a
        # subclass) and must be refused.
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


# ===========================================================================
# GATE — acceptance probes
# ===========================================================================


class TestGateAccepts(unittest.TestCase):
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
        self.assertEqual(payload, [
            {'id': 'u1', 'display_name': 'Jane'},
            {'id': 'u2', 'display_name': 'John'},
        ])


# ===========================================================================
# run_tool — edge cases
# ===========================================================================


class TestRunToolEdgeCases(unittest.TestCase):
    def test_run_tool_with_named_logger(self):
        logger = logging.getLogger('test_run_tool_named')

        def failing_tool():
            raise RuntimeError('boom')

        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(failing_tool, logger=logger)
        self.assertEqual(payload['status'], 'error')

    def test_run_tool_default_logger_when_none_supplied(self):
        # Default ``llm_core_lib.safety`` logger; assertLogs captures
        # whatever name run_tool falls back to.
        def failing_tool():
            raise RuntimeError('boom')

        with self.assertLogs('llm_core_lib.safety', level='ERROR'):
            payload = run_tool(failing_tool)
        self.assertEqual(payload['status'], 'error')

    def test_run_tool_passes_through_args_and_kwargs(self):
        captured = {}

        def tool(arg1, kw1=None):
            captured['arg1'] = arg1
            captured['kw1'] = kw1
            return _UserView(id=arg1, display_name=kw1)

        payload = run_tool(tool, 'u1', kw1='Jane')
        self.assertEqual(captured, {'arg1': 'u1', 'kw1': 'Jane'})
        self.assertEqual(payload, {'id': 'u1', 'display_name': 'Jane'})

    def test_run_tool_catches_unsafe_tool_result_error(self):
        # The gate raises inside ``run_tool``'s try; the wrapper turns
        # that into the same generic envelope, no type-name leak.
        def leaky_tool():
            return {'id': 'u1', 'email': 'jane@example.com'}

        logger = logging.getLogger('test_run_tool_unsafe')
        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(leaky_tool, logger=logger)
        self.assertEqual(payload['status'], 'error')
        # The type-name detail from the gate's exception message must
        # not appear in the LLM-bound payload.
        self.assertNotIn('dict', str(payload))
        self.assertNotIn('jane@example.com', str(payload))

    def test_run_tool_callable_without_name_uses_repr(self):
        # A bare callable object (no ``__name__``) — the log line
        # falls back to ``repr(fn)`` so the wrapper doesn't crash.
        class _NamelessCallable(object):
            def __call__(self):
                raise RuntimeError('boom')

        logger = logging.getLogger('test_run_tool_repr')
        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(_NamelessCallable(), logger=logger)
        self.assertEqual(payload['status'], 'error')


# ===========================================================================
# new_error_ref / sanitized_error_payload — shape contract
# ===========================================================================


class TestNewErrorRef(unittest.TestCase):
    def test_returns_8_char_hex(self):
        ref = new_error_ref()
        self.assertEqual(len(ref), 8)
        # Hex characters only.
        self.assertTrue(all(c in '0123456789abcdef' for c in ref))

    def test_successive_refs_are_distinct(self):
        refs = {new_error_ref() for _ in range(50)}
        # uuid4 collisions in a sample of 50 are astronomically unlikely;
        # this asserts the helper isn't accidentally constant.
        self.assertEqual(len(refs), 50)


class TestSanitizedErrorPayload(unittest.TestCase):
    def test_shape_is_status_detail_ref(self):
        payload = sanitized_error_payload('abc12345')
        self.assertEqual(set(payload.keys()), {'status', 'detail', 'ref'})

    def test_detail_string_is_fixed(self):
        # The contract: detail is a fixed string the LLM can rely on
        # to detect "request failed" without parsing.
        self.assertEqual(
            sanitized_error_payload('x')['detail'],
            'The request could not be completed.',
        )

    def test_detail_is_pii_free(self):
        # The fixed detail must contain no email / SSN / etc. — that
        # contract is what makes the generic envelope safe by
        # construction.
        detail = sanitized_error_payload('x')['detail']
        self.assertNotIn('@', detail)
        self.assertNotIn('-', detail)  # the SSN/credit-card separator


# ===========================================================================
# NO-LEAK — round-trip checks
# ===========================================================================


class TestNoLeakRoundTrip(unittest.TestCase):
    def test_no_leak_failure_payload_carries_no_arbitrary_data(self):
        # A tool that raises with arbitrary data in the message must
        # not surface any of that data in the returned payload.
        secret = 'classified-tag-AB12-secret-data-XYZ'

        def failing_tool():
            raise RuntimeError(f'failed because {secret}')

        log = logging.getLogger('test_no_leak_failure')
        with self.assertLogs(log, level='ERROR'):
            payload = run_tool(failing_tool, logger=log)
        self.assertNotIn(secret, str(payload))

    def test_no_leak_gate_error_does_not_include_type_name_in_payload(self):
        # ``UnsafeToolResultError``'s message carries the offending
        # type's name (e.g. "tool tried to return dict to the LLM").
        # The wrapper must NOT propagate that into the LLM-bound payload.
        class _LeakyORM(object):
            email = 'jane@example.com'

        def leaky_tool():
            return _LeakyORM()

        log = logging.getLogger('test_no_leak_type_name')
        with self.assertLogs(log, level='ERROR'):
            payload = run_tool(leaky_tool, logger=log)
        self.assertNotIn('_LeakyORM', str(payload))
        self.assertNotIn('jane@example.com', str(payload))

    def test_no_leak_view_to_dict_only_exposes_declared_fields(self):
        # If a subclass declares only ``id`` / ``display_name`` but the
        # instance carries an extra Python attribute (set via
        # ``self.email = ...`` in ``__init__``), ``model_dump`` should
        # return only what the subclass chooses to return — the marker
        # imposes no constraint.
        class _LooseView(RefLLMView):
            def __init__(self):
                self.id = 'u1'
                self.display_name = 'Jane'
                self.email = 'jane@example.com'  # NOT in model_dump

            def model_dump(self):
                # The subclass's allowlist is whatever this method returns.
                return {'id': self.id, 'display_name': self.display_name}

        payload = to_llm_payload(_LooseView())
        self.assertEqual(payload, {'id': 'u1', 'display_name': 'Jane'})
        self.assertNotIn('email', payload)


if __name__ == '__main__':
    unittest.main()
