"""Tests for :mod:`llm_core_lib.safety.payload_gate`.

The gate is the "one choke point" the task description specifies — a
single function every tool result must pass through. These tests lock
the four behaviors the gate is responsible for:

1. :func:`to_llm_payload` accepts :class:`LLMView` and lists of
   :class:`LLMView`, projecting both into JSON-safe shapes.
2. :func:`to_llm_payload` refuses raw dicts / ORM-shaped objects / bare
   models — failing loud is the contract.
3. :func:`run_tool` forwards the success path through the gate.
4. :func:`run_tool` swallows exceptions, logs the full detail with a
   correlation ref, and returns the generic error envelope. The
   exception's *message* must never reach the returned payload.
"""
from __future__ import annotations

import json
import logging
import re
import unittest
from http import HTTPStatus

from core_lib.error_handling.status_code_exception import StatusCodeException

from llm_core_lib.safety.llm_view import RefLLMView
from llm_core_lib.safety.payload_gate import (
    UnsafeToolResultError,
    run_tool,
    sanitized_error_payload,
    to_llm_payload,
)


# Inline PII assertions so ``llm-core-lib`` stays free of regex-PII
# concerns (the canonical patterns live in ``agent-core-lib``
# ``helpers/pii_patterns`` per the workspace's single-source-of-truth
# rule). What this file cares about is the *contract* — exception
# messages must not survive into the LLM-bound payload. The simple
# ``@`` / dash-digit checks below are sufficient for that contract:
# the inputs we feed in carry an email and an SSN respectively, and
# the assertion is that neither leaks back out.
def _payload_text(payload):
    return json.dumps(payload, default=str)


def _assert_no_email_or_ssn(test_case, payload):
    blob = _payload_text(payload)
    test_case.assertNotIn('@', blob, f'email-shaped data leaked: {blob!r}')
    test_case.assertIsNone(
        re.search(r'\d{3}-\d{2}-\d{4}', blob),
        f'ssn-shaped data leaked: {blob!r}',
    )


class _UserLLMView(RefLLMView):
    """Stdlib subclass of the transport-layer marker — the gate's
    ``isinstance(item, LLMView)`` check passes, and ``model_dump``
    returns a JSON-safe dict. The Pydantic-flavored equivalent
    (``ConfigDict(extra='forbid', frozen=True)`` + field declarations)
    lives in ``agent_core_lib.safety.llm_view.LLMView`` and is tested
    there; this fixture only needs to exercise the gate's contract."""

    def __init__(self, id, display_name):
        self.id = id
        self.display_name = display_name

    def model_dump(self):
        return {'id': self.id, 'display_name': self.display_name}


class TestToLlmPayloadAcceptsLLMView(unittest.TestCase):
    def test_single_view_returns_dict(self):
        payload = to_llm_payload(_UserLLMView(id='u1', display_name='Jane'))
        self.assertEqual(payload, {'id': 'u1', 'display_name': 'Jane'})

    def test_list_of_views_returns_list_of_dicts(self):
        payload = to_llm_payload([
            _UserLLMView(id='u1', display_name='Jane'),
            _UserLLMView(id='u2', display_name='John'),
        ])
        self.assertEqual(payload, [
            {'id': 'u1', 'display_name': 'Jane'},
            {'id': 'u2', 'display_name': 'John'},
        ])

    def test_none_passes_through(self):
        self.assertIsNone(to_llm_payload(None))


class TestToLlmPayloadRefusesUnsafeShapes(unittest.TestCase):
    def test_raw_dict_raises(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload({'id': 'u1', 'email': 'jane@example.com'})

    def test_list_with_raw_dict_raises(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload([{'id': 'u1'}])

    def test_orm_shaped_object_raises(self):
        class _OrmLike(object):
            id = 'u1'
            email = 'jane@example.com'

        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload(_OrmLike())

    def test_mixed_list_raises_on_first_unsafe_item(self):
        with self.assertRaises(UnsafeToolResultError):
            to_llm_payload([
                _UserLLMView(id='u1', display_name='Jane'),
                {'id': 'u2'},
            ])


class TestRunToolSuccessPath(unittest.TestCase):
    def test_success_passes_through_gate(self):
        def search_users(_query):
            return [_UserLLMView(id='u1', display_name='Jane')]

        payload = run_tool(search_users, 'jane')
        self.assertEqual(payload, [{'id': 'u1', 'display_name': 'Jane'}])
        # And it's PII-free by construction — the view didn't declare email.
        _assert_no_email_or_ssn(self, payload)


class TestRunToolErrorPath(unittest.TestCase):
    """The most important tests in the file — exception strings are
    the canonical PII-leak vector the task description calls out."""

    def test_exception_message_is_not_returned_to_caller(self):
        def failing_tool():
            # The kind of message a real SDK would raise — and exactly
            # the kind we must never propagate to the model.
            raise RuntimeError('User jane@example.com not found in region EU-WEST')

        logger = logging.getLogger('test_run_tool_error')
        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(failing_tool, logger=logger)

        # Generic envelope, no leaked detail.
        self.assertEqual(payload['status'], 'error')
        self.assertEqual(payload['detail'], 'The request could not be completed.')
        self.assertIn('ref', payload)
        self.assertNotIn('jane@example.com', payload['detail'])
        self.assertNotIn('EU-WEST', payload['detail'])
        # And the assert holds for the entire payload, not just detail.
        _assert_no_email_or_ssn(self, payload)

    def test_error_payload_is_assert_no_pii_safe(self):
        def banned_user_tool():
            raise PermissionError(
                'access denied for jane@example.com (123-45-6789)'
            )

        logger = logging.getLogger('test_run_tool_error_pii')
        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(banned_user_tool, logger=logger)

        _assert_no_email_or_ssn(self, payload)
        self.assertEqual(payload['status'], 'error')

    def test_unsafe_tool_result_failure_is_also_sanitized(self):
        # A tool that returns a raw dict raises ``UnsafeToolResultError``
        # inside ``run_tool`` — and that error itself must not leak the
        # raw-dict contents back to the LLM. The same generic envelope
        # is what comes out.
        def leaky_tool():
            return {'id': 'u1', 'email': 'jane@example.com'}

        logger = logging.getLogger('test_run_tool_unsafe')
        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(leaky_tool, logger=logger)

        _assert_no_email_or_ssn(self, payload)
        self.assertEqual(payload['status'], 'error')

    def test_correlation_ref_is_stable_length(self):
        def failing_tool():
            raise RuntimeError('x')

        logger = logging.getLogger('test_run_tool_ref')
        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(failing_tool, logger=logger)
        # 8-char hex slice from uuid4 — locks the operator-facing format.
        self.assertEqual(len(payload['ref']), 8)


class TestUnsafeToolResultErrorIsStatusCodeException(unittest.TestCase):
    """The framework convention is that every core-lib exception
    inherits from ``core_lib.StatusCodeException`` so the Flask layer's
    ``@HandleException`` decorator can map it to an HTTP response. Lock
    the inheritance and the chosen status code."""

    def test_subclass_of_status_code_exception(self):
        self.assertTrue(issubclass(UnsafeToolResultError, StatusCodeException))

    def test_http_status_is_internal_server_error(self):
        # 500 because this is a server-side contract violation by a
        # tool author — no client action would fix it.
        err = UnsafeToolResultError('a tool returned a raw dict')
        # ``StatusCodeException`` exposes the chosen status via
        # ``.status_code``; tolerate both ``HTTPStatus`` and ``int``
        # forms since the constructor accepts either.
        status = getattr(err, 'status_code', None)
        self.assertIn(status, (HTTPStatus.INTERNAL_SERVER_ERROR, 500))


class TestSanitizedErrorPayload(unittest.TestCase):
    def test_shape_is_status_detail_ref(self):
        payload = sanitized_error_payload('abc12345')
        self.assertEqual(set(payload.keys()), {'status', 'detail', 'ref'})
        self.assertEqual(payload['status'], 'error')
        self.assertEqual(payload['ref'], 'abc12345')
        # The fixed detail message is the contract the LLM relies on.
        self.assertEqual(payload['detail'], 'The request could not be completed.')


if __name__ == '__main__':
    unittest.main()
