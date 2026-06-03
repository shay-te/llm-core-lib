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

import logging
import unittest
from dataclasses import dataclass

from llm_core_lib.safety.llm_view import LLMView
from llm_core_lib.safety.payload_gate import (
    UnsafeToolResultError,
    run_tool,
    sanitized_error_payload,
    to_llm_payload,
)
from llm_core_lib.safety.pii_scanner import assert_no_pii


@dataclass(frozen=True)
class _UserLLMView(LLMView):
    id: str
    display_name: str


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
        assert_no_pii(payload)


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
        assert_no_pii(payload)

    def test_error_payload_is_assert_no_pii_safe(self):
        def banned_user_tool():
            raise PermissionError(
                'access denied for jane@example.com (123-45-6789)'
            )

        logger = logging.getLogger('test_run_tool_error_pii')
        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(banned_user_tool, logger=logger)

        assert_no_pii(payload)
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

        assert_no_pii(payload)
        self.assertEqual(payload['status'], 'error')

    def test_correlation_ref_is_stable_length(self):
        def failing_tool():
            raise RuntimeError('x')

        logger = logging.getLogger('test_run_tool_ref')
        with self.assertLogs(logger, level='ERROR'):
            payload = run_tool(failing_tool, logger=logger)
        # 8-char hex slice from uuid4 — locks the operator-facing format.
        self.assertEqual(len(payload['ref']), 8)


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
