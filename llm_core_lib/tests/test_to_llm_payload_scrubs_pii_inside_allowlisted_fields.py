"""``to_llm_payload`` scrubs PII inside allowlisted free-text fields.

The gate does TWO things, in order:

  1. Projects an :class:`LLMView` through its declared field allowlist
     (covered separately in ``test_safety_payload_gate.py``).
  2. Scrubs the projected payload via the module-level
     :class:`PiiService` so any PII that ended up inside an
     allowlisted free-text field (a ``comment``, a ``note``) is
     rewritten to ``[redacted:<pattern>]`` before the LLM sees it.

This file owns the lock for the SECOND behaviour. It exists because
admin-backend is PII-blind by design — every host that wires
``llm-core-lib``'s gate inherits the same scrubbing for free, and
that promise needs a test at the layer it actually lives.

Per the workspace-wide "one TestCase per file" rule (see
``llm-core-lib``'s ``AGENTS.md``), this file owns exactly one
TestCase. Shared fixtures live in ``gate_pii_helpers``.
"""
from __future__ import annotations

import logging
import unittest
from unittest import mock

from llm_core_lib.safety.payload_gate import to_llm_payload

from llm_core_lib.tests.gate_pii_helpers import CommentStubLLMView


class TestToLlmPayloadScrubsPiiInsideAllowlistedFields(unittest.TestCase):

    def test_email_inside_comment_is_redacted(self):
        result = to_llm_payload(CommentStubLLMView(
            id='c1',
            comment='follow up with jane@example.com next week',
        ))
        self.assertEqual(result['id'], 'c1')
        self.assertNotIn('jane@example.com', result['comment'])
        self.assertIn('[redacted:email', result['comment'])

    def test_clean_comment_passes_through_unchanged(self):
        # No PII → scrub returns the same content; the projected dict
        # is structurally identical to what bare projection produced.
        result = to_llm_payload(CommentStubLLMView(
            id='c2', comment='please ping the user once available',
        ))
        self.assertEqual(result, {
            'id': 'c2',
            'comment': 'please ping the user once available',
        })

    def test_list_of_views_each_scrubbed_independently(self):
        result = to_llm_payload([
            CommentStubLLMView(id='c1', comment='clean'),
            CommentStubLLMView(id='c2', comment='ping jane@example.com'),
        ])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['comment'], 'clean')
        self.assertNotIn('jane@example.com', result[1]['comment'])

    def test_pii_detection_fires_an_audit_log_warning_through_caller_logger(self):
        # When the caller passes ``audit_logger``, the PII scrub's
        # WARNING line lands on that logger so operators can route
        # detections per call site.
        logger = logging.getLogger('test_gate_audit')
        with self.assertLogs(logger, level='WARNING') as captured:
            to_llm_payload(
                CommentStubLLMView(id='c3', comment='reach jane@example.com'),
                audit_logger=logger,
                context='tool result: comment',
            )
        joined = ' '.join(captured.output)
        self.assertIn('PII detected in tool result: comment', joined)
        # And the raw value never appears in the audit line — zero-byte
        # preview policy is inherited from PiiService.
        self.assertNotIn('jane@example.com', joined)

    def test_scrub_crash_propagates_so_run_tool_can_sanitize(self):
        # If the underlying PiiService raises, ``to_llm_payload`` does
        # NOT swallow it — that's ``run_tool``'s job. The raw exception
        # must reach the caller's try/except so the standard sanitized
        # envelope is emitted (and not silently a half-scrubbed payload).
        class _BoomPii(object):
            def scrub(self, payload, *, strict=False, raise_on_pii=False,
                      audit_logger=None, context='payload'):
                raise RuntimeError('simulated scrub crash')

        from llm_core_lib.safety import payload_gate

        with mock.patch.object(payload_gate, '_PII_SERVICE', _BoomPii()):
            with self.assertRaises(RuntimeError):
                to_llm_payload(CommentStubLLMView(id='c4', comment='x'))


if __name__ == '__main__':
    unittest.main()
