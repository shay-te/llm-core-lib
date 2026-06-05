"""``audit_text`` is scan-only — never modifies the text.

The companion to :func:`to_llm_payload` for free-text outputs the
caller can't (or won't) rewrite: the model's final response, system
prompts being logged, anywhere PII detection matters but the text
itself must round-trip unchanged.

Per the workspace-wide "one TestCase per file" rule (see
``llm-core-lib``'s ``AGENTS.md``), this file owns exactly one
TestCase.
"""
from __future__ import annotations

import logging
import unittest

from llm_core_lib.safety.payload_gate import audit_text


class TestAuditTextScansWithoutModifying(unittest.TestCase):

    def test_clean_text_returns_empty_findings_and_no_log(self):
        logger = logging.getLogger('test_audit_text_clean')
        # ``self.assertNoLogs`` is Python 3.10+. Reimplemented inline so
        # the suite runs on the workspace's 3.9 baseline.
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        capture = _Capture(level=logging.WARNING)
        logger.addHandler(capture)
        try:
            findings = audit_text(
                'please ping the user once available',
                audit_logger=logger,
            )
        finally:
            logger.removeHandler(capture)
        self.assertEqual(
            [r for r in records if r.levelno >= logging.WARNING], [],
        )
        self.assertEqual(findings, [])

    def test_pii_text_returns_findings_and_audit_logs(self):
        logger = logging.getLogger('test_audit_text_pii')
        with self.assertLogs(logger, level='WARNING') as captured:
            findings = audit_text(
                'response said: jane@example.com is reachable',
                audit_logger=logger,
                context='admin main_chat response',
            )
        self.assertTrue(findings, 'expected at least one PII finding')
        joined = ' '.join(captured.output)
        self.assertIn('admin main_chat response', joined)
        # Zero-byte preview policy: the raw value never appears.
        self.assertNotIn('jane@example.com', joined)

    def test_findings_list_contains_pattern_name(self):
        findings = audit_text('reach jane@example.com')
        pattern_names = {finding.pattern_name for finding in findings}
        self.assertIn('email', pattern_names)

    def test_empty_text_returns_empty_findings(self):
        self.assertEqual(audit_text(''), [])

    def test_audit_text_does_not_modify_the_input(self):
        # The function returns findings, not a scrubbed string. The
        # caller's original string survives the call unmodified —
        # ``audit_text`` is for the response surface where rewriting
        # would change what the user sees.
        original = 'reach jane@example.com'
        audit_text(original)
        self.assertEqual(original, 'reach jane@example.com')


if __name__ == '__main__':
    unittest.main()
