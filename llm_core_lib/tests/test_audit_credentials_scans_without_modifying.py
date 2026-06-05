"""``audit_credentials`` is scan-only — never modifies the text.

Companion to :func:`audit_text` (PII) for the credential / phishing
detector set. Same call shape, same audit-only contract — the text
itself is never rewritten. Host apps call both on the same string
to get full sensitive-data coverage (PII + secrets) through one
layer (the llm-core-lib safety gate) so admin-backend stays fully
credential- and PII-blind.

Per the workspace-wide "one TestCase per file" rule (see this repo's
``AGENTS.md``), this file owns exactly one TestCase.
"""
from __future__ import annotations

import logging
import unittest

from llm_core_lib.safety.payload_gate import audit_credentials


class TestAuditCredentialsScansWithoutModifying(unittest.TestCase):

    def test_clean_text_does_not_log(self):
        logger = logging.getLogger('test_audit_credentials_clean')
        with self.assertNoLogs(logger, level='WARNING'):
            audit_credentials(
                'please ping the user once available',
                audit_logger=logger,
            )

    def test_credential_in_text_logs_warning_with_context(self):
        logger = logging.getLogger('test_audit_credentials_aws')
        with self.assertLogs(logger, level='WARNING') as captured:
            audit_credentials(
                # An AWS-shaped access key id — the detector is the
                # one in pii_core_lib.credential_patterns, real call.
                'response said: AKIAIOSFODNN7EXAMPLE is your key',
                audit_logger=logger,
                context='admin main_chat response',
            )
        joined = ' '.join(captured.output)
        self.assertIn('admin main_chat response', joined)
        # Zero-byte preview policy: the raw key never appears in logs.
        self.assertNotIn('AKIAIOSFODNN7EXAMPLE', joined)

    def test_empty_text_does_not_crash(self):
        # Audit-only — returns None either way. No log fired.
        logger = logging.getLogger('test_audit_credentials_empty')
        with self.assertNoLogs(logger, level='WARNING'):
            audit_credentials('', audit_logger=logger)

    def test_does_not_modify_the_input(self):
        # The function is detective-only — the caller's string survives
        # the call unmodified. ``audit_credentials`` is for the
        # response surface where rewriting would change what the user sees.
        original = 'AKIAIOSFODNN7EXAMPLE is your key'
        audit_credentials(original)
        self.assertEqual(original, 'AKIAIOSFODNN7EXAMPLE is your key')

    def test_returns_none(self):
        # Unlike ``audit_text`` (which returns findings for advanced
        # consumers), ``audit_credentials`` is fire-and-forget. The
        # underlying scanner logs at WARNING level and returns nothing.
        result = audit_credentials('clean text', audit_logger=logging.getLogger('x'))
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
