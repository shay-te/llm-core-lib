"""Tests for :mod:`llm_core_lib.safety.pii_scanner`.

The scanner is the runtime backstop — the structural defense is the
allowlist, but if an allowlisted string field carries raw user PII
(e.g. a comment that quoted an email address) the scanner must catch
it so the test suite and the live ``run_tool`` failure path both have
something to lean on.
"""
from __future__ import annotations

import unittest

from llm_core_lib.safety.pii_patterns import PII_PATTERN_NAMES
from llm_core_lib.safety.pii_scanner import (
    PIIDetectedError,
    assert_no_pii,
    find_pii,
    scrub_pii,
)


class TestFindPii(unittest.TestCase):
    def test_clean_payload_returns_no_findings(self):
        payload = {'id': 'u1', 'display_name': 'Jane', 'rank': 5}
        self.assertEqual(find_pii(payload), [])

    def test_email_is_detected(self):
        findings = find_pii({'note': 'reach out to jane@example.com'})
        self.assertEqual([finding.pattern_name for finding in findings], ['email'])

    def test_ssn_is_detected(self):
        findings = find_pii({'ref': 'SSN 123-45-6789 on file'})
        self.assertIn('ssn', [finding.pattern_name for finding in findings])

    def test_credit_card_is_detected(self):
        findings = find_pii({'card': '4242 4242 4242 4242'})
        self.assertIn('credit_card', [finding.pattern_name for finding in findings])

    def test_nested_structures_are_walked(self):
        payload = {
            'users': [
                {'id': 'u1', 'note': 'see jane@example.com'},
                {'id': 'u2', 'note': 'clean'},
            ]
        }
        findings = find_pii(payload)
        self.assertEqual([finding.pattern_name for finding in findings], ['email'])

    def test_none_payload_is_no_findings(self):
        self.assertEqual(find_pii(None), [])

    def test_redacted_preview_never_contains_full_value(self):
        # The raw email must never appear in a finding — only the safe
        # prefix + length marker.
        findings = find_pii({'note': 'jane@example.com'})
        self.assertEqual(len(findings), 1)
        self.assertNotIn('jane@example.com', findings[0].redacted_preview)
        self.assertIn('REDACTED', findings[0].redacted_preview)


class TestAssertNoPii(unittest.TestCase):
    def test_clean_payload_does_not_raise(self):
        assert_no_pii({'id': 'u1', 'display_name': 'Jane'})

    def test_email_payload_raises_pii_detected(self):
        with self.assertRaises(PIIDetectedError) as ctx:
            assert_no_pii({'note': 'jane@example.com'})
        # The error message must not echo the raw email back.
        self.assertNotIn('jane@example.com', str(ctx.exception))
        self.assertIn('email', str(ctx.exception))

    def test_assert_lists_every_matched_pattern(self):
        with self.assertRaises(PIIDetectedError) as ctx:
            assert_no_pii({
                'a': 'jane@example.com',
                'b': '123-45-6789',
            })
        message = str(ctx.exception)
        self.assertIn('email', message)
        self.assertIn('ssn', message)


class TestScrubPii(unittest.TestCase):
    def test_email_in_string_is_replaced(self):
        scrubbed = scrub_pii('reach out to jane@example.com please')
        self.assertNotIn('jane@example.com', scrubbed)
        self.assertIn('[REDACTED:email]', scrubbed)

    def test_dict_is_walked_recursively(self):
        payload = {
            'id': 'u1',
            'note': 'email is jane@example.com',
            'nested': {'card': '4242 4242 4242 4242'},
        }
        scrubbed = scrub_pii(payload)
        self.assertEqual(scrubbed['id'], 'u1')
        self.assertNotIn('jane@example.com', scrubbed['note'])
        self.assertNotIn('4242 4242 4242 4242', scrubbed['nested']['card'])

    def test_list_is_walked(self):
        scrubbed = scrub_pii(['jane@example.com', 'clean'])
        self.assertNotIn('jane@example.com', scrubbed[0])
        self.assertEqual(scrubbed[1], 'clean')

    def test_non_text_primitives_pass_through(self):
        # Numbers, bools, None carry no text PII; scrubber must not coerce them.
        self.assertEqual(scrub_pii(42), 42)
        self.assertEqual(scrub_pii(True), True)
        self.assertIsNone(scrub_pii(None))


class TestPatternNames(unittest.TestCase):
    def test_named_set_matches_expected_categories(self):
        # The task description calls out email, ssn, phone, credit_card,
        # and an "address / billing" intent — billing-shaped data lands
        # under credit_card / iban. Lock the contract.
        self.assertEqual(
            PII_PATTERN_NAMES,
            frozenset({'email', 'ssn', 'phone', 'credit_card', 'iban'}),
        )


if __name__ == '__main__':
    unittest.main()
