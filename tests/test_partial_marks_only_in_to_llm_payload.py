"""Only ``to_llm_payload`` asks pii-core-lib for partial marks: the matchmaker reads that data through the AI and keeps
a clue (an email's host). Scrubbed history, saved messages and error envelopes stay full marks.

Per the workspace-wide "one TestCase per file" rule (see this repo's ``AGENTS.md``), this file owns exactly one
TestCase. Shared fixtures live in ``gate_pii_helpers``."""
from __future__ import annotations

import unittest

from llm_core_lib.safety.payload_gate import scrub_history_for_llm, scrub_messages_for_persistence, \
    scrub_user_error_payload, to_llm_payload

from tests.helpers.gate_pii_helpers import CommentStubLLMView

TEXT = 'ping jane@example.com'
FULL_MARK = 'ping [redacted:email]'
PARTIAL_MARK = 'ping [redacted:email:host=example.com]'


class TestPartialMarksOnlyInToLlmPayload(unittest.TestCase):

    def test_a_tool_result_keeps_the_email_host(self):
        result = to_llm_payload(CommentStubLLMView(id='c1', comment=TEXT))
        self.assertEqual(PARTIAL_MARK, result['comment'])

    def test_scrubbed_history_is_full(self):
        scrubbed = scrub_history_for_llm([{'role': 'user', 'content': TEXT}, {'role': 'user', 'content': 'now'}])
        self.assertEqual(FULL_MARK, scrubbed[0]['content'])

    def test_saved_messages_are_full(self):
        scrubbed = scrub_messages_for_persistence([{'role': 'user', 'content': TEXT}])
        self.assertEqual(FULL_MARK, scrubbed[0]['content'])

    def test_a_user_error_is_full(self):
        self.assertEqual(FULL_MARK, scrub_user_error_payload(TEXT)['detail'])


if __name__ == '__main__':
    unittest.main()
