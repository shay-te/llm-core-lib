"""``scrub_messages_for_persistence`` — pre-persistence PII scrub.

The chat loop returns ``(response, messages)`` and the caller is
expected to persist ``messages`` so the next turn has the
conversation history. The messages list accumulates user commands
(unscrubbed by design — admins type real PII for lookups),
tool-result outputs, and provider-specific assistant items. Storing
the list verbatim leaks every prior turn's PII into the host app's
DB / session store.

This helper walks the structure through :meth:`PiiService.scrub` so
any user-typed PII is replaced with ``[redacted:<pattern>]``
placeholders before persistence. The original is never mutated.

Per the workspace-wide "one TestCase per file" rule (see this repo's
``AGENTS.md``), this file owns exactly one TestCase.
"""
from __future__ import annotations

import unittest

from llm_core_lib.safety.payload_gate import scrub_messages_for_persistence


class TestScrubMessagesForPersistence(unittest.TestCase):

    def test_email_in_user_command_is_redacted(self):
        messages = [
            {'role': 'user', 'content': 'find user jane@example.com'},
            {'role': 'assistant', 'content': 'I will look them up.'},
        ]
        scrubbed = scrub_messages_for_persistence(messages)
        self.assertNotIn('jane@example.com', scrubbed[0]['content'])
        self.assertIn('[redacted:email', scrubbed[0]['content'])

    def test_clean_history_passes_through_structurally_equal(self):
        messages = [
            {'role': 'user', 'content': 'list packages'},
            {'role': 'assistant', 'content': 'You have 3 packages.'},
        ]
        scrubbed = scrub_messages_for_persistence(messages)
        self.assertEqual(scrubbed, messages)

    def test_input_messages_are_not_mutated(self):
        original = [{'role': 'user', 'content': 'reach jane@example.com'}]
        snapshot = [dict(message) for message in original]
        scrub_messages_for_persistence(original)
        # The original list still carries the raw value — scrub
        # returns a new structure, never rewrites in place.
        self.assertEqual(original, snapshot)
        self.assertEqual(original[0]['content'], 'reach jane@example.com')

    def test_empty_messages_returns_empty(self):
        self.assertEqual(scrub_messages_for_persistence([]), [])

    def test_nested_bedrock_message_shape_is_walked(self):
        # Bedrock messages have nested ``content`` block lists. The
        # scrubber walks dicts / lists recursively, so PII inside the
        # nested text block is redacted just like a flat string.
        messages = [{
            'role': 'user',
            'content': [{'text': 'reach jane@example.com'}],
        }]
        scrubbed = scrub_messages_for_persistence(messages)
        self.assertNotIn(
            'jane@example.com',
            scrubbed[0]['content'][0]['text'],
        )


if __name__ == '__main__':
    unittest.main()
