"""``scrub_history_for_llm`` — strip PII from PRIOR turns before resending to the LLM.

The chat loop sends the full ``input_messages`` list on every turn
(stateless LLM APIs). Without this scrub, every prior turn's PII
gets re-transmitted to the LLM provider on every subsequent turn,
forever, multiplying provider-side exposure. This helper redacts
all but the LAST message in the list — the current turn's user
command stays raw (admins legitimately type real emails for
lookups); everything older is scrubbed.

Per the workspace-wide "one TestCase per file" rule (see this repo's
``AGENTS.md``), this file owns exactly one TestCase.
"""
from __future__ import annotations

import unittest

from llm_core_lib.safety.payload_gate import scrub_history_for_llm


class TestScrubHistoryForLlm(unittest.TestCase):

    def test_last_message_is_preserved_verbatim(self):
        # The current turn's user command needs to reach the LLM in
        # original form — the admin typed "find jane@example.com"
        # because they actually want to look up jane@example.com.
        messages = [
            {'role': 'user', 'content': 'prior: ping bob@example.com'},
            {'role': 'assistant', 'content': 'will do'},
            {'role': 'user', 'content': 'now: find alice@example.com'},
        ]
        scrubbed = scrub_history_for_llm(messages)
        # Last message untouched.
        self.assertEqual(scrubbed[-1]['content'], 'now: find alice@example.com')

    def test_prior_user_messages_get_pii_redacted(self):
        messages = [
            {'role': 'user', 'content': 'first turn: ping bob@example.com'},
            {'role': 'user', 'content': 'second turn — current'},
        ]
        scrubbed = scrub_history_for_llm(messages)
        self.assertNotIn('bob@example.com', scrubbed[0]['content'])
        self.assertIn('[redacted:email', scrubbed[0]['content'])

    def test_prior_assistant_messages_get_pii_redacted(self):
        # The model may have echoed PII in an earlier reply. That
        # text re-flows to the LLM on every turn unless we scrub.
        messages = [
            {'role': 'user', 'content': 'find user 42'},
            {'role': 'assistant', 'content': 'Found Jane Doe at jane@example.com'},
            {'role': 'user', 'content': 'now what?'},
        ]
        scrubbed = scrub_history_for_llm(messages)
        self.assertNotIn('jane@example.com', scrubbed[1]['content'])

    def test_function_call_output_strings_get_scrubbed(self):
        # OpenAI Responses appends ``function_call_output`` items
        # whose ``output`` is a string built from the tool result.
        # Gate scrubbed the result on the way in; we re-scrub the
        # string on the way back to defend against any drift.
        messages = [
            {'role': 'user', 'content': 'lookup'},
            {
                'type': 'function_call_output',
                'call_id': 'c1',
                'output': '<TOOL_DATA>{"email": "alice@example.com"}</TOOL_DATA>',
            },
            {'role': 'user', 'content': 'current'},
        ]
        scrubbed = scrub_history_for_llm(messages)
        self.assertNotIn('alice@example.com', scrubbed[1]['output'])
        # call_id stays so the LLM can correlate calls with results.
        self.assertEqual(scrubbed[1]['call_id'], 'c1')

    def test_bedrock_tool_result_blocks_get_scrubbed(self):
        # Bedrock Converse wraps tool results inside the messages
        # ``content`` list under a ``toolResult`` block. The scrubber
        # walks dicts recursively, so PII inside the nested text is
        # redacted just like a flat string.
        messages = [
            {'role': 'user', 'content': [{'text': 'lookup'}]},
            {
                'role': 'user',
                'content': [{
                    'toolResult': {
                        'toolUseId': 'tu_1',
                        'content': [{'text': 'email: bob@example.com'}],
                    },
                }],
            },
            {'role': 'user', 'content': [{'text': 'current'}]},
        ]
        scrubbed = scrub_history_for_llm(messages)
        result_block = scrubbed[1]['content'][0]['toolResult']
        self.assertEqual(result_block['toolUseId'], 'tu_1')
        self.assertNotIn(
            'bob@example.com',
            result_block['content'][0]['text'],
        )

    def test_empty_history_returns_empty(self):
        self.assertEqual(scrub_history_for_llm([]), [])

    def test_single_message_passes_through_unchanged(self):
        # Single message IS the last message — nothing prior to scrub.
        messages = [{'role': 'user', 'content': 'find jane@example.com'}]
        scrubbed = scrub_history_for_llm(messages)
        self.assertEqual(scrubbed, messages)
        # And the raw email survives so the LLM can act on the lookup.
        self.assertIn('jane@example.com', scrubbed[0]['content'])

    def test_input_list_is_not_mutated(self):
        # Defensive — callers may still hold a reference to the
        # original messages for in-memory UI rendering.
        original = [
            {'role': 'user', 'content': 'prior: jane@example.com'},
            {'role': 'user', 'content': 'current'},
        ]
        snapshot_first = dict(original[0])
        scrub_history_for_llm(original)
        self.assertEqual(original[0], snapshot_first)


if __name__ == '__main__':
    unittest.main()
