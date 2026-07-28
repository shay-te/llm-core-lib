"""``truncate_to_token_budget`` — drop oldest messages until fit.

The runtime safety net behind ``max_chat_history``: a
message-count cap can still let a small number of huge messages
blow past the LLM's context window. This truncator runs after the
fetch and trims the input list to a token budget.

Two invariants the truncator MUST preserve, locked here:

  1. The LAST message survives unconditionally — it's the current
     user prompt; truncating it defeats the turn.
  2. Older messages drop first; we never reorder by size (the LLM
     follows conversation order).

Per the workspace-wide "one TestCase per file" rule, this file owns
exactly one TestCase.
"""
from __future__ import annotations

import unittest

from llm_core_lib.chat.history_budget import (
    estimate_tokens,
    truncate_to_token_budget,
)


class TestHistoryBudget(unittest.TestCase):

    def test_estimate_tokens_uses_4_chars_per_token(self):
        # A 40-char JSON body should clock in around 10 tokens.
        # We allow some slack for JSON quote/key overhead — the
        # important property is "scales with content size".
        small = estimate_tokens({'role': 'user', 'content': 'hi'})
        large = estimate_tokens({'role': 'user', 'content': 'x' * 400})
        self.assertGreater(large, small * 10)

    def test_estimate_tokens_returns_zero_for_none(self):
        self.assertEqual(estimate_tokens(None), 0)

    def test_estimate_tokens_returns_nonzero_floor_for_tiny_payload(self):
        # Even a 1-char message costs at least 1 token so the budget
        # math always makes progress.
        self.assertGreaterEqual(estimate_tokens({'a': 'b'}), 1)

    def test_returns_empty_list_unchanged(self):
        self.assertEqual(truncate_to_token_budget([], 1000), [])

    def test_zero_or_negative_budget_disables_truncation(self):
        # Opt-out without branching at call site.
        messages = [{'role': 'user', 'content': 'x' * 1000}] * 5
        self.assertEqual(len(truncate_to_token_budget(messages, 0)), 5)
        self.assertEqual(len(truncate_to_token_budget(messages, -1)), 5)

    def test_short_list_under_budget_passes_through(self):
        messages = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
            {'role': 'user', 'content': 'how are you'},
        ]
        out = truncate_to_token_budget(messages, max_tokens=1000)
        self.assertEqual(out, messages)

    def test_drops_oldest_first(self):
        # 3 messages, ~50 tokens each in this content. Set a budget
        # of ~100 so we expect the last 2 (most-recent) to survive.
        messages = [
            {'role': 'user', 'content': 'A' * 200},      # ~50 tokens
            {'role': 'assistant', 'content': 'B' * 200},  # ~50 tokens
            {'role': 'user', 'content': 'C' * 200},      # ~50 tokens
        ]
        out = truncate_to_token_budget(messages, max_tokens=120)
        # Last message is always kept. The next-oldest drops.
        self.assertEqual(out[-1]['content'][0], 'C')
        # The first message is gone, second may be in or out depending
        # on exact byte math — but we never see the oldest.
        contents = [m['content'][0] for m in out]
        self.assertNotIn('A', contents)

    def test_always_keeps_last_message_even_when_over_budget(self):
        # The current user prompt MUST reach the LLM even if it
        # alone exceeds the budget; truncating it would be worse
        # than going over. Operators see provider rejections, not
        # silent drops of the user's input.
        huge_prompt = {'role': 'user', 'content': 'x' * 10000}
        out = truncate_to_token_budget([huge_prompt], max_tokens=10)
        self.assertEqual(out, [huge_prompt])

    def test_does_not_mutate_input_list(self):
        messages = [
            {'role': 'user', 'content': 'A' * 200},
            {'role': 'user', 'content': 'B' * 200},
        ]
        before = list(messages)
        truncate_to_token_budget(messages, max_tokens=10)
        self.assertEqual(messages, before)

    def test_does_not_reorder_kept_messages(self):
        # Ordering matters — the LLM follows the conversation in
        # input-list order. The truncator must not pick "smallest
        # first" or otherwise reorder; it only drops a prefix.
        messages = [
            {'role': 'user', 'content': 'A' * 200},      # ~50 tokens — gets dropped
            {'role': 'assistant', 'content': 'B' * 8},   # tiny
            {'role': 'user', 'content': 'C' * 200},      # ~50 tokens
        ]
        out = truncate_to_token_budget(messages, max_tokens=80)
        # The tiny 'B' message survives because it's adjacent to the
        # kept tail, and survives in the same position relative to
        # the rest — even though dropping the huge 'A' lets us keep
        # 'B', we don't promote 'B' ahead of 'C'.
        self.assertEqual(out[-1]['content'][0], 'C')
        self.assertEqual([m['content'][0] for m in out], ['B', 'C'])


if __name__ == '__main__':
    unittest.main()
