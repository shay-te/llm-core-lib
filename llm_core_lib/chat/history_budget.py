"""Token-budget truncator for chat history.

``max_chat_history`` (message-count cap) bounds how many messages
:class:`ChatSession` fetches from the store. That alone isn't safe:
50 messages whose tool outputs are each 8 KB of JSON would still
blow past the LLM's context window.

This module sits between "we loaded N messages from the store" and
"we sent them to the connection" — it drops oldest messages until
the remaining ones fit a configurable token budget.

We use a **character-based heuristic** (4 chars ≈ 1 token) instead
of importing ``tiktoken``:

  * No extra dependency for llm-core-lib (tiktoken is OpenAI-only;
    Bedrock would need a different tokenizer anyway).
  * The estimate is intentionally conservative — real tokens are
    usually 3.5–5 chars; rounding to 4 errs on the safe side for
    English text. Non-English text varies more but the budget
    should still be a soft ceiling (callers leave headroom for the
    completion).
  * A precise per-provider counter can replace ``estimate_tokens``
    without changing the truncator's interface.

Two invariants the truncator preserves:

  1. The LAST message is ALWAYS kept — it's the current-turn user
     prompt. Truncating it would defeat the whole turn.
  2. Truncation drops from the OLDEST end so the most-recent
     conversational context stays intact.
"""
from __future__ import annotations

import json
from typing import Any, List


# Conservative chars-per-token estimate. Lower than the real
# OpenAI-English average (~4.0) so we err on the side of "fewer
# tokens than reality" → more headroom in the budget.
_CHARS_PER_TOKEN = 4


def estimate_tokens(message: Any) -> int:
    """Estimate the token cost of one input-list message.

    The orchestrator passes provider-shaped dicts here. JSON-serialise
    and divide by the chars-per-token estimate. Sentinel: anything
    that fails to serialise is charged a small constant so a
    pathological row can't escape the budget.
    """
    if message is None:
        return 0
    try:
        text = json.dumps(message, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return 8
    if not text:
        return 0
    # +1 so a 1-char message still costs 1, not 0 (integer division).
    return max(1, len(text) // _CHARS_PER_TOKEN)


def truncate_to_token_budget(
    messages: List[Any], max_tokens: int,
) -> List[Any]:
    """Drop oldest messages until the list fits ``max_tokens``.

    The LAST message is always preserved — even if it alone exceeds
    the budget. This matches the chat-session invariant that the
    current-turn user prompt must reach the LLM regardless of history
    cost.

    Args:
        messages: ordered oldest-first.
        max_tokens: per-turn token budget for input messages
            combined. ``<= 0`` disables truncation (returns the list
            unchanged) so callers can opt-out without branching.

    Returns:
        A NEW list. The input is never mutated.
    """
    if not messages:
        return list(messages)
    if max_tokens <= 0:
        return list(messages)

    # Always keep the tail (current user prompt). Build the kept list
    # back-to-front to make "drop oldest first" trivial.
    tail = messages[-1]
    tail_cost = estimate_tokens(tail)
    kept_reversed = [tail]
    running = tail_cost
    for message in reversed(messages[:-1]):
        cost = estimate_tokens(message)
        if running + cost > max_tokens:
            # Stop here — any older message would push us over.
            # We don't keep scanning for a small-enough older
            # message: order matters for the LLM to follow the
            # conversation, so we can't reorder by size.
            break
        kept_reversed.append(message)
        running += cost
    kept_reversed.reverse()
    return kept_reversed
