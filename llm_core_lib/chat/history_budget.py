"""Token-budget truncator for chat history.

Drops oldest messages until the list fits ``max_tokens``.
Conservative 4-chars-per-token heuristic (no tiktoken dep) — fine
because callers leave completion headroom in the budget.

The last message is always kept (current user prompt); truncation
drops oldest first (never reorders).
"""
from __future__ import annotations

import json
from typing import Any, List


_CHARS_PER_TOKEN = 4


def estimate_tokens(message: Any) -> int:
    if message is None:
        return 0
    try:
        text = json.dumps(message, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return 8
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def truncate_to_token_budget(
    messages: List[Any], max_tokens: int,
) -> List[Any]:
    """Drop oldest messages until the list fits ``max_tokens``.

    ``max_tokens <= 0`` disables truncation. Returns a new list;
    input is never mutated. The last message is preserved even if
    it alone exceeds the budget — the current user prompt must
    reach the LLM regardless of history cost.
    """
    if not messages or max_tokens is None or max_tokens <= 0:
        return list(messages)
    tail = messages[-1]
    kept_reversed = [tail]
    running = estimate_tokens(tail)
    for message in reversed(messages[:-1]):
        cost = estimate_tokens(message)
        if running + cost > max_tokens:
            break
        kept_reversed.append(message)
        running += cost
    kept_reversed.reverse()
    return kept_reversed
