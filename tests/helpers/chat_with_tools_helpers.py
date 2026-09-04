"""Shared fixtures for the ``test_*_chat_with_tools_*`` test files.

The helpers here are real-collaborator-only — no ``mock.Mock``. The
SDK client is the one legitimate mock surface per this repo's
``AGENTS.md`` and lives in ``tests/mock/``; everything else (the
``invoke_tool`` callback, the tools list, the connection itself) is
a real Python value the test wires together.

This module has no ``test_`` prefix on purpose; the discoverers skip it.
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple


class RecordingInvokeTool(object):
    """Real callable that records every ``(name, kwargs)`` it was
    called with and returns a scripted result.

    Used as the ``invoke_tool`` argument to ``chat_with_tools`` so the
    test can assert on what the connection asked the host to execute
    AND return a deterministic result the connection feeds back to
    the model. Plain Python — no ``mock.Mock.return_value`` setup.
    """

    def __init__(self, results=None):
        self._results = list(results) if results is not None else ['TOOL_RESULT']
        self._index = 0
        self.calls: List[Tuple[str, dict]] = []

    def __call__(self, name: str, kwargs: dict) -> Any:
        self.calls.append((name, kwargs))
        if self._index >= len(self._results):
            return self._results[-1] if self._results else None
        result = self._results[self._index]
        self._index += 1
        return result


def sample_openai_tools() -> list:
    """A small OpenAI function-tool schema sufficient to exercise the
    tool-call loop. The Connection forwards this to the OpenAI
    Responses API as-is and (for Bedrock) converts it internally."""
    return [
        {
            'name': 'list_packages',
            'description': 'List packages for the user.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'integer'},
                },
                'required': ['user_id'],
            },
        },
    ]
