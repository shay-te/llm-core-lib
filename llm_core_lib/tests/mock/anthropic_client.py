"""Mock ``anthropic.Anthropic`` client for the connection-factory tests.

Mirrors the ``.messages.create`` surface :class:`AnthropicConnection`
touches and records every call into ``calls`` for assertions. Passed
into the factory via the ``client`` config key:
``AnthropicConnectionFactory({'model': ..., 'client': MockAnthropicClient()})``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class _MockAnthropicMessages(object):
    def __init__(self, parent: 'MockAnthropicClient'):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._respond(kwargs)


class MockAnthropicClient(object):
    """Mocks ``anthropic.Anthropic`` for ``.messages.create``."""

    def __init__(
        self,
        content: str = 'hi from anthropic',
        usage: Optional[Dict[str, int]] = None,
        model: str = 'claude-3-5-sonnet-latest',
        raise_exc: Optional[BaseException] = None,
        omit_usage: bool = False,
        content_blocks: Optional[List[Dict[str, str]]] = None,
    ):
        self.messages = _MockAnthropicMessages(self)
        self.calls: List[Dict[str, Any]] = []
        self._content = content
        self._content_blocks = content_blocks
        self._usage = usage or {'input': 11, 'output': 4}
        self._model = model
        self._raise = raise_exc
        self._omit_usage = omit_usage

    def _respond(self, kwargs: Dict[str, Any]):
        self.calls.append(kwargs)
        if self._raise is not None:
            raise self._raise

        if self._content_blocks is not None:
            content = [
                SimpleNamespace(type=b['type'], text=b.get('text', ''))
                for b in self._content_blocks
            ]
        else:
            content = [SimpleNamespace(type='text', text=self._content)]

        usage_obj = None
        if not self._omit_usage:
            usage_obj = SimpleNamespace(
                input_tokens=self._usage['input'],
                output_tokens=self._usage['output'],
            )

        return SimpleNamespace(
            content=content,
            model=self._model,
            stop_reason='end_turn',
            usage=usage_obj,
        )
