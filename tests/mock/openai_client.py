"""Mock ``openai.OpenAI`` client for the connection-factory tests.

Mirrors the *exact* attribute surface :class:`OpenAiConnection` touches
— ``chat.completions.create`` + ``embeddings.create`` — and records
every call into ``chat_calls`` / ``embedding_calls`` for assertions.
Returned objects use ``types.SimpleNamespace`` so attribute access
matches the real SDK response shapes.

Passed into the factory via the ``client`` config key:
``OpenAiConnectionFactory({'model': ..., 'client': MockOpenAIClient()})``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class _MockOpenAICompletions(object):
    def __init__(self, parent: 'MockOpenAIClient'):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._respond_chat(kwargs)


class _MockOpenAIChat(object):
    def __init__(self, parent: 'MockOpenAIClient'):
        self.completions = _MockOpenAICompletions(parent)


class _MockOpenAIEmbeddings(object):
    def __init__(self, parent: 'MockOpenAIClient'):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._respond_embedding(kwargs)


class _MockOpenAIResponses(object):
    """Mocks ``client.responses.create`` (the Responses API the
    Connection's ``chat_with_tools`` drives for OpenAI function calling).

    The factory takes a list of pre-canned responses — one per call —
    so a test can script a multi-round tool-call dance: round 1
    returns a ``function_call`` item, round 2 returns a terminal
    ``message`` item, etc.
    """

    def __init__(self, parent: 'MockOpenAIClient'):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._respond_responses(kwargs)


class MockOpenAIClient(object):
    """Mocks ``openai.OpenAI`` for the subset the connection uses:
    ``chat.completions.create`` (Chat Completions API — drives
    ``complete_text``), ``embeddings.create`` (drives ``embed``), and
    ``responses.create`` (Responses API — drives ``chat_with_tools``)."""

    def __init__(
        self,
        content: str = 'hello',
        usage: Optional[Dict[str, int]] = None,
        model: str = 'gpt-4o-mini',
        raise_exc: Optional[BaseException] = None,
        omit_usage: bool = False,
        embedding: Optional[List[float]] = None,
        responses_script: Optional[List[Any]] = None,
    ):
        self.chat = _MockOpenAIChat(self)
        self.embeddings = _MockOpenAIEmbeddings(self)
        self.responses = _MockOpenAIResponses(self)
        self.chat_calls: List[Dict[str, Any]] = []
        self.embedding_calls: List[Dict[str, Any]] = []
        self.responses_calls: List[Dict[str, Any]] = []
        self._content = content
        self._usage = usage or {'prompt': 5, 'completion': 7, 'total': 12}
        self._model = model
        self._raise = raise_exc
        self._omit_usage = omit_usage
        self._embedding = embedding if embedding is not None else [0.1, 0.2, 0.3]
        # Default script: one terminal message (no tool calls). Tests
        # that exercise tool calling pass a longer script.
        self._responses_script = list(responses_script) if responses_script is not None else None
        self._responses_index = 0

    def _respond_chat(self, kwargs: Dict[str, Any]):
        self.chat_calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        usage_obj = None
        if not self._omit_usage:
            usage_obj = SimpleNamespace(
                prompt_tokens=self._usage['prompt'],
                completion_tokens=self._usage['completion'],
                total_tokens=self._usage['total'],
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content),
                    finish_reason='stop',
                ),
            ],
            model=self._model,
            usage=usage_obj,
        )

    def _respond_embedding(self, kwargs: Dict[str, Any]):
        self.embedding_calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=list(self._embedding))],
        )

    def _respond_responses(self, kwargs: Dict[str, Any]):
        self.responses_calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        if self._responses_script is None:
            # Default: a single terminal message; ``chat_with_tools``
            # returns immediately without calling any tool.
            return _make_openai_message_response(self._content, self._model)
        if self._responses_index >= len(self._responses_script):
            raise AssertionError(
                'MockOpenAIClient responses_script exhausted; '
                'add more entries or shorten the test loop.'
            )
        scripted = self._responses_script[self._responses_index]
        self._responses_index += 1
        return scripted


def make_openai_function_call_response(
    *,
    name: str,
    arguments: str = '{}',
    call_id: str = 'call_1',
    model: str = 'gpt-4o-mini',
):
    """Build a Responses-API response whose output is a function_call.

    Public so test files can script a multi-round tool-call dance.
    """
    return SimpleNamespace(
        error=None,
        model=model,
        output=[
            SimpleNamespace(
                type='function_call',
                name=name,
                arguments=arguments,
                call_id=call_id,
            ),
        ],
    )


def _make_openai_message_response(text: str, model: str):
    return SimpleNamespace(
        error=None,
        model=model,
        output=[
            SimpleNamespace(
                type='message',
                content=[SimpleNamespace(text=text)],
            ),
        ],
    )


def make_openai_message_response(text: str = 'final', model: str = 'gpt-4o-mini'):
    """Build a Responses-API response whose output is a terminal message."""
    return _make_openai_message_response(text, model)


def make_openai_error_response(message: str = 'rate limited', model: str = 'gpt-4o-mini'):
    """Build a Responses-API response that carries an error."""
    return SimpleNamespace(
        error=SimpleNamespace(message=message),
        model=model,
        output=[],
    )
