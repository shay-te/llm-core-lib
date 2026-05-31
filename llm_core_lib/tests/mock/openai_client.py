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


class MockOpenAIClient(object):
    """Mocks ``openai.OpenAI`` for the subset the connection uses:
    ``chat.completions.create`` + ``embeddings.create``."""

    def __init__(
        self,
        content: str = 'hello',
        usage: Optional[Dict[str, int]] = None,
        model: str = 'gpt-4o-mini',
        raise_exc: Optional[BaseException] = None,
        omit_usage: bool = False,
        embedding: Optional[List[float]] = None,
    ):
        self.chat = _MockOpenAIChat(self)
        self.embeddings = _MockOpenAIEmbeddings(self)
        self.chat_calls: List[Dict[str, Any]] = []
        self.embedding_calls: List[Dict[str, Any]] = []
        self._content = content
        self._usage = usage or {'prompt': 5, 'completion': 7, 'total': 12}
        self._model = model
        self._raise = raise_exc
        self._omit_usage = omit_usage
        self._embedding = embedding if embedding is not None else [0.1, 0.2, 0.3]

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
