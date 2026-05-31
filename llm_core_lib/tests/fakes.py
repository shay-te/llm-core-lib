"""Fake SDK clients for the adapter tests.

Each fake mirrors the *exact* attribute/method surface the corresponding
adapter touches — no more — and records every call into ``calls`` for
assertions. Returned objects use ``types.SimpleNamespace`` so attribute
access matches the real SDK response shapes.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


# ---- OpenAI -----------------------------------------------------------

class _FakeOpenAICompletions(object):
    def __init__(self, parent: 'FakeOpenAIClient'):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._respond(kwargs)


class _FakeOpenAIChat(object):
    def __init__(self, parent: 'FakeOpenAIClient'):
        self.completions = _FakeOpenAICompletions(parent)


class FakeOpenAIClient(object):
    """Mimics ``openai.OpenAI`` for ``.chat.completions.create``."""

    def __init__(
        self,
        content: str = 'hello',
        finish_reason: str = 'stop',
        usage: Optional[Dict[str, int]] = None,
        model: str = 'gpt-4o-mini',
        raise_exc: Optional[BaseException] = None,
        omit_usage: bool = False,
    ):
        self.chat = _FakeOpenAIChat(self)
        self.calls: List[Dict[str, Any]] = []
        self._content = content
        self._finish_reason = finish_reason
        self._usage = usage or {'prompt': 5, 'completion': 7, 'total': 12}
        self._model = model
        self._raise = raise_exc
        self._omit_usage = omit_usage

    def _respond(self, kwargs: Dict[str, Any]):
        self.calls.append(kwargs)
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
                    finish_reason=self._finish_reason,
                ),
            ],
            model=self._model,
            usage=usage_obj,
        )


# ---- Anthropic --------------------------------------------------------

class _FakeAnthropicMessages(object):
    def __init__(self, parent: 'FakeAnthropicClient'):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._respond(kwargs)


class FakeAnthropicClient(object):
    """Mimics ``anthropic.Anthropic`` for ``.messages.create``."""

    def __init__(
        self,
        content: str = 'hi from anthropic',
        finish_reason: str = 'end_turn',
        usage: Optional[Dict[str, int]] = None,
        model: str = 'claude-3-5-sonnet-latest',
        raise_exc: Optional[BaseException] = None,
        omit_usage: bool = False,
        content_blocks: Optional[List[Dict[str, str]]] = None,
    ):
        self.messages = _FakeAnthropicMessages(self)
        self.calls: List[Dict[str, Any]] = []
        self._content = content
        self._content_blocks = content_blocks
        self._finish_reason = finish_reason
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
            stop_reason=self._finish_reason,
            usage=usage_obj,
        )


# ---- Bedrock ----------------------------------------------------------

class _FakeBedrockBody(object):
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeBedrockClient(object):
    """Mimics ``boto3.client('bedrock-runtime')`` for ``invoke_model``."""

    def __init__(
        self,
        content: str = 'bedrock reply',
        finish_reason: str = 'end_turn',
        usage: Optional[Dict[str, int]] = None,
        raise_exc: Optional[BaseException] = None,
        omit_usage: bool = False,
        legacy_completion: Optional[str] = None,
    ):
        self.calls: List[Dict[str, Any]] = []
        self._content = content
        self._finish_reason = finish_reason
        self._usage = usage or {'input': 9, 'output': 4}
        self._raise = raise_exc
        self._omit_usage = omit_usage
        self._legacy_completion = legacy_completion

    def invoke_model(self, modelId: str, body: bytes, contentType: str, accept: str):
        call = {
            'modelId': modelId,
            'body': json.loads(body.decode('utf-8')),
            'contentType': contentType,
            'accept': accept,
        }
        self.calls.append(call)
        if self._raise is not None:
            raise self._raise

        if self._legacy_completion is not None:
            payload = {
                'completion': self._legacy_completion,
                'stop_reason': self._finish_reason,
            }
        else:
            payload = {
                'content': [{'type': 'text', 'text': self._content}],
                'stop_reason': self._finish_reason,
            }
        if not self._omit_usage:
            payload['usage'] = {
                'input_tokens': self._usage['input'],
                'output_tokens': self._usage['output'],
            }

        return {'body': _FakeBedrockBody(json.dumps(payload).encode('utf-8'))}
