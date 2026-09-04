"""Mock ``boto3.client('bedrock-runtime')`` for the connection-factory
tests. Mirrors ``invoke_model`` (Chat Completions / Vision / Embedding
via Anthropic-on-Bedrock body) and ``converse`` (Bedrock Converse API
with tool use — drives ``chat_with_tools``).

Returns Anthropic-on-Bedrock-shaped payloads by default; pass
``legacy_completion`` or ``titan_results`` to exercise the legacy /
Titan fallback paths, or ``embedding`` to exercise embed. For the
``converse`` API, pass ``converse_script`` — a list of pre-canned
responses, one per call, so a test can script a multi-round
tool-call dance.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class _MockBedrockBody(object):
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class MockBedrockClient(object):
    """Mocks ``boto3.client('bedrock-runtime')`` for ``invoke_model``
    and ``converse``."""

    def __init__(
        self,
        content: str = 'bedrock reply',
        usage: Optional[Dict[str, int]] = None,
        raise_exc: Optional[BaseException] = None,
        omit_usage: bool = False,
        legacy_completion: Optional[str] = None,
        titan_results: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        converse_script: Optional[List[Any]] = None,
    ):
        self.calls: List[Dict[str, Any]] = []
        self.converse_calls: List[Dict[str, Any]] = []
        self._content = content
        self._usage = usage or {'input': 9, 'output': 4}
        self._raise = raise_exc
        self._omit_usage = omit_usage
        self._legacy_completion = legacy_completion
        self._titan_results = titan_results
        self._embedding = embedding
        self._converse_script = list(converse_script) if converse_script is not None else None
        self._converse_index = 0

    def invoke_model(
        self,
        modelId: str,
        body: bytes,
        contentType: str,
        accept: str,
    ):
        call = {
            'modelId': modelId,
            'body': json.loads(body.decode('utf-8')),
            'contentType': contentType,
            'accept': accept,
        }
        self.calls.append(call)
        if self._raise is not None:
            raise self._raise

        # Embedding requests have an `inputText` field in the body —
        # respond with the configured embedding vector.
        if isinstance(call['body'], dict) and 'inputText' in call['body']:
            payload = (
                {'embedding': list(self._embedding)}
                if self._embedding is not None
                else {'embeddings': [[0.1, 0.2]]}
            )
            return {'body': _MockBedrockBody(json.dumps(payload).encode('utf-8'))}

        # Chat / vision request — pick a response shape.
        if self._legacy_completion is not None:
            payload = {'completion': self._legacy_completion}
        elif self._titan_results is not None:
            payload = {'results': [{'outputText': t} for t in self._titan_results]}
        else:
            payload = {'content': [{'type': 'text', 'text': self._content}]}
        if not self._omit_usage and (
            self._legacy_completion is None and self._titan_results is None
        ):
            payload['usage'] = {
                'input_tokens': self._usage['input'],
                'output_tokens': self._usage['output'],
            }

        return {'body': _MockBedrockBody(json.dumps(payload).encode('utf-8'))}

    def converse(
        self,
        modelId: str,
        messages: list,
        system: list,
        toolConfig: dict,
    ):
        call = {
            'modelId': modelId,
            'messages': messages,
            'system': system,
            'toolConfig': toolConfig,
        }
        self.converse_calls.append(call)
        if self._raise is not None:
            raise self._raise
        if self._converse_script is None:
            # Default: terminal end_turn message with no tool use.
            return {
                'output': {
                    'message': {
                        'role': 'assistant',
                        'content': [{'text': self._content}],
                    }
                },
                'stopReason': 'end_turn',
            }
        if self._converse_index >= len(self._converse_script):
            raise AssertionError(
                'MockBedrockClient converse_script exhausted; '
                'add more entries or shorten the test loop.'
            )
        scripted = self._converse_script[self._converse_index]
        self._converse_index += 1
        return scripted


def make_bedrock_tool_use_response(
    *,
    name: str,
    tool_input: dict,
    tool_use_id: str = 'tu_1',
):
    """Build a Converse response whose content carries a single
    ``toolUse`` block (with ``stopReason='tool_use'``). Public so test
    files can script a multi-round tool-call dance."""
    return {
        'output': {
            'message': {
                'role': 'assistant',
                'content': [
                    {'toolUse': {
                        'name': name,
                        'input': tool_input,
                        'toolUseId': tool_use_id,
                    }},
                ],
            }
        },
        'stopReason': 'tool_use',
    }


def make_bedrock_end_turn_response(text: str = 'final'):
    """Build a Converse response with a terminal ``end_turn`` message."""
    return {
        'output': {
            'message': {
                'role': 'assistant',
                'content': [{'text': text}],
            }
        },
        'stopReason': 'end_turn',
    }
