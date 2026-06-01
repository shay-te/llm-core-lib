"""Mock ``boto3.client('bedrock-runtime')`` for the connection-factory
tests. Mirrors ``invoke_model`` and records every call into ``calls``
for assertions. Returns Anthropic-on-Bedrock-shaped payloads by
default; pass ``legacy_completion`` or ``titan_results`` to exercise
the legacy / Titan fallback paths, or ``embedding`` to exercise embed.
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
    """Mocks ``boto3.client('bedrock-runtime')`` for ``invoke_model``."""

    def __init__(
        self,
        content: str = 'bedrock reply',
        usage: Optional[Dict[str, int]] = None,
        raise_exc: Optional[BaseException] = None,
        omit_usage: bool = False,
        legacy_completion: Optional[str] = None,
        titan_results: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ):
        self.calls: List[Dict[str, Any]] = []
        self._content = content
        self._usage = usage or {'input': 9, 'output': 4}
        self._raise = raise_exc
        self._omit_usage = omit_usage
        self._legacy_completion = legacy_completion
        self._titan_results = titan_results
        self._embedding = embedding

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
