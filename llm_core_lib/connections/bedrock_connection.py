"""``BedrockConnection`` — per-call wrapper around a ``boto3``
``bedrock-runtime`` client (Anthropic-shaped request body).

The companion :class:`BedrockConnectionFactory` in
``bedrock_connection_factory.py`` owns SDK-client construction.
"""
import base64
import json
import logging
from typing import Any, Callable, List, Optional, Tuple

from core_lib.connection.connection import Connection

from llm_core_lib.connections.openai_connection import format_tool_result_for_llm
from llm_core_lib.errors import LlmConfigError, LlmError, LlmProviderError
from llm_core_lib.types import LlmCompletion


_ANTHROPIC_VERSION = 'bedrock-2023-05-31'

# Matches the OpenAI connection's cap so the per-provider behaviour is
# uniform from the caller's perspective. See that constant's docstring.
DEFAULT_MAX_TOOL_CALL_ROUNDS = 100


class BedrockConnection(Connection):
    """
    Thin provider-abstracted wrapper. The methods are intentionally narrow
    — chat completion, vision, and embedding generation. Anything more
    exotic (tool use, streaming, etc.) belongs in a dedicated client at
    the edge, not in services.

    Implements the ``core_lib.connection.Connection`` context-manager
    contract so callers can write::

        with factory.get() as conn:
            completion = conn.complete_text('hello')
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        vision_model_id: str,
        embedding_model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self._client = client
        self._model_id = model_id
        self._vision_model_id = vision_model_id
        self._embedding_model = embedding_model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vision_model_id(self) -> str:
        return self._vision_model_id

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def complete_text(
        self, prompt: str, system: Optional[str] = None
    ) -> LlmCompletion:
        body = self._build_messages_body(prompt, system, image_bytes=None)
        text, usage = self._invoke(self._model_id, body)
        return LlmCompletion(text=text, model=self._model_id, usage=usage)

    def complete_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        image_mime: str = 'image/png',
        system: Optional[str] = None,
    ) -> LlmCompletion:
        # Bedrock (Anthropic-shaped) vision: image first, text second.
        body = self._build_messages_body(prompt, system, image_bytes, image_mime)
        text, usage = self._invoke(self._vision_model_id, body)
        return LlmCompletion(text=text, model=self._vision_model_id, usage=usage)

    def embed(self, text: str) -> List[float]:
        if not self._embedding_model:
            # Mirror the OpenAI adapter — raise a clear config error
            # rather than letting boto3 fail with an empty modelId.
            raise LlmConfigError(
                'bedrock connection has no embedding_model configured'
            )

        body = json.dumps({'inputText': text}).encode('utf-8')
        try:
            resp = self._client.invoke_model(
                modelId=self._embedding_model,
                body=body,
                contentType='application/json',
                accept='application/json',
            )
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'bedrock embed failed: {exc}') from exc
        payload = json.loads(resp['body'].read())
        # 1. fetch — every key we may consume out of the JSON response.
        # Titan returns `embedding`; Cohere-on-Bedrock returns
        # `embeddings`. Pull both up front, decide which shape we got
        # next, then return last.
        embedding = payload.get('embedding')
        embeddings_array = payload.get('embeddings')

        # 2. decide which shape is populated.
        if embedding:
            chosen = embedding
        elif embeddings_array:
            chosen = embeddings_array[0]
        else:
            chosen = []

        # 3. use.
        return chosen

    def chat_with_tools(
        self,
        *,
        input_messages: list,
        tools: list,
        instructions: str,
        invoke_tool: Callable[[str, dict], Any],
        max_tool_call_rounds: int = DEFAULT_MAX_TOOL_CALL_ROUNDS,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[Any, list]:
        """Multi-round Bedrock Converse tool-call loop.

        Each round either produces ``stopReason='tool_use'`` (we run
        every ``toolUse`` block via ``invoke_tool`` and feed the
        results back) or anything else (we return). Iterative — never
        recursive — so a long tool sequence cannot exhaust the Python
        stack. Capped at ``max_tool_call_rounds``.

        Args:
            input_messages: Converse-API messages list in Bedrock
                format (``[{'role': 'user', 'content': [{'text': ...}]}, ...]``).
                Messages whose ``content`` is a bare string (the
                provider-agnostic history-replay shape from ChatSession)
                are normalized to ``[{'text': ...}]`` blocks on entry. The
                caller owns the initial user message; the loop appends the
                assistant's reply and the user-role ``toolResult``
                follow-up as rounds progress.
            tools: OpenAI function-tool schema (the workspace's
                standard tool shape). Converted to Bedrock's
                ``toolConfig`` format internally so the caller stays
                provider-agnostic.
            instructions: passed as the Converse ``system`` text.
            invoke_tool: callback ``(name, kwargs) -> result``. The
                caller's choke point — authorization, gate, scrub,
                sanitized errors all live there. ``result`` is
                ``str``-encoded before being fed back as a
                ``toolResult.content`` text block.
            max_tool_call_rounds: hard cap; hit-the-cap logs WARNING
                and returns the in-flight response.
            logger: destination for round-level debug lines. Defaults
                to the connection module's logger.

        Returns:
            ``(response, input_messages)`` — the final Converse
            response dict and the final messages list.
        """
        effective_logger = logger or logging.getLogger(__name__)
        # Normalize replayed history (provider-agnostic string content)
        # into Bedrock content blocks before the first Converse call —
        # the current prompt + tool-result messages are already blocks.
        input_messages = [_ensure_bedrock_content_blocks(m) for m in input_messages]
        tool_config = {'tools': _openai_tools_to_bedrock_tool_config(tools)}
        last_response: Any = None
        for _round in range(max_tool_call_rounds):
            response = self._client.converse(
                modelId=self._model_id,
                messages=input_messages,
                system=[{'text': instructions}],
                toolConfig=tool_config,
            )
            last_response = response
            output_message = response['output']['message']
            input_messages.append(output_message)
            if response.get('stopReason') != 'tool_use':
                return response, input_messages
            tool_results = []
            for block in output_message.get('content', []):
                tool_use = block.get('toolUse')
                if not tool_use:
                    continue
                effective_logger.debug('tool call: %r', tool_use.get('name'))
                result = invoke_tool(
                    tool_use.get('name'),
                    tool_use.get('input') or {},
                )
                tool_results.append({
                    'toolResult': {
                        'toolUseId': tool_use.get('toolUseId'),
                        'content': [{'text': format_tool_result_for_llm(result)}],
                    }
                })
            input_messages.append({'role': 'user', 'content': tool_results})
        effective_logger.warning(
            'Bedrock tool-call loop hit max_tool_call_rounds (%d); returning early',
            max_tool_call_rounds,
        )
        return last_response, input_messages

    def close(self) -> None:
        # boto3 clients don't require explicit close.
        pass

    def __enter__(self) -> 'BedrockConnection':
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        # Always call close(); let any exception propagate by returning None.
        self.close()

    def _invoke(self, model_id: str, body_dict: dict) -> Tuple[str, Optional[dict]]:
        try:
            resp = self._client.invoke_model(
                modelId=model_id,
                body=json.dumps(body_dict, default=_json_default).encode('utf-8'),
                contentType='application/json',
                accept='application/json',
            )
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'bedrock invoke_model failed: {exc}') from exc

        payload = json.loads(resp['body'].read())
        # 1. fetch — pull every field _invoke touches out of the
        # response payload into named locals.
        payload_is_dict = isinstance(payload, dict)
        usage = payload.get('usage') if payload_is_dict else None

        # 2. normalize text via the shape-aware extractor.
        text = _extract_text(payload)

        # 3. use.
        return text, usage

    def _build_messages_body(
        self,
        prompt: str,
        system: Optional[str],
        image_bytes: Optional[bytes],
        image_mime: str = 'image/png',
    ) -> dict:
        content = []
        if image_bytes is not None:
            content.append(
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': image_mime,
                        'data': base64.b64encode(image_bytes).decode('ascii'),
                    },
                }
            )
        content.append({'type': 'text', 'text': prompt})

        body = {
            'anthropic_version': _ANTHROPIC_VERSION,
            'max_tokens': self._max_tokens,
            'temperature': self._temperature,
            'messages': [{'role': 'user', 'content': content}],
        }
        if system:
            body['system'] = system
        return body


def _extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ''

    # 1. fetch — every key we might consume, pulled into named locals.
    content_blocks = payload.get('content')
    legacy_completion = payload.get('completion')
    titan_results = payload.get('results')

    # 2. decide which response shape is populated.
    # Bedrock Anthropic-shaped → content is a list of blocks.
    if isinstance(content_blocks, list):
        text_parts = []
        for part in content_blocks:
            if isinstance(part, dict):
                part_text = part.get('text', '')
                text_parts.append(part_text)
        return ''.join(text_parts)

    # Legacy / Titan-shaped fallback.
    if legacy_completion:
        return legacy_completion
    if titan_results:
        first_result = titan_results[0] if titan_results else {}
        first_output = first_result.get('outputText', '') if isinstance(first_result, dict) else ''
        return first_output or ''

    # 3. use — no recognised shape.
    return ''


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('ascii')
    raise TypeError(
        f'Object of type {type(obj).__name__} is not JSON serializable'
    )


def _ensure_bedrock_content_blocks(message: dict) -> dict:
    """Coerce a message's ``content`` into Bedrock Converse block form.

    Prior-turn history arrives from :class:`ChatSession` in its
    provider-agnostic replay shape ``{role, content: <str>}`` (that
    string is OpenAI-native and Bedrock-invalid), whereas the current
    prompt and tool-result messages are already ``[{'text': ...}, ...]``
    block lists. Converse rejects bare-string ``content``
    (``ParamValidationError: valid types: list, tuple``), so wrap any
    string into a single ``[{'text': ...}]`` block; anything already a
    list is returned unchanged. Returns a NEW dict (never mutates the
    input) so the caller's pre-call snapshot — used for the positional
    ``diff_new_messages`` — is left intact.
    """
    content = message.get('content')
    if isinstance(content, str):
        return {**message, 'content': [{'text': content}]}
    return message


def _openai_tools_to_bedrock_tool_config(openai_tools: list) -> list:
    """Translate the OpenAI function-tool schema into Bedrock's
    Converse ``toolConfig.tools`` shape.

    Kept inside the BedrockConnection module so callers can pass tools
    in the workspace's standard (OpenAI) shape and not branch on
    provider. The OpenAI schema lives at the top level of each tool
    entry (``{'name': ..., 'description': ..., 'parameters': ...}``);
    Bedrock wraps it in ``toolSpec`` and uses ``inputSchema.json``.
    """
    bedrock_tools = []
    for function in openai_tools:
        bedrock_tools.append({
            'toolSpec': {
                'name': function['name'],
                'description': function.get('description', ''),
                'inputSchema': {'json': function['parameters']},
            }
        })
    return bedrock_tools
