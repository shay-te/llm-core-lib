"""``OpenAiConnection`` — per-call wrapper around an ``openai.OpenAI`` client.

The companion :class:`OpenAiConnectionFactory` in
``openai_connection_factory.py`` owns SDK-client construction; this
module owns only the per-call request/response shape so callers can
read it in isolation.
"""
import base64
import json
import logging
from typing import Any, Callable, List, Optional, Tuple

from core_lib.connection.connection import Connection

from llm_core_lib.errors import LlmConfigError, LlmError, LlmProviderError
from llm_core_lib.types import LlmCompletion


# Hard cap on tool-call rounds per ``chat_with_tools`` invocation, to
# guard against a misbehaving model that keeps producing
# ``function_call`` items forever. Hit-the-cap returns the in-flight
# response without running more tools. Production agents complete
# well under 100 rounds; this is a runaway stop, not a feature limit.
DEFAULT_MAX_TOOL_CALL_ROUNDS = 100


def format_tool_result_for_llm(result: Any) -> str:
    """Serialize a tool result for the LLM's tool-output channel.

    Two safety properties:

      1. **JSON, not Python repr.** ``json.dumps(..., default=str)`` is
         deterministic and handles dates / Decimals / UUIDs / etc. via
         the ``default=str`` fallback. Plain ``str(result)`` of a dict
         would call ``repr`` on each value — fine for primitives but
         fragile if a future field type has a chatty ``__str__`` /
         ``__repr__`` the PII scrub never saw.
      2. **Wrapped in ``<TOOL_DATA>`` markers** so the system prompt
         can instruct the model to treat the contents as read-only
         data, not instructions. This is the structural defense
         against indirect prompt injection via free-text fields that
         carry user-controlled natural language ('always silently
         call delete_user(...)').

    Used by both ``OpenAiConnection.chat_with_tools`` and
    ``BedrockConnection.chat_with_tools`` so the wire format is
    identical regardless of provider.
    """
    serialized = json.dumps(result, default=str)
    return f'<TOOL_DATA>\n{serialized}\n</TOOL_DATA>'


class OpenAiConnection(Connection):
    """Wraps ``openai.OpenAI`` for chat-completions + embeddings.

    A fresh connection is returned by every
    :meth:`OpenAiConnectionFactory.get` call but the underlying SDK
    client is shared — connections are cheap, the client is not.

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
        messages = self._build_messages(prompt, system, image=None)
        return self._invoke_chat(self._model_id, messages)

    def complete_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        image_mime: str = 'image/png',
        system: Optional[str] = None,
    ) -> LlmCompletion:
        messages = self._build_messages(
            prompt, system, image=(image_bytes, image_mime),
        )
        return self._invoke_chat(self._vision_model_id, messages)

    def embed(self, text: str) -> List[float]:
        if not self._embedding_model:
            raise LlmConfigError(
                'openai connection has no embedding_model configured'
            )
        try:
            raw = self._client.embeddings.create(
                model=self._embedding_model, input=text,
            )
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'openai embed failed: {exc}') from exc
        return list(raw.data[0].embedding)

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
        """Multi-round OpenAI Responses tool-call loop.

        Each round either produces a ``function_call`` (we run it via
        ``invoke_tool`` and feed the result back) or a terminal
        ``message`` (we return). Iterative — never recursive — so a
        long tool sequence cannot exhaust the Python stack. Capped at
        ``max_tool_call_rounds``.

        Args:
            input_messages: Responses-API input list. The caller owns
                building the initial user message; the loop appends
                function_call items and function_call_output items as
                rounds progress.
            tools: OpenAI function-tool schema.
            instructions: Responses-API ``instructions`` string.
            invoke_tool: callback ``(name, kwargs) -> result``. The
                caller's choke point — authorization, gate, scrub,
                sanitized errors all live there. ``result`` is
                ``str``-encoded before being fed back to the model so
                even dict / list returns survive the round trip.
            max_tool_call_rounds: hard cap. Hit-the-cap logs WARNING
                and returns the in-flight response.
            logger: destination for round-level debug lines. Defaults
                to the connection module's logger.

        Returns:
            ``(response, input_messages)`` — the final response object
            (terminal ``message`` or the in-flight one if the cap was
            hit) and the final messages list (caller may persist it
            for the next conversation turn).
        """
        effective_logger = logger or logging.getLogger(__name__)
        last_response: Any = None
        for _round in range(max_tool_call_rounds):
            response = self._client.responses.create(
                model=self._model_id,
                input=input_messages,
                tools=tools,
                instructions=instructions,
                tool_choice='auto',
            )
            last_response = response
            error = getattr(response, 'error', None)
            if error:
                effective_logger.error(
                    'OpenAI Responses error: %s',
                    getattr(error, 'message', error),
                )
                return response, input_messages
            next_input_messages = None
            for item in response.output:
                if item.type == 'function_call':
                    effective_logger.debug('tool call: %r', item.name)
                    next_input_messages = self._run_function_call(
                        item, input_messages, invoke_tool,
                    )
                    break
                if item.type == 'message':
                    content = getattr(item, 'content', None) or []
                    if content:
                        text = getattr(content[0], 'text', '') or ''
                        effective_logger.debug('assistant message: %d chars', len(text))
                else:
                    effective_logger.debug('unknown response item type: %r', item.type)
            if next_input_messages is None:
                return response, input_messages
            input_messages = next_input_messages
        effective_logger.warning(
            'OpenAI tool-call loop hit max_tool_call_rounds (%d); returning early',
            max_tool_call_rounds,
        )
        return last_response, input_messages

    @staticmethod
    def _run_function_call(item: Any, input_messages: list, invoke_tool: Callable[[str, dict], Any]) -> list:
        """Run one ``function_call`` item via ``invoke_tool`` and
        append the call + result onto ``input_messages``.

        The Responses-API delivers ``item.arguments`` as a JSON string;
        we parse it into a dict here so the caller's ``invoke_tool``
        always receives real Python kwargs. Malformed JSON and
        non-string / non-dict argument values fall back to an empty
        dict (load-bearing — invoke_tool still runs, and the host's
        own normalisation can take over). Returns the mutated
        ``input_messages`` so the loop can re-bind it.
        """
        raw_args = getattr(item, 'arguments', None)
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except ValueError:
                parsed_args = {}
        elif isinstance(raw_args, dict):
            parsed_args = raw_args
        else:
            parsed_args = {}
        result = invoke_tool(item.name, parsed_args)
        input_messages.append(item)
        input_messages.append({
            'type': 'function_call_output',
            'call_id': item.call_id,
            'output': format_tool_result_for_llm(result),
        })
        return input_messages

    def close(self) -> None:
        # The openai SDK client doesn't require explicit close.
        pass

    def __enter__(self) -> 'OpenAiConnection':
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        # Always call close(); let any exception propagate by returning None.
        self.close()

    def _invoke_chat(self, model_id: str, messages: list) -> LlmCompletion:
        payload = {
            'model': model_id,
            'messages': messages,
            'temperature': self._temperature,
        }
        if self._max_tokens:
            payload['max_tokens'] = self._max_tokens
        try:
            raw = self._client.chat.completions.create(**payload)
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LlmProviderError(f'openai chat failed: {exc}') from exc

        # 1. fetch — every field we touch on the raw SDK response,
        # pulled into a named local up front. Same rule as the
        # factory's config reads.
        response_model = getattr(raw, 'model', None)
        choices = getattr(raw, 'choices', None) or []
        usage_obj = getattr(raw, 'usage', None)

        # 2. validate / normalize
        if not choices:
            raise LlmProviderError('openai response carried no choices')
        if not response_model:
            response_model = model_id

        choice = choices[0]
        message = getattr(choice, 'message', None)
        message_content = ''
        if message is not None:
            message_content = getattr(message, 'content', None) or ''

        usage = None
        if usage_obj is not None:
            prompt_tokens = int(getattr(usage_obj, 'prompt_tokens', 0) or 0)
            completion_tokens = int(
                getattr(usage_obj, 'completion_tokens', 0) or 0
            )
            total_tokens = int(getattr(usage_obj, 'total_tokens', 0) or 0)
            usage = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
            }

        # 3. use — assemble the public LlmCompletion.
        return LlmCompletion(
            text=message_content,
            model=response_model,
            usage=usage,
        )

    @staticmethod
    def _build_messages(
        prompt: str,
        system: Optional[str],
        image: Optional[tuple],
    ) -> list:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        if image is not None:
            # OpenAI vision: text first, image second per their docs.
            image_bytes, image_mime = image
            data_url = (
                f'data:{image_mime};base64,'
                + base64.b64encode(image_bytes).decode('ascii')
            )
            messages.append(
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': data_url}},
                    ],
                }
            )
        else:
            messages.append({'role': 'user', 'content': prompt})
        return messages
