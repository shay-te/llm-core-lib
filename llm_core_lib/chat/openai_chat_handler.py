"""OpenAI Responses-API chat handler.

Stores messages in the same shape OpenAI returns — that way
``stored_to_input_message`` is the identity function and we don't
lose any field on the round trip.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from llm_core_lib.chat.chat_handler import ChatHandler
from llm_core_lib.chat.chat_history_store import (
    SENDER_ASSISTANT,
    SENDER_TOOL,
    SENDER_USER,
)


KIND_OPENAI = 'main_chat_openai'


class OpenAiChatHandler(ChatHandler):
    """Handles the OpenAI Responses input-list shape.

    Two flavours appear in the list:

      * Plain user / assistant messages:
        ``{'role': 'user'|'assistant', 'content': '...'}``
      * Tool plumbing:
        ``{'type': 'function_call', 'call_id': ..., 'name': ..., 'arguments': '...'}``
        ``{'type': 'function_call_output', 'call_id': ..., 'output': '...'}``

    Both are JSON-serialisable so persisting verbatim is safe.
    """

    KIND = KIND_OPENAI

    def build_user_prompt(self, command: str) -> Dict[str, Any]:
        return {'role': 'user', 'content': command}

    def stored_to_input_message(self, stored_meta_data: Dict[str, Any]) -> Dict[str, Any]:
        # Stored shape == provider shape. Return as-is.
        return stored_meta_data

    def diff_new_messages(
        self,
        input_messages_before: List[Dict[str, Any]],
        final_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # The connection's loop appends; nothing in the head is
        # rewritten. Slice off the prefix the orchestrator already had.
        return list(final_messages[len(input_messages_before):])

    def summarize_for_storage(self, message: Dict[str, Any]) -> str:
        msg_type = message.get('type')
        if msg_type == 'function_call':
            name = message.get('name', '?')
            args = message.get('arguments') or ''
            return self.truncate_summary(f'tool: {name}({args})')
        if msg_type == 'function_call_output':
            output = message.get('output') or ''
            return self.truncate_summary(f'tool_result: {output}')

        role = message.get('role')
        if role == 'user':
            content = message.get('content') or ''
            if isinstance(content, list):
                # OpenAI vision shape — list of {type, text/image_url}.
                # Pull the first text block for the summary.
                content = next(
                    (part.get('text', '') for part in content
                     if isinstance(part, dict) and part.get('type') == 'text'),
                    '',
                )
            return self.truncate_summary(str(content))
        if role == 'assistant':
            content = message.get('content') or ''
            if isinstance(content, list):
                # The connection's loop appends raw response items —
                # an assistant turn that produced both text and a tool
                # call will land here as a list. Take the first text.
                content = next(
                    (block.get('text', '') for block in content
                     if isinstance(block, dict) and block.get('type') == 'output_text'),
                    '',
                )
            return self.truncate_summary(str(content))

        # Unknown shape — log-friendly fallback. Keeps storage safe
        # without dropping the row (the full original is in meta_data).
        return self.truncate_summary(json.dumps(message, default=str))

    def extract_response_text(self, response: Any) -> str:
        # response.output is a list of items; the terminal one is a
        # message item with .content[0].text. Defensive at every hop
        # because the SDK shape has drifted before.
        output = getattr(response, 'output', None) or []
        try:
            items = list(output)
        except TypeError:
            return ''
        for item in items:
            if getattr(item, 'type', None) == 'message':
                content = getattr(item, 'content', None) or []
                if content:
                    return getattr(content[0], 'text', '') or ''
        return ''

    def sender_for(self, message: Dict[str, Any]) -> str:
        msg_type = message.get('type')
        if msg_type in ('function_call', 'function_call_output'):
            return SENDER_TOOL
        role = message.get('role')
        if role == 'assistant':
            return SENDER_ASSISTANT
        return SENDER_USER
