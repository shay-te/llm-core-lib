"""OpenAI Responses-API chat handler.

Stores messages in the provider shape so ``stored_to_input_message``
is identity. Input-list items are either plain
``{'role', 'content'}`` messages or tool plumbing
(``{'type': 'function_call' | 'function_call_output', ...}``).
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

    KIND = KIND_OPENAI

    def build_user_prompt(self, command: str) -> Dict[str, Any]:
        return {'role': 'user', 'content': command}

    def stored_to_input_message(self, stored_meta_data: Dict[str, Any]) -> Dict[str, Any]:
        return stored_meta_data

    def diff_new_messages(
        self,
        input_messages_before: List[Dict[str, Any]],
        final_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # Loop appends only; slice off the head we sent in.
        return list(final_messages[len(input_messages_before):])

    def summarize_for_storage(self, message: Dict[str, Any]) -> str:
        msg_type = message.get('type')
        if msg_type == 'function_call':
            name = message.get('name', '?')
            args = message.get('arguments') or ''
            return self.truncate_summary(f'tool: {name}({args})')
        if msg_type == 'function_call_output':
            return self.truncate_summary(f'tool_result: {message.get("output") or ""}')

        role = message.get('role')
        if role in ('user', 'assistant'):
            content = message.get('content') or ''
            if isinstance(content, list):
                # Vision (user) and tool-mixed assistant turns: pull
                # the first text block.
                text_key = 'output_text' if role == 'assistant' else 'text'
                content = next(
                    (b.get('text', '') for b in content
                     if isinstance(b, dict) and b.get('type') == text_key),
                    '',
                )
            return self.truncate_summary(str(content))

        # Unknown shape — JSON fallback so the row still persists.
        return self.truncate_summary(json.dumps(message, default=str))

    def extract_response_text(self, response: Any) -> str:
        # Defensive at every hop — the SDK shape has drifted before.
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
        if message.get('type') in ('function_call', 'function_call_output'):
            return SENDER_TOOL
        if message.get('role') == 'assistant':
            return SENDER_ASSISTANT
        return SENDER_USER
