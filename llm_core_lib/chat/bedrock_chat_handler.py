"""Bedrock Converse-API chat handler.

Bedrock's input shape differs from OpenAI's: messages are
``{'role': 'user'|'assistant', 'content': [<blocks>]}`` where each
block is a ``{'text': ...}``, ``{'toolUse': ...}``, or
``{'toolResult': ...}`` dict. Stored verbatim (no normalisation).
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


KIND_BEDROCK = 'main_chat_bedrock'


class BedrockChatHandler(ChatHandler):
    """Handles the Bedrock Converse messages-list shape."""

    KIND = KIND_BEDROCK

    def build_user_prompt(self, command: str) -> Dict[str, Any]:
        return {'role': 'user', 'content': [{'text': command}]}

    def stored_to_input_message(self, stored_meta_data: Dict[str, Any]) -> Dict[str, Any]:
        return stored_meta_data

    def diff_new_messages(
        self,
        input_messages_before: List[Dict[str, Any]],
        final_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return list(final_messages[len(input_messages_before):])

    def summarize_for_storage(self, message: Dict[str, Any]) -> str:
        role = message.get('role')
        content = message.get('content')
        if not isinstance(content, list):
            return self.truncate_summary(json.dumps(message, default=str))

        # A single message can carry text + a toolUse block at once
        # (Bedrock's assistant turn shape). The summary prefers the
        # text; if absent, names the tool.
        text_parts = []
        tool_use_names = []
        tool_result_summary = None
        for block in content:
            if not isinstance(block, dict):
                continue
            if 'text' in block:
                text_parts.append(str(block.get('text') or ''))
            tool_use = block.get('toolUse')
            if isinstance(tool_use, dict):
                tool_use_names.append(str(tool_use.get('name') or '?'))
            tool_result = block.get('toolResult')
            if isinstance(tool_result, dict):
                # toolResult.content is a list of text blocks too.
                inner = tool_result.get('content') or []
                texts = [
                    part.get('text', '') for part in inner
                    if isinstance(part, dict) and part.get('text')
                ]
                tool_result_summary = ' '.join(texts)

        if text_parts:
            return self.truncate_summary(' '.join(text_parts))
        if tool_use_names:
            return self.truncate_summary(f'tool: {", ".join(tool_use_names)}')
        if tool_result_summary is not None:
            return self.truncate_summary(f'tool_result: {tool_result_summary}')
        return self.truncate_summary(f'[{role or "?"}: empty]')

    def extract_response_text(self, response: Any) -> str:
        if not isinstance(response, dict):
            return ''
        message = (response.get('output') or {}).get('message') or {}
        for block in message.get('content', []) or []:
            if isinstance(block, dict) and block.get('text'):
                return block['text']
        return ''

    def sender_for(self, message: Dict[str, Any]) -> str:
        role = message.get('role')
        content = message.get('content') or []
        # Bedrock's "user" follow-up that carries a toolResult is the
        # tool plumbing — classify as SENDER_TOOL even though the
        # protocol role is 'user'.
        if isinstance(content, list) and any(
            isinstance(block, dict) and 'toolResult' in block
            for block in content
        ):
            return SENDER_TOOL
        if isinstance(content, list) and any(
            isinstance(block, dict) and 'toolUse' in block
            for block in content
        ):
            return SENDER_TOOL
        if role == 'assistant':
            return SENDER_ASSISTANT
        return SENDER_USER
