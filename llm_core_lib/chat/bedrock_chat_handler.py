"""Bedrock Converse-API chat handler.

Messages are ``{'role', 'content': [<blocks>]}`` where each block is
``{'text'}``, ``{'toolUse'}``, or ``{'toolResult'}``. Stored verbatim.
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

        # One message can carry text + toolUse together (assistant
        # turn shape). Prefer text; fall back to tool name / result.
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
        content = message.get('content') or []
        # Bedrock's "user"-role follow-up carrying toolResult is tool
        # plumbing, not user input — classify accordingly.
        if isinstance(content, list) and any(
            isinstance(b, dict) and ('toolResult' in b or 'toolUse' in b)
            for b in content
        ):
            return SENDER_TOOL
        if message.get('role') == 'assistant':
            return SENDER_ASSISTANT
        return SENDER_USER
