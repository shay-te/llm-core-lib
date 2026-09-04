"""``BedrockChatHandler`` — provider-shape adapter for Bedrock Converse.

Per the workspace-wide "one TestCase per file" rule, this file owns
exactly one TestCase.
"""
from __future__ import annotations

import unittest

from llm_core_lib.chat.bedrock_chat_handler import BedrockChatHandler, KIND_BEDROCK
from llm_core_lib.chat.chat_history_store import (
    SENDER_ASSISTANT,
    SENDER_TOOL,
    SENDER_USER,
)


class TestBedrockChatHandler(unittest.TestCase):

    def setUp(self):
        self.handler = BedrockChatHandler()

    def test_kind_constant(self):
        self.assertEqual(self.handler.KIND, KIND_BEDROCK)

    def test_build_user_prompt_uses_text_block(self):
        # Bedrock content is a list of blocks, NOT a flat string.
        self.assertEqual(
            self.handler.build_user_prompt('hi'),
            {'role': 'user', 'content': [{'text': 'hi'}]},
        )

    def test_stored_to_input_message_is_identity(self):
        stored = {'role': 'assistant', 'content': [{'text': 'echo'}]}
        self.assertIs(self.handler.stored_to_input_message(stored), stored)

    def test_summarize_assistant_text_block(self):
        message = {'role': 'assistant', 'content': [{'text': 'hello'}]}
        self.assertEqual(self.handler.summarize_for_storage(message), 'hello')

    def test_summarize_tool_use_block_names_the_tool(self):
        message = {
            'role': 'assistant',
            'content': [{'toolUse': {'name': 'list_users', 'toolUseId': 'tu1'}}],
        }
        summary = self.handler.summarize_for_storage(message)
        self.assertIn('tool: list_users', summary)

    def test_summarize_tool_result_block(self):
        message = {
            'role': 'user',
            'content': [{
                'toolResult': {
                    'toolUseId': 'tu1',
                    'content': [{'text': 'rows: 2'}],
                },
            }],
        }
        summary = self.handler.summarize_for_storage(message)
        self.assertTrue(summary.startswith('tool_result:'))
        self.assertIn('rows: 2', summary)

    def test_summarize_falls_back_for_unknown_shape(self):
        # No 'content' list at all → json fallback. Doesn't drop.
        summary = self.handler.summarize_for_storage({'role': 'tool'})
        self.assertIn('role', summary)

    def test_extract_response_text_walks_converse_message(self):
        response = {
            'output': {
                'message': {
                    'content': [{'text': 'hello world'}],
                },
            },
        }
        self.assertEqual(self.handler.extract_response_text(response), 'hello world')

    def test_extract_response_text_empty_for_non_dict(self):
        self.assertEqual(self.handler.extract_response_text(None), '')
        self.assertEqual(self.handler.extract_response_text('oops'), '')

    def test_extract_response_text_empty_for_missing_content(self):
        # Loop hit cap → response may not carry text. No crash.
        self.assertEqual(
            self.handler.extract_response_text({'output': {'message': {}}}),
            '',
        )

    def test_sender_for_user_text_message(self):
        self.assertEqual(
            self.handler.sender_for({'role': 'user', 'content': [{'text': 'hi'}]}),
            SENDER_USER,
        )

    def test_sender_for_assistant_text_message(self):
        self.assertEqual(
            self.handler.sender_for({'role': 'assistant', 'content': [{'text': 'r'}]}),
            SENDER_ASSISTANT,
        )

    def test_sender_for_tool_use_block_is_tool(self):
        message = {
            'role': 'assistant',
            'content': [{'toolUse': {'name': 'x', 'toolUseId': 'tu1'}}],
        }
        self.assertEqual(self.handler.sender_for(message), SENDER_TOOL)

    def test_sender_for_user_role_carrying_tool_result_is_tool_not_user(self):
        # Bedrock's tool-result follow-up has role='user' by protocol.
        # Classify as TOOL so a UI doesn't render it as if the human
        # typed the tool's JSON.
        message = {
            'role': 'user',
            'content': [{'toolResult': {'toolUseId': 'tu1', 'content': [{'text': 'r'}]}}],
        }
        self.assertEqual(self.handler.sender_for(message), SENDER_TOOL)


if __name__ == '__main__':
    unittest.main()
