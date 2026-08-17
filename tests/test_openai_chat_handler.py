"""``OpenAiChatHandler`` — provider-shape adapter for OpenAI Responses.

Per the workspace-wide "one TestCase per file" rule, this file owns
exactly one TestCase.
"""
from __future__ import annotations

import unittest

from llm_core_lib.chat.chat_history_store import (
    SENDER_ASSISTANT,
    SENDER_TOOL,
    SENDER_USER,
)
from llm_core_lib.chat.chat_handler import MAX_CONTENT_SUMMARY_LEN
from llm_core_lib.chat.openai_chat_handler import KIND_OPENAI, OpenAiChatHandler


class _ResponseStub(object):
    def __init__(self, output):
        self.output = output


class _MessageStub(object):
    def __init__(self, content_items):
        self.type = 'message'
        self.content = content_items


class _TextContent(object):
    def __init__(self, text):
        self.text = text


class TestOpenAiChatHandler(unittest.TestCase):

    def setUp(self):
        self.handler = OpenAiChatHandler()

    def test_kind_constant(self):
        self.assertEqual(self.handler.KIND, KIND_OPENAI)

    def test_build_user_prompt_returns_responses_shape(self):
        message = self.handler.build_user_prompt('hi')
        self.assertEqual(message, {'role': 'user', 'content': 'hi'})

    def test_stored_to_input_message_is_identity(self):
        # OpenAI handler stores raw provider shape; round-trip is
        # the identity. Other handlers may differ.
        stored = {'role': 'assistant', 'content': 'echo'}
        self.assertIs(self.handler.stored_to_input_message(stored), stored)

    def test_diff_returns_tail_after_prefix(self):
        before = [{'role': 'user', 'content': 'a'}]
        final = [
            {'role': 'user', 'content': 'a'},
            {'type': 'function_call', 'call_id': 'c1', 'name': 'list_x', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'c1', 'output': '[]'},
            {'role': 'assistant', 'content': 'done'},
        ]
        diff = self.handler.diff_new_messages(before, final)
        self.assertEqual(len(diff), 3)
        self.assertEqual(diff[-1]['content'], 'done')

    def test_summarize_user_string_content(self):
        summary = self.handler.summarize_for_storage(
            {'role': 'user', 'content': 'find jane'},
        )
        self.assertEqual(summary, 'find jane')

    def test_summarize_function_call_includes_name_and_args(self):
        summary = self.handler.summarize_for_storage({
            'type': 'function_call',
            'name': 'list_users',
            'arguments': '{"q":"jane"}',
            'call_id': 'c1',
        })
        self.assertIn('tool: list_users', summary)
        self.assertIn('jane', summary)

    def test_summarize_function_call_output_includes_output(self):
        summary = self.handler.summarize_for_storage({
            'type': 'function_call_output',
            'call_id': 'c1',
            'output': '<TOOL_DATA>{"users": []}</TOOL_DATA>',
        })
        self.assertTrue(summary.startswith('tool_result:'))
        self.assertIn('TOOL_DATA', summary)

    def test_summarize_truncates_to_content_column_limit(self):
        long_text = 'x' * (MAX_CONTENT_SUMMARY_LEN + 200)
        summary = self.handler.summarize_for_storage(
            {'role': 'user', 'content': long_text},
        )
        self.assertLessEqual(len(summary), MAX_CONTENT_SUMMARY_LEN)

    def test_summarize_unknown_shape_falls_back_to_json(self):
        # Defensive — never drops a message because we don't know
        # the shape; the full original is in meta_data anyway.
        summary = self.handler.summarize_for_storage({'unknown': 1})
        self.assertIn('unknown', summary)

    def test_extract_response_text_walks_responses_output(self):
        response = _ResponseStub(output=[
            _MessageStub([_TextContent('hello world')]),
        ])
        self.assertEqual(self.handler.extract_response_text(response), 'hello world')

    def test_extract_response_text_empty_for_missing_message_item(self):
        # When the loop hit max_tool_call_rounds the response may
        # not contain a terminal message — we return '' rather than
        # raising.
        response = _ResponseStub(output=[])
        self.assertEqual(self.handler.extract_response_text(response), '')

    def test_extract_response_text_empty_for_none_response(self):
        # The connection's leak guard can return None — handler
        # must tolerate it.
        self.assertEqual(self.handler.extract_response_text(None), '')

    def test_sender_for_user_message(self):
        self.assertEqual(
            self.handler.sender_for({'role': 'user', 'content': 'hi'}),
            SENDER_USER,
        )

    def test_sender_for_assistant_message(self):
        self.assertEqual(
            self.handler.sender_for({'role': 'assistant', 'content': 'reply'}),
            SENDER_ASSISTANT,
        )

    def test_sender_for_function_call(self):
        self.assertEqual(
            self.handler.sender_for(
                {'type': 'function_call', 'call_id': 'c1', 'name': 'x', 'arguments': '{}'},
            ),
            SENDER_TOOL,
        )

    def test_sender_for_function_call_output(self):
        self.assertEqual(
            self.handler.sender_for(
                {'type': 'function_call_output', 'call_id': 'c1', 'output': '[]'},
            ),
            SENDER_TOOL,
        )


if __name__ == '__main__':
    unittest.main()
