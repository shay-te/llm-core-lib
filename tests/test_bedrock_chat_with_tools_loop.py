"""``BedrockConnection.chat_with_tools`` — multi-round tool-call loop.

The Connection drives the Bedrock Converse API. Each round either
produces ``stopReason='tool_use'`` (we run every ``toolUse`` block
via ``invoke_tool`` and feed the results back) or anything else (we
return). The OpenAI function-tool schema is converted to Bedrock's
``toolConfig`` shape inside the connection so the caller stays
provider-agnostic. Capped by ``max_tool_call_rounds``.

Per the workspace-wide "one TestCase per file" rule (see this repo's
``AGENTS.md``), this file owns exactly one TestCase. Shared fixtures
live in ``chat_with_tools_helpers``; the SDK client comes from
``tests/mock/bedrock_client`` (the one legitimate mock surface).
"""
from __future__ import annotations

import logging
import unittest

from llm_core_lib.connections.bedrock_connection import BedrockConnection
from llm_core_lib.connections.bedrock_connection_factory import BedrockConnectionFactory

from tests.helpers.chat_with_tools_helpers import (
    RecordingInvokeTool,
    sample_openai_tools,
)
from tests.mock.bedrock_client import (
    MockBedrockClient,
    make_bedrock_tool_use_response,
    make_bedrock_end_turn_response,
)


def _real_connection(client: MockBedrockClient) -> BedrockConnection:
    """Build a real BedrockConnection wrapping the mock SDK client."""
    factory = BedrockConnectionFactory({
        'region': 'us-east-1',
        'model': 'anthropic.claude-3',
        'vision_model': 'anthropic.claude-3',
        'embedding_model': 'titan',
        'max_tokens': 4096,
        'temperature': 0.0,
        'client': client,
    })
    return factory.get()


class TestBedrockChatWithToolsLoop(unittest.TestCase):

    def test_end_turn_first_round_returns_immediately(self):
        client = MockBedrockClient(converse_script=[
            make_bedrock_end_turn_response('all done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool()

        response, messages = connection.chat_with_tools(
            input_messages=[{'role': 'user', 'content': [{'text': 'hello'}]}],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        self.assertEqual(len(client.converse_calls), 1)
        self.assertEqual(invoke_tool.calls, [])
        self.assertEqual(response['stopReason'], 'end_turn')
        # The assistant message was appended; the original user one survives.
        self.assertEqual(messages[0], {'role': 'user', 'content': [{'text': 'hello'}]})
        self.assertEqual(messages[-1]['role'], 'assistant')

    def test_tool_use_round_invokes_tool_then_returns_on_next_end_turn(self):
        client = MockBedrockClient(converse_script=[
            make_bedrock_tool_use_response(
                name='list_packages',
                tool_input={'user_id': 42},
                tool_use_id='tu_xyz',
            ),
            make_bedrock_end_turn_response('here are your packages'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=[{'packages': ['Gold']}])

        response, messages = connection.chat_with_tools(
            input_messages=[{'role': 'user', 'content': [{'text': 'show packages'}]}],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        self.assertEqual(len(client.converse_calls), 2)
        self.assertEqual(invoke_tool.calls, [('list_packages', {'user_id': 42})])
        self.assertEqual(response['stopReason'], 'end_turn')
        # Tool-result follow-up message was appended in Bedrock shape;
        # the inner text is wrapped in <TOOL_DATA> markers + JSON
        # serialized (not Python repr).
        tool_result_msg = messages[2]
        self.assertEqual(tool_result_msg['role'], 'user')
        result_block = tool_result_msg['content'][0]['toolResult']
        self.assertEqual(result_block['toolUseId'], 'tu_xyz')
        result_text = result_block['content'][0]['text']
        self.assertTrue(result_text.startswith('<TOOL_DATA>\n'))
        self.assertTrue(result_text.endswith('\n</TOOL_DATA>'))
        self.assertIn('"packages"', result_text)
        self.assertIn('"Gold"', result_text)

    def test_openai_tool_schema_is_translated_to_bedrock_tool_config(self):
        # The caller passes tools in OpenAI function-tool format.
        # BedrockConnection converts to Converse ``toolConfig`` shape
        # internally — the SDK receives the Bedrock shape.
        client = MockBedrockClient(converse_script=[
            make_bedrock_end_turn_response('done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool()

        connection.chat_with_tools(
            input_messages=[{'role': 'user', 'content': [{'text': 'hi'}]}],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        sent_tool_config = client.converse_calls[0]['toolConfig']
        self.assertIn('tools', sent_tool_config)
        first = sent_tool_config['tools'][0]
        self.assertEqual(first['toolSpec']['name'], 'list_packages')
        self.assertEqual(
            first['toolSpec']['inputSchema']['json'],
            sample_openai_tools()[0]['parameters'],
        )

    def test_instructions_forwarded_as_system_text(self):
        client = MockBedrockClient(converse_script=[
            make_bedrock_end_turn_response('done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool()

        connection.chat_with_tools(
            input_messages=[{'role': 'user', 'content': [{'text': 'hi'}]}],
            tools=sample_openai_tools(),
            instructions='you are helpful',
            invoke_tool=invoke_tool,
        )

        self.assertEqual(
            client.converse_calls[0]['system'],
            [{'text': 'you are helpful'}],
        )

    def test_non_tool_use_content_block_is_skipped(self):
        # Bedrock content blocks can mix text + toolUse in the same
        # round. The loop skips blocks that don't carry ``toolUse``
        # (e.g., a text block alongside the tool call) without invoking
        # invoke_tool for that block.
        client = MockBedrockClient(converse_script=[
            {
                'output': {
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {'text': 'thinking...'},  # non-tool_use block
                            {'toolUse': {
                                'name': 'list_packages',
                                'input': {'user_id': 1},
                                'toolUseId': 'tu_1',
                            }},
                        ],
                    }
                },
                'stopReason': 'tool_use',
            },
            make_bedrock_end_turn_response('done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=['ok'])

        connection.chat_with_tools(
            input_messages=[{'role': 'user', 'content': [{'text': 'hi'}]}],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        # Only the toolUse block triggered an invocation — the text
        # block was skipped silently.
        self.assertEqual(invoke_tool.calls, [('list_packages', {'user_id': 1})])

    def test_max_tool_call_rounds_cap_returns_in_flight_response(self):
        many_tool_uses = [
            make_bedrock_tool_use_response(
                name='list_packages',
                tool_input={'user_id': i},
                tool_use_id=f'tu_{i}',
            )
            for i in range(10)
        ]
        client = MockBedrockClient(converse_script=many_tool_uses)
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=['x'] * 10)
        logger = logging.getLogger('test_bedrock_chat_with_tools_cap')

        with self.assertLogs(logger, level='WARNING') as captured:
            connection.chat_with_tools(
                input_messages=[{'role': 'user', 'content': [{'text': 'hi'}]}],
                tools=sample_openai_tools(),
                instructions='sys',
                invoke_tool=invoke_tool,
                max_tool_call_rounds=3,
                logger=logger,
            )

        self.assertEqual(len(client.converse_calls), 3)
        self.assertEqual(len(invoke_tool.calls), 3)
        self.assertTrue(any('max_tool_call_rounds' in line for line in captured.output))

    def test_bare_string_history_content_is_normalized_to_blocks(self):
        # Regression: prior-turn history arrives from ChatSession in the
        # provider-agnostic ``{role, content: <str>}`` replay shape.
        # Converse rejects bare-string content (ParamValidationError:
        # "valid types: list, tuple"), so the connection wraps each
        # string into a ``[{'text': ...}]`` block before the API call.
        # Already-block content (the current prompt) passes through
        # unchanged (no double-wrapping).
        client = MockBedrockClient(converse_script=[
            make_bedrock_end_turn_response('ok'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool()

        connection.chat_with_tools(
            input_messages=[
                {'role': 'user', 'content': 'show me a user'},         # history — bare string
                {'role': 'assistant', 'content': 'which one?'},         # history — bare string
                {'role': 'user', 'content': [{'text': 'the first'}]},   # current — already blocks
            ],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        sent_messages = client.converse_calls[0]['messages']
        self.assertEqual(sent_messages[0]['content'], [{'text': 'show me a user'}])
        self.assertEqual(sent_messages[1]['content'], [{'text': 'which one?'}])
        self.assertEqual(sent_messages[2]['content'], [{'text': 'the first'}])


if __name__ == '__main__':
    unittest.main()
