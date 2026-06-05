"""``OpenAiConnection.chat_with_tools`` — multi-round tool-call loop.

The Connection drives the OpenAI Responses API. Each round either
produces a ``function_call`` (we run it via ``invoke_tool`` and feed
the result back) or a terminal ``message`` (we return). Capped by
``max_tool_call_rounds`` so a misbehaving model can't pin a request.

Per the workspace-wide "one TestCase per file" rule (see this repo's
``AGENTS.md``), this file owns exactly one TestCase. Shared fixtures
live in ``chat_with_tools_helpers``; the SDK client comes from
``tests/mock/openai_client`` (the one legitimate mock surface).
"""
from __future__ import annotations

import logging
import unittest

from llm_core_lib.connections.openai_connection import OpenAiConnection
from llm_core_lib.connections.openai_connection_factory import OpenAiConnectionFactory

from llm_core_lib.tests.chat_with_tools_helpers import (
    RecordingInvokeTool,
    sample_openai_tools,
)
from llm_core_lib.tests.mock.openai_client import (
    MockOpenAIClient,
    make_openai_function_call_response,
    make_openai_message_response,
    make_openai_error_response,
)


def _real_connection(client: MockOpenAIClient) -> OpenAiConnection:
    """Build a real OpenAiConnection wrapping the mock SDK client.

    The factory does the config validation + builds the Connection —
    the same code path the registry uses at boot.
    """
    factory = OpenAiConnectionFactory({
        'model': 'gpt-4o-mini',
        'vision_model': 'gpt-4o-mini',
        'embedding_model': 'emb',
        'max_tokens': 4096,
        'temperature': 0.0,
        'api_key': 'sk-test',
        'client': client,
    })
    return factory.get()


class TestOpenAiChatWithToolsLoop(unittest.TestCase):

    def test_terminal_message_first_round_returns_immediately(self):
        client = MockOpenAIClient(responses_script=[
            make_openai_message_response('all done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool()

        response, messages = connection.chat_with_tools(
            input_messages=[{'role': 'user', 'content': 'hello'}],
            tools=sample_openai_tools(),
            instructions='you are helpful',
            invoke_tool=invoke_tool,
        )

        # Single round; no tool calls.
        self.assertEqual(len(client.responses_calls), 1)
        self.assertEqual(invoke_tool.calls, [])
        self.assertEqual(response.output[0].type, 'message')
        # The initial user message survives in the messages list.
        self.assertEqual(messages, [{'role': 'user', 'content': 'hello'}])

    def test_function_call_round_invokes_tool_then_returns_on_next_message(self):
        client = MockOpenAIClient(responses_script=[
            make_openai_function_call_response(
                name='list_packages',
                arguments='{"user_id": 42}',
                call_id='call_abc',
            ),
            make_openai_message_response('here are your packages'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=[{'packages': ['Gold']}])

        response, messages = connection.chat_with_tools(
            input_messages=[{'role': 'user', 'content': 'show me packages'}],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        # Two rounds, one tool call.
        self.assertEqual(len(client.responses_calls), 2)
        self.assertEqual(invoke_tool.calls, [('list_packages', {'user_id': 42})])
        # The final response is the terminal message.
        self.assertEqual(response.output[0].type, 'message')
        # Messages list now carries the function_call + function_call_output
        # (appended by the connection's loop), in addition to the
        # original user message.
        self.assertEqual(messages[0], {'role': 'user', 'content': 'show me packages'})
        # Tool result is wrapped in <TOOL_DATA> markers + JSON-serialized
        # (not Python repr) so the system prompt can instruct the model
        # to treat the inner content as read-only data.
        last = messages[-1]
        self.assertEqual(last['type'], 'function_call_output')
        self.assertEqual(last['call_id'], 'call_abc')
        self.assertTrue(last['output'].startswith('<TOOL_DATA>\n'))
        self.assertTrue(last['output'].endswith('\n</TOOL_DATA>'))
        self.assertIn('"packages"', last['output'])
        self.assertIn('"Gold"', last['output'])

    def test_arguments_are_json_parsed_before_invoke_tool(self):
        # The OpenAI Responses API delivers ``arguments`` as a JSON
        # string; the connection must parse it into a dict before
        # calling ``invoke_tool``. ``invoke_tool`` receives a real dict.
        client = MockOpenAIClient(responses_script=[
            make_openai_function_call_response(
                name='list_packages', arguments='{"user_id": 7}', call_id='c',
            ),
            make_openai_message_response('done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=['ok'])

        connection.chat_with_tools(
            input_messages=[],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        self.assertEqual(invoke_tool.calls[0][1], {'user_id': 7})

    def test_error_response_returns_early_without_tool_calls(self):
        client = MockOpenAIClient(responses_script=[
            make_openai_error_response('rate limited'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool()
        logger = logging.getLogger('test_openai_chat_with_tools_error')

        with self.assertLogs(logger, level='ERROR') as captured:
            response, _ = connection.chat_with_tools(
                input_messages=[],
                tools=sample_openai_tools(),
                instructions='sys',
                invoke_tool=invoke_tool,
                logger=logger,
            )

        self.assertEqual(len(client.responses_calls), 1)
        self.assertEqual(invoke_tool.calls, [])
        self.assertEqual(response.error.message, 'rate limited')
        self.assertTrue(any('rate limited' in line for line in captured.output))

    def test_unknown_response_item_type_is_logged_and_loop_returns(self):
        # Defensive: if the SDK ever returns an item with a type that's
        # neither ``function_call`` nor ``message``, the loop debug-logs
        # the unknown type and exits the round (next_input_messages
        # stays None -> return).
        from types import SimpleNamespace
        client = MockOpenAIClient(responses_script=[
            SimpleNamespace(error=None, output=[
                SimpleNamespace(type='reasoning'),  # not message, not function_call
            ]),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool()
        logger = logging.getLogger('test_openai_unknown_item_type')

        with self.assertLogs(logger, level='DEBUG') as captured:
            response, _ = connection.chat_with_tools(
                input_messages=[],
                tools=sample_openai_tools(),
                instructions='sys',
                invoke_tool=invoke_tool,
                logger=logger,
            )

        self.assertEqual(invoke_tool.calls, [])
        self.assertTrue(any('unknown response item type' in line for line in captured.output))

    def test_malformed_arguments_json_falls_back_to_empty_dict(self):
        # ``item.arguments`` is a JSON string; if the model emits
        # garbage (or empty / non-JSON), the connection parses leniently
        # — invoke_tool is still called with ``{}`` so the caller's
        # tool sees defaults rather than a crash mid-loop.
        client = MockOpenAIClient(responses_script=[
            make_openai_function_call_response(
                name='list_packages', arguments='this is not json',
                call_id='c',
            ),
            make_openai_message_response('done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=['ok'])

        connection.chat_with_tools(
            input_messages=[],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        self.assertEqual(invoke_tool.calls, [('list_packages', {})])

    def test_empty_arguments_string_falls_back_to_empty_dict(self):
        client = MockOpenAIClient(responses_script=[
            make_openai_function_call_response(
                name='list_packages', arguments='', call_id='c',
            ),
            make_openai_message_response('done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=['ok'])

        connection.chat_with_tools(
            input_messages=[],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        self.assertEqual(invoke_tool.calls, [('list_packages', {})])

    def test_dict_arguments_pass_through_without_json_parse(self):
        # Some SDK versions / mocks deliver ``arguments`` already as a
        # dict; the connection accepts that shape without re-parsing.
        from types import SimpleNamespace
        client = MockOpenAIClient(responses_script=[
            SimpleNamespace(error=None, output=[
                SimpleNamespace(
                    type='function_call',
                    name='list_packages',
                    arguments={'user_id': 5},
                    call_id='c',
                ),
            ]),
            make_openai_message_response('done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=['ok'])

        connection.chat_with_tools(
            input_messages=[],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        self.assertEqual(invoke_tool.calls, [('list_packages', {'user_id': 5})])

    def test_non_string_non_dict_arguments_fall_back_to_empty_dict(self):
        # ``arguments`` is None / int / other unexpected type — the
        # connection treats it as "no args" and the tool runs with {}.
        from types import SimpleNamespace
        client = MockOpenAIClient(responses_script=[
            SimpleNamespace(error=None, output=[
                SimpleNamespace(
                    type='function_call',
                    name='list_packages',
                    arguments=None,
                    call_id='c',
                ),
            ]),
            make_openai_message_response('done'),
        ])
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=['ok'])

        connection.chat_with_tools(
            input_messages=[],
            tools=sample_openai_tools(),
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        self.assertEqual(invoke_tool.calls, [('list_packages', {})])

    def test_max_tool_call_rounds_cap_returns_in_flight_response(self):
        # Script: every round is a function_call. The loop hits the cap
        # and returns the last in-flight response WITHOUT calling more.
        many_function_calls = [
            make_openai_function_call_response(
                name='list_packages',
                arguments='{}',
                call_id=f'c{i}',
            )
            for i in range(10)
        ]
        client = MockOpenAIClient(responses_script=many_function_calls)
        connection = _real_connection(client)
        invoke_tool = RecordingInvokeTool(results=['x'] * 10)
        logger = logging.getLogger('test_openai_chat_with_tools_cap')

        with self.assertLogs(logger, level='WARNING') as captured:
            connection.chat_with_tools(
                input_messages=[],
                tools=sample_openai_tools(),
                instructions='sys',
                invoke_tool=invoke_tool,
                max_tool_call_rounds=3,
                logger=logger,
            )

        # Cap hit after 3 rounds.
        self.assertEqual(len(client.responses_calls), 3)
        self.assertEqual(len(invoke_tool.calls), 3)
        self.assertTrue(any('max_tool_call_rounds' in line for line in captured.output))


if __name__ == '__main__':
    unittest.main()
