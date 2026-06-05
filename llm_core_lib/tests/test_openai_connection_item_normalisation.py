"""``OpenAiConnection`` — items appended to ``input_messages`` must be dicts.

OpenAI's Responses SDK returns Pydantic-shaped objects whose
attribute access works locally but breaks downstream: chat handlers
call ``message.get(...)`` and the conversation store serializes
``meta_data`` to a JSON column. Either step fails on raw SDK
objects. The connection therefore normalises every appended item to
a plain dict via ``_item_to_dict``.

Per the workspace-wide "one TestCase per file" rule, this file owns
exactly one TestCase.
"""
from __future__ import annotations

import unittest

from llm_core_lib.connections.openai_connection import (
    OpenAiConnection,
    _item_to_dict,
    format_tool_result_for_llm,
)


class _PydanticLike(object):
    """Stand-in for an SDK Pydantic object; exposes ``model_dump``."""

    def __init__(self, data):
        self._data = data
        # Attribute access mirrors SDK behaviour so attempts to
        # ``item.type`` in the connection still work.
        for k, v in data.items():
            setattr(self, k, v)

    def model_dump(self):
        return dict(self._data)


class _FunctionCallItem(_PydanticLike):
    pass


class _MessageItem(_PydanticLike):
    pass


class _ResponseStub(object):
    def __init__(self, output):
        self.output = output
        self.error = None


class _ClientStub(object):
    """Returns one function_call then one terminal message, mirroring
    a realistic two-step tool conversation."""

    def __init__(self):
        self._calls = 0

    class _Responses(object):
        def __init__(self, parent):
            self._parent = parent

        def create(self, **_):
            self._parent._calls += 1
            if self._parent._calls == 1:
                return _ResponseStub([
                    _FunctionCallItem({
                        'type': 'function_call',
                        'name': 'list_users',
                        'arguments': '{"q":"jane"}',
                        'call_id': 'c1',
                    }),
                ])
            return _ResponseStub([
                _MessageItem({
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'done'}],
                }),
            ])

    @property
    def responses(self):
        return _ClientStub._Responses(self)


def _make_connection():
    return OpenAiConnection(
        client=_ClientStub(),
        model_id='gpt-4o',
        vision_model_id='gpt-4o',
        embedding_model='text-embedding-3-small',
    )


class TestOpenAiConnectionItemNormalisation(unittest.TestCase):

    def test_item_to_dict_passes_dicts_through(self):
        d = {'type': 'message', 'role': 'assistant'}
        self.assertIs(_item_to_dict(d), d)

    def test_item_to_dict_uses_model_dump_when_available(self):
        item = _PydanticLike({'type': 'function_call', 'name': 'x'})
        self.assertEqual(
            _item_to_dict(item),
            {'type': 'function_call', 'name': 'x'},
        )

    def test_function_call_item_persisted_as_dict_not_sdk_object(self):
        # The whole point of this test file: after one round-trip
        # through chat_with_tools the function_call item in
        # input_messages MUST be a dict that handlers can ``.get()`` on
        # and SQLAlchemy can JSON-serialise.
        connection = _make_connection()
        input_messages = []
        invocations = []

        def invoke_tool(name, kwargs):
            invocations.append((name, kwargs))
            return {'status': 'ok'}

        _, final_messages = connection.chat_with_tools(
            input_messages=input_messages,
            tools=[],
            instructions='sys',
            invoke_tool=invoke_tool,
        )

        # The loop ran one tool then a terminal message. Final list
        # should contain: function_call dict, function_call_output dict,
        # assistant message dict.
        self.assertEqual(len(final_messages), 3)
        for message in final_messages:
            self.assertIsInstance(message, dict)
            # ``.get(...)`` works on dicts; SDK-object regressions
            # would fail this with AttributeError.
            self.assertIsNotNone(message.get('type'))

    def test_terminal_assistant_message_is_appended_to_input_messages(self):
        # The chat persistence layer relies on every new turn item
        # appearing in the returned input_messages list. Before the
        # fix, the assistant ``message`` item was logged but never
        # appended, so history round-trips silently dropped the reply.
        connection = _make_connection()
        _, final_messages = connection.chat_with_tools(
            input_messages=[],
            tools=[],
            instructions='sys',
            invoke_tool=lambda name, kwargs: {'status': 'ok'},
        )
        last = final_messages[-1]
        self.assertEqual(last.get('type'), 'message')
        self.assertEqual(last.get('role'), 'assistant')


if __name__ == '__main__':
    unittest.main()
