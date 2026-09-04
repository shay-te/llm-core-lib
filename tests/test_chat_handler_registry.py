"""``ChatHandlerRegistry`` — register / get / has + the missing-kind error.

Per the workspace-wide "one TestCase per file" rule, this file owns
exactly one TestCase.
"""
from __future__ import annotations

import unittest

from llm_core_lib.chat.chat_handler import ChatHandler
from llm_core_lib.chat.chat_handler_registry import (
    ChatHandlerRegistry,
    UnknownChatKindError,
)


class _Stub(ChatHandler):
    KIND = 'stub-kind'

    def build_user_prompt(self, command):
        return {'role': 'user', 'content': command}

    def stored_to_input_message(self, stored_meta_data):
        return stored_meta_data

    def diff_new_messages(self, input_messages_before, final_messages):
        return list(final_messages[len(input_messages_before):])

    def summarize_for_storage(self, message):
        return str(message)

    def extract_response_text(self, response):
        return ''

    def sender_for(self, message):
        return 'user'


class _Unnamed(_Stub):
    KIND = ''


class TestChatHandlerRegistry(unittest.TestCase):

    def test_register_and_get_round_trips_by_kind(self):
        registry = ChatHandlerRegistry()
        handler = _Stub()
        registry.register(handler)
        self.assertIs(registry.get('stub-kind'), handler)

    def test_get_unknown_kind_raises_with_available_list(self):
        registry = ChatHandlerRegistry()
        registry.register(_Stub())
        with self.assertRaises(UnknownChatKindError) as cm:
            registry.get('nope')
        # The error message names what IS registered so the operator
        # can correlate the kind mismatch quickly.
        self.assertIn('stub-kind', str(cm.exception))
        self.assertIn('nope', str(cm.exception))

    def test_register_handler_without_kind_is_rejected(self):
        # Failing fast on register catches typo-style bugs at boot
        # rather than at first-call time.
        registry = ChatHandlerRegistry()
        with self.assertRaises(ValueError):
            registry.register(_Unnamed())

    def test_re_register_overwrites(self):
        # Useful for tests that swap a stub in for the real handler.
        registry = ChatHandlerRegistry()
        first = _Stub()
        second = _Stub()
        registry.register(first)
        registry.register(second)
        self.assertIs(registry.get('stub-kind'), second)

if __name__ == '__main__':
    unittest.main()
