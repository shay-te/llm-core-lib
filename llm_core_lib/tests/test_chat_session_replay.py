"""``ChatSession`` history-replayer wiring.

Verifies that when a ``history_replayer`` is provided at construction
the connection receives the COMPACT ``{role, content}`` projection
for prior turns (with assistant ``llm_payload`` folded into the text)
instead of the raw provider-shaped store rows. The CURRENT turn's
user prompt is appended via the handler's normal path and reaches the
connection unchanged.

Per the workspace-wide "one TestCase per file" rule, this file owns
exactly one TestCase.
"""
from __future__ import annotations

import json
import unittest

from llm_core_lib.chat.chat_handler_registry import ChatHandlerRegistry
from llm_core_lib.chat.chat_history_store import (
    SENDER_USER,
    ChatHistoryStore,
)
from llm_core_lib.chat.chat_session import ChatSession
from llm_core_lib.chat.openai_chat_handler import KIND_OPENAI, OpenAiChatHandler


class _InMemoryStore(ChatHistoryStore):
    """Minimal store. The replayer makes ``list_recent_messages``
    irrelevant for the historical-context path, but other methods
    (``conversation_by_hash``, ``append_message``) still drive."""

    def __init__(self):
        self._conversations = {}
        self._messages = {}
        self._next_conv_id = 1
        self._next_msg_id = 1
        self.appended = []
        self.list_recent_calls = []  # for the negative assertion

    def create_conversation(self, owner_user_id, kind, name, scope_meta_data=None):
        hash_id = f'h{self._next_conv_id}'
        conv_id = self._next_conv_id
        self._next_conv_id += 1
        self._conversations[hash_id] = {
            'id': conv_id,
            'kind': kind,
            'name': name,
            'meta_data': {'kind': kind, 'name': name, **(scope_meta_data or {})},
        }
        self._messages[conv_id] = []
        return hash_id

    def conversation_by_hash(self, hash_id):
        return self._conversations.get(hash_id)

    def list_recent_messages(self, conversation_id, limit):
        self.list_recent_calls.append((conversation_id, limit))
        messages = self._messages.get(conversation_id, [])
        return messages[-limit:] if limit else list(messages)

    def append_message(self, conversation_id, sender, content, meta_data):
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        row = {
            'id': msg_id,
            'sender_user_id': 0 if sender != SENDER_USER else 1,
            'content': content,
            'meta_data': meta_data,
        }
        self._messages.setdefault(conversation_id, []).append(row)
        self.appended.append((sender, content, meta_data))
        return msg_id

    def list_conversations(self, owner_user_id, scope_meta_data=None):
        return [
            {**conv, 'hash_id': hid}
            for hid, conv in self._conversations.items()
        ]

    def rename_conversation(self, hash_id, new_name):
        return None

    def delete_conversation(self, hash_id):
        return None


class _ResponseStub(object):
    def __init__(self, output):
        self.output = output


class _MessageStub(object):
    def __init__(self, text):
        self.type = 'message'
        self.content = [_TextContent(text)]


class _TextContent(object):
    def __init__(self, text):
        self.text = text


class _RecordingConnection(object):
    """Captures the exact input_messages handed to ``chat_with_tools``
    so the test can prove the replayer's compact shape reaches the
    connection (not the raw provider-shaped store rows)."""

    def __init__(self):
        self.received_input_messages = None

    def chat_with_tools(self, *, input_messages, tools, instructions,
                        invoke_tool, logger, max_tool_call_rounds=None):
        self.received_input_messages = [
            dict(m) if isinstance(m, dict) else m for m in input_messages
        ]
        input_messages.append({'role': 'assistant', 'content': 'final'})
        return _ResponseStub(output=[_MessageStub('final')]), input_messages


class _ConnectionFactoryStub(object):
    def __init__(self, connection):
        self._connection = connection

    def get(self):
        return self

    def __enter__(self):
        return self._connection

    def __exit__(self, *args):
        return None


class TestChatSessionReplay(unittest.TestCase):

    def _make_session(self, replayer):
        store = _InMemoryStore()
        connection = _RecordingConnection()
        registry = ChatHandlerRegistry()
        registry.register(OpenAiChatHandler())

        session = ChatSession(
            history_store=store,
            connection_factory=_ConnectionFactoryStub(connection),
            handler_registry=registry,
            instructions='sys',
            tools=[],
            invoke_tool=lambda name, kwargs: {'status': 'ok'},
            max_chat_history=10,
            history_replayer=replayer,
        )
        return session, store, connection

    def test_replayer_drives_historical_context_shape(self):
        # Canned replay output: one prior user prompt + one prior
        # assistant turn carrying an ``llm_payload`` (the ids that
        # tools surfaced last turn). A tool-role entry that MUST be
        # dropped from the replay projection.
        canned = [
            {'role': 'user', 'text': 'hello', 'llm_payload': {}},
            {'role': 'tool', 'text': 'list_packages output', 'llm_payload': {}},
            {
                'role': 'assistant',
                'text': 'found 2 packages',
                'llm_payload': {'package_ids': [10, 11]},
            },
        ]
        replayer_calls = []

        def replayer(hash_id, limit):
            replayer_calls.append((hash_id, limit))
            return canned

        session, store, connection = self._make_session(replayer)
        hash_id = session.create_conversation(owner_user_id=1, kind=KIND_OPENAI)
        # Drop some raw rows in the store too — they MUST NOT appear
        # in what the connection receives, proving the replayer (not
        # ``list_recent_messages``) drove the historical context.
        store.append_message(
            conversation_id=store.conversation_by_hash(hash_id)['id'],
            sender=SENDER_USER,
            content='SHOULD NOT APPEAR',
            meta_data={
                'role': 'user', 'content': 'SHOULD NOT APPEAR',
                'type': 'message',
            },
        )

        session.run_command(hash_id, 'next prompt')

        received = connection.received_input_messages
        self.assertIsNotNone(received)

        # The replayer was actually called for this turn.
        self.assertEqual(len(replayer_calls), 1)
        self.assertEqual(replayer_calls[0][0], hash_id)

        # The store's raw row must NOT be in the LLM input list.
        for msg in received:
            self.assertNotEqual(
                msg.get('content'), 'SHOULD NOT APPEAR',
                msg='raw store row leaked past the replayer',
            )

        # The current-turn user prompt is last (handler path).
        self.assertEqual(received[-1], {'role': 'user', 'content': 'next prompt'})

        # Historical user prompt comes through unmodified.
        self.assertIn({'role': 'user', 'content': 'hello'}, received)

        # Tool-role entry was dropped — no message contains the tool
        # text, since orphan tool messages have no matching
        # function_call sibling once the replay shape collapses them.
        for msg in received:
            self.assertNotIn(
                'list_packages output', str(msg.get('content', '')),
                msg='tool-role replay entry leaked into the LLM input',
            )

        # Assistant turn carries the llm_payload folded into the text.
        assistant_messages = [
            m for m in received
            if isinstance(m, dict) and m.get('role') == 'assistant'
        ]
        self.assertEqual(len(assistant_messages), 1)
        content = assistant_messages[0]['content']
        self.assertIn('found 2 packages', content)
        self.assertIn('Tool results', content)
        # Payload is serialized JSON so the model can parse it.
        payload_str = json.dumps({'package_ids': [10, 11]}, sort_keys=True)
        self.assertIn(payload_str, content)

    def test_no_replayer_falls_back_to_store_projection(self):
        # When no replayer is given, the legacy
        # ``list_recent_messages`` + ``stored_to_input_message`` path
        # must still drive (backward-compat).
        session, store, connection = self._make_session(replayer=None)
        hash_id = session.create_conversation(owner_user_id=1, kind=KIND_OPENAI)
        # Seed a prior turn via the store directly.
        store.append_message(
            conversation_id=store.conversation_by_hash(hash_id)['id'],
            sender=SENDER_USER,
            content='earlier',
            meta_data={'role': 'user', 'content': 'earlier', 'type': 'message'},
        )

        session.run_command(hash_id, 'now')

        received = connection.received_input_messages
        # Legacy projection: stored row reaches the LLM input list.
        self.assertTrue(any(
            isinstance(m, dict) and m.get('content') == 'earlier'
            for m in received
        ))
        # And the store was consulted (no replayer to short-circuit it).
        self.assertTrue(store.list_recent_calls)

    def test_replayer_exception_falls_back_to_store(self):
        # Defensive: a thrown replayer must not break a turn — fall
        # back to the store path so chat keeps working.
        def bad_replayer(hash_id, limit):
            raise RuntimeError('replay broke')

        session, store, connection = self._make_session(replayer=bad_replayer)
        hash_id = session.create_conversation(owner_user_id=1, kind=KIND_OPENAI)
        store.append_message(
            conversation_id=store.conversation_by_hash(hash_id)['id'],
            sender=SENDER_USER,
            content='fallback-prompt',
            meta_data={
                'role': 'user', 'content': 'fallback-prompt', 'type': 'message',
            },
        )

        # Must not raise.
        session.run_command(hash_id, 'now')

        received = connection.received_input_messages
        self.assertTrue(any(
            isinstance(m, dict) and m.get('content') == 'fallback-prompt'
            for m in received
        ))


if __name__ == '__main__':
    unittest.main()
