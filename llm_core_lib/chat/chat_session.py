"""``ChatSession`` — drives one chat turn end-to-end.

    response_text = chat_session.run_command(hash_id, command)

History is fetched from :class:`ChatHistoryStore` (never from the
caller), bounded by ``max_chat_history`` (count) + ``max_chat_tokens``
(token estimate). The conversation's stored ``kind`` is the
dispatch key — the same handler reads and writes its turns, so the
on-wire shape never drifts.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from llm_core_lib.chat.chat_handler_registry import ChatHandlerRegistry
from llm_core_lib.chat.chat_history_store import ChatHistoryStore, SENDER_USER
from llm_core_lib.chat.history_budget import truncate_to_token_budget
from llm_core_lib.safety.payload_gate import scrub_history_for_llm


# Host can override via constructor (typically from
# ``core_lib.llm.max_chat_history`` / ``...max_chat_tokens``).
DEFAULT_MAX_CHAT_HISTORY = 50
# 0 disables the token cap (message-count cap still applies). 12000
# leaves headroom for completion on a 16k-context model.
DEFAULT_MAX_CHAT_TOKENS = 12000


class ChatSessionNotFound(LookupError):
    """Distinguishable so the host's web layer can map it to 404."""


class ChatSession(object):

    def __init__(
        self,
        history_store: ChatHistoryStore,
        connection_factory: Any,
        handler_registry: ChatHandlerRegistry,
        instructions: str,
        tools: list,
        invoke_tool: Callable[[str, dict], Any],
        max_chat_history: int = DEFAULT_MAX_CHAT_HISTORY,
        max_chat_tokens: int = DEFAULT_MAX_CHAT_TOKENS,
        max_tool_call_rounds: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # ``invoke_tool`` is the host's choke point (auth + gate +
        # scrub). ``max_tool_call_rounds=None`` defers to the
        # connection's own default.
        self._history_store = history_store
        self._connection_factory = connection_factory
        self._handler_registry = handler_registry
        self._instructions = instructions
        self._tools = tools
        self._invoke_tool = invoke_tool
        self._max_chat_history = max_chat_history
        self._max_chat_tokens = max_chat_tokens
        self._max_tool_call_rounds = max_tool_call_rounds
        self._logger = logger or logging.getLogger(__name__)

    # ---- public API: intentionally minimal ------------------------

    def create_conversation(
        self,
        owner_user_id: int,
        kind: str,
        name: str = 'New chat',
        scope_meta_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Fail fast on unknown kind — registry.get raises.
        self._handler_registry.get(kind)
        return self._history_store.create_conversation(
            owner_user_id=owner_user_id,
            kind=kind,
            name=name,
            scope_meta_data=scope_meta_data,
        )

    def run_command(self, hash_id: str, command: str) -> str:
        """Drive one turn; persist user prompt + every loop message;
        return assistant text."""
        conversation = self._history_store.conversation_by_hash(hash_id)
        if conversation is None:
            raise ChatSessionNotFound(f'no conversation for hash_id={hash_id!r}')
        kind = conversation.get('kind')
        handler = self._handler_registry.get(kind)
        conversation_id = conversation['id']

        history = self._history_store.list_recent_messages(
            conversation_id, self._max_chat_history,
        )
        # Rebuild the provider-shape input list from stored meta_data.
        input_messages: List[Dict[str, Any]] = [
            handler.stored_to_input_message(message['meta_data'])
            for message in history
        ]
        # Append the new user prompt (also persisted below).
        user_prompt_msg = handler.build_user_prompt(command)
        input_messages.append(user_prompt_msg)
        # Persist the user prompt BEFORE calling the LLM so a crash
        # mid-turn still leaves the user's last command recoverable.
        self._history_store.append_message(
            conversation_id,
            sender=SENDER_USER,
            content=handler.summarize_for_storage(user_prompt_msg),
            meta_data=user_prompt_msg,
        )

        # Token-budget truncation first — drops oldest messages
        # until the input list fits ``max_chat_tokens``. The
        # message-count cap already bounded the fetch; this is the
        # safety net for "50 messages but each one is huge".
        bounded = truncate_to_token_budget(input_messages, self._max_chat_tokens)
        # Scrub PII out of prior-turn messages before re-sending to
        # the LLM. The last message (the just-built user prompt)
        # stays raw — admins legitimately type PII to look users up.
        # Storage stays raw too; the scrub is read-path only.
        scrubbed_for_llm = scrub_history_for_llm(bounded)
        # Snapshot the EXACT list the connection will see, so the
        # handler's diff has the true "before" image — even if a
        # future connection rewrites the head in place.
        sent_to_connection = list(scrubbed_for_llm)

        with self._connection_factory.get() as connection:
            chat_kwargs = dict(
                input_messages=scrubbed_for_llm,
                tools=self._tools,
                instructions=self._instructions,
                invoke_tool=self._invoke_tool,
                logger=self._logger,
            )
            if self._max_tool_call_rounds is not None:
                chat_kwargs['max_tool_call_rounds'] = self._max_tool_call_rounds
            response, final_messages = connection.chat_with_tools(**chat_kwargs)

        # Diff against the snapshot (not against a slice of the
        # post-call list — that would lie if the connection ever
        # rewrites the head).
        new_messages = handler.diff_new_messages(
            input_messages_before=sent_to_connection,
            final_messages=final_messages,
        )
        for message in new_messages:
            self._history_store.append_message(
                conversation_id,
                sender=handler.sender_for(message),
                content=handler.summarize_for_storage(message),
                meta_data=message,
            )

        return handler.extract_response_text(response)

    def list_conversations(
        self, owner_user_id: int, scope_meta_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self._history_store.list_conversations(
            owner_user_id, scope_meta_data=scope_meta_data,
        )

    def get_conversation_messages(
        self, hash_id: str, limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        conversation = self._history_store.conversation_by_hash(hash_id)
        if conversation is None:
            raise ChatSessionNotFound(f'no conversation for hash_id={hash_id!r}')
        effective_limit = limit if limit is not None else self._max_chat_history
        return self._history_store.list_recent_messages(
            conversation['id'], effective_limit,
        )

    def rename_conversation(self, hash_id: str, new_name: str) -> None:
        if self._history_store.conversation_by_hash(hash_id) is None:
            raise ChatSessionNotFound(f'no conversation for hash_id={hash_id!r}')
        self._history_store.rename_conversation(hash_id, new_name)

    def delete_conversation(self, hash_id: str) -> None:
        if self._history_store.conversation_by_hash(hash_id) is None:
            raise ChatSessionNotFound(f'no conversation for hash_id={hash_id!r}')
        self._history_store.delete_conversation(hash_id)
