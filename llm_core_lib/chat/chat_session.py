"""``ChatSession`` — drives one chat turn end-to-end.

The orchestrator lives here so the rule "history is server-side; the
frontend only sends a prompt" is enforced in one place. Its
``run_command`` signature is intentionally tiny:

    response_text = chat_session.run_command(hash_id, command)

Everything else (history fetch + handler dispatch + tool-loop +
persistence + text extraction) is hidden.

Hard-coded properties:

  * History is fetched from :class:`ChatHistoryStore`, never from
    the caller. The connection's ``chat_with_tools`` still accepts
    ``input_messages`` because the connection is a thin SDK adapter
    — but only this orchestrator ever calls it.
  * The recent-history fetch is bounded by ``max_chat_history`` so
    provider-side exposure stays O(1) per turn regardless of how
    long-lived the conversation is.
  * The conversation's stored ``kind`` is the dispatch key — the
    same handler that wrote a turn reads it back, so the on-wire
    shape never drifts between writer and reader.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from llm_core_lib.chat.chat_handler import ChatHandler
from llm_core_lib.chat.chat_handler_registry import ChatHandlerRegistry
from llm_core_lib.chat.chat_history_store import ChatHistoryStore, SENDER_USER
from llm_core_lib.safety.payload_gate import scrub_history_for_llm


# Default cap on the recent-history fetch. The host can override via
# the constructor (typically from ``core_lib.llm.chat.max_chat_history``
# in its Hydra config). Picked high enough that normal admin chats
# don't truncate but low enough that a malicious/long-lived
# conversation can't blow up the provider request size.
DEFAULT_MAX_CHAT_HISTORY = 50


class ChatSessionNotFound(LookupError):
    """The hash id didn't resolve to a conversation in the store.

    Distinguishable from other ``LookupError``s so the host's web
    layer can map it to a 404 if desired (otherwise the choke-point
    sanitizer envelopes it like any other failure)."""


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
        max_tool_call_rounds: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """All collaborators in once at composition root.

        Args:
            history_store: persistence boundary.
            connection_factory: anything whose ``.get()`` returns a
                context manager whose ``__enter__`` yields a
                Connection (matching this project's
                ``OpenAiConnection`` / ``BedrockConnection`` shape).
            handler_registry: dispatch by conversation ``kind``.
            instructions: system-prompt text passed to the connection.
            tools: function-tool schema list.
            invoke_tool: callback ``(name, kwargs) -> result`` — the
                caller's choke point (authorization + gate + scrub).
            max_chat_history: cap on the recent-history fetch.
            max_tool_call_rounds: forwarded to the connection's loop;
                ``None`` uses the connection's own default.
            logger: scoped logger; defaults to the module logger.
        """
        self._history_store = history_store
        self._connection_factory = connection_factory
        self._handler_registry = handler_registry
        self._instructions = instructions
        self._tools = tools
        self._invoke_tool = invoke_tool
        self._max_chat_history = max_chat_history
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
        """Open a new conversation; return its client-facing hash id."""
        # Fail fast on unknown kind — better than letting an orphan
        # conversation get created that no handler can ever read. The
        # registry's ``get`` raises ``UnknownChatKindError``.
        self._handler_registry.get(kind)
        return self._history_store.create_conversation(
            owner_user_id=owner_user_id,
            kind=kind,
            name=name,
            scope_meta_data=scope_meta_data,
        )

    def run_command(self, hash_id: str, command: str) -> str:
        """Drive one chat turn end-to-end.

        Returns:
            The assistant's final text reply (provider-extracted by
            the matching handler). The full new exchange is already
            persisted in the store by the time this returns.
        """
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

        # Snapshot the prefix length so the handler can diff the new
        # tail later. ``input_messages`` is what the connection sees
        # going IN; ``final_messages`` is what it returns.
        prefix_len = len(input_messages)

        # Scrub PII out of prior-turn messages before re-sending to
        # the LLM. The last message (the just-built user prompt)
        # stays raw — admins legitimately type PII to look users up.
        # Storage stays raw too; the scrub is read-path only.
        scrubbed_for_llm = scrub_history_for_llm(input_messages)

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

        # Persist everything the loop added on top of our snapshot.
        new_messages = handler.diff_new_messages(
            input_messages_before=input_messages[:prefix_len],
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
        """Return the stored messages for a conversation, oldest-first.

        Used by the UI to load a past chat. The default ``limit`` is
        the session's ``max_chat_history`` so the same bound applies
        to UI and to LLM-input fetches.
        """
        conversation = self._history_store.conversation_by_hash(hash_id)
        if conversation is None:
            raise ChatSessionNotFound(f'no conversation for hash_id={hash_id!r}')
        effective_limit = limit if limit is not None else self._max_chat_history
        return self._history_store.list_recent_messages(
            conversation['id'], effective_limit,
        )

    def rename_conversation(self, hash_id: str, new_name: str) -> None:
        # Lookup-first lets the host's verifier verify cross-tenant
        # access before the write; the store implementation
        # presumably also enforces but defense-in-depth.
        conversation = self._history_store.conversation_by_hash(hash_id)
        if conversation is None:
            raise ChatSessionNotFound(f'no conversation for hash_id={hash_id!r}')
        self._history_store.rename_conversation(hash_id, new_name)

    @property
    def handler_registry(self) -> ChatHandlerRegistry:
        # Exposed so the host can use the registry's ``has`` /
        # ``kinds`` introspection (the registry is otherwise injected
        # at construct time).
        return self._handler_registry
