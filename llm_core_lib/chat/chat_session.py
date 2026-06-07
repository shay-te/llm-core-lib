"""``ChatSession`` — drives one chat turn end-to-end.

    response_text = chat_session.run_command(hash_id, command)

History is fetched from :class:`ChatHistoryStore` (never from the
caller), bounded by ``max_chat_history`` (count) + ``max_chat_tokens``
(token estimate). The conversation's stored ``kind`` is the
dispatch key — the same handler reads and writes its turns, so the
on-wire shape never drifts.

Historical-context replay shape
-------------------------------
When a ``history_replayer`` callable is provided at construction the
PRIOR turns (everything BEFORE the new user prompt) are rebuilt from
the host's compact replay projection — ``[{role, text, llm_payload},
...]`` — rather than from raw provider-shaped store rows. Each
replayed dict is flattened into a single plain ``{role, content}``
message; provider-native ``function_call`` / ``function_call_output``
pairs are deliberately DROPPED for historical context because:

  * They reference SDK-internal ``call_id`` handles whose lifetime is
    bounded by a single response; replaying them out-of-loop confuses
    the model into thinking there are unmatched tool calls in flight.
  * The render-path metadata (``llm_payload``) already captures the
    ids the assistant reasoned over — those are folded into the
    assistant text so the next turn sees "I previously did X and got
    these results" without dangling call_ids.

The CURRENT turn's in-flight tool loop continues to use the handler's
provider-shaped path — the SDK still needs matched function_call /
function_call_output pairs to drive its own loop. Replay is read-side
only; the persistence shape never changes.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from llm_core_lib.chat.chat_handler_registry import ChatHandlerRegistry
from llm_core_lib.chat.chat_history_store import ChatHistoryStore, SENDER_USER
from llm_core_lib.chat.history_budget import truncate_to_token_budget
from llm_core_lib.safety.payload_gate import scrub_history_for_llm


# Defaults sized for 8k-context models (system prompt + tool schemas
# + tool results + completion eat most of the window). Raise via
# constructor / yaml on bigger models. 0 disables the token cap.
DEFAULT_MAX_CHAT_HISTORY = 20
DEFAULT_MAX_CHAT_TOKENS = 4000


# Stored-only sidecar keys on ``ConversationMessage.meta_data``. These
# carry the render-path metadata (``refs`` for batched hydration,
# ``llm_payload`` for the next-turn replay summary) and MUST be
# stripped before the dict is re-sent to the LLM — the model would
# otherwise see internal keys it never produced and start echoing /
# arguing over them.
META_KEY_REFS = 'refs'
META_KEY_LLM_PAYLOAD = 'llm_payload'

# Cap how much JSON-encoded llm_payload may inflate a replayed assistant
# turn — without this, multi-turn conversations over large id lists
# silently blow past the token budget.
_REPLAY_PAYLOAD_MAX_CHARS = 1200
_PERSISTENCE_ONLY_META_KEYS = (META_KEY_REFS, META_KEY_LLM_PAYLOAD)


# Replay-shape role discriminators. Mirrors the host's
# ``ChatMessageReplayService`` constants so the read-side projection
# stays a plain dict contract — llm-core-lib must not import the
# admin-side replay service.
_REPLAY_ROLE_USER = 'user'
_REPLAY_ROLE_ASSISTANT = 'assistant'
_REPLAY_ROLE_TOOL = 'tool'


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
        on_tool_result: Optional[Callable[[str, Any], Dict[str, Any]]] = None,
        history_replayer: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    ):
        # ``invoke_tool`` is the host's choke point (auth + gate +
        # scrub). ``max_tool_call_rounds=None`` defers to the
        # connection's own default.
        #
        # ``on_tool_result`` is the host-supplied metadata extractor
        # called after every tool dispatch with ``(tool_name, gated_result)``.
        # It returns ``{'refs': [...], 'llm_payload_fragment': {key: val}}``
        # — both are accumulated for the current turn and merged into
        # the assistant / tool messages persisted at the end. Keeps
        # ``ChatSession`` the single source of truth for "what gets
        # written to the history store" without leaking tool-name
        # awareness into ``llm-core-lib``.
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
        self._on_tool_result = on_tool_result
        # ``history_replayer(hash_id, limit) -> [{role, text, llm_payload}, ...]``
        # When set, drives the historical-context portion of run_command;
        # see module docstring for the compact replay rationale. When
        # None, falls back to the original list_recent_messages +
        # stored_to_input_message path (backward-compat).
        self._history_replayer = history_replayer

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

    def run_command(
        self,
        hash_id: str,
        command: str,
        tools: Optional[list] = None,
    ) -> str:
        """Drive one turn; persist user prompt + every loop message;
        return assistant text.

        ``tools`` overrides the constructor list for this turn (host
        passes per-user filtered subset). ``None`` uses the full list
        registered at construction.
        """
        conversation = self._history_store.conversation_by_hash(hash_id)
        if conversation is None:
            raise ChatSessionNotFound(f'no conversation for hash_id={hash_id!r}')
        handler = self._handler_registry.get(conversation.get('kind'))
        conversation_id = conversation['id']

        input_messages: List[Dict[str, Any]] = self._build_historical_context(
            hash_id=hash_id,
            conversation_id=conversation_id,
            handler=handler,
        )
        user_prompt_msg = handler.build_user_prompt(command)
        input_messages.append(user_prompt_msg)
        # Persist BEFORE the LLM call so a mid-turn crash still leaves
        # the user's prompt recoverable.
        self._history_store.append_message(
            conversation_id,
            sender=SENDER_USER,
            content=handler.summarize_for_storage(user_prompt_msg),
            meta_data=user_prompt_msg,
        )

        bounded = truncate_to_token_budget(input_messages, self._max_chat_tokens)
        # Scrub prior-turn PII (read-path only; storage stays raw so
        # admins see what they typed). The current prompt is the last
        # item and stays raw — admins legitimately look users up by PII.
        scrubbed_for_llm = scrub_history_for_llm(bounded)
        # Snapshot the exact list the connection will see — the
        # handler's diff needs the true "before" image.
        sent_to_connection = list(scrubbed_for_llm)

        # Per-turn accumulators populated by the wrapped invoke_tool
        # below. Keyed by tool call order, but only the merged totals
        # are persisted — the render path treats the whole turn as a
        # single "what entities did this turn touch?" question.
        turn_refs: List[Dict[str, Any]] = []
        turn_llm_payload: Dict[str, Any] = {}

        invoke_tool = self._build_capturing_invoke_tool(
            turn_refs=turn_refs,
            turn_llm_payload=turn_llm_payload,
        )

        with self._connection_factory.get() as connection:
            chat_kwargs = dict(
                input_messages=scrubbed_for_llm,
                tools=tools if tools is not None else self._tools,
                instructions=self._instructions,
                invoke_tool=invoke_tool,
                logger=self._logger,
            )
            if self._max_tool_call_rounds is not None:
                chat_kwargs['max_tool_call_rounds'] = self._max_tool_call_rounds
            response, final_messages = connection.chat_with_tools(**chat_kwargs)

        new_messages = handler.diff_new_messages(
            input_messages_before=sent_to_connection,
            final_messages=final_messages,
        )
        for message in new_messages:
            meta_data = self._meta_data_with_tool_metadata(
                message=message,
                sender=handler.sender_for(message),
                turn_refs=turn_refs,
                turn_llm_payload=turn_llm_payload,
            )
            self._history_store.append_message(
                conversation_id,
                sender=handler.sender_for(message),
                content=handler.summarize_for_storage(message),
                meta_data=meta_data,
            )

        return handler.extract_response_text(response)

    # ---- internal helpers ----------------------------------------

    def _build_capturing_invoke_tool(
        self,
        turn_refs: List[Dict[str, Any]],
        turn_llm_payload: Dict[str, Any],
    ) -> Callable[[str, dict], Any]:
        """Wrap the host's ``invoke_tool`` so each call's result is
        also handed to the metadata extractor and accumulated.

        The wrapped function preserves the host callback's return
        value verbatim — the LLM gets exactly what the gate produced;
        the capture is a side-channel for persistence.
        """
        host_invoke = self._invoke_tool
        on_tool_result = self._on_tool_result
        if on_tool_result is None:
            return host_invoke

        def capturing_invoke_tool(name: str, tool_kwargs: dict):
            result = host_invoke(name, tool_kwargs)
            try:
                fragment = on_tool_result(name, result) or {}
            except Exception:  # noqa: BLE001 — never let metadata extraction break a turn
                self._logger.exception(
                    'on_tool_result raised for tool %r; skipping metadata capture', name,
                )
                fragment = {}
            refs = fragment.get('refs') or []
            payload = fragment.get('llm_payload_fragment') or {}
            if refs:
                turn_refs.extend(refs)
            if payload:
                turn_llm_payload.update(payload)
            return result

        return capturing_invoke_tool

    def _build_historical_context(
        self,
        hash_id: str,
        conversation_id: Any,
        handler: Any,
    ) -> List[Dict[str, Any]]:
        """Project prior turns into the LLM input list.

        When a ``history_replayer`` is wired, prefer the compact
        ``{role, text, llm_payload}`` shape — see module docstring.
        Otherwise fall back to the legacy stored-meta-data path so
        callers without a replay service keep working.
        """
        if self._history_replayer is not None:
            try:
                replayed = self._history_replayer(
                    hash_id, self._max_chat_history,
                ) or []
            except Exception:  # noqa: BLE001 — never let replay break the turn
                self._logger.exception(
                    'history_replayer raised for hash_id=%r; '
                    'falling back to raw store projection', hash_id,
                )
                replayed = None
            if replayed is not None:
                messages: List[Dict[str, Any]] = []
                for entry in replayed:
                    projected = self._replay_message_to_input_shape(entry)
                    if projected is not None:
                        messages.append(projected)
                return messages

        history = self._history_store.list_recent_messages(
            conversation_id, self._max_chat_history,
        )
        return [
            handler.stored_to_input_message(
                self._strip_persistence_keys(message['meta_data']),
            )
            for message in history
        ]

    @staticmethod
    def _replay_message_to_input_shape(
        replay_msg: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Convert a single ``{role, text, llm_payload}`` replay dict
        into the provider-agnostic ``{role, content}`` shape every
        handler treats as plain text.

        Tool-role entries are DROPPED — their ``llm_payload`` belongs
        to the assistant turn that produced them, and the upstream
        replay service already folds them out / collapses them. A
        bare tool message replayed in isolation has no matching
        ``function_call`` sibling, so the SDK would treat it as an
        orphan and either error or hallucinate around it.

        For assistant messages, the ``llm_payload`` dict is appended
        to the text as a JSON-ish suffix so the model can read back
        the ids it acted on last turn without those ids being
        re-hydrated from the live DB.
        """
        if not isinstance(replay_msg, dict):
            return None
        role = replay_msg.get('role')
        text = replay_msg.get('text') or ''
        if role == _REPLAY_ROLE_USER:
            return {'role': _REPLAY_ROLE_USER, 'content': text}
        if role == _REPLAY_ROLE_ASSISTANT:
            llm_payload = replay_msg.get('llm_payload') or {}
            if llm_payload:
                payload_str = json.dumps(llm_payload, sort_keys=True)
                # Cap injected payload so long-list turns can't inflate
                # next-turn context past the token budget; truncation
                # mark tells the model the list was bigger than shown.
                if len(payload_str) > _REPLAY_PAYLOAD_MAX_CHARS:
                    payload_str = payload_str[:_REPLAY_PAYLOAD_MAX_CHARS] + '…(truncated)'
                content = f'{text}\n\n[Tool results: {payload_str}]' if text else f'[Tool results: {payload_str}]'
            else:
                content = text
            return {'role': _REPLAY_ROLE_ASSISTANT, 'content': content}
        # Tool role + anything unknown → drop. See docstring.
        return None

    @staticmethod
    def _strip_persistence_keys(stored_meta_data: Dict[str, Any]) -> Dict[str, Any]:
        """Drop sidecar keys (``refs`` / ``llm_payload``) before the
        handler re-projects the stored row into the LLM input list.

        Returns a shallow copy when stripping is needed so the row
        cached upstream is not mutated; returns the original when
        clean for the fast path.
        """
        if not isinstance(stored_meta_data, dict):
            return stored_meta_data
        if not any(key in stored_meta_data for key in _PERSISTENCE_ONLY_META_KEYS):
            return stored_meta_data
        return {
            key: value for key, value in stored_meta_data.items()
            if key not in _PERSISTENCE_ONLY_META_KEYS
        }

    @staticmethod
    def _meta_data_with_tool_metadata(
        message: Dict[str, Any],
        sender: str,
        turn_refs: List[Dict[str, Any]],
        turn_llm_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a copy of ``message`` with the turn's tool-metadata
        merged into ``llm_payload`` + ``refs`` keys.

        Both accumulators are turn-wide — every assistant / tool
        message persisted in this turn carries the same merged view.
        The render path reads from any of them, so an admin scrolling
        through the message list sees the same hydrated entities the
        assistant reasoned over.

        Mutates a shallow copy of the input dict (not the original)
        so the in-memory provider-shaped message used elsewhere is
        unchanged.
        """
        if not turn_refs and not turn_llm_payload:
            return message
        merged = dict(message)
        if turn_refs:
            # Dedupe by (type, id) — the same entity may be returned
            # by multiple tools in the same turn.
            seen = set()
            deduped: List[Dict[str, Any]] = []
            for ref in turn_refs:
                key = (ref.get('type'), ref.get('id'))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append({'type': ref['type'], 'id': ref['id']})
            merged[META_KEY_REFS] = deduped
        if turn_llm_payload:
            merged[META_KEY_LLM_PAYLOAD] = dict(turn_llm_payload)
        return merged

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
