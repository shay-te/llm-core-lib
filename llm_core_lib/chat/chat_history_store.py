"""``ChatHistoryStore`` — the persistence boundary chat sessions talk to.

A ``Protocol`` not a class so ``llm-core-lib`` does NOT depend on any
specific storage backend (``conversation-core-lib``, a noop, an
in-memory test stub). The host wires whichever implementation it owns.

The protocol is intentionally tiny — orchestration logic lives in
:class:`ChatSession`; the store just persists and reads dicts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# A ``StoredMessage`` is the dict shape :class:`ChatSession` consumes
# back from :meth:`ChatHistoryStore.list_recent_messages`. Keys:
#
#   * ``id``           int    — unique message id (store-provided).
#   * ``sender_user_id`` int  — who wrote it (admin id or the
#                                synthetic LLM-bot id, see SENDER_*).
#   * ``content``      str    — short human-readable summary
#                                (truncated to fit the underlying
#                                ``content`` column's char limit).
#   * ``meta_data``    dict   — the provider-shaped original message
#                                the handler will replay verbatim.
#
# Stored as a plain dict for cheap interop — no class import needed.


# Sender markers — :class:`ChatSession` uses them to distinguish who
# wrote each stored message when handing them to handlers. Stores must
# preserve the value the session passed in unchanged.
SENDER_USER = 'user'        # the admin (or end-user)
SENDER_ASSISTANT = 'assistant'  # the LLM-bot
SENDER_TOOL = 'tool'        # tool-call dispatch / tool-result rows


class ChatHistoryStore(object):
    """Persistence interface — implement these four methods to plug in
    any storage backend.

    The class is documented as concrete (not ``Protocol``-typed) so
    runtime ``isinstance`` checks work in test fixtures; subclassing is
    the recommended pattern but duck-typed implementations also pass.
    """

    def create_conversation(
        self,
        owner_user_id: int,
        kind: str,
        name: str,
        scope_meta_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Open a new conversation and return its client-facing hash id.

        Args:
            owner_user_id: the human user who owns the conversation
                (admin id for the admin chat). The store decides how to
                model the "other party" (LLM bot synthetic id).
            kind: dispatch key (e.g. ``'main_chat_openai'``); also
                stored in conversation meta_data so a future load can
                resolve the right :class:`ChatHandler`.
            name: human label (e.g. ``'New chat'``); the store stores
                it in conversation meta_data so it round-trips on list.
            scope_meta_data: extra keys the host wants on the
                conversation row (e.g. ``{'project_id': 7}`` so the
                host can verify cross-tenant access on every read).

        Returns:
            A short stable identifier (UUID4 hex) the client uses to
            address this conversation in every subsequent request.
            Never the internal DB id (which is implementation detail).
        """
        raise NotImplementedError

    def conversation_by_hash(self, hash_id: str) -> Optional[Dict[str, Any]]:
        """Resolve ``hash_id`` to a conversation row dict.

        Returns ``None`` if no conversation matches — callers MUST
        treat that as "not found" and surface a generic error (never a
        distinguishable "wrong id" — that would let the LLM probe).

        Expected dict keys (the store decides the shape internally,
        but :class:`ChatSession` reads exactly these):

          * ``id``        int  — internal DB id (passed back to
                                 ``list_recent_messages`` and
                                 ``append_message``).
          * ``kind``      str  — value from :meth:`create_conversation`.
          * ``name``      str  — value from :meth:`create_conversation`.
          * ``meta_data`` dict — whatever the host stuffed in
                                 ``scope_meta_data`` plus the store's
                                 own bookkeeping.
        """
        raise NotImplementedError

    def list_recent_messages(
        self, conversation_id: int, limit: int,
    ) -> List[Dict[str, Any]]:
        """Return up to ``limit`` most-recent messages in chronological order.

        Order is oldest-first (the LLM's expected input order). Each
        dict has the ``StoredMessage`` shape documented at module top.
        """
        raise NotImplementedError

    def append_message(
        self,
        conversation_id: int,
        sender: str,
        content: str,
        meta_data: Dict[str, Any],
    ) -> int:
        """Persist one message; return its new id.

        Args:
            conversation_id: the internal DB id returned by
                :meth:`conversation_by_hash`.
            sender: one of the ``SENDER_*`` constants above.
            content: short human-readable summary (the store truncates
                if its column is shorter; callers should still keep
                this brief).
            meta_data: the full provider-shaped original message.
                Stored verbatim so the next turn's
                :meth:`list_recent_messages` round-trips it back
                unchanged to the same handler that wrote it.
        """
        raise NotImplementedError

    def list_conversations(
        self, owner_user_id: int, scope_meta_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List the user's conversations (newest first).

        Each dict carries the same keys as :meth:`conversation_by_hash`
        plus a ``hash_id`` key and any timestamps the store exposes.
        ``scope_meta_data`` filters server-side (e.g. by ``project_id``)
        so cross-tenant entries never reach the caller.
        """
        raise NotImplementedError

    def rename_conversation(self, hash_id: str, new_name: str) -> None:
        """Update the conversation's display name. Idempotent."""
        raise NotImplementedError
