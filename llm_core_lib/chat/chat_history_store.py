"""``ChatHistoryStore`` — persistence boundary chat sessions talk to.

Duck-typed interface so ``llm-core-lib`` doesn't depend on any
specific storage backend; the host wires the implementation.

Stored messages are dicts shaped:
    {id, sender_user_id, content, meta_data}
where ``meta_data`` is the provider-shaped original message
(round-tripped verbatim).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Sender markers — preserved verbatim through the store so a UI can
# render different bubbles per category.
SENDER_USER = 'user'
SENDER_ASSISTANT = 'assistant'
SENDER_TOOL = 'tool'


class ChatHistoryStore(object):

    def create_conversation(
        self,
        owner_user_id: int,
        kind: str,
        name: str,
        scope_meta_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    def conversation_by_hash(self, hash_id: str) -> Optional[Dict[str, Any]]:
        # Returns dict with keys: id, kind, name, meta_data. ``None``
        # means "not found" — callers must surface a generic error
        # (never let the LLM probe).
        raise NotImplementedError

    def list_recent_messages(
        self, conversation_id: int, limit: int,
    ) -> List[Dict[str, Any]]:
        # Oldest-first (the LLM's expected input order).
        raise NotImplementedError

    def append_message(
        self,
        conversation_id: int,
        sender: str,
        content: str,
        meta_data: Dict[str, Any],
    ) -> int:
        raise NotImplementedError

    def list_conversations(
        self, owner_user_id: int, scope_meta_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Newest-first; ``scope_meta_data`` filters server-side (e.g.
        # ``{'project_id': 7}``) so cross-tenant entries never leak.
        raise NotImplementedError

    def rename_conversation(self, hash_id: str, new_name: str) -> None:
        raise NotImplementedError

    def delete_conversation(self, hash_id: str) -> None:
        # Idempotent — unknown hash is a no-op. Hard- vs soft-delete
        # is the backing store's choice; callers don't differentiate.
        raise NotImplementedError
