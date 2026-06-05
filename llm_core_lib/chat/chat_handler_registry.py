"""``ChatHandlerRegistry`` — name → :class:`ChatHandler` lookup.

The host (admin-backend) builds one registry at composition root and
hands it to :class:`ChatSession`. When a turn comes in for a
conversation, the session reads the conversation's stored ``kind``
and asks the registry for the matching handler.

A registry instead of a hard-coded dict keeps llm-core-lib agnostic
of which handlers a given host actually wires (host with only OpenAI
support registers only the OpenAI handler).
"""
from __future__ import annotations

from typing import Dict

from llm_core_lib.chat.chat_handler import ChatHandler


class UnknownChatKindError(LookupError):
    """No handler registered for the conversation's ``kind``.

    Surfaced via the orchestrator's normal error path; the host's
    web-layer ``@HandleException`` turns it into a 500 (server-side
    config mistake — a conversation got persisted with a kind no
    handler is registered for).
    """


class ChatHandlerRegistry(object):

    def __init__(self) -> None:
        self._by_kind: Dict[str, ChatHandler] = {}

    def register(self, handler: ChatHandler) -> None:
        """Register a handler. The kind comes from
        :attr:`ChatHandler.KIND`. Re-registering the same kind
        overwrites (useful in tests that swap a stub in)."""
        if not handler.KIND:
            raise ValueError(
                f'{type(handler).__name__} did not set ChatHandler.KIND'
            )
        self._by_kind[handler.KIND] = handler

    def get(self, kind: str) -> ChatHandler:
        handler = self._by_kind.get(kind)
        if handler is None:
            raise UnknownChatKindError(
                f'no chat handler registered for kind={kind!r}; '
                f'available={sorted(self._by_kind)}'
            )
        return handler

    def has(self, kind: str) -> bool:
        return kind in self._by_kind

    def kinds(self) -> list:
        return sorted(self._by_kind)
