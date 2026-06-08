"""``ChatHandlerRegistry`` — name → :class:`ChatHandler` lookup.

Host registers concrete handlers at composition root; the orchestrator
resolves them per turn by the conversation's stored ``kind``.
"""
from __future__ import annotations

from typing import Dict

from llm_core_lib.chat.chat_handler import ChatHandler


class UnknownChatKindError(LookupError):
    """No handler registered for the conversation's ``kind``."""


class ChatHandlerRegistry(object):

    def __init__(self) -> None:
        self._by_kind: Dict[str, ChatHandler] = {}

    def register(self, handler: ChatHandler) -> None:
        # Re-registering the same kind overwrites — useful in tests.
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
