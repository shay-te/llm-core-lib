"""``ChatHandler`` — provider-shape adapter for :class:`ChatSession`.

Each conversation is tagged with a ``kind`` at create-time; the
matching handler subclass owns (de)serialisation in both directions
and stores raw provider-shape dicts in ``ConversationMessage.meta_data``
— no canonical normalisation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


# ``conversation_message.content`` is VARCHAR(500). Handlers must
# clip their summary to this.
MAX_CONTENT_SUMMARY_LEN = 500


class ChatHandler(ABC):

    # Dispatch key — host registers against this in ``ChatHandlerRegistry``
    # and the value is stored on the conversation row so the next
    # turn re-picks the same handler.
    KIND: str = ''

    @abstractmethod
    def build_user_prompt(self, command: str) -> Dict[str, Any]: ...

    @abstractmethod
    def stored_to_input_message(self, stored_meta_data: Dict[str, Any]) -> Dict[str, Any]:
        # Almost always identity; the hook exists for handlers that
        # compress storage and need to re-expand on read.
        ...

    @abstractmethod
    def diff_new_messages(
        self,
        input_messages_before: List[Dict[str, Any]],
        final_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # ``input_messages_before`` is the exact pre-call snapshot
        # the session sent to the connection; ``final_messages`` is
        # what the connection returned (typically the same list with
        # new items appended). Length-based slicing works for both
        # current providers.
        ...

    @abstractmethod
    def summarize_for_storage(self, message: Dict[str, Any]) -> str:
        # MUST clip to MAX_CONTENT_SUMMARY_LEN.
        ...

    @abstractmethod
    def extract_response_text(self, response: Any) -> str:
        # ``''`` when the response has no text (e.g. it stopped on a
        # function_call without a closing message).
        ...

    @abstractmethod
    def sender_for(self, message: Dict[str, Any]) -> str:
        # SENDER_USER / SENDER_ASSISTANT / SENDER_TOOL.
        ...

    @staticmethod
    def truncate_summary(text: str) -> str:
        if text is None:
            return ''
        if len(text) <= MAX_CONTENT_SUMMARY_LEN:
            return text
        return text[: MAX_CONTENT_SUMMARY_LEN - 1] + '…'
