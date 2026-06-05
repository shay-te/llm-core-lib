"""``ChatHandler`` — provider-specific (de)serializer + text extractor.

Each chat provider (OpenAI Responses, Bedrock Converse, future) has a
different on-wire shape for ``input_messages``. Rather than normalise
to a canonical shape (lossy, complex), every conversation is tagged
at create-time with a ``kind`` string and stores raw provider-shape
dicts in ``ConversationMessage.meta_data``. The matching
:class:`ChatHandler` subclass owns the conversion both directions.

Keep the interface tiny — the orchestrator (:class:`ChatSession`)
calls exactly these four methods. Adding a fifth requires changing
the orchestrator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


# Per-message content-column cap. ``conversation_core_lib``'s
# ``conversation_message.content`` is ``VARCHAR(500)``. Handlers
# truncate their summary to this so the store accepts the row.
MAX_CONTENT_SUMMARY_LEN = 500


class ChatHandler(ABC):
    """Provider adapter for the chat-session orchestrator.

    Subclasses MUST set :attr:`KIND` — the dispatch key the host
    registers against in :class:`ChatHandlerRegistry`, also stored on
    the conversation row so the next turn can re-pick the same handler.
    """

    KIND: str = ''

    @abstractmethod
    def build_user_prompt(self, command: str) -> Dict[str, Any]:
        """Return the provider-shaped dict for a new user message.

        Stored verbatim in ``ConversationMessage.meta_data`` and also
        passed to the connection as the latest input message of the
        turn.
        """

    @abstractmethod
    def stored_to_input_message(self, stored_meta_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one stored message's ``meta_data`` back to the
        provider-shape dict the connection expects in ``input_messages``.

        Almost always identity for handlers that already store the
        raw provider shape — but kept as an explicit hook so handlers
        that want to compress storage (e.g. drop redundant fields)
        can re-expand on read.
        """

    @abstractmethod
    def diff_new_messages(
        self,
        input_messages_before: List[Dict[str, Any]],
        final_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return the messages that the connection ADDED during the
        tool-call loop.

        ``input_messages_before`` is the list the orchestrator sent in
        (history + the new user prompt). ``final_messages`` is what
        the connection returned. The diff is the model output +
        function_call / function_call_output pairs that need to be
        persisted as new ``ConversationMessage`` rows.

        Most providers can return ``final_messages[len(input_messages_before):]``
        verbatim — the hook exists so a provider whose loop mutates
        the head (none today, but defensive) can recover the right
        suffix.
        """

    @abstractmethod
    def summarize_for_storage(self, message: Dict[str, Any]) -> str:
        """Produce the short ``content``-column summary for one message.

        Handlers MUST return at most :data:`MAX_CONTENT_SUMMARY_LEN`
        characters. The full original lives in ``meta_data``; this is
        the line a UI shows in a conversation listing.
        """

    @abstractmethod
    def extract_response_text(self, response: Any) -> str:
        """Pull the assistant's final text out of the connection's
        ``response`` object (provider-specific shape). Returns ``''``
        if the response has no text component (e.g. it stopped on a
        function_call without a closing message).
        """

    @abstractmethod
    def sender_for(self, message: Dict[str, Any]) -> str:
        """Classify a provider-shaped message as ``SENDER_USER``,
        ``SENDER_ASSISTANT``, or ``SENDER_TOOL`` for storage.

        :class:`ChatSession` passes this to the store so the store
        can render different bubbles for user vs LLM in a UI.
        """

    @staticmethod
    def truncate_summary(text: str) -> str:
        """Helper for subclasses: clip ``text`` to the storage cap."""
        if text is None:
            return ''
        if len(text) <= MAX_CONTENT_SUMMARY_LEN:
            return text
        # -1 to leave room for the ellipsis marker; reading "...truncated"
        # eats too much of the limit on a 500-char field.
        return text[: MAX_CONTENT_SUMMARY_LEN - 1] + '…'
