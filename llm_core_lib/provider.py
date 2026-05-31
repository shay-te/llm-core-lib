"""``LlmProvider`` — the single interface every backend adapter
implements.

Kept as an ABC (not a Protocol) so adapters get a concrete base they
can inherit ``stream`` from for free: the default implementation calls
``chat`` once and yields one delta + one stop event, which is enough
for non-streaming backends without forcing every adapter to write the
fallback themselves.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from llm_core_lib.types import (
    LlmChatRequest,
    LlmChatResponse,
    LlmStreamEvent,
)


class LlmProvider(ABC):
    """Common interface for OpenAI / Anthropic / Bedrock adapters.

    Concrete adapters set the class-level ``id`` attribute to one of
    ``LlmProviderId`` values; the registry / factory rely on it for
    routing.
    """

    id: str = ''

    @abstractmethod
    def chat(self, request: LlmChatRequest) -> LlmChatResponse:
        """Send a chat request and return the normalized response.

        Adapters MUST wrap SDK errors in :class:`LlmProviderError` so
        callers don't have to import the underlying SDK's exception
        hierarchy.
        """

    def stream(self, request: LlmChatRequest) -> Iterator[LlmStreamEvent]:
        """Default streaming = one delta + one stop derived from
        :meth:`chat`.

        Adapters whose SDK supports true streaming should override this
        method and emit incremental deltas. Callers can rely on the
        interface uniformly either way.
        """
        try:
            response = self.chat(request)
        except Exception as exc:  # noqa: BLE001 — surface as a stream error
            yield LlmStreamEvent(type='error', content=str(exc))
            return
        yield LlmStreamEvent(type='delta', content=response.content)
        yield LlmStreamEvent(type='stop', finish_reason=response.finish_reason)
