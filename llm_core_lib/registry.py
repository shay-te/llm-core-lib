"""In-memory named LLM connection registry.

Host apps register an :class:`LlmConnectionConfig` once at boot; library
code resolves a provider by id at call time. The registry builds the
provider through :func:`create_llm_provider` (so config validation
fires at *register* time, not first-call time), then caches the
instance — calling :meth:`get` repeatedly returns the same
:class:`LlmProvider`.

No persistence by design — re-register on process start.
"""
from __future__ import annotations

from typing import Dict, List

from llm_core_lib.errors import (
    LlmConfigError,
    LlmDuplicateConnectionError,
    LlmMissingConnectionError,
)
from llm_core_lib.factory import create_llm_provider
from llm_core_lib.provider import LlmProvider
from llm_core_lib.types import LlmConnectionConfig


class LlmConnectionRegistry(object):
    """Holds named LLM connections + their built providers in-process."""

    def __init__(self) -> None:
        self._configs: Dict[str, LlmConnectionConfig] = {}
        self._providers: Dict[str, LlmProvider] = {}

    def register(self, config: LlmConnectionConfig) -> None:
        """Build + cache the provider. Raises on duplicate id or bad config."""
        if not config.id:
            raise LlmConfigError('connection config requires non-empty id')
        if config.id in self._configs:
            raise LlmDuplicateConnectionError(
                f'connection {config.id!r} already registered'
            )
        # Build eagerly so config errors surface here, not later at get().
        provider = create_llm_provider(config.to_provider_config())
        self._configs[config.id] = config
        self._providers[config.id] = provider

    def unregister(self, connection_id: str) -> None:
        if connection_id not in self._configs:
            raise LlmMissingConnectionError(
                f'connection {connection_id!r} not registered'
            )
        del self._configs[connection_id]
        del self._providers[connection_id]

    def get(self, connection_id: str) -> LlmProvider:
        try:
            return self._providers[connection_id]
        except KeyError as exc:
            raise LlmMissingConnectionError(
                f'connection {connection_id!r} not registered'
            ) from exc

    def has(self, connection_id: str) -> bool:
        return connection_id in self._configs

    def list(self) -> List[LlmConnectionConfig]:
        """Snapshot of currently-registered configs (insertion order)."""
        return list(self._configs.values())

    def clear(self) -> None:
        """Drop every registration. Provided for tests."""
        self._configs.clear()
        self._providers.clear()
