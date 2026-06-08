"""In-memory named LLM connection registry.

Host apps register an :class:`LlmConnectionConfig` once at boot; library
code resolves a ``ConnectionFactory`` by id at call time. The registry
builds the factory through :func:`create_connection_factory` at
registration (so config errors surface at boot, not at first call) and
caches it — calling :meth:`get` repeatedly returns the same factory.

Callers then go through the factory the same way they would with a
hand-rolled ``BedrockConnectionFactory``::

    factory = registry.get('openai-default')
    conn = factory.get()
    try:
        result = conn.complete_text('hello')
    finally:
        conn.close()

No persistence by design — re-register on process start.
"""
from __future__ import annotations

from typing import Dict, List

from core_lib.connection.connection_factory import ConnectionFactory

from llm_core_lib.errors import (
    LlmConfigError,
    LlmDuplicateConnectionError,
    LlmMissingConnectionError,
)
from llm_core_lib.factory import create_connection_factory
from llm_core_lib.types import LlmConnectionConfig


class LlmConnectionRegistry(object):
    """Holds named LLM connections + their built factories in-process."""

    def __init__(self) -> None:
        self._configs: Dict[str, LlmConnectionConfig] = {}
        self._factories: Dict[str, ConnectionFactory] = {}

    def register(self, config: LlmConnectionConfig) -> None:
        """Build + cache the factory. Raises on duplicate id or bad config."""
        if not config.id:
            raise LlmConfigError('connection config requires non-empty id')
        if config.id in self._configs:
            raise LlmDuplicateConnectionError(
                f'connection {config.id!r} already registered'
            )
        # Eager build → config errors surface here, not at first call.
        factory = create_connection_factory(config)
        self._configs[config.id] = config
        self._factories[config.id] = factory

    def get(self, connection_id: str) -> ConnectionFactory:
        """Return the cached ``ConnectionFactory`` for ``connection_id``.

        Caller drives the per-call lifecycle::

            factory = registry.get(id)
            conn = factory.get()
            try:
                conn.complete_text(...)
            finally:
                conn.close()
        """
        try:
            return self._factories[connection_id]
        except KeyError as exc:
            raise LlmMissingConnectionError(
                f'connection {connection_id!r} not registered'
            ) from exc

    def list(self) -> List[LlmConnectionConfig]:
        """Snapshot of currently-registered configs (insertion order)."""
        return list(self._configs.values())
