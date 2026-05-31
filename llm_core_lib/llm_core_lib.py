"""Composition root: ``LlmCoreLib(CoreLib)``.

Exposes a single attribute — ``self.registry`` — that's an
:class:`LlmConnectionRegistry`. If a config is passed (Hydra
``DictConfig`` or a plain mapping with the same shape as
``example_data.yaml``), the constructor pre-registers every entry under
``core_lib.llm.connections``.

Thin :class:`core_lib.core_lib.CoreLib` subclass: it owns one
registry attribute and delegates everything real to it. The class is
the transport-layer composition root only — prompt preparation belongs
upstream (in the caller / workflow layer) and is intentionally out of
scope here. See README → "Boundary with agent-core-lib".
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from omegaconf import DictConfig

from core_lib.core_lib import CoreLib

from llm_core_lib.registry import LlmConnectionRegistry
from llm_core_lib.types import LlmConnectionConfig


class LlmCoreLib(CoreLib):
    """Hosts a single :class:`LlmConnectionRegistry`.

    Args:
        conf: Optional Hydra ``DictConfig`` whose
            ``core_lib.llm.connections`` is a list of connection
            entries. Each entry is passed to
            :meth:`LlmConnectionRegistry.register` in order, so any
            config error fails the constructor — host apps find out
            about a bad connection at boot, not at first-call time.
    """

    def __init__(self, conf: Optional[DictConfig] = None):
        super().__init__()
        self.config = conf
        self.registry = LlmConnectionRegistry()
        if conf is not None:
            self._hydrate_from_config(conf)

    def _hydrate_from_config(self, conf: Any) -> None:
        connections = _read_connections(conf)
        if not connections:
            return
        for entry in connections:
            self.registry.register(_connection_from_mapping(entry))


def _read_connections(conf: Any) -> Iterable[Any]:
    """Walk ``conf.core_lib.llm.connections`` defensively.

    Accepts a Hydra ``DictConfig``, a plain dict, or any object whose
    nested attribute / item access returns a list of mappings. Returns
    an empty list if the path doesn't exist.
    """
    core = _attr_or_item(conf, 'core_lib')
    llm = _attr_or_item(core, 'llm') if core is not None else None
    connections = _attr_or_item(llm, 'connections') if llm is not None else None
    if connections is None:
        return []
    try:
        return list(connections)
    except TypeError:
        return []


def _attr_or_item(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    # Mapping path wins so a plain dict works without owning attrs.
    if isinstance(obj, Mapping):
        return obj.get(key)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return None


def _connection_from_mapping(entry: Any) -> LlmConnectionConfig:
    """Build :class:`LlmConnectionConfig` from a dict-like config entry.

    Pulled out so the same shape works for plain dicts and for Hydra
    ``DictConfig`` items — both support attr-or-item access.
    """
    extra_raw = _attr_or_item(entry, 'extra') or {}
    try:
        extra = dict(extra_raw)
    except (TypeError, ValueError):
        extra = {}

    return LlmConnectionConfig(
        id=str(_attr_or_item(entry, 'id') or ''),
        provider=str(_attr_or_item(entry, 'provider') or ''),
        model=str(_attr_or_item(entry, 'model') or ''),
        api_key=_optional_str(_attr_or_item(entry, 'api_key')),
        base_url=_optional_str(_attr_or_item(entry, 'base_url')),
        organization=_optional_str(_attr_or_item(entry, 'organization')),
        region=_optional_str(_attr_or_item(entry, 'region')),
        access_key=_optional_str(_attr_or_item(entry, 'access_key')),
        secret_key=_optional_str(_attr_or_item(entry, 'secret_key')),
        endpoint_url=_optional_str(_attr_or_item(entry, 'endpoint_url')),
        vision_model=_optional_str(_attr_or_item(entry, 'vision_model')),
        embedding_model=_optional_str(_attr_or_item(entry, 'embedding_model')),
        max_tokens=int(_attr_or_item(entry, 'max_tokens') or 4096),
        temperature=float(_attr_or_item(entry, 'temperature') or 0.0),
        extra=extra,
    )


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None
