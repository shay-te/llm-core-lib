"""LLM-view marker base — stdlib-only.

Tool results that cross the LLM boundary must subclass :class:`LLMView`
so the choke point in :mod:`llm_core_lib.safety.payload_gate` can
distinguish a declared, allowlisted shape from an accidentally-leaking
raw dict / ORM row / SQLAlchemy model.

The base lives here, in the transport layer, so :func:`to_llm_payload`'s
``isinstance(item, LLMView)`` check is meaningful — but the base
itself is a plain Python class with **no Pydantic dependency**. The
enforcement contract (``ConfigDict(extra='forbid', frozen=True)``,
the allowlist of declared fields, ``project`` / ``project_list``
adapters) lives in :mod:`agent_core_lib.safety.llm_view`, where the
Pydantic dependency belongs (the agent layer owns the data-shape
contract for tool returns; the transport layer just polices the
boundary).

In practice tool authors subclass the **agent-core-lib** concrete
:class:`agent_core_lib.safety.llm_view.LLMView`, not this one directly.
The transport-layer marker exists so the gate has a single name to
``isinstance`` against without forcing a dependency from
``llm-core-lib`` onto ``agent-core-lib`` (the existing boundary test
in ``test_boundary.py`` enforces that direction).

The contract every concrete subclass must satisfy is one method:
``model_dump() -> dict``. The name is deliberately Pydantic's so the
canonical agent-core-lib subclass satisfies it without a wrapper.
We do NOT declare it as ``@abstractmethod`` here because Pydantic v2
generates ``model_dump`` dynamically on its ``BaseModel`` subclasses
(not as a method on the class statement), which an ``ABC`` abstract
slot doesn't recognize — the subclass would be uninstantiable. The
marker is therefore a plain class; the runtime check that
``model_dump`` exists is performed by the gate when it calls
``item.model_dump()``.
"""
from __future__ import annotations


class LLMView(object):
    """Marker base — any value returned to the LLM must subclass this.

    Subclass contract (not enforced statically; runtime-enforced by
    the gate calling ``item.model_dump()``):

      * ``model_dump(self) -> dict`` returns the allowlisted field
        set as a JSON-safe ``dict``.

    The canonical concrete subclass is
    :class:`agent_core_lib.safety.llm_view.LLMView` — a Pydantic v2
    ``BaseModel`` with ``ConfigDict(extra='forbid', frozen=True)``.
    That's what tool authors subclass.
    """
