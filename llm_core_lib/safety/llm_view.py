"""LLM-view marker base — stdlib-only.

The transport-layer marker that
:func:`llm_core_lib.safety.payload_gate.to_llm_payload`
``isinstance``-checks against. Concrete subclasses are the
Pydantic-backed view in
:class:`agent_core_lib.safety.llm_view.LLMView`; the Pydantic
dependency lives in that repo so this one stays a pure transport
library (the ``test_boundary.py`` rule forbids importing
``agent_core_lib`` from here, which is why the marker is in this
package rather than the concrete class).

Subclasses must expose ``model_dump() -> dict``. The contract is
runtime-checked by the gate calling the method, not statically by
``@abstractmethod`` — Pydantic v2 generates ``model_dump``
dynamically on ``BaseModel`` subclasses, which an ``ABC`` slot does
not recognise; declaring the method as abstract would make every
concrete view uninstantiable.
"""
from __future__ import annotations


class LLMView(object):
    """Marker base for any value returned to the LLM."""
