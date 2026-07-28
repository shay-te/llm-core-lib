"""LLM-view marker bases — stdlib-only.

Transport-layer markers that
:func:`llm_core_lib.safety.payload_gate.to_llm_payload`
``isinstance``-checks against. Concrete Pydantic subclasses live in
``agent_core_lib.safety`` — the Pydantic dep stays in that repo so
``llm-core-lib`` remains a pure transport library (``test_boundary.py``
forbids importing ``agent_core_lib`` from here).

Two markers:

* :class:`LLMView` — any value a tool may return. Used internally by
  hydration views that never cross the model boundary.
* :class:`RefLLMView` — the ONLY shape the gate lets reach the LLM.
  Carries ``id`` (and optionally non-PII signals like a score) — never
  raw entity data. Subclassing ``RefLLMView`` is the marker contract
  that says "this view is safe for the model boundary".

Subclasses must expose ``model_dump() -> dict``. Runtime-checked by the
gate calling it — declaring abstract would conflict with Pydantic v2's
dynamic generation.
"""
from __future__ import annotations


class LLMView(object):
    """Marker base for any value returned to the LLM."""


class RefLLMView(LLMView):
    """Marker base for the MODEL boundary — id-only (or id + non-PII signal).

    Concrete subclass with the ``id: int`` field lives in
    ``agent_core_lib.safety.ref_llm_view.RefLLMView``. The gate refuses
    anything that isn't a ``RefLLMView`` subclass — that's how PII is
    structurally prevented from reaching the model.
    """
