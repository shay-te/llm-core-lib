"""Shared fixtures for the ``test_llm_view_*`` test files.

Per the workspace-wide test-organization rule (one TestCase per file,
filename mirrors the class name in snake_case — see ``architecture.md``
"Test file organization" and the matching note in this repo's
``AGENTS.md``), every behaviour of :class:`LLMView` lives in its own
file. The shared stub view that both files instantiate lives here so
it isn't duplicated.

The stub is a stdlib-only class that satisfies the marker's
``model_dump`` contract — the gate's only requirement — without
pulling in Pydantic. The real Pydantic-backed subclass lives one
layer up in :mod:`agent_core_lib.safety.llm_view` and is tested
there.

This module has no ``test_`` prefix on purpose; ``unittest discover``
and ``pytest`` both skip it.
"""
from __future__ import annotations

from llm_core_lib.safety.llm_view import LLMView


class StubLLMView(LLMView):
    """Non-Pydantic concrete subclass of :class:`LLMView`.

    Exercises the contract the transport marker advertises —
    ``model_dump`` is the one method the gate calls — without
    requiring Pydantic. Used by both ``test_llm_view_is_a_class`` and
    ``test_llm_view_isinstance_check``.
    """

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self):
        return dict(self._fields)
