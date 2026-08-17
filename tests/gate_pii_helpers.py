"""Shared fixtures for the ``test_to_llm_payload_scrubs_*`` and
``test_audit_text_*`` test files.

Per the workspace-wide test-organization rule (one TestCase per file,
filename mirrors the class name in snake_case), each behaviour of
the gate's PII layer lives in its own file. The stub view that
exercises an allowlisted free-text field — a ``comment`` value the
gate's scrub step is expected to walk into — lives here so both
files share one definition.

The stub is a stdlib-only :class:`LLMView` subclass; the
Pydantic-backed concrete views live one layer up in agent-core-lib.

This module has no ``test_`` prefix on purpose; the discoverers skip it.
"""
from __future__ import annotations

from llm_core_lib.safety.llm_view import RefLLMView


class CommentStubLLMView(RefLLMView):
    """RefLLMView marker subclass that exercises the in-payload PII scrub.

    Inherits the RefLLMView marker so the strict gate accepts it.
    Carries a free-text ``comment`` field so the gate's scrub step
    has something to walk into — the fixture's job is to verify that
    PII in any allowlisted free-text field is rewritten before the
    LLM sees it.
    """

    def __init__(self, id, comment):
        self.id = id
        self.comment = comment

    def model_dump(self):
        return {'id': self.id, 'comment': self.comment}
