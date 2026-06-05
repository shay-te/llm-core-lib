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

from llm_core_lib.safety.llm_view import LLMView


class CommentStubLLMView(LLMView):
    """LLMView whose allowlisted shape exposes a free-text ``comment``.

    The gate's projection step accepts the view (it's an LLMView
    subclass) and emits ``{'id': ..., 'comment': ...}``. The gate's
    scrub step then walks the projected dict and rewrites any PII it
    finds inside the comment text. Together they make this fixture
    the minimal case the gate is designed for: a typed allowlist that
    still has at least one free-text field a tool author might
    accidentally fill with user PII.
    """

    def __init__(self, id, comment):
        self.id = id
        self.comment = comment

    def model_dump(self):
        return {'id': self.id, 'comment': self.comment}
