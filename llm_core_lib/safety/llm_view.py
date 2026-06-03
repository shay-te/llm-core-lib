"""LLM-view marker base.

Tool results that cross the LLM boundary must subclass :class:`LLMView`
so the choke point in :mod:`llm_core_lib.safety.payload_gate` can
distinguish a declared, allowlisted shape from an accidentally-leaking
raw dict / ORM row / SQLAlchemy model.

The same intent the task description writes against Pydantic v2
(``ConfigDict(extra='forbid')``) is achieved here with a frozen
dataclass: unknown init kwargs raise ``TypeError`` at construction,
post-init assignment is rejected, and the field list is the complete
allowlist of what the LLM will see. Pydantic isn't pulled in as a
dependency — ``llm-core-lib`` keeps ``requirements.txt`` to just
``core-lib`` (see ``AGENTS.md``) and stdlib dataclasses give us the
same guarantee at the boundary.
"""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict


class LLMView(object):
    """Marker base for any value that may be returned to the LLM.

    Concrete subclasses MUST be ``@dataclass(frozen=True)`` and declare
    every field the LLM is permitted to see. The frozen-dataclass shape
    enforces three properties the safety layer relies on:

    1. **Allowlisted fields** — construction with an unknown kwarg
       raises ``TypeError``; no caller can sneak an extra value in.
    2. **Immutability** — ``frozen=True`` blocks post-init assignment,
       so nothing downstream can splice a raw email / address onto an
       already-built view.
    3. **JSON-shape** — ``to_dict`` walks dataclass fields only, which
       means ORM rows / SQLAlchemy models attached as attributes won't
       leak (they'd have to be declared as a field first).
    """

    def to_dict(self) -> Dict[str, Any]:
        """Return the allowlisted field set as a plain ``dict``.

        Raises ``TypeError`` if the subclass forgot to apply
        ``@dataclass(frozen=True)`` — failing loud is the point: a
        subclass without declared fields has no allowlist.
        """
        if not is_dataclass(self):
            raise TypeError(
                f'{type(self).__name__} subclasses LLMView but is not a '
                f'dataclass — declare fields with @dataclass(frozen=True).'
            )
        return asdict(self)

    @classmethod
    def allowed_field_names(cls) -> frozenset:
        """Return the set of field names this view will expose to the LLM.

        Used by ``test_*_view_fields_are_locked`` style tests so a
        reviewer can codify the allowlist and the test fails the moment
        someone widens it without updating the test.
        """
        if not is_dataclass(cls):
            raise TypeError(
                f'{cls.__name__} subclasses LLMView but is not a '
                f'dataclass — declare fields with @dataclass(frozen=True).'
            )
        return frozenset(f.name for f in fields(cls))
