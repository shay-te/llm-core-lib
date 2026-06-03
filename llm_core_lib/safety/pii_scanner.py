"""Runtime backstop scanner for PII in LLM-bound payloads.

Three entry points:

* :func:`find_pii` — returns a list of :class:`PIIFinding` (no raise),
  for diagnostics, logging, and test assertions.
* :func:`assert_no_pii` — raises :class:`PIIDetectedError` if anything
  matched; this is what test suites use to lock the contract that a
  given tool's payload is PII-free.
* :func:`scrub_pii` — recursively replaces matched substrings with
  ``[REDACTED:<pattern_name>]`` so a tool that legitimately needs to
  return free-text (a user comment, a search-blob) can be sanitized
  before crossing the LLM boundary.

The matching is performed against ``json.dumps(payload, default=str)``
so dates / Decimal / UUID values don't crash the scan. Tuples / lists /
nested dicts are walked in place by :func:`scrub_pii`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List

from llm_core_lib.safety.pii_patterns import PII_PATTERNS


@dataclass(frozen=True)
class PIIFinding(object):
    """One match of a named PII pattern. ``redacted_preview`` is safe to log."""

    pattern_name: str
    redacted_preview: str


class PIIDetectedError(ValueError):
    """:func:`assert_no_pii` found at least one match.

    The error message names the matched patterns but never the raw
    matched values — the redacted preview is the only thing safe to
    surface to logs / test failures.
    """


def _redact(matched_text: str) -> str:
    prefix_len = min(4, len(matched_text))
    prefix = matched_text[:prefix_len]
    return f'{prefix}…[REDACTED, len={len(matched_text)}]'


def _payload_blob(payload: Any) -> str:
    # ``default=str`` so dates / Decimal / UUID don't crash. ``sort_keys``
    # makes the scan deterministic, which matters for test assertions.
    return json.dumps(payload, default=str, sort_keys=True)


def find_pii(payload: Any) -> List[PIIFinding]:
    """Return every PII-pattern match found anywhere in ``payload``.

    The payload is serialized once via ``json.dumps`` and scanned with
    every pattern; nested dicts / lists / tuples are covered by the
    serialization. Returns an empty list if nothing matched.
    """
    if payload is None:
        return []
    blob = _payload_blob(payload)
    findings: List[PIIFinding] = []
    for pattern_name, regex in PII_PATTERNS.items():
        for match in regex.finditer(blob):
            findings.append(PIIFinding(
                pattern_name=pattern_name,
                redacted_preview=_redact(match.group(0)),
            ))
    return findings


def assert_no_pii(payload: Any) -> None:
    """Raise :class:`PIIDetectedError` if ``payload`` contains any PII.

    The error message lists pattern names and *redacted* previews — the
    raw matched value is never re-surfaced. Use this in tests to lock
    the contract that a given tool's payload is PII-free.
    """
    findings = find_pii(payload)
    if not findings:
        return
    summary = ', '.join(
        f'{finding.pattern_name}={finding.redacted_preview}'
        for finding in findings
    )
    raise PIIDetectedError(f'PII detected in LLM-bound payload: {summary}')


def scrub_pii(payload: Any) -> Any:
    """Recursively replace PII matches inside ``payload`` with placeholders.

    Strings are rewritten in place via the pattern set. Dicts and lists
    are walked recursively; tuples are returned as tuples. Anything that
    isn't a container or string is returned unchanged (numbers, bools,
    None, dates, etc. — those don't carry text PII).
    """
    if isinstance(payload, str):
        scrubbed = payload
        for pattern_name, regex in PII_PATTERNS.items():
            scrubbed = regex.sub(f'[REDACTED:{pattern_name}]', scrubbed)
        return scrubbed
    if isinstance(payload, dict):
        return {key: scrub_pii(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [scrub_pii(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(scrub_pii(value) for value in payload)
    return payload
