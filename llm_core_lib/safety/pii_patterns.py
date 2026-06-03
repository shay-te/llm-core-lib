"""Regex patterns for data shapes that must never reach the LLM.

These are deliberately broad — false positives are acceptable (the
runtime scan can be tightened on a per-pattern basis later), false
negatives are not. The structural defense (``LLMView`` allowlist +
``to_llm_payload`` choke point) is the primary layer; this pattern set
is the backstop for the case where an allowlisted *string* field still
happens to contain raw user PII (e.g. a free-text comment that quoted
an email address).

Address detection intentionally is **not** regex-based — postal
addresses don't have a parsable shape and a regex catches nothing
useful while emitting wild false positives. The address defense lives
at the allowlist layer (no ``LLMView`` subclass declares an address
field unless the field name itself is the contract that the value will
be scrubbed before the view is built).
"""
from __future__ import annotations

import re
from typing import Dict, Pattern


PII_PATTERNS: Dict[str, Pattern[str]] = {
    # RFC-5322-ish email; the local-part allows the common punctuation
    # set and the host requires at least one dot.
    'email': re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'),
    # US SSN format. Narrow on purpose — broader SSN-shaped numbers
    # collide with phone fragments and order ids.
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    # E.164-ish phone numbers: optional leading +, then 10+ digits with
    # optional separators (space, dash, paren, dot).
    'phone': re.compile(r'\+?\d[\d \-().]{8,}\d'),
    # 13-16 digit card numbers, with optional space or dash separators.
    # Doesn't validate the Luhn checksum — that's a job for the scrubber,
    # not the detector.
    'credit_card': re.compile(r'\b(?:\d[ \-]?){13,16}\b'),
    # IBAN: 2-letter country, 2-digit check, 11-30 alphanumerics.
    'iban': re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b'),
}


PII_PATTERN_NAMES: frozenset = frozenset(PII_PATTERNS.keys())
