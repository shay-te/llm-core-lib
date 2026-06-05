"""The single choke point between tool results and the LLM.

Every tool invocation's return must pass through :func:`to_llm_payload`.
The gate does TWO things, in order:

  1. Projects the result via :class:`LLMView`'s declared allowlist —
     refuses anything that isn't an :class:`LLMView` subclass so we
     review one gate instead of trusting every tool author independently.
  2. Scrubs the projected payload through :class:`PiiService` — any
     PII inside an allowlisted free-text field (a ``comment``, a
     ``note``) gets rewritten to ``[redacted:<pattern>]`` before the
     LLM sees it.

The host app does not need to know PII exists. It hands the gate a
tool's return, gets back an LLM-safe dict. The two safety layers
(allowlist projection + PII scrub) are reviewed once here and apply
uniformly to every chat tool in every repo.

:func:`run_tool` wraps the invoke path so success and failure get the
same sanitisation: the LLM always sees the generic
:func:`sanitized_error_payload` envelope on error, never an exception
message (which routinely carries user PII like ``User john@acme.com
not found in region EU-WEST``).

:func:`audit_text` is the audit-only companion for free-text outputs
(the model's final response, system prompts, tool args being logged).
It scans, audit-logs detections, and returns the findings — it does
NOT modify the text. The caller decides whether to act.
"""
from __future__ import annotations

import logging
import uuid
from http import HTTPStatus
from typing import Any, Callable, List, Optional

from core_lib.error_handling.status_code_exception import StatusCodeException

from pii_core_lib.pii_patterns import PIIPatternFinding
from pii_core_lib.pii_service import PiiService

from llm_core_lib.safety.llm_view import LLMView


# Module-level singleton. ``PiiService`` is stateless — every method
# walks the passed-in payload and returns a new value — so one
# process-wide instance is fine and there's nothing to inject.
_PII_SERVICE = PiiService()


class UnsafeToolResultError(StatusCodeException):
    """A tool returned a non-:class:`LLMView` shape to the LLM.

    HTTP 500 — server-side contract violation by the tool author. The
    web layer's ``@HandleException`` maps it to a 500 response; the
    chat loop's ``run_tool`` separately catches it and emits the
    sanitized envelope (so the type-name detail in the message never
    reaches the model either).
    """

    def __init__(self, message: str) -> None:
        super().__init__(HTTPStatus.INTERNAL_SERVER_ERROR, message)


def _ensure_llm_view(item: Any) -> None:
    if not isinstance(item, LLMView):
        raise UnsafeToolResultError(
            f'tool returned {type(item).__name__} to the LLM; '
            f'tool results must be LLMView subclasses.'
        )


def _project(result: Any) -> Any:
    """Allowlist projection only — no PII handling here.

    Split out so :func:`to_llm_payload` can layer the PII scrub on top
    with a single function body, and so the projection rule is the
    one thing :class:`UnsafeToolResultError` is about.
    """
    if result is None:
        return None
    if isinstance(result, list):
        dumped_items = []
        for item in result:
            _ensure_llm_view(item)
            dumped_items.append(item.model_dump())
        return dumped_items
    _ensure_llm_view(result)
    return result.model_dump()


def to_llm_payload(
    result: Any,
    *,
    audit_logger: Optional[logging.Logger] = None,
    context: str = 'tool result',
) -> Any:
    """Project a tool's return into an LLM-safe JSON shape + scrub PII.

    Accepts an :class:`LLMView` (returns dict) or a list of
    :class:`LLMView` (returns list of dicts). ``None`` passes through.
    Anything else raises :class:`UnsafeToolResultError`.

    After the allowlist projection, the result is passed through
    :meth:`PiiService.scrub` so any free-text PII inside allowlisted
    fields is rewritten to ``[redacted:<pattern>]`` before the LLM
    ever sees it.

    Args:
        result: tool's raw return value.
        audit_logger: destination for the WARNING line emitted when a
            scrub actually fires (zero-byte preview + pattern name).
            Defaults to :class:`PiiService`'s own module logger so
            detections never disappear silently.
        context: short label woven into the audit log so operators can
            locate the source (e.g. ``'admin tool result: search_users'``).
    """
    projected = _project(result)
    return _PII_SERVICE.scrub(
        projected,
        audit_logger=audit_logger,
        context=context,
    )


def audit_text(
    text: str,
    *,
    audit_logger: Optional[logging.Logger] = None,
    context: str = 'response text',
) -> List[PIIPatternFinding]:
    """Audit-only PII scan over a free-text output.

    Use on the post-LLM response text, system prompt text, or anywhere
    the caller wants to know "did PII slip into this string?" without
    modifying it. Returns the findings list (empty when clean); the
    audit log fires inside the call if findings exist.

    The text itself is never rewritten — that's the caller's decision
    if they want to act. Most chat flows just leave the response
    unchanged and rely on the prevention path :func:`to_llm_payload`
    to keep PII out of inputs in the first place.
    """
    return _PII_SERVICE.validate(
        text,
        audit_logger=audit_logger,
        context=context,
    )


def new_error_ref() -> str:
    """Mint a fresh 8-char hex correlation ref."""
    return uuid.uuid4().hex[:8]


def sanitized_error_payload(ref: str) -> dict:
    """Return the generic error envelope the LLM is allowed to see."""
    return {
        'status': 'error',
        'detail': 'The request could not be completed.',
        'ref': ref,
    }


def run_tool(
    fn: Callable[..., Any],
    *args: Any,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Any:
    """Single entry point for the LLM-tool-invoke path.

    Success goes through :func:`to_llm_payload` (which gates + scrubs).
    Failure (including :class:`UnsafeToolResultError` and any crash
    inside the PII scrub) logs the traceback locally with a
    correlation ref and returns :func:`sanitized_error_payload`.
    """
    effective_logger = logger or logging.getLogger('llm_core_lib.safety')
    try:
        # Gate + scrub inside the try so an unsafe-shape result OR a
        # scrub crash is also sanitized — neither the gate's leaked
        # type-name nor a PII-layer bug must reach the model.
        result = fn(*args, **kwargs)
        return to_llm_payload(
            result,
            audit_logger=effective_logger,
            context=getattr(fn, '__name__', 'tool result'),
        )
    except Exception:  # noqa: BLE001 — sanitized; detail to log only
        ref = new_error_ref()
        effective_logger.exception(
            'llm tool %r failed [ref=%s]',
            getattr(fn, '__name__', repr(fn)),
            ref,
        )
        return sanitized_error_payload(ref)
