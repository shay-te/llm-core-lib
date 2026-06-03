"""The single choke point between tool results and the LLM.

Every tool invocation's return value must pass through
:func:`to_llm_payload`. The gate refuses to forward anything that isn't
a declared :class:`~llm_core_lib.safety.llm_view.LLMView` subclass (or a
list of them) — that's the whole point: instead of trusting eight
separate tool authors to remember to project their data correctly, we
trust *one* gate to refuse anything that doesn't carry a declared
allowlist.

:func:`run_tool` wraps the whole tool-invoke path so the failure path
gets the same treatment as the success path: the LLM sees a generic
"could not be completed" envelope with a correlation ref, while the
full traceback goes only to the configured logger. This is the
"sanitized errors" half of the task — exception strings often carry
PII (``User john@acme.com not found in region EU-WEST`` is the
canonical shape) and the gate is the one place we can guarantee that
message never reaches the model.
"""
from __future__ import annotations

import logging
import uuid
from http import HTTPStatus
from typing import Any, Callable, Optional

from core_lib.error_handling.status_code_exception import StatusCodeException

from llm_core_lib.safety.llm_view import LLMView


class UnsafeToolResultError(StatusCodeException):
    """A tool tried to return a non-:class:`LLMView` shape to the LLM.

    Failing loud is the contract — the model must never see something
    that hasn't been routed through a declared, allowlisted view. A
    raw dict / ORM row / SQLAlchemy model reaching this point is a bug
    in the tool, not something to paper over with a best-effort
    coercion.

    HTTP status: **500 INTERNAL SERVER ERROR**. The cause is a
    server-side contract violation by a tool author (returning a shape
    the safety gate refuses) — there is no client action that would
    make the request succeed, and surfacing it as a 4xx would let the
    bug hide as "client problem" in monitoring. The web layer's
    ``@HandleException`` decorator picks the status code off this
    exception and turns it into the matching HTTP response; the chat
    loop's ``run_tool`` separately catches it and returns the generic,
    PII-free error envelope to the LLM (so the type-name detail in
    the message never reaches the model either).
    """

    def __init__(self, message: str) -> None:
        super().__init__(HTTPStatus.INTERNAL_SERVER_ERROR, message)


def _ensure_llm_view(item: Any) -> None:
    if not isinstance(item, LLMView):
        raise UnsafeToolResultError(
            f'tool tried to return {type(item).__name__} to the LLM; '
            f'tool results must be LLMView subclasses (the safety '
            f'choke point in llm_core_lib.safety.payload_gate refuses '
            f'anything else by design — see the LLMView docstring).'
        )


def to_llm_payload(result: Any) -> Any:
    """Project a tool's return value into an LLM-safe JSON shape.

    Accepts a single :class:`LLMView` (returns a dict) or a list of
    :class:`LLMView` instances (returns a list of dicts). Anything else
    raises :class:`UnsafeToolResultError` — no silent passthrough, no
    coercion, no best-effort. ``None`` is allowed because some tools
    legitimately return "nothing to report".
    """
    if result is None:
        return None
    if isinstance(result, list):
        for item in result:
            _ensure_llm_view(item)
        return [item.model_dump() for item in result]
    _ensure_llm_view(result)
    return result.model_dump()


def new_error_ref() -> str:
    """Mint a fresh correlation ref for an LLM tool failure.

    Returns an 8-char hex slice from ``uuid4`` — the operator-facing
    format that ``sanitized_error_payload`` puts in the ``ref`` field
    and that ``self.logger.exception('... [ref=%s] ...', ref)`` writes
    to the log line. **Use this instead of inlining
    ``uuid.uuid4().hex[:8]``** anywhere you're about to build a
    sanitized error envelope — one definition of the ref format means
    one place to change if the length / encoding ever shifts.
    """
    return uuid.uuid4().hex[:8]


def sanitized_error_payload(ref: str) -> dict:
    """Return the generic error envelope the LLM is allowed to see.

    The detail string is intentionally fixed and PII-free; the ``ref``
    is the only correlation hook back to the log line that holds the
    full traceback. Lifting this into a function (rather than inlining
    the literal) keeps tests honest — they can assert against this
    exact shape rather than reproducing the magic string.
    """
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

    Success: forward the return through :func:`to_llm_payload`.

    Failure: log the full traceback locally (with a correlation ref)
    and return the generic :func:`sanitized_error_payload` envelope to
    the caller. This guarantees the LLM never sees an exception string
    — those routinely carry the very user data ("not found in region
    EU-WEST", "duplicate key violates unique constraint Key
    (email)=(john@acme.com)") that the success path is structured to
    keep out.
    """
    log = logger if logger is not None else logging.getLogger('llm_core_lib.safety')
    try:
        result = fn(*args, **kwargs)
        # The gate is inside the try on purpose — an UnsafeToolResultError
        # means the tool returned a non-LLMView shape (a bug in the tool),
        # and we want the LLM to see the generic envelope, not the
        # type-name detail the gate's message carries.
        return to_llm_payload(result)
    except Exception:  # noqa: BLE001 — sanitized error path; full detail goes to the log
        ref = new_error_ref()
        log.exception(
            'llm tool %s failed [ref=%s]',
            getattr(fn, '__name__', repr(fn)),
            ref,
        )
        return sanitized_error_payload(ref)
