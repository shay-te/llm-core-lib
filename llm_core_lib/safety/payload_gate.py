"""The single choke point between tool results and the LLM.

Every tool invocation's return must pass through :func:`to_llm_payload`.
The gate refuses anything that isn't an :class:`LLMView` subclass — one
gate to review instead of trusting every tool author independently.

:func:`run_tool` wraps the invoke path so success and failure get the
same sanitisation: the LLM always sees the generic
:func:`sanitized_error_payload` envelope on error, never an exception
message (which routinely carries user PII like ``User john@acme.com
not found in region EU-WEST``).
"""
from __future__ import annotations

import logging
import uuid
from http import HTTPStatus
from typing import Any, Callable, Optional

from core_lib.error_handling.status_code_exception import StatusCodeException

from llm_core_lib.safety.llm_view import LLMView


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


def to_llm_payload(result: Any) -> Any:
    """Project a tool's return into an LLM-safe JSON shape.

    Accepts an :class:`LLMView` (returns dict) or a list of
    :class:`LLMView` (returns list of dicts). ``None`` passes through.
    Anything else raises :class:`UnsafeToolResultError`.
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

    Success goes through :func:`to_llm_payload`. Failure (including
    :class:`UnsafeToolResultError`) logs the traceback locally with a
    correlation ref and returns :func:`sanitized_error_payload`.
    """
    effective_logger = logger or logging.getLogger('llm_core_lib.safety')
    try:
        # Gate inside the try so an unsafe-shape result is also
        # sanitized — the gate's exception message names the leaked
        # type, which must not reach the model.
        result = fn(*args, **kwargs)
        return to_llm_payload(result)
    except Exception:  # noqa: BLE001 — sanitized; detail to log only
        ref = new_error_ref()
        effective_logger.exception(
            'llm tool %s failed [ref=%s]',
            getattr(fn, '__name__', repr(fn)),
            ref,
        )
        return sanitized_error_payload(ref)
