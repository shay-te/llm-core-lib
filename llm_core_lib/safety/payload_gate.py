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

from pii_core_lib.credential_scan import scan_text_for_credentials_and_phishing
from pii_core_lib.pii_patterns import PIIPatternFinding
from pii_core_lib.data_layers.service.pii_service import PiiService

from llm_core_lib.safety.llm_view import LLMView, RefLLMView


# Module-level singleton. ``PiiService`` is stateless — every method
# walks the passed-in payload and returns a new value — so one
# process-wide instance is fine and there's nothing to inject.
_PII_SERVICE = PiiService()


class UnsafeToolResultError(StatusCodeException):
    """A tool returned a non-:class:`RefLLMView` shape to the LLM.

    HTTP 500 — server-side contract violation. The web layer's
    ``@HandleException`` maps to 500; the chat loop's ``run_tool``
    catches and emits the sanitized envelope so the type-name never
    reaches the model.
    """

    def __init__(self, message: str) -> None:
        super().__init__(HTTPStatus.INTERNAL_SERVER_ERROR, message)


def _ensure_ref_llm_view(item: Any) -> None:
    # Strict: ONLY RefLLMView (or its subclasses) may cross the model
    # boundary. Plain ``LLMView`` carrying entity data — name, email,
    # status — is rejected. Tools must project to ``RefLLMView`` (ids
    # only) and rely on the render path's hydration for display.
    if not isinstance(item, RefLLMView):
        raise UnsafeToolResultError(
            f'tool returned {type(item).__name__} to the LLM; '
            f'tool results must be RefLLMView subclasses.'
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
            _ensure_ref_llm_view(item)
            dumped_items.append(item.model_dump())
        return dumped_items
    _ensure_ref_llm_view(result)
    return result.model_dump()


def to_llm_payload(
    result: Any,
    *,
    audit_logger: Optional[logging.Logger] = None,
    context: str = 'tool result',
) -> Any:
    """Project a tool's return into an LLM-safe JSON shape + scrub PII.

    Accepts a :class:`RefLLMView` (returns dict) or a list of
    :class:`RefLLMView` (returns list of dicts). ``None`` passes
    through. Anything else (plain ``LLMView`` with entity data, dict,
    list of dicts, primitives) raises :class:`UnsafeToolResultError`.

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


def audit_credentials(
    text: str,
    *,
    audit_logger: Optional[logging.Logger] = None,
    context: str = 'response text',
) -> None:
    """Audit-only credential + phishing scan over a free-text output.

    Companion to :func:`audit_text` with the credential / phishing
    detector set instead of PII patterns. Audit-only — never rewrites
    the text. Defaults ``audit_logger`` to the gate's module logger so
    detections never disappear silently.
    """
    scan_text_for_credentials_and_phishing(
        text,
        logger=audit_logger if audit_logger is not None else logging.getLogger(__name__),
        context_label=context,
    )


def scrub_history_for_llm(input_messages: list) -> list:
    """Scrub every prior turn in ``input_messages`` before resending to the LLM.

    Stateless chat APIs re-send the full history every turn — without
    intervention, prior-turn PII (in user commands, assistant text,
    and the ``function_call`` arguments the model itself chose) keeps
    getting re-transmitted, multiplying provider-side exposure.

    Scrubs every message EXCEPT the last (the current user command,
    which must reach the LLM intact). Correlation IDs (OpenAI
    ``call_id``, Bedrock ``toolUseId``) are preserved verbatim so
    the LLM can still match calls to results. The input list is not
    mutated; a new list is returned.

    Tail-share invariant: the returned list shares the tail dict's
    identity with the input. Safe because ``chat_with_tools`` only
    appends; if any caller starts in-place-mutating message dicts,
    copy the tail (``tail = dict(tail)``) here first.

    Log noise: emits a WARNING for every PII finding, typically
    duplicating the input-side audit. Filter the
    ``llm_core_lib.safety.payload_gate`` logger to suppress.
    """
    if not input_messages:
        return list(input_messages)
    # Last message is the current turn's user input — pass through.
    head, tail = input_messages[:-1], input_messages[-1]
    scrubbed_head = _PII_SERVICE.scrub(
        head,
        audit_logger=logging.getLogger(__name__),
        context='conversation history (prior turns)',
    )
    return [*scrubbed_head, tail]


def scrub_messages_for_persistence(messages: Any) -> Any:
    """Scrub PII out of a conversation-history list before storage.

    Walks the whole structure through :meth:`PiiService.scrub` so
    persisted history doesn't carry the user-typed PII that
    ``audit_text`` flagged but did not rewrite. The scrubbed
    structure is safe to thread back into the next
    ``MainChatService.run_command(..., input_messages=...)``. The
    original ``messages`` is never mutated.

    **Log noise note:** same as :func:`scrub_history_for_llm` —
    every detection emits a WARNING, typically duplicating the
    input-side audit. Filter the
    ``llm_core_lib.safety.payload_gate`` logger to suppress.
    """
    return _PII_SERVICE.scrub(
        messages,
        audit_logger=logging.getLogger(__name__),
        context='conversation persistence',
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


def scrub_user_error_payload(
    detail: str,
    *,
    audit_logger: Optional[logging.Logger] = None,
    context: str = 'user-visible tool error',
) -> dict:
    """Curated tool-error envelope, scrubbed for PII.

    ``LLMUserError`` carries human-readable detail (e.g. "No user found
    with email <email>") that the LLM needs verbatim to re-prompt — but
    the email itself must NOT cross the model boundary. Run the PII
    scrub on the detail before wrapping into the envelope.
    """
    scrubbed = _PII_SERVICE.scrub(
        {'status': 'invalid_input', 'detail': str(detail)},
        audit_logger=audit_logger,
        context=context,
    )
    return scrubbed


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
