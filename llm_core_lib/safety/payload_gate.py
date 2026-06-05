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


def audit_credentials(
    text: str,
    *,
    audit_logger: Optional[logging.Logger] = None,
    context: str = 'response text',
) -> None:
    """Audit-only credential + phishing scan over a free-text output.

    Companion to :func:`audit_text` — same call shape, same audit-only
    contract, but the detector set is API keys / tokens / OAuth bearers
    / Stripe & AWS & GitHub keys / phishing-URL patterns instead of
    PII. Host apps that want full sensitive-data coverage on a single
    string call both; the two are intentionally separate scans so a
    caller can route the two findings to different audit destinations
    if they want.

    Emits one WARNING per scan that finds anything (the detector itself
    handles the line) and returns ``None`` — the text is never
    rewritten. As with :func:`audit_text`, prevention is done by
    :func:`to_llm_payload` upstream of the LLM call; this is the
    detective post-hoc on whatever made it through.

    When the caller doesn't pass ``audit_logger``, the gate's own
    module logger is used so detections never disappear silently
    (mirrors :meth:`PiiService.validate`'s default-logger behaviour).
    """
    scan_text_for_credentials_and_phishing(
        text,
        logger=audit_logger if audit_logger is not None else logging.getLogger(__name__),
        context_label=context,
    )


def scrub_history_for_llm(input_messages: list) -> list:
    """Scrub every prior turn in ``input_messages`` before resending to the LLM.

    The chat loop accumulates messages across turns: user commands,
    assistant text responses, and ``function_call`` / ``toolUse``
    items the model itself emitted (which carry the arguments the
    model chose, often containing PII the user originally typed).
    Every subsequent turn re-sends the *full* history to the LLM
    provider — that's how stateless chat APIs work. Without
    intervention, the same PII keeps getting re-transmitted on every
    turn, multiplying provider-side exposure.

    This helper scrubs all but the **last** message in the list:

      * Prior ``user`` / ``assistant`` text content → PII redacted in
        place (string scrub).
      * Prior ``function_call`` / ``function_call_output`` items
        (OpenAI Responses) → arguments + output strings scrubbed; the
        ``call_id`` is preserved verbatim so the LLM can still
        correlate the call with its result.
      * Prior Bedrock ``toolUse`` blocks → ``input`` dict scrubbed;
        ``toolUseId`` preserved.

    The LAST message is left intact — it's the current user command,
    which must reach the LLM in its original form for the admin's
    lookup to work. (Audit-logging of the current command happens
    separately in :meth:`MainChatService.run_command`.)

    The input list is never mutated; a new list is returned.
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

    The chat loop returns ``(response, messages)`` so the caller can
    persist ``messages`` and thread it back on the next turn. The
    messages list accumulates: the user's commands (which the audit
    layer flags but does NOT rewrite), tool-result outputs that the
    gate has already scrubbed, and provider-specific assistant
    items. Storing the list verbatim puts every prior turn's user
    PII into the host app's DB / session store.

    This helper walks the whole messages structure through
    :meth:`PiiService.scrub` so any PII the user typed is replaced
    with ``[redacted:<pattern>]`` placeholders before persistence.
    The scrubbed structure is safe to store; on the next turn,
    callers can thread the scrubbed messages back into
    ``MainChatService.run_command(..., input_messages=scrubbed)``
    without losing conversational continuity.

    The original ``messages`` is never mutated — a new structure
    comes back.
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
