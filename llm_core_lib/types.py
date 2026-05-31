"""Public contracts: messages, requests, responses, configs.

Mirrors the TypeScript-style shapes in the task spec:

  - ``LlmProviderId``         — Literal of the supported backend ids.
  - ``LlmRole``               — Literal of message roles.
  - ``LlmMessage``            — one item in a chat request.
  - ``LlmChatRequest``        — chat call input.
  - ``LlmChatResponse``       — normalized output across providers.
  - ``LlmStreamEvent``        — delta / stop / error envelope for streaming.
  - ``LlmUsage``              — normalized token counts.
  - ``LlmProviderConfig``     — what :func:`create_llm_provider` consumes.
  - ``LlmConnectionConfig``   — what :class:`LlmConnectionRegistry` stores
                                (provider config + the registry's lookup id).

Every dataclass is frozen so accidental mutation by adapters can't drift
a config behind a caller's back. ``extra`` is the only mutable field —
it's a plain ``dict`` for provider-specific overflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:  # pragma: no cover  (Python 3.7 backport — not in our supported range)
    from typing_extensions import Literal  # type: ignore[assignment]


LlmProviderId = Literal['openai', 'anthropic', 'bedrock']
LlmRole = Literal['system', 'user', 'assistant']
LlmStreamEventType = Literal['delta', 'stop', 'error']

# Runtime-checkable tuple matching ``LlmProviderId``. Keep in sync with
# the Literal above; the factory uses this for validation, the tests
# enforce parity.
LLM_PROVIDER_IDS: Tuple[str, ...] = ('openai', 'anthropic', 'bedrock')


@dataclass(frozen=True)
class LlmMessage:
    """One message in a chat request.

    ``role`` is constrained at type-check time to ``LlmRole``; at
    runtime any string is accepted so adapters can forward
    provider-specific roles (e.g. ``tool``) if a host extends the
    contract.
    """
    role: str
    content: str


@dataclass(frozen=True)
class LlmUsage:
    """Normalized token counts. Providers that don't return usage leave
    this ``None`` on the response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LlmChatRequest:
    """Inputs to :meth:`LlmProvider.chat` / :meth:`LlmProvider.stream`.

    ``model`` overrides the provider's default model when set; ``stop``
    accepts either a single string or a list of strings (adapters
    normalize to whatever shape the SDK expects).
    """
    messages: List[LlmMessage] = field(default_factory=list)
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Any] = None      # str | List[str]
    system: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmChatResponse:
    """Normalized output from any provider. ``content`` is the assistant
    text; ``finish_reason`` and ``usage`` are best-effort (None if the
    underlying SDK didn't surface them)."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[LlmUsage] = None


@dataclass(frozen=True)
class LlmStreamEvent:
    """Single event in a streaming chat. ``type``:

    - ``'delta'``  — incremental text in ``content``.
    - ``'stop'``   — final event; ``finish_reason`` populated.
    - ``'error'``  — terminal failure; ``content`` carries the error message.

    Providers that don't stream natively still emit one delta + one stop
    via the default ``LlmProvider.stream`` implementation, so callers
    can use the streaming interface uniformly.
    """
    type: str
    content: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class LlmProviderConfig:
    """What :func:`create_llm_provider` consumes.

    Provider-specific knobs are all here as ``Optional`` fields; the
    adapter constructors validate which are required. ``extra`` is the
    overflow bag for fields the contract doesn't model yet (kept so
    early adopters don't need a contract bump for one-off knobs).
    """
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmConnectionConfig:
    """What :class:`LlmConnectionRegistry` stores.

    Same shape as :class:`LlmProviderConfig` plus an ``id`` for lookup.
    Kept as a sibling dataclass (rather than subclass) so frozen-dataclass
    field-order rules don't force ``id`` to carry a default value.
    """
    id: str
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_provider_config(self) -> LlmProviderConfig:
        """Strip ``id`` and return the equivalent provider config."""
        return LlmProviderConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            region=self.region,
            access_key=self.access_key,
            secret_key=self.secret_key,
            endpoint_url=self.endpoint_url,
            extra=dict(self.extra),
        )
