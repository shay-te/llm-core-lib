"""Public contracts: provider id, completion envelope, connection config.

Shape mirrors the ``library-core-lib`` ``BedrockConnectionFactory`` /
``BedrockCompletion`` pattern so consumers of either lib see the same
vocabulary. The cross-provider :class:`LlmCompletion` carries the
single text + model + usage envelope every connection returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

try:
    from typing import Literal
except ImportError:  # pragma: no cover  (Python 3.7 backport — outside our supported range)
    from typing_extensions import Literal  # type: ignore[assignment]


LlmProviderId = Literal['openai', 'anthropic', 'bedrock']

# Runtime-checkable mirror of ``LlmProviderId``. Tests keep them in sync.
LLM_PROVIDER_IDS: Tuple[str, ...] = ('openai', 'anthropic', 'bedrock')


@dataclass(frozen=True)
class LlmCompletion:
    """Normalized completion envelope returned by every ``*Connection``.

    ``usage`` is the SDK's token-count dict (or ``None`` if the SDK
    didn't surface one) — kept as a plain ``dict`` rather than a
    typed ``LlmUsage`` because the keys vary subtly across SDKs and
    callers that care can dig in directly.
    """

    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LlmConnectionConfig:
    """What :class:`LlmConnectionRegistry` stores.

    ``provider`` selects which ``*ConnectionFactory`` to build.
    ``model`` is the chat / completion model id. Vision and embedding
    models default to the chat model id but can be overridden.

    Every backend-specific knob lives here as ``Optional`` and is
    passed to the matching factory; the factory validates which keys
    it actually needs (e.g. Bedrock requires ``region``, OpenAI
    requires ``api_key``).
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
    vision_model: Optional[str] = None
    embedding_model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_factory_config(self) -> Dict[str, Any]:
        """Render this config as the plain ``dict`` the connection
        factories consume. Strips ``id`` and ``provider`` (the registry
        + factory dispatch use those) and merges ``extra`` last so
        explicit overrides win.
        """
        out: Dict[str, Any] = {
            'model': self.model,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'organization': self.organization,
            'region': self.region,
            'access_key': self.access_key,
            'secret_key': self.secret_key,
            'endpoint_url': self.endpoint_url,
            'vision_model': self.vision_model,
            'embedding_model': self.embedding_model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
        }
        if self.extra:
            for k, v in self.extra.items():
                out[k] = v
        return out
