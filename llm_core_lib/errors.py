"""Typed errors raised across ``llm_core_lib``.

All errors derive from :class:`LlmError` so callers can catch the whole
family with one ``except`` clause. The leaf subclasses are deliberately
specific — registry duplicate/missing, factory invalid-provider, config
validation, and provider-call wrapping each have their own type so a
caller can react to one without parsing exception messages.
"""
from __future__ import annotations


class LlmError(Exception):
    """Base for every error this package raises."""


class LlmConfigError(LlmError):
    """A provider / connection config is missing required fields or has
    a value the adapter cannot use (e.g. an OpenAI provider without an
    ``api_key``, a Bedrock provider without a ``region``)."""


class LlmInvalidProviderError(LlmError):
    """The factory was asked to build a provider whose id is not one of
    the supported backends (``openai`` / ``anthropic`` / ``bedrock``)."""


class LlmDuplicateConnectionError(LlmError):
    """``LlmConnectionRegistry.register`` was called twice with the same
    connection id."""


class LlmMissingConnectionError(LlmError):
    """``LlmConnectionRegistry.get`` / ``unregister`` was called with a
    connection id that has not been registered."""


class LlmProviderError(LlmError):
    """An adapter caught an exception from the underlying SDK call and
    re-raised it through this normalized error type so callers don't
    have to know which SDK was used."""
