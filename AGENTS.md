# AGENTS Notes

> This is the canonical home of the fetch → validate → use rule. The
> workspace-wide pointer lives in `architecture.md` under "Coding
> conventions (workspace-wide, all Python repos)"; every sibling
> repo's AGENTS.md points back here.

## Config reads — always fetch → validate → use, in that order

Any method that reads from `DictConfig` (every `*ConnectionFactory.__init__`
and every `_build_client`) must be structured as three explicit
blocks, in this order:

1. **fetch** — pull every value the method needs out of the config
   into named locals at the top. One `config.get('key')` per value,
   **with no fallback default**. Inline defaults like
   `config.get('region', 'us-east-1')` are banned — they hide the
   requirement.
2. **validate** — every required key gets a None / falsy check
   followed by `raise LlmConfigError(...)`, in one contiguous block
   immediately after the fetch. Strings use truthy-check (rejects
   both `None` and `''`); numerics use `is None` (so `max_tokens=0`
   stays valid).
3. **use** — assign to `self.*`, call collaborators, build SDK
   clients. By this point every value is a named local that has
   passed validation.

Add `# 1. fetch` / `# 2. validate` / `# 3. use` markers — the
operator has explicitly accepted them and they double as anchors for
the next reviewer.

**No aliases. No fallbacks. No silent defaults.** Pick one canonical
key per value (`model`, not `model` OR `model_id`; `vision_model`,
not `vision_model_id`). If a value is missing from the config the
factory must raise `LlmConfigError` — never paper over it with a
default and never silently fall back to `model_id` or `model` for
the vision/embedding model. The matching `LlmConnectionConfig`
dataclass in `types.py` mirrors the requirement: every cross-provider
field (model, vision_model, embedding_model, max_tokens, temperature)
is a required positional field with no default.

See `BedrockConnectionFactory`, `AnthropicConnectionFactory`, and
`OpenAiConnectionFactory` for the canonical shape.

### The same pattern applies to SDK response parsing

Any method that *reads* fields off a raw SDK response object — the
`_invoke` / `_invoke_chat` methods in the three `*Connection` classes,
the `_extract_text` helper in `bedrock_connection.py`, the `embed`
return-shape pick in `bedrock_connection.py` — also follows fetch →
validate → use. Don't sprinkle `getattr(raw, 'choices', None)`,
`payload.get('embedding')`, or `getattr(block, 'text', '')` inside
loops, generator expressions, or final return statements. Pull
*every* field you'll touch into a named local at the top of the
method, normalize / decide which shape you got next, then build the
public `LlmCompletion` (or chosen embedding vector) last.

In a `for ... in raw_content_blocks` loop, fetch the per-iteration
fields into named locals (`block_type = getattr(block, 'type', None)`,
`block_text = getattr(block, 'text', '') or ''`) at the top of the
loop body instead of inlining them in the conditional / append.

## The connection-factory shape

Every backend in `llm_core_lib/connections/` follows the same shape —
this is the canonical layout for all three (OpenAI / Anthropic /
Bedrock) and any future backend:

- `*ConnectionFactory(core_lib.connection.ConnectionFactory)` —
  takes a `Mapping` config, builds the shared SDK client once in
  `__init__`, exposes `get()` returning a fresh `*Connection`.
- `*Connection` — `complete_text(prompt, system=None)`,
  `complete_vision(prompt, image_bytes, image_mime, system=None)`,
  `embed(text)` (where the SDK supports it), `close()`.
- All three return / accept the shared `LlmCompletion` envelope.

The pattern was lifted from `library-core-lib`'s original Bedrock
factory (now removed from that repo in favor of this one — see the
"Sibling repos" section in the README).

**Do not** reintroduce a generic provider ABC with
`chat(LlmChatRequest)` / `stream(...)` signatures here — the reviewer
preferred the connection-factory style and we standardized on it.
Stream support, multi-turn messages, tool calls, etc. belong in
separate per-call helpers if/when needed, not in a parallel API.

## No empty / re-export-only `__init__.py`

Same rule as the sibling core libs: `__init__.py` files carry **no**
imports. Every `__init__.py` is empty (Python package marker); the
package-root `llm_core_lib/__init__.py` holds only `__version__`. There
is no re-export facade.

In-tree code — and external consumers — import from the defining
submodule:

```python
from llm_core_lib.connections.bedrock_connection_factory import BedrockConnectionFactory   # ✓
from llm_core_lib.connections import BedrockConnectionFactory                              # ✗
from llm_core_lib import BedrockConnectionFactory                                          # ✗ (no root facade)
```

## `requirements.txt` is `core-lib` + `pydantic>=2.0`

`core-lib` carries the framework transitively (`SQLAlchemy`,
`alembic`, `omegaconf`, `hydra-core`, `boto3`, etc.).
`pydantic>=2.0` is required directly because
`llm_core_lib.safety.llm_view.LLMView` is a Pydantic v2 ``BaseModel``
with ``ConfigDict(extra='forbid', frozen=True)`` — that's the
allowlist contract the safety choke point depends on, and Pydantic
v2's ``extra='forbid'`` is what enforces it. The three LLM SDKs
(`openai`, `anthropic`, `boto3`) are declared in `extras_require`
so consumers opt in:

```bash
pip install 'llm-core-lib[openai]'
```

Do not add `openai` / `anthropic` / `boto3` to `requirements.txt`.

## SDK imports are lazy inside `_build_client`

Each factory's `_build_client` does the SDK import inside the method
body, **not** at module top. This keeps an OpenAI-only install from
importing the Anthropic + Bedrock trees on package import.

Tests inject a fake client through `config['client']`; the SDK build
path stays untouched by the suite.

## Errors are typed, not stringly-checked

Catch the specific error class (`LlmDuplicateConnectionError`,
`LlmMissingConnectionError`, `LlmInvalidProviderError`,
`LlmConfigError`, `LlmProviderError`) — do not parse error messages.
Every connection wraps SDK exceptions in `LlmProviderError` so
callers never need to import the underlying SDK's exception
hierarchy.

## No persistence in the registry

`LlmConnectionRegistry` is in-memory by design. Re-register on
process boot. Don't add a DB-backed implementation here — that
belongs in a host app or a separate library that *consumes* this one.
