# AGENTS Notes

## Config reads — always fetch → validate → use, in that order

Any method that reads from `DictConfig` (every `*ConnectionFactory.__init__`
and every `_build_client`) must be structured as three explicit
blocks, in this order:

1. **fetch** — pull every value the method needs out of the config
   into named locals at the top. One `config.get('key')` per value.
   Required keys are pulled with no fallback (`config.get('key')`);
   genuinely-optional values with a real default may keep one
   (`config.get('temperature', 0.0)`).
2. **validate** — every None / falsy check + `raise LlmConfigError(...)`
   lives in one contiguous block immediately after the fetch.
3. **use** — assign to `self.*`, call collaborators, build SDK
   clients. By this point every value is a named local that has
   passed validation.

Add `# 1. fetch` / `# 2. validate` / `# 3. use` markers — they're
load-bearing for readability and the operator has explicitly accepted
them. Do **not** call `config.get(...)` from inside the "use" block;
hoist it up to the fetch block. The pattern applies recursively —
`_build_client` follows the same internal layout.

See `BedrockConnectionFactory`, `AnthropicConnectionFactory`, and
`OpenAiConnectionFactory` for the canonical shape.

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

Same rule as the sibling repos: do **not** create `__init__.py` files
whose only content is `from .x import Y` aggregators. Either leave
the `__init__.py` empty (Python package marker) or — only for the
package-root `llm_core_lib/__init__.py` — list *intentional* public
exports.

In-tree code imports from the defining submodule:

```python
from llm_core_lib.connections.bedrock_connection_factory import BedrockConnectionFactory   # ✓
from llm_core_lib.connections import BedrockConnectionFactory                              # ✗
```

The package-root `__init__.py` re-exports for user convenience
(`from llm_core_lib import BedrockConnectionFactory`); that's the
documented exception.

## `requirements.txt` is just `core-lib`

Every other dep (`SQLAlchemy`, `alembic`, `omegaconf`, `hydra-core`,
`boto3`, etc.) comes in transitively through `core-lib`. The three
LLM SDKs (`openai`, `anthropic`, `boto3`) are declared in
`extras_require` so consumers opt in:

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
