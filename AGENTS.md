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

## `requirements.txt` is just `core-lib`

`core-lib` carries the framework transitively (`SQLAlchemy`,
`alembic`, `omegaconf`, `hydra-core`, `boto3`, etc.). **No other
runtime dependency lives here** — in particular this repo does NOT
depend on Pydantic. The transport-layer ``LLMView`` marker in
``llm_core_lib.safety.llm_view`` is a plain Python class (no Pydantic
import); the Pydantic-backed concrete view with
``ConfigDict(extra='forbid', frozen=True)`` is in
``agent_core_lib.safety.llm_view`` and that's where the Pydantic
dependency lives. The split keeps this repo a pure transport library
and respects the boundary test (``test_boundary.py``) that forbids
``agent_core_lib`` imports from this side.

The three LLM SDKs (`openai`, `anthropic`, `boto3`) are declared in
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

## Test file organization — one TestCase per file, filename mirrors the class

**Every new test file owns exactly one `unittest.TestCase` subclass,
and the filename is the snake_case form of that class name.** This is
a workspace-wide rule — see the "Test file organization" sub-section
of "Coding conventions (workspace-wide, all Python repos)" in
`architecture.md` for the full rationale, the helper-module pattern,
and the canonical examples.

Inside this repo: any new file under `llm_core_lib/tests/` follows
the rule. The safety subpackage is the first area to land under it
(`test_to_llm_payload_accepts_llm_view.py`,
`test_run_tool_error_path.py`, etc. — one TestCase per file, shared
fixtures like the test `_UserLLMView` type or the mock SDK clients in
a sibling `<topic>_helpers.py` module without a `test_` prefix).
Genuinely pre-existing multi-class files (`test_boundary.py`,
`test_exports.py`) are **not** required to be split retroactively —
apply the rule forward, with new files and any time you're materially
touching an old one. The `test_safety_*` set was added in this PR
(UNA-2727) and still has multiple TestCase classes per file
(`test_safety_llm_view.py` 6, `test_safety_payload_gate.py` 6,
`test_safety_adversarial.py` 13); those should be split into
one-TestCase-per-file the next time they're materially touched —
documented as a debt, not endorsement.

## Tests prefer real collaborators over mocks

**Mock at infrastructure boundaries, not at internal seams.**
Workspace-wide rule — see the "Tests prefer real collaborators over
mocks" sub-section of "Coding conventions (workspace-wide, all
Python repos)" in `architecture.md` for the full rule.

This repo is the **canonical example** for the workspace-wide rule —
the `*ConnectionFactory` / `*Connection` tests in
`llm_core_lib/tests/test_connections.py` run the real factory and the
real connection end-to-end; only the SDK client (OpenAI / Anthropic /
Bedrock) is mocked, via the existing fakes under
`llm_core_lib/tests/mock/` (`MockOpenAIClient`,
`MockAnthropicClient`, `MockBedrockClient`). The response-parsing
logic, the prompt-forwarding-verbatim guarantees, the registry
plumbing — all of those exercise the real code paths against
synthetic SDK payloads.

For new tests in this repo:
- The SUT's direct collaborators (a factory's `Connection`, a
  registry's stored configs) are pure Python — wire the real types.
- The legitimate mock surfaces here are the **SDK client** (use the
  `MockOpenAIClient` / `MockAnthropicClient` / `MockBedrockClient`
  shape — they're already in `tests/mock/`), the **logger**, and the
  **clock** if a test asserts on timing.
- Pre-existing tests are **not** required to be rewritten — apply
  the rule forward, with new tests and any time you're materially
  rewriting an old one.
