# AGENTS Notes

## No empty / re-export-only `__init__.py`

Same rule as the sibling repos: do **not** create `__init__.py` files
whose only content is `from .x import Y` aggregators. Either:

- Leave the `__init__.py` empty (Python package marker).
- OR — only for the package-root `llm_core_lib/__init__.py` — list
  *intentional* public exports. The root init is the documented
  exception because it carries `__version__` and the public API
  surface.

Import from the defining submodule at the consumer site:

```python
from llm_core_lib.providers.openai_provider import OpenAiLlmProvider   # ✓
from llm_core_lib.providers import OpenAiLlmProvider                   # ✗ (don't aggregate here)
```

The package root re-exports `OpenAiLlmProvider` for *user* convenience
(`from llm_core_lib import OpenAiLlmProvider`), but in-tree code uses
the defining submodule path.

## `requirements.txt` is just `core-lib`

Every other dep (`SQLAlchemy`, `alembic`, `omegaconf`, `hydra-core`,
`boto3`, etc.) comes in transitively through `core-lib`. The three LLM
SDKs (`openai`, `anthropic`, `boto3`) are declared in
`extras_require` so consumers opt in:

```bash
pip install 'llm-core-lib[openai]'
```

Do not add `openai` / `anthropic` / `boto3` to `requirements.txt`.

## Provider SDK imports are lazy

Each adapter's `_resolved_client()` does the SDK import inside the
method body, **not** at module top. This keeps an OpenAI-only install
from importing the Anthropic + Bedrock trees on package import.

Tests inject a fake client through the adapter constructor's
`client=` kwarg; the SDK import path stays untouched by the suite.

## Errors are typed, not stringly-checked

Catch the specific error class (`LlmDuplicateConnectionError`,
`LlmMissingConnectionError`, `LlmInvalidProviderError`,
`LlmConfigError`, `LlmProviderError`) — do not parse error messages.
Every adapter wraps SDK exceptions in `LlmProviderError` so callers
never need to import the underlying SDK's exception hierarchy.

## No persistence in the registry

`LlmConnectionRegistry` is in-memory by design. Re-register on
process boot. Don't add a DB-backed implementation here — that belongs
in a host app or a separate library that *consumes* this one.
