# llm-core-lib

Shared **LLM connection abstraction** for the Una workspace. Wraps
OpenAI, Anthropic, and AWS Bedrock behind one `ConnectionFactory` +
`*Connection` pair per backend, plus a small in-memory named
**connection registry** so other libraries and host apps can register
an LLM connection once and resolve a normalized
`core_lib.connection.ConnectionFactory` by id at call time.

The canonical Bedrock connection factory lives in this package, at
[`llm_core_lib/connections/bedrock_connection_factory.py`](llm_core_lib/connections/bedrock_connection_factory.py)
(also exported as `llm_core_lib.BedrockConnectionFactory`). The
matching `OpenAi` and `Anthropic` factories follow the same shape so a
caller who learns one picks up the others unchanged. `library-core-lib`
imports its Bedrock factory from here.

## Architecture

```
LlmCoreLib(CoreLib)             <- composition root
  └─ self.registry: LlmConnectionRegistry
                       │
                       ├─ register(LlmConnectionConfig)
                       ├─ get(connection_id)  ──▶ ConnectionFactory
                       └─ list / has / unregister / clear

create_connection_factory(LlmConnectionConfig) ──▶ ConnectionFactory
                                                    (OpenAi | Anthropic | Bedrock)

Each backend exposes:
    *ConnectionFactory(core_lib.connection.ConnectionFactory)
      __init__(config: Mapping)   # builds the shared SDK client once
      get()                       ──▶ *Connection

    *Connection
      complete_text(prompt, system=None)              ──▶ LlmCompletion
      complete_vision(prompt, image_bytes, ...)       ──▶ LlmCompletion
      embed(text)                                     ──▶ list[float]   (OpenAi / Bedrock)
      close()                                         (no-op for these SDKs)
      .model_id / .vision_model_id / .embedding_model

Errors:
    LlmError, LlmConfigError, LlmInvalidProviderError,
    LlmDuplicateConnectionError, LlmMissingConnectionError, LlmProviderError
```

Provider SDKs (`openai`, `anthropic`, `boto3`) are imported lazily
inside each factory's `_build_client`, so installs that only use one
backend never pay the import cost of the others. Tests inject fake
clients through `config['client']`; the SDK-instantiation branches are
marked `pragma: no cover` because they need the real SDKs + credentials
to exercise.

## Installation

```bash
pip install llm-core-lib                # core only
pip install 'llm-core-lib[openai]'      # + openai SDK
pip install 'llm-core-lib[anthropic]'   # + anthropic SDK
pip install 'llm-core-lib[bedrock]'     # + boto3
pip install 'llm-core-lib[all]'         # all three
```

## Provider-factory example

The cross-provider factory hands you the matching
`*ConnectionFactory`:

```python
import os
from llm_core_lib import (
    LlmConnectionConfig,
    create_connection_factory,
)

factory = create_connection_factory(LlmConnectionConfig(
    id='_',                       # id is unused outside the registry
    provider='openai',
    model='gpt-4.1-mini',
    api_key=os.environ['OPENAI_API_KEY'],
))

conn = factory.get()
try:
    completion = conn.complete_text('Write a short welcome message.')
    print(completion.text)
finally:
    conn.close()
```

Or build a single backend factory directly:

```python
from llm_core_lib import BedrockConnectionFactory

factory = BedrockConnectionFactory({
    'model_id': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
    'region': 'us-east-1',
    'embedding_model': 'amazon.titan-embed-text-v1',
})

conn = factory.get()
try:
    print(conn.complete_text('hello').text)
    print(conn.embed('vectorize me'))
finally:
    conn.close()
```

## Connection-registry example

```python
import os
from llm_core_lib import (
    LlmConnectionConfig,
    LlmConnectionRegistry,
)

registry = LlmConnectionRegistry()

registry.register(LlmConnectionConfig(
    id='openai-default',
    provider='openai',
    model='gpt-4.1-mini',
    api_key=os.environ['OPENAI_API_KEY'],
))
registry.register(LlmConnectionConfig(
    id='anthropic-default',
    provider='anthropic',
    model='claude-3-5-sonnet-latest',
    api_key=os.environ['ANTHROPIC_API_KEY'],
))
registry.register(LlmConnectionConfig(
    id='bedrock-prod',
    provider='bedrock',
    model='anthropic.claude-3-5-sonnet-20241022-v2:0',
    region='us-east-1',
    embedding_model='amazon.titan-embed-text-v1',
))

factory = registry.get('openai-default')   # ConnectionFactory
conn = factory.get()                       # OpenAiConnection
try:
    completion = conn.complete_text(
        'Write a short welcome message.',
    )
finally:
    conn.close()
```

Factories are built **at register time** (config errors surface at
boot, not at first call) and **cached** — repeated
`registry.get(id)` returns the same `ConnectionFactory`. Each
`factory.get()` returns a fresh `Connection` wrapping the shared SDK
client.

## `LlmCoreLib` (Hydra-friendly composition root)

```python
from llm_core_lib import LlmCoreLib

# Hydra DictConfig, a plain dict, or a namespace whose nested
# attribute/item access resolves `core_lib.llm.connections` all work.
core = LlmCoreLib(cfg)

factory = core.registry.get('openai-default')
conn = factory.get()
try:
    conn.complete_text(...)
finally:
    conn.close()
```

Reads `core_lib.llm.connections` (see `example_data.yaml`) and
pre-registers every entry on the shared registry.

## Testing / no-network note

All tests are **mocked**. Each factory accepts a `client` key in its
config dict; the test suite uses
[`llm_core_lib/tests/fakes.py`](llm_core_lib/tests/fakes.py) which
mirrors the exact attribute surfaces of `openai.OpenAI`,
`anthropic.Anthropic`, and a Bedrock-runtime boto3 client. No SDK
needs to be installed to run the suite, and no test reaches the
network.

When `llm-core-lib` is installed via `pip` the test command is just:

```bash
python -m unittest discover -s llm_core_lib/tests -p 'test_*.py'
```

When running from an uninstalled checkout in the Kato workspace
(`core-lib` not on `sys.path`), prepend `PYTHONPATH`:

```bash
PYTHONPATH=/Users/shaytessler/Desktop/dev_kato/UNA-2719/core-lib:/Users/shaytessler/Desktop/dev_kato/UNA-2719/llm-core-lib \
    python -m unittest discover -s llm_core_lib/tests -p 'test_*.py'
```

Coverage targets the package modules (`llm_core_lib/*.py`,
`llm_core_lib/connections/*.py`); the SDK-build branches are marked
`pragma: no cover` because they require the real SDKs + credentials.

## Sibling repos

`library-core-lib` consumes `llm_core_lib.BedrockConnectionFactory`
directly (`from llm_core_lib import BedrockConnectionFactory`); the
local `library_core_lib.connections.bedrock_connection_factory` module
has been removed in favor of this one. Other siblings can do the same
when they need an LLM backend.

## Architecture boundaries

- This library never imports a host application package.
- Backend SDK imports (`openai`, `anthropic`, `boto3`) are
  function-local inside each factory — the base package depends on
  none of them at import time.
- The registry holds no persistence; re-register on process start.
- Connection config carries provider-specific overflow in an `extra`
  dict so new SDK knobs don't require a contract bump.
