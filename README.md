# llm-core-lib

Shared **LLM provider abstraction** for the Una workspace. Wraps OpenAI,
Anthropic, and AWS Bedrock behind one provider interface plus a small
in-memory named **connection registry** so other libraries and host
apps can register an LLM connection once and resolve a normalized
provider by id at call time.

`llm-core-lib` deliberately does **not** ship its own model API client
implementations beyond thin adapters — provider SDKs (`openai`,
`anthropic`, `boto3`) are imported lazily inside each adapter, so an
install that only uses one backend never pays the import cost of the
others. All three SDKs are declared as `extras_require` (see
[Installation](#installation)).

## Architecture

```
LlmCoreLib(CoreLib)              <- composition root
  └─ self.registry: LlmConnectionRegistry
                       │
                       ├─ register(LlmConnectionConfig)
                       ├─ get(connection_id)  ──▶ LlmProvider
                       └─ list / has / unregister / clear

create_llm_provider(LlmProviderConfig) ──▶ LlmProvider
                                            (OpenAi | Anthropic | Bedrock)

LlmProvider  (ABC)
  ├─ chat(LlmChatRequest)   ──▶ LlmChatResponse
  └─ stream(LlmChatRequest) ──▶ Iterator[LlmStreamEvent]
                                 (default = one delta + one stop)

Adapters (one per backend, SDK imports lazy, client injectable for tests):
  - OpenAiLlmProvider      (openai.OpenAI().chat.completions.create)
  - AnthropicLlmProvider   (anthropic.Anthropic().messages.create)
  - BedrockLlmProvider     (boto3.client('bedrock-runtime').invoke_model,
                            Anthropic-shaped body)

Errors:
  LlmError, LlmConfigError, LlmInvalidProviderError,
  LlmDuplicateConnectionError, LlmMissingConnectionError, LlmProviderError
```

The package mirrors `agent-core-lib`'s shape: thin `CoreLib` subclass
+ one shared interface + lazy per-backend imports + a factory. Adapter
classes never reach into each other; the factory is the only seam
that knows about the union of backends.

## Installation

```bash
pip install llm-core-lib                # core only
pip install 'llm-core-lib[openai]'      # + openai SDK
pip install 'llm-core-lib[anthropic]'   # + anthropic SDK
pip install 'llm-core-lib[bedrock]'     # + boto3
pip install 'llm-core-lib[all]'         # all three
```

The adapter for a backend you didn't install is still importable; it
will raise `LlmConfigError("openai SDK not installed; ...")` the first
time `chat` is called without an injected client.

## Provider factory example

```python
from llm_core_lib import (
    LlmChatRequest,
    LlmMessage,
    LlmProviderConfig,
    create_llm_provider,
)

provider = create_llm_provider(LlmProviderConfig(
    provider='openai',
    model='gpt-4.1-mini',
    api_key=os.environ['OPENAI_API_KEY'],
))

response = provider.chat(LlmChatRequest(
    messages=[LlmMessage(role='user', content='Write a short welcome message.')],
    temperature=0.2,
))
print(response.content)
```

## Connection registry example

```python
import os
from llm_core_lib import (
    LlmChatRequest,
    LlmConnectionConfig,
    LlmConnectionRegistry,
    LlmMessage,
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
))

llm = registry.get('openai-default')
response = llm.chat(LlmChatRequest(
    messages=[LlmMessage(role='user', content='Write a short welcome message.')],
    temperature=0.2,
))
```

Provider instances are built **at register time** (config errors fire
at boot, not at first call) and **cached** — repeated `registry.get(id)`
returns the same instance.

## `LlmCoreLib` (Hydra-friendly composition root)

```python
from llm_core_lib import LlmCoreLib

# Hydra `DictConfig`, a plain dict, or a namespace whose nested
# attribute/item access resolves `core_lib.llm.connections` all work.
core = LlmCoreLib(cfg)
core.registry.get('openai-default').chat(...)
```

Reads `core_lib.llm.connections` (see `example_data.yaml`) and
pre-registers every entry on the shared registry.

## Testing / no-network note

All tests are **mocked**. The adapter constructors accept an injected
`client=` kwarg; the test suite uses
[`llm_core_lib/tests/fakes.py`](llm_core_lib/tests/fakes.py) which
mirrors the exact attribute surfaces of `openai.OpenAI`,
`anthropic.Anthropic`, and a Bedrock-runtime boto3 client. No SDK needs
to be installed to run the suite, and no test reaches the network.

```bash
python -m unittest discover -s llm_core_lib/tests -p 'test_*.py'
```

Coverage targets the package modules (`llm_core_lib/*.py`,
`llm_core_lib/providers/*.py`) — the adapter SDK-import branches are
marked `pragma: no cover` because they depend on which SDKs are
installed in the test environment.

## Sibling repo migrations — future work

`library-core-lib` currently has its own `BedrockConnectionFactory`
that wraps `boto3` directly. Migrating it (and any other sibling that
talks to an LLM SDK) to consume `llm-core-lib`'s registry is **planned
but out of scope for this pass** — this PR only stands up the new
package.

## Architecture boundaries

- This library never imports a host application package.
- Backend SDK imports (`openai`, `anthropic`, `boto3`) are
  function-local inside each adapter — the base package depends on
  none of them at import time.
- The registry holds no persistence; re-register on process start.
- Connection config carries provider-specific overflow in an `extra`
  dict so new SDK knobs don't require a contract bump.
