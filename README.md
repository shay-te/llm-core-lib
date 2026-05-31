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
      __init__(config: DictConfig)  # builds the shared SDK client once
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

## Boundary with `agent-core-lib`

`llm-core-lib` is the **transport layer** for LLM calls — it owns the
SDK clients, the provider payload shape, the response normalization,
and the connection registry. It does **not** know about agent
workflows, repo context, review mode, resumed sessions, AGENTS.md
files, or backend-specific (Codex / Claude-agent / OpenHands) prompt
shaping. Those belong in `agent-core-lib`.

> **`llm-core-lib` must not import `agent-core-lib`** (or
> `claude_core_lib` / `codex_core_lib` / `openhands_core_lib` /
> `kato_core_lib`). This is asserted at test time by
> [`tests/test_boundary.py`](llm_core_lib/tests/test_boundary.py),
> which AST-parses every module under `llm_core_lib/` and fails the
> suite on any forbidden import.

### Responsibilities (what `llm-core-lib` owns)

- Provider connection **factories** (`OpenAiConnectionFactory`,
  `AnthropicConnectionFactory`, `BedrockConnectionFactory`).
- The in-memory **`LlmConnectionRegistry`** (named, eagerly-built,
  cached).
- **Text completion** calls (`complete_text(prompt, system=None)`).
- **Vision completion** calls (`complete_vision(prompt, image_bytes,
  image_mime, system=None)`) — including image bytes → base64 / MIME
  packaging and the provider-specific content-block shape.
- **Embedding** calls (`embed(text)`) on OpenAI / Bedrock.
- **Provider request payload formatting** — message role wrapping,
  Anthropic image-first ordering, Bedrock `invoke_model` body, OpenAI
  `chat.completions` envelope.
- **Provider response normalization** to the shared `LlmCompletion`
  envelope (`text`, `model`, `usage`).
- **Lazy SDK imports** inside each factory's `_build_client` — an
  OpenAI-only install never pays the Anthropic / Bedrock import cost.
- **Provider config handling** via `LlmConnectionConfig` plus an
  `extra` overflow dict for SDK-specific knobs.

### Non-responsibilities (what `llm-core-lib` does NOT own)

- Agent **prompt preparation** of any kind.
- Loading or rendering **AGENTS.md** files.
- Loading or rendering **architecture-doc** / **lessons-doc** files.
- **Repo / workspace scope** guardrails and forbidden-folder text.
- **Review-comment** context (file/line snippets, prior-comment
  threading, batch formatting).
- **Task / ticket / PR / branch** framing and conversation titles.
- **Resume / continuity** prompt rendering or session-id reconciliation.
- **Claude-agent / Codex / OpenHands** runtime behavior, success-flag
  extraction, or backend-specific result shapes.
- **Kato-specific** injected guidance, refusal text, or workflow logic.
- **System-prompt policy** beyond accepting a caller-provided `system`
  string and forwarding it verbatim into the provider payload.

The dependency direction is strict and enforced by a boundary test
(`tests/test_boundary.py`):

```
app / workflow / Kato
    │
    ├─ agent-core-lib   ──▶ prepares (prompt, system) for agent workflows
    │     (repo scope, AGENTS.md, review framing, resume context,
    │      workspace inventory, guardrails, …)
    │
    └─ llm-core-lib     ──▶ sends already-prepared (prompt, system)
          (connection registry, provider payload, SDK calls,
           response normalization)

llm-core-lib  ──▶ core-lib  (only).
llm-core-lib  ──/▶ agent-core-lib   (never).
```

In code, the integration always looks the same — the caller composes
the prompt with `agent-core-lib`, then hands the already-prepared
strings to a `*Connection`. There are four canonical shapes.

#### A. Agent / coding-agent workflow

```python
# In the caller / workflow layer — NOT in llm_core_lib.
from agent_core_lib.helpers.agent_prompt_utils import (
    prepend_chat_workspace_context,
    security_guardrails_text,
    workspace_scope_block,
)
from agent_core_lib.helpers.agents_instruction_utils import (
    agents_instructions_for_path,
)
from agent_core_lib.helpers.architecture_doc_utils import (
    read_architecture_doc,
)

scope = workspace_scope_block(
    allowed_paths=[workspace_path],
    extra_refusal_guidance=host_specific_guidance,
)
guardrails = security_guardrails_text()

prepared_prompt = prepend_chat_workspace_context(
    raw_user_prompt,
    cwd=workspace_path,
    additional_dirs=repo_paths,
    is_resumed_session=False,
)
prepared_system = '\n\n'.join(filter(None, [
    scope,
    guardrails,
    agents_instructions_for_path(workspace_path),
    read_architecture_doc(architecture_doc_path),
]))

# Now hand the prepared strings to llm-core-lib:
factory = registry.get('bedrock-prod')
with factory.get() as conn:
    completion = conn.complete_text(
        prompt=prepared_prompt,
        system=prepared_system,
    )
```

The `*Connection` does no prompt building, no string concatenation, no
guardrail injection — `prompt` and `system` are passed verbatim into
the provider payload. If `agent-core-lib` produced an empty string,
that's what the provider sees. This is enforced by both the existing
connection tests and a dedicated verbatim-pass-through test in
`tests/test_boundary.py`.

#### B. Non-agent / domain workflow

`agent-core-lib` is **not** involved. Domain code owns the prompt.

```python
factory = registry.get('bedrock-prod')
with factory.get() as conn:
    completion = conn.complete_text(
        prompt='Summarize this customer email...\n\n' + email_text,
        system='You are a concise assistant.',
    )
```

#### C. Vision workflow

`agent-core-lib` may prepare the visual-task `prompt` and `system`
(same helpers as A). `llm-core-lib` still owns the image bytes,
MIME / base64 packaging, and the provider-specific content-block
shape (e.g. Anthropic puts the image block first; OpenAI uses
`image_url` with a `data:...;base64,...` URL).

```python
factory = registry.get('anthropic-default')
with factory.get() as conn:
    completion = conn.complete_vision(
        prompt=prepared_visual_prompt,    # caller-prepared
        image_bytes=open(path, 'rb').read(),
        image_mime='image/png',
        system=prepared_system,            # caller-prepared
    )
```

#### D. Embeddings

`agent-core-lib` is usually not involved — there's no prompt to shape.
The caller decides what text to embed; `llm-core-lib` makes the
provider call.

```python
factory = registry.get('openai-default')
with factory.get() as conn:
    vector = conn.embed('the quick brown fox')
```

### When to use which

| Workflow shape          | `agent-core-lib` involvement                                         | `llm-core-lib` involvement                                  |
| ----------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------- |
| Agent / coding-agent    | Prepares prompt + system from repo / task / review / resume context. | Sends the prepared pair via the matching `*Connection`.     |
| Non-agent / domain      | **Not used.** Domain code owns the prompt.                           | Sends the domain-prepared prompt via a `*Connection`.       |
| Vision                  | May prepare the visual-task prompt + system.                         | Still owns image bytes, MIME/base64, payload, normalization. |
| Embeddings              | Usually not involved (no prompt shaping needed).                     | Owns `embed(text)` on OpenAI / Bedrock connections.          |

See [`INTEGRATION.md`](INTEGRATION.md) for the full Kato recipe — what
`agent-core-lib` helpers map to which step of the pipeline, and how
the four workflow shapes above are wired end-to-end.


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

with factory.get() as conn:
    completion = conn.complete_text('Write a short welcome message.')
    print(completion.text)
```

Every `*Connection` extends `core_lib.connection.Connection`, so the
context-manager form above is the idiomatic one — `__exit__` calls
`close()` for you and exceptions propagate normally. The explicit
`try / finally` form still works for callers that want manual
lifecycle control.

Or build a single backend factory directly:

```python
from llm_core_lib import BedrockConnectionFactory

factory = BedrockConnectionFactory({
    'model_id': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
    'region': 'us-east-1',
    'embedding_model': 'amazon.titan-embed-text-v1',
})

with factory.get() as conn:
    print(conn.complete_text('hello').text)
    print(conn.embed('vectorize me'))
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
with factory.get() as conn:                # OpenAiConnection
    completion = conn.complete_text(
        'Write a short welcome message.',
    )
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
- This library never imports `agent_core_lib`, `claude_core_lib`,
  `codex_core_lib`, `openhands_core_lib`, or `kato_core_lib` — agent
  workflows compose those upstream and pass the prepared
  `(prompt, system)` pair down. Enforced by
  [`tests/test_boundary.py`](llm_core_lib/tests/test_boundary.py).
- Backend SDK imports (`openai`, `anthropic`, `boto3`) are
  function-local inside each factory — the base package depends on
  none of them at import time.
- The registry holds no persistence; re-register on process start.
- Connection config carries provider-specific overflow in an `extra`
  dict so new SDK knobs don't require a contract bump.
