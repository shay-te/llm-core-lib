# AGENTS Notes — llm-core-lib

## Built on core-lib — read its rulebook first

This library is a `*-core-lib`. The shared architecture, conventions, and
scaffolding are **not** repeated here — they live in the `core-lib` package
(the `../core-lib/` checkout beside this repo, or the installed `core-lib`):

| Read | For |
|---|---|
| `core-lib/AGENTS.md` | **Read first.** The AGNOSTIC principles, then the canonical recipe (§0–§17): naming, folder tree, config, entities, DataAccess, services, composition root, observers, jobs, migrations, tests, build order, things to avoid, final checklist. |
| `core-lib/skills/` | Copy-paste scaffolding templates, one per core-lib part. |

**MANDATORY — before you create or modify any part below, first load the
matching core-lib skill** (open and follow it). This is a hard rule: match the
row and load the skill *before* writing code. Never write core-lib code from
memory when a matching skill exists.

| If you are about to… | You MUST first load |
|---|---|
| add or change an entity / table / model / column / nested enum | [`core-lib-entity`](../core-lib/skills/core-lib-entity/SKILL.md) |
| add or change a DataAccess / DAO / repository / query / get_by / list | [`core-lib-data-access`](../core-lib/skills/core-lib-data-access/SKILL.md) |
| add or change a Service / business logic / public method / caching | [`core-lib-service`](../core-lib/skills/core-lib-service/SKILL.md) |
| add or change an external client / provider / SDK / connection factory | [`core-lib-connection`](../core-lib/skills/core-lib-connection/SKILL.md) |
| add a migration / alter / create / drop a table, column, index, constraint | [`core-lib-migration`](../core-lib/skills/core-lib-migration/SKILL.md) |
| add / fix / restructure tests or raise coverage | [`core-lib-tests`](../core-lib/skills/core-lib-tests/SKILL.md) |

Everything below this line is **llm-core-lib-specific** — lessons that apply only to
this library. Anything generic belongs in `core-lib/AGENTS.md` instead, so every
core-lib inherits it.

---

## The connection-factory shape (this repo is the reference implementation)

Core-lib **§5.1** describes the shape; these are this repo's concrete
contracts. Every backend in `llm_core_lib/connections/` follows it — OpenAI,
Anthropic, Bedrock, and any future one:

- `*ConnectionFactory(core_lib.connection.ConnectionFactory)` — takes a
  `Mapping` config, builds the shared SDK client once in `__init__`, exposes
  `get()` returning a fresh `*Connection`.
- `*Connection` — `complete_text(prompt, system=None)`,
  `complete_vision(prompt, image_bytes, image_mime, system=None)`,
  `embed(text)` (where the SDK supports it), `close()`.
- All return / accept the shared `LlmCompletion` envelope.

`BedrockConnectionFactory`, `AnthropicConnectionFactory`, and
`OpenAiConnectionFactory` are the canonical examples of fetch → validate → use
(§1.1) — including their `_invoke` / `_invoke_chat` / `_extract_text` / `embed`
response parsing.

**Do not** reintroduce a generic provider ABC with `chat(LlmChatRequest)` /
`stream(...)` signatures — we standardized on the connection-factory style.
Stream support, multi-turn messages, and tool calls belong in separate per-call
helpers, not a parallel API.

## Config contract

Missing config raises `LlmConfigError` — never a default, never an alias chain
(`model`, not `model` OR `model_id`; `vision_model`, not `vision_model_id`).
The `LlmConnectionConfig` dataclass in `types.py` mirrors this: every
cross-provider field (`model`, `vision_model`, `embedding_model`, `max_tokens`,
`temperature`) is a required positional field with **no default**.

## Typed errors

Catch the specific class — never parse messages: `LlmDuplicateConnectionError`,
`LlmMissingConnectionError`, `LlmInvalidProviderError`, `LlmConfigError`,
`LlmProviderError`. Every connection wraps SDK exceptions in
`LlmProviderError` so callers never import the SDK's exception hierarchy
(core-lib §5.3).

## No Pydantic here — this is a pure transport library

`core-lib` carries the framework transitively; **no other runtime dependency
lives here.** In particular this repo does NOT depend on Pydantic: the
transport-layer `LLMView` marker in `llm_core_lib.safety.llm_view` is a plain
Python class. The Pydantic-backed concrete view
(`ConfigDict(extra='forbid', frozen=True)`) lives in
`agent_core_lib.safety.llm_view`. The boundary test (`test_boundary.py`)
enforces the direction by forbidding `agent_core_lib` imports from this side —
the concrete instance of core-lib §2.4.

The three LLM SDKs (`openai`, `anthropic`, `boto3`) are `extras_require`
(`pip install 'llm-core-lib[openai]'`), imported lazily inside `_build_client`
(§5.2, §5.4). Tests inject a fake client through `config['client']`, so the SDK
build path stays untouched by the suite.

`LlmConnectionRegistry` is in-memory by design — re-register on process boot
(§5.5).

## Testing in this repo

Rules: core-lib **§7.1** (one `TestCase` per file) and **§7.2** (real
collaborators). This repo is the reference for §7.2 —
`llm_core_lib/tests/test_connections.py` runs the real factory and real
connection end-to-end; **only** the SDK client is mocked, via
`MockOpenAIClient` / `MockAnthropicClient` / `MockBedrockClient` in
`llm_core_lib/tests/mock/`.

- Legitimate mock surfaces here: the SDK client, the logger, and the clock.
- Shared fixtures go in a sibling `<topic>_helpers.py` (no `test_` prefix) —
  e.g. `safety_llm_view_helpers.py`, shared by `test_llm_view_is_a_class.py`
  and `test_llm_view_isinstance_check.py`.
- **Known debt:** `test_safety_payload_gate.py` (6 TestCases) and
  `test_safety_adversarial.py` (13) are still multi-class; split them the next
  time they're materially touched. `test_boundary.py` / `test_exports.py` are
  grandfathered.
