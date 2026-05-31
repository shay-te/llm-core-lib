# Kato integration recipe — `agent-core-lib` + `llm-core-lib`

This is a **documentation-only** recipe. None of the code below lives
in `llm-core-lib`; it describes how Kato (or any host app driving an
agent workflow) composes the two libraries.

The rule, restated:

```
caller / Kato workflow
   ├─ agent-core-lib       ──▶ produces (prepared_prompt, prepared_system)
   └─ llm-core-lib         ──▶ sends them via a *Connection

llm-core-lib  ──▶ core-lib  (only).
llm-core-lib  ──/▶ agent-core-lib   (never — enforced by tests/test_boundary.py).
```

## 1. Capabilities Kato should use BEFORE calling `llm-core-lib`

The list below maps every relevant `agent-core-lib` helper to the
pipeline stage where Kato should call it. Helper paths assume the
`agent_core_lib.helpers.*` namespace; the public surface is described
in `agent-core-lib`'s own AGENTS.md.

### 1.1 Workspace / repo scope (every agent call)

| Helper                                                                          | Produces                                                            |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `agent_prompt_utils.workspace_scope_block(allowed_paths, extra_refusal_guidance='')` | Strict workspace-boundary block for the agent's task scope.         |
| `agent_prompt_utils.workspace_inventory_block(cwd, additional_dirs)`            | Lists the repositories the agent is allowed to see.                 |
| `agent_prompt_utils.forbidden_repository_guardrails_text(raw_value=None)`       | Text block describing forbidden-repo boundaries.                    |
| `agent_prompt_utils.prepend_forbidden_repository_guardrails(prompt, raw_value=None)` | Prefixes the user prompt with the forbidden-repo protocol.          |
| `agent_prompt_utils.prepend_chat_workspace_context(...)`                        | **Top-level convenience** — prepends continuity + inventory + forbidden-repo blocks in one call. Use this if you don't need to handle the pieces yourself. |
| `agent_prompt_utils.repository_scope_text(task, prepared_task=None)`            | Repository-specific scope instruction for a task with a `repository_id`. |

Caller-injected guidance: every host can pass its own refusal text via
`workspace_scope_block(..., extra_refusal_guidance=host_text)` (or, at
the `AgentCoreLib` constructor level, via `workspace_refusal_guidance`).
That string flows through `agent-core-lib` and ends up in the prepared
`system`; `llm-core-lib` never knows it exists.

### 1.2 Security / policy guardrails

| Helper                                                  | Produces                                                          |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| `agent_prompt_utils.security_guardrails_text()`         | Pure guardrails text block on credential / inspection safety.     |
| `credential_scan.scan_text_for_credentials_and_phishing(text, *, logger, context_label)` | Detective-only WARNING log; does NOT block. Run AFTER the agent runs, not before the LLM call. |

### 1.3 AGENTS.md / architecture / lessons rendering

| Helper                                                                       | Produces                                                                   |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `agents_instruction_utils.agents_instructions_for_path(workspace_path, *, repository_id='')` | Walks AGENTS.md files in the workspace, wraps them with context framing.   |
| `agents_instruction_utils.repository_agents_instructions_text(repositories)` | Same surface but takes pre-resolved Repository objects.                    |
| `architecture_doc_utils.read_architecture_doc(path, *, logger=None)`         | Reads + wraps the architecture doc with a living-document directive.       |
| `lessons_doc_utils.read_lessons_file(path, *, logger=None)`                  | Reads + wraps the lessons doc with a constraint directive.                 |
| `cached_file_render.cached_file_render(path, renderer, *, ...)`              | Mtime/size-based cache used internally by the architecture / lessons helpers — Kato can reuse for its own doc renders. |

### 1.4 Task / review framing

| Helper                                                                          | Produces                                                                                |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `agent_prompt_utils.task_branch_name(task, prepared_task=None)`                 | Canonical branch name for a task.                                                       |
| `agent_prompt_utils.task_conversation_title(task, suffix='')`                   | Task-scoped conversation title for logging / UI.                                        |
| `agent_prompt_utils.agents_instructions_text(prepared_task=None)`               | Extract AGENTS.md text from a prepared task object.                                     |
| `agent_prompt_utils.review_conversation_title(comment, task_id='', task_summary='')` | Title for a review-comment-driven conversation.                                          |
| `agent_prompt_utils.review_comment_location_text(comment)`                      | `file:line` location for one review comment.                                            |
| `agent_prompt_utils.review_comment_code_snippet(comment, workspace_path, *, context_lines=3)` | Renders the source snippet around the commented line.                                   |
| `agent_prompt_utils.review_comment_context_text(comment)`                       | Extracts prior comments in the thread (filters self-replies).                           |
| `agent_prompt_utils.review_comments_batch_text(comments, workspace_path='')`    | **Top-level convenience** for review mode — formats every comment with context + snippets. |
| `agent_prompt_utils.review_repository_context(comment)`                         | Returns the repository-id context string for a review comment.                          |

### 1.5 Resume / continuity

| Helper                                                                          | Produces                                                                                |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `resume_prompt_utils.ResumePromptInputs` (dataclass)                            | Container for the session snapshot Kato passes to the renderer.                         |
| `resume_prompt_utils.build_inputs_from_session(*, task_id, task_summary, branch_name, workspace_path, repository_paths, recent_events, agent_session_id='', max_recent_assistant=6)` | Adapter that turns live session events into `ResumePromptInputs`.                       |
| `resume_prompt_utils.render_resume_prompt(inputs)`                              | Renders the markdown snapshot that can be pasted into another model to continue.        |
| `agent_prompt_utils.chat_continuity_ground_truth_block(*, is_resumed_session)` | "Conversation history is the source of truth" block to inject when continuing a session. |
| `session_id_utils.fix_session_id(value)` / `has_session_id` / `same_session_id` / `read_session_id_from{,_mapping}` | Session-id normalization + comparison.                                                  |

### 1.6 Result handling (after the agent runs — still NOT inside `llm-core-lib`)

| Helper                                                                          | Produces                                                                |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `result_utils.openhands_success_flag(payload, *, default=False)`                | Extracts the success boolean from an OpenHands payload.                  |
| `result_utils.openhands_session_id(payload)`                                    | Extracts the session id (checks the three known payload keys).           |
| `result_utils.build_openhands_result(payload, *, branch_name='', summary_fallback='', default_commit_message=None, default_success=False)` | Builds the normalized result dict from an OpenHands payload.             |
| `text_utils.normalized_text(value)` / `condensed_text(value)` / `text_from_attr` / `text_from_mapping` | Text-normalization helpers Kato can reuse for its own response shaping.  |

## 2. Capabilities Kato MUST NOT route through `llm-core-lib`

These are real `agent-core-lib` capabilities, but they belong to the
caller — not to a transport library. Routing them through
`llm-core-lib` would either inject a forbidden dependency or duplicate
agent runtime behavior.

- `agent_core_lib.agent_core_lib.AgentCoreLib` (composition root with
  `self.agent: AgentProvider`) — Kato builds and owns this; it is the
  *agent runtime*, not an LLM transport.
- `agent_core_lib.client.agent_client_factory.AgentClientFactory` and
  `resolve_platform` — backend-agent factory (Claude / Codex /
  OpenHands selection). `llm-core-lib` knows nothing about agent
  backends.
- `agent_core_lib.platform.AgentPlatform` — the agent-backend enum;
  unrelated to `LlmProviderId`.
- Anything inside `claude_core_lib`, `codex_core_lib`,
  `openhands_core_lib`, `sandbox_core_lib` — agent-backend runtimes.
- Anything Kato-specific (UI text, refusal copy, workflow
  orchestration, ticket integrations) — keep in Kato, optionally
  passed through agent-core-lib's `extra_refusal_guidance` /
  `workspace_refusal_guidance` channels.

## 3. End-to-end pseudocode (Kato side)

```python
# kato/workflows/run_agent_task.py  (illustrative; not in any core-lib)

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
from agent_core_lib.helpers.lessons_doc_utils import read_lessons_file
from agent_core_lib.helpers.resume_prompt_utils import (
    build_inputs_from_session,
    render_resume_prompt,
)


def run(kato_task, *, llm_registry, workspace_path, repo_paths,
        architecture_doc_path, lessons_doc_path,
        host_refusal_guidance, recent_events=()):

    # ---- 1. Build the system block (caller side) ----
    scope = workspace_scope_block(
        allowed_paths=[workspace_path],
        extra_refusal_guidance=host_refusal_guidance,
    )
    prepared_system = '\n\n'.join(filter(None, [
        scope,
        security_guardrails_text(),
        agents_instructions_for_path(workspace_path),
        read_architecture_doc(architecture_doc_path),
        read_lessons_file(lessons_doc_path),
    ]))

    # ---- 2. Build the user prompt (caller side) ----
    prepared_prompt = prepend_chat_workspace_context(
        kato_task.prompt,
        cwd=workspace_path,
        additional_dirs=repo_paths,
        is_resumed_session=bool(recent_events),
    )

    # ---- 2b. If resuming, prepend the resume snapshot ----
    if recent_events:
        inputs = build_inputs_from_session(
            task_id=kato_task.id,
            task_summary=kato_task.summary,
            branch_name=kato_task.branch,
            workspace_path=workspace_path,
            repository_paths=repo_paths,
            recent_events=recent_events,
        )
        prepared_prompt = render_resume_prompt(inputs) + '\n\n' + prepared_prompt

    # ---- 3. Send through llm-core-lib (transport only) ----
    factory = llm_registry.get('bedrock-prod')
    with factory.get() as conn:
        completion = conn.complete_text(
            prompt=prepared_prompt,
            system=prepared_system,
        )

    return completion
```

## 4. Why this split

- **One direction, ever.** `llm-core-lib` evolving cannot break
  `agent-core-lib`; `agent-core-lib` evolving cannot force a
  `llm-core-lib` redeploy. If the dependency arrow ever pointed the
  other way, an OpenAI SDK bump could cascade into agent-runtime tests.
- **Testability.** `llm-core-lib`'s suite injects fake SDK clients via
  `config['client']` and runs in seconds, with no agent context to
  mock. `agent-core-lib` tests its prompt helpers without touching
  any LLM.
- **Reusability.** A non-agent caller (e.g. a CRM domain service)
  doesn't have to install `agent-core-lib` or any agent backend just
  to send a single completion request.
- **Single audit point for transport.** SDK choice, retries, payload
  shape, response normalization all live in one library — no agent
  workflow can quietly reshape a Bedrock request.

## 5. Validation Kato should run

After wiring the two libraries together, Kato (or any consumer) can
sanity-check the boundary the same way `llm-core-lib`'s suite does:

```bash
# Boundary grep — must return no source-import matches:
rg "^\s*(?:from|import)\s+(agent_core_lib|claude_core_lib|codex_core_lib|openhands_core_lib|kato_core_lib)\b" llm_core_lib/

# llm-core-lib's own boundary + verbatim-forward + vision/embed tests:
python -m unittest discover -s llm_core_lib/tests -p 'test_*.py'
```
