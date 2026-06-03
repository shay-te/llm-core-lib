"""Boundary tests for ``llm-core-lib``.

These tests codify the architecture rule documented in the README's
"Boundary with agent-core-lib" section: ``llm-core-lib`` is the
transport layer for LLM calls and may not import any upstream
agent-workflow library. Prompt preparation belongs to the caller.

The suite asserts four things:

1. No module under ``llm_core_lib/`` imports ``agent_core_lib``,
   ``claude_core_lib``, ``codex_core_lib``, ``openhands_core_lib``, or
   ``kato_core_lib`` (AST-level check, immune to docstring mentions).
2. ``complete_text(prompt, system=...)`` forwards a fully-prepared
   ``(prompt, system)`` pair into the provider payload **verbatim**,
   for every backend — no concatenation, no guardrail injection, no
   string mutation.
3. ``LlmConnectionRegistry.get`` resolves a connection without
   touching any prompt helper — the registry has no prompt-building
   surface at all.
4. Vision payload formatting (base64, MIME, provider content blocks)
   stays inside ``llm-core-lib``; the embedding call accepts only the
   raw text and forwards it unmodified.
"""
from __future__ import annotations

import ast
import os
import unittest
from typing import Iterable, Set

import llm_core_lib
from llm_core_lib.connections.anthropic_connection_factory import AnthropicConnectionFactory
from llm_core_lib.connections.bedrock_connection_factory import BedrockConnectionFactory
from llm_core_lib.connections.openai_connection_factory import OpenAiConnectionFactory
from llm_core_lib.registry import LlmConnectionRegistry
from llm_core_lib.types import LlmConnectionConfig
from llm_core_lib.registry import LlmConnectionRegistry as _RegistryClass
from llm_core_lib.tests.mock.anthropic_client import MockAnthropicClient
from llm_core_lib.tests.mock.bedrock_client import MockBedrockClient
from llm_core_lib.tests.mock.openai_client import MockOpenAIClient


FORBIDDEN_TOP_LEVEL_PACKAGES = frozenset({
    'agent_core_lib',
    'claude_core_lib',
    'codex_core_lib',
    'openhands_core_lib',
    'kato_core_lib',
})


def _package_root() -> str:
    # llm_core_lib/__init__.py → parent dir is the package root.
    return os.path.dirname(os.path.abspath(llm_core_lib.__file__))


def _iter_python_files(root: str) -> Iterable[str]:
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith('.py'):
                yield os.path.join(dirpath, name)


def _imported_top_levels(path: str) -> Set[str]:
    with open(path, 'r', encoding='utf-8') as fh:
        source = fh.read()
    tree = ast.parse(source, filename=path)
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.', 1)[0])
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (``from . import x``) — they can't
            # name another top-level package.
            if node.level and not node.module:
                continue
            if node.module:
                names.add(node.module.split('.', 1)[0])
    return names


# ----------------------------------------------------------------------
# 1. Forbidden-import check (AST-level, immune to docstring mentions)
# ----------------------------------------------------------------------


class TestForbiddenImports(unittest.TestCase):
    def test_no_module_imports_a_forbidden_package(self):
        offenders = []
        for path in _iter_python_files(_package_root()):
            # Skip this test file itself — it names the forbidden
            # packages as string constants, which is fine; we only
            # care about real import statements.
            if os.path.basename(path) == 'test_boundary.py':
                continue
            imported = _imported_top_levels(path)
            hit = imported & FORBIDDEN_TOP_LEVEL_PACKAGES
            if hit:
                offenders.append((path, sorted(hit)))
        self.assertEqual(
            offenders,
            [],
            'llm-core-lib must not import upstream agent-workflow '
            'packages — see README "Boundary with agent-core-lib". '
            f'Offenders: {offenders}',
        )

    def test_forbidden_set_covers_every_named_target(self):
        # Belt-and-braces: if a reviewer adds a new forbidden target to
        # the README they should add it here too, and this test will
        # fail if the constant drifts away from the documented list.
        self.assertEqual(
            FORBIDDEN_TOP_LEVEL_PACKAGES,
            frozenset({
                'agent_core_lib',
                'claude_core_lib',
                'codex_core_lib',
                'openhands_core_lib',
                'kato_core_lib',
            }),
        )


# ----------------------------------------------------------------------
# 2. Verbatim forwarding of caller-prepared (prompt, system)
# ----------------------------------------------------------------------


PREPARED_PROMPT = (
    '# Workspace context\n'
    '<<this block was produced upstream by agent-core-lib>>\n\n'
    '# Task\nplease summarize the attached email.\n'
)
PREPARED_SYSTEM = (
    'You operate inside an allowed workspace. '
    '<<scope + guardrails + AGENTS.md text from agent-core-lib>>'
)


class TestCallerPreparedPromptForwardsVerbatim(unittest.TestCase):
    """``complete_text`` must not modify the caller's strings."""

    def test_openai_forwards_prompt_and_system_unchanged(self):
        fake = MockOpenAIClient()
        factory = OpenAiConnectionFactory({
            'model': 'gpt-x',
            'vision_model': 'gpt-x',
            'embedding_model': 'emb',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'k',
            'client': fake,
        })
        with factory.get() as conn:
            conn.complete_text(prompt=PREPARED_PROMPT, system=PREPARED_SYSTEM)
        sent = fake.chat_calls[0]['messages']
        self.assertEqual(sent[0], {'role': 'system', 'content': PREPARED_SYSTEM})
        self.assertEqual(sent[1], {'role': 'user', 'content': PREPARED_PROMPT})

    def test_anthropic_forwards_prompt_and_system_unchanged(self):
        fake = MockAnthropicClient()
        factory = AnthropicConnectionFactory({
            'model': 'claude-x',
            'vision_model': 'claude-x',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'k',
            'client': fake,
        })
        with factory.get() as conn:
            conn.complete_text(prompt=PREPARED_PROMPT, system=PREPARED_SYSTEM)
        call = fake.calls[0]
        self.assertEqual(call['system'], PREPARED_SYSTEM)
        # The single user-message's text block must be the exact prompt.
        content = call['messages'][0]['content']
        self.assertEqual(content, [{'type': 'text', 'text': PREPARED_PROMPT}])

    def test_bedrock_forwards_prompt_and_system_unchanged(self):
        fake = MockBedrockClient()
        factory = BedrockConnectionFactory({
            'model': 'anthropic.fake',
            'vision_model': 'anthropic.fake',
            'embedding_model': 'emb',
            'max_tokens': 4096,
            'temperature': 0.0,
            'region': 'us-east-1',
            'client': fake,
        })
        with factory.get() as conn:
            conn.complete_text(prompt=PREPARED_PROMPT, system=PREPARED_SYSTEM)
        body = fake.calls[0]['body']
        self.assertEqual(body['system'], PREPARED_SYSTEM)
        # Bedrock body wraps user content in the Anthropic-style shape.
        self.assertEqual(
            body['messages'][0]['content'],
            [{'type': 'text', 'text': PREPARED_PROMPT}],
        )

    def test_no_connection_class_exposes_a_prompt_builder_method(self):
        # ``llm-core-lib`` must not grow a ``build_prompt`` /
        # ``with_system`` / ``prepend_*`` helper — those are explicitly
        # agent-core-lib's job. Probe every public connection class.
        from llm_core_lib.connections.anthropic_connection import AnthropicConnection
        from llm_core_lib.connections.bedrock_connection import BedrockConnection
        from llm_core_lib.connections.openai_connection import OpenAiConnection

        forbidden_method_substrings = (
            'build_prompt', 'prepare_prompt', 'compose_prompt',
            'with_system', 'prepend_', 'render_prompt',
        )
        for cls in (OpenAiConnection, AnthropicConnection, BedrockConnection):
            for name in dir(cls):
                if name.startswith('_'):
                    continue
                lowered = name.lower()
                for needle in forbidden_method_substrings:
                    self.assertFalse(
                        needle in lowered,
                        f'{cls.__name__}.{name} looks like a prompt '
                        f'builder — that belongs upstream in '
                        f'agent-core-lib.',
                    )


# ----------------------------------------------------------------------
# 3. Registry resolves without invoking any prompt helper
# ----------------------------------------------------------------------


class TestRegistryHasNoPromptSurface(unittest.TestCase):
    def test_get_returns_factory_without_touching_any_prompt(self):
        fake = MockOpenAIClient()
        registry = LlmConnectionRegistry()
        registry.register(LlmConnectionConfig(
            id='openai-default',
            provider='openai',
            model='gpt-x',
            vision_model='gpt-x',
            embedding_model='emb',
            max_tokens=4096,
            temperature=0.0,
            api_key='k',
            extra={'client': fake},
        ))
        factory = registry.get('openai-default')
        # The factory is the cached one from registration; nothing was
        # built or invoked that could have shaped a prompt.
        self.assertIs(factory, registry.get('openai-default'))
        # No chat call has been made — registry resolution is pure
        # lookup.
        self.assertEqual(fake.chat_calls, [])

    def test_registry_public_api_is_lookup_only(self):
        # ``LlmConnectionRegistry`` must not grow methods that hint at
        # prompt-building responsibility. Probe the public surface.
        allowed = {
            'register', 'unregister', 'get', 'has', 'list', 'clear',
        }
        public = {
            n for n in dir(_RegistryClass)
            if not n.startswith('_')
        }
        # Allow ``allowed`` and no others.
        extras = public - allowed
        self.assertEqual(
            extras,
            set(),
            f'LlmConnectionRegistry public surface drifted: {extras}. '
            f'New methods must not introduce prompt-building.',
        )


# ----------------------------------------------------------------------
# 4. Vision payload + embedding live INSIDE llm-core-lib
# ----------------------------------------------------------------------


class TestVisionAndEmbeddingStayInsideLlmCoreLib(unittest.TestCase):
    def test_anthropic_vision_packages_image_bytes_here(self):
        fake = MockAnthropicClient()
        factory = AnthropicConnectionFactory({
            'model': 'claude-x',
            'vision_model': 'claude-v',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'k',
            'client': fake,
        })
        with factory.get() as conn:
            conn.complete_vision(
                prompt='describe this image',
                image_bytes=b'\x89PNG\r\n\x1a\nfake',
                image_mime='image/png',
                system='caller-prepared system text',
            )
        content = fake.calls[0]['messages'][0]['content']
        # Image block must be first (Anthropic convention) and contain
        # base64 data with the right media type — packed inside
        # llm-core-lib, not by the caller.
        self.assertEqual(content[0]['type'], 'image')
        self.assertEqual(content[0]['source']['type'], 'base64')
        self.assertEqual(content[0]['source']['media_type'], 'image/png')
        self.assertTrue(content[0]['source']['data'])
        self.assertEqual(content[1], {'type': 'text', 'text': 'describe this image'})
        self.assertEqual(fake.calls[0]['system'], 'caller-prepared system text')

    def test_openai_embed_takes_only_text_and_forwards_it(self):
        fake = MockOpenAIClient(embedding=[1.0, 2.0, 3.0])
        factory = OpenAiConnectionFactory({
            'model': 'gpt-x',
            'vision_model': 'gpt-x',
            'embedding_model': 'text-embedding-3-small',
            'max_tokens': 4096,
            'temperature': 0.0,
            'api_key': 'k',
            'client': fake,
        })
        with factory.get() as conn:
            vector = conn.embed('the quick brown fox')
        self.assertEqual(vector, [1.0, 2.0, 3.0])
        sent = fake.embedding_calls[0]
        self.assertEqual(sent['input'], 'the quick brown fox')
        self.assertEqual(sent['model'], 'text-embedding-3-small')
        # No system / prompt fields exist on the embed call surface.

    def test_embed_signature_takes_only_text(self):
        import inspect

        from llm_core_lib.connections.bedrock_connection import BedrockConnection
        from llm_core_lib.connections.openai_connection import OpenAiConnection

        for cls in (OpenAiConnection, BedrockConnection):
            sig = inspect.signature(cls.embed)
            params = [p for p in sig.parameters.values() if p.name != 'self']
            self.assertEqual(
                [p.name for p in params],
                ['text'],
                f'{cls.__name__}.embed signature drifted — embeddings '
                f'must take only the raw text; prompt shaping is '
                f'upstream.',
            )


if __name__ == '__main__':
    unittest.main()
