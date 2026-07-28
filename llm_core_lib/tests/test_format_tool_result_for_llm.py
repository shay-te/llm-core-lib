"""``format_tool_result_for_llm`` — JSON-not-repr + ``<TOOL_DATA>`` wrapper.

Two safety contracts, both load-bearing:

  1. **JSON serialization** instead of ``str(result)`` so dates,
     Decimals, UUIDs, and any custom ``__str__`` / ``__repr__`` can't
     leak around the PII scrub.
  2. **``<TOOL_DATA>...</TOOL_DATA>`` wrapper** so the system prompt
     can instruct the model to treat the inner content as read-only
     data — the structural defense against indirect prompt injection
     via free-text fields that carry user-controlled natural language.

Both ``OpenAiConnection.chat_with_tools`` and
``BedrockConnection.chat_with_tools`` route every tool result
through this function so the wire format is identical regardless
of provider.

Per the workspace-wide "one TestCase per file" rule (see this repo's
``AGENTS.md``), this file owns exactly one TestCase.
"""
from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal
from uuid import UUID

from llm_core_lib.connections.openai_connection import format_tool_result_for_llm


class TestFormatToolResultForLlm(unittest.TestCase):

    def test_wraps_dict_in_tool_data_markers(self):
        result = format_tool_result_for_llm({'id': 1, 'name': 'Jane'})
        self.assertTrue(result.startswith('<TOOL_DATA>\n'))
        self.assertTrue(result.endswith('\n</TOOL_DATA>'))
        # And the JSON body is in between.
        self.assertIn('"id": 1', result)
        self.assertIn('"name": "Jane"', result)

    def test_serializes_with_json_not_python_repr(self):
        # Plain ``str(dict)`` produces ``"{'id': 1, 'name': 'Jane'}"`` —
        # Python repr with single quotes. JSON uses double quotes and
        # is the de-facto interchange format the model expects.
        result = format_tool_result_for_llm({'id': 1})
        self.assertIn('"id"', result)         # JSON double quote
        self.assertNotIn("'id'", result)       # not Python repr

    def test_serializes_date_via_default_str(self):
        # ``json.dumps`` doesn't natively handle ``date``; ``default=str``
        # turns it into the ISO string. Tests the safety net.
        result = format_tool_result_for_llm({'created_at': date(2026, 6, 5)})
        self.assertIn('"2026-06-05"', result)

    def test_serializes_decimal_via_default_str(self):
        result = format_tool_result_for_llm({'amount': Decimal('100.50')})
        self.assertIn('"100.50"', result)

    def test_serializes_uuid_via_default_str(self):
        u = UUID('12345678-1234-5678-1234-567812345678')
        result = format_tool_result_for_llm({'request_id': u})
        self.assertIn('"12345678-1234-5678-1234-567812345678"', result)

    def test_none_is_serialized_as_null(self):
        result = format_tool_result_for_llm(None)
        # ``<TOOL_DATA>\nnull\n</TOOL_DATA>``
        self.assertIn('null', result)

    def test_list_of_dicts_serializes_as_json_array(self):
        result = format_tool_result_for_llm([{'id': 1}, {'id': 2}])
        self.assertIn('[{"id": 1}, {"id": 2}]', result)

    def test_wrapper_is_always_present_even_for_empty_dict(self):
        result = format_tool_result_for_llm({})
        self.assertEqual(result, '<TOOL_DATA>\n{}\n</TOOL_DATA>')


if __name__ == '__main__':
    unittest.main()
