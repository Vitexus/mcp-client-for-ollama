"""Tests for schema validation of MCP tool arguments."""

import pytest

from mcp_client_for_ollama.utils.schema_validation import (
    SchemaValidationError,
    normalize_tool_arguments,
    validate_tool_arguments,
)


def test_normalize_tool_arguments_accepts_dict():
    assert normalize_tool_arguments({"q": "hello"}) == {"q": "hello"}


def test_normalize_tool_arguments_parses_json_string():
    assert normalize_tool_arguments('{"q":"hello"}') == {"q": "hello"}


def test_normalize_tool_arguments_rejects_invalid_json_string():
    with pytest.raises(SchemaValidationError, match="not valid JSON"):
        normalize_tool_arguments('{"q":')


def test_validate_tool_arguments_rejects_missing_required():
    schema = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    }

    with pytest.raises(SchemaValidationError, match="required"):
        validate_tool_arguments({"status": "hi"}, schema)


def test_validate_tool_arguments_rejects_type_mismatch():
    schema = {
        "type": "object",
        "properties": {"limit": {"type": "integer"}},
        "required": ["limit"],
    }

    with pytest.raises(SchemaValidationError, match="integer"):
        validate_tool_arguments({"limit": "10"}, schema)


def test_validate_tool_arguments_rejects_unknown_fields_when_disallowed():
    schema = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
        "additionalProperties": False,
    }

    with pytest.raises(SchemaValidationError, match="unsupported field"):
        validate_tool_arguments({"content": "ok", "status": "extra"}, schema)


def test_validate_tool_arguments_accepts_valid_payload():
    schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "visibility": {"type": "string", "enum": ["public", "private"]},
        },
        "required": ["content"],
    }

    validate_tool_arguments({"content": "hello", "visibility": "public"}, schema)
