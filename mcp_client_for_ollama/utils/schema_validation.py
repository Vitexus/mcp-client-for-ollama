"""Utilities for validating LLM-generated tool arguments against MCP input schemas."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable


class SchemaValidationError(ValueError):
    """Raised when tool arguments do not satisfy the declared JSON schema."""


def normalize_tool_arguments(raw_arguments: Any) -> Dict[str, Any]:
    """Normalize tool arguments emitted by models into a dictionary.

    Ollama models usually emit a dict, but some may emit a JSON string.
    """
    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, str):
        text = raw_arguments.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(decoded, dict):
            raise SchemaValidationError("arguments JSON must decode to an object")
        return decoded

    raise SchemaValidationError(f"arguments must be an object or JSON string, got {type(raw_arguments).__name__}")


def validate_tool_arguments(arguments: Dict[str, Any], schema: Dict[str, Any] | None) -> None:
    """Validate arguments against a minimal JSON Schema subset used by MCP tools."""
    if not schema:
        return
    _validate_schema_node(arguments, schema, path="$")


def _validate_schema_node(value: Any, schema: Dict[str, Any], path: str) -> None:
    if "enum" in schema:
        allowed = schema["enum"]
        if value not in allowed:
            raise SchemaValidationError(f"{path} must be one of {allowed}, got {value!r}")

    declared_type = schema.get("type")
    if declared_type is not None:
        if isinstance(declared_type, str):
            declared_types: Iterable[str] = (declared_type,)
        else:
            declared_types = declared_type
        if not any(_matches_type(value, item) for item in declared_types):
            pretty = ", ".join(str(t) for t in declared_types)
            raise SchemaValidationError(f"{path} must be of type {pretty}, got {type(value).__name__}")

    if schema.get("type") == "object":
        _validate_object(value, schema, path)
    elif schema.get("type") == "array":
        _validate_array(value, schema, path)


def _validate_object(value: Any, schema: Dict[str, Any], path: str) -> None:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path} must be an object")

    properties = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    for field in required:
        if field not in value:
            raise SchemaValidationError(f"{path}.{field} is required")

    additional = schema.get("additionalProperties", True)
    if additional is False:
        extras = [key for key in value.keys() if key not in properties]
        if extras:
            raise SchemaValidationError(f"{path} contains unsupported field(s): {', '.join(sorted(extras))}")

    for key, item in value.items():
        child_path = f"{path}.{key}"
        if key in properties:
            _validate_schema_node(item, properties[key], child_path)
        elif isinstance(additional, dict):
            _validate_schema_node(item, additional, child_path)


def _validate_array(value: Any, schema: Dict[str, Any], path: str) -> None:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{path} must be an array")
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            _validate_schema_node(item, item_schema, f"{path}[{index}]")


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True
