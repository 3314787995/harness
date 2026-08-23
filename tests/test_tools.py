from __future__ import annotations

import pytest

from qwen3vl_agent.tools import FunctionTool, ToolContext, ToolRegistry, build_default_registry

EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def test_function_tool_and_registry_decorator() -> None:
    registry = ToolRegistry()

    @registry.tool(
        name="echo",
        description="Echo a value.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )
    def echo(context: ToolContext, value: str):
        return {"question": context.question, "value": value}

    result = registry.get("echo").invoke(ToolContext(question="q", messages=[]), {"value": "x"})
    assert result.tool_name == "echo"
    assert result.data == {"question": "q", "value": "x"}
    assert registry.manifests()[0]["name"] == "echo"


def test_duplicate_tool_is_rejected() -> None:
    registry = ToolRegistry()
    tool = FunctionTool(
        name="noop",
        description="No operation.",
        parameters=EMPTY_SCHEMA,
        handler=lambda context: None,
    )
    registry.register(tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_context_requires_media() -> None:
    context = ToolContext(question="q", messages=[])
    with pytest.raises(ValueError, match="requires video"):
        context.require_video()


def test_default_registry_exposes_video_metadata() -> None:
    registry = build_default_registry()
    manifest = registry.manifests()[0]
    assert manifest["name"] == "video_metadata"
    assert manifest["parameters"]["properties"]["video_index"]["default"] == 0
