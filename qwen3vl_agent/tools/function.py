from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from qwen3vl_agent.tools.base import BaseTool, ToolContext, ToolResult


class FunctionTool(BaseTool):
    """Adapt a Python callable to the tool protocol."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        if not name or not name.replace("_", "").isalnum():
            raise ValueError("Tool name must contain only letters, numbers, and underscores")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self.name = name
        self.description = description.strip()
        self.parameters = dict(parameters)
        self.handler = handler

    def invoke(self, context: ToolContext, arguments: Mapping[str, Any]) -> ToolResult:
        value = self.handler(context, **dict(arguments))
        if isinstance(value, ToolResult):
            if value.tool_name != self.name:
                raise ValueError(
                    f"Tool returned name {value.tool_name!r}; expected {self.name!r}"
                )
            return value
        return ToolResult(tool_name=self.name, data=value)
