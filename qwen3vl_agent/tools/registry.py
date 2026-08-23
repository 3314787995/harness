from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from qwen3vl_agent.tools.base import BaseTool
from qwen3vl_agent.tools.function import FunctionTool


class ToolRegistry:
    """Ordered collection of tools exposed to the planner."""

    def __init__(self) -> None:
        self._tools: OrderedDict[str, BaseTool] = OrderedDict()

    def register(self, tool: BaseTool, *, replace: bool = False) -> BaseTool:
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        if tool.name in self._tools:
            self._tools[tool.name].unload()
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str, *, unload: bool = True) -> BaseTool:
        tool = self._tools.pop(name)
        if unload:
            tool.unload()
        return tool

    def tool(
        self,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for registering a function while returning it unchanged."""

        schema = parameters or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

        def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                FunctionTool(
                    name=name,
                    description=description,
                    parameters=schema,
                    handler=handler,
                ),
                replace=replace,
            )
            return handler

        return decorator

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def manifests(self) -> list[dict[str, Any]]:
        return [tool.manifest() for tool in self._tools.values()]

    def load_all(self) -> None:
        loaded: list[BaseTool] = []
        try:
            for tool in self._tools.values():
                tool.load()
                loaded.append(tool)
        except Exception:
            for tool in reversed(loaded):
                tool.unload()
            raise

    def unload_all(self) -> None:
        for tool in reversed(list(self._tools.values())):
            tool.unload()

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._tools.values())
