from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    """Inputs shared with a tool at invocation time."""

    question: str
    messages: Sequence[Mapping[str, Any]]
    videos: Sequence[str] = ()
    images: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def require_video(self, index: int = 0) -> str:
        try:
            return self.videos[index]
        except IndexError as exc:
            raise ValueError(f"Tool requires video index {index}, but no such video was provided") from exc

    def require_image(self, index: int = 0) -> str:
        try:
            return self.images[index]
        except IndexError as exc:
            raise ValueError(f"Tool requires image index {index}, but no such image was provided") from exc


@dataclass
class ToolResult:
    tool_name: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Minimal tool contract used by :class:`Qwen3VLAgent`."""

    name: str
    description: str
    parameters: Mapping[str, Any]

    def load(self) -> None:
        """Allocate optional resources. Stateless tools can keep the default."""

    def unload(self) -> None:
        """Release optional resources. Stateless tools can keep the default."""

    @abstractmethod
    def invoke(self, context: ToolContext, arguments: Mapping[str, Any]) -> ToolResult:
        """Execute the tool with model-planned arguments."""

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }
