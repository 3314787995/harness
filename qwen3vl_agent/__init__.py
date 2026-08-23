from qwen3vl_agent.active_tree import ActiveTreeConfig, ActiveTreeVideoAgent
from qwen3vl_agent.agent import Qwen3VLAgent
from qwen3vl_agent.coarse_to_fine import CoarseToFineConfig, CoarseToFineVideoAgent
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.tools import BaseTool, FunctionTool, ToolContext, ToolRegistry, ToolResult

__all__ = [
    "ActiveTreeConfig",
    "ActiveTreeVideoAgent",
    "BaseTool",
    "BaseVideoModel",
    "CoarseToFineConfig",
    "CoarseToFineVideoAgent",
    "FunctionTool",
    "ModelOutput",
    "Qwen3VLAgent",
    "Qwen3VLModel",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
]


def __getattr__(name: str):
    if name == "Qwen3VLModel":
        from qwen3vl_agent.models.qwen3vl import Qwen3VLModel

        return Qwen3VLModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
