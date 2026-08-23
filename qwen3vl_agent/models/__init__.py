from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput

__all__ = ["BaseVideoModel", "ModelOutput", "Qwen3VLModel"]


def __getattr__(name: str):
    if name == "Qwen3VLModel":
        from qwen3vl_agent.models.qwen3vl import Qwen3VLModel

        return Qwen3VLModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
