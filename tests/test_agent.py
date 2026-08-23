from __future__ import annotations

from typing import Any

from qwen3vl_agent import ModelOutput, Qwen3VLAgent, ToolContext, ToolRegistry
from qwen3vl_agent.models.base import BaseVideoModel


class FakeModel(BaseVideoModel):
    def __init__(self, responses: list[str]):
        super().__init__("fake", device="cpu", dtype="float32")
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, messages, *, videos=None, images=None, **kwargs):
        self.calls.append(
            {"messages": messages, "videos": videos, "images": images, "kwargs": kwargs}
        )
        return ModelOutput(next(self.responses), {"fake": True})


def test_agent_plans_executes_and_injects_tool_evidence() -> None:
    model = FakeModel(
        [
            '{"tools":[{"name":"metadata","arguments":{"detail":true}}]}',
            "final answer",
        ]
    )
    registry = ToolRegistry()

    @registry.tool(
        name="metadata",
        description="Read video metadata.",
        parameters={"type": "object", "properties": {"detail": {"type": "boolean"}}},
    )
    def metadata(context: ToolContext, detail: bool):
        return {"path": context.require_video(), "detail": detail}

    agent = Qwen3VLAgent(model, tools=registry)
    agent.load()
    output = agent.generate(
        [{"role": "user", "content": "How long is it?"}],
        videos=["demo.mp4"],
    )

    assert output.text == "final answer"
    assert output.metadata["tool_calls"][0]["ok"] is True
    assert output.metadata["tool_calls"][0]["result"]["path"] == "demo.mp4"
    assert model.calls[0]["videos"] is None
    assert model.calls[1]["videos"] == ["demo.mp4"]
    final_prompt = model.calls[1]["messages"][0]["content"]
    assert "supplementary evidence" in final_prompt
    assert "demo.mp4" in final_prompt


def test_invalid_plan_falls_back_to_direct_generation() -> None:
    model = FakeModel(["not json", "direct answer"])
    registry = ToolRegistry()
    registry.tool(name="noop", description="No operation.")(lambda context: None)
    agent = Qwen3VLAgent(model, tools=registry)
    agent.load()

    output = agent.generate([{"role": "user", "content": "question"}])

    assert output.text == "direct answer"
    assert output.metadata["tool_calls"] == []
    assert "error" in output.metadata["tool_planner"]


def test_empty_external_registry_is_preserved() -> None:
    registry = ToolRegistry()
    agent = Qwen3VLAgent(FakeModel([]), tools=registry)
    assert agent.tools is registry
