"""Register a small ffprobe-backed tool without changing package source."""

from __future__ import annotations

import json
import os
import subprocess

from qwen3vl_agent import Qwen3VLAgent, Qwen3VLModel, ToolContext, ToolRegistry


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool(
        name="video_metadata",
        description="Read exact duration, dimensions and frame rate from the input video.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    def video_metadata(context: ToolContext) -> dict:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate",
            "-of",
            "json",
            context.require_video(),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)

    return registry


def main(video_path: str, question: str) -> None:
    model = Qwen3VLModel(
        os.environ.get("QWEN3VL_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
    )
    agent = Qwen3VLAgent(model, tools=build_registry())
    agent.load()
    try:
        result = agent.generate(
            [{"role": "user", "content": question}],
            videos=[video_path],
        )
        print(result.text)
    finally:
        agent.unload()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("question")
    values = parser.parse_args()
    main(values.video, values.question)
