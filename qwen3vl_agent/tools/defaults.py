from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen3vl_agent.tools.base import ToolContext
from qwen3vl_agent.tools.registry import ToolRegistry


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_video_metadata(context: ToolContext, video_index: int = 0) -> dict[str, Any]:
    """Read container and primary video-stream metadata with PyAV."""

    import av

    path = Path(context.require_video(video_index)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    with av.open(str(path)) as container:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            raise ValueError(f"No video stream found: {path}")

        frame_rate = _as_float(stream.average_rate or stream.base_rate or stream.guessed_rate)
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)

        frame_count = int(stream.frames) if stream.frames else None
        if frame_count is None and duration is not None and frame_rate is not None:
            frame_count = round(duration * frame_rate)

        return {
            "video_index": video_index,
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "duration_seconds": round(duration, 6) if duration is not None else None,
            "width": int(stream.width),
            "height": int(stream.height),
            "frame_rate_fps": round(frame_rate, 6) if frame_rate is not None else None,
            "frame_count": frame_count,
            "codec": stream.codec_context.name,
            "container_format": container.format.name,
        }


def build_default_registry() -> ToolRegistry:
    """Build the small, dependency-light registry used by the CLI baseline."""

    registry = ToolRegistry()
    registry.tool(
        name="video_metadata",
        description=(
            "Read exact file metadata for an input video: duration, resolution, frame rate, "
            "frame count, codec, container format, and file size. Use this for questions about "
            "those properties instead of estimating them visually."
        ),
        parameters={
            "type": "object",
            "properties": {
                "video_index": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Zero-based index into the supplied videos.",
                }
            },
            "additionalProperties": False,
        },
    )(read_video_metadata)
    return registry
