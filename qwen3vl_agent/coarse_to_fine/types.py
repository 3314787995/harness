from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class CoarseToFineError(RuntimeError):
    """Base error for recoverable coarse-to-fine failures."""


class DecisionError(CoarseToFineError):
    """Raised when a model decision cannot be parsed or validated."""


class BudgetExhausted(CoarseToFineError):
    """Raised when a model call cannot fit inside the configured frame budget."""


@dataclass(frozen=True)
class TimeWindow:
    id: str
    start_seconds: float
    end_seconds: float
    depth: int = 1
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("window start must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("window end must be greater than start")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def midpoint_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2

    def contains(self, timestamp: float) -> bool:
        return self.start_seconds <= timestamp <= self.end_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "depth": self.depth,
            "parent_id": self.parent_id,
        }


def partition_window(
    window: TimeWindow,
    *,
    parts: int,
    round_index: int,
    id_offset: int = 0,
) -> list[TimeWindow]:
    if parts < 1:
        raise ValueError("parts must be positive")
    step = window.duration_seconds / parts
    children: list[TimeWindow] = []
    for index in range(parts):
        start = window.start_seconds + index * step
        end = window.end_seconds if index == parts - 1 else start + step
        children.append(
            TimeWindow(
                id=f"R{round_index}-W{id_offset + index}",
                start_seconds=start,
                end_seconds=end,
                depth=round_index,
                parent_id=window.id,
            )
        )
    return children


@dataclass(frozen=True)
class FrameRef:
    id: str
    timestamp_seconds: float
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "path": self.path,
        }


@dataclass
class FrameBudget:
    unique_limit: int
    cumulative_limit: int
    unique_ids: set[str] = field(default_factory=set)
    cumulative_views: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def consume(
        self,
        frames: Iterable[FrameRef],
        *,
        purpose: str,
        min_count: int = 1,
    ) -> list[FrameRef]:
        remaining_views = self.cumulative_limit - self.cumulative_views
        accepted: list[FrameRef] = []
        pending_new: set[str] = set()
        for frame in frames:
            if len(accepted) >= remaining_views:
                break
            is_new = frame.id not in self.unique_ids and frame.id not in pending_new
            if is_new and len(self.unique_ids) + len(pending_new) >= self.unique_limit:
                continue
            accepted.append(frame)
            if is_new:
                pending_new.add(frame.id)

        if len(accepted) < min_count:
            raise BudgetExhausted(
                f"Frame budget cannot satisfy {purpose!r}: requested at least {min_count}, "
                f"accepted {len(accepted)}"
            )

        self.unique_ids.update(pending_new)
        self.cumulative_views += len(accepted)
        self.events.append(
            {
                "purpose": purpose,
                "frame_views": len(accepted),
                "new_unique_frames": len(pending_new),
                "unique_frames_after": len(self.unique_ids),
                "cumulative_views_after": self.cumulative_views,
            }
        )
        return accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "unique_limit": self.unique_limit,
            "cumulative_limit": self.cumulative_limit,
            "unique_frames": len(self.unique_ids),
            "cumulative_views": self.cumulative_views,
            "events": list(self.events),
        }


class EvidenceMemory:
    """Bounded ordered store of frames that have been shown to the model."""

    def __init__(self, max_frames: int) -> None:
        self.max_frames = max_frames
        self._frames: OrderedDict[str, FrameRef] = OrderedDict()

    def add(self, frames: Iterable[FrameRef]) -> None:
        for frame in frames:
            self._frames.pop(frame.id, None)
            self._frames[frame.id] = frame
            while len(self._frames) > self.max_frames:
                self._frames.popitem(last=False)

    def working_set(
        self,
        preferred: Iterable[FrameRef],
        *,
        limit: int,
    ) -> list[FrameRef]:
        selected: list[FrameRef] = []
        seen: set[str] = set()
        for frame in preferred:
            if frame.id in seen:
                continue
            selected.append(frame)
            seen.add(frame.id)
            if len(selected) >= limit:
                return selected
        for frame in reversed(self._frames.values()):
            if frame.id in seen:
                continue
            selected.append(frame)
            seen.add(frame.id)
            if len(selected) >= limit:
                break
        return selected

    def to_list(self) -> list[dict[str, Any]]:
        return [frame.to_dict() for frame in self._frames.values()]
