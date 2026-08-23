from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

from qwen3vl_agent.active_tree.config import ActiveTreeConfig
from qwen3vl_agent.active_tree.types import SceneNode, SceneTree
from qwen3vl_agent.coarse_to_fine.cache import CachedVideo, SubtitleTrack
from qwen3vl_agent.coarse_to_fine.types import FrameRef


class SceneTreeBuilder:
    """Build and persist a query-independent scene-aware temporal hierarchy."""

    INDEX_VERSION = 2

    def __init__(self, config: ActiveTreeConfig) -> None:
        self.config = config

    def build(
        self,
        cached: CachedVideo,
        subtitles: SubtitleTrack | None = None,
    ) -> SceneTree:
        signature = self._signature(subtitles)
        index_path = Path(cached.cache_dir) / f"active_tree_{signature}.json"
        if index_path.is_file():
            try:
                return self._load(index_path, signature=signature, cache_hit=True)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        scores = self._visual_change_scores(cached.frames)
        boundaries = self._boundaries(cached, subtitles, scores)
        tree = self._hierarchy(cached.duration_seconds, boundaries, scores, cached.frames)
        tree.index_path = str(index_path.resolve())
        self._save(index_path, tree, signature=signature)
        return tree

    def storyboard_frames(self, cached: CachedVideo, node: SceneNode) -> list[FrameRef]:
        targets = [
            node.start_seconds + min(0.25, node.duration_seconds * 0.05),
            node.midpoint_seconds,
            max(node.start_seconds, node.end_seconds - min(0.25, node.duration_seconds * 0.05)),
        ]
        if node.peak_timestamp_seconds is not None:
            targets.append(node.peak_timestamp_seconds)
        else:
            targets.append(node.midpoint_seconds)
        frames = self._deduplicate(cached.nearest_frame(target) for target in targets)
        if len(frames) < self.config.storyboard_frames:
            frames = self._deduplicate(
                [*frames, *cached.uniform_frames(self.config.storyboard_frames, node.as_window())]
            )
        return frames[: self.config.storyboard_frames]

    def _signature(self, subtitles: SubtitleTrack | None) -> str:
        subtitle_key: str | None = None
        if subtitles is not None:
            subtitle_payload = [
                (round(cue.start_seconds, 3), round(cue.end_seconds, 3))
                for cue in subtitles.cues
            ]
            subtitle_key = hashlib.sha1(
                json.dumps(subtitle_payload).encode()
            ).hexdigest()[:12]
        payload = {
            "version": self.INDEX_VERSION,
            "scene_min_seconds": self.config.scene_min_seconds,
            "scene_max_seconds": self.config.scene_max_seconds,
            "scene_change_threshold": self.config.scene_change_threshold,
            "subtitle_gap_seconds": self.config.subtitle_gap_seconds,
            "min_children": self.config.min_children,
            "max_children": self.config.max_children,
            "subtitle_timing": subtitle_key,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    @staticmethod
    def _deduplicate(frames: Any) -> list[FrameRef]:
        result: list[FrameRef] = []
        seen: set[str] = set()
        for frame in frames:
            if frame.id in seen:
                continue
            seen.add(frame.id)
            result.append(frame)
        return result

    @staticmethod
    def _visual_change_scores(frames: tuple[FrameRef, ...]) -> dict[str, float]:
        if not frames:
            return {}
        from PIL import Image, ImageChops, ImageStat

        scores: dict[str, float] = {frames[0].id: 0.0}
        previous = None
        for frame in frames:
            with Image.open(frame.path) as image:
                current = image.convert("L").resize((64, 36))
            if previous is not None:
                difference = ImageChops.difference(previous, current)
                scores[frame.id] = float(ImageStat.Stat(difference).mean[0]) / 255.0
            previous = current
        return scores

    def _boundaries(
        self,
        cached: CachedVideo,
        subtitles: SubtitleTrack | None,
        scores: dict[str, float],
    ) -> list[float]:
        candidates: list[tuple[float, float]] = []
        for frame in cached.frames[1:]:
            score = scores.get(frame.id, 0.0)
            if score >= self.config.scene_change_threshold:
                candidates.append((frame.timestamp_seconds, score))

        if subtitles is not None and self.config.subtitle_gap_seconds > 0:
            for left, right in zip(subtitles.cues, subtitles.cues[1:]):
                gap = right.start_seconds - left.end_seconds
                if gap >= self.config.subtitle_gap_seconds:
                    timestamp = (left.end_seconds + right.start_seconds) / 2
                    candidates.append((timestamp, min(1.0, 0.5 + gap / 20)))

        candidates.sort(key=lambda item: item[0])
        filtered: list[tuple[float, float]] = []
        for timestamp, score in candidates:
            if timestamp <= 0 or timestamp >= cached.duration_seconds:
                continue
            if not filtered or timestamp - filtered[-1][0] >= self.config.scene_min_seconds:
                filtered.append((timestamp, score))
                continue
            if score > filtered[-1][1]:
                filtered[-1] = (timestamp, score)

        boundaries = [0.0]
        for timestamp, _ in filtered:
            while timestamp - boundaries[-1] > self.config.scene_max_seconds:
                boundaries.append(boundaries[-1] + self.config.scene_max_seconds)
            if timestamp - boundaries[-1] >= self.config.scene_min_seconds:
                boundaries.append(timestamp)
        while cached.duration_seconds - boundaries[-1] > self.config.scene_max_seconds:
            boundaries.append(boundaries[-1] + self.config.scene_max_seconds)
        if cached.duration_seconds - boundaries[-1] < self.config.scene_min_seconds and len(boundaries) > 1:
            boundaries.pop()
        boundaries.append(cached.duration_seconds)
        return boundaries

    def _hierarchy(
        self,
        duration_seconds: float,
        boundaries: list[float],
        scores: dict[str, float],
        frames: tuple[FrameRef, ...],
    ) -> SceneTree:
        nodes: dict[str, SceneNode] = {}
        score_points = self._score_points(scores, frames)
        current: list[SceneNode] = []
        for index, (start, end) in enumerate(pairwise(boundaries)):
            peak_timestamp, peak_score = self._peak(score_points, start, end)
            node = SceneNode(
                id=f"L0-N{index:04d}",
                start_seconds=start,
                end_seconds=end,
                level=0,
                peak_timestamp_seconds=peak_timestamp,
                boundary_score=peak_score,
            )
            nodes[node.id] = node
            current.append(node)

        level = 1
        while len(current) > self.config.max_children:
            grouped: list[SceneNode] = []
            for index, group in enumerate(self._balanced_groups(current)):
                peak_child = max(group, key=lambda item: item.boundary_score)
                parent = SceneNode(
                    id=f"L{level}-N{index:04d}",
                    start_seconds=group[0].start_seconds,
                    end_seconds=group[-1].end_seconds,
                    level=level,
                    child_ids=[item.id for item in group],
                    peak_timestamp_seconds=peak_child.peak_timestamp_seconds,
                    boundary_score=peak_child.boundary_score,
                )
                for child in group:
                    child.parent_id = parent.id
                nodes[parent.id] = parent
                grouped.append(parent)
            current = grouped
            level += 1

        root_peak = max(current, key=lambda item: item.boundary_score) if current else None
        root = SceneNode(
            id="ROOT",
            start_seconds=0.0,
            end_seconds=duration_seconds,
            level=level,
            child_ids=[item.id for item in current],
            peak_timestamp_seconds=(root_peak.peak_timestamp_seconds if root_peak else None),
            boundary_score=(root_peak.boundary_score if root_peak else 0.0),
        )
        for child in current:
            child.parent_id = root.id
        nodes[root.id] = root
        self._assign_depths(nodes, root.id, depth=0)
        return SceneTree(root_id=root.id, nodes=nodes)

    def _balanced_groups(self, nodes: list[SceneNode]) -> list[list[SceneNode]]:
        count = len(nodes)
        group_count = max(
            self.config.min_children,
            math.ceil(count / self.config.max_children),
        )
        group_count = min(group_count, max(1, count // 2))
        base, remainder = divmod(count, group_count)
        sizes = [base + (1 if index < remainder else 0) for index in range(group_count)]
        groups: list[list[SceneNode]] = []
        offset = 0
        for size in sizes:
            groups.append(nodes[offset : offset + size])
            offset += size
        return groups

    @staticmethod
    def _score_points(
        scores: dict[str, float],
        frames: tuple[FrameRef, ...],
    ) -> list[tuple[float, float]]:
        return [
            (frame.timestamp_seconds, scores.get(frame.id, 0.0))
            for frame in frames
        ]

    @staticmethod
    def _peak(
        score_points: list[tuple[float, float]],
        start: float,
        end: float,
    ) -> tuple[float, float]:
        matches = [point for point in score_points if start <= point[0] <= end]
        if not matches:
            return (start + end) / 2, 0.0
        return max(matches, key=lambda item: item[1])

    @staticmethod
    def _assign_depths(nodes: dict[str, SceneNode], node_id: str, *, depth: int) -> None:
        node = nodes[node_id]
        node.level = depth
        for child_id in node.child_ids:
            SceneTreeBuilder._assign_depths(nodes, child_id, depth=depth + 1)

    def _save(self, path: Path, tree: SceneTree, *, signature: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.INDEX_VERSION,
            "signature": signature,
            "root_id": tree.root_id,
            "nodes": [node.to_dict() for node in tree.nodes.values()],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self, path: Path, *, signature: str, cache_hit: bool) -> SceneTree:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != self.INDEX_VERSION:
            raise ValueError("unsupported active-tree index version")
        if payload.get("signature") != signature:
            raise ValueError("active-tree index config mismatch")
        nodes = {
            str(item["id"]): SceneNode(
                id=str(item["id"]),
                start_seconds=float(item["start_seconds"]),
                end_seconds=float(item["end_seconds"]),
                level=int(item["level"]),
                parent_id=item.get("parent_id"),
                child_ids=[str(value) for value in item.get("child_ids", [])],
                peak_timestamp_seconds=(
                    float(item["peak_timestamp_seconds"])
                    if item.get("peak_timestamp_seconds") is not None
                    else None
                ),
                boundary_score=float(item.get("boundary_score", 0.0)),
            )
            for item in payload["nodes"]
        }
        tree = SceneTree(
            root_id=str(payload["root_id"]),
            nodes=nodes,
            index_path=str(path.resolve()),
            cache_hit=cache_hit,
        )
        return replace(tree, cache_hit=cache_hit)
