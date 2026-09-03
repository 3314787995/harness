from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from qwen3vl_agent.coarse_to_fine.cache import CachedVideo, VideoEvidenceCache
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.p01.types import BoundingBox, TimeSpan


@dataclass(frozen=True)
class FrameMetrics:
    frame_id: str
    timestamp_seconds: float
    change_score: float
    motion_score: float
    clarity_score: float
    text_score: float
    is_black: bool
    is_duplicate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "change_score": round(self.change_score, 6),
            "motion_score": round(self.motion_score, 6),
            "clarity_score": round(self.clarity_score, 6),
            "text_score": round(self.text_score, 6),
            "is_black": self.is_black,
            "is_duplicate": self.is_duplicate,
        }


@dataclass(frozen=True)
class VideoMetadata:
    source_path: str
    duration_seconds: float
    source_fps: float | None
    width: int
    height: int


@dataclass(frozen=True)
class ShotRef:
    shot_id: str
    span: TimeSpan
    frame_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "span": self.span.to_dict(),
            "frame_ids": list(self.frame_ids),
        }


@dataclass(frozen=True)
class TemporalNode:
    node_id: str
    span: TimeSpan
    level: int
    child_ids: tuple[str, ...]
    shot_ids: tuple[str, ...]

    @property
    def is_leaf(self) -> bool:
        return not self.child_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "span": self.span.to_dict(),
            "level": self.level,
            "child_ids": list(self.child_ids),
            "shot_ids": list(self.shot_ids),
            "is_leaf": self.is_leaf,
        }


@dataclass(frozen=True)
class P01VideoIndex:
    cached_video: CachedVideo
    metrics: tuple[FrameMetrics, ...]
    shots: tuple[ShotRef, ...]
    nodes: dict[str, TemporalNode]
    root_id: str

    @property
    def duration_seconds(self) -> float:
        return self.cached_video.duration_seconds

    def node(self, node_id: str) -> TemporalNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise ValueError(f"unknown temporal node: {node_id}") from exc

    def children(self, node_id: str) -> tuple[TemporalNode, ...]:
        return tuple(self.node(child_id) for child_id in self.node(node_id).child_ids)

    def frames_in_span(self, span: TimeSpan) -> tuple[FrameRef, ...]:
        return tuple(
            frame
            for frame in self.cached_video.frames
            if span.decode_start_seconds - 1e-6
            <= frame.timestamp_seconds
            <= span.decode_end_seconds + 1e-6
        )

    def shot_ids_for_span(self, span: TimeSpan) -> tuple[str, ...]:
        return tuple(
            shot.shot_id
            for shot in self.shots
            if shot.span.end_seconds >= span.start_seconds
            and shot.span.start_seconds <= span.end_seconds
        )

    def representative_frames(
        self,
        node: TemporalNode,
        mode: str,
    ) -> tuple[FrameRef, FrameRef]:
        frames = self.frames_in_span(node.span)
        if not frames:
            nearest = self.cached_video.nearest_frame(node.span.midpoint_seconds)
            return nearest, nearest
        metrics = {item.frame_id: item for item in self.metrics}
        center = min(
            frames,
            key=lambda frame: abs(frame.timestamp_seconds - node.span.midpoint_seconds),
        )

        def best(score_name: str) -> FrameRef:
            return max(
                frames,
                key=lambda frame: getattr(metrics[frame.id], score_name),
            )

        if mode == "static_visual":
            return center, best("clarity_score")
        if mode == "dynamic_action":
            return center, best("motion_score")
        if mode == "ocr":
            return best("clarity_score"), best("text_score")
        if mode == "subscene_caption":
            event = max(
                frames,
                key=lambda frame: max(
                    metrics[frame.id].change_score,
                    metrics[frame.id].motion_score,
                ),
            )
            return center, event
        raise ValueError(f"unsupported observation mode: {mode}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.cached_video.to_dict(),
            "root_id": self.root_id,
            "shots": [shot.to_dict() for shot in self.shots],
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "frame_quality": [item.to_dict() for item in self.metrics],
        }


class P01IndexBuilder:
    """Build a low-resolution, non-destructive navigation index and bounded time tree."""

    def __init__(
        self,
        config: P01Config,
        *,
        cache: VideoEvidenceCache | None = None,
    ) -> None:
        self.config = config
        self.cache = cache or VideoEvidenceCache(
            config.resolved_cache_dir / "navigation",
            sample_fps=config.index_fps,
            max_side=config.index_max_side,
            jpeg_quality=config.cache_jpeg_quality,
            lru_size=config.cache_lru_size,
        )

    def prepare(self, video_path: str | Path) -> P01VideoIndex:
        cached = self.cache.prepare(video_path)
        metrics = self._measure_frames(cached.frames)
        shots = self._build_shots(cached, metrics)
        nodes, root_id = self._build_tree(shots)
        return P01VideoIndex(cached, metrics, shots, nodes, root_id)

    def probe(self, video_path: str | Path) -> VideoMetadata:
        """Read container metadata without decoding the full navigation stream."""
        import av

        source = Path(video_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Video does not exist: {source}")
        with av.open(str(source)) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                raise ValueError(f"No video stream found: {source}")
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            if duration is None or duration <= 0:
                raise ValueError(f"Video duration is unavailable: {source}")
            return VideoMetadata(
                source_path=str(source),
                duration_seconds=duration,
                source_fps=float(stream.average_rate) if stream.average_rate else None,
                width=int(stream.width),
                height=int(stream.height),
            )

    def prepare_interval(
        self,
        video_path: str | Path,
        span: TimeSpan,
        *,
        metadata: VideoMetadata | None = None,
    ) -> P01VideoIndex:
        """Build navigation metrics only inside a deterministic explicit interval."""
        source = Path(video_path).expanduser().resolve()
        details = metadata or self.probe(source)
        start = max(0.0, span.decode_start_seconds)
        end = min(details.duration_seconds, span.decode_end_seconds)
        if end <= start:
            raise ValueError("explicit interval has no decodable duration")
        step = 1.0 / self.config.index_fps
        sample_count = max(1, math.floor((end - start) / step) + 1)
        timestamps = [min(end, start + index * step) for index in range(sample_count)]
        if not timestamps or timestamps[-1] < end - 1e-6:
            timestamps.append(end)
        frames = SourceFrameStore(self.config).extract(
            source,
            timestamps,
            purpose="explicit_interval_navigation",
            max_side=self.config.index_max_side,
        )
        if not frames:
            raise ValueError(f"Video decoder returned no interval frames: {source}")
        cached = CachedVideo(
            source_path=str(source),
            cache_dir=str(Path(frames[0].path).parent),
            duration_seconds=details.duration_seconds,
            source_fps=details.source_fps,
            width=details.width,
            height=details.height,
            sample_fps=self.config.index_fps,
            frames=frames,
            cache_hit=False,
        )
        metrics = self._measure_frames(frames)
        indexed_span = TimeSpan(
            span.start_seconds,
            span.end_seconds,
            source="explicit_interval_index",
            context_start_seconds=start,
            context_end_seconds=end,
        )
        shots = (ShotRef("S0000", indexed_span, tuple(frame.id for frame in frames)),)
        nodes, root_id = self._build_tree(shots)
        return P01VideoIndex(cached, metrics, shots, nodes, root_id)

    def _measure_frames(self, frames: Sequence[FrameRef]) -> tuple[FrameMetrics, ...]:
        from PIL import Image, ImageChops, ImageFilter, ImageStat

        results: list[FrameMetrics] = []
        previous = None
        for frame in frames:
            with Image.open(frame.path) as source:
                gray = source.convert("L")
                gray.thumbnail((96, 54), Image.Resampling.BILINEAR)
                mean = ImageStat.Stat(gray).mean[0] / 255.0
                edges = gray.filter(ImageFilter.FIND_EDGES)
                edge_stat = ImageStat.Stat(edges)
                clarity = min(1.0, math.sqrt(edge_stat.var[0]) / 64.0)
                text_score = min(1.0, edge_stat.mean[0] / 32.0 + clarity * 0.35)
                if previous is None:
                    change = 0.0
                else:
                    difference = ImageChops.difference(previous, gray)
                    change = ImageStat.Stat(difference).mean[0] / 255.0
                previous = gray.copy()
            results.append(
                FrameMetrics(
                    frame_id=frame.id,
                    timestamp_seconds=frame.timestamp_seconds,
                    change_score=change,
                    motion_score=change,
                    clarity_score=clarity,
                    text_score=text_score,
                    is_black=mean < 0.035,
                    is_duplicate=bool(results) and change < 0.002,
                )
            )
        return tuple(results)

    def _build_shots(
        self,
        cached: CachedVideo,
        metrics: Sequence[FrameMetrics],
    ) -> tuple[ShotRef, ...]:
        duration = cached.duration_seconds
        boundaries = [0.0]
        current_start = 0.0
        for metric in metrics[1:]:
            elapsed = metric.timestamp_seconds - current_start
            forced = elapsed >= self.config.shot_max_seconds
            changed = (
                metric.change_score >= self.config.shot_change_threshold
                and elapsed >= self.config.shot_min_seconds
            )
            if forced or changed:
                boundary = min(duration, metric.timestamp_seconds)
                if boundary - boundaries[-1] >= self.config.shot_min_seconds:
                    boundaries.append(boundary)
                    current_start = boundary
        if duration - boundaries[-1] < self.config.shot_min_seconds and len(boundaries) > 1:
            boundaries.pop()
        if duration > boundaries[-1] + 1e-6:
            boundaries.append(duration)
        if len(boundaries) == 1:
            boundaries.append(duration)

        shots: list[ShotRef] = []
        for index, (start, end) in enumerate(pairwise(boundaries)):
            shot_frames = tuple(
                frame.id
                for frame in cached.frames
                if start - 1e-6 <= frame.timestamp_seconds <= end + 1e-6
            )
            if not shot_frames:
                shot_frames = (cached.nearest_frame((start + end) / 2).id,)
            shots.append(
                ShotRef(
                    shot_id=f"S{index:04d}",
                    span=TimeSpan(start, end, source="shot"),
                    frame_ids=shot_frames,
                )
            )
        return tuple(shots)

    def _build_tree(
        self,
        shots: Sequence[ShotRef],
    ) -> tuple[dict[str, TemporalNode], str]:
        if not shots:
            raise ValueError("cannot build a temporal tree without shots")
        nodes: dict[str, TemporalNode] = {}
        shot_node_ids: list[str] = []
        for shot in shots:
            nodes[shot.shot_id] = TemporalNode(
                node_id=shot.shot_id,
                span=shot.span,
                level=0,
                child_ids=(),
                shot_ids=(shot.shot_id,),
            )
            shot_node_ids.append(shot.shot_id)

        branch = self.config.locator_branch_factor
        node_counter = 0

        def balanced_groups(values: Sequence[str], count: int) -> tuple[tuple[str, ...], ...]:
            base, remainder = divmod(len(values), count)
            groups: list[tuple[str, ...]] = []
            offset = 0
            for group_index in range(count):
                size = base + (1 if group_index < remainder else 0)
                groups.append(tuple(values[offset : offset + size]))
                offset += size
            return tuple(groups)

        def build_children(values: Sequence[str], depth: int) -> tuple[str, ...]:
            nonlocal node_counter
            if len(values) <= branch:
                return tuple(values)
            children: list[str] = []
            for group in balanced_groups(values, branch):
                if len(group) == 1:
                    children.append(group[0])
                    continue
                child_ids = build_children(group, depth + 1)
                child_nodes = [nodes[item] for item in child_ids]
                node_id = f"L{depth}-N{node_counter:04d}"
                node_counter += 1
                nodes[node_id] = TemporalNode(
                    node_id=node_id,
                    span=TimeSpan(
                        child_nodes[0].span.start_seconds,
                        child_nodes[-1].span.end_seconds,
                        source="time_tree",
                    ),
                    level=depth,
                    child_ids=child_ids,
                    shot_ids=tuple(shot_id for child in child_nodes for shot_id in child.shot_ids),
                )
                children.append(node_id)
            return tuple(children)

        root_id = "ROOT"
        root_children = build_children(shot_node_ids, 1)
        root_nodes = [nodes[item] for item in root_children]
        nodes[root_id] = TemporalNode(
            node_id=root_id,
            span=TimeSpan(
                root_nodes[0].span.start_seconds,
                root_nodes[-1].span.end_seconds,
                source="time_tree_root",
            ),
            level=0,
            child_ids=root_children,
            shot_ids=tuple(shot.shot_id for shot in shots),
        )
        return nodes, root_id


class SourceFrameStore:
    """Timestamp-addressed original-resolution frames and validated normalized crops."""

    def __init__(self, config: P01Config) -> None:
        self.config = config
        self.root = config.resolved_cache_dir / "source"

    def extract(
        self,
        video_path: str | Path,
        timestamps: Iterable[float],
        *,
        purpose: str,
        max_side: int | None = None,
    ) -> tuple[FrameRef, ...]:
        from PIL import Image

        source = Path(video_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Video does not exist: {source}")
        targets = sorted({max(0.0, float(item)) for item in timestamps})
        if not targets:
            return ()
        source_key = self._source_key(source)
        variant = "full" if max_side is None else f"max{max_side}"
        output_dir = self.root / source_key / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        decoded = self._decode_nearest(source, targets)
        refs: list[FrameRef] = []
        for _target, actual, image in decoded:
            milliseconds = max(0, round(actual * 1000))
            frame_id = f"SRC-{source_key}-{milliseconds:012d}"
            target_path = output_dir / f"{frame_id}.jpg"
            if not target_path.is_file():
                rendered = image.convert("RGB")
                if max_side is not None:
                    rendered.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                rendered.save(
                    target_path,
                    format="JPEG",
                    quality=self.config.cache_jpeg_quality,
                    optimize=True,
                )
            refs.append(FrameRef(frame_id, actual, str(target_path)))
        return tuple(_deduplicate_frames(refs))

    def crop(
        self,
        frame: FrameRef,
        bbox: BoundingBox,
        *,
        padding_fraction: float = 0.1,
    ) -> FrameRef:
        if bbox.frame_id != frame.id:
            raise ValueError("crop bbox frame_id does not match source frame")
        if padding_fraction < 0:
            raise ValueError("crop padding must be non-negative")
        from PIL import Image

        source = Path(frame.path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Frame does not exist: {source}")
        crop_dir = source.parent.parent / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(
            f"{bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2},{padding_fraction}".encode()
        ).hexdigest()[:12]
        crop_id = f"CROP-{frame.id}-{digest}"
        target = crop_dir / f"{crop_id}.jpg"
        if not target.is_file():
            with Image.open(source) as image:
                width, height = image.size
                box_width = (bbox.x2 - bbox.x1) / 1000 * width
                box_height = (bbox.y2 - bbox.y1) / 1000 * height
                pad_x = box_width * padding_fraction
                pad_y = box_height * padding_fraction
                left = max(0, math.floor(bbox.x1 / 1000 * width - pad_x))
                top = max(0, math.floor(bbox.y1 / 1000 * height - pad_y))
                right = min(width, math.ceil(bbox.x2 / 1000 * width + pad_x))
                bottom = min(height, math.ceil(bbox.y2 / 1000 * height + pad_y))
                image.convert("RGB").crop((left, top, right, bottom)).save(
                    target,
                    format="JPEG",
                    quality=95,
                    optimize=True,
                )
        return FrameRef(crop_id, frame.timestamp_seconds, str(target))

    @staticmethod
    def _source_key(source: Path) -> str:
        stat = source.stat()
        payload = f"{source}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _decode_nearest(source: Path, targets: Sequence[float]) -> list[tuple[float, float, Any]]:
        import av

        results: list[tuple[float, float, Any]] = []
        with av.open(str(source)) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                raise ValueError(f"No video stream found: {source}")
            if stream.time_base is not None:
                seek_time = max(0.0, targets[0] - 1.0)
                container.seek(
                    int(seek_time / float(stream.time_base)),
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )
            target_index = 0
            previous: tuple[float, Any] | None = None
            for decoded in container.decode(stream):
                if decoded.pts is None or stream.time_base is None:
                    continue
                timestamp = float(decoded.pts * stream.time_base)
                while target_index < len(targets) and targets[target_index] <= timestamp:
                    target = targets[target_index]
                    chosen = (timestamp, decoded)
                    if previous is not None and abs(previous[0] - target) <= abs(
                        timestamp - target
                    ):
                        chosen = previous
                    results.append((target, chosen[0], chosen[1].to_image()))
                    target_index += 1
                previous = (timestamp, decoded)
                if target_index >= len(targets):
                    break
            if target_index < len(targets) and previous is not None:
                while target_index < len(targets):
                    results.append((targets[target_index], previous[0], previous[1].to_image()))
                    target_index += 1
        if len(results) != len(targets):
            raise ValueError(f"Unable to decode requested frames from {source}")
        return results


def _deduplicate_frames(frames: Iterable[FrameRef]) -> list[FrameRef]:
    result: list[FrameRef] = []
    seen: set[str] = set()
    for frame in frames:
        if frame.id in seen:
            continue
        seen.add(frame.id)
        result.append(frame)
    return result


def safe_media_name(value: str) -> str:
    """Return a trace-friendly token without letting role text influence file paths."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized[:64] or "media"
