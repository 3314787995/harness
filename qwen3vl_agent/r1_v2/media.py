"""Bounded real shot indexing and non-clustered navigation for R1 V2."""

import math
from itertools import pairwise
from pathlib import Path

from qwen3vl_agent.coarse_to_fine.cache import CachedVideo
from qwen3vl_agent.p01.media import P01IndexBuilder, P01VideoIndex, ShotRef, SourceFrameStore
from qwen3vl_agent.p01.types import TimeSpan
from qwen3vl_agent.r1.media import R1Media


def navigation_frames(frames, shown_ids, limit=3):
    """First/end/middle initially; then bisect the largest remaining temporal gaps."""
    frames = sorted({f.id: f for f in frames}.values(), key=lambda f: f.timestamp_seconds)
    available = [f for f in frames if f.id not in shown_ids]
    if not available:
        return ()
    shown = [f.timestamp_seconds for f in frames if f.id in shown_ids]
    if not shown:
        picks = (available[0], available[len(available) // 2], available[-1])
        return tuple({f.id: f for f in picks}.values())[:limit]
    selected = []
    while available and len(selected) < limit:
        boundaries = sorted({frames[0].timestamp_seconds, frames[-1].timestamp_seconds, *shown})
        gaps = sorted(pairwise(boundaries), key=lambda p: (-(p[1] - p[0]), p[0]))
        pick = None
        for start, end in gaps:
            candidates = [f for f in available if start <= f.timestamp_seconds <= end]
            if candidates:
                middle = (start + end) / 2
                pick = min(
                    candidates,
                    key=lambda f: (abs(f.timestamp_seconds - middle), f.timestamp_seconds),
                )
                break
        if pick is None:
            pick = available[0]
        selected.append(pick)
        shown.append(pick.timestamp_seconds)
        available = [f for f in available if f.id != pick.id]
    return tuple(sorted(selected, key=lambda f: f.timestamp_seconds))


class R1V2IndexBuilder(P01IndexBuilder):
    def __init__(self, config, source_store=None):
        super().__init__(config)
        self.source_store = source_store or SourceFrameStore(config)

    def prepare_interval(self, video_path, span, *, metadata=None):
        source = Path(video_path).expanduser().resolve()
        details = metadata or self.probe(source)
        start = max(0.0, span.start_seconds)
        end = min(details.duration_seconds, span.end_seconds)
        if end <= start:
            raise ValueError("allowed scope has no decodable duration")
        allowed = TimeSpan(start, end, source="r1_v2_navigation")
        step = 1.0 / self.config.index_fps
        times = [start + i * step for i in range(math.floor((end - start) / step) + 1)]
        if times[-1] < end - 1e-6:
            times.append(end)
        decoded = self.source_store.extract(
            source, times, purpose="r1_v2_navigation", max_side=self.config.index_max_side
        )
        frames = tuple(
            sorted(
                {
                    f.id: f
                    for f in decoded
                    if allowed.contains_evidence(f.timestamp_seconds, tolerance=1e-6)
                }.values(),
                key=lambda f: f.timestamp_seconds,
            )
        )
        if not frames:
            raise ValueError("decoder returned no navigation frames inside allowed scope")
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
        # The P01 whole-video splitter starts at zero; bounded indexing must retain absolute PTS.
        boundaries = [start]
        for metric in metrics[1:]:
            elapsed = metric.timestamp_seconds - boundaries[-1]
            forced = elapsed >= self.config.shot_max_seconds
            changed = (
                metric.change_score >= self.config.shot_change_threshold
                and elapsed >= self.config.shot_min_seconds
            )
            if (forced or changed) and elapsed >= self.config.shot_min_seconds:
                boundaries.append(min(end, metric.timestamp_seconds))
        if end - boundaries[-1] < self.config.shot_min_seconds and len(boundaries) > 1:
            boundaries.pop()
        if end > boundaries[-1] + 1e-6:
            boundaries.append(end)
        shots = tuple(
            ShotRef(
                f"S{i:04d}",
                TimeSpan(a, b, source="shot"),
                tuple(f.id for f in frames if a - 1e-6 <= f.timestamp_seconds <= b + 1e-6),
            )
            for i, (a, b) in enumerate(pairwise(boundaries))
        )
        nodes, root_id = self._build_tree(shots)
        return P01VideoIndex(cached, metrics, shots, nodes, root_id)


class R1V2Media(R1Media):
    def __init__(self, config, index_builder=None, source_store=None):
        store = source_store or SourceFrameStore(config.media)
        super().__init__(config, index_builder or R1V2IndexBuilder(config.media, store), store)


def index_report(index):
    """Describe the real index without claiming the model observed its cached frames."""
    return {
        "kind": "r1_v2_bounded_shot_tree",
        "root_id": index.root_id,
        "scope": index.node(index.root_id).span.to_dict(),
        "navigation_fps": index.cached_video.sample_fps,
        "cached_frame_count": len(index.cached_video.frames),
        "shot_count": len(index.shots),
        "node_count": len(index.nodes),
        "nodes": [node.to_dict() for node in index.nodes.values()],
        "shots": [
            {
                **shot.to_dict(),
                "initial_representatives": [
                    {"id": f.id, "timestamp_seconds": f.timestamp_seconds}
                    for f in navigation_frames(
                        [f for f in index.cached_video.frames if f.id in set(shot.frame_ids)], set()
                    )
                ],
            }
            for shot in index.shots
        ],
    }
