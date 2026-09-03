from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qwen3vl_agent.coarse_to_fine.types import FrameRef, TimeWindow


def _deduplicate_frames(frames: Iterable[FrameRef]) -> list[FrameRef]:
    result: list[FrameRef] = []
    seen: set[str] = set()
    for frame in frames:
        if frame.id in seen:
            continue
        result.append(frame)
        seen.add(frame.id)
    return result


@dataclass(frozen=True)
class CachedVideo:
    source_path: str
    cache_dir: str
    duration_seconds: float
    source_fps: float | None
    width: int
    height: int
    sample_fps: float
    frames: tuple[FrameRef, ...]
    cache_hit: bool

    def nearest_frame(self, timestamp_seconds: float) -> FrameRef:
        if not self.frames:
            raise ValueError("cached video has no frames")
        return min(
            self.frames,
            key=lambda frame: abs(frame.timestamp_seconds - timestamp_seconds),
        )

    def uniform_frames(
        self,
        count: int,
        window: TimeWindow | None = None,
    ) -> list[FrameRef]:
        if count < 1:
            return []
        start = 0.0 if window is None else window.start_seconds
        end = self.duration_seconds if window is None else window.end_seconds
        if end <= start:
            return [self.nearest_frame(start)]
        targets = [start + (index + 0.5) * (end - start) / count for index in range(count)]
        return _deduplicate_frames(self.nearest_frame(target) for target in targets)

    def frames_for_windows(
        self,
        windows: list[TimeWindow],
        *,
        total: int,
    ) -> list[FrameRef]:
        if not windows or total < 1:
            return []
        base, remainder = divmod(total, len(windows))
        result: list[FrameRef] = []
        for index, window in enumerate(windows):
            count = base + (1 if index < remainder else 0)
            result.extend(self.uniform_frames(max(1, count), window))
        return _deduplicate_frames(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "cache_dir": self.cache_dir,
            "duration_seconds": round(self.duration_seconds, 3),
            "source_fps": self.source_fps,
            "width": self.width,
            "height": self.height,
            "sample_fps": self.sample_fps,
            "cached_frames": len(self.frames),
            "cache_hit": self.cache_hit,
        }


class VideoEvidenceCache:
    """Decode a video once, persist timestamped JPEGs, and retain recent manifests in memory."""

    MANIFEST_VERSION = 1

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        sample_fps: float = 1.0,
        max_side: int = 768,
        jpeg_quality: int = 85,
        lru_size: int = 4,
    ) -> None:
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if max_side < 64:
            raise ValueError("max_side must be at least 64")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.sample_fps = sample_fps
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self.lru_size = max(1, lru_size)
        self._memory: OrderedDict[str, CachedVideo] = OrderedDict()

    def prepare(self, video_path: str | Path) -> CachedVideo:
        source = Path(video_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Video does not exist: {source}")
        key = self._cache_key(source)
        cached = self._memory.get(key)
        if cached is not None:
            self._memory.move_to_end(key)
            return replace(cached, cache_hit=True)

        target_dir = self.cache_dir / key
        manifest_path = target_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                cached = self._load_manifest(manifest_path, source, cache_hit=True)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                cached = self._extract(source, target_dir)
        else:
            cached = self._extract(source, target_dir)
        self._remember(key, cached)
        return cached

    def _cache_key(self, source: Path) -> str:
        stat = source.stat()
        payload = (
            f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{self.sample_fps}|"
            f"{self.max_side}|{self.jpeg_quality}|{self.MANIFEST_VERSION}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]

    def _remember(self, key: str, cached: CachedVideo) -> None:
        self._memory.pop(key, None)
        self._memory[key] = cached
        while len(self._memory) > self.lru_size:
            self._memory.popitem(last=False)

    def _load_manifest(
        self,
        manifest_path: Path,
        source: Path,
        *,
        cache_hit: bool,
    ) -> CachedVideo:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("version") != self.MANIFEST_VERSION:
            raise ValueError(f"Unsupported cache manifest: {manifest_path}")
        frames = tuple(
            FrameRef(
                id=str(item["id"]),
                timestamp_seconds=float(item["timestamp_seconds"]),
                path=str((manifest_path.parent / item["relative_path"]).resolve()),
            )
            for item in manifest["frames"]
        )
        if not frames or any(not Path(frame.path).is_file() for frame in frames):
            raise ValueError(f"Incomplete frame cache: {manifest_path.parent}")
        return CachedVideo(
            source_path=str(source),
            cache_dir=str(manifest_path.parent),
            duration_seconds=float(manifest["duration_seconds"]),
            source_fps=(
                float(manifest["source_fps"]) if manifest.get("source_fps") is not None else None
            ),
            width=int(manifest["width"]),
            height=int(manifest["height"]),
            sample_fps=float(manifest["sample_fps"]),
            frames=frames,
            cache_hit=cache_hit,
        )

    def _extract(self, source: Path, target_dir: Path) -> CachedVideo:
        import av
        from PIL import Image

        frame_dir = target_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "manifest.json"
        entries: list[dict[str, Any]] = []

        with av.open(str(source)) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                raise ValueError(f"No video stream found: {source}")
            source_fps = float(stream.average_rate) if stream.average_rate else None
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)

            next_timestamp = 0.0
            step = 1.0 / self.sample_fps
            last_timestamp = 0.0
            for decoded in container.decode(stream):
                if decoded.pts is None or stream.time_base is None:
                    continue
                timestamp = float(decoded.pts * stream.time_base)
                last_timestamp = max(last_timestamp, timestamp)
                if timestamp + 1e-6 < next_timestamp:
                    continue
                image = decoded.to_image().convert("RGB")
                image.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
                frame_id = f"F{len(entries):06d}"
                relative_path = Path("frames") / f"{frame_id}.jpg"
                image.save(
                    target_dir / relative_path,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=True,
                )
                entries.append(
                    {
                        "id": frame_id,
                        "timestamp_seconds": round(timestamp, 6),
                        "relative_path": relative_path.as_posix(),
                    }
                )
                next_timestamp += step
                if timestamp >= next_timestamp:
                    next_timestamp = (math.floor(timestamp / step) + 1) * step

            if not entries:
                raise ValueError(f"Video decoder returned no frames: {source}")
            if duration is None or duration <= 0:
                duration = last_timestamp + step

            manifest = {
                "version": self.MANIFEST_VERSION,
                "source_path": str(source),
                "duration_seconds": duration,
                "source_fps": source_fps,
                "width": int(stream.width),
                "height": int(stream.height),
                "sample_fps": self.sample_fps,
                "frames": entries,
            }

        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        return self._load_manifest(manifest_path, source, cache_hit=False)


@dataclass(frozen=True)
class SubtitleCue:
    start_seconds: float
    end_seconds: float
    text: str

    def overlaps(self, window: TimeWindow) -> bool:
        return self.end_seconds >= window.start_seconds and self.start_seconds <= window.end_seconds


class SubtitleTrack:
    TIMESTAMP_PATTERN = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})")

    def __init__(self, cues: Iterable[SubtitleCue], *, source_path: str | None = None) -> None:
        self.cues = tuple(cues)
        self.source_path = source_path

    @classmethod
    def from_srt(cls, path: str | Path) -> SubtitleTrack:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8-sig", errors="replace") as file:
            content = file.read()
        cues: list[SubtitleCue] = []
        for block in re.split(r"\r?\n\s*\r?\n", content.strip()):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            timestamp_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
            if timestamp_index is None:
                continue
            left, right = lines[timestamp_index].split("-->", maxsplit=1)
            start = cls._parse_timestamp(left.strip())
            end = cls._parse_timestamp(right.strip().split()[0])
            text = " ".join(lines[timestamp_index + 1 :]).strip()
            text = html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
            if text:
                cues.append(SubtitleCue(start, end, text))
        return cls(cues, source_path=str(source))

    @classmethod
    def _parse_timestamp(cls, value: str) -> float:
        match = cls.TIMESTAMP_PATTERN.search(value)
        if match is None:
            raise ValueError(f"Invalid SRT timestamp: {value!r}")
        return (
            int(match.group("h")) * 3600
            + int(match.group("m")) * 60
            + int(match.group("s"))
            + int(match.group("ms")) / 1000
        )

    def text_for_windows(
        self,
        windows: list[TimeWindow],
        *,
        max_chars: int,
    ) -> str:
        if not windows or max_chars <= 0:
            return ""
        selected: list[str] = []
        seen: set[tuple[float, float, str]] = set()
        for cue in self.cues:
            if not any(cue.overlaps(window) for window in windows):
                continue
            key = (cue.start_seconds, cue.end_seconds, cue.text)
            if key in seen:
                continue
            seen.add(key)
            selected.append(f"[{cue.start_seconds:.3f}s-{cue.end_seconds:.3f}s] {cue.text}")
        text = "\n".join(selected)
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 15)] + "\n...<truncated>"


def build_contact_sheet(
    frames: list[FrameRef],
    labels: list[str],
    *,
    output_dir: str | Path,
    columns: int = 3,
    tile_width: int = 320,
    image_height: int = 180,
    label_height: int = 30,
) -> str:
    if not frames or len(frames) != len(labels):
        raise ValueError("contact sheet requires one label per frame")

    from PIL import Image, ImageDraw, ImageOps

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest_payload = [
        f"layout:{columns}:{tile_width}:{image_height}:{label_height}",
        *(f"{frame.id}:{label}" for frame, label in zip(frames, labels)),
    ]
    digest = hashlib.sha1("|".join(digest_payload).encode("utf-8")).hexdigest()[:16]
    target = output / f"contact_{digest}.jpg"
    if target.is_file():
        return str(target)

    if min(tile_width, image_height, label_height) < 1:
        raise ValueError("contact sheet dimensions must be positive")
    columns = max(1, min(columns, len(frames)))
    rows = math.ceil(len(frames) / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (image_height + label_height)),
        color=(20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (frame, label) in enumerate(zip(frames, labels)):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * (image_height + label_height)
        with Image.open(frame.path) as image:
            tile = ImageOps.fit(image.convert("RGB"), (tile_width, image_height))
        canvas.paste(tile, (x, y))
        draw.rectangle(
            (x, y + image_height, x + tile_width, y + image_height + label_height), fill=(0, 0, 0)
        )
        draw.text((x + 8, y + image_height + 8), label, fill=(255, 255, 255))
    canvas.save(target, format="JPEG", quality=88, optimize=True)
    return str(target)
