from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VIDEOMME_ROOT = "data/videomme"


@dataclass(frozen=True)
class VideoMMEPaths:
    root: Path
    annotation: Path
    videos: Path
    subtitles: Path


def default_videomme_paths() -> VideoMMEPaths:
    """Return the conventional Video-MME layout, overridable with VIDEOMME_ROOT."""

    root = Path(os.environ.get("VIDEOMME_ROOT", DEFAULT_VIDEOMME_ROOT)).expanduser()
    return VideoMMEPaths(
        root=root,
        annotation=root / "videomme" / "test-00000-of-00001.parquet",
        videos=root / "videos",
        subtitles=root / "subtitle",
    )
