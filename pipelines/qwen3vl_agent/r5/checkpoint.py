"""R5-specific version identity over the shared append-only checkpoint engine."""

import hashlib
from pathlib import Path

from qwen3vl_agent.r3.checkpoint import Checkpoint, file_digest


def implementation_digest() -> str:
    package = Path(__file__).parent.parent
    paths = list(Path(__file__).parent.glob("*.py"))
    for directory in ("r1", "r3", "r4", "p01", "models"):
        paths.extend((package / directory).glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = ["Checkpoint", "file_digest", "implementation_digest"]
