"""Reuse the append-only checkpoint engine with an R4-specific implementation hash."""

import hashlib
from pathlib import Path

from qwen3vl_agent.r3.checkpoint import Checkpoint, file_digest


def model_signature(model):
    """Cheap pre-load identity: config hashes and immutable-shard file metadata."""
    root = Path(model.model_path).expanduser()
    files = {}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model"}:
                files[path.relative_to(root).as_posix()] = ({"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                    if path.suffix in {".safetensors", ".bin"} else file_digest(path))
    return {"path": model.model_path, "revision": getattr(model, "revision", None), "files": files}


def implementation_digest() -> str:
    package = Path(__file__).parent.parent
    paths = list(Path(__file__).parent.glob("*.py"))
    for name in (
        "r1/media.py",
        "r1/types.py",
        "r1/providers.py",
        "r1/config.py",
        "r3/runtime.py",
        "r3/checkpoint.py",
        "r3/planning.py",
        "r3/types.py",
        "r3/media.py",
        "p01/media.py",
        "p01/config.py",
        "models/base.py",
        "models/qwen3vl.py",
    ):
        paths.append(package / name)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = ["Checkpoint", "file_digest", "implementation_digest"]
