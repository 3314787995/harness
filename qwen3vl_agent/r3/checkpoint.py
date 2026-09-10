"""Version-checked append-only checkpoints, including in-flight resource receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _difference(old: Any, new: Any, path: list[Any] | None = None) -> list[list[Any]]:
    path = path or []
    if old == new:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        changes = [["delete", [*path, key], None] for key in old.keys() - new.keys()]
        for key, value in new.items():
            changes.extend(
                _difference(old[key], value, [*path, key])
                if key in old
                else [["set", [*path, key], value]]
            )
        return changes
    if isinstance(old, list) and isinstance(new, list) and len(new) >= len(old):
        changes = []
        for i, value in enumerate(old):
            changes.extend(_difference(value, new[i], [*path, i]))
        if len(new) > len(old):
            changes.append(["append", path, new[len(old) :]])
        return changes
    return [["set", path, new]]


def _apply_changes(state: Any, changes: list[list[Any]]) -> Any:
    for operation, path, value in changes:
        if operation == "set" and not path:
            state = value
            continue
        parent = state
        for key in path[:-1] if operation != "append" else path:
            parent = parent[key]
        if operation == "append":
            parent.extend(value)
        elif operation == "set":
            parent[path[-1]] = value
        elif operation == "delete":
            del parent[path[-1]]
        else:
            raise ValueError("unknown checkpoint delta operation")
    return state


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_digest() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).parent.parent
    shared = (
        "temporal_media.py",
        "r1/media.py",
        "r1/config.py",
        "r1/providers.py",
        "r1/types.py",
        "p01/media.py",
        "p01/config.py",
        "models/base.py",
        "models/qwen3vl.py",
    )
    for path in sorted([*Path(__file__).parent.glob("*.py"), *(package / name for name in shared)]):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Checkpoint:
    def __init__(self, path: str | None, fingerprint: dict[str, Any], *, resume: bool) -> None:
        self.path = Path(path).expanduser().resolve() if path else None
        self.fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True, default=str))
        self.restored: dict[str, Any] | None = None
        self._previous: dict[str, Any] | None = None
        if not self.path:
            return
        if self.path.exists():
            if not resume:
                raise ValueError("checkpoint already exists; use resume or a new checkpoint path")
            last_good, size = 0, self.path.stat().st_size
            with self.path.open("rb") as stream:
                for i, line in enumerate(stream):
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        if stream.tell() != size or i == 0:
                            raise ValueError("corrupt checkpoint before final record") from None
                        break
                    if i == 0:
                        if record != {"kind": "header", "fingerprint": self.fingerprint}:
                            raise ValueError(
                                "checkpoint media/request/model/config/implementation mismatch"
                            )
                    elif record.get("kind") == "state":
                        self.restored = record["state"]
                    elif record.get("kind") == "delta":
                        self.restored = _apply_changes(self.restored, record["changes"])
                    else:
                        raise ValueError("invalid checkpoint record")
                    last_good = stream.tell()
            if not last_good:
                raise ValueError("empty checkpoint")
            self._previous = json.loads(json.dumps(self.restored))
            if last_good < size:
                # Only the incomplete final transaction of this validated checkpoint is removed.
                with self.path.open("r+b") as stream:
                    stream.truncate(last_good)
            elif not line.endswith(b"\n"):
                with self.path.open("ab") as stream:
                    stream.write(b"\n")
        elif resume:
            raise ValueError("checkpoint to resume does not exist")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._append({"kind": "header", "fingerprint": self.fingerprint})

    def _append(self, value: dict[str, Any]) -> None:
        if self.path:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str
                    )
                    + "\n"
                )
                stream.flush()

    def save(self, state: dict[str, Any]) -> None:
        if not self.path:
            return
        snapshot = json.loads(json.dumps(state, allow_nan=False, default=str))
        if self._previous is None:
            self._append({"kind": "state", "state": snapshot})
        else:
            changes = _difference(self._previous, snapshot)
            if changes:
                self._append({"kind": "delta", "changes": changes})
        self._previous = snapshot
