from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_ENV_WITH_DEFAULT = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*):-(?P<default>[^}]*)\}"
)


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    return _expand_env(value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        value = _ENV_WITH_DEFAULT.sub(
            lambda match: os.environ.get(match.group("name")) or match.group("default"),
            value,
        )
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value
