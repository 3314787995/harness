from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from qwen3vl_agent.models.qwen3vl import Qwen3VLModel

DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"


def build_model(config: Mapping[str, Any] | None = None) -> Qwen3VLModel:
    model_config = dict(config or {})
    return Qwen3VLModel(
        model_config.pop(
            "path",
            os.environ.get("QWEN3VL_MODEL_PATH", DEFAULT_MODEL_PATH),
        ),
        device=model_config.pop("device", "auto"),
        dtype=model_config.pop("dtype", "bfloat16"),
        generation=model_config.pop("generation", None),
        video=model_config.pop("video", None),
        attn_implementation=model_config.pop("attn_implementation", None),
        **model_config,
    )
