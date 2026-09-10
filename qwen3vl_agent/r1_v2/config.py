from dataclasses import dataclass, field

from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.r1.config import R1Config


@dataclass(frozen=True)
class R1V2Config(R1Config):
    """The original R1 limits and media settings, with an independent cache."""

    media: P01Config = field(
        default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r1_v2")
    )
