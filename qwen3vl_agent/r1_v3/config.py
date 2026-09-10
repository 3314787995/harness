from dataclasses import dataclass, field

from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.r1.config import R1Config


@dataclass(frozen=True)
class R1V3Config(R1Config):
    """Unchanged R1 model/media/budget settings with a separate cache."""

    media: P01Config = field(
        default_factory=lambda: P01Config(cache_dir=".cache/qwen3vl_agent/r1_v3")
    )
