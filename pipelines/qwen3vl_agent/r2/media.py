"""R2 compatibility entry for the shared, unchanged PTS media service."""

from qwen3vl_agent.temporal_media import (
    TemporalMedia,
    TemporalPrepared,
    source_point,
)

R2Prepared = TemporalPrepared
__all__ = ["R2Media", "R2Prepared", "source_point"]


class R2Media(TemporalMedia):
    def __init__(self, config):
        super().__init__(config, namespace="r2")
