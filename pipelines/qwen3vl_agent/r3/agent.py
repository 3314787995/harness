"""Production R3 v5 facade. Historical v4 implementation is kept separately."""
from typing import Any
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from .config import R3Config
from .types import R3Request, R3Result
from .query_media import QueryMedia
from .query_engine import QueryEngine

class R3VideoAgent:
    def __init__(self, model: BaseVideoModel, *, config=None, index_builder=None,
                 source_store=None, provider=None):
        self.model=model
        self.config=config if isinstance(config,R3Config) else R3Config.from_mapping(config)
        self.config.validate()
        self.media=QueryMedia(self.config)
        if index_builder is not None: self.media.shared.index_builder=index_builder
        if source_store is not None: self.media.shared.source_store=source_store
        self.provider=provider

    def load(self):
        if not self.model.is_loaded: self.model.load()

    def unload(self):
        self.model.unload()

    def solve(self, request: R3Request) -> R3Result:
        # Each question has independent candidates, frame visibility and checkpoints.
        self.media.catalog={}
        return QueryEngine(self,request).run()

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[Any] | None = None,
        images: list[str] | None = None,
        choices: Any = None,
        given_interval: Any = None,
        **kwargs: Any,
    ) -> ModelOutput:
        if not videos or len(videos) != 1 or not isinstance(videos[0], str) or images:
            raise ValueError("R3 requires exactly one video and no standalone images")
        if kwargs.pop("subtitle_path", None) is not None:
            raise ValueError("inject a permitted ExternalEvidenceProvider for subtitle/ASR access")
        users = [m for m in messages if m.get("role") == "user"]
        if not users or not isinstance(users[-1].get("content"), str):
            raise ValueError("R3 requires a text question")
        if given_interval is not None:
            if kwargs.get("query_scope") is not None and tuple(given_interval) != tuple(
                kwargs["query_scope"]
            ):
                raise ValueError("given_interval conflicts with query_scope")
            kwargs["query_scope"] = given_interval
        kwargs.setdefault("budget", self.config.budget)
        result = self.solve(
            R3Request(videos[0], users[-1]["content"], choices=choices or (), **kwargs)
        )
        return ModelOutput(result.prediction or "", {"r3": result.to_dict()})
