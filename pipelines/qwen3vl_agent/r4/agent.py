"""Public R4 facade for the v5 collection controller."""
from __future__ import annotations
from typing import Any
from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.r1.config import R1Config
from qwen3vl_agent.r3.media import R3SourceFrameStore
from .config import R4Config
from .media import R4Media
from .types import R4Request, R4Result
from .controller import CollectionController
from .session import StageFailure

class R4VideoAgent:
    def __init__(
        self,
        model: BaseVideoModel,
        config: R4Config | dict[str, Any] | None = None,
        provider: Any = None,
        *,
        index_builder: Any = None,
        source_store: Any = None,
    ) -> None:
        self.model = model
        self.config = config if isinstance(config, R4Config) else R4Config.from_mapping(config)
        self.config.validate()
        self.provider = provider
        self.media = R4Media(
            R1Config(media=self.config.media),
            index_builder,
            source_store or R3SourceFrameStore(self.config.media),
        )

    def load(self) -> None:
        if not self.model.is_loaded:
            self.model.load()

    def unload(self) -> None:
        self.model.unload()

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[str] | None = None,
        images: list[str] | None = None,
        choices: Any = None,
        given_interval: Any = None,
        **kwargs: Any,
    ) -> ModelOutput:
        if not videos or len(videos) != 1 or images:
            raise ValueError(
                "R4 compatibility entry accepts one video; use R4Request.sources for history"
            )
        users = [m for m in messages if m.get("role") == "user"]
        if not users or not isinstance(users[-1].get("content"), str):
            raise ValueError("R4 requires a text question")
        if given_interval is not None:
            if kwargs.get("query_scope") is not None and list(kwargs["query_scope"]) != list(
                given_interval
            ):
                raise ValueError("given_interval conflicts with query_scope")
            kwargs["query_scope"] = given_interval
        subtitle = kwargs.pop("subtitle_path", None)
        if subtitle:
            kwargs["external_files"] = [{"path": subtitle, "kind": "subtitle"}]
            kwargs["available_modalities"] = ("video", "screen_text", "subtitle")
        kwargs.setdefault("budget", self.config.budget)
        result = self.solve(
            R4Request(
                question=users[-1]["content"], video_path=videos[0], choices=choices or (), **kwargs
            )
        )
        return ModelOutput(result.prediction or "", {"r4": result.to_dict()})

    def solve(self, request: R4Request) -> R4Result:
        try:
            controller = CollectionController(self, request)
        except (ValueError, OSError, ImportError, StageFailure) as exc:
            # A signature mismatch is a caller error, never permission to reuse old state.
            if request.resume and isinstance(exc, ValueError):
                raise
            failure = exc.failure if isinstance(exc, StageFailure) else {
                "stage": "input", "code": type(exc).__name__, "message": str(exc)}
            return R4Result(None, {"results": [], "sets": {}}, "execution_error", "unsupported",
                            "pipeline_failure", {}, [], [str(exc)], {}, {"model_calls": 0}, {}, failure,
                            "execution_failed")
        return controller.run()
