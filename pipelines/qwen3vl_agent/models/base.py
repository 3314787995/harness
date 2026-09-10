from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

VideoSource = str | list[str]


@dataclass
class ModelOutput:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVideoModel(ABC):
    def __init__(self, model_path: str, *, device: str = "auto", dtype: str = "bfloat16"):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[VideoSource] | None = None,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded
