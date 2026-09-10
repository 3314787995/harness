from __future__ import annotations

import gc
import logging
import math
import re
import time
import json
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from collections.abc import Mapping, Sequence
from typing import Any

from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput, VideoSource

logger = logging.getLogger(__name__)

DEFAULT_GENERATION = {
    "max_new_tokens": 64,
    "temperature": 0.0,
    "top_p": None,
    "top_k": None,
    "num_beams": 1,
    "repetition_penalty": 1.0,
}

DEFAULT_VIDEO = {
    "min_pixels": 4 * 32 * 32,
    "max_pixels": 256 * 32 * 32,
    "total_pixels": 4096 * 32 * 32,
    "fps": 1.0,
    "max_frames": 32,
}


class InputContextExceeded(ValueError):
    """The optional caller budget was exceeded before model generation."""

    def __init__(self, actual_tokens: int, limit: int):
        self.actual_tokens, self.limit = actual_tokens, limit
        super().__init__(f"processed input tokens {actual_tokens} exceed limit {limit}")


class VisualBudgetExceeded(ValueError):
    """A prepared visual input was rejected before GPU transfer/generation."""


def enforce_visual_budget(tokens, pixels, token_limit=None, pixel_limit=None):
    for name, value, limit in (("visual tokens", tokens, token_limit), ("processed pixels", pixels, pixel_limit)):
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError(f"invalid {name} budget")
            if value is None or value > limit:
                raise VisualBudgetExceeded(f"{name}: {value} exceeds {limit}")


class Qwen3VLModel(BaseVideoModel):
    """Focused Qwen3-VL inference wrapper following the official input flow."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        generation: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
        attn_implementation: str | None = None,
        use_cache: bool = True,
        device_map: str | Mapping[str, Any] | None = None,
        max_memory: Mapping[Any, Any] | None = None,
        required_cuda_devices: Sequence[int] | None = None,
        forbid_offload: bool = False,
        revision: str | None = None,
    ) -> None:
        super().__init__(model_path, device=device, dtype=dtype)
        self.generation = {**DEFAULT_GENERATION, **(generation or {})}
        self.video = {**DEFAULT_VIDEO, **(video or {})}
        self.attn_implementation = attn_implementation
        self.use_cache = use_cache
        self.device_map = device_map if device_map is not None else device
        self.max_memory = dict(max_memory or {}) or None
        self.required_cuda_devices = tuple(int(item) for item in (required_cuda_devices or ()))
        self.forbid_offload = bool(forbid_offload)
        self.revision = revision
        self.model = None
        self.processor = None
        self.hf_device_map: dict[str, Any] = {}

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        if self.dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {self.dtype!r}")

        logger.info("Loading Qwen3-VL from %s on %s", self.model_path, self.device)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype_map[self.dtype],
            "device_map": self.device_map,
        }
        if self.max_memory is not None:
            model_kwargs["max_memory"] = self.max_memory
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        if self.revision is not None:
            model_kwargs["revision"] = self.revision

        processor_kwargs = {"revision": self.revision} if self.revision is not None else {}
        self.processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, **model_kwargs
        ).eval()
        self.hf_device_map = dict(getattr(self.model, "hf_device_map", {}) or {})
        self._validate_device_map()
        self._loaded = True

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[VideoSource] | None = None,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        if not self._loaded or self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        import torch
        from qwen_vl_utils import process_vision_info

        video_frame_metadata = kwargs.pop("video_frame_metadata", None)
        input_token_limit = kwargs.pop("input_token_limit", None)
        visual_token_limit = kwargs.pop("visual_token_limit", None)
        processed_pixel_limit = kwargs.pop("processed_pixel_limit", None)
        receipt_path = kwargs.pop("preparation_receipt_path", None)
        prepared = self._build_messages(messages, videos=videos, images=images)
        prompt = self.processor.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            prepared,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadata = None
        if video_inputs is not None:
            video_inputs, video_metadata = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadata = list(video_metadata)

        if video_frame_metadata is not None:
            video_metadata = self._validated_frame_metadata(video_inputs, video_frame_metadata)
            video_kwargs["do_sample_frames"] = False

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        # Persist actual processor accounting even if the subsequent gate refuses input.
        receipt = {"visual_tokens": self._visual_token_count(inputs),
                   "processed_visual_grids": self._visual_grids(inputs),
                   "processed_pixels": self._processed_pixels(inputs),
                   "input_tokens": int(inputs.input_ids.shape[-1]),
                   "image_patch_size": 16, "do_resize": False,
                   "video_frame_metadata": video_frame_metadata, "versions": {}}
        for package in ("torch", "transformers", "qwen-vl-utils", "flash-attn"):
            try:
                receipt["versions"][package] = version(package)
            except PackageNotFoundError:
                receipt["versions"][package] = None
        if receipt_path:
            path = Path(receipt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
        enforce_visual_budget(receipt["visual_tokens"], receipt["processed_pixels"],
                              visual_token_limit, processed_pixel_limit)
        if input_token_limit is not None:
            if isinstance(input_token_limit, bool) or not isinstance(input_token_limit, int) or input_token_limit <= 0:
                raise ValueError("input_token_limit requires a positive integer")
            actual_tokens = int(inputs.input_ids.shape[-1])
            if actual_tokens > input_token_limit:
                raise InputContextExceeded(actual_tokens, input_token_limit)
        inputs = inputs.to(self._input_device())

        generation = {**self.generation, **kwargs}
        generation["use_cache"] = self.use_cache
        temperature = generation.get("temperature")
        generation["do_sample"] = bool(temperature is not None and temperature > 0)
        if not generation["do_sample"]:
            generation.pop("temperature", None)
            generation.pop("top_p", None)
            generation.pop("top_k", None)
        generation = {key: value for key, value in generation.items() if value is not None}

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation)
        latency = time.perf_counter() - started

        generated = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, output_ids)
        ]
        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return ModelOutput(
            text=text,
            metadata={
                "preparation": receipt,
                "input_tokens": int(inputs.input_ids.shape[-1]),
                "output_tokens": int(generated[0].shape[-1]),
                "latency_seconds": latency,
                "generation": generation,
                "visual_tokens": self._visual_token_count(inputs),
                "processed_visual_grids": self._visual_grids(inputs),
                "processed_pixels": self._processed_pixels(inputs),
                "hf_device_map": self._serializable_device_map(),
                "gpu_memory": self._gpu_memory_snapshot(),
                **({"video_timing_verified": True,
                    "encoded_video_frames": [int(v.shape[0]) for v in video_inputs]}
                   if video_frame_metadata is not None else {}),
            },
        )

    @staticmethod
    def _processed_pixels(inputs):
        """Encoded RGB pixel positions, including the processor's temporal patch expansion."""
        count = 0
        for name in ("pixel_values", "pixel_values_videos"):
            value = inputs.get(name) if hasattr(inputs, "get") else None
            if value is not None:
                if not hasattr(value, "numel"):
                    return None
                count += int(value.numel()) // 3
        return count

    @staticmethod
    def _visual_grids(inputs):
        result = {}
        for name in ("image_grid_thw", "video_grid_thw"):
            grid = inputs.get(name) if hasattr(inputs, "get") else None
            if grid is not None:
                try:
                    result[name] = grid.detach().cpu().tolist()
                except AttributeError:
                    result[name] = grid.tolist() if hasattr(grid, "tolist") else grid
        return result

    @staticmethod
    def _validated_frame_metadata(video_inputs, selected):
        """Optional preselected-frame contract; legacy callers retain their input path."""
        if not isinstance(selected, list) or len(selected) != len(video_inputs or []):
            raise ValueError("selected metadata must match the encoded video count")
        result = []
        for tensor, entry in zip(video_inputs or [], selected):
            count = int(tensor.shape[0])
            indices, times, ids = (entry.get(key, []) for key in
                                   ("frames_indices", "source_timestamps", "frame_ids"))
            fps = entry.get("fps")
            if (isinstance(fps, bool) or not isinstance(fps, (int, float))
                    or not math.isfinite(fps) or fps <= 0):
                raise ValueError("selected video metadata requires positive source FPS")
            if count < 2 or count % 2 or any(len(v) != count for v in (indices, times, ids)):
                raise ValueError("implicit frame padding or mismatched selected frame counts")
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
                raise ValueError("selected source frame indices must be nonnegative integers")
            if any(indices[i] >= indices[i+1] for i in range(count-1)) or len(set(ids)) != count:
                raise ValueError("selected video frames must be ordered and distinct")
            if any(not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t)
                   or abs(i/fps-t) > 1e-6 for i, t in zip(indices, times)):
                raise ValueError("source timestamps do not match the actual source frame indices")
            total = entry.get("total_num_frames")
            if isinstance(total, bool) or not isinstance(total, int) or total <= max(indices, default=-1):
                raise ValueError("invalid source video frame extent")
            result.append({"fps": fps, "frames_indices": list(indices), "total_num_frames": total})
        return result

    def unload(self) -> None:
        import torch

        self.model = None
        self.processor = None
        self.hf_device_map = {}
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[VideoSource] | None,
        images: list[str] | None,
    ) -> list[dict[str, Any]]:
        if not messages:
            raise ValueError("messages must not be empty")

        prepared: list[dict[str, Any]] = []
        media_injected = self._contains_media(messages)
        for message in messages:
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported message role: {role!r}")
            content = message.get("content", "")

            if isinstance(content, list):
                parts = list(content)
                if role == "user" and not media_injected:
                    media_parts = self._video_parts(videos or [])
                    media_parts.extend(
                        {"type": "image", "image": path} for path in images or []
                    )
                    if media_parts:
                        parts = media_parts + parts
                        media_injected = True
                prepared.append({"role": role, "content": parts})
                continue

            parts: list[dict[str, Any]] = []
            if role == "user" and not media_injected:
                parts.extend(self._video_parts(videos or []))
                parts.extend({"type": "image", "image": path} for path in images or [])
                media_injected = bool(parts)
            parts.append({"type": "text", "text": str(content)})
            prepared.append({"role": role, "content": parts})

        return prepared

    def _video_parts(self, videos: list[VideoSource]) -> list[dict[str, Any]]:
        return [
            {
                "type": "video",
                "video": path,
                **{key: value for key, value in self.video.items() if value is not None},
            }
            for path in videos
        ]

    @staticmethod
    def _contains_media(messages: list[dict[str, Any]]) -> bool:
        return any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") in {"image", "image_url", "video"}
                for part in message["content"]
            )
            for message in messages
        )

    def _validate_device_map(self) -> None:
        placements = {self._placement_name(value) for value in self.hf_device_map.values()}
        if self.forbid_offload and placements.intersection({"cpu", "disk", "meta"}):
            raise RuntimeError(
                "model device map contains forbidden CPU/disk offload: "
                + ", ".join(sorted(placements))
            )
        if self.required_cuda_devices:
            present = {
                int(match.group(1))
                for value in placements
                if (match := re.match(r"(?:cuda:)?(\d+)$", value))
            }
            missing = set(self.required_cuda_devices) - present
            if missing:
                raise RuntimeError(
                    "model was not sharded across required CUDA devices: "
                    f"required={list(self.required_cuda_devices)}, present={sorted(present)}, "
                    f"device_map={self._serializable_device_map()}"
                )

    @staticmethod
    def _placement_name(value: Any) -> str:
        if isinstance(value, int):
            return str(value)
        return str(value).casefold()

    def _serializable_device_map(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.hf_device_map.items()}

    def _input_device(self) -> Any:
        if self.model is None:
            raise RuntimeError("model is unavailable")
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, RuntimeError):
            return self.model.device

    def _visual_token_count(self, inputs: Any) -> int:
        if self.model is None:
            return 0
        raw_tokens = 0
        for name in ("image_grid_thw", "video_grid_thw"):
            grid = inputs.get(name) if hasattr(inputs, "get") else None
            if grid is None:
                continue
            try:
                rows = grid.detach().cpu().tolist()
            except AttributeError:
                rows = grid
            for row in rows:
                if len(row) == 3:
                    raw_tokens += int(row[0]) * int(row[1]) * int(row[2])
        vision_config = getattr(getattr(self.model, "config", None), "vision_config", None)
        merge = int(getattr(vision_config, "spatial_merge_size", 2) or 2)
        return (raw_tokens + merge * merge - 1) // (merge * merge)

    @staticmethod
    def _gpu_memory_snapshot() -> list[dict[str, Any]]:
        try:
            import torch

            if not torch.cuda.is_available():
                return []
            return [
                {
                    "device_index": index,
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
                for index in range(torch.cuda.device_count())
            ]
        except Exception:  # noqa: BLE001 - telemetry cannot invalidate generation
            return []
