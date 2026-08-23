from __future__ import annotations

import gc
import logging
import time
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
    ) -> None:
        super().__init__(model_path, device=device, dtype=dtype)
        self.generation = {**DEFAULT_GENERATION, **(generation or {})}
        self.video = {**DEFAULT_VIDEO, **(video or {})}
        self.attn_implementation = attn_implementation
        self.use_cache = use_cache
        self.model = None
        self.processor = None

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
            "device_map": self.device,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, **model_kwargs
        ).eval()
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

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(self.model.device)

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
                "input_tokens": int(inputs.input_ids.shape[-1]),
                "output_tokens": int(generated[0].shape[-1]),
                "latency_seconds": latency,
                "generation": generation,
            },
        )

    def unload(self) -> None:
        import torch

        self.model = None
        self.processor = None
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
