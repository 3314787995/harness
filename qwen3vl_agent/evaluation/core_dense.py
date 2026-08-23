from __future__ import annotations

import gc
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from qwen3vl_agent.coarse_to_fine import (
    CachedVideo,
    SubtitleTrack,
    TimeWindow,
    VideoEvidenceCache,
)
from qwen3vl_agent.coarse_to_fine.adapters import MultipleChoiceAdapter
from qwen3vl_agent.coarse_to_fine.types import FrameRef
from qwen3vl_agent.evaluation.ablations import (
    ABLATION_POLICY_ID,
    build_unified_answer_prompt,
    choose_oracle_set,
)
from qwen3vl_agent.evaluation.evidence30 import (
    Exposure,
    canonical_sha256,
    score_relaxed_grounding,
    sha256_file,
    subtitle_exposures,
)
from qwen3vl_agent.evaluation.videomme import VideoMMEQuestion
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.models.base import BaseVideoModel

CORE_DENSE_POLICY_ID = "evidence30-oracle-core-density/1.1"
CORE_VARIANTS = ("core_16", "core_32")
EXPECTED_ORACLE_FAILURES = 9


@dataclass(frozen=True)
class DevReferenceSet:
    """Dev-only Evidence30 view that never opens locked.jsonl."""

    root: Path
    manifest: Mapping[str, Any]
    question_ids: tuple[str, ...]
    references: Mapping[str, Mapping[str, Any]]
    dev_sha256: str

    @classmethod
    def load(cls, root: str | Path) -> DevReferenceSet:
        resolved = Path(root).expanduser().resolve()
        manifest_path = resolved / "manifest.json"
        dev_path = resolved / "dev.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        question_ids = tuple(str(item) for item in manifest["splits"]["dev_question_ids"])
        records: dict[str, Mapping[str, Any]] = {}
        for line in dev_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = str(record["source"]["question_id"])
            if str(record.get("split")) != "dev":
                raise ValueError(f"non-dev reference found in dev.jsonl: {question_id}")
            if question_id in records:
                raise ValueError(f"duplicate dev reference: {question_id}")
            records[question_id] = record
        if set(records) != set(question_ids):
            missing = sorted(set(question_ids) - set(records))
            extra = sorted(set(records) - set(question_ids))
            raise ValueError(f"dev manifest/reference mismatch; missing={missing}, extra={extra}")
        digest = sha256_file(dev_path)
        expected = manifest.get("artifact_sha256", {}).get("dev")
        if expected and digest != expected:
            raise ValueError("dev reference hash does not match manifest")
        return cls(
            root=resolved,
            manifest=manifest,
            question_ids=question_ids,
            references=records,
            dev_sha256=digest,
        )

    def reference(self, question_id: str) -> Mapping[str, Any]:
        try:
            return self.references[question_id]
        except KeyError as exc:
            raise KeyError(f"unknown dev question_id: {question_id}") from exc


@dataclass(frozen=True)
class CoreVariantConfig:
    max_frames: int
    min_pixels: int = 4_096
    max_pixels: int = 131_072
    total_pixels: int = 2_097_152
    video_fps: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CoreVariantConfig:
        config = cls(
            max_frames=int(value["max_frames"]),
            min_pixels=int(value.get("min_pixels", 4_096)),
            max_pixels=int(value.get("max_pixels", 131_072)),
            total_pixels=int(value.get("total_pixels", 2_097_152)),
            video_fps=float(value.get("video_fps", 1.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_frames < 1:
            raise ValueError("core variant max_frames must be positive")
        if self.min_pixels < 1 or self.max_pixels < self.min_pixels:
            raise ValueError("invalid core variant per-frame pixel limits")
        if self.total_pixels < self.max_pixels:
            raise ValueError("core variant total_pixels must be at least max_pixels")
        if self.video_fps <= 0:
            raise ValueError("core variant video_fps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_frames": self.max_frames,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "total_pixels": self.total_pixels,
            "video_fps": self.video_fps,
        }


@dataclass(frozen=True)
class CoreDenseConfig:
    cache_dir: str
    variants: Mapping[str, CoreVariantConfig]
    sample_fps: float = 1.0
    cache_max_side: int = 768
    cache_jpeg_quality: int = 85
    cache_lru_size: int = 4
    subtitle_max_chars: int = 8_000
    max_new_tokens: int = 16

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> CoreDenseConfig:
        data = dict(value or {})
        if "cache_dir" not in data:
            raise ValueError("core_dense.cache_dir is required")
        raw_variants = data.get("variants")
        if not isinstance(raw_variants, Mapping):
            raise TypeError("core_dense.variants must be a mapping")
        variants = {
            str(name): CoreVariantConfig.from_mapping(item)
            for name, item in raw_variants.items()
            if isinstance(item, Mapping)
        }
        if set(variants) != set(CORE_VARIANTS):
            raise ValueError(f"core_dense.variants must be exactly {CORE_VARIANTS}")
        config = cls(
            cache_dir=str(data["cache_dir"]),
            variants=variants,
            sample_fps=float(data.get("sample_fps", 1.0)),
            cache_max_side=int(data.get("cache_max_side", 768)),
            cache_jpeg_quality=int(data.get("cache_jpeg_quality", 85)),
            cache_lru_size=int(data.get("cache_lru_size", 4)),
            subtitle_max_chars=int(data.get("subtitle_max_chars", 8_000)),
            max_new_tokens=int(data.get("max_new_tokens", 16)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("core_dense.sample_fps must be positive")
        if self.cache_max_side < 64:
            raise ValueError("core_dense.cache_max_side must be at least 64")
        if self.cache_lru_size < 1:
            raise ValueError("core_dense.cache_lru_size must be positive")
        if self.subtitle_max_chars < 0:
            raise ValueError("core_dense.subtitle_max_chars must be non-negative")
        if self.max_new_tokens < 1:
            raise ValueError("core_dense.max_new_tokens must be positive")

    def variant(self, name: str) -> CoreVariantConfig:
        if name not in CORE_VARIANTS:
            raise ValueError(f"unsupported core variant: {name}")
        return self.variants[name]


@dataclass(frozen=True)
class CoreWindowSpec:
    evidence_id: str
    slot_id: str
    modality: str
    window: TimeWindow

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "slot_id": self.slot_id,
            "modality": self.modality,
            "window": self.window.to_dict(),
        }


@dataclass(frozen=True)
class CoreEvidencePacket:
    variant: str
    frames: tuple[FrameRef, ...]
    subtitles: str
    source_frame_count: int
    oracle_set_id: str
    frame_specs: tuple[CoreWindowSpec, ...]
    subtitle_specs: tuple[CoreWindowSpec, ...]
    frame_allocations: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if self.variant not in CORE_VARIANTS:
            raise ValueError(f"unsupported core packet variant: {self.variant}")
        if not self.frames:
            raise ValueError("core packet has no frames")

    def exposures(self) -> list[dict[str, Any]]:
        result = [
            Exposure(
                stage="core_dense_answer_head",
                modality="visual",
                start_seconds=frame.timestamp_seconds,
                end_seconds=frame.timestamp_seconds,
                frame_id=frame.id,
            ).to_dict()
            for frame in self.frames
        ]
        result.extend(
            item.to_dict()
            for item in subtitle_exposures(
                self.subtitles,
                stage="core_dense_answer_head",
            )
        )
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "source_frame_count": self.source_frame_count,
            "selected_frame_count": len(self.frames),
            "frames": [frame.to_dict() for frame in self.frames],
            "subtitle_chars": len(self.subtitles),
            "subtitle_intervals": [
                {
                    "start_seconds": item.start_seconds,
                    "end_seconds": item.end_seconds,
                }
                for item in subtitle_exposures(
                    self.subtitles,
                    stage="core_dense_answer_head",
                )
            ],
            "oracle_set_id": self.oracle_set_id,
            "frame_specs": [item.to_dict() for item in self.frame_specs],
            "subtitle_specs": [item.to_dict() for item in self.subtitle_specs],
            "frame_allocations": [dict(item) for item in self.frame_allocations],
        }


def load_oracle_failure_rows(
    path: str | Path,
    *,
    dev_question_ids: Sequence[str],
    expected_count: int = EXPECTED_ORACLE_FAILURES,
) -> dict[str, Mapping[str, Any]]:
    """Select wrong Oracle-context rows while proving the source contains dev only."""

    source = Path(path).expanduser().resolve()
    allowed = set(dev_question_ids)
    oracle_rows: dict[str, Mapping[str, Any]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row.get("question_id"))
        if question_id not in allowed:
            raise ValueError("baseline ablation contains a non-dev question ID")
        if row.get("packet_source") != "oracle_context":
            continue
        if question_id in oracle_rows:
            raise ValueError(f"duplicate Oracle-context baseline row: {question_id}")
        if row.get("policy_id") != ABLATION_POLICY_ID:
            raise ValueError("Oracle baseline policy is not the official v2 policy")
        if not bool(row.get("completed")) or row.get("error"):
            raise ValueError(f"incomplete Oracle baseline row: {question_id}")
        if not bool(row.get("relaxed_grounding", {}).get("grounded")):
            raise ValueError(f"ungrounded Oracle baseline row: {question_id}")
        if float(row.get("relaxed_grounding", {}).get("core_item_recall", 0)) != 1.0:
            raise ValueError(f"Oracle baseline lacks full core recall: {question_id}")
        if not isinstance(row.get("correct"), bool):
            raise TypeError(f"Oracle baseline has invalid correctness: {question_id}")
        oracle_rows[question_id] = row
    if set(oracle_rows) != allowed:
        missing = sorted(allowed - set(oracle_rows))
        raise ValueError(f"Oracle baseline is missing {len(missing)} dev rows")
    failures = {
        question_id: oracle_rows[question_id]
        for question_id in dev_question_ids
        if not bool(oracle_rows[question_id]["correct"])
    }
    if len(failures) != expected_count:
        raise ValueError(
            f"expected {expected_count} Oracle-context failures, found {len(failures)}"
        )
    return failures


def core_window_specs(
    reference: Mapping[str, Any],
    *,
    modalities: set[str] | None = None,
) -> tuple[str, list[CoreWindowSpec]]:
    evidence_set = choose_oracle_set(reference)
    required_slots = {
        str(item["slot_id"])
        for item in reference["evidence_contract"]["evidence_slots"]
        if bool(item.get("required", True))
    }
    specs: list[CoreWindowSpec] = []
    for index, item in enumerate(evidence_set["items"]):
        modality = str(item["modality"])
        if str(item["slot_id"]) not in required_slots:
            continue
        if modalities is not None and modality not in modalities:
            continue
        interval = item["core_interval"]
        start = float(interval["start_sec"])
        end = max(float(interval["end_sec"]), start + 0.001)
        specs.append(
            CoreWindowSpec(
                evidence_id=str(item["evidence_id"]),
                slot_id=str(item["slot_id"]),
                modality=modality,
                window=TimeWindow(f"CORE-{index:02d}", start, end, depth=0),
            )
        )
    return str(evidence_set["set_id"]), specs


def _evenly_select_frames(values: Sequence[FrameRef], limit: int) -> list[FrameRef]:
    ordered = list(values)
    if limit < 1:
        return []
    if len(ordered) <= limit:
        return ordered
    return [
        ordered[min(len(ordered) - 1, (2 * index + 1) * len(ordered) // (2 * limit))]
        for index in range(limit)
    ]


def _equal_allocations(total: int, count: int) -> list[int]:
    if count < 1:
        return []
    if total < count:
        raise ValueError("frame budget cannot give every required core item one frame")
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def sample_core_specs(
    cached: CachedVideo,
    specs: Sequence[CoreWindowSpec],
    *,
    total: int,
) -> tuple[list[FrameRef], int, list[dict[str, Any]]]:
    """Sample each required item independently so broad intervals cannot hide local slots."""

    if not specs:
        return [], 0, []
    quotas = _equal_allocations(total, len(specs))
    selected: list[FrameRef] = []
    selected_ids: set[str] = set()
    all_candidates: dict[str, FrameRef] = {}
    allocations: list[dict[str, Any]] = []
    candidates_by_spec: list[list[FrameRef]] = []
    for spec, quota in zip(specs, quotas):
        candidates = [
            frame
            for frame in cached.frames
            if spec.window.contains(frame.timestamp_seconds)
        ]
        candidates_by_spec.append(candidates)
        for frame in candidates:
            all_candidates[frame.id] = frame
        chosen = _evenly_select_frames(candidates, quota)
        for frame in chosen:
            if frame.id not in selected_ids:
                selected.append(frame)
                selected_ids.add(frame.id)
        allocations.append(
            {
                "evidence_id": spec.evidence_id,
                "slot_id": spec.slot_id,
                "modality": spec.modality,
                "requested_frames": quota,
                "candidate_frames": len(candidates),
                "initial_unique_frames": len({frame.id for frame in chosen}),
            }
        )
    if len(selected) < total:
        remaining = [
            frame
            for frame in sorted(
                all_candidates.values(),
                key=lambda item: (item.timestamp_seconds, item.id),
            )
            if frame.id not in selected_ids
        ]
        for frame in _evenly_select_frames(remaining, total - len(selected)):
            selected.append(frame)
            selected_ids.add(frame.id)
    selected.sort(key=lambda item: (item.timestamp_seconds, item.id))
    return selected[:total], len(all_candidates), allocations


def _subtitles_for_specs(
    track: SubtitleTrack | None,
    specs: Sequence[CoreWindowSpec],
    *,
    max_chars: int,
) -> str:
    if track is None or not specs or max_chars <= 0:
        return ""
    return track.text_for_windows(
        [spec.window for spec in specs],
        max_chars=max_chars,
    )


def build_core_packet(
    variant: str,
    *,
    reference: Mapping[str, Any],
    cached: CachedVideo,
    subtitle_track: SubtitleTrack | None,
    config: CoreDenseConfig,
) -> CoreEvidencePacket:
    variant_config = config.variant(variant)
    set_id, visual_specs = core_window_specs(
        reference,
        modalities={"visual", "ocr"},
    )
    _, subtitle_specs = core_window_specs(reference, modalities={"subtitle"})
    _, all_specs = core_window_specs(reference)
    frame_specs = visual_specs or all_specs
    selected, source_count, allocations = sample_core_specs(
        cached,
        frame_specs,
        total=variant_config.max_frames,
    )
    subtitles = _subtitles_for_specs(
        subtitle_track,
        subtitle_specs,
        max_chars=config.subtitle_max_chars,
    )
    return CoreEvidencePacket(
        variant=variant,
        frames=tuple(selected),
        subtitles=subtitles,
        source_frame_count=source_count,
        oracle_set_id=set_id,
        frame_specs=tuple(frame_specs),
        subtitle_specs=tuple(subtitle_specs),
        frame_allocations=tuple(allocations),
    )


def _chosen_set_core_recall(
    reference: Mapping[str, Any],
    score: Mapping[str, Any],
    set_id: str,
) -> float:
    evidence_set = next(
        item
        for item in reference["evidence_contract"]["sufficient_evidence_sets"]
        if str(item["set_id"]) == set_id
    )
    required_slots = {
        str(item["slot_id"])
        for item in reference["evidence_contract"]["evidence_slots"]
        if bool(item.get("required", True))
    }
    required_ids = {
        str(item["evidence_id"])
        for item in evidence_set["items"]
        if str(item["slot_id"]) in required_slots
    }
    hits = {
        str(item["evidence_id"])
        for item in score["items"]
        if str(item["set_id"]) == set_id and bool(item["core_hit"])
    }
    return len(required_ids & hits) / len(required_ids) if required_ids else 1.0


def validate_core_packet(
    reference: Mapping[str, Any],
    packet: CoreEvidencePacket,
    *,
    max_frames: int,
) -> dict[str, Any]:
    if len(packet.frames) > max_frames:
        raise ValueError("core packet exceeds its frame budget")
    missing = [frame.path for frame in packet.frames if not Path(frame.path).is_file()]
    if missing:
        raise FileNotFoundError(f"core packet has {len(missing)} missing frames")
    score = score_relaxed_grounding(dict(reference), packet.exposures())
    chosen_core_recall = _chosen_set_core_recall(reference, score, packet.oracle_set_id)
    if chosen_core_recall != 1.0:
        raise ValueError("core packet does not hit every required item in its Oracle set")
    if float(score["core_item_recall"]) != 1.0:
        raise ValueError("core packet does not have full reference-level core recall")
    if not bool(score["grounded"]):
        raise ValueError("core packet is not relaxed-grounded")
    return {**score, "chosen_set_core_recall": chosen_core_recall}


class CoreDenseSession:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        model: BaseVideoModel | None = None,
        model_factory: Callable[[Mapping[str, Any] | None], BaseVideoModel] = build_model,
    ) -> None:
        self.raw_config = dict(config)
        self.config = CoreDenseConfig.from_mapping(self.raw_config.get("core_dense"))
        self.model = model or model_factory(self.raw_config.get("model"))
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self.model.load()
            self._loaded = True

    def close(self) -> None:
        if self._loaded:
            self.model.unload()
            self._loaded = False

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def evaluate(
        self,
        question: VideoMMEQuestion,
        packet: CoreEvidencePacket,
        *,
        reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("CoreDenseSession is not loaded")
        variant = self.config.variant(packet.variant)
        prompt = build_unified_answer_prompt(question, packet)  # type: ignore[arg-type]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": [frame.path for frame in packet.frames],
                        "min_pixels": variant.min_pixels,
                        "max_pixels": variant.max_pixels,
                        "total_pixels": variant.total_pixels,
                        "fps": variant.video_fps,
                        "max_frames": variant.max_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        started = time.perf_counter()
        try:
            output = self.model.generate(
                messages,
                max_new_tokens=self.config.max_new_tokens,
            )
            adapter = MultipleChoiceAdapter(question.options)
            prediction = adapter.normalize(output.text)
            exposures = packet.exposures()
            score = score_relaxed_grounding(dict(reference), exposures)
            return {
                **question.to_dict(),
                "policy_id": CORE_DENSE_POLICY_ID,
                "variant": packet.variant,
                "prediction": prediction,
                "correct": prediction == question.answer,
                "completed": prediction in adapter.letters,
                "model_output": output.text,
                "prompt_sha256": canonical_sha256(prompt),
                "wall_seconds": time.perf_counter() - started,
                "packet": packet.metadata(),
                "exposures": exposures,
                "relaxed_grounding": {
                    **score,
                    "chosen_set_core_recall": _chosen_set_core_recall(
                        reference,
                        score,
                        packet.oracle_set_id,
                    ),
                },
                "model_metadata": output.metadata,
                "vision_budget": variant.to_dict(),
            }
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - inference dependency
                pass


def _new_cache(config: CoreDenseConfig) -> VideoEvidenceCache:
    return VideoEvidenceCache(
        config.cache_dir,
        sample_fps=config.sample_fps,
        max_side=config.cache_max_side,
        jpeg_quality=config.cache_jpeg_quality,
        lru_size=config.cache_lru_size,
    )


def _forbidden_prompt_keys(prompt: str) -> list[str]:
    return [
        item
        for item in (
            "atomic_fact",
            "context_interval",
            "core_interval",
            "official_answer",
            "hard_negative",
        )
        if item in prompt
    ]


def preflight_core_dense(
    *,
    dataset: DevReferenceSet,
    questions: Sequence[VideoMMEQuestion],
    baseline_failures: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    video_dir: str | Path,
    subtitle_dir: str | Path,
    baseline_items_path: str | Path,
    variants: Sequence[str] = CORE_VARIANTS,
) -> dict[str, Any]:
    dense = CoreDenseConfig.from_mapping(config.get("core_dense"))
    invalid = sorted(set(variants) - set(CORE_VARIANTS))
    if invalid:
        raise ValueError(f"unsupported core variants: {', '.join(invalid)}")
    selected_ids = list(baseline_failures)
    selected = [question for question in questions if question.question_id in baseline_failures]
    errors: list[str] = []
    packets = Counter()
    frame_counts: dict[str, list[int]] = {variant: [] for variant in variants}
    source_counts: dict[str, list[int]] = {variant: [] for variant in variants}
    if [question.question_id for question in selected] != selected_ids:
        errors.append("selected question order does not match dev manifest order")
    if len(selected_ids) != EXPECTED_ORACLE_FAILURES:
        errors.append(f"expected {EXPECTED_ORACLE_FAILURES} selected Oracle failures")
    cache = _new_cache(dense)
    resolved_subtitle_dir = Path(subtitle_dir).expanduser().resolve()
    for question in selected:
        try:
            cached = cache.prepare(question.video_path(video_dir))
            subtitle_path = question.subtitle_path(resolved_subtitle_dir)
            track = SubtitleTrack.from_srt(subtitle_path) if subtitle_path.is_file() else None
            reference = dataset.reference(question.question_id)
            for variant in variants:
                packet = build_core_packet(
                    variant,
                    reference=reference,
                    cached=cached,
                    subtitle_track=track,
                    config=dense,
                )
                prompt = build_unified_answer_prompt(question, packet)  # type: ignore[arg-type]
                leaked = _forbidden_prompt_keys(prompt)
                if leaked:
                    raise ValueError(f"reference schema keys leaked into prompt: {leaked}")
                validate_core_packet(
                    reference,
                    packet,
                    max_frames=dense.variant(variant).max_frames,
                )
                packets[variant] += 1
                frame_counts[variant].append(len(packet.frames))
                source_counts[variant].append(packet.source_frame_count)
        except Exception as exc:  # noqa: BLE001 - aggregate all preflight failures
            errors.append(f"{question.question_id}: {type(exc).__name__}: {exc}")
    return {
        "policy_id": CORE_DENSE_POLICY_ID,
        "ok": not errors,
        "split": "dev",
        "locked_records_read": False,
        "selected_questions": len(selected_ids),
        "selected_question_ids": selected_ids,
        "expected_packets": len(selected_ids) * len(variants),
        "packets": dict(packets),
        "frame_count_ranges": {
            variant: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for variant, values in frame_counts.items()
        },
        "source_frame_count_ranges": {
            variant: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for variant, values in source_counts.items()
        },
        "vision_budgets": {
            variant: dense.variant(variant).to_dict() for variant in variants
        },
        "dev_reference_sha256": dataset.dev_sha256,
        "baseline_items_sha256": sha256_file(baseline_items_path),
        "errors": errors,
    }


def _mean(
    rows: Sequence[Mapping[str, Any]],
    getter: Callable[[Mapping[str, Any]], float],
) -> float | None:
    values = [getter(row) for row in rows]
    return statistics.mean(values) if values else None


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if bool(row.get("completed"))]
    return {
        "items": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows) if rows else 0.0,
        "correct": sum(bool(row.get("correct")) for row in rows),
        "accuracy": sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else 0.0,
        "recovered_from_oracle_context_failure": sum(
            bool(row.get("correct")) for row in rows
        ),
        "relaxed_grounded_rate": (
            sum(bool(row.get("relaxed_grounding", {}).get("grounded")) for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "mean_core_item_recall": _mean(
            rows,
            lambda row: float(row.get("relaxed_grounding", {}).get("core_item_recall", 0)),
        ),
        "mean_selected_frames": _mean(
            rows,
            lambda row: float(row.get("packet", {}).get("selected_frame_count", 0)),
        ),
        "mean_source_frames": _mean(
            rows,
            lambda row: float(row.get("packet", {}).get("source_frame_count", 0)),
        ),
        "mean_subtitle_chars": _mean(
            rows,
            lambda row: float(row.get("packet", {}).get("subtitle_chars", 0)),
        ),
        "mean_input_tokens": _mean(
            rows,
            lambda row: float(row.get("model_metadata", {}).get("input_tokens", 0)),
        ),
        "mean_wall_seconds": _mean(rows, lambda row: float(row.get("wall_seconds", 0))),
    }


def _paired(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    question_ids = sorted(set(left) & set(right))
    wins = losses = ties = same_prediction = 0
    for question_id in question_ids:
        left_row = left[question_id]
        right_row = right[question_id]
        left_correct = bool(left_row.get("correct"))
        right_correct = bool(right_row.get("correct"))
        if left_correct and not right_correct:
            wins += 1
        elif right_correct and not left_correct:
            losses += 1
        else:
            ties += 1
        same_prediction += int(left_row.get("prediction") == right_row.get("prediction"))
    return {
        "comparable": len(question_ids),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "same_prediction": same_prediction,
        "different_prediction": len(question_ids) - same_prediction,
    }


def summarize_core_dense(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_failures: Mapping[str, Mapping[str, Any]],
    variants: Sequence[str],
    expected_items: int,
    run_signature: str,
) -> dict[str, Any]:
    by_variant_rows = {
        variant: [row for row in rows if row.get("variant") == variant]
        for variant in variants
    }
    by_variant = {
        variant: {str(row["question_id"]): row for row in values}
        for variant, values in by_variant_rows.items()
    }
    duration_groups: dict[str, dict[str, Any]] = {}
    for variant, values in by_variant_rows.items():
        for duration in sorted({str(row.get("duration")) for row in values}):
            duration_groups[f"{variant}:{duration}"] = _metrics(
                [row for row in values if str(row.get("duration")) == duration]
            )
    paired = None
    if set(CORE_VARIANTS) <= set(by_variant):
        paired = _paired(by_variant["core_32"], by_variant["core_16"])
    baseline_predictions = {
        question_id: str(row.get("prediction"))
        for question_id, row in baseline_failures.items()
    }
    prediction_changes = {
        variant: sum(
            row.get("prediction") != baseline_predictions.get(str(row["question_id"]))
            for row in values
        )
        for variant, values in by_variant_rows.items()
    }
    return {
        "policy_id": CORE_DENSE_POLICY_ID,
        "split": "dev",
        "locked_records_read": False,
        "baseline_policy_id": ABLATION_POLICY_ID,
        "baseline_oracle_context": {
            "items": len(baseline_failures),
            "correct": 0,
            "accuracy": 0.0,
        },
        "expected_items": expected_items,
        "observed_items": len(rows),
        "engineering_pass": (
            len(rows) == expected_items
            and all(bool(row.get("completed")) for row in rows)
            and all(not row.get("error") for row in rows)
            and all(
                float(row.get("relaxed_grounding", {}).get("chosen_set_core_recall", 0))
                == 1.0
                for row in rows
            )
        ),
        "run_signature": run_signature,
        "variant_metrics": {
            variant: _metrics(values) for variant, values in by_variant_rows.items()
        },
        "core_32_vs_core_16": paired,
        "prediction_changes_vs_oracle_context": prediction_changes,
        "duration_metrics": duration_groups,
    }


def run_core_dense_suite(
    *,
    dataset: DevReferenceSet,
    questions: Sequence[VideoMMEQuestion],
    baseline_failures: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    video_dir: str | Path,
    subtitle_dir: str | Path,
    baseline_items_path: str | Path,
    output_dir: str | Path,
    variants: Sequence[str] = CORE_VARIANTS,
    session_factory: Callable[..., CoreDenseSession] = CoreDenseSession,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    invalid = sorted(set(variants) - set(CORE_VARIANTS))
    if invalid:
        raise ValueError(f"unsupported core variants: {', '.join(invalid)}")
    dense = CoreDenseConfig.from_mapping(config.get("core_dense"))
    selected_questions = [
        question for question in questions if question.question_id in baseline_failures
    ]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    items_path = output / "items.jsonl"
    manifest_path = output / "run_manifest.json"
    baseline_sha = sha256_file(baseline_items_path)
    run_signature = canonical_sha256(
        {
            "policy_id": CORE_DENSE_POLICY_ID,
            "config": config,
            "selected_question_ids": list(baseline_failures),
            "dev_reference_sha256": dataset.dev_sha256,
            "baseline_items_sha256": baseline_sha,
            "variants": list(variants),
        }
    )
    run_manifest = {
        "policy_id": CORE_DENSE_POLICY_ID,
        "split": "dev",
        "locked_records_read": False,
        "run_signature": run_signature,
        "dev_reference_path": str(dataset.root / "dev.jsonl"),
        "dev_reference_sha256": dataset.dev_sha256,
        "baseline_items_path": str(Path(baseline_items_path).expanduser().resolve()),
        "baseline_items_sha256": baseline_sha,
        "selected_question_ids": list(baseline_failures),
        "variants": list(variants),
        "vision_budgets": {
            variant: dense.variant(variant).to_dict() for variant in variants
        },
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != run_manifest:
            raise ValueError("existing core-density output has a different run signature")
    else:
        manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    history: list[dict[str, Any]] = []
    if items_path.is_file():
        history = [
            json.loads(line)
            for line in items_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    expected_keys = {
        (question.question_id, variant)
        for variant in variants
        for question in selected_questions
    }
    history_keys = {
        (str(row.get("question_id")), str(row.get("variant"))) for row in history
    }
    if history_keys - expected_keys:
        raise ValueError("core-density checkpoint contains unexpected records")
    latest: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["question_id"]), str(row["variant"])): row for row in history
    }
    completed_keys = {
        key
        for key, row in latest.items()
        if bool(row.get("completed")) and not row.get("error")
    }

    cache = _new_cache(dense)
    resolved_subtitle_dir = Path(subtitle_dir).expanduser().resolve()
    with session_factory(config) as session:
        for variant in variants:
            for question in selected_questions:
                key = (question.question_id, variant)
                if key in completed_keys:
                    continue
                reference = dataset.reference(question.question_id)
                started = time.perf_counter()
                try:
                    cached = cache.prepare(question.video_path(video_dir))
                    subtitle_path = question.subtitle_path(resolved_subtitle_dir)
                    track = SubtitleTrack.from_srt(subtitle_path) if subtitle_path.is_file() else None
                    packet = build_core_packet(
                        variant,
                        reference=reference,
                        cached=cached,
                        subtitle_track=track,
                        config=dense,
                    )
                    validate_core_packet(
                        reference,
                        packet,
                        max_frames=dense.variant(variant).max_frames,
                    )
                    row = session.evaluate(question, packet, reference=reference)
                except Exception as exc:  # noqa: BLE001 - retain resumable diagnostics
                    row = {
                        **question.to_dict(),
                        "policy_id": CORE_DENSE_POLICY_ID,
                        "variant": variant,
                        "prediction": None,
                        "correct": False,
                        "completed": False,
                        "wall_seconds": time.perf_counter() - started,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                row["run_signature"] = run_signature
                with items_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                latest[key] = row
                if progress is not None:
                    progress(
                        {
                            "variant": variant,
                            "question_id": question.question_id,
                            "completed": sum(
                                bool(item.get("completed")) and not item.get("error")
                                for item in latest.values()
                            ),
                            "expected": len(expected_keys),
                            "item": row,
                        }
                    )

    ordered_rows = [
        latest[(question.question_id, variant)]
        for variant in variants
        for question in selected_questions
        if (question.question_id, variant) in latest
    ]
    summary = summarize_core_dense(
        ordered_rows,
        baseline_failures=baseline_failures,
        variants=variants,
        expected_items=len(expected_keys),
        run_signature=run_signature,
    )
    summary["items_path"] = str(items_path)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
